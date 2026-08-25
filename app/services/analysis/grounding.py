"""Claim-level grounding verification and conservative repair for summaries."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Sequence

from pydantic import ValidationError

from app.core.exceptions import AnalysisResponseError
from app.models.analysis import (
    GroundingRepairResponse,
    GroundingVerificationResponse,
    SummaryStyle,
)
from app.services.analysis.prompts import (
    GROUNDING_REPAIR_SYSTEM_PROMPT,
    GROUNDING_VERIFIER_SYSTEM_PROMPT,
    build_grounding_repair_prompt,
    build_grounding_verification_prompt,
)
from app.services.llm.ollama import OllamaClient

if TYPE_CHECKING:
    from app.services.analysis.summarizer import AnalysisChunk

logger = logging.getLogger(__name__)
_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\r?\n+")
_SUMMARY_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_SUMMARY_IDENTIFIER_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*(?:[-/][A-Z0-9]+)+\b")
_SUMMARY_RELATION_WORDS = frozenset(
    {
        "associated",
        "associate",
        "supports",
        "support",
        "supported",
        "referenced",
        "references",
        "relates",
        "related",
        "specifies",
        "specify",
        "establishes",
        "establish",
        "defines",
        "define",
        "indicates",
        "indicate",
        "causes",
        "cause",
        "requires",
        "require",
        "proves",
        "prove",
        "because",
        "therefore",
        "purpose",
        "applies",
        "covers",
        "governs",
        "outlines",
    }
)
_SUMMARY_FRAMING_WORDS = frozenset(
    {
        "according",
        "content",
        "contain",
        "contains",
        "describe",
        "describes",
        "document",
        "evidence",
        "explicitly",
        "following",
        "mention",
        "mentions",
        "report",
        "reports",
        "said",
        "say",
        "source",
        "state",
        "states",
        "text",
    }
)
_SUMMARY_STOP_WORDS = frozenset(
    {
        "a",
        "after",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "during",
        "for",
        "from",
        "he",
        "her",
        "his",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "of",
        "on",
        "or",
        "our",
        "over",
        "she",
        "than",
        "that",
        "the",
        "their",
        "these",
        "this",
        "those",
        "to",
        "under",
        "was",
        "were",
        "we",
        "with",
        "without",
        "would",
        "you",
        "your",
    }
)
_SUMMARY_HEADINGS = frozenset(
    {
        "dates / time periods",
        "explicit facts",
        "explicit obligations",
        "identifiers / references",
        "not specified",
        "overview",
        "qualifications",
    }
)


def render_evidence_block(chunk: Any, *, source_id: str) -> str:
    """Render one source block for verifier prompts and extractive fallback."""

    page = "(unavailable)"
    if chunk.start_page is not None:
        page = str(chunk.start_page)
        if chunk.end_page is not None and chunk.end_page != chunk.start_page:
            page = f"{chunk.start_page}-{chunk.end_page}"
    section = chunk.section_heading or "(none)"
    return (
        f"[{source_id}]\nDocument: {chunk.filename}\nChunk: {chunk.chunk_id}\n"
        f"Sequence: {chunk.sequence_number}\nPage: {page}\nSection: {section}\n"
        f"Content:\n{chunk.text}"
    )


def render_evidence_context(
    chunks: Sequence[Any],
    *,
    source_ids: Sequence[str] | None = None,
) -> str:
    """Render ordered evidence with stable source labels."""

    labels = list(source_ids or (f"S{index}" for index in range(1, len(chunks) + 1)))
    if len(labels) != len(chunks):
        raise ValueError("source_ids must align one-to-one with chunks.")
    return "\n\n".join(
        render_evidence_block(chunk, source_id=label)
        for label, chunk in zip(labels, chunks, strict=True)
    )


def deterministically_grounded_summary(
    summary: str,
    chunks: Sequence[Any],
    *,
    style: SummaryStyle,
    source_ids: Sequence[str] | None = None,
) -> bool:
    """Conservatively accept only cited claims whose content occurs in the evidence.

    This is deliberately a lexical safety boundary, not a semantic entailment model. It
    allows light reporting framing and inflection differences, but any new content word,
    number, identifier, actor, evaluative term, or relational verb causes rejection. A
    rejected draft is replaced by the extractive fallback by the caller.
    """

    labels = list(source_ids or (f"S{index}" for index in range(1, len(chunks) + 1)))
    if len(labels) != len(chunks) or not summary.strip():
        return False
    available = set(labels)
    source_by_label = dict(zip(labels, chunks, strict=True))
    segments = _summary_segments(summary, style=style)
    if not segments:
        return False

    for segment in segments:
        citations = [f"S{number}" for number in _CITATION_PATTERN.findall(segment)]
        if not citations or any(label not in available for label in citations):
            return False
        claim_text = _CITATION_PATTERN.sub("", segment)
        claim_text = re.sub(r"^\s*[-*•�]\s*", "", claim_text).strip()
        claim_tokens = _meaningful_summary_tokens(claim_text)
        if not claim_tokens:
            return False

        cited_chunks = [source_by_label[label] for label in dict.fromkeys(citations)]
        cited_tokens = set().union(
            *(_meaningful_summary_tokens(chunk.text) for chunk in cited_chunks)
        )
        if not claim_tokens.issubset(cited_tokens):
            return False

        has_identifier = bool(_SUMMARY_IDENTIFIER_PATTERN.search(claim_text))
        has_relation = bool(
            _SUMMARY_RELATION_WORDS.intersection(
                _normalized_summary_tokens(claim_text)
            )
        )
        if has_identifier or has_relation:
            sentence_tokens = [
                _meaningful_summary_tokens(sentence)
                for chunk in cited_chunks
                for sentence in _SENTENCE_PATTERN.split(chunk.text)
            ]
            if not any(claim_tokens.issubset(tokens) for tokens in sentence_tokens):
                return False
    return True


def _summary_segments(summary: str, *, style: SummaryStyle) -> list[str]:
    """Split generated text into atomic, cited claims while ignoring known headings."""

    segments: list[str] = []
    for raw_line in summary.splitlines() or [summary]:
        line = raw_line.strip()
        if not line:
            continue
        if style is SummaryStyle.BULLET_POINTS:
            if not line.startswith("-"):
                return []
            line = line[1:].strip()
        line = re.sub(r"^#{1,6}\s*", "", line).strip()
        heading = line.rstrip(":").strip().lower()
        if not _CITATION_PATTERN.search(line) and heading in _SUMMARY_HEADINGS:
            continue
        for part in (part.strip() for part in _SENTENCE_PATTERN.split(line) if part.strip()):
            if re.fullmatch(r"(?:\[S\d+\]\s*)+", part) and segments:
                segments[-1] = f"{segments[-1]} {part}"
            else:
                segments.append(part)
    return segments


def _normalized_summary_tokens(text: str) -> set[str]:
    """Normalize words for conservative comparison without a model dependency."""

    normalized: set[str] = set()
    for raw_token in _SUMMARY_TOKEN_PATTERN.findall(text.lower()):
        token = raw_token
        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        normalized.add(token)
    return normalized


def _meaningful_summary_tokens(text: str) -> set[str]:
    """Remove only grammar and reporting-frame words from a claim."""

    return {
        token
        for token in _normalized_summary_tokens(text)
        if token not in _SUMMARY_STOP_WORDS and token not in _SUMMARY_FRAMING_WORDS
    }


class GroundingVerifier:
    """Use the existing local Ollama client to verify and repair summary claims."""

    def __init__(self, ollama_client: OllamaClient) -> None:
        self.ollama_client = ollama_client

    async def verify(
        self,
        draft_summary: str,
        chunks: Sequence[Any],
        *,
        style: SummaryStyle,
        source_ids: Sequence[str] | None = None,
    ) -> GroundingVerificationResponse:
        """Verify every draft claim against the supplied source blocks."""

        labels = list(source_ids or (f"S{index}" for index in range(1, len(chunks) + 1)))
        evidence = render_evidence_context(chunks, source_ids=labels)
        raw = await self.ollama_client.generate_json(
            system_prompt=GROUNDING_VERIFIER_SYSTEM_PROMPT,
            user_prompt=build_grounding_verification_prompt(
                draft_summary=draft_summary,
                evidence=evidence,
                style=style,
            ),
        )
        result = self._validate_provider_model(raw, GroundingVerificationResponse)
        self._validate_claim_sources(result, labels, chunks)
        return result.model_copy(
            update={
                "has_unsupported_claims": any(
                    not claim.supported for claim in result.claims
                )
            }
        )

    async def repair(
        self,
        draft_summary: str,
        verification: GroundingVerificationResponse,
        chunks: Sequence[Any],
        *,
        style: SummaryStyle,
        source_ids: Sequence[str] | None = None,
    ) -> GroundingRepairResponse:
        """Repair one failed draft using one structured, bounded provider call."""

        labels = list(source_ids or (f"S{index}" for index in range(1, len(chunks) + 1)))
        evidence = render_evidence_context(chunks, source_ids=labels)
        verification_json = json.dumps(verification.model_dump(mode="json"), ensure_ascii=False)
        raw = await self.ollama_client.generate_json(
            system_prompt=GROUNDING_REPAIR_SYSTEM_PROMPT,
            user_prompt=build_grounding_repair_prompt(
                draft_summary=draft_summary,
                evidence=evidence,
                verification=verification_json,
                style=style,
            ),
        )
        return self._validate_provider_model(raw, GroundingRepairResponse)

    @staticmethod
    def _validate_provider_model(raw: dict[str, Any], model: Any) -> Any:
        """Map malformed provider JSON to the existing safe analysis error."""

        try:
            return model.model_validate(raw)
        except (ValidationError, TypeError, ValueError) as exc:
            logger.warning("Ollama grounding response failed structured validation")
            raise AnalysisResponseError(
                "The local model returned an invalid grounding response."
            ) from exc

    @staticmethod
    def _validate_claim_sources(
        result: GroundingVerificationResponse,
        source_ids: Sequence[str],
        chunks: Sequence[Any],
    ) -> None:
        """Reject unknown citations and unsupported claim provenance."""

        available = set(source_ids)
        source_by_label = dict(zip(source_ids, chunks, strict=True))
        for claim in result.claims:
            unknown = set(claim.source_labels) - available
            if unknown:
                raise AnalysisResponseError(
                    "The local model returned a citation for a source that was not provided."
                )
            if claim.supported and not claim.source_labels:
                raise AnalysisResponseError(
                    "The local model marked a claim supported without source evidence."
                )
            if claim.supported and not claim.supporting_evidence:
                raise AnalysisResponseError(
                    "The local model marked a claim supported without a supporting evidence quote."
                )
            if claim.supported:
                quote = _normalize_quote(claim.supporting_evidence)
                if not quote or not any(
                    quote in _normalize_quote(source_by_label[label].text)
                    for label in claim.source_labels
                ):
                    raise AnalysisResponseError(
                        "The local model returned supporting evidence that was not present in the cited source."
                    )


def _normalize_quote(text: str) -> str:
    """Normalize whitespace for exact supporting-span comparison."""

    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def conservative_fallback(
    chunks: Sequence[Any],
    *,
    style: SummaryStyle,
    max_chars: int,
    source_ids: Sequence[str] | None = None,
) -> str:
    """Build an extractive evidence-only summary after bounded verification failure."""

    labels = list(source_ids or (f"S{index}" for index in range(1, len(chunks) + 1)))
    if len(labels) != len(chunks):
        raise ValueError("source_ids must align one-to-one with chunks.")
    items: list[str] = []
    for label, chunk in zip(labels, chunks, strict=True):
        sentences = [part.strip() for part in _SENTENCE_PATTERN.split(chunk.text) if part.strip()]
        if not sentences and chunk.text.strip():
            sentences = [chunk.text.strip()]
        for sentence in sentences:
            items.append(f"{sentence} [{label}]")

    if style is SummaryStyle.BULLET_POINTS:
        candidate = "\n".join(f"- {item}" for item in items)
    else:
        candidate = " ".join(items)
    if len(candidate) <= max_chars:
        return candidate
    marker = "\n[content truncated to the configured analysis summary budget]"
    if max_chars <= len(marker):
        return candidate[:max_chars]
    return candidate[: max_chars - len(marker)] + marker
