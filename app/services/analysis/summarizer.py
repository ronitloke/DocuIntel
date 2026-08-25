"""Bounded hierarchical document summarization over persisted chunks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from time import perf_counter
from uuid import UUID

from app.core.exceptions import AnalysisResponseError, OllamaServiceError
from app.models.analysis import SummaryStyle
from app.services.analysis.grounding import (
    GroundingVerifier,
    conservative_fallback,
    deterministically_grounded_summary,
    render_evidence_block,
)
from app.services.analysis.prompts import (
    SUMMARY_SYSTEM_PROMPT,
    build_final_summary_prompt,
    build_summary_batch_prompt,
)
from app.services.llm.ollama import OllamaClient

logger = logging.getLogger(__name__)
_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
_IDENTIFIER_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*(?:[-/][A-Z0-9]+)+\b")
_IDENTIFIER_RELATION_PATTERN = re.compile(
    r"\b(?:specifies|establishes|defines|belongs to|is associated with|associated with|"
    r"is related to|related to|relates to|corresponds to|indicates|sets|governs|covers|"
    r"applies to|reference to)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AnalysisChunk:
    """Small detached chunk projection used by analysis services."""

    document_id: UUID
    chunk_id: UUID
    sequence_number: int
    text: str
    start_page: int | None
    end_page: int | None
    section_heading: str | None
    filename: str


@dataclass(frozen=True, slots=True)
class SummaryBatch:
    """One bounded ordered batch and the chunks represented in it."""

    text: str
    chunks: list[AnalysisChunk]
    source_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SummaryGeneration:
    """Provider output and generation-stage timings."""

    summary: str
    partial_generation_time_ms: float
    final_synthesis_time_ms: float
    grounding_verification_time_ms: float = 0.0
    grounding_repair_time_ms: float = 0.0
    grounding_verification_passes: int = 0


class DocumentSummarizer:
    """Generate one summary using one call or hierarchical partial synthesis."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        *,
        grounding_verifier: GroundingVerifier | None = None,
        grounding_enabled: bool = False,
        grounding_max_passes: int = 2,
    ) -> None:
        self.ollama_client = ollama_client
        self.grounding_verifier = grounding_verifier
        self.grounding_enabled = grounding_enabled
        if grounding_max_passes < 1 or grounding_max_passes > 2:
            raise ValueError("grounding_max_passes must be between 1 and 2.")
        self.grounding_max_passes = grounding_max_passes
        if grounding_enabled and grounding_verifier is None:
            raise ValueError("grounding_verifier is required when grounding is enabled.")

    def build_batches(self, chunks: list[AnalysisChunk], *, max_chars: int) -> list[SummaryBatch]:
        """Group ordered chunks without exceeding the configured character budget."""

        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero.")
        batches: list[SummaryBatch] = []
        current_text: list[str] = []
        current_chunks: list[AnalysisChunk] = []
        current_source_ids: list[str] = []
        current_length = 0
        for index, chunk in enumerate(chunks, start=1):
            source_id = f"S{index}"
            block = self._chunk_block(chunk, source_id=source_id)
            if len(block) > max_chars:
                block = self._truncate_block(block, max_chars)
            separator_length = 2 if current_text else 0
            if current_text and current_length + separator_length + len(block) > max_chars:
                batches.append(
                    SummaryBatch(
                        "\n\n".join(current_text),
                        current_chunks,
                        current_source_ids,
                    )
                )
                current_text = []
                current_chunks = []
                current_source_ids = []
                current_length = 0
                separator_length = 0
            current_text.append(block)
            current_chunks.append(chunk)
            current_source_ids.append(source_id)
            current_length += separator_length + len(block)
        if current_text:
            batches.append(
                SummaryBatch("\n\n".join(current_text), current_chunks, current_source_ids)
            )
        return batches

    async def summarize(
        self,
        chunks: list[AnalysisChunk],
        *,
        style: SummaryStyle,
        batch_max_chars: int,
        final_max_chars: int,
    ) -> SummaryGeneration:
        """Generate partial summaries and synthesize them when multiple batches exist."""

        batches = self.build_batches(chunks, max_chars=batch_max_chars)
        if not batches:
            raise ValueError("At least one document chunk is required for summarization.")

        partial_summaries: list[str] = []
        grounding_verification_time_ms = 0.0
        grounding_repair_time_ms = 0.0
        grounding_verification_passes = 0
        partial_generation_time_ms = 0.0
        for batch in batches:
            partial_generation_started = perf_counter()
            draft_summary = await self.ollama_client.generate(
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                user_prompt=build_summary_batch_prompt(style=style, context=batch.text),
            )
            partial_generation_time_ms += (perf_counter() - partial_generation_started) * 1000
            partial_summary, grounding = await self._ground_summary(
                draft_summary,
                batch.chunks,
                source_ids=batch.source_ids,
                style=style,
                max_chars=final_max_chars,
            )
            grounding_verification_time_ms += grounding[0]
            grounding_repair_time_ms += grounding[1]
            grounding_verification_passes += grounding[2]
            self._validate_citations(partial_summary, source_ids=batch.source_ids)
            partial_summaries.append(partial_summary)
        partial_generation_time_ms = round(partial_generation_time_ms, 3)

        if len(partial_summaries) == 1:
            return SummaryGeneration(
                summary=partial_summaries[0],
                partial_generation_time_ms=partial_generation_time_ms,
                final_synthesis_time_ms=0.0,
                grounding_verification_time_ms=round(grounding_verification_time_ms, 3),
                grounding_repair_time_ms=round(grounding_repair_time_ms, 3),
                grounding_verification_passes=grounding_verification_passes,
            )

        bounded_partials = self._bound_partial_summaries(partial_summaries, final_max_chars)
        final_generation_started = perf_counter()
        final_draft = await self.ollama_client.generate(
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            user_prompt=build_final_summary_prompt(
                style=style,
                partial_summaries=bounded_partials,
            ),
        )
        final_synthesis_time_ms = round(
            (perf_counter() - final_generation_started) * 1000,
            3,
        )
        final_summary, grounding = await self._ground_summary(
            final_draft,
            chunks,
            source_ids=[f"S{index}" for index in range(1, len(chunks) + 1)],
            style=style,
            max_chars=final_max_chars,
        )
        grounding_verification_time_ms += grounding[0]
        grounding_repair_time_ms += grounding[1]
        grounding_verification_passes += grounding[2]
        self._validate_citations(
            final_summary,
            source_ids=[f"S{index}" for index in range(1, len(chunks) + 1)],
        )
        logger.info(
            "Document summary generated batches=%s model=%s partial_ms=%.3f final_ms=%.3f "
            "grounding_ms=%.3f repair_ms=%.3f passes=%s",
            len(batches),
            self.ollama_client.model,
            partial_generation_time_ms,
            final_synthesis_time_ms,
            grounding_verification_time_ms,
            grounding_repair_time_ms,
            grounding_verification_passes,
        )
        return SummaryGeneration(
            summary=final_summary,
            partial_generation_time_ms=partial_generation_time_ms,
            final_synthesis_time_ms=final_synthesis_time_ms,
            grounding_verification_time_ms=round(grounding_verification_time_ms, 3),
            grounding_repair_time_ms=round(grounding_repair_time_ms, 3),
            grounding_verification_passes=grounding_verification_passes,
        )

    async def _ground_summary(
        self,
        draft_summary: str,
        chunks: list[AnalysisChunk],
        *,
        source_ids: list[str],
        style: SummaryStyle,
        max_chars: int,
    ) -> tuple[str, tuple[float, float, int]]:
        """Verify, repair once, and safely fall back for one summary stage."""

        candidate = self._normalize_summary(
            self._sanitize_unsupported_identifier_relationships(draft_summary, chunks),
            style=style,
        )
        if not self.grounding_enabled or self.grounding_verifier is None:
            return candidate, (0.0, 0.0, 0)

        verification_time_ms = 0.0
        repair_time_ms = 0.0
        passes = 0
        current = candidate
        for pass_number in range(1, self.grounding_max_passes + 1):
            verification_started = perf_counter()
            try:
                verification = await self.grounding_verifier.verify(
                    current,
                    chunks,
                    style=style,
                    source_ids=source_ids,
                )
            except (AnalysisResponseError, OllamaServiceError):
                verification_time_ms += (perf_counter() - verification_started) * 1000
                logger.warning(
                    "Summary grounding response was malformed; using extractive fallback"
                )
                fallback = conservative_fallback(
                    chunks,
                    style=style,
                    max_chars=max_chars,
                    source_ids=source_ids,
                )
                return self._normalize_summary(fallback, style=style), (
                    round(verification_time_ms, 3),
                    round(repair_time_ms, 3),
                    pass_number,
                )
            verification_time_ms += (perf_counter() - verification_started) * 1000
            passes = pass_number
            if not verification.has_unsupported_claims:
                if deterministically_grounded_summary(
                    current,
                    chunks,
                    style=style,
                    source_ids=source_ids,
                ):
                    return current, (
                        round(verification_time_ms, 3),
                        round(repair_time_ms, 3),
                        passes,
                    )
                logger.warning(
                    "Deterministic summary grounding rejected a clean verifier result; "
                    "using extractive fallback"
                )
                break
            if pass_number == self.grounding_max_passes:
                break
            repair_started = perf_counter()
            try:
                repaired = await self.grounding_verifier.repair(
                    current,
                    verification,
                    chunks,
                    style=style,
                    source_ids=source_ids,
                )
            except (AnalysisResponseError, OllamaServiceError):
                repair_time_ms += (perf_counter() - repair_started) * 1000
                logger.warning(
                    "Summary repair response was malformed; using extractive fallback"
                )
                fallback = conservative_fallback(
                    chunks,
                    style=style,
                    max_chars=max_chars,
                    source_ids=source_ids,
                )
                return self._normalize_summary(fallback, style=style), (
                    round(verification_time_ms, 3),
                    round(repair_time_ms, 3),
                    passes,
                )
            repair_time_ms += (perf_counter() - repair_started) * 1000
            current = self._normalize_summary(
                self._sanitize_unsupported_identifier_relationships(
                    repaired.repaired_summary,
                    chunks,
                ),
                style=style,
            )

        logger.warning(
            "Summary grounding failed after bounded passes=%s; using extractive fallback",
            passes,
        )
        fallback = conservative_fallback(
            chunks,
            style=style,
            max_chars=max_chars,
            source_ids=source_ids,
        )
        return self._normalize_summary(fallback, style=style), (
            round(verification_time_ms, 3),
            round(repair_time_ms, 3),
            passes,
        )

    def enforce_final_grounding_boundary(
        self,
        summary: str,
        chunks: list[AnalysisChunk],
        *,
        style: SummaryStyle,
        source_ids: list[str],
        max_chars: int,
    ) -> str:
        """Guarantee that only deterministic evidence-derived text reaches the API."""

        candidate = self._normalize_summary(
            self._sanitize_unsupported_identifier_relationships(summary, chunks),
            style=style,
        )
        if deterministically_grounded_summary(
            candidate,
            chunks,
            style=style,
            source_ids=source_ids,
        ):
            return candidate
        logger.warning("Final summary boundary rejected generated text; using extractive fallback")
        fallback = conservative_fallback(
            chunks,
            style=style,
            max_chars=max_chars,
            source_ids=source_ids,
        )
        return self._normalize_summary(fallback, style=style)

    @staticmethod
    def _chunk_block(chunk: AnalysisChunk, *, source_id: str) -> str:
        """Render chunk metadata and data without treating it as instructions."""

        return render_evidence_block(chunk, source_id=source_id)

    @staticmethod
    def _normalize_summary(summary: str, *, style: SummaryStyle) -> str:
        """Normalize provider formatting without changing factual content."""

        cleaned = summary.strip()
        if style is not SummaryStyle.BULLET_POINTS:
            return cleaned
        normalized = cleaned.replace("•", "-").replace("�", "-")
        normalized = re.sub(r"\s+[-*]\s+", "\n- ", normalized)
        lines = normalized.splitlines()
        output: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("*"):
                stripped = "-" + stripped[1:]
            output.append(stripped)
        return "\n".join(line for line in output if line)

    @staticmethod
    def _truncate_block(block: str, max_chars: int) -> str:
        """Bound one pathological chunk while retaining its source header."""

        marker = "\n[content truncated to the configured analysis batch budget]"
        if max_chars <= len(marker):
            return block[:max_chars]
        return block[: max_chars - len(marker)] + marker

    @staticmethod
    def _bound_partial_summaries(summaries: list[str], max_chars: int) -> str:
        """Keep final synthesis input bounded without adding a tokenizer dependency."""

        if max_chars <= 0:
            raise ValueError("final_max_chars must be greater than zero.")
        blocks: list[str] = []
        used = 0
        for index, summary in enumerate(summaries, start=1):
            block = f"[Partial {index}]\n{summary}"
            separator = 2 if blocks else 0
            available = max_chars - used - separator
            if available <= 0:
                break
            blocks.append(block[:available])
            used += separator + min(len(block), available)
        return "\n\n".join(blocks)

    @staticmethod
    def _sanitize_unsupported_identifier_relationships(
        summary: str,
        chunks: list[AnalysisChunk],
    ) -> str:
        """Keep source identifiers from being linked to unsupported claims.

        This is intentionally narrow: it only rewrites a generated fragment when an
        identifier present in the source is paired with a relationship phrase that is
        absent from the source sentence containing that identifier. It does not attempt
        general factual verification or rewrite ordinary paraphrases.
        """

        evidence = "\n".join(chunk.text for chunk in chunks)
        identifiers = set(_IDENTIFIER_PATTERN.findall(evidence))
        if not identifiers:
            return summary

        explicit_relationships: set[str] = set()
        evidence_sentences = re.split(r"(?<=[.!?])\s+|\r?\n+", evidence)
        for sentence in evidence_sentences:
            sentence_identifiers = _IDENTIFIER_PATTERN.findall(sentence)
            if _IDENTIFIER_RELATION_PATTERN.search(sentence):
                explicit_relationships.update(sentence_identifiers)
        unsafe_identifiers = identifiers - explicit_relationships
        if not unsafe_identifiers:
            return summary

        rewritten = False
        fragments = re.split(r"(?<=[.!?])\s+|\r?\n+", summary)
        for index, fragment in enumerate(fragments):
            fragment_identifiers = [
                identifier
                for identifier in _IDENTIFIER_PATTERN.findall(fragment)
                if identifier in unsafe_identifiers
            ]
            if not fragment_identifiers or not _IDENTIFIER_RELATION_PATTERN.search(fragment):
                continue
            prefix_match = re.match(r"^(\s*[-*•�]\s*)", fragment)
            prefix = prefix_match.group(1) if prefix_match else ""
            citations = " ".join(
                f"[S{citation}]" for citation in _CITATION_PATTERN.findall(fragment)
            )
            unique_identifiers = list(dict.fromkeys(fragment_identifiers))
            replacement = f"{prefix}Identifier/reference mentioned: {', '.join(unique_identifiers)}."
            if citations:
                replacement = f"{replacement} {citations}"
            fragments[index] = replacement
            rewritten = True

        if rewritten:
            logger.warning(
                "Rewrote unsupported identifier relationship in generated summary identifiers=%s",
                sorted(unsafe_identifiers),
            )
            return "\n".join(fragments)
        return summary

    @staticmethod
    def _validate_citations(summary: str, *, source_ids: list[str]) -> None:
        """Reject source labels that do not map to real supplied chunks."""

        available = set(source_ids)
        unknown = {
            f"S{match}"
            for match in _CITATION_PATTERN.findall(summary)
            if f"S{match}" not in available
        }
        if unknown:
            raise AnalysisResponseError(
                "The local model returned a citation for a source that was not provided."
            )
