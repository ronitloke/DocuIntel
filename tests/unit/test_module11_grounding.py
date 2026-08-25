"""Deterministic claim-level grounding verification coverage for Module 11.2."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.core.exceptions import AnalysisResponseError, OllamaServiceError
from app.models.analysis import (
    GroundingVerificationResponse,
    SummaryStyle,
)
from app.services.analysis.grounding import GroundingVerifier
from app.services.analysis.prompts import (
    GROUNDING_VERIFIER_SYSTEM_PROMPT,
    build_grounding_verification_prompt,
)
from app.services.analysis.summarizer import AnalysisChunk, DocumentSummarizer


def make_chunk(text: str, sequence_number: int = 1) -> AnalysisChunk:
    """Create one deterministic-shaped source chunk for verifier tests."""

    return AnalysisChunk(
        document_id=uuid4(),
        chunk_id=uuid4(),
        sequence_number=sequence_number,
        text=text,
        start_page=sequence_number,
        end_page=sequence_number,
        section_heading="Employment Notice Policy",
        filename="manual_module5_employment.pdf",
    )


class FakeGroundingProvider:
    """Record text/JSON calls while returning scripted local-model output."""

    model = "test/module11-grounding-model"

    def __init__(
        self,
        *,
        summaries: list[str] | None = None,
        structured: list[dict[str, object]] | None = None,
    ) -> None:
        self.summaries = list(summaries or [])
        self.structured = list(structured or [])
        self.generate_calls: list[tuple[str, str]] = []
        self.json_calls: list[tuple[str, str]] = []

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.generate_calls.append((system_prompt, user_prompt))
        return self.summaries.pop(0)

    async def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.json_calls.append((system_prompt, user_prompt))
        return self.structured.pop(0)


def supported_claim(
    claim: str = "Employees must give thirty days written notice.",
    source_labels: list[str] | None = None,
    supporting_evidence: str | None = None,
) -> dict[str, object]:
    """Build one supported verifier claim."""

    return {
        "claim": claim,
        "supported": True,
        "source_labels": source_labels or ["S1"],
        "supporting_evidence": supporting_evidence
        or claim.replace(" [S1]", "").replace(" [S2]", ""),
        "reason": "Directly stated by S1.",
    }


def unsupported_claim(claim: str, reason: str = "The evidence does not establish this.") -> dict[str, object]:
    """Build one unsupported verifier claim."""

    return {
        "claim": claim,
        "supported": False,
        "source_labels": [],
        "reason": reason,
    }


@pytest.mark.parametrize(
    ("evidence", "draft"),
    [
        (
            "Employees must give thirty days written notice before resignation. "
            "Invoice reference INV-2026-0043.",
            "The notice period is referenced in invoice INV-2026-0043.",
        ),
        (
            "Employees must give written notice.",
            "Employers must acknowledge and process employee resignation notices.",
        ),
        (
            "Employment Notice Policy\nEmployees must give written notice.",
            "The purpose of the policy is to define the formal resignation process.",
        ),
        (
            "Invoice reference INV-2026-0043.",
            "INV-2026-0043 indicates that the transaction occurred in 2026.",
        ),
    ],
)
def test_verifier_rejects_unsupported_claim_classes(evidence: str, draft: str) -> None:
    """Adjacency, actor, purpose, and identifier interpretations are unsupported."""

    provider = FakeGroundingProvider(
        structured=[
            {
                "claims": [unsupported_claim(draft)],
                "has_unsupported_claims": True,
                "repaired_summary": "",
            }
        ]
    )
    result = asyncio.run(
        GroundingVerifier(provider).verify(
            draft,
            [make_chunk(evidence)],
            style=SummaryStyle.BRIEF,
        )
    )

    assert result.has_unsupported_claims is True
    assert result.claims[0].supported is False


def test_verifier_accepts_valid_paraphrase_and_explicit_relationship() -> None:
    """Faithful paraphrases and source-stated relationships remain supported."""

    evidence = "Invoice INV-2026-0043 records the employee's thirty-day resignation notice."
    draft = "Invoice INV-2026-0043 records the thirty-day resignation notice."
    provider = FakeGroundingProvider(
        structured=[
            {
                "claims": [
                    supported_claim(
                        draft,
                        supporting_evidence=(
                            "Invoice INV-2026-0043 records the employee's "
                            "thirty-day resignation notice."
                        ),
                    )
                ],
                "has_unsupported_claims": False,
                "repaired_summary": "",
            }
        ]
    )
    result = asyncio.run(
        GroundingVerifier(provider).verify(
            draft,
            [make_chunk(evidence)],
            style=SummaryStyle.BRIEF,
        )
    )

    assert result.has_unsupported_claims is False
    assert result.claims[0].source_labels == ["S1"]


def test_verifier_prompt_defines_claim_level_rules() -> None:
    """The verifier prompt states the safety rules and keeps evidence separate."""

    prompt = build_grounding_verification_prompt(
        draft_summary="The invoice specifies the notice period.",
        evidence="[S1]\nContent: Employees must give written notice. Invoice reference INV-2026-0043.",
        style=SummaryStyle.DETAILED,
    ).lower()

    assert "factual grounding verifier" in GROUNDING_VERIFIER_SYSTEM_PROMPT.lower()
    assert "sentence proximity never proves a relationship" in GROUNDING_VERIFIER_SYSTEM_PROMPT.lower()
    assert "do not infer document purpose" in GROUNDING_VERIFIER_SYSTEM_PROMPT.lower()
    assert "do not treat the draft as evidence" in prompt
    assert "source_evidence" in prompt


def test_verifier_rejects_unknown_or_unattributed_supported_claim() -> None:
    """Supported claims must cite supplied sources and cannot invent source labels."""

    unknown_provider = FakeGroundingProvider(
        structured=[
            {
                "claims": [
                    {
                        **supported_claim(),
                        "source_labels": ["S9"],
                    }
                ],
                "has_unsupported_claims": False,
            }
        ]
    )
    with pytest.raises(AnalysisResponseError, match="source"):
        asyncio.run(
            GroundingVerifier(unknown_provider).verify(
                "Supported claim [S9].",
                [make_chunk("Evidence")],
                style=SummaryStyle.BRIEF,
            )
        )

    unattributed_provider = FakeGroundingProvider(
        structured=[
            {
                "claims": [
                    {
                        **supported_claim(),
                        "source_labels": [],
                    }
                ],
                "has_unsupported_claims": False,
            }
        ]
    )
    with pytest.raises(AnalysisResponseError, match="without source"):
        asyncio.run(
            GroundingVerifier(unattributed_provider).verify(
                "Supported claim.",
                [make_chunk("Evidence")],
                style=SummaryStyle.BRIEF,
            )
        )

    fabricated_evidence_provider = FakeGroundingProvider(
        structured=[
            {
                "claims": [
                    supported_claim(
                        supporting_evidence="This sentence is not in the source."
                    )
                ],
                "has_unsupported_claims": False,
            }
        ]
    )
    with pytest.raises(AnalysisResponseError, match="supporting evidence"):
        asyncio.run(
            GroundingVerifier(fabricated_evidence_provider).verify(
                "Supported claim.",
                [make_chunk("Evidence")],
                style=SummaryStyle.BRIEF,
            )
        )


def test_repair_pass_is_verified_and_returns_repaired_summary() -> None:
    """One failed verification is repaired and checked again before returning."""

    provider = FakeGroundingProvider(
        summaries=["The notice is referenced in invoice INV-2026-0043."],
        structured=[
            {
                "claims": [unsupported_claim("The notice is referenced in invoice INV-2026-0043.")],
                "has_unsupported_claims": True,
            },
            {"repaired_summary": "Employees must give thirty days written notice. [S1]"},
            {
                "claims": [supported_claim()],
                "has_unsupported_claims": False,
            },
        ],
    )
    result = asyncio.run(
        DocumentSummarizer(
            provider,
            grounding_verifier=GroundingVerifier(provider),
            grounding_enabled=True,
            grounding_max_passes=2,
        ).summarize(
            [
                make_chunk(
                    "Employees must give thirty days written notice before resignation. "
                    "Invoice reference INV-2026-0043."
                )
            ],
            style=SummaryStyle.BRIEF,
            batch_max_chars=2000,
            final_max_chars=2000,
        )
    )

    assert result.summary == "Employees must give thirty days written notice. [S1]"
    assert len(provider.json_calls) == 3
    assert result.grounding_verification_passes == 2
    assert result.grounding_repair_time_ms >= 0


def test_bounded_grounding_failure_returns_extractive_fallback_without_third_pass() -> None:
    """Two failed verification passes return evidence-derived fallback text only."""

    unsafe = "The invoice defines the resignation process."
    provider = FakeGroundingProvider(
        summaries=[unsafe],
        structured=[
            {
                "claims": [unsupported_claim(unsafe)],
                "has_unsupported_claims": True,
            },
            {"repaired_summary": unsafe},
            {
                "claims": [unsupported_claim(unsafe)],
                "has_unsupported_claims": True,
            },
        ],
    )
    result = asyncio.run(
        DocumentSummarizer(
            provider,
            grounding_verifier=GroundingVerifier(provider),
            grounding_enabled=True,
            grounding_max_passes=2,
        ).summarize(
            [make_chunk("Employees must give written notice. Invoice reference INV-2026-0043.")],
            style=SummaryStyle.BRIEF,
            batch_max_chars=2000,
            final_max_chars=2000,
        )
    )

    assert "The invoice defines" not in result.summary
    assert "Employees must give written notice." in result.summary
    assert "Invoice reference INV-2026-0043." in result.summary
    assert len(provider.json_calls) == 3
    assert result.grounding_verification_passes == 2


def test_bullet_summary_has_one_markdown_item_per_line() -> None:
    """Provider bullet markers are normalized to separate Markdown lines."""

    provider = FakeGroundingProvider(
        summaries=["• Employees must give written notice. • Identifier/reference mentioned: INV-2026-0043."],
        structured=[
            {
                "claims": [
                    supported_claim("Employees must give written notice."),
                    supported_claim(
                        "Identifier/reference mentioned: INV-2026-0043.",
                        supporting_evidence="Invoice reference INV-2026-0043.",
                    ),
                ],
                "has_unsupported_claims": False,
            }
        ],
    )
    result = asyncio.run(
        DocumentSummarizer(
            provider,
            grounding_verifier=GroundingVerifier(provider),
            grounding_enabled=True,
        ).summarize(
            [make_chunk("Employees must give written notice. Invoice reference INV-2026-0043.")],
            style=SummaryStyle.BULLET_POINTS,
            batch_max_chars=2000,
            final_max_chars=2000,
        )
    )

    lines = [line for line in result.summary.splitlines() if line.strip().startswith("-")]
    assert len(lines) == 2
    assert "notice" in lines[0]
    assert "INV-2026-0043" in lines[1]


def test_hierarchical_summary_verifies_partials_before_final_synthesis() -> None:
    """Only repaired/verified partial summaries are passed into final synthesis."""

    provider = FakeGroundingProvider(
        summaries=[
            "The notice is referenced in invoice INV-2026-0043.",
            "Employees must give written notice. [S2]",
            "Employees must give written notice. [S1] [S2]",
        ],
        structured=[
            {
                "claims": [unsupported_claim("The notice is referenced in invoice INV-2026-0043.")],
                "has_unsupported_claims": True,
            },
            {"repaired_summary": "Employees must give written notice. [S1]"},
            {"claims": [supported_claim()], "has_unsupported_claims": False},
            {"claims": [supported_claim("Employees must give written notice. [S2]", ["S2"])], "has_unsupported_claims": False},
            {"claims": [supported_claim("Employees must give written notice. [S1] [S2]")], "has_unsupported_claims": False},
        ],
    )
    result = asyncio.run(
        DocumentSummarizer(
            provider,
            grounding_verifier=GroundingVerifier(provider),
            grounding_enabled=True,
        ).summarize(
            [make_chunk("Employees must give written notice. Invoice reference INV-2026-0043.", 1), make_chunk("Employees must give written notice.", 2)],
            style=SummaryStyle.BRIEF,
            batch_max_chars=180,
            final_max_chars=500,
        )
    )

    assert "The notice is referenced" not in provider.generate_calls[-1][1]
    assert "Employees must give written notice" in provider.generate_calls[-1][1]
    assert "The notice is referenced" not in result.summary
    assert result.grounding_verification_passes == 4


def test_grounding_settings_default_to_enabled_and_two_passes() -> None:
    """Grounding protection is enabled by default and explicitly bounded."""

    settings = Settings()
    assert settings.summary_grounding_enabled is True
    assert settings.summary_grounding_max_passes == 2


def test_grounding_response_flag_cannot_hide_unsupported_claim() -> None:
    """Structured verifier output cannot mark an unsupported claim as clean."""

    with pytest.raises(ValueError, match="has_unsupported_claims"):
        GroundingVerificationResponse(
            claims=[unsupported_claim("unsupported")],
            has_unsupported_claims=False,
        )


REGRESSION_EVIDENCE = (
    "Employees must give thirty days written notice before resignation. "
    "Invoice reference INV-2026-0043."
)


REGRESSION_UNSUPPORTED_CLAIMS = [
    "The Employment Notice Policy is a crucial aspect of the manual.",
    "The policy outlines formal procedures.",
    "The notice period is a mandatory requirement.",
    "Employees must provide written notice to their employer.",
    "Employees must specify their intention to resign.",
    "The employee must continue to perform their duties and responsibilities during the notice period.",
    "The invoice is supporting evidence for the notice period requirement.",
    "The notice period is referenced in invoice INV-2026-0043.",
    "The policy is a critical component of the manual.",
]


def approved_by_incorrect_verifier(summary: str) -> dict[str, object]:
    """Simulate the faulty verifier behavior seen in the real local request."""

    return {
        "claims": [supported_claim(summary, supporting_evidence=REGRESSION_EVIDENCE)],
        "has_unsupported_claims": False,
        "repaired_summary": "",
    }


def test_single_chunk_summary_has_final_safe_boundary_even_without_synthesis() -> None:
    """A one-chunk response cannot bypass deterministic grounding after partial verification."""

    unsafe = " ".join(REGRESSION_UNSUPPORTED_CLAIMS)
    provider = FakeGroundingProvider(
        summaries=[unsafe],
        structured=[approved_by_incorrect_verifier(unsafe)],
    )
    result = asyncio.run(
        DocumentSummarizer(
            provider,
            grounding_verifier=GroundingVerifier(provider),
            grounding_enabled=True,
        ).summarize(
            [make_chunk(REGRESSION_EVIDENCE)],
            style=SummaryStyle.DETAILED,
            batch_max_chars=2000,
            final_max_chars=2000,
        )
    )

    assert result.final_synthesis_time_ms == 0.0
    assert result.summary == (
        "Employees must give thirty days written notice before resignation. [S1] "
        "Invoice reference INV-2026-0043. [S1]"
    )
    assert all(claim not in result.summary for claim in REGRESSION_UNSUPPORTED_CLAIMS)


def test_detailed_summary_rejects_inferred_content_when_verifier_approves_it() -> None:
    """Detailed formatting cannot turn absent policy meaning into grounded content."""

    unsafe = "The policy is a critical component of the manual. The policy outlines formal procedures."
    provider = FakeGroundingProvider(
        summaries=[unsafe],
        structured=[approved_by_incorrect_verifier(unsafe)],
    )
    result = asyncio.run(
        DocumentSummarizer(
            provider,
            grounding_verifier=GroundingVerifier(provider),
            grounding_enabled=True,
        ).summarize(
            [make_chunk(REGRESSION_EVIDENCE)],
            style=SummaryStyle.DETAILED,
            batch_max_chars=2000,
            final_max_chars=2000,
        )
    )

    assert result.summary == (
        "Employees must give thirty days written notice before resignation. [S1] "
        "Invoice reference INV-2026-0043. [S1]"
    )
    assert "critical" not in result.summary.lower()
    assert "formal procedures" not in result.summary.lower()


def test_bullet_summary_keeps_invoice_as_independent_fact() -> None:
    """Bullet output cannot connect the invoice to the notice rule through formatting."""

    unsafe = (
        "• Employees must give thirty days written notice before resignation. "
        "• The notice period is referenced in invoice INV-2026-0043."
    )
    provider = FakeGroundingProvider(
        summaries=[unsafe],
        structured=[approved_by_incorrect_verifier(unsafe)],
    )
    result = asyncio.run(
        DocumentSummarizer(
            provider,
            grounding_verifier=GroundingVerifier(provider),
            grounding_enabled=True,
        ).summarize(
            [make_chunk(REGRESSION_EVIDENCE)],
            style=SummaryStyle.BULLET_POINTS,
            batch_max_chars=2000,
            final_max_chars=2000,
        )
    )

    assert result.summary.splitlines() == [
        "- Employees must give thirty days written notice before resignation. [S1]",
        "- Invoice reference INV-2026-0043. [S1]",
    ]
    assert "referenced in invoice" not in result.summary.lower()


def test_brief_summary_does_not_link_invoice_to_notice_requirement() -> None:
    """Brief style may preserve the notice rule but cannot invent invoice purpose."""

    unsafe = "Employees must give thirty days written notice before resignation. The invoice supports the notice requirement."
    provider = FakeGroundingProvider(
        summaries=[unsafe],
        structured=[approved_by_incorrect_verifier(unsafe)],
    )
    result = asyncio.run(
        DocumentSummarizer(
            provider,
            grounding_verifier=GroundingVerifier(provider),
            grounding_enabled=True,
        ).summarize(
            [make_chunk(REGRESSION_EVIDENCE)],
            style=SummaryStyle.BRIEF,
            batch_max_chars=2000,
            final_max_chars=2000,
        )
    )

    assert result.summary == (
        "Employees must give thirty days written notice before resignation. [S1] "
        "Invoice reference INV-2026-0043. [S1]"
    )
    assert "supports the notice requirement" not in result.summary.lower()


def test_multi_chunk_final_synthesis_has_safe_boundary_after_grounded_partials() -> None:
    """Final synthesis is checked independently after every partial is grounded."""

    provider = FakeGroundingProvider(
        summaries=[
            "The policy is crucial.",
            "The invoice supports the notice period.",
            "The policy is a critical component of the manual.",
        ],
        structured=[
            approved_by_incorrect_verifier("The policy is crucial."),
            approved_by_incorrect_verifier("The invoice supports the notice period."),
            approved_by_incorrect_verifier("The policy is a critical component of the manual."),
        ],
    )
    result = asyncio.run(
        DocumentSummarizer(
            provider,
            grounding_verifier=GroundingVerifier(provider),
            grounding_enabled=True,
        ).summarize(
            [
                make_chunk("Employees must give thirty days written notice before resignation.", 1),
                make_chunk("Invoice reference INV-2026-0043.", 2),
            ],
            style=SummaryStyle.BRIEF,
            batch_max_chars=220,
            final_max_chars=2000,
        )
    )

    assert "critical" not in result.summary.lower()
    assert "supports" not in result.summary.lower()
    assert "thirty days written notice" in result.summary
    assert "Invoice reference INV-2026-0043" in result.summary


class TimeoutGroundingProvider:
    """Simulate an unavailable verifier provider after draft generation."""

    model = "test/timeout-grounding-model"

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return "The policy is crucial."

    async def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        raise OllamaServiceError("The verifier timed out.")


def test_verifier_timeout_fails_closed_to_extractive_fallback() -> None:
    """A verifier timeout cannot return the raw generated draft."""

    result = asyncio.run(
        DocumentSummarizer(
            TimeoutGroundingProvider(),
            grounding_verifier=GroundingVerifier(TimeoutGroundingProvider()),
            grounding_enabled=True,
        ).summarize(
            [make_chunk(REGRESSION_EVIDENCE)],
            style=SummaryStyle.BRIEF,
            batch_max_chars=2000,
            final_max_chars=2000,
        )
    )

    assert "crucial" not in result.summary.lower()
    assert "Employees must give thirty days written notice before resignation." in result.summary
    assert "Invoice reference INV-2026-0043." in result.summary
