import pytest

from app.data.validated.quality import QualityReport, QualityStatus
from app.evidence.quality_gate import (
    EvidenceDecision,
    evaluate_market_data,
)


@pytest.mark.parametrize(
    "status, expected_decision, expected_approval",
    [
        (QualityStatus.VALID, EvidenceDecision.ALLOW, True),
        (QualityStatus.WARNING, EvidenceDecision.REVIEW, False),
        (QualityStatus.SUSPECT, EvidenceDecision.BLOCK, False),
        (QualityStatus.INVALID, EvidenceDecision.BLOCK, False),
    ],
)
def test_evidence_gate_decisions(
    status, expected_decision, expected_approval
):
    report = QualityReport(
        symbol="NVDA",
        status=status,
        bars_checked=23,
    )

    result = evaluate_market_data(report)

    assert result.decision == expected_decision
    assert result.approved_for_research is expected_approval
    assert result.reason
