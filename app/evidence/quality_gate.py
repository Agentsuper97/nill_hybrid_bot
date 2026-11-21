from dataclasses import dataclass
from enum import Enum

from app.data.validated.quality import QualityReport, QualityStatus


class EvidenceDecision(str, Enum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


@dataclass(frozen=True)
class EvidenceGateResult:
    decision: EvidenceDecision
    reason: str
    approved_for_research: bool


def evaluate_market_data(report: QualityReport) -> EvidenceGateResult:
    """Fail closed unless market data passes the quality gate."""

    if report.status == QualityStatus.VALID:
        return EvidenceGateResult(
            decision=EvidenceDecision.ALLOW,
            reason="Market data passed the current quality checks.",
            approved_for_research=True,
        )

    if report.status == QualityStatus.WARNING:
        return EvidenceGateResult(
            decision=EvidenceDecision.REVIEW,
            reason="Market data contains warnings requiring investigation.",
            approved_for_research=False,
        )

    return EvidenceGateResult(
        decision=EvidenceDecision.BLOCK,
        reason=f"Market data quality is {report.status.value}.",
        approved_for_research=False,
    )
