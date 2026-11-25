from dataclasses import dataclass
from enum import Enum

from app.evidence.provenance import AvailabilityBasis


class ResearchPolicy(str, Enum):
    EXPLORATORY = "exploratory"
    STRICT = "strict"


@dataclass(frozen=True)
class AvailabilityGateResult:
    policy: ResearchPolicy
    approved_for_research: bool
    reason: str
    limitations: tuple[str, ...] = ()


def evaluate_availability(
    bases: tuple[AvailabilityBasis, ...],
    *,
    policy: ResearchPolicy,
) -> AvailabilityGateResult:
    """Evaluate availability evidence for a research purpose."""

    policy = ResearchPolicy(policy)

    if not bases:
        return AvailabilityGateResult(
            policy=policy,
            approved_for_research=False,
            reason="No availability evidence supplied.",
        )

    try:
        bases = tuple(AvailabilityBasis(basis) for basis in bases)
    except (ValueError, TypeError):
        return AvailabilityGateResult(
            policy=policy,
            approved_for_research=False,
            reason="Invalid availability evidence classification.",
        )

    if AvailabilityBasis.UNKNOWN in bases:
        return AvailabilityGateResult(
            policy=policy,
            approved_for_research=False,
            reason="Availability evidence contains unknown classifications.",
        )

    if policy == ResearchPolicy.STRICT:
        if any(
            basis != AvailabilityBasis.INDEPENDENTLY_VERIFIED
            for basis in bases
        ):
            return AvailabilityGateResult(
                policy=policy,
                approved_for_research=False,
                reason=(
                    "Strict research requires independently "
                    "verified availability for every observation."
                ),
            )

        return AvailabilityGateResult(
            policy=policy,
            approved_for_research=True,
            reason="All observations have independently verified availability classifications.",
        )

    limitations = []

    if AvailabilityBasis.ESTIMATED_POLICY in bases:
        limitations.append(
            "Estimated availability is not proof of historical "
            "information availability."
        )

    if AvailabilityBasis.PROVIDER_REPORTED in bases:
        limitations.append(
            "Provider-reported availability has not been "
            "independently verified."
        )

    return AvailabilityGateResult(
        policy=policy,
        approved_for_research=True,
        reason="Availability evidence meets exploratory research policy.",
        limitations=tuple(limitations),
    )
