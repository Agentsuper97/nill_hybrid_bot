import pytest

from app.evidence.provenance import AvailabilityBasis
from app.evidence.availability_gate import (
    ResearchPolicy,
    evaluate_availability,
)


@pytest.mark.parametrize(
    "policy, basis, approved",
    [
        (ResearchPolicy.EXPLORATORY, AvailabilityBasis.UNKNOWN, False),
        (ResearchPolicy.EXPLORATORY, AvailabilityBasis.ESTIMATED_POLICY, True),
        (ResearchPolicy.EXPLORATORY, AvailabilityBasis.PROVIDER_REPORTED, True),
        (ResearchPolicy.EXPLORATORY, AvailabilityBasis.INDEPENDENTLY_VERIFIED, True),
        (ResearchPolicy.STRICT, AvailabilityBasis.UNKNOWN, False),
        (ResearchPolicy.STRICT, AvailabilityBasis.ESTIMATED_POLICY, False),
        (ResearchPolicy.STRICT, AvailabilityBasis.PROVIDER_REPORTED, False),
        (ResearchPolicy.STRICT, AvailabilityBasis.INDEPENDENTLY_VERIFIED, True),
    ],
)
def test_availability_policy_decisions(policy, basis, approved):
    result = evaluate_availability((basis,), policy=policy)

    assert result.approved_for_research is approved
    assert result.reason


@pytest.mark.parametrize("policy", list(ResearchPolicy))
def test_empty_availability_evidence_is_withheld(policy):
    result = evaluate_availability((), policy=policy)

    assert result.approved_for_research is False
    assert result.reason


def test_exploratory_research_withholds_mixed_unknown_evidence():
    result = evaluate_availability(
        (
            AvailabilityBasis.INDEPENDENTLY_VERIFIED,
            AvailabilityBasis.UNKNOWN,
        ),
        policy=ResearchPolicy.EXPLORATORY,
    )

    assert result.approved_for_research is False


def test_strict_research_withholds_mixed_estimated_evidence():
    result = evaluate_availability(
        (
            AvailabilityBasis.INDEPENDENTLY_VERIFIED,
            AvailabilityBasis.ESTIMATED_POLICY,
        ),
        policy=ResearchPolicy.STRICT,
    )

    assert result.approved_for_research is False


def test_strict_research_accepts_multiple_verified_observations():
    result = evaluate_availability(
        (AvailabilityBasis.INDEPENDENTLY_VERIFIED,) * 3,
        policy=ResearchPolicy.STRICT,
    )

    assert result.approved_for_research is True


def test_estimated_evidence_reports_limitations():
    result = evaluate_availability(
        (AvailabilityBasis.ESTIMATED_POLICY,),
        policy=ResearchPolicy.EXPLORATORY,
    )

    assert result.approved_for_research is True
    assert result.limitations
