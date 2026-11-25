import pytest

from app.evidence.provenance import AvailabilityBasis
from app.evidence.availability_gate import (
    ResearchPolicy,
    evaluate_availability,
)


@pytest.mark.parametrize("policy", list(ResearchPolicy))
@pytest.mark.parametrize("invalid_basis", ["unverified", None])
def test_invalid_availability_classification_is_blocked(
    policy, invalid_basis
):
    result = evaluate_availability(
        (
            AvailabilityBasis.INDEPENDENTLY_VERIFIED,
            invalid_basis,
        ),
        policy=policy,
    )

    assert result.approved_for_research is False
    assert "invalid" in result.reason.lower()


@pytest.mark.parametrize(
    "invalid_policy",
    ["", "unknown", "STRICT", None],
)
def test_invalid_research_policy_is_rejected(invalid_policy):
    with pytest.raises(ValueError):
        evaluate_availability(
            (AvailabilityBasis.INDEPENDENTLY_VERIFIED,),
            policy=invalid_policy,
        )
