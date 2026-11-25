from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.market_data import MarketBar
from app.evidence.provenance import AvailabilityBasis, MarketEvidence
from app.evidence.availability_gate import ResearchPolicy
from app.evidence.quality_gate import EvidenceDecision
from app.data.validated.research_loader import load_approved_market_data


@pytest.mark.parametrize(
    "policy, bar_basis, evidence_basis",
    [
        (
            ResearchPolicy.EXPLORATORY,
            AvailabilityBasis.UNKNOWN,
            AvailabilityBasis.ESTIMATED_POLICY,
        ),
        (
            ResearchPolicy.EXPLORATORY,
            AvailabilityBasis.ESTIMATED_POLICY,
            AvailabilityBasis.PROVIDER_REPORTED,
        ),
        (
            ResearchPolicy.STRICT,
            AvailabilityBasis.ESTIMATED_POLICY,
            AvailabilityBasis.INDEPENDENTLY_VERIFIED,
        ),
        (
            ResearchPolicy.STRICT,
            AvailabilityBasis.UNKNOWN,
            AvailabilityBasis.INDEPENDENTLY_VERIFIED,
        ),
    ],
)
def test_loader_blocks_inconsistent_availability_evidence(
    policy, bar_basis, evidence_basis
):
    start = datetime(2026, 11, 2, tzinfo=timezone.utc)
    observations = []

    for index in range(3):
        timestamp = start + timedelta(days=index)

        bar = MarketBar(
            symbol="NVDA",
            timeframe="1d",
            timestamp=timestamp,
            available_at=timestamp + timedelta(days=1),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1_000_000,
            source="synthetic_consistency_test",
            availability_basis=bar_basis,
        )

        evidence = MarketEvidence(
            observation_id=index + 1,
            symbol=bar.symbol,
            timeframe=bar.timeframe.value,
            timestamp=bar.timestamp,
            available_at=bar.available_at,
            retrieved_at=bar.available_at,
            source=bar.source,
            adjusted=bar.adjusted,
            availability_basis=bar_basis,
        )

        evidence = replace(
            evidence,
            availability_basis=evidence_basis,
        )

        observations.append((bar, evidence))

    class InconsistentStore:
        def load_observations_as_of(self, symbol, as_of):
            return observations

    result = load_approved_market_data(
        "NVDA",
        store=InconsistentStore(),
        as_of=start + timedelta(days=10),
        policy=policy,
    )

    assert result.gate.decision == EvidenceDecision.BLOCK
    assert result.gate.approved_for_research is False
    assert result.availability_gate.approved_for_research is False
    assert "disagree" in result.availability_gate.reason.lower()
    assert result.bars == ()
    assert result.evidence == ()
