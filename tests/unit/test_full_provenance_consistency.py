from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.market_data import MarketBar
from app.evidence.provenance import AvailabilityBasis, MarketEvidence
from app.evidence.availability_gate import ResearchPolicy
from app.evidence.quality_gate import EvidenceDecision
from app.data.validated.research_loader import load_approved_market_data


START = datetime(2026, 11, 2, tzinfo=timezone.utc)
CUTOFF = START + timedelta(days=10)


def make_observations():
    observations = []

    for index in range(3):
        timestamp = START + timedelta(days=index)

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
            source="synthetic_provenance_test",
            adjusted=False,
            availability_basis=AvailabilityBasis.ESTIMATED_POLICY,
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
            availability_basis=bar.availability_basis,
        )

        observations.append((bar, evidence))

    return observations


class SyntheticStore:
    def __init__(self, observations):
        self.observations = observations

    def load_observations_as_of(self, symbol, as_of):
        return self.observations


def load(observations, symbol="NVDA"):
    return load_approved_market_data(
        symbol,
        store=SyntheticStore(observations),
        as_of=CUTOFF,
        policy=ResearchPolicy.EXPLORATORY,
    )


def assert_blocked(result):
    assert result.gate.decision == EvidenceDecision.BLOCK
    assert result.gate.approved_for_research is False
    assert result.bars == ()
    assert result.evidence == ()


@pytest.mark.parametrize(
    "field, value",
    [
        ("symbol", "AMD"),
        ("timeframe", "15m"),
        ("timestamp", START - timedelta(days=1)),
        ("available_at", START + timedelta(hours=12)),
        ("source", "different_provider"),
        ("adjusted", True),
    ],
)
def test_mismatched_provenance_is_blocked(field, value):
    observations = make_observations()
    bar, evidence = observations[0]

    observations[0] = (
        bar,
        replace(evidence, **{field: value}),
    )

    assert_blocked(load(observations))


def test_duplicate_evidence_identity_is_blocked():
    observations = make_observations()
    bar, evidence = observations[1]

    observations[1] = (
        bar,
        replace(evidence, observation_id=1),
    )

    assert_blocked(load(observations))


def test_future_availability_is_blocked():
    observations = make_observations()
    bar, evidence = observations[0]
    future = CUTOFF + timedelta(seconds=1)

    observations[0] = (
        bar.model_copy(update={"available_at": future}),
        replace(evidence, available_at=future),
    )

    assert_blocked(load(observations))


def test_future_retrieval_is_blocked():
    observations = make_observations()
    bar, evidence = observations[0]

    observations[0] = (
        bar,
        replace(
            evidence,
            retrieved_at=CUTOFF + timedelta(seconds=1),
        ),
    )

    assert_blocked(load(observations))


def test_wrong_requested_symbol_is_blocked():
    assert_blocked(load(make_observations(), symbol="AMD"))


def test_consistent_provenance_is_released():
    result = load(make_observations())

    assert result.gate.approved_for_research is True
    assert len(result.bars) == 3
    assert len(result.evidence) == 3
