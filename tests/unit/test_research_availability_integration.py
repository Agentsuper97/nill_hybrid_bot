from datetime import datetime, timedelta, timezone

import pytest

from app.domain.market_data import MarketBar
from app.storage.market_store import MarketStore
from app.evidence.provenance import AvailabilityBasis
from app.evidence.availability_gate import ResearchPolicy
from app.data.validated.research_loader import load_approved_market_data


def make_store(tmp_path, bases, closes=None):
    store = MarketStore(tmp_path / "policy_test.sqlite")
    start = datetime(2026, 11, 2, tzinfo=timezone.utc)
    closes = closes if closes is not None else [100.0] * len(bases)

    bars = []

    for index, (basis, close) in enumerate(zip(bases, closes)):
        timestamp = start + timedelta(days=index)

        bars.append(
            MarketBar(
                symbol="NVDA",
                timeframe="1d",
                timestamp=timestamp,
                available_at=timestamp + timedelta(days=1),
                open=close,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=1_000_000,
                source="synthetic_policy_test",
                availability_basis=basis,
            )
        )

    assert store.save_bars(bars) == len(bars)

    return store


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
        (ResearchPolicy.STRICT, AvailabilityBasis.INDEPENDENTLY_VERIFIED, False),
    ],
)
def test_loader_enforces_availability_policy(
    tmp_path, policy, basis, approved
):
    store = make_store(tmp_path, [basis] * 3)

    result = load_approved_market_data(
        "NVDA",
        store=store,
        policy=policy,
    )

    assert result.gate.approved_for_research is approved
    assert result.availability_gate.approved_for_research is approved
    assert result.availability_gate.policy == policy

    if approved:
        assert len(result.bars) == 3
        assert len(result.evidence) == 3
    else:
        assert result.bars == ()
        assert result.evidence == ()


def test_default_policy_withholds_unknown_evidence(tmp_path):
    store = make_store(
        tmp_path, [AvailabilityBasis.UNKNOWN] * 3
    )

    result = load_approved_market_data("NVDA", store=store)

    assert result.gate.approved_for_research is False
    assert result.bars == ()
    assert result.evidence == ()


def test_default_policy_reports_estimated_limitations(tmp_path):
    store = make_store(
        tmp_path, [AvailabilityBasis.ESTIMATED_POLICY] * 3
    )

    result = load_approved_market_data("NVDA", store=store)

    assert result.gate.approved_for_research is True
    assert result.availability_gate.policy == ResearchPolicy.EXPLORATORY
    assert result.availability_gate.limitations


def test_strict_loader_withholds_mixed_evidence(tmp_path):
    store = make_store(
        tmp_path,
        [
            AvailabilityBasis.INDEPENDENTLY_VERIFIED,
            AvailabilityBasis.ESTIMATED_POLICY,
        ],
    )

    result = load_approved_market_data(
        "NVDA",
        store=store,
        policy=ResearchPolicy.STRICT,
    )

    assert result.gate.approved_for_research is False
    assert result.bars == ()
    assert result.evidence == ()


def test_approved_availability_cannot_override_quality_gate(tmp_path):
    store = make_store(
        tmp_path,
        [AvailabilityBasis.INDEPENDENTLY_VERIFIED] * 3,
        closes=[100.0, 101.0, 85.0],
    )

    result = load_approved_market_data(
        "NVDA",
        store=store,
        policy=ResearchPolicy.EXPLORATORY,
    )

    assert result.availability_gate.approved_for_research is True
    assert result.gate.approved_for_research is False
    assert result.bars == ()
    assert result.evidence == ()
