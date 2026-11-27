from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.market_data import MarketBar
from app.evidence.provenance import (
    AvailabilityBasis,
    MarketEvidence,
)
from app.evidence.fingerprint import fingerprint_observation


START = datetime(2026, 11, 2, tzinfo=timezone.utc)


def make_observation():
    bar = MarketBar(
        symbol="NVDA",
        timeframe="1d",
        timestamp=START,
        available_at=START + timedelta(days=1),
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1_000_000,
        source="synthetic_defensive_test",
        adjusted=False,
        availability_basis=AvailabilityBasis.ESTIMATED_POLICY,
    )

    evidence = MarketEvidence(
        observation_id=1,
        symbol=bar.symbol,
        timeframe=bar.timeframe.value,
        timestamp=bar.timestamp,
        available_at=bar.available_at,
        retrieved_at=START + timedelta(days=2),
        source=bar.source,
        adjusted=bar.adjusted,
        availability_basis=bar.availability_basis,
    )

    return bar, evidence


@pytest.mark.parametrize(
    "field,value",
    [
        ("symbol", "AMD"),
        ("timeframe", "1wk"),
        ("timestamp", START - timedelta(days=1)),
        ("available_at", START + timedelta(days=2)),
        ("source", "different_provider"),
        ("adjusted", True),
        (
            "availability_basis",
            AvailabilityBasis.PROVIDER_REPORTED,
        ),
    ],
)
def test_changed_provenance_changes_fingerprint(field, value):
    bar, evidence = make_observation()
    updated = replace(evidence, **{field: value})

    assert fingerprint_observation(
        bar, evidence
    ) != fingerprint_observation(
        bar, updated
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("symbol", "AMD"),
        ("timeframe", "1wk"),
        ("timestamp", START - timedelta(days=1)),
        ("available_at", START + timedelta(days=2)),
        ("source", "different_provider"),
        ("adjusted", True),
        (
            "availability_basis",
            AvailabilityBasis.PROVIDER_REPORTED,
        ),
    ],
)
def test_changed_bar_metadata_changes_fingerprint(field, value):
    bar, evidence = make_observation()

    updated = MarketBar.model_validate(
        {**bar.model_dump(), field: value}
    )

    assert fingerprint_observation(
        bar, evidence
    ) != fingerprint_observation(
        updated, evidence
    )


@pytest.mark.parametrize("invalid", [None, {}, "invalid"])
def test_invalid_bar_type_is_rejected(invalid):
    _, evidence = make_observation()

    with pytest.raises(TypeError):
        fingerprint_observation(invalid, evidence)


@pytest.mark.parametrize("invalid", [None, {}, "invalid"])
def test_invalid_evidence_type_is_rejected(invalid):
    bar, _ = make_observation()

    with pytest.raises(TypeError):
        fingerprint_observation(bar, invalid)


@pytest.mark.parametrize(
    "invalid",
    [float("nan"), float("inf"), float("-inf")],
)
def test_non_finite_values_are_rejected(invalid):
    bar, evidence = make_observation()

    corrupted = bar.model_copy(
        update={"close": invalid}
    )

    with pytest.raises(ValueError):
        fingerprint_observation(corrupted, evidence)


def test_naive_timestamp_is_rejected():
    bar, evidence = make_observation()

    corrupted = bar.model_copy(
        update={"timestamp": START.replace(tzinfo=None)}
    )

    with pytest.raises(ValueError):
        fingerprint_observation(corrupted, evidence)


def test_signed_zero_has_identical_fingerprint():
    bar, evidence = make_observation()

    positive = bar.model_copy(update={"volume": 0.0})
    negative = bar.model_copy(update={"volume": -0.0})

    assert fingerprint_observation(
        positive, evidence
    ) == fingerprint_observation(
        negative, evidence
    )
