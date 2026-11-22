from datetime import datetime, timedelta, timezone

import pytest

from app.evidence.provenance import MarketEvidence


def make_evidence(**overrides):
    now = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)

    values = {
        "observation_id": 1,
        "symbol": "NVDA",
        "timeframe": "1d",
        "timestamp": now,
        "available_at": now,
        "retrieved_at": now,
        "source": "test",
        "adjusted": False,
    }

    values.update(overrides)
    return MarketEvidence(**values)


def test_valid_provenance():
    evidence = make_evidence()

    assert evidence.observation_id == 1
    assert evidence.symbol == "NVDA"
    assert evidence.source == "test"


def test_invalid_observation_id():
    with pytest.raises(ValueError, match="observation_id"):
        make_evidence(observation_id=0)


@pytest.mark.parametrize(
    "field",
    ["timestamp", "available_at", "retrieved_at"],
)
def test_naive_timestamps_rejected(field):
    naive = datetime(2026, 11, 21, 12)

    with pytest.raises(ValueError, match="timezone-aware"):
        make_evidence(**{field: naive})


def test_availability_before_observation_rejected():
    now = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="available_at"):
        make_evidence(
            timestamp=now,
            available_at=now - timedelta(hours=1),
        )
