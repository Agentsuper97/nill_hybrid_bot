from datetime import datetime, timedelta, timezone

import pytest

from app.domain.market_data import MarketBar
from app.storage.market_store import MarketStore


def test_historical_revision_firewall(tmp_path):
    store = MarketStore(tmp_path / "market.sqlite")

    timestamp = datetime(2026, 11, 20, tzinfo=timezone.utc)

    original = MarketBar(
        symbol="NVDA",
        timeframe="1d",
        timestamp=timestamp,
        available_at=timestamp + timedelta(days=1),
        open=150.0,
        high=155.0,
        low=148.0,
        close=152.0,
        volume=1_000_000,
        source="test",
        adjusted=False,
    )

    revised = original.model_copy(update={"close": 153.0})

    assert store.save_bars([original]) == 1
    assert store.save_bars([revised]) == 1
    assert store.count_bars("NVDA") == 2

    # Control retrieval timestamps in this isolated test database.
    with store._connect() as connection:
        connection.execute(
            "UPDATE market_bars SET retrieved_at = ? WHERE id = 1",
            ("2026-11-21T12:00:00+00:00",),
        )
        connection.execute(
            "UPDATE market_bars SET retrieved_at = ? WHERE id = 2",
            ("2026-11-22T12:00:00+00:00",),
        )

    before = store.load_bars_as_of(
        "NVDA",
        datetime(2026, 11, 21, 18, tzinfo=timezone.utc),
    )
    after = store.load_bars_as_of(
        "NVDA",
        datetime(2026, 11, 23, tzinfo=timezone.utc),
    )

    assert len(before) == 1
    assert before[0].close == 152.0

    assert len(after) == 1
    assert after[0].close == 153.0

    assert len(store.load_bars("NVDA")) == 1
    assert store.load_bars("NVDA")[0].close == 153.0


def test_as_of_rejects_naive_datetime(tmp_path):
    store = MarketStore(tmp_path / "market.sqlite")

    with pytest.raises(ValueError, match="timezone-aware"):
        store.load_bars_as_of("NVDA", datetime(2026, 11, 21))
