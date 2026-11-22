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


def test_evidence_provenance_matches_stored_observation(tmp_path):
    store = MarketStore(tmp_path / "provenance.sqlite")

    timestamp = datetime(2026, 11, 20, tzinfo=timezone.utc)
    cutoff = datetime.now(timezone.utc) + timedelta(minutes=1)

    bar = MarketBar(
        symbol="NVDA",
        timeframe="1d",
        timestamp=timestamp,
        available_at=timestamp + timedelta(days=1),
        open=150.0,
        high=155.0,
        low=148.0,
        close=152.0,
        volume=1_000_000,
        source="provenance_test",
        adjusted=False,
    )

    assert store.save_bars([bar]) == 1

    evidence = store.load_evidence_as_of("NVDA", cutoff)
    prices = store.load_bars_as_of("NVDA", cutoff)

    assert len(evidence) == 1
    assert len(prices) == 1

    record = evidence[0]
    price = prices[0]

    assert record.observation_id > 0
    assert record.symbol == price.symbol
    assert record.timeframe == price.timeframe.value
    assert record.timestamp == price.timestamp
    assert record.available_at == price.available_at
    assert record.source == price.source
    assert record.adjusted == price.adjusted
    assert record.retrieved_at <= cutoff


def test_provenance_tracks_historical_revisions(tmp_path):
    import sqlite3

    store = MarketStore(tmp_path / "revisions.sqlite")

    timestamp = datetime(2026, 11, 1, tzinfo=timezone.utc)
    first_retrieval = datetime(2026, 11, 2, tzinfo=timezone.utc)
    second_retrieval = datetime(2026, 11, 4, tzinfo=timezone.utc)

    def make_bar(close):
        return MarketBar(
            symbol="NVDA",
            timeframe="1d",
            timestamp=timestamp,
            available_at=timestamp + timedelta(days=1),
            open=100.0,
            high=110.0,
            low=95.0,
            close=close,
            volume=1_000_000,
            source="revision_test",
            adjusted=False,
        )

    assert store.save_bars([make_bar(100.0)]) == 1
    assert store.save_bars([make_bar(105.0)]) == 1

    with sqlite3.connect(store.path) as connection:
        rows = connection.execute(
            "SELECT id FROM market_bars ORDER BY id"
        ).fetchall()

        assert len(rows) == 2

        original_id = rows[0][0]
        revised_id = rows[1][0]

        connection.execute(
            "UPDATE market_bars SET retrieved_at = ? WHERE id = ?",
            (first_retrieval.isoformat(), original_id),
        )
        connection.execute(
            "UPDATE market_bars SET retrieved_at = ? WHERE id = ?",
            (second_retrieval.isoformat(), revised_id),
        )

    early_cutoff = datetime(2026, 11, 3, tzinfo=timezone.utc)
    late_cutoff = datetime(2026, 11, 5, tzinfo=timezone.utc)

    early_prices = store.load_bars_as_of("NVDA", early_cutoff)
    early_evidence = store.load_evidence_as_of("NVDA", early_cutoff)

    late_prices = store.load_bars_as_of("NVDA", late_cutoff)
    late_evidence = store.load_evidence_as_of("NVDA", late_cutoff)

    assert len(early_prices) == len(early_evidence) == 1
    assert len(late_prices) == len(late_evidence) == 1

    assert early_prices[0].close == 100.0
    assert early_evidence[0].observation_id == original_id

    assert late_prices[0].close == 105.0
    assert late_evidence[0].observation_id == revised_id

    assert original_id != revised_id


def test_market_store_normalizes_timestamps_to_utc(tmp_path):
    import sqlite3

    store = MarketStore(tmp_path / "utc_normalization.sqlite")

    eastern = timezone(timedelta(hours=-5))

    timestamp = datetime(2026, 11, 20, 10, tzinfo=eastern)
    available_at = timestamp + timedelta(days=1)

    bar = MarketBar(
        symbol="NVDA",
        timeframe="1d",
        timestamp=timestamp,
        available_at=available_at,
        open=150.0,
        high=155.0,
        low=148.0,
        close=152.0,
        volume=1_000_000,
        source="timezone_test",
        adjusted=False,
    )

    assert store.save_bars([bar]) == 1

    with sqlite3.connect(store.path) as connection:
        stored = connection.execute(
            "SELECT timestamp, available_at, retrieved_at FROM market_bars"
        ).fetchone()

    for value in stored:
        parsed = datetime.fromisoformat(value)
        assert parsed.utcoffset() == timedelta(0)

    assert datetime.fromisoformat(stored[0]) == timestamp
    assert datetime.fromisoformat(stored[1]) == available_at


def test_equivalent_timezone_timestamps_share_one_candle(tmp_path):
    store = MarketStore(tmp_path / "timezone_equivalence.sqlite")

    utc = datetime(2026, 11, 20, 15, tzinfo=timezone.utc)
    eastern = datetime(
        2026, 11, 20, 10,
        tzinfo=timezone(timedelta(hours=-5)),
    )

    assert utc == eastern

    def make_bar(timestamp):
        return MarketBar(
            symbol="NVDA",
            timeframe="1d",
            timestamp=timestamp,
            available_at=timestamp + timedelta(days=1),
            open=150.0,
            high=155.0,
            low=148.0,
            close=152.0,
            volume=1_000_000,
            source="timezone_equivalence_test",
            adjusted=False,
        )

    inserted = store.save_bars([
        make_bar(utc),
        make_bar(eastern),
    ])

    assert inserted == 1
    assert store.count_bars("NVDA") == 1

    bars = store.load_bars("NVDA")

    assert len(bars) == 1
    assert bars[0].timestamp == utc
    assert bars[0].timestamp.utcoffset() == timedelta(0)


def test_historical_observations_pair_prices_with_provenance(tmp_path):
    store = MarketStore(tmp_path / "paired_observations.sqlite")

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
        source="paired_test",
        adjusted=False,
    )

    revised = original.model_copy(update={"close": 153.0})

    assert store.save_bars([original]) == 1
    assert store.save_bars([revised]) == 1

    with store._connect() as connection:
        connection.execute(
            "UPDATE market_bars SET retrieved_at = ? WHERE id = 1",
            ("2026-11-21T12:00:00+00:00",),
        )
        connection.execute(
            "UPDATE market_bars SET retrieved_at = ? WHERE id = 2",
            ("2026-11-22T12:00:00+00:00",),
        )

    early = store.load_observations_as_of(
        "NVDA",
        datetime(2026, 11, 21, 18, tzinfo=timezone.utc),
    )

    late = store.load_observations_as_of(
        "NVDA",
        datetime(2026, 11, 23, tzinfo=timezone.utc),
    )

    assert len(early) == len(late) == 1

    early_bar, early_evidence = early[0]
    late_bar, late_evidence = late[0]

    assert early_bar.close == 152.0
    assert late_bar.close == 153.0

    assert early_evidence.observation_id == 1
    assert late_evidence.observation_id == 2

    for bar, evidence in (early[0], late[0]):
        assert bar.symbol == evidence.symbol
        assert bar.timeframe.value == evidence.timeframe
        assert bar.timestamp == evidence.timestamp
        assert bar.available_at == evidence.available_at
        assert bar.source == evidence.source
        assert bar.adjusted == evidence.adjusted
