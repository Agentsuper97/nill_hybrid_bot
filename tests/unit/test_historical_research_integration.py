from datetime import datetime, timedelta, timezone

from app.data.validated.research_loader import load_approved_market_data
from app.domain.market_data import MarketBar
from app.evidence.quality_gate import EvidenceDecision
from app.storage.market_store import MarketStore


def test_historical_research_integrity(tmp_path):
    store = MarketStore(tmp_path / "historical_test.sqlite")
    utc = timezone.utc

    def make_bar(day, close, available_day):
        timestamp = datetime(2026, 11, day, tzinfo=utc)
        return MarketBar(
            symbol="NVDA",
            timeframe="1d",
            timestamp=timestamp,
            available_at=datetime(
                2026, 11, available_day, tzinfo=utc
            ),
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=1_000_000,
            source="integration_test",
            adjusted=False,
            availability_basis="estimated_policy",
        )

    original = make_bar(2, 100, 3)
    revised = make_bar(2, 101, 3)
    second = make_bar(3, 102, 4)
    future = make_bar(5, 80, 6)

    assert store.save_bars([original, second, future]) == 3
    assert store.save_bars([revised]) == 1

    with store._connect() as connection:
        retrieval_times = [
            "2026-11-03T12:00:00+00:00",
            "2026-11-04T12:00:00+00:00",
            "2026-11-06T12:00:00+00:00",
            "2026-11-07T12:00:00+00:00",
        ]
        for row_id, retrieved_at in enumerate(
            retrieval_times, start=1
        ):
            connection.execute(
                "UPDATE market_bars SET retrieved_at = ? WHERE id = ?",
                (retrieved_at, row_id),
            )

    historical = load_approved_market_data(
        "NVDA",
        store=store,
        as_of=datetime(2026, 11, 4, 18, tzinfo=utc),
    )

    assert historical.gate.decision == EvidenceDecision.ALLOW
    assert [bar.close for bar in historical.bars] == [100, 102]

    later = load_approved_market_data(
        "NVDA",
        store=store,
        as_of=datetime(2026, 11, 8, tzinfo=utc),
    )

    assert later.gate.decision == EvidenceDecision.REVIEW
    assert later.bars == ()

    print("PASS: Historical data excludes later revisions.")
    print("PASS: Future observations stay outside earlier research.")
    print("PASS: Later quality warnings prevent data approval.")


def test_yahoo_candle_is_withheld_until_available_and_retrieved(
    tmp_path, monkeypatch
):
    import pandas as pd

    from app.ingestion.yahoo import fetch_daily_bars

    data = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [103.0],
            "Low": [99.0],
            "Close": [102.0],
            "Volume": [1_000_000],
        },
        index=pd.DatetimeIndex(["2026-11-20"]),
    )

    monkeypatch.setattr(
        "app.ingestion.yahoo.yf.download",
        lambda *args, **kwargs: data.copy(),
    )

    bars = fetch_daily_bars("NVDA")
    assert len(bars) == 1

    store = MarketStore(tmp_path / "yahoo_availability_test.sqlite")
    assert store.save_bars(bars) == 1

    # Simulate when this observation entered our database.
    retrieved_at = "2026-11-21T12:00:00+00:00"

    with store._connect() as connection:
        connection.execute(
            "UPDATE market_bars SET retrieved_at = ?",
            (retrieved_at,),
        )

    before = load_approved_market_data(
        "NVDA",
        store=store,
        as_of=datetime(2026, 11, 20, 15, tzinfo=timezone.utc),
    )

    assert before.bars == ()
    assert before.evidence == ()
    assert before.gate.approved_for_research is False

    # Availability has passed, but retrieval has not occurred.
    before_retrieval = load_approved_market_data(
        "NVDA",
        store=store,
        as_of=datetime(2026, 11, 21, 8, tzinfo=timezone.utc),
    )

    assert before_retrieval.bars == ()
    assert before_retrieval.evidence == ()
    assert before_retrieval.gate.approved_for_research is False

    after = load_approved_market_data(
        "NVDA",
        store=store,
        as_of=datetime(2026, 11, 21, 13, tzinfo=timezone.utc),
    )

    assert after.gate.decision == EvidenceDecision.ALLOW
    assert len(after.bars) == 1
    assert len(after.evidence) == 1
    assert after.bars[0].close == 102.0
    assert after.evidence[0].retrieved_at == datetime(
        2026, 11, 21, 12, tzinfo=timezone.utc
    )


def test_yahoo_candle_is_withheld_even_if_retrieved_early(
    tmp_path, monkeypatch
):
    import pandas as pd

    from app.ingestion.yahoo import fetch_daily_bars

    data = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [103.0],
            "Low": [99.0],
            "Close": [102.0],
            "Volume": [1_000_000],
        },
        index=pd.DatetimeIndex(["2026-11-20"]),
    )

    monkeypatch.setattr(
        "app.ingestion.yahoo.yf.download",
        lambda *args, **kwargs: data.copy(),
    )

    bars = fetch_daily_bars("NVDA")
    store = MarketStore(tmp_path / "early_retrieval_test.sqlite")
    assert store.save_bars(bars) == 1

    with store._connect() as connection:
        connection.execute(
            "UPDATE market_bars SET retrieved_at = ?",
            ("2026-11-21T03:00:00+00:00",),
        )

    # Retrieved at 03:00, but estimated availability is 05:00.
    before_availability = load_approved_market_data(
        "NVDA",
        store=store,
        as_of=datetime(2026, 11, 21, 4, tzinfo=timezone.utc),
    )

    assert before_availability.bars == ()
    assert before_availability.evidence == ()
    assert before_availability.gate.approved_for_research is False

    after_availability = load_approved_market_data(
        "NVDA",
        store=store,
        as_of=datetime(2026, 11, 21, 6, tzinfo=timezone.utc),
    )

    assert after_availability.gate.decision == EvidenceDecision.ALLOW
    assert len(after_availability.bars) == 1
    assert len(after_availability.evidence) == 1
    assert after_availability.bars[0].available_at == datetime(
        2026, 11, 21, 5, tzinfo=timezone.utc
    )
