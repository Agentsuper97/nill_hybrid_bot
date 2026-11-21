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
