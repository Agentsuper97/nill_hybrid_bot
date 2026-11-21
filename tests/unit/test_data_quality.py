from datetime import datetime, timedelta, timezone

from app.domain.market_data import MarketBar
from app.data.validated.quality import (
    QualityStatus,
    check_market_bars,
)


def make_bar(day, close=100.0, symbol="NVDA"):
    timestamp = datetime(2026, 11, day, tzinfo=timezone.utc)

    return MarketBar(
        symbol=symbol,
        timeframe="1d",
        timestamp=timestamp,
        available_at=timestamp + timedelta(days=1),
        open=close,
        high=close + 2,
        low=close - 2,
        close=close,
        volume=1_000_000,
        source="test",
        adjusted=False,
    )


def test_normal_data_is_valid():
    bars = [make_bar(10, 100), make_bar(11, 102)]
    report = check_market_bars(bars)

    assert report.status == QualityStatus.VALID


def test_large_price_move_is_warning():
    bars = [make_bar(10, 100), make_bar(11, 85)]
    report = check_market_bars(bars)

    assert report.status == QualityStatus.WARNING
    assert any("Large price move" in w for w in report.warnings)


def test_duplicate_timestamp_is_suspect():
    bars = [make_bar(10), make_bar(10)]
    report = check_market_bars(bars)

    assert report.status == QualityStatus.SUSPECT
    assert any("Duplicate timestamp" in w for w in report.warnings)


def test_mixed_symbols_are_invalid():
    bars = [make_bar(10), make_bar(11, symbol="AMD")]
    report = check_market_bars(bars)

    assert report.status == QualityStatus.INVALID
