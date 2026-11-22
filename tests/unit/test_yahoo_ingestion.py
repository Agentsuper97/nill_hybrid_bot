from datetime import timedelta

import pandas as pd

from app.ingestion.yahoo import fetch_daily_bars


def test_daily_bar_uses_documented_availability_placeholder(monkeypatch):
    index = pd.DatetimeIndex(["2026-11-02"])

    data = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [103.0],
            "Low": [99.0],
            "Close": [102.0],
            "Volume": [1_000_000],
        },
        index=index,
    )

    def fake_download(*args, **kwargs):
        return data.copy()

    monkeypatch.setattr(
        "app.ingestion.yahoo.yf.download",
        fake_download,
    )

    bars = fetch_daily_bars("NVDA")

    assert len(bars) == 1

    bar = bars[0]

    assert bar.symbol == "NVDA"
    assert bar.timeframe.value == "1d"
    assert bar.source == "yfinance"

    # Current policy: NY-local candle date plus 24 hours.
    # This is an estimate, not verified provider availability.
    assert bar.available_at == bar.timestamp + timedelta(days=1)

    assert bar.timestamp.tzinfo is not None
    assert bar.available_at.tzinfo is not None


def test_daily_timestamps_respect_new_york_daylight_saving(monkeypatch):
    from datetime import timezone

    index = pd.DatetimeIndex([
        "2026-03-06",
        "2026-03-09",
    ])

    data = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [103.0, 104.0],
            "Low": [99.0, 100.0],
            "Close": [102.0, 103.0],
            "Volume": [1_000_000, 1_100_000],
        },
        index=index,
    )

    monkeypatch.setattr(
        "app.ingestion.yahoo.yf.download",
        lambda *args, **kwargs: data.copy(),
    )

    bars = fetch_daily_bars("NVDA")

    assert len(bars) == 2

    # March 6 is before the US daylight saving transition.
    assert bars[0].timestamp.hour == 5

    # March 9 is after the transition.
    assert bars[1].timestamp.hour == 4

    assert all(
        bar.timestamp.tzinfo == timezone.utc
        for bar in bars
    )

    # Document the existing 24-hour UTC availability policy.
    assert all(
        bar.available_at == bar.timestamp + timedelta(days=1)
        for bar in bars
    )
