from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from app.domain.market_data import MarketBar


def fetch_daily_bars(symbol: str, period: str = "1mo") -> list[MarketBar]:
    """Download and validate historical daily bars for research."""
    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError("Symbol cannot be empty")

    data = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No market data returned for {symbol}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.sort_index()
    bars = []

    for timestamp, row in data.iterrows():
        timestamp = pd.Timestamp(timestamp)

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                ZoneInfo("America/New_York")
            )

        timestamp_utc = timestamp.tz_convert("UTC").to_pydatetime()

        # Conservative placeholder, not verified point-in-time availability.
        available_at = timestamp_utc + timedelta(days=1)

        bar = MarketBar(
            symbol=symbol,
            timeframe="1d",
            timestamp=timestamp_utc,
            available_at=available_at,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
            source="yfinance",
            adjusted=False,
            availability_basis="estimated_policy",
        )

        bars.append(bar)

    return bars
