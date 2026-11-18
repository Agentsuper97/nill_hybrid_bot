import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# READ-ONLY RESEARCH TOOL: this script contains no order-submission calls.
SYMBOLS = [
    "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO",
    "TSLA", "NFLX", "ORCL", "CRM", "MU", "QCOM", "PLTR", "SPY",
    "QQQ", "IWM", "JPM", "LLY",
]

LOOKBACK = 10
EMA_LENGTH = 20
RVOL_LENGTH = 20
RVOL_MIN = 1.5
ATR_LENGTH = 14
ADX_LENGTH = 14
ADX_MIN = 40.0
STOP_ATR_MULTIPLE = 1.5
TARGET_R_MULTIPLE = 2.0
BAR_MINUTES = 15

NY = ZoneInfo("America/New_York")


def calculate_indicators(df):
    df = df.sort_index().copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ema"] = df["close"].ewm(span=EMA_LENGTH, adjust=False).mean()
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = true_range.rolling(ATR_LENGTH).mean()

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    atr_for_adx = true_range.rolling(ADX_LENGTH).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(ADX_LENGTH).mean() / atr_for_adx
    minus_di = 100 * minus_dm.rolling(ADX_LENGTH).mean() / atr_for_adx
    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denominator
    df["adx"] = dx.rolling(ADX_LENGTH).mean()

    df["prior_high"] = df["high"].shift(1).rolling(LOOKBACK).max()
    df["average_volume"] = df["volume"].shift(1).rolling(RVOL_LENGTH).mean()
    df["rvol"] = df["volume"] / df["average_volume"].replace(0, np.nan)
    df["crossed_prior_high"] = (
        (df["close"] > df["prior_high"])
        & (df["close"].shift(1) <= df["prior_high"].shift(1))
    )
    return df


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing Alpaca credentials in .env; values were not displayed.")

    client = StockHistoricalDataClient(key, secret)

    # Fetch enough 15-minute history to warm up the indicators.
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=10)
    request = StockBarsRequest(
        symbol_or_symbols=SYMBOLS,
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=start,
        end=end,
        limit=10000,
        feed=DataFeed.IEX,
    )

    print("ALPACA READ-ONLY 15-MINUTE RESEARCH SCAN")
    print("No orders will be submitted.")
    print("Times shown in America/New_York.")
    print()

    bars = client.get_stock_bars(request).df
    if bars.empty:
        raise SystemExit("Alpaca returned no bars. Check market-data access and subscription.")

    if isinstance(bars.index, pd.MultiIndex):
        symbols_in_data = set(bars.index.get_level_values(0))
    else:
        symbols_in_data = set(SYMBOLS)

    now_ny = datetime.now(NY)
    results = []

    for symbol in SYMBOLS:
        if symbol not in symbols_in_data:
            results.append({"symbol": symbol, "status": "NO DATA"})
            continue

        try:
            if isinstance(bars.index, pd.MultiIndex):
                df = bars.xs(symbol, level=0).copy()
            else:
                df = bars.copy()

            if df.empty:
                results.append({"symbol": symbol, "status": "NO DATA"})
                continue

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert(NY)

            # Ignore a bar whose 15-minute interval has not finished.
            df = df[(df.index + pd.Timedelta(minutes=BAR_MINUTES)) <= pd.Timestamp(now_ny)]
            if len(df) < 60:
                results.append({"symbol": symbol, "status": "INSUFFICIENT BARS"})
                continue

            df = calculate_indicators(df)
            row = df.iloc[-1]

            ready = all(
                pd.notna(row.get(col))
                for col in ["ema", "atr", "adx", "prior_high", "rvol"]
            )
            if not ready:
                results.append({"symbol": symbol, "status": "INDICATORS NOT READY"})
                continue

            breakout = bool(row["crossed_prior_high"])
            above_ema = bool(row["close"] > row["ema"])
            volume_ok = bool(row["rvol"] >= RVOL_MIN)
            trend_ok = bool(row["adx"] >= ADX_MIN)
            signal = breakout and above_ema and volume_ok and trend_ok and row["atr"] > 0

            entry_reference = float(row["close"])
            stop_reference = entry_reference - STOP_ATR_MULTIPLE * float(row["atr"])
            target_reference = entry_reference + TARGET_R_MULTIPLE * (
                entry_reference - stop_reference
            )

            results.append({
                "symbol": symbol,
                "bar_time_ny": row.name.strftime("%Y-%m-%d %H:%M"),
                "close": round(float(row["close"]), 2),
                "ema20": round(float(row["ema"]), 2),
                "rvol": round(float(row["rvol"]), 2),
                "adx14": round(float(row["adx"]), 1),
                "breakout": breakout,
                "above_ema": above_ema,
                "volume_ok": volume_ok,
                "adx_ok": trend_ok,
                "signal": "POTENTIAL LONG" if signal else "-",
                "reference_stop": round(stop_reference, 2),
                "reference_target": round(target_reference, 2),
                "status": "OK",
            })
        except Exception as exc:
            results.append({
                "symbol": symbol,
                "status": f"ERROR: {type(exc).__name__}: {exc}",
            })

    output = pd.DataFrame(results)
    print(output.to_string(index=False))
    print()
    print("Research only: signals are unverified candidates, not trade instructions.")
    print("Reference stop/target are illustrative calculations from the last close,")
    print("not executable orders or a guarantee of the strategy's backtest results.")


if __name__ == "__main__":
    main()
