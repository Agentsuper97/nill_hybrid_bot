"""
MEAN REVERSION v0.1 — RESEARCH ONLY

Hypothesis:
Intraday price extensions below VWAP followed by a reversal confirmation
may provide profitable mean-reversion opportunities.

This module is historical research only.
No Alpaca connection. No orders.
No look-ahead: signals are generated from completed bars and are intended
for entry on the following bar.
"""

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SYMBOLS = [
    "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO",
    "TSLA", "NFLX", "ORCL", "CRM", "MU", "QCOM", "PLTR", "SPY",
    "QQQ", "IWM", "JPM", "LLY",
]

NY = "America/New_York"

ATR_LENGTH = 14
EXTENSION_ATR = 1.0
MAX_REVERSAL_BARS = 3
MIN_RVOL = 1.0


def load_symbol(symbol):
    path = ROOT / "intraday_snapshot" / f"{symbol}.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.set_index("timestamp").sort_index()

    if df.index.tz is None:
        df.index = df.index.tz_localize(NY)
    else:
        df.index = df.index.tz_convert(NY)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["open", "high", "low", "close", "volume"]
    )

    # Regular US session only.
    df = df.between_time("09:30", "16:00")

    return df


def calculate_indicators(df):
    df = df.copy()

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

    # Session VWAP.
    typical_price = (
        df["high"] + df["low"] + df["close"]
    ) / 3.0

    session_date = df.index.date

    cumulative_pv = (
        typical_price * df["volume"]
    ).groupby(session_date).cumsum()

    cumulative_volume = (
        df["volume"].groupby(session_date).cumsum()
    )

    df["vwap"] = (
        cumulative_pv / cumulative_volume.replace(0, np.nan)
    )

    df["average_volume"] = (
        df["volume"]
        .shift(1)
        .rolling(20)
        .mean()
    )

    df["rvol"] = (
        df["volume"]
        / df["average_volume"].replace(0, np.nan)
    )

    df["extension_atr"] = (
        (df["vwap"] - df["close"])
        / df["atr"].replace(0, np.nan)
    )

    # A reversal confirmation:
    # previous bar was meaningfully below VWAP,
    # current bar closes above previous close.
    df["reversal"] = (
        (df["close"] > df["close"].shift(1))
        & (df["close"] > df["open"])
    )

    return df


def detect_signal(df, index):
    """
    Returns a signal dictionary or None.

    The signal uses only information available at `index`.
    Entry should occur on index + 1.
    """

    if index < ATR_LENGTH + 20:
        return None

    row = df.iloc[index]

    if not np.isfinite(row["atr"]):
        return None

    if not np.isfinite(row["vwap"]):
        return None

    if not np.isfinite(row["rvol"]):
        return None

    # Current bar must be the reversal confirmation.
    if not bool(row["reversal"]):
        return None

    if row["rvol"] < MIN_RVOL:
        return None

    # Look backward up to MAX_REVERSAL_BARS bars for an extension.
    extension_index = None

    for lookback in range(1, MAX_REVERSAL_BARS + 1):
        candidate_index = index - lookback

        if candidate_index < 0:
            continue

        candidate = df.iloc[candidate_index]

        if (
            np.isfinite(candidate["extension_atr"])
            and candidate["extension_atr"] >= EXTENSION_ATR
        ):
            extension_index = candidate_index
            break

    if extension_index is None:
        return None

    extension = df.iloc[extension_index]

    return {
        "SignalTime": df.index[index],
        "ExtensionTime": df.index[extension_index],
        "SignalClose": float(row["close"]),
        "VWAP": float(row["vwap"]),
        "ATR": float(row["atr"]),
        "ExtensionATR": float(extension["extension_atr"]),
        "RVOL": float(row["rvol"]),
        "EntryTime": (
            df.index[index + 1]
            if index + 1 < len(df)
            else pd.NaT
        ),
    }


def scan_period(start, end):
    results = []
    symbols_loaded = 0

    for symbol in SYMBOLS:
        df = load_symbol(symbol)

        if df is None or df.empty:
            continue

        symbols_loaded += 1
        df = calculate_indicators(df)

        mask = (
            (df.index >= pd.Timestamp(start, tz=NY))
            & (df.index < pd.Timestamp(end, tz=NY))
        )

        period_df = df.loc[mask]

        for i in range(len(period_df)):
            # Map period position back to original dataframe index.
            timestamp = period_df.index[i]
            original_index = df.index.get_loc(timestamp)

            signal = detect_signal(df, original_index)

            if signal is not None:
                signal["Symbol"] = symbol
                results.append(signal)

    return symbols_loaded, pd.DataFrame(results)


def main():
    periods = [
        ("TRAIN", "2026-07-30", "2026-09-01"),
        ("VALIDATION", "2026-09-01", "2026-10-01"),
        ("OOS", "2026-10-01", "2026-10-23"),
    ]

    print("=" * 100)
    print("MEAN REVERSION v0.1 — INITIAL SIGNAL RESEARCH")
    print("=" * 100)
    print("Research only. No Alpaca. No orders.")
    print(
        f"Extension threshold: {EXTENSION_ATR:.1f} ATR below VWAP"
    )
    print(
        f"Maximum reversal window: {MAX_REVERSAL_BARS} bars"
    )
    print()

    all_results = []

    for name, start, end in periods:
        loaded, signals = scan_period(start, end)

        print("-" * 100)
        print(
            f"{name}: {start} -> {end} | "
            f"Symbols: {loaded}/{len(SYMBOLS)} | "
            f"Signals: {len(signals)}"
        )

        if not signals.empty:
            by_symbol = (
                signals.groupby("Symbol")
                .size()
                .sort_values(ascending=False)
            )

            print("\nSignals by symbol:")
            print(by_symbol.to_string())

            print("\nSignal characteristics:")
            print(
                f"  Avg extension: "
                f"{signals['ExtensionATR'].mean():.2f} ATR"
            )
            print(
                f"  Avg RVOL: "
                f"{signals['RVOL'].mean():.2f}x"
            )

            signals["Period"] = name
            all_results.append(signals)

        print()

    if all_results:
        output = pd.concat(all_results, ignore_index=True)

        output_dir = ROOT / "research_lab" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_dir / "mean_reversion_signals_v01.csv"
        )

        output.to_csv(output_path, index=False)

        print("=" * 100)
        print(f"Total signals: {len(output)}")
        print(f"Saved signals to: {output_path}")
        print("=" * 100)
    else:
        print("=" * 100)
        print("No signals detected.")
        print("=" * 100)


if __name__ == "__main__":
    main()
