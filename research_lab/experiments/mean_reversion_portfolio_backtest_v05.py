from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research_lab.experiments.mean_reversion_v01 import (
    load_symbol,
    calculate_indicators,
)

# ============================================================
# MEAN REVERSION v0.4
# EXHAUSTION + STABILISATION
#
# Research only.
# No Alpaca.
# No orders.
# No live trading.
# ============================================================

SYMBOLS = [
    "NVDA", "AMD", "AAPL", "MSFT", "AMZN",
    "META", "GOOGL", "AVGO", "TSLA", "NFLX",
    "ORCL", "CRM", "MU", "QCOM", "PLTR",
    "SPY", "QQQ", "IWM", "JPM", "LLY",
]

STARTING_EQUITY = 10_000.00
RISK_PER_TRADE = 0.005
MAX_OPEN_POSITIONS = 5
MAX_POSITION_FRACTION = 0.20
ROUND_TRIP_COST = 0.001

ATR_LENGTH = 14
EMA_LENGTH = 20

# ------------------------------------------------------------
# New hypothesis
# ------------------------------------------------------------
# 1. Price must be genuinely stretched below VWAP.
# 2. The signal must show stabilisation rather than another
#    aggressive lower-low breakdown.
# 3. The signal candle should recover, but not be an extreme
#    top-of-range spike.
# 4. Entry is next-bar open.
#
# These are deliberately broad research thresholds rather than
# optimised values.
# ------------------------------------------------------------

MIN_EXTENSION_ATR = 2.0

# Signal low must not be materially below the original
# extension low.  We want evidence that selling pressure
# has stopped accelerating.
MAX_NEW_LOW_ATR = 0.25

# Avoid the very top-of-range candles that behaved poorly
# in the pre-entry diagnostic.
MAX_CLOSE_LOCATION = 0.75

# Require at least some bullish recovery.
MIN_SIGNAL_RECLAIM_ATR = 0.0

STOP_ATR = 1.5
MAX_HOLD_BARS = 8

# No new entries late in the session.
LAST_ENTRY_TIME = "11:30"

DATA_DIR = ROOT / "intraday_snapshot"

# Default period; runner temporarily changes these.
OOS_START = pd.Timestamp("2026-10-01", tz="America/New_York")
OOS_END = pd.Timestamp("2026-10-23", tz="America/New_York")


def load_local_symbol(symbol):
    path = DATA_DIR / f"{symbol}.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)

    if "Timestamp" not in df.columns:
        return None

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["Timestamp"]).copy()

    df = df.rename(
        columns={
            "Timestamp": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df["timestamp"] = (
        df["timestamp"]
        .dt.tz_convert("America/New_York")
    )

    df = df.set_index("timestamp").sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["open", "high", "low", "close", "volume"]
    )

    # Regular US session only.
    df = df.between_time("09:30", "16:00").copy()

    return df


def calculate_v04_indicators(df):
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

    df["ema"] = (
        df["close"]
        .ewm(span=EMA_LENGTH, adjust=False)
        .mean()
    )

    # Intraday VWAP.
    typical_price = (
        df["high"] + df["low"] + df["close"]
    ) / 3.0

    session = df.index.date

    cumulative_pv = (
        typical_price * df["volume"]
    ).groupby(session).cumsum()

    cumulative_volume = (
        df["volume"]
        .groupby(session)
        .cumsum()
    )

    df["vwap"] = (
        cumulative_pv / cumulative_volume.replace(0, np.nan)
    )

    df["vwap_distance_atr"] = (
        (df["vwap"] - df["close"])
        / df["atr"].replace(0, np.nan)
    )

    # Signal candle structure.
    candle_range = (
        df["high"] - df["low"]
    ).replace(0, np.nan)

    df["close_location"] = (
        (df["close"] - df["low"])
        / candle_range
    )

    df["signal_reclaim_atr"] = (
        (df["close"] - previous_close)
        / df["atr"].replace(0, np.nan)
    )

    return df


def detect_v04_signal(df, index):
    if index < ATR_LENGTH + 5:
        return None

    row = df.iloc[index]

    required = [
        row["atr"],
        row["vwap"],
        row["close_location"],
        row["signal_reclaim_atr"],
        row["vwap_distance_atr"],
    ]

    if not all(np.isfinite(x) for x in required):
        return None

    # Must be a bullish reversal candle.
    if row["close"] <= row["open"]:
        return None

    if row["close"] <= df.iloc[index - 1]["close"]:
        return None

    # --------------------------------------------------------
    # Find strongest VWAP extension during preceding 3 bars.
    # --------------------------------------------------------
    extension_indices = range(
        max(0, index - 3),
        index,
    )

    extension_index = None
    extension_atr = -np.inf

    for j in extension_indices:
        r = df.iloc[j]

        if not np.isfinite(r["atr"]) or r["atr"] <= 0:
            continue

        value = (
            (r["vwap"] - r["close"])
            / r["atr"]
        )

        if value > extension_atr:
            extension_atr = value
            extension_index = j

    if extension_index is None:
        return None

    if extension_atr < MIN_EXTENSION_ATR:
        return None

    extension_row = df.iloc[extension_index]

    # --------------------------------------------------------
    # Stabilisation:
    #
    # The signal low may be slightly below the extension low,
    # but not by a large ATR amount.
    # --------------------------------------------------------
    signal_low_vs_extension_atr = (
        (row["low"] - extension_row["low"])
        / row["atr"]
    )

    if signal_low_vs_extension_atr < -MAX_NEW_LOW_ATR:
        return None

    # --------------------------------------------------------
    # Avoid the very strongest top-of-range spikes.
    # --------------------------------------------------------
    if row["close_location"] > MAX_CLOSE_LOCATION:
        return None

    # The signal must actually reclaim relative to the
    # preceding close.
    if row["signal_reclaim_atr"] < MIN_SIGNAL_RECLAIM_ATR:
        return None

    entry_index = index + 1

    if entry_index >= len(df):
        return None

    entry_time = df.index[entry_index]

    # No new positions after the cutoff.
    if entry_time.strftime("%H:%M") > LAST_ENTRY_TIME:
        return None

    return {
        "SignalTime": df.index[index],
        "EntryTime": entry_time,
        "ExtensionTime": df.index[extension_index],
        "SignalClose": float(row["close"]),
        "VWAP": float(row["vwap"]),
        "ATR": float(row["atr"]),
        "ExtensionATR": float(extension_atr),
        "CloseLocation": float(row["close_location"]),
        "SignalReclaimATR": float(row["signal_reclaim_atr"]),
        "SignalLowVsExtensionATR": float(
            signal_low_vs_extension_atr
        ),
    }


def build_signals(data_by_symbol, start, end):
    signals = []

    for symbol, df in data_by_symbol.items():
        if df is None or df.empty:
            continue

        df = calculate_v04_indicators(df)

        # Only generate signals inside requested period.
        mask = (
            (df.index >= start)
            & (df.index < end)
        )

        indices = np.where(mask)[0]

        for index in indices:
            signal = detect_v04_signal(df, index)

            if signal is None:
                continue

            signal["Symbol"] = symbol
            signals.append(signal)

    return sorted(
        signals,
        key=lambda x: x["EntryTime"],
    )


def run_backtest(start, end):
    data_by_symbol = {}

    for symbol in SYMBOLS:
        data_by_symbol[symbol] = load_local_symbol(symbol)

    available = sum(
        df is not None and not df.empty
        for df in data_by_symbol.values()
    )

    print(f"Symbols with data: {available} / {len(SYMBOLS)}")

    signals = build_signals(
        data_by_symbol,
        start,
        end,
    )

    print(f"Candidate signals: {len(signals)}")

    cash = STARTING_EQUITY
    completed = []
    open_positions = []

    for signal in signals:
        timestamp = signal["EntryTime"]

        # --------------------------------------------------------
        # Process exits before new entries.
        # --------------------------------------------------------
        remaining = []

        for position in open_positions:
            symbol = position["Symbol"]
            df = data_by_symbol[symbol]

            if timestamp not in df.index:
                remaining.append(position)
                continue

            entry_idx = position["entry_idx"]
            current_idx = df.index.get_loc(timestamp)

            if current_idx <= entry_idx:
                remaining.append(position)
                continue

            row = df.iloc[current_idx]

            exit_price = None
            exit_reason = None

            stop = position["stop"]
            target = position["target"]

            # Conservative stop-first handling.
            if row["low"] <= stop:
                if row["open"] < stop:
                    exit_price = row["open"]
                    exit_reason = "Stop (gap)"
                else:
                    exit_price = stop
                    exit_reason = "Stop"

            elif row["high"] >= target:
                if row["open"] > target:
                    exit_price = row["open"]
                    exit_reason = "Target (gap)"
                else:
                    exit_price = target
                    exit_reason = "Target"

            elif (
                current_idx - entry_idx
                >= MAX_HOLD_BARS
            ):
                exit_price = row["close"]
                exit_reason = "Time"

            elif timestamp.strftime("%H:%M") >= "15:45":
                exit_price = row["close"]
                exit_reason = "End of day"

            if exit_price is None:
                remaining.append(position)
                continue

            gross = (
                (exit_price - position["entry"])
                * position["shares"]
            )

            notional = (
                position["entry"] * position["shares"]
                + exit_price * position["shares"]
            )

            cost = notional * ROUND_TRIP_COST

            net = gross - cost

            cash += (
                position["entry"] * position["shares"]
                + net
            )

            completed.append(
                {
                    "Symbol": symbol,
                    "SignalTime": position["signal_time"],
                    "EntryTime": position["entry_time"],
                    "ExitTime": timestamp,
                    "Entry": position["entry"],
                    "Exit": exit_price,
                    "Shares": position["shares"],
                    "GrossPnL": gross,
                    "Costs": cost,
                    "NetPnL": net,
                    "ExitReason": exit_reason,
                    "ExtensionATR": position["extension_atr"],
                    "CloseLocation": position["close_location"],
                    "SignalReclaimATR": position["signal_reclaim_atr"],
                    "SignalLowVsExtensionATR": position[
                        "signal_low_vs_extension_atr"
                    ],
                }
            )

        open_positions = remaining

        # --------------------------------------------------------
        # Do not exceed maximum open positions.
        # --------------------------------------------------------
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue

        symbol = signal["Symbol"]
        df = data_by_symbol[symbol]

        if timestamp not in df.index:
            continue

        entry_idx = df.index.get_loc(timestamp)
        entry_row = df.iloc[entry_idx]

        entry = float(entry_row["open"])
        atr = float(signal["ATR"])

        if not np.isfinite(entry) or not np.isfinite(atr):
            continue

        if entry <= 0 or atr <= 0:
            continue

        # Target is VWAP at signal time.
        target = float(signal["VWAP"])

        if target <= entry:
            continue

        stop = entry - STOP_ATR * atr

        stop_distance = entry - stop

        if stop_distance <= 0:
            continue

        equity = cash + sum(
            p["entry"] * p["shares"]
            for p in open_positions
        )

        risk_budget = equity * RISK_PER_TRADE

        shares_by_risk = int(
            risk_budget / stop_distance
        )

        shares_by_value = int(
            (equity * MAX_POSITION_FRACTION)
            / entry
        )

        shares = min(
            shares_by_risk,
            shares_by_value,
        )

        if shares < 1:
            continue

        cash -= entry * shares

        open_positions.append(
            {
                "Symbol": symbol,
                "signal_time": signal["SignalTime"],
                "entry_time": timestamp,
                "entry": entry,
                "shares": shares,
                "stop": stop,
                "target": target,
                "entry_idx": entry_idx,
                "extension_atr": signal["ExtensionATR"],
                "close_location": signal["CloseLocation"],
                "signal_reclaim_atr": signal[
                    "SignalReclaimATR"
                ],
                "signal_low_vs_extension_atr": signal[
                    "SignalLowVsExtensionATR"
                ],
            }
        )

    # ------------------------------------------------------------
    # Close anything remaining at final available bar.
    # ------------------------------------------------------------
    for position in open_positions:
        symbol = position["Symbol"]
        df = data_by_symbol[symbol]

        future = df[
            df.index >= position["entry_time"]
        ]

        if future.empty:
            continue

        exit_time = future.index[-1]
        exit_row = future.iloc[-1]
        exit_price = float(exit_row["close"])

        gross = (
            (exit_price - position["entry"])
            * position["shares"]
        )

        notional = (
            position["entry"] * position["shares"]
            + exit_price * position["shares"]
        )

        cost = notional * ROUND_TRIP_COST
        net = gross - cost

        cash += (
            position["entry"] * position["shares"]
            + net
        )

        completed.append(
            {
                "Symbol": symbol,
                "SignalTime": position["signal_time"],
                "EntryTime": position["entry_time"],
                "ExitTime": exit_time,
                "Entry": position["entry"],
                "Exit": exit_price,
                "Shares": position["shares"],
                "GrossPnL": gross,
                "Costs": cost,
                "NetPnL": net,
                "ExitReason": "End of test",
                "ExtensionATR": position["extension_atr"],
                "CloseLocation": position["close_location"],
                "SignalReclaimATR": position["signal_reclaim_atr"],
                "SignalLowVsExtensionATR": position[
                    "signal_low_vs_extension_atr"
                ],
            }
        )

    trades = pd.DataFrame(completed)

    if trades.empty:
        print("No completed trades.")
        return

    final_equity = (
        STARTING_EQUITY
        + trades["NetPnL"].sum()
    )

    wins = trades[trades["NetPnL"] > 0]
    losses = trades[trades["NetPnL"] <= 0]

    gross_profit = wins["NetPnL"].sum()
    gross_loss = abs(losses["NetPnL"].sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else np.inf
    )

    win_rate = (
        len(wins) / len(trades) * 100
    )

    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"Completed trades: {len(trades)}")
    print(f"Starting equity: ${STARTING_EQUITY:,.2f}")
    print(f"Final equity: ${final_equity:,.2f}")
    print(
        f"Net P&L: ${final_equity - STARTING_EQUITY:,.2f}"
    )
    print(
        f"Net return: "
        f"{(final_equity / STARTING_EQUITY - 1) * 100:.2f}%"
    )
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Profit factor: {profit_factor:.2f}")
    print(
        f"Estimated trading costs: "
        f"${trades['Costs'].sum():,.2f}"
    )

    print()
    print("Exit reasons:")
    print(trades["ExitReason"].value_counts().to_string())

    print()
    print("By symbol:")
    by_symbol = (
        trades.groupby("Symbol")
        .agg(
            Trades=("NetPnL", "count"),
            NetPnL=("NetPnL", "sum"),
            Wins=("NetPnL", lambda x: (x > 0).sum()),
        )
        .sort_values("NetPnL", ascending=False)
    )

    by_symbol["WinRate"] = (
        by_symbol["Wins"]
        / by_symbol["Trades"]
        * 100
    )

    print(
        by_symbol.to_string(
            float_format=lambda x: f"{x:,.2f}"
        )
    )

    out_name = (
        "mean_reversion_v04_"
        f"{start.strftime('%Y%m%d')}_"
        f"{end.strftime('%Y%m%d')}.csv"
    )

    output = ROOT / "research_lab" / "results" / out_name
    trades.to_csv(output, index=False)

    print()
    print(f"Saved trades: {output}")


def main():
    print("=" * 100)
    print("MEAN REVERSION v0.5 — EXHAUSTION + STABILISATION + MORNING")
    print("=" * 100)
    print("Historical simulation only.")
    print("No Alpaca connection. No orders.")
    print()
    print(
        "Hypothesis: the exhaustion/reversal setup may work better "
        "during the US opening price-discovery period."
    )
    print()
    print(
        f"Entry window: 09:30–11:30 America/New_York"
    )
    print(
        f"Extension >= {MIN_EXTENSION_ATR:.1f} ATR | "
        f"New low >= {-MAX_NEW_LOW_ATR:.2f} ATR | "
        f"Close location <= {MAX_CLOSE_LOCATION:.2f}"
    )

    periods = [
        (
            "TRAIN",
            pd.Timestamp("2026-07-30", tz="America/New_York"),
            pd.Timestamp("2026-09-01", tz="America/New_York"),
        ),
        (
            "VALIDATION",
            pd.Timestamp("2026-09-01", tz="America/New_York"),
            pd.Timestamp("2026-10-01", tz="America/New_York"),
        ),
        (
            "OOS",
            pd.Timestamp("2026-10-01", tz="America/New_York"),
            pd.Timestamp("2026-10-23", tz="America/New_York"),
        ),
    ]

    for name, start, end in periods:
        print()
        print("#" * 100)
        print(f"{name}: {start.date()} -> {end.date()}")
        print("#" * 100)
        run_backtest(start, end)


if __name__ == "__main__":
    main()
