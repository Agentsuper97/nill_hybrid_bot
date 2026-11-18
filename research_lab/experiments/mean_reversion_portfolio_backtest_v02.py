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

STARTING_EQUITY = 10_000.00
RISK_PER_TRADE = 0.005
MAX_OPEN_POSITIONS = 5
MAX_POSITION_FRACTION = 0.20
ROUND_TRIP_COST = 0.001

ATR_LENGTH = 14
EXTENSION_ATR = 1.0
MAX_REVERSAL_BARS = 3
MIN_RVOL = 1.0

STOP_ATR = 1.5
MAX_HOLD_BARS = 8

OOS_START = pd.Timestamp("2026-10-01", tz=NY)
OOS_END = pd.Timestamp("2026-10-23", tz=NY)


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
        cumulative_pv
        / cumulative_volume.replace(0, np.nan)
    )

    df["average_volume"] = (
        df["volume"].shift(1).rolling(20).mean()
    )

    df["rvol"] = (
        df["volume"]
        / df["average_volume"].replace(0, np.nan)
    )

    df["extension_atr"] = (
        (df["vwap"] - df["close"])
        / df["atr"].replace(0, np.nan)
    )

    df["reversal"] = (
        (df["close"] > df["close"].shift(1))
        & (df["close"] > df["open"])
    )

    return df


def build_signals(df):
    signals = []

    # Work entirely with timestamps.
    for i in range(ATR_LENGTH + 20, len(df) - 1):

        row = df.iloc[i]

        if not np.isfinite(row["atr"]):
            continue

        if not np.isfinite(row["vwap"]):
            continue

        if not np.isfinite(row["rvol"]):
            continue

        if not bool(row["reversal"]):
            continue

        if row["rvol"] < MIN_RVOL:
            continue

        extension_timestamp = None

        for lookback in range(1, MAX_REVERSAL_BARS + 1):

            candidate = df.iloc[i - lookback]

            if (
                np.isfinite(candidate["extension_atr"])
                and candidate["extension_atr"] >= EXTENSION_ATR
            ):
                extension_timestamp = candidate.name
                break

        if extension_timestamp is None:
            continue

        entry_timestamp = df.index[i + 1]

        if entry_timestamp.strftime("%H:%M") > "15:30":
            continue

        signals.append(
            {
                "Symbol": None,
                "SignalTime": df.index[i],
                "EntryTime": entry_timestamp,
                "ExtensionTime": extension_timestamp,
                "SignalVWAP": float(row["vwap"]),
                "SignalATR": float(row["atr"]),
                "ExtensionATR": float(
                    df.loc[extension_timestamp, "extension_atr"]
                ),
                "EntryRVOL": float(row["rvol"]),
            }
        )

    return signals


def simulate_trade(df, signal, entry_price, shares):

    entry_time = signal["EntryTime"]

    # Find the actual entry row by timestamp.
    if entry_time not in df.index:
        return None

    entry_pos = df.index.get_loc(entry_time)

    if not isinstance(entry_pos, (int, np.integer)):
        return None

    stop = (
        entry_price
        - STOP_ATR * signal["SignalATR"]
    )

    target = signal["SignalVWAP"]

    # A VWAP target must actually be above the entry.
    if target <= entry_price:
        return None

    last_position = min(
        entry_pos + MAX_HOLD_BARS - 1,
        len(df) - 1,
    )

    exit_pos = None
    exit_price = None
    exit_reason = None

    for j in range(entry_pos, last_position + 1):

        bar = df.iloc[j]
        timestamp = df.index[j]

        # No overnight positions.
        if timestamp.strftime("%H:%M") >= "15:45":
            exit_pos = j
            exit_price = float(bar["close"])
            exit_reason = "End of day"
            break

        # Conservative stop-first assumption if both are touched.
        if (
            bar["low"] <= stop
            and bar["high"] >= target
        ):
            exit_pos = j
            exit_price = stop
            exit_reason = "Stop"
            break

        if bar["low"] <= stop:
            exit_pos = j
            exit_price = stop
            exit_reason = "Stop"
            break

        if bar["high"] >= target:
            exit_pos = j
            exit_price = target
            exit_reason = "Target"
            break

        if j == last_position:
            exit_pos = j
            exit_price = float(bar["close"])
            exit_reason = "Time"
            break

    if exit_pos is None:
        return None

    exit_time = df.index[exit_pos]

    # Hard integrity check.
    if exit_time < entry_time:
        raise RuntimeError(
            f"INVALID TRADE: exit {exit_time} "
            f"before entry {entry_time}"
        )

    entry_value = entry_price * shares
    exit_value = exit_price * shares

    gross_pnl = exit_value - entry_value

    estimated_costs = (
        entry_value + exit_value
    ) * ROUND_TRIP_COST / 2.0

    net_pnl = gross_pnl - estimated_costs

    return {
        "Symbol": signal["Symbol"],
        "SignalTime": signal["SignalTime"],
        "ExtensionTime": signal["ExtensionTime"],
        "EntryTime": entry_time,
        "ExitTime": exit_time,
        "EntryPrice": entry_price,
        "ExitPrice": exit_price,
        "Shares": shares,
        "Stop": stop,
        "Target": target,
        "EntryRVOL": signal["EntryRVOL"],
        "EntryExtensionATR": signal["ExtensionATR"],
        "EntryATR": signal["SignalATR"],
        "BarsHeld": exit_pos - entry_pos + 1,
        "ExitReason": exit_reason,
        "GrossPnL": gross_pnl,
        "EstimatedCosts": estimated_costs,
        "NetPnL": net_pnl,
    }


def main():

    data = {}

    for symbol in SYMBOLS:

        df = load_symbol(symbol)

        if df is None or df.empty:
            continue

        data[symbol] = calculate_indicators(df)

    print("=" * 100)
    print("MEAN REVERSION PORTFOLIO BACKTEST v0.2 — INDEX-SAFE")
    print("=" * 100)
    print("Historical simulation only.")
    print("No Alpaca connection. No orders.")
    print(f"OOS: {OOS_START} -> {OOS_END}")
    print(
        f"Extension: {EXTENSION_ATR:.1f} ATR | "
        f"Stop: {STOP_ATR:.1f} ATR | "
        f"Max hold: {MAX_HOLD_BARS} bars"
    )

    signals = []

    for symbol, df in data.items():

        period_df = df.loc[
            (df.index >= OOS_START)
            & (df.index < OOS_END)
        ]

        for signal in build_signals(period_df):
            signal["Symbol"] = symbol
            signals.append(signal)

    signals.sort(
        key=lambda x: x["EntryTime"]
    )

    print(
        f"Symbols with data: "
        f"{len(data)} / {len(SYMBOLS)}"
    )
    print(
        f"Candidate signals: {len(signals)}"
    )

    cash = STARTING_EQUITY
    open_positions = []
    completed = []

    for signal in signals:

        timestamp = signal["EntryTime"]

        # Remove positions that have already exited.
        open_positions = [
            p for p in open_positions
            if p["ExitTime"] > timestamp
        ]

        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue

        symbol = signal["Symbol"]

        if any(
            p["Symbol"] == symbol
            and p["ExitTime"] > timestamp
            for p in open_positions
        ):
            continue

        df = data[symbol]

        if timestamp not in df.index:
            continue

        entry_pos = df.index.get_loc(timestamp)

        if not isinstance(entry_pos, (int, np.integer)):
            continue

        entry_price = float(
            df.iloc[entry_pos]["open"]
        )

        if not np.isfinite(entry_price):
            continue

        stop_distance = (
            STOP_ATR * signal["SignalATR"]
        )

        if stop_distance <= 0:
            continue

        equity = cash + sum(
            p["EntryPrice"] * p["Shares"]
            for p in open_positions
        )

        risk_budget = (
            equity * RISK_PER_TRADE
        )

        shares_by_risk = int(
            risk_budget / stop_distance
        )

        shares_by_cap = int(
            equity
            * MAX_POSITION_FRACTION
            / entry_price
        )

        shares = min(
            shares_by_risk,
            shares_by_cap,
        )

        if shares < 1:
            continue

        trade = simulate_trade(
            df,
            signal,
            entry_price,
            shares,
        )

        if trade is None:
            continue

        open_positions.append(trade)
        completed.append(trade)

        cash += trade["NetPnL"]

    trades = pd.DataFrame(completed)

    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)

    if trades.empty:
        print("No completed trades.")
        return

    # Final integrity verification.
    invalid = (
        trades["ExitTime"]
        < trades["EntryTime"]
    ).sum()

    if invalid:
        raise RuntimeError(
            f"Integrity failure: {invalid} "
            "trades have exits before entries."
        )

    gross_profit = trades.loc[
        trades["NetPnL"] > 0,
        "NetPnL",
    ].sum()

    gross_loss = abs(
        trades.loc[
            trades["NetPnL"] <= 0,
            "NetPnL",
        ].sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    wins = (
        trades["NetPnL"] > 0
    ).sum()

    final_equity = (
        STARTING_EQUITY
        + trades["NetPnL"].sum()
    )

    print(
        f"Completed trades: {len(trades)}"
    )
    print(
        f"Starting equity: "
        f"${STARTING_EQUITY:,.2f}"
    )
    print(
        f"Final equity: "
        f"${final_equity:,.2f}"
    )
    print(
        f"Net P&L: "
        f"${trades['NetPnL'].sum():,.2f}"
    )
    print(
        f"Net return: "
        f"{trades['NetPnL'].sum() / STARTING_EQUITY * 100:.2f}%"
    )
    print(
        f"Win rate: "
        f"{wins / len(trades) * 100:.1f}%"
    )
    print(
        f"Profit factor: "
        f"{profit_factor:.2f}"
    )
    print(
        f"Estimated trading costs: "
        f"${trades['EstimatedCosts'].sum():,.2f}"
    )

    print("\nExit reasons:")
    print(
        trades["ExitReason"]
        .value_counts()
        .to_string()
    )

    print("\nResults by symbol:")

    summary = (
        trades.groupby("Symbol")
        .agg(
            Trades=("NetPnL", "size"),
            Wins=(
                "NetPnL",
                lambda x: (x > 0).sum(),
            ),
            NetPnL=("NetPnL", "sum"),
            EstimatedCosts=(
                "EstimatedCosts",
                "sum",
            ),
        )
        .sort_values(
            "NetPnL",
            ascending=False,
        )
    )

    print(
        summary.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    output_dir = (
        ROOT
        / "research_lab"
        / "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "mean_reversion_trades_v02.csv"
    )

    trades.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved trades to: {output_path}"
    )

    print()
    print("=" * 100)
    print("INTEGRITY CHECK PASSED")
    print(
        "Every recorded exit occurs on or after its entry."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
