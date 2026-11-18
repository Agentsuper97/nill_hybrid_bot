from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research_lab.experiments.mean_reversion_v01 import (
    load_symbol,
    calculate_indicators,
    detect_signal,
)

SYMBOLS = [
    "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO",
    "TSLA", "NFLX", "ORCL", "CRM", "MU", "QCOM", "PLTR", "SPY",
    "QQQ", "IWM", "JPM", "LLY",
]

STARTING_EQUITY = 10_000.00
RISK_PER_TRADE = 0.005
MAX_OPEN_POSITIONS = 5
MAX_POSITION_FRACTION = 0.20
ROUND_TRIP_COST = 0.001

STOP_ATR = 1.5
MAX_HOLD_BARS = 8

OOS_START = pd.Timestamp("2026-10-01", tz="America/New_York")
OOS_END = pd.Timestamp("2026-10-23", tz="America/New_York")


def add_regime_features(df):
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

    df["atr14_regime"] = true_range.rolling(14).mean()

    df["ema20_regime"] = (
        df["close"].ewm(span=20, adjust=False).mean()
    )

    df["ema20_slope"] = df["ema20_regime"].diff(4)

    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    atr_adx = (
        true_range.rolling(14).mean()
        .replace(0, np.nan)
    )

    plus_di = (
        100
        * plus_dm.rolling(14).mean()
        / atr_adx
    )

    minus_di = (
        100
        * minus_dm.rolling(14).mean()
        / atr_adx
    )

    denominator = (
        plus_di + minus_di
    ).replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    df["adx14_regime"] = dx.rolling(14).mean()

    return df


def prepare_symbol(symbol):
    df = load_symbol(symbol)

    if df is None or df.empty:
        return None

    df = calculate_indicators(df)
    df = add_regime_features(df)

    return df


def prepare_market_context():
    context = {}

    for symbol in ["SPY", "QQQ"]:
        df = prepare_symbol(symbol)

        if df is None:
            raise RuntimeError(
                f"Unable to load market context for {symbol}"
            )

        context[symbol] = df

    return context


def get_row_at_or_before(df, timestamp):
    timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "America/New_York"
        )
    else:
        timestamp = timestamp.tz_convert(
            "America/New_York"
        )

    pos = df.index.searchsorted(
        timestamp,
        side="right",
    ) - 1

    if pos < 0 or pos >= len(df):
        return None

    return df.iloc[pos]


def regime_allows_trade(
    stock_row,
    spy_row,
    qqq_row,
):
    if stock_row is None:
        return False

    if spy_row is None or qqq_row is None:
        return False

    stock_bearish = (
        stock_row["close"]
        < stock_row["ema20_regime"]
        and stock_row["ema20_slope"] < 0
        and stock_row["adx14_regime"] >= 40
    )

    broad_bearish = (
        spy_row["close"] < spy_row["ema20_regime"]
        and qqq_row["close"] < qqq_row["ema20_regime"]
    )

    # Structural hypothesis:
    # Do not attempt long mean reversion when the
    # stock itself is in a strong bearish trend OR
    # when both major market proxies are bearish.
    if stock_bearish:
        return False

    if broad_bearish:
        return False

    return True


def main():
    print("=" * 100)
    print("MEAN REVERSION PORTFOLIO BACKTEST v0.3 — REGIME FILTER")
    print("=" * 100)
    print("Historical simulation only.")
    print("No Alpaca connection. No orders.")
    print()
    print(
        "Hypothesis: avoid long mean reversion during "
        "strong stock downtrends or broad-market bearish regimes."
    )
    print()
    print(
        f"OOS: {OOS_START} -> {OOS_END}"
    )
    print(
        f"Extension: 1.0 ATR | Stop: {STOP_ATR} ATR | "
        f"Max hold: {MAX_HOLD_BARS} bars"
    )
    print()

    market = prepare_market_context()

    data = {}

    for symbol in SYMBOLS:
        df = prepare_symbol(symbol)

        if df is not None and not df.empty:
            data[symbol] = df

    print(
        f"Symbols with data: "
        f"{len(data)} / {len(SYMBOLS)}"
    )

    candidates = []

    for symbol, df in data.items():
        signals = []
        for i in range(len(df)):
            signal = detect_signal(df, i)
            if signal is not None:
                signals.append(signal)

        if signals is None:
            continue

        if isinstance(signals, pd.DataFrame):
            records = signals.to_dict("records")
        else:
            records = signals

        for signal in records:
            signal_time = signal.get("SignalTime")

            if signal_time is None:
                continue

            signal_time = pd.Timestamp(signal_time)

            if signal_time.tzinfo is None:
                signal_time = signal_time.tz_localize(
                    "America/New_York"
                )
            else:
                signal_time = signal_time.tz_convert(
                    "America/New_York"
                )

            if not (
                OOS_START <= signal_time < OOS_END
            ):
                continue

            stock_row = get_row_at_or_before(
                df,
                signal_time,
            )

            spy_row = get_row_at_or_before(
                market["SPY"],
                signal_time,
            )

            qqq_row = get_row_at_or_before(
                market["QQQ"],
                signal_time,
            )

            if not regime_allows_trade(
                stock_row,
                spy_row,
                qqq_row,
            ):
                continue

            entry_time = signal.get("EntryTime")

            if entry_time is None:
                pos = df.index.searchsorted(
                    signal_time,
                    side="right",
                )

                if pos >= len(df):
                    continue

                entry_time = df.index[pos]
            else:
                entry_time = pd.Timestamp(entry_time)

                if entry_time.tzinfo is None:
                    entry_time = entry_time.tz_localize(
                        "America/New_York"
                    )
                else:
                    entry_time = entry_time.tz_convert(
                        "America/New_York"
                    )

            if entry_time >= OOS_END:
                continue

            if entry_time.strftime("%H:%M") > "15:30":
                continue

            candidates.append(
                {
                    "Symbol": symbol,
                    "SignalTime": signal_time,
                    "EntryTime": entry_time,
                    "SignalClose": float(
                        signal["SignalClose"]
                    ),
                    "VWAP": float(
                        signal["VWAP"]
                    ),
                    "ATR": float(
                        signal["ATR"]
                    ),
                    "ExtensionATR": float(
                        signal["ExtensionATR"]
                    ),
                    "RVOL": float(
                        signal["RVOL"]
                    ),
                }
            )

    candidates.sort(
        key=lambda x: x["EntryTime"]
    )

    print(
        f"Filtered candidate signals: "
        f"{len(candidates)}"
    )

    cash = STARTING_EQUITY
    positions = []
    completed = []

    for candidate in candidates:
        timestamp = candidate["EntryTime"]

        still_open = []

        for position in positions:
            df = data[position["Symbol"]]

            entry_time = position["EntryTime"]
            entry_pos = df.index.searchsorted(
                entry_time
            )

            if entry_pos >= len(df):
                continue

            current_pos = df.index.searchsorted(
                timestamp,
                side="right",
            ) - 1

            if current_pos < entry_pos:
                still_open.append(position)
                continue

            bars_held = (
                current_pos - entry_pos
            )

            row = df.iloc[current_pos]

            exit_price = None
            exit_reason = None

            if row["low"] <= position["stop"]:
                exit_price = position["stop"]
                exit_reason = "Stop"

            elif row["high"] >= position["target"]:
                exit_price = position["target"]
                exit_reason = "Target"

            elif bars_held >= MAX_HOLD_BARS:
                exit_price = row["close"]
                exit_reason = "Time"

            elif row.name.strftime("%H:%M") >= "15:45":
                exit_price = row["close"]
                exit_reason = "End of day"

            if exit_price is None:
                still_open.append(position)
                continue

            gross_pnl = (
                exit_price - position["entry_price"]
            ) * position["shares"]

            costs = (
                (
                    position["entry_price"]
                    + exit_price
                )
                * position["shares"]
                * ROUND_TRIP_COST
            )

            net_pnl = gross_pnl - costs

            cash += net_pnl

            completed.append(
                {
                    "Symbol": position["Symbol"],
                    "SignalTime": position["SignalTime"],
                    "EntryTime": position["EntryTime"],
                    "ExitTime": row.name,
                    "EntryPrice": position["entry_price"],
                    "ExitPrice": exit_price,
                    "Shares": position["shares"],
                    "Stop": position["stop"],
                    "Target": position["target"],
                    "GrossPnL": gross_pnl,
                    "EstimatedCosts": costs,
                    "NetPnL": net_pnl,
                    "ExitReason": exit_reason,
                    "ExtensionATR": position["ExtensionATR"],
                    "RVOL": position["RVOL"],
                }
            )

        positions = still_open

        if len(positions) >= MAX_OPEN_POSITIONS:
            continue

        symbol = candidate["Symbol"]
        df = data[symbol]

        entry_pos = df.index.searchsorted(
            candidate["EntryTime"]
        )

        if entry_pos >= len(df):
            continue

        entry_row = df.iloc[entry_pos]

        entry_price = float(entry_row["open"])
        atr = float(candidate["ATR"])

        if not np.isfinite(entry_price):
            continue

        if not np.isfinite(atr) or atr <= 0:
            continue

        stop_distance = STOP_ATR * atr
        stop = entry_price - stop_distance
        target = float(candidate["VWAP"])

        if target <= entry_price:
            continue

        equity = cash

        risk_budget = (
            equity * RISK_PER_TRADE
        )

        shares_by_risk = int(
            risk_budget / stop_distance
        )

        shares_by_value = int(
            equity
            * MAX_POSITION_FRACTION
            / entry_price
        )

        shares = min(
            shares_by_risk,
            shares_by_value,
        )

        if shares < 1:
            continue

        positions.append(
            {
                "Symbol": symbol,
                "SignalTime": candidate["SignalTime"],
                "EntryTime": candidate["EntryTime"],
                "entry_price": entry_price,
                "shares": shares,
                "stop": stop,
                "target": target,
                "ExtensionATR": candidate[
                    "ExtensionATR"
                ],
                "RVOL": candidate["RVOL"],
            }
        )

    # Force-close anything still open at the final available
    # regular-session bar so the backtest cannot leave capital
    # artificially floating outside the test period.
    for position in positions:
        df = data[position["Symbol"]]

        eligible = df.loc[
            (df.index >= position["EntryTime"])
            & (df.index < OOS_END)
        ]

        if eligible.empty:
            continue

        row = eligible.iloc[-1]

        exit_price = float(row["close"])

        gross_pnl = (
            exit_price - position["entry_price"]
        ) * position["shares"]

        costs = (
            (
                position["entry_price"]
                + exit_price
            )
            * position["shares"]
            * ROUND_TRIP_COST
        )

        net_pnl = gross_pnl - costs

        cash += net_pnl

        completed.append(
            {
                "Symbol": position["Symbol"],
                "SignalTime": position["SignalTime"],
                "EntryTime": position["EntryTime"],
                "ExitTime": row.name,
                "EntryPrice": position["entry_price"],
                "ExitPrice": exit_price,
                "Shares": position["shares"],
                "Stop": position["stop"],
                "Target": position["target"],
                "GrossPnL": gross_pnl,
                "EstimatedCosts": costs,
                "NetPnL": net_pnl,
                "ExitReason": "End of test",
                "ExtensionATR": position["ExtensionATR"],
                "RVOL": position["RVOL"],
            }
        )

    trades = pd.DataFrame(completed)

    if trades.empty:
        print()
        print("No completed trades.")
        return

    wins = trades["NetPnL"] > 0

    gross_wins = trades.loc[
        wins,
        "NetPnL",
    ].sum()

    gross_losses = -trades.loc[
        ~wins,
        "NetPnL",
    ].sum()

    profit_factor = (
        gross_wins / gross_losses
        if gross_losses > 0
        else np.inf
    )

    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)

    final_equity = cash
    net_pnl = final_equity - STARTING_EQUITY
    win_rate = wins.mean() * 100

    print(
        f"Completed trades: {len(trades)}"
    )
    print(
        f"Starting equity: ${STARTING_EQUITY:,.2f}"
    )
    print(
        f"Final equity: ${final_equity:,.2f}"
    )
    print(
        f"Net P&L: ${net_pnl:,.2f}"
    )
    print(
        f"Net return: "
        f"{net_pnl / STARTING_EQUITY * 100:.2f}%"
    )
    print(
        f"Win rate: {win_rate:.1f}%"
    )
    print(
        f"Profit factor: {profit_factor:.2f}"
    )
    print(
        f"Estimated trading costs: "
        f"${trades['EstimatedCosts'].sum():,.2f}"
    )

    print()
    print("Exit reasons:")
    print(
        trades["ExitReason"]
        .value_counts()
        .to_string()
    )

    print()
    print("Results by symbol:")

    summary = (
        trades.groupby("Symbol")
        .agg(
            Trades=("NetPnL", "count"),
            Wins=(
                "NetPnL",
                lambda x: int((x > 0).sum()),
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

    output_path = (
        ROOT
        / "research_lab"
        / "results"
        / "mean_reversion_trades_v03.csv"
    )

    trades.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 100)
    print(
        f"Saved trades to: {output_path}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
