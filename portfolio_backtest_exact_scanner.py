import numpy as np
import pandas as pd
from pathlib import Path

SYMBOLS = [
    "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO",
    "TSLA", "NFLX", "ORCL", "CRM", "MU", "QCOM", "PLTR", "SPY",
    "QQQ", "IWM", "JPM", "LLY",
]

SNAPSHOT_DIR = Path("intraday_snapshot")
TRADES_FILE = Path("portfolio_trades_exact_scanner.csv")
EQUITY_FILE = Path("portfolio_equity_exact_scanner.csv")

STARTING_CASH = 10_000.00
RISK_PER_TRADE = 0.005
MAX_OPEN_POSITIONS = 5
MAX_POSITION_FRACTION = 0.20

LOOKBACK = 10
EMA_LENGTH = 20
RVOL_LENGTH = 20
RVOL_MIN = 1.5
ATR_LENGTH = 14
ADX_LENGTH = 14
ADX_MIN = 40.0

STOP_ATR_MULTIPLE = 1.5
TARGET_R_MULTIPLE = 2.0
MAX_HOLD_BARS = 16
ROUND_TRIP_COST_RATE = 0.001

OOS_START = pd.Timestamp("2026-10-01", tz="America/New_York")
OOS_END = pd.Timestamp("2026-10-23", tz="America/New_York")


def calculate_indicators(df):
    """Exact indicator formulas copied from alpaca_paper_monitor.py."""
    df = df.sort_index().copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["EMA"] = df["Close"].ewm(
        span=EMA_LENGTH,
        adjust=False,
    ).mean()

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = true_range.rolling(ATR_LENGTH).mean()

    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

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

    atr_for_adx = (
        true_range
        .rolling(ADX_LENGTH)
        .mean()
        .replace(0, np.nan)
    )

    plus_di = (
        100
        * plus_dm.rolling(ADX_LENGTH).mean()
        / atr_for_adx
    )

    minus_di = (
        100
        * minus_dm.rolling(ADX_LENGTH).mean()
        / atr_for_adx
    )

    denominator = (
        plus_di + minus_di
    ).replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    df["ADX"] = dx.rolling(ADX_LENGTH).mean()

    df["PriorHigh"] = (
        df["High"]
        .shift(1)
        .rolling(LOOKBACK)
        .max()
    )

    df["AverageVolume"] = (
        df["Volume"]
        .shift(1)
        .rolling(RVOL_LENGTH)
        .mean()
    )

    df["RVOL"] = (
        df["Volume"]
        / df["AverageVolume"].replace(0, np.nan)
    )

    df["CrossedPriorHigh"] = (
        (df["Close"] > df["PriorHigh"])
        & (
            df["Close"].shift(1)
            <= df["PriorHigh"].shift(1)
        )
    )

    df["Signal"] = (
        df["CrossedPriorHigh"]
        & (df["Close"] > df["EMA"])
        & (df["RVOL"] >= RVOL_MIN)
        & (df["ADX"] >= ADX_MIN)
        & (df["ATR"] > 0)
    )

    return df


def load_symbol(symbol):
    path = SNAPSHOT_DIR / f"{symbol}.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)

    required = {
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    if not required.issubset(df.columns):
        return None

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        utc=True,
    ).dt.tz_convert("America/New_York")

    df = df.set_index("Timestamp")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Match the scanner's regular-session treatment.
    bar_end = df.index + pd.Timedelta(minutes=15)

    df = df[
        (df.index.time >= pd.Timestamp("09:30").time())
        & (bar_end.time <= pd.Timestamp("16:00").time())
    ]

    if len(df) < 60:
        return None

    return calculate_indicators(df)


def price_at_or_before(df, timestamp):
    prior = df.loc[df.index <= timestamp, "Close"]

    if prior.empty:
        return None

    return float(prior.iloc[-1])


def main():
    market_data = {}

    for symbol in SYMBOLS:
        df = load_symbol(symbol)

        if df is not None:
            market_data[symbol] = df

    if not market_data:
        raise SystemExit("No historical snapshot data loaded.")

    timeline = sorted(
        set().union(
            *(df.index.tolist() for df in market_data.values())
        )
    )

    last_bar_by_day = {}

    for symbol, df in market_data.items():
        for day, group in df.groupby(df.index.date):
            last_bar_by_day[(symbol, day)] = group.index[-1]

    cash = STARTING_CASH
    positions = {}
    trades = []
    equity_rows = []

    for timestamp in timeline:
        if timestamp < OOS_START or timestamp >= OOS_END:
            continue

        # ------------------------------------------------------------
        # 1. Manage existing positions.
        # ------------------------------------------------------------
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            df = market_data[symbol]

            if timestamp not in df.index:
                continue

            bar = df.loc[timestamp]

            pos["bars_held"] += 1

            exit_price = None
            exit_reason = None

            # Conservative gap handling.
            if float(bar["Open"]) <= pos["stop"]:
                exit_price = float(bar["Open"])
                exit_reason = "Stop (gap)"

            elif float(bar["Open"]) >= pos["target"]:
                exit_price = float(bar["Open"])
                exit_reason = "Target (gap)"

            elif (
                float(bar["Low"]) <= pos["stop"]
                and float(bar["High"]) >= pos["target"]
            ):
                exit_price = pos["stop"]
                exit_reason = "Stop (same bar)"

            elif float(bar["Low"]) <= pos["stop"]:
                exit_price = pos["stop"]
                exit_reason = "Stop"

            elif float(bar["High"]) >= pos["target"]:
                exit_price = pos["target"]
                exit_reason = "Target"

            elif pos["bars_held"] >= MAX_HOLD_BARS:
                exit_price = float(bar["Close"])
                exit_reason = "Time"

            # No overnight positions.
            elif last_bar_by_day.get(
                (symbol, timestamp.date())
            ) == timestamp:
                exit_price = float(bar["Close"])
                exit_reason = "End of day"

            if exit_price is not None:
                shares = pos["shares"]

                gross_pnl = (
                    exit_price - pos["entry_price"]
                ) * shares

                total_cost = pos["estimated_round_trip_cost"]
                net_pnl = gross_pnl - total_cost

                cash += shares * exit_price

                trades.append(
                    {
                        "Symbol": symbol,
                        "EntryTime": pos["entry_time"],
                        "ExitTime": timestamp,
                        "EntryPrice": pos["entry_price"],
                        "ExitPrice": exit_price,
                        "Shares": shares,
                        "Stop": pos["stop"],
                        "Target": pos["target"],
                        "EntryADX": pos["entry_adx"],
                        "EntryRVOL": pos["entry_rvol"],
                        "BarsHeld": pos["bars_held"],
                        "ExitReason": exit_reason,
                        "GrossPnL": gross_pnl,
                        "EstimatedCosts": total_cost,
                        "NetPnL": net_pnl,
                    }
                )

                del positions[symbol]

        # ------------------------------------------------------------
        # 2. Find new signals using the completed prior bar.
        # ------------------------------------------------------------
        candidates = []

        for symbol, df in market_data.items():
            if last_bar_by_day.get((symbol, timestamp.date())) == timestamp:
                continue
                continue

            if symbol in positions:
                continue

            if timestamp not in df.index:
                continue

            prior_rows = df.loc[df.index < timestamp]

            if prior_rows.empty:
                continue

            signal_bar = prior_rows.iloc[-1]
            current_bar = df.loc[timestamp]

            if not bool(signal_bar["Signal"]):
                continue

            entry_price = float(current_bar["Open"])
            atr = float(signal_bar["ATR"])

            if (
                not np.isfinite(entry_price)
                or not np.isfinite(atr)
                or entry_price <= 0
                or atr <= 0
            ):
                continue

            stop_distance = STOP_ATR_MULTIPLE * atr

            candidates.append(
                {
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "stop_distance": stop_distance,
                    "adx": float(signal_bar["ADX"]),
                    "rvol": float(signal_bar["RVOL"]),
                }
            )

        # Match the previous portfolio test's candidate priority.
        candidates.sort(
            key=lambda x: x["rvol"],
            reverse=True,
        )

        for candidate in candidates:
            if len(positions) >= MAX_OPEN_POSITIONS:
                break

            symbol = candidate["symbol"]

            if symbol in positions:
                continue

            entry = candidate["entry_price"]
            stop_distance = candidate["stop_distance"]

            stop_price = (
                entry - stop_distance
            )

            target_price = (
                entry
                + TARGET_R_MULTIPLE * stop_distance
            )

            # Mark existing positions using closes strictly before
            # the entry timestamp.
            equity_for_sizing = cash

            for held_symbol, pos in positions.items():
                held_df = market_data[held_symbol]

                prior_close = price_at_or_before(
                    held_df,
                    timestamp - pd.Timedelta(microseconds=1),
                )

                if prior_close is not None:
                    equity_for_sizing += (
                        pos["shares"] * prior_close
                    )

            if equity_for_sizing <= 0:
                continue

            risk_budget = (
                equity_for_sizing
                * RISK_PER_TRADE
            )

            shares_by_risk = int(
                risk_budget / stop_distance
            )

            max_position_value = (
                equity_for_sizing
                * MAX_POSITION_FRACTION
            )

            shares_by_cap = int(
                max_position_value / entry
            )

            per_share_cash_requirement = (
                entry * (1 + ROUND_TRIP_COST_RATE)
            )

            shares_by_cash = int(
                cash / per_share_cash_requirement
            )

            shares = min(
                shares_by_risk,
                shares_by_cap,
                shares_by_cash,
            )

            if shares < 1:
                continue

            entry_notional = shares * entry
            estimated_cost = (
                entry_notional
                * ROUND_TRIP_COST_RATE
            )

            cash -= (
                entry_notional
                + estimated_cost
            )

            positions[symbol] = {
                "entry_time": timestamp,
                "entry_price": entry,
                "entry_adx": candidate["adx"],
                "entry_rvol": candidate["rvol"],
                "shares": shares,
                "stop": stop_price,
                "target": target_price,
                "bars_held": 0,
                "estimated_round_trip_cost": estimated_cost,
            }

        # ------------------------------------------------------------
        # 3. End-of-bar mark-to-market.
        # ------------------------------------------------------------
        market_value = 0.0

        for symbol, pos in positions.items():
            df = market_data[symbol]

            if timestamp in df.index:
                mark_price = float(
                    df.loc[timestamp, "Close"]
                )
            else:
                prior_close = price_at_or_before(
                    df,
                    timestamp,
                )

                mark_price = (
                    prior_close
                    if prior_close is not None
                    else pos["entry_price"]
                )

            market_value += (
                pos["shares"] * mark_price
            )

        equity_rows.append(
            {
                "Timestamp": timestamp,
                "Cash": cash,
                "OpenPositionValue": market_value,
                "Equity": cash + market_value,
                "OpenPositions": len(positions),
            }
        )

    # ------------------------------------------------------------
    # 4. Safety cleanup: no overnight positions.
    # ------------------------------------------------------------
    for symbol in list(positions.keys()):
        pos = positions[symbol]
        df = market_data[symbol]

        test_df = df.loc[
            (df.index >= OOS_START)
            & (df.index < OOS_END)
        ]

        if test_df.empty:
            continue

        exit_time = test_df.index[-1]
        exit_price = float(
            test_df.iloc[-1]["Close"]
        )

        gross_pnl = (
            exit_price - pos["entry_price"]
        ) * pos["shares"]

        total_cost = pos[
            "estimated_round_trip_cost"
        ]

        net_pnl = gross_pnl - total_cost

        cash += pos["shares"] * exit_price

        trades.append(
            {
                "Symbol": symbol,
                "EntryTime": pos["entry_time"],
                "ExitTime": exit_time,
                "EntryPrice": pos["entry_price"],
                "ExitPrice": exit_price,
                "Shares": pos["shares"],
                "Stop": pos["stop"],
                "Target": pos["target"],
                "EntryADX": pos["entry_adx"],
                "EntryRVOL": pos["entry_rvol"],
                "BarsHeld": pos["bars_held"],
                "ExitReason": "End of data",
                "GrossPnL": gross_pnl,
                "EstimatedCosts": total_cost,
                "NetPnL": net_pnl,
            }
        )

        del positions[symbol]

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)

    if not equity_df.empty:
        equity_df.loc[
            equity_df.index[-1],
            "Cash",
        ] = cash

        equity_df.loc[
            equity_df.index[-1],
            "OpenPositionValue",
        ] = 0.0

        equity_df.loc[
            equity_df.index[-1],
            "Equity",
        ] = cash

        equity_df.loc[
            equity_df.index[-1],
            "OpenPositions",
        ] = 0

    trades_df.to_csv(
        TRADES_FILE,
        index=False,
    )

    equity_df.to_csv(
        EQUITY_FILE,
        index=False,
    )

    final_equity = cash
    net_pnl_total = (
        final_equity - STARTING_CASH
    )

    net_return = (
        net_pnl_total / STARTING_CASH
    )

    if not equity_df.empty:
        equity_series = (
            equity_df["Equity"]
            .astype(float)
        )

        running_peak = equity_series.cummax()

        drawdown = (
            equity_series / running_peak
        ) - 1

        max_drawdown = float(
            drawdown.min()
        )
    else:
        max_drawdown = 0.0

    print()
    print("========== EXACT SCANNER PORTFOLIO BACKTEST ==========")
    print("Historical simulation only.")
    print("No Alpaca connection. No orders.")
    print()
    print(f"Symbols with data: {len(market_data)} / {len(SYMBOLS)}")
    print(f"OOS period: {OOS_START} to {OOS_END}")
    print(f"Completed trades: {len(trades_df)}")
    print(f"Starting equity: ${STARTING_CASH:,.2f}")
    print(f"Final equity: ${final_equity:,.2f}")
    print(f"Net P&L: ${net_pnl_total:,.2f}")
    print(f"Net return: {net_return:.2%}")
    print(
        f"Max recorded equity drawdown: "
        f"{max_drawdown:.2%}"
    )

    if not trades_df.empty:
        wins = int(
            (trades_df["NetPnL"] > 0).sum()
        )

        losses = int(
            (trades_df["NetPnL"] < 0).sum()
        )

        win_rate = (
            wins / len(trades_df)
        )

        gross_profit = float(
            trades_df.loc[
                trades_df["NetPnL"] > 0,
                "NetPnL",
            ].sum()
        )

        gross_loss = abs(
            float(
                trades_df.loc[
                    trades_df["NetPnL"] < 0,
                    "NetPnL",
                ].sum()
            )
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf")
        )

        total_costs = float(
            trades_df["EstimatedCosts"].sum()
        )

        print(f"Win rate: {win_rate:.1%}")
        print(
            "Profit factor "
            f"(net winners / net losers): "
            f"{profit_factor:.2f}"
        )
        print(
            f"Estimated trading costs: "
            f"${total_costs:,.2f}"
        )

        print("\nResults by symbol:")

        by_symbol = (
            trades_df
            .groupby("Symbol")
            .agg(
                Trades=("NetPnL", "count"),
                GrossPnL=("GrossPnL", "sum"),
                EstimatedCosts=(
                    "EstimatedCosts",
                    "sum",
                ),
                NetPnL=("NetPnL", "sum"),
            )
            .sort_values(
                "NetPnL",
                ascending=False,
            )
        )

        print(
            by_symbol
            .round(2)
            .to_string()
        )

        print("\nExit reasons:")

        print(
            trades_df["ExitReason"]
            .value_counts()
            .to_string()
        )

    else:
        print("No trades were generated.")

    print()
    print(
        f"Saved trades to: "
        f"{TRADES_FILE.resolve()}"
    )

    print(
        f"Saved equity curve to: "
        f"{EQUITY_FILE.resolve()}"
    )

    print()
    print(
        "Research note: this is a historical simulation, "
        "not a prediction."
    )
    print(
        "Intrabar ordering, slippage, spreads, "
        "and data quality can materially change results."
    )


if __name__ == "__main__":
    main()
