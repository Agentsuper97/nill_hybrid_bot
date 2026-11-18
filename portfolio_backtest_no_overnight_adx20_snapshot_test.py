
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------- SETTINGS ----------------
SYMBOLS = [
    "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO",
    "TSLA", "NFLX", "ORCL", "CRM", "MU", "QCOM", "PLTR", "SPY",
    "QQQ", "IWM", "JPM", "LLY",
]

PERIOD = "60d"
INTERVAL = "15m"

STARTING_CASH = 10_000.00
RISK_PER_TRADE = 0.005       # Risk budget: 0.5% of estimated equity per trade
MAX_POSITION_FRACTION = 0.20 # Maximum 20% of equity in one position
MAX_OPEN_POSITIONS = 5

LOOKBACK = 10
EMA_LENGTH = 20
RVOL_LENGTH = 20
RVOL_MIN = 1.5
ATR_LENGTH = 14
ADX_LENGTH = 14
ADX_MIN = 20.0
STOP_ATR_MULTIPLE = 1.5
TARGET_R_MULTIPLE = 2.0
MAX_HOLD_BARS = 16

# Estimated total round-trip costs as a fraction of entry notional.
# This is a simplified assumption, not a broker-specific fee/slippage model.
ROUND_TRIP_COST_RATE = 0.001

OUTPUT_DIR = Path(".")
TRADES_FILE = OUTPUT_DIR / "portfolio_trades_no_overnight_adx20_snapshot_test.csv"
EQUITY_FILE = OUTPUT_DIR / "portfolio_equity_no_overnight_adx20_snapshot_test.csv"


def download_symbol(symbol):
    """Load saved intraday bars and prepare indicators."""
    try:
        input_file = Path("intraday_snapshot") / f"{symbol}.csv"
        if not input_file.exists():
            print(f"Skipping {symbol}: snapshot file not found.")
            return None

        df = pd.read_csv(input_file, index_col="Timestamp", parse_dates=False)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")

        needed = ["Open", "High", "Low", "Close", "Volume"]
        if any(col not in df.columns for col in needed):
            print(f"Skipping {symbol}: expected price/volume columns missing.")
            return None

        df = df[needed].copy()
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df = df[~df.index.duplicated(keep="last")].sort_index()

        if len(df) < 60:
            print(f"Skipping {symbol}: not enough bars ({len(df)}).")
            return None

        return prepare_indicators(df)

    except Exception as exc:
        print(f"Skipping {symbol}: snapshot/preparation error: {exc}")
        return None


def prepare_indicators(df):
    """Calculate indicators using current and historical bars only."""
    df = df.copy()

    # At each row, PREV_HIGH is the highest high over the preceding LOOKBACK bars.
    df["PREV_HIGH"] = df["High"].shift(1).rolling(LOOKBACK).max()
    df["EMA"] = df["Close"].ewm(span=EMA_LENGTH, adjust=False).mean()

    prior_volume_average = df["Volume"].shift(1).rolling(RVOL_LENGTH).mean()
    df["RVOL"] = df["Volume"] / prior_volume_average.replace(0, np.nan)

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

    # Wilder-style ADX: trend strength from directional movement.
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    smoothed_tr = true_range.ewm(
        alpha=1 / ADX_LENGTH, adjust=False, min_periods=ADX_LENGTH
    ).mean()
    smoothed_plus_dm = plus_dm.ewm(
        alpha=1 / ADX_LENGTH, adjust=False, min_periods=ADX_LENGTH
    ).mean()
    smoothed_minus_dm = minus_dm.ewm(
        alpha=1 / ADX_LENGTH, adjust=False, min_periods=ADX_LENGTH
    ).mean()

    plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df["ADX"] = dx.ewm(
        alpha=1 / ADX_LENGTH, adjust=False, min_periods=ADX_LENGTH
    ).mean()

    previous_reference_high = df["PREV_HIGH"].shift(1)
    previous_bar_close = df["Close"].shift(1)

    df["SIGNAL"] = (
        (df["Close"] > df["PREV_HIGH"])
        & (previous_bar_close <= previous_reference_high)
        & (df["Close"] > df["EMA"])
        & (df["RVOL"] >= RVOL_MIN)
        & (df["ATR"] > 0)
        & (df["ADX"] >= ADX_MIN)
    )

    return df


def price_at_or_before(df, timestamp):
    """Return the latest close strictly before timestamp, if one exists."""
    available = df.loc[df.index < timestamp, "Close"]
    if available.empty:
        return None
    return float(available.iloc[-1])


def main():
    print("Downloading intraday data...")
    market_data = {}

    for symbol in SYMBOLS:
        df = download_symbol(symbol)
        if df is not None and not df.empty:
            market_data[symbol] = df
            print(f"  {symbol}: {len(df)} bars")

    if not market_data:
        print("No usable market data was downloaded. Nothing to backtest.")
        return

    timeline = sorted(
        set().union(*(set(df.index) for df in market_data.values()))
    )

    cash = STARTING_CASH
    positions = {}
    trades = []
    equity_rows = []

    print("\nRunning portfolio backtest...")

    for timestamp in timeline:
        # 1) Manage positions that were already open before this bar.
        # Stops/targets are checked using this bar's OHLC. If both are touched,
        # the conservative assumption is that the stop is hit first.
        for symbol in list(positions.keys()):
            df = market_data[symbol]
            if timestamp not in df.index:
                continue

            bar = df.loc[timestamp]
            pos = positions[symbol]
            pos["bars_held"] += 1

            exit_price = None
            exit_reason = None

            if float(bar["Open"]) <= pos["stop"]:
                exit_price = float(bar["Open"])
                exit_reason = "Stop (gap)"
            elif float(bar["Open"]) >= pos["target"]:
                exit_price = float(bar["Open"])
                exit_reason = "Target (gap)"
            elif float(bar["Low"]) <= pos["stop"] and float(bar["High"]) >= pos["target"]:
                exit_price = pos["stop"]
                exit_reason = "Stop (both touched; conservative)"
            elif float(bar["Low"]) <= pos["stop"]:
                exit_price = pos["stop"]
                exit_reason = "Stop"
            elif float(bar["High"]) >= pos["target"]:
                exit_price = pos["target"]
                exit_reason = "Target"
            elif pos["bars_held"] >= MAX_HOLD_BARS:
                exit_price = float(bar["Close"])
                exit_reason = "Time exit"

            # Experiment A: close remaining positions on the final
            # available regular-session bar (15:45 New York time).
            if (
                exit_price is None
                and timestamp.hour == 15
                and timestamp.minute == 45
            ):
                exit_price = float(bar["Close"])
                exit_reason = "End of day"

            if exit_price is not None:
                gross_pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                total_cost = pos["estimated_round_trip_cost"]
                net_pnl = gross_pnl - total_cost

                # The estimated round-trip cost was deducted from cash at entry.
                # At exit, add the sale proceeds.
                cash += pos["shares"] * exit_price

                trades.append(
                    {
                        "Symbol": symbol,
                        "EntryTime": pos["entry_time"],
                        "ExitTime": timestamp,
                        "EntryPrice": pos["entry_price"],
                        "EntryADX": pos["entry_adx"],
                        "ExitPrice": exit_price,
                        "Shares": pos["shares"],
                        "Stop": pos["stop"],
                        "Target": pos["target"],
                        "BarsHeld": pos["bars_held"],
                        "ExitReason": exit_reason,
                        "GrossPnL": gross_pnl,
                        "EstimatedCosts": total_cost,
                        "NetPnL": net_pnl,
                    }
                )
                del positions[symbol]

        # 2) Find new signals from the previous completed bar.
        # New entries are blocked on the final regular-session bar below.
        # Rank candidates by the previous bar's RVOL so SYMBOLS list order
        # does not decide which trades get the limited portfolio slots.
        candidates = []

        for symbol, df in market_data.items():
            # Do not open fresh positions on the 15:45 bar.
            if (
                (timestamp.hour == 15 and timestamp.minute == 45)
                or symbol in positions
                or timestamp not in df.index
            ):
                continue

            prior_rows = df.loc[df.index < timestamp]
            if prior_rows.empty:
                continue

            signal_bar = prior_rows.iloc[-1]
            current_bar = df.loc[timestamp]

            if bool(signal_bar["SIGNAL"]):
                entry_price = float(current_bar["Open"])
                atr = float(signal_bar["ATR"])
                if not np.isfinite(entry_price) or not np.isfinite(atr) or entry_price <= 0 or atr <= 0:
                    continue

                stop_distance = STOP_ATR_MULTIPLE * atr
                candidates.append(
                    {
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "atr": atr,
                        "stop_distance": stop_distance,
                        "rvol": float(signal_bar["RVOL"]),
                        "adx": float(signal_bar["ADX"]),
                    }
                )

        candidates.sort(key=lambda item: item["rvol"], reverse=True)

        for candidate in candidates:
            if len(positions) >= MAX_OPEN_POSITIONS:
                break

            symbol = candidate["symbol"]
            if symbol in positions:
                continue

            entry = candidate["entry_price"]
            stop_distance = candidate["stop_distance"]
            stop_price = entry - stop_distance
            target_price = entry + TARGET_R_MULTIPLE * stop_distance

            # Estimate equity using cash plus the most recent closes strictly
            # before this timestamp. This avoids using the current bar's close
            # to size a trade that enters at the current bar's open.
            equity_for_sizing = cash
            for held_symbol, pos in positions.items():
                held_df = market_data[held_symbol]
                prior_close = price_at_or_before(held_df, timestamp)
                if prior_close is not None:
                    equity_for_sizing += pos["shares"] * prior_close

            if equity_for_sizing <= 0:
                continue

            risk_budget = equity_for_sizing * RISK_PER_TRADE
            shares_by_risk = int(risk_budget / stop_distance)

            max_position_value = equity_for_sizing * MAX_POSITION_FRACTION
            shares_by_cap = int(max_position_value / entry)

            # Reserve the estimated round-trip cost as well as purchase value.
            per_share_cash_requirement = entry * (1 + ROUND_TRIP_COST_RATE)
            shares_by_cash = int(cash / per_share_cash_requirement)

            shares = min(shares_by_risk, shares_by_cap, shares_by_cash)
            if shares < 1:
                continue

            entry_notional = shares * entry
            estimated_cost = entry_notional * ROUND_TRIP_COST_RATE

            cash -= entry_notional + estimated_cost

            positions[symbol] = {
                "entry_time": timestamp,
                "entry_price": entry,
                "entry_adx": float(candidate["adx"]),
                "shares": shares,
                "stop": stop_price,
                "target": target_price,
                "bars_held": 0,
                "estimated_round_trip_cost": estimated_cost,
            }

        # 3) Record end-of-bar mark-to-market equity.
        market_value = 0.0
        for symbol, pos in positions.items():
            df = market_data[symbol]
            if timestamp in df.index:
                mark_price = float(df.loc[timestamp, "Close"])
            else:
                prior_close = price_at_or_before(df, timestamp)
                mark_price = prior_close if prior_close is not None else pos["entry_price"]
            market_value += pos["shares"] * mark_price

        equity_rows.append(
            {
                "Timestamp": timestamp,
                "Cash": cash,
                "OpenPositionValue": market_value,
                "Equity": cash + market_value,
                "OpenPositions": len(positions),
            }
        )

    # 4) Close any remaining positions at each symbol's final available close.
    for symbol in list(positions.keys()):
        pos = positions[symbol]
        df = market_data[symbol]
        exit_time = df.index[-1]
        exit_price = float(df.iloc[-1]["Close"])

        gross_pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        total_cost = pos["estimated_round_trip_cost"]
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

    # Replace the final equity point with the actual cash after closing any
    # positions left open at the end of the data.
    if not equity_df.empty:
        equity_df.loc[equity_df.index[-1], "Cash"] = cash
        equity_df.loc[equity_df.index[-1], "OpenPositionValue"] = 0.0
        equity_df.loc[equity_df.index[-1], "Equity"] = cash
        equity_df.loc[equity_df.index[-1], "OpenPositions"] = 0

    trades_df.to_csv(TRADES_FILE, index=False)
    equity_df.to_csv(EQUITY_FILE, index=False)

    final_equity = cash
    net_pnl_total = final_equity - STARTING_CASH
    net_return = net_pnl_total / STARTING_CASH

    if not equity_df.empty:
        equity_series = equity_df["Equity"].astype(float)
        running_peak = equity_series.cummax()
        drawdown = (equity_series / running_peak) - 1
        max_drawdown = float(drawdown.min())
    else:
        max_drawdown = 0.0

    print("\n========== PORTFOLIO BACKTEST RESULTS ==========")
    print(f"Symbols with data: {len(market_data)} / {len(SYMBOLS)}")
    print(f"Completed trades: {len(trades_df)}")
    print(f"Starting equity: ${STARTING_CASH:,.2f}")
    print(f"Final equity after closing positions: ${final_equity:,.2f}")
    print(f"Net P&L: ${net_pnl_total:,.2f}")
    print(f"Net return: {net_return:.2%}")
    print(f"Max recorded equity drawdown: {max_drawdown:.2%}")

    if not trades_df.empty:
        wins = int((trades_df["NetPnL"] > 0).sum())
        losses = int((trades_df["NetPnL"] < 0).sum())
        win_rate = wins / len(trades_df)

        gross_profit = float(trades_df.loc[trades_df["NetPnL"] > 0, "NetPnL"].sum())
        gross_loss = abs(float(trades_df.loc[trades_df["NetPnL"] < 0, "NetPnL"].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        print(f"Win rate: {win_rate:.1%}")
        print(f"Profit factor (net winners / net losers): {profit_factor:.2f}")
        print(f"Estimated trading costs: ${trades_df['EstimatedCosts'].sum():,.2f}")

        by_symbol = (
            trades_df.groupby("Symbol")
            .agg(
                Trades=("NetPnL", "count"),
                GrossPnL=("GrossPnL", "sum"),
                EstimatedCosts=("EstimatedCosts", "sum"),
                NetPnL=("NetPnL", "sum"),
            )
            .sort_values("NetPnL", ascending=False)
        )
        print("\nResults by symbol:")
        print(by_symbol.round(2).to_string())
    else:
        print("No trades were generated.")

    print(f"\nSaved trades to: {TRADES_FILE.resolve()}")
    print(f"Saved equity curve to: {EQUITY_FILE.resolve()}")
    print("\nResearch note: this is a simplified historical simulation, not a prediction.")
    print("Intrabar stop/target ordering, slippage, spreads, and data quality can materially")
    print("change results. Do not use these results as evidence of live profitability.")


if __name__ == "__main__":
    main()
