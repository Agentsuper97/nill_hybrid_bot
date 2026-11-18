import time
import numpy as np
import pandas as pd
import yfinance as yf

SYMBOLS = [
    "NVDA", "AMD", "AAPL", "MSFT", "AMZN",
    "META", "GOOGL", "AVGO", "TSLA", "NFLX",
    "ORCL", "CRM", "MU", "QCOM", "PLTR",
    "SPY", "QQQ", "IWM", "JPM", "LLY",
]

LOOKBACK = 10
EMA_PERIOD = 20
RVOL_PERIOD = 20
ATR_PERIOD = 14
MIN_RVOL = 1.5

STOP_ATR = 1.5
TARGET_R = 2.0
MAX_HOLD_BARS = 16

# Approximate total round-trip trading costs as a fraction
# of entry notional. 0.001 = 0.10%, including slippage.
ROUND_TRIP_COST = 0.001


def load_data(symbol):
    df = yf.download(
        symbol,
        period="60d",
        interval="15m",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if df.index.tz is None:
        return None

    df.index = df.index.tz_convert("America/New_York")

    # Remove any currently forming candle.
    now = pd.Timestamp.now(tz="America/New_York")
    df = df.loc[
        df.index + pd.Timedelta(minutes=15) <= now
    ].copy()

    return df


def prepare(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["EMA"] = close.ewm(
        span=EMA_PERIOD, adjust=False
    ).mean()

    df["PREV_HIGH"] = high.shift(1).rolling(
        LOOKBACK
    ).max()

    df["AVG_VOLUME"] = volume.shift(1).rolling(
        RVOL_PERIOD
    ).mean()

    df["RVOL"] = volume / df["AVG_VOLUME"]

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = true_range.rolling(
        ATR_PERIOD
    ).mean()

    previous_reference_high = df["PREV_HIGH"].shift(1)

    df["SIGNAL"] = (
        (close > df["PREV_HIGH"])
        & (previous_close <= previous_reference_high)
        & (close > df["EMA"])
        & (df["RVOL"] >= MIN_RVOL)
        & (df["ATR"] > 0)
    )

    return df


def backtest_symbol(symbol, df):
    df = prepare(df)
    trades = []
    i = max(LOOKBACK, EMA_PERIOD, RVOL_PERIOD, ATR_PERIOD) + 2

    while i < len(df) - 1:
        row = df.iloc[i]

        if not bool(row["SIGNAL"]):
            i += 1
            continue

        # Signal forms at candle close; enter next candle open.
        entry_i = i + 1
        entry = float(df["Open"].iloc[entry_i])
        atr = float(row["ATR"])

        if not np.isfinite(entry) or not np.isfinite(atr):
            i += 1
            continue

        risk = STOP_ATR * atr
        stop = entry - risk
        target = entry + TARGET_R * risk

        exit_i = None
        exit_price = None
        exit_reason = None

        last_i = min(
            entry_i + MAX_HOLD_BARS - 1,
            len(df) - 1,
        )

        for j in range(entry_i, last_i + 1):
            bar = df.iloc[j]
            bar_low = float(bar["Low"])
            bar_high = float(bar["High"])

            # Conservative assumption if both levels are
            # touched within the same candle: stop first.
            if bar_low <= stop:
                exit_i = j
                exit_price = min(
                    stop, float(bar["Open"])
                )
                exit_reason = "stop"
                break

            if bar_high >= target:
                exit_i = j
                exit_price = max(
                    target, float(bar["Open"])
                )
                exit_reason = "target"
                break

        if exit_i is None:
            exit_i = last_i
            exit_price = float(df["Close"].iloc[exit_i])
            exit_reason = "time"

        # Approximate costs as a percentage of entry notional.
        gross_return = (exit_price - entry) / entry
        net_return = gross_return - ROUND_TRIP_COST

        trades.append({
            "Symbol": symbol,
            "EntryTime": str(df.index[entry_i]),
            "ExitTime": str(df.index[exit_i]),
            "Entry": entry,
            "Exit": exit_price,
            "ExitReason": exit_reason,
            "RVOL": float(row["RVOL"]),
            "GrossReturnPct": gross_return * 100,
            "NetReturnPct": net_return * 100,
            "InitialRiskPct": risk / entry * 100,
        })

        # Do not overlap trades in the same symbol.
        i = exit_i + 1

    return trades


def main():
    all_trades = []

    for symbol in SYMBOLS:
        print(f"Backtesting {symbol}...")
        try:
            df = load_data(symbol)

            if df is None or len(df) < 100:
                print("  Skipped: insufficient usable data")
                continue

            all_trades.extend(backtest_symbol(symbol, df))

        except Exception as exc:
            print(f"  Error: {type(exc).__name__}: {exc}")

        time.sleep(0.2)

    if not all_trades:
        print(
            "\nNo trades generated. This is not proof "
            "that the strategy is unprofitable."
        )
        return

    trades = pd.DataFrame(all_trades)
    trades.to_csv("backtest_trades.csv", index=False)

    print("\n--- RESULTS BY SYMBOL ---")
    summaries = []

    for symbol, group in trades.groupby("Symbol"):
        returns = group["NetReturnPct"] / 100
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1

        winners = returns[returns > 0]
        losers = returns[returns < 0]

        summaries.append({
            "Symbol": symbol,
            "Trades": len(group),
            "WinRatePct": (returns > 0).mean() * 100,
            "MeanNetPct": returns.mean() * 100,
            "ProfitFactor": (
                winners.sum() / abs(losers.sum())
                if len(losers) and abs(losers.sum()) > 0
                else np.nan
            ),
            "MaxDrawdownPct": drawdown.min() * 100,
            "NetCompoundedPct": (
                equity.iloc[-1] - 1
            ) * 100,
        })

    summary = pd.DataFrame(summaries)
    summary = summary.sort_values(
        "MeanNetPct", ascending=False
    )

    print(summary.to_string(
        index=False,
        formatters={
            "WinRatePct": "{:.1f}".format,
            "MeanNetPct": "{:.3f}".format,
            "ProfitFactor": "{:.2f}".format,
            "MaxDrawdownPct": "{:.2f}".format,
            "NetCompoundedPct": "{:.2f}".format,
        },
    ))

    print(f"\nTotal trades: {len(trades)}")
    print("Saved detailed trades to backtest_trades.csv")
    print(
        "\nWARNING: exploratory backtest only. Results use "
        "limited intraday history and simplified costs. "
        "They are not evidence of future profitability."
    )


if __name__ == "__main__":
    main()
