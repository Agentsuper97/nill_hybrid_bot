from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "intraday_snapshot"
TRADES_PATH = ROOT / "research_lab" / "results" / "mean_reversion_trades_v02.csv"

SYMBOLS = [
    "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO",
    "TSLA", "NFLX", "ORCL", "CRM", "MU", "QCOM", "PLTR", "SPY",
    "QQQ", "IWM", "JPM", "LLY",
]

def load_symbol(symbol):
    path = DATA_DIR / f"{symbol}.csv"
    df = pd.read_csv(path)

    df.columns = [str(c).strip().lower() for c in df.columns]

    timestamp_col = "timestamp"
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

    df = df.set_index(timestamp_col).sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_indicators(df):
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

    df["atr14"] = true_range.rolling(14).mean()

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema20_slope"] = df["ema20"].diff(4)

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

    atr_adx = true_range.rolling(14).mean().replace(0, np.nan)

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

    denominator = (plus_di + minus_di).replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    df["adx14"] = dx.rolling(14).mean()

    # Intraday session VWAP.
    ny = df.index.tz_convert("America/New_York")
    session = ny.date

    typical_price = (
        df["high"] + df["low"] + df["close"]
    ) / 3.0

    pv = typical_price * df["volume"]

    df["session_date"] = session
    df["cum_pv"] = pv.groupby(session).cumsum()
    df["cum_volume"] = df["volume"].groupby(session).cumsum()

    df["vwap"] = (
        df["cum_pv"]
        / df["cum_volume"].replace(0, np.nan)
    )

    df["extension_atr"] = (
        (df["vwap"] - df["close"])
        / df["atr14"].replace(0, np.nan)
    )

    df["rvol20"] = (
        df["volume"]
        / df["volume"]
        .shift(1)
        .rolling(20)
        .mean()
        .replace(0, np.nan)
    )

    df["below_ema"] = df["close"] < df["ema20"]

    # Trend classification using information available at that bar.
    df["trend_regime"] = np.select(
        [
            (df["adx14"] >= 40) & (df["ema20_slope"] < 0),
            (df["adx14"] >= 40) & (df["ema20_slope"] >= 0),
            (df["adx14"] < 25),
        ],
        [
            "Strong downtrend",
            "Strong uptrend",
            "Weak/range-like",
        ],
        default="Moderate trend",
    )

    return df


def find_row_at_or_after(df, timestamp):
    timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    pos = df.index.searchsorted(timestamp)

    if pos >= len(df):
        return None

    return df.iloc[pos]


def build_market_context():
    market = {}

    for symbol in ["SPY", "QQQ"]:
        df = load_symbol(symbol)
        df = add_indicators(df)

        market[symbol] = df[
            [
                "close",
                "ema20",
                "ema20_slope",
                "adx14",
                "vwap",
            ]
        ].copy()

    return market


def main():
    print("=" * 100)
    print("MEAN REVERSION v0.2 — REGIME DIAGNOSTIC")
    print("=" * 100)
    print("Research only.")
    print("Uses corrected mean_reversion_trades_v02.csv.")
    print("No Alpaca connection. No orders.")
    print()

    if not TRADES_PATH.exists():
        raise FileNotFoundError(
            f"Missing trade file: {TRADES_PATH}"
        )

    trades = pd.read_csv(TRADES_PATH)

    for col in ["EntryTime", "ExitTime"]:
        trades[col] = pd.to_datetime(
            trades[col],
            utc=True,
        )

    market = build_market_context()

    cache = {}

    for symbol in SYMBOLS:
        df = load_symbol(symbol)
        cache[symbol] = add_indicators(df)

    rows = []

    for _, trade in trades.iterrows():
        symbol = trade["Symbol"]
        entry_time = trade["EntryTime"]

        df = cache[symbol]

        row = find_row_at_or_after(
            df,
            entry_time,
        )

        if row is None:
            continue

        spy_row = find_row_at_or_after(
            market["SPY"],
            entry_time,
        )

        qqq_row = find_row_at_or_after(
            market["QQQ"],
            entry_time,
        )

        if spy_row is None or qqq_row is None:
            continue

        stock_trend = row["trend_regime"]

        spy_trend = (
            "Bullish"
            if spy_row["close"] > spy_row["ema20"]
            else "Bearish"
        )

        qqq_trend = (
            "Bullish"
            if qqq_row["close"] > qqq_row["ema20"]
            else "Bearish"
        )

        # Broad market agreement with a long mean-reversion trade.
        if spy_trend == "Bullish" and qqq_trend == "Bullish":
            market_regime = "Broad market bullish"
        elif spy_trend == "Bearish" and qqq_trend == "Bearish":
            market_regime = "Broad market bearish"
        else:
            market_regime = "Mixed market"

        stock_below_ema = bool(row["close"] < row["ema20"])

        rows.append(
            {
                "Symbol": symbol,
                "EntryTime": entry_time,
                "NetPnL": float(trade["NetPnL"]),
                "ExitReason": trade["ExitReason"],
                "StockTrend": stock_trend,
                "MarketRegime": market_regime,
                "StockBelowEMA20": stock_below_ema,
                "ADX": float(row["adx14"]),
                "EMASlope": float(row["ema20_slope"]),
                "ExtensionATR": float(row["extension_atr"]),
                "RVOL": float(row["rvol20"]),
                "SPY_ADX": float(spy_row["adx14"]),
                "QQQ_ADX": float(qqq_row["adx14"]),
                "SPY_EMA20_Slope": float(spy_row["ema20_slope"]),
                "QQQ_EMA20_Slope": float(qqq_row["ema20_slope"]),
            }
        )

    diag = pd.DataFrame(rows)

    if diag.empty:
        raise RuntimeError("No diagnostic rows were created.")

    diag["Win"] = diag["NetPnL"] > 0

    def grouped_summary(group):
        gross_wins = group.loc[
            group["NetPnL"] > 0,
            "NetPnL",
        ].sum()

        gross_losses = -group.loc[
            group["NetPnL"] < 0,
            "NetPnL",
        ].sum()

        pf = (
            gross_wins / gross_losses
            if gross_losses > 0
            else np.inf
        )

        return pd.Series(
            {
                "Trades": len(group),
                "Wins": int(group["Win"].sum()),
                "WinRate": group["Win"].mean() * 100,
                "NetPnL": group["NetPnL"].sum(),
                "AvgPnL": group["NetPnL"].mean(),
                "ProfitFactor": pf,
            }
        )

    print("=" * 100)
    print("BY STOCK TREND REGIME")
    print("=" * 100)

    print(
        diag.groupby("StockTrend")
        .apply(grouped_summary, include_groups=False)
        .sort_values("NetPnL", ascending=False)
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print()
    print("=" * 100)
    print("BY BROAD MARKET REGIME")
    print("=" * 100)

    print(
        diag.groupby("MarketRegime")
        .apply(grouped_summary, include_groups=False)
        .sort_values("NetPnL", ascending=False)
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print()
    print("=" * 100)
    print("BY STOCK BELOW/ABOVE EMA20")
    print("=" * 100)

    print(
        diag.groupby("StockBelowEMA20")
        .apply(grouped_summary, include_groups=False)
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print()
    print("=" * 100)
    print("BY EXTENSION SIZE")
    print("=" * 100)

    diag["ExtensionBucket"] = pd.cut(
        diag["ExtensionATR"],
        bins=[-np.inf, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf],
        labels=[
            "<1.0",
            "1.0–1.5",
            "1.5–2.0",
            "2.0–2.5",
            "2.5–3.0",
            ">3.0",
        ],
    )

    print(
        diag.groupby(
            "ExtensionBucket",
            observed=False,
        )
        .apply(grouped_summary, include_groups=False)
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print()
    print("=" * 100)
    print("BY TIME OF DAY")
    print("=" * 100)

    ny_entry = diag["EntryTime"].dt.tz_convert(
        "America/New_York"
    )

    diag["TimeBucket"] = pd.cut(
        (
            ny_entry.dt.hour * 60
            + ny_entry.dt.minute
        ),
        bins=[
            570,   # 09:30
            630,   # 10:30
            690,   # 11:30
            750,   # 12:30
            810,   # 13:30
            870,   # 14:30
            930,   # 15:30
        ],
        labels=[
            "09:30–10:30",
            "10:30–11:30",
            "11:30–12:30",
            "12:30–13:30",
            "13:30–14:30",
            "14:30–15:30",
        ],
        include_lowest=True,
    )

    print(
        diag.groupby(
            "TimeBucket",
            observed=False,
        )
        .apply(grouped_summary, include_groups=False)
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print()
    print("=" * 100)
    print("BY ADX")
    print("=" * 100)

    diag["ADXBucket"] = pd.cut(
        diag["ADX"],
        bins=[-np.inf, 20, 25, 30, 40, 50, np.inf],
        labels=[
            "<20",
            "20–25",
            "25–30",
            "30–40",
            "40–50",
            ">50",
        ],
    )

    print(
        diag.groupby(
            "ADXBucket",
            observed=False,
        )
        .apply(grouped_summary, include_groups=False)
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    output_path = (
        ROOT
        / "research_lab"
        / "results"
        / "mean_reversion_regime_diagnostic_v02.csv"
    )

    diag.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 100)
    print(f"Saved diagnostic: {output_path}")
    print(f"Diagnostic trades: {len(diag)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
