from pathlib import Path
import numpy as np
import pandas as pd

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

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df.set_index("timestamp")
        .sort_index()
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


def add_features(df):
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

    df["ema20"] = (
        df["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["ema_slope_4"] = df["ema20"].diff(4)

    df["avg_volume20"] = (
        df["volume"]
        .shift(1)
        .rolling(20)
        .mean()
    )

    df["rvol"] = (
        df["volume"]
        / df["avg_volume20"].replace(0, np.nan)
    )

    # Intraday VWAP.
    ny = df.index.tz_convert(
        "America/New_York"
    )

    session_date = ny.date

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3.0

    pv = typical_price * df["volume"]

    df["session_date"] = session_date

    df["cum_pv"] = (
        pv.groupby(session_date)
        .cumsum()
    )

    df["cum_volume"] = (
        df["volume"]
        .groupby(session_date)
        .cumsum()
    )

    df["vwap"] = (
        df["cum_pv"]
        / df["cum_volume"].replace(0, np.nan)
    )

    # Candle mechanics.
    df["range"] = (
        df["high"] - df["low"]
    )

    df["body"] = (
        df["close"] - df["open"]
    )

    df["body_fraction"] = (
        df["body"].abs()
        / df["range"].replace(0, np.nan)
    )

    df["bullish_body_fraction"] = np.where(
        df["body"] > 0,
        df["body"] / df["range"].replace(0, np.nan),
        0.0,
    )

    # Distance from VWAP and EMA.
    df["vwap_distance_atr"] = (
        (df["vwap"] - df["close"])
        / df["atr14"].replace(0, np.nan)
    )

    df["ema_distance_atr"] = (
        (df["close"] - df["ema20"])
        / df["atr14"].replace(0, np.nan)
    )

    # Whether this bar actually stopped making a new
    # short-term low.
    prior_low_3 = (
        df["low"]
        .shift(1)
        .rolling(3)
        .min()
    )

    df["stopped_new_low"] = (
        df["low"] >= prior_low_3
    )

    # Close location within candle.
    df["close_location"] = (
        (df["close"] - df["low"])
        / df["range"].replace(0, np.nan)
    )

    return df


def row_at_or_before(df, timestamp):
    timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    pos = df.index.searchsorted(
        timestamp,
        side="right",
    ) - 1

    if pos < 0 or pos >= len(df):
        return None, None

    return df.iloc[pos], pos


def grouped_summary(group):
    wins = group["NetPnL"] > 0

    gross_wins = group.loc[
        wins,
        "NetPnL",
    ].sum()

    gross_losses = -group.loc[
        ~wins,
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
            "Wins": int(wins.sum()),
            "WinRate": wins.mean() * 100,
            "NetPnL": group["NetPnL"].sum(),
            "AvgPnL": group["NetPnL"].mean(),
            "ProfitFactor": pf,
        }
    )


def main():
    print("=" * 100)
    print("MEAN REVERSION v0.2 — TRADE MECHANICS DIAGNOSTIC")
    print("=" * 100)
    print("Uses the corrected v0.2 trade file.")
    print("Research only. No Alpaca. No orders.")
    print()

    trades = pd.read_csv(
        TRADES_PATH
    )

    trades["EntryTime"] = pd.to_datetime(
        trades["EntryTime"],
        utc=True,
    )

    trades["ExitTime"] = pd.to_datetime(
        trades["ExitTime"],
        utc=True,
    )

    data = {}

    for symbol in SYMBOLS:
        data[symbol] = add_features(
            load_symbol(symbol)
        )

    rows = []

    for _, trade in trades.iterrows():
        symbol = trade["Symbol"]
        df = data[symbol]

        signal_row, signal_pos = row_at_or_before(
            df,
            trade["SignalTime"],
        )

        entry_row, entry_pos = row_at_or_before(
            df,
            trade["EntryTime"],
        )

        if signal_row is None or entry_row is None:
            continue

        if entry_pos <= 0:
            continue

        previous_entry_row = df.iloc[
            entry_pos - 1
        ]

        # How far the signal candle moved toward VWAP.
        signal_vwap_distance = (
            float(signal_row["vwap"])
            - float(signal_row["close"])
        )

        # Forward movement from entry toward the VWAP target.
        target_distance = (
            float(signal_row["vwap"])
            - float(trade["EntryPrice"])
        )

        # First post-entry bar behaviour.
        next_pos = entry_pos + 1

        if next_pos < len(df):
            next_row = df.iloc[next_pos]

            first_bar_return = (
                float(next_row["close"])
                / float(entry_row["open"])
                - 1.0
            )

            first_bar_low_distance = (
                float(next_row["low"])
                - float(entry_row["open"])
            ) / float(entry_row["atr14"])
        else:
            first_bar_return = np.nan
            first_bar_low_distance = np.nan

        ny_entry = (
            trade["EntryTime"]
            .tz_convert("America/New_York")
        )

        rows.append(
            {
                "Symbol": symbol,
                "EntryTime": trade["EntryTime"],
                "ExitTime": trade["ExitTime"],
                "NetPnL": float(trade["NetPnL"]),
                "ExitReason": trade["ExitReason"],
                "Win": float(trade["NetPnL"]) > 0,

                "ExtensionATR": (
                    float(signal_row["vwap"] - signal_row["close"])
                    / float(signal_row["atr14"])
                    if float(signal_row["atr14"]) > 0
                    else np.nan
                ),

                "RVOL": float(
                    signal_row["rvol"]
                ),

                "SignalBodyFraction": float(
                    signal_row["body_fraction"]
                ),

                "SignalBullishBodyFraction": float(
                    signal_row["bullish_body_fraction"]
                ),

                "SignalCloseLocation": float(
                    signal_row["close_location"]
                ),

                "SignalRangeATR": (
                    float(signal_row["range"])
                    / float(signal_row["atr14"])
                    if signal_row["atr14"] > 0
                    else np.nan
                ),

                "SignalVWAPDistanceATR": (
                    float(
                        signal_row["vwap_distance_atr"]
                    )
                ),

                "SignalEMADistanceATR": (
                    float(
                        signal_row["ema_distance_atr"]
                    )
                ),

                "EMASlopeATR": (
                    float(
                        signal_row["ema_slope_4"]
                    )
                    / float(signal_row["atr14"])
                    if signal_row["atr14"] > 0
                    else np.nan
                ),

                "StoppedNewLow": bool(
                    signal_row["stopped_new_low"]
                ),

                "TargetDistance": target_distance,

                "FirstBarReturnPct": (
                    first_bar_return * 100
                ),

                "FirstBarLowATR": (
                    first_bar_low_distance
                ),

                "EntryHour": ny_entry.hour,
                "EntryMinute": ny_entry.minute,
            }
        )

    diag = pd.DataFrame(rows)

    print(
        f"Diagnostic trades: {len(diag)}"
    )

    print()
    print("=" * 100)
    print("OVERALL WINNERS VS LOSERS")
    print("=" * 100)

    print(
        diag.groupby("Win")
        .apply(
            grouped_summary,
            include_groups=False,
        )
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # Helper for displaying feature buckets.
    def show_bucket(title, column, bins, labels):
        print()
        print("=" * 100)
        print(title)
        print("=" * 100)

        diag[f"{column}_Bucket"] = pd.cut(
            diag[column],
            bins=bins,
            labels=labels,
        )

        result = (
            diag.groupby(
                f"{column}_Bucket",
                observed=False,
            )
            .apply(
                grouped_summary,
                include_groups=False,
            )
        )

        print(
            result.to_string(
                float_format=lambda x: f"{x:.2f}"
            )
        )

    show_bucket(
        "SIGNAL BULLISH BODY STRENGTH",
        "SignalBullishBodyFraction",
        [-np.inf, 0.20, 0.40, 0.60, 0.80, 1.0, np.inf],
        [
            "<0.20",
            "0.20–0.40",
            "0.40–0.60",
            "0.60–0.80",
            "0.80–1.00",
            ">1.00",
        ],
    )

    show_bucket(
        "SIGNAL RANGE SIZE",
        "SignalRangeATR",
        [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf],
        [
            "<0.5 ATR",
            "0.5–1.0 ATR",
            "1.0–1.5 ATR",
            "1.5–2.0 ATR",
            ">2.0 ATR",
        ],
    )

    show_bucket(
        "RVOL AT SIGNAL",
        "RVOL",
        [-np.inf, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf],
        [
            "<1.0",
            "1.0–1.5",
            "1.5–2.0",
            "2.0–3.0",
            "3.0–5.0",
            ">5.0",
        ],
    )

    show_bucket(
        "EMA DISTANCE AT SIGNAL",
        "SignalEMADistanceATR",
        [-np.inf, -3.0, -2.0, -1.0, 0.0, 1.0, np.inf],
        [
            "<-3 ATR",
            "-3 to -2",
            "-2 to -1",
            "-1 to 0",
            "0 to 1",
            ">1",
        ],
    )

    show_bucket(
        "VWAP EXTENSION AT SIGNAL",
        "ExtensionATR",
        [-np.inf, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf],
        [
            "<1.0",
            "1.0–1.5",
            "1.5–2.0",
            "2.0–2.5",
            "2.5–3.0",
            ">3.0",
        ],
    )

    show_bucket(
        "FIRST-BAR RETURN AFTER ENTRY",
        "FirstBarReturnPct",
        [-np.inf, -0.50, -0.25, 0.0, 0.25, 0.50, np.inf],
        [
            "<-0.50%",
            "-0.50 to -0.25%",
            "-0.25 to 0%",
            "0 to 0.25%",
            "0.25–0.50%",
            ">0.50%",
        ],
    )

    print()
    print("=" * 100)
    print("REVERSAL STOPPING NEW LOW")
    print("=" * 100)

    print(
        diag.groupby("StoppedNewLow")
        .apply(
            grouped_summary,
            include_groups=False,
        )
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print()
    print("=" * 100)
    print("TIME OF DAY")
    print("=" * 100)

    diag["TimeBucket"] = pd.cut(
        diag["EntryHour"] * 60
        + diag["EntryMinute"],
        bins=[
            570,
            630,
            690,
            750,
            810,
            870,
            930,
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
        .apply(
            grouped_summary,
            include_groups=False,
        )
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    output_path = (
        ROOT
        / "research_lab"
        / "results"
        / "mean_reversion_trade_mechanics_v02.csv"
    )

    diag.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 100)
    print(
        f"Saved diagnostic: {output_path}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
