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
    df = pd.read_csv(DATA_DIR / f"{symbol}.csv")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.set_index("timestamp").sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_features(df):
    df = df.copy()

    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr14"] = tr.rolling(14).mean()

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

    # Session VWAP.
    ny = df.index.tz_convert("America/New_York")
    session_date = ny.date

    typical = (
        df["high"] + df["low"] + df["close"]
    ) / 3.0

    pv = typical * df["volume"]

    df["session_date"] = session_date
    df["cum_pv"] = pv.groupby(session_date).cumsum()
    df["cum_volume"] = df["volume"].groupby(session_date).cumsum()

    df["vwap"] = (
        df["cum_pv"]
        / df["cum_volume"].replace(0, np.nan)
    )

    # Candle geometry.
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]

    df["body_fraction"] = (
        df["body"].abs()
        / df["range"].replace(0, np.nan)
    )

    df["bullish_body_fraction"] = np.where(
        df["body"] > 0,
        df["body"] / df["range"].replace(0, np.nan),
        0.0,
    )

    df["close_location"] = (
        (df["close"] - df["low"])
        / df["range"].replace(0, np.nan)
    )

    # Momentum before the signal.
    df["return_1"] = df["close"].pct_change(1)
    df["return_2"] = df["close"].pct_change(2)
    df["return_3"] = df["close"].pct_change(3)

    df["momentum_3_atr"] = (
        (df["close"] - df["close"].shift(3))
        / df["atr14"].replace(0, np.nan)
    )

    # Recent low structure.
    df["low_1"] = df["low"].shift(1)
    df["low_2"] = df["low"].shift(2)
    df["low_3"] = df["low"].shift(3)

    df["new_low_vs_prev3"] = (
        df["low"] < df["low"].shift(1).rolling(3).min()
    )

    # VWAP and EMA distances.
    df["vwap_distance_atr"] = (
        (df["vwap"] - df["close"])
        / df["atr14"].replace(0, np.nan)
    )

    df["ema_distance_atr"] = (
        (df["close"] - df["ema20"])
        / df["atr14"].replace(0, np.nan)
    )

    return df


def get_row(df, timestamp):
    ts = pd.Timestamp(timestamp)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    pos = df.index.searchsorted(
        ts,
        side="right",
    ) - 1

    if pos < 0 or pos >= len(df):
        return None, None

    return df.iloc[pos], pos


def summary(group):
    wins = group["Win"]

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


def bucket_report(df, column, bins, labels, title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    bucket = pd.cut(
        df[column],
        bins=bins,
        labels=labels,
    )

    result = (
        df.assign(Bucket=bucket)
        .groupby("Bucket", observed=False)
        .apply(summary, include_groups=False)
    )

    print(
        result.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )


def main():
    print("=" * 100)
    print("MEAN REVERSION v0.2 — PRE-ENTRY MECHANICS DIAGNOSTIC")
    print("=" * 100)
    print("Corrected v0.2 trades.")
    print("Only information available by the signal/entry is used.")
    print("Research only. No Alpaca. No orders.")
    print()

    trades = pd.read_csv(TRADES_PATH)

    trades["SignalTime"] = pd.to_datetime(
        trades["SignalTime"],
        utc=True,
    )

    trades["EntryTime"] = pd.to_datetime(
        trades["EntryTime"],
        utc=True,
    )

    data = {
        symbol: add_features(load_symbol(symbol))
        for symbol in SYMBOLS
    }

    rows = []

    for _, trade in trades.iterrows():
        symbol = trade["Symbol"]
        df = data[symbol]

        signal, signal_pos = get_row(
            df,
            trade["SignalTime"],
        )

        entry, entry_pos = get_row(
            df,
            trade["EntryTime"],
        )

        if signal is None or entry is None:
            continue

        if signal_pos < 3 or entry_pos <= signal_pos:
            continue

        # Find the actual extension bar used by the original
        # signal logic: the strongest VWAP extension in the
        # preceding three bars.
        lookback_start = max(
            0,
            signal_pos - 3,
        )

        preceding = df.iloc[
            lookback_start:signal_pos
        ].copy()

        preceding["extension_atr"] = (
            (preceding["vwap"] - preceding["close"])
            / preceding["atr14"].replace(0, np.nan)
        )

        extension = preceding[
            "extension_atr"
        ].idxmax()

        extension_row = preceding.loc[
            extension
        ]

        extension_atr = float(
            extension_row["extension_atr"]
        )

        # First bar after entry is deliberately used only
        # as the OUTCOME label, never as a predictor.
        next_pos = entry_pos + 1

        if next_pos >= len(df):
            continue

        next_bar = df.iloc[next_pos]

        next_bar_return = (
            float(next_bar["close"])
            / float(entry["open"])
            - 1.0
        )

        # Did the next bar move upward from the entry?
        next_bar_up = next_bar_return > 0

        # Did the signal candle reclaim the prior bar?
        prior_signal = df.iloc[signal_pos - 1]

        signal_reclaim_fraction = (
            (
                float(signal["close"])
                - float(prior_signal["close"])
            )
            / float(signal["atr14"])
            if float(signal["atr14"]) > 0
            else np.nan
        )

        # Did the signal candle make a lower low than
        # the extension bar?
        signal_low_vs_extension_atr = (
            (
                float(signal["low"])
                - float(extension_row["low"])
            )
            / float(signal["atr14"])
            if float(signal["atr14"]) > 0
            else np.nan
        )

        ny = trade["SignalTime"].tz_convert(
            "America/New_York"
        )

        rows.append(
            {
                "Symbol": symbol,
                "SignalTime": trade["SignalTime"],
                "EntryTime": trade["EntryTime"],
                "NetPnL": float(trade["NetPnL"]),
                "Win": float(trade["NetPnL"]) > 0,

                # Outcome label — NOT used as predictor.
                "NextBarReturnPct": (
                    next_bar_return * 100
                ),
                "NextBarUp": next_bar_up,

                # Pre-entry features.
                "ExtensionATR": extension_atr,

                "SignalRVOL": float(
                    signal["rvol"]
                ),

                "SignalBodyFraction": float(
                    signal["body_fraction"]
                ),

                "SignalBullishBodyFraction": float(
                    signal["bullish_body_fraction"]
                ),

                "SignalCloseLocation": float(
                    signal["close_location"]
                ),

                "SignalRangeATR": (
                    float(signal["range"])
                    / float(signal["atr14"])
                    if float(signal["atr14"]) > 0
                    else np.nan
                ),

                "SignalVWAPDistanceATR": float(
                    signal["vwap_distance_atr"]
                ),

                "SignalEMADistanceATR": float(
                    signal["ema_distance_atr"]
                ),

                "EMASlopeATR": (
                    float(signal["ema_slope_4"])
                    / float(signal["atr14"])
                    if float(signal["atr14"]) > 0
                    else np.nan
                ),

                "Return1Pct": float(
                    signal["return_1"] * 100
                ),

                "Return2Pct": float(
                    signal["return_2"] * 100
                ),

                "Return3Pct": float(
                    signal["return_3"] * 100
                ),

                "Momentum3ATR": float(
                    signal["momentum_3_atr"]
                ),

                "SignalReclaimATR": signal_reclaim_fraction,

                "SignalLowVsExtensionATR": (
                    signal_low_vs_extension_atr
                ),

                "ExtensionLow": float(
                    extension_row["low"]
                ),

                "SignalLow": float(
                    signal["low"]
                ),

                "EntryHour": ny.hour,
                "EntryMinute": ny.minute,
            }
        )

    diag = pd.DataFrame(rows)

    print(f"Diagnostic trades: {len(diag)}")

    print()
    print("=" * 100)
    print("NEXT-BAR OUTCOME")
    print("=" * 100)

    print(
        diag.groupby("NextBarUp")
        .apply(summary, include_groups=False)
        .to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    bucket_report(
        diag,
        "SignalReclaimATR",
        [-np.inf, -0.50, 0.0, 0.25, 0.50, 1.0, np.inf],
        [
            "<-0.50 ATR",
            "-0.50–0",
            "0–0.25 ATR",
            "0.25–0.50 ATR",
            "0.50–1.0 ATR",
            ">1.0 ATR",
        ],
        "SIGNAL CANDLE RECLAIM VS PRIOR CLOSE",
    )

    bucket_report(
        diag,
        "SignalCloseLocation",
        [0.0, 0.25, 0.50, 0.75, 1.0],
        [
            "0–25%",
            "25–50%",
            "50–75%",
            "75–100%",
        ],
        "SIGNAL CLOSE LOCATION WITHIN CANDLE",
    )

    bucket_report(
        diag,
        "Momentum3ATR",
        [-np.inf, -3.0, -2.0, -1.0, -0.5, 0.0, np.inf],
        [
            "<-3 ATR",
            "-3 to -2",
            "-2 to -1",
            "-1 to -0.5",
            "-0.5 to 0",
            ">0",
        ],
        "3-BAR PRE-SIGNAL MOMENTUM",
    )

    bucket_report(
        diag,
        "SignalLowVsExtensionATR",
        [-np.inf, -1.0, -0.25, 0.0, 0.25, np.inf],
        [
            "<-1 ATR",
            "-1 to -0.25",
            "-0.25 to 0",
            "0 to 0.25",
            ">0.25",
        ],
        "SIGNAL LOW VS EXTENSION-BAR LOW",
    )

    bucket_report(
        diag,
        "ExtensionATR",
        [-np.inf, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf],
        [
            "<1.0",
            "1.0–1.5",
            "1.5–2.0",
            "2.0–2.5",
            "2.5–3.0",
            ">=3.0",
        ],
        "ORIGINAL EXTENSION SIZE",
    )

    bucket_report(
        diag,
        "SignalBullishBodyFraction",
        [-np.inf, 0.20, 0.40, 0.60, 0.80, 1.0],
        [
            "<0.20",
            "0.20–0.40",
            "0.40–0.60",
            "0.60–0.80",
            "0.80–1.00",
        ],
        "BULLISH REVERSAL BODY STRENGTH",
    )

    bucket_report(
        diag,
        "SignalRVOL",
        [-np.inf, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf],
        [
            "<1.0",
            "1.0–1.5",
            "1.5–2.0",
            "2.0–3.0",
            "3.0–5.0",
            ">5.0",
        ],
        "SIGNAL RVOL",
    )

    bucket_report(
        diag,
        "SignalEMADistanceATR",
        [-np.inf, -3.0, -2.0, -1.0, 0.0, np.inf],
        [
            "<-3 ATR",
            "-3 to -2",
            "-2 to -1",
            "-1 to 0",
            ">0",
        ],
        "EMA DISTANCE AT SIGNAL",
    )

    output = (
        ROOT
        / "research_lab"
        / "results"
        / "mean_reversion_preentry_v02.csv"
    )

    diag.to_csv(output, index=False)

    print()
    print("=" * 100)
    print(f"Saved diagnostic: {output}")
    print("=" * 100)


if __name__ == "__main__":
    main()
