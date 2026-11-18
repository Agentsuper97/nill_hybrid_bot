from pathlib import Path
import pandas as pd
import yfinance as yf

SYMBOLS = [
    "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO",
    "TSLA", "NFLX", "ORCL", "CRM", "MU", "QCOM", "PLTR", "SPY",
    "QQQ", "IWM", "JPM", "LLY",
]

PERIOD = "60d"
INTERVAL = "15m"
OUTPUT_DIR = Path("intraday_snapshot")
OUTPUT_DIR.mkdir(exist_ok=True)

for symbol in SYMBOLS:
    print(f"Downloading {symbol}...")
    try:
        df = yf.download(
            symbol,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            print(f"  Skipped {symbol}: no data returned.")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in needed):
            print(f"  Skipped {symbol}: missing required columns.")
            continue

        df = df[needed].copy()
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df = df[~df.index.duplicated(keep="last")].sort_index()

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")

        # Exclude the latest bar, which may still be forming.
        if len(df) > 1:
            df = df.iloc[:-1].copy()

        output_file = OUTPUT_DIR / f"{symbol}.csv"
        df.to_csv(output_file, index_label="Timestamp")
        print(f"  Saved {len(df)} bars to {output_file}")

    except Exception as exc:
        print(f"  Failed {symbol}: {exc}")

print(f"\nSnapshot folder: {OUTPUT_DIR.resolve()}")
print("Keep these files unchanged while comparing ADX thresholds.")
