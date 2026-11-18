import time
import yfinance as yf
import pandas as pd

# Research scanner only: no orders are placed.
SYMBOLS = [
    "NVDA", "AMD", "AAPL", "MSFT", "AMZN",
    "META", "GOOGL", "AVGO", "TSLA", "NFLX",
    "ORCL", "CRM", "MU", "QCOM", "PLTR",
    "SPY", "QQQ", "IWM", "JPM", "LLY",
]

LOOKBACK = 10
EMA_PERIOD = 20
RVOL_PERIOD = 20
MIN_RVOL = 1.5
MIN_PRICE = 10.0
MIN_AVG_DOLLAR_VOLUME = 20_000_000

def scan_symbol(symbol):
    try:
        data = yf.download(
            symbol,
            period="10d",
            interval="15m",
            progress=False,
            auto_adjust=True,
            threads=False,
        )

        if data.empty:
            return None, f"{symbol}: no data"

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.sort_index()
        data = data[~data.index.duplicated(keep="last")]

        if data.index.tz is None:
            return None, f"{symbol}: missing timezone"

        data.index = data.index.tz_convert("America/New_York")

        # Exclude the currently forming candle.
        now = pd.Timestamp.now(tz="America/New_York")
        data = data.loc[
            data.index + pd.Timedelta(minutes=15) <= now
        ].copy()

        required = max(LOOKBACK + 2, EMA_PERIOD + 2,
                       RVOL_PERIOD + 2)

        if len(data) < required:
            return None, f"{symbol}: insufficient candles"

        close = data["Close"]
        high = data["High"]
        volume = data["Volume"]

        entry = float(close.iloc[-1])
        previous_close = float(close.iloc[-2])
        previous_high = float(
            high.iloc[-LOOKBACK-1:-1].max()
        )

        ema20 = close.ewm(
            span=EMA_PERIOD, adjust=False
        ).mean()
        latest_ema20 = float(ema20.iloc[-1])

        average_volume = float(
            volume.iloc[-RVOL_PERIOD-1:-1].mean()
        )
        current_volume = float(volume.iloc[-1])

        # Approximate liquidity filter using available candles.
        avg_dollar_volume = float(
            (close * volume).tail(100).mean()
        )

        if average_volume <= 0:
            return None, f"{symbol}: invalid volume baseline"

        rvol = current_volume / average_volume

        breakout = (
            previous_close <= previous_high
            and entry > previous_high
        )
        momentum = entry > latest_ema20
        volume_ok = rvol >= MIN_RVOL
        liquid = avg_dollar_volume >= MIN_AVG_DOLLAR_VOLUME
        price_ok = entry >= MIN_PRICE

        score = (
            40 * breakout
            + 25 * momentum
            + 25 * volume_ok
            + 10 * liquid
        )

        result = {
            "Symbol": symbol,
            "Close": entry,
            "EMA20": latest_ema20,
            "RVOL": rvol,
            "Breakout": breakout,
            "Above EMA20": momentum,
            "Volume OK": volume_ok,
            "Liquidity OK": liquid,
            "Score": score,
            "Candle": str(data.index[-1]),
            "Passed": (
                breakout and momentum and volume_ok
                and liquid and price_ok
            ),
        }

        return result, None

    except Exception as exc:
        return None, f"{symbol}: {type(exc).__name__}: {exc}"


def main():
    print("MULTI-STOCK 15-MINUTE RESEARCH SCANNER")
    print("Research only — no orders will be placed.\n")

    results = []
    errors = []

    for symbol in SYMBOLS:
        print(f"Scanning {symbol}...")
        result, error = scan_symbol(symbol)

        if result:
            results.append(result)
        if error:
            errors.append(error)

        time.sleep(0.2)

    if not results:
        print("\nNo usable market data received.")
        for error in errors:
            print(error)
        return

    table = pd.DataFrame(results)
    table = table.sort_values(
        ["Passed", "Score", "RVOL"],
        ascending=[False, False, False],
    )

    print("\n--- RANKED CANDIDATES ---")
    print(table[
        ["Symbol", "Close", "EMA20", "RVOL",
         "Breakout", "Above EMA20", "Volume OK",
         "Liquidity OK", "Score", "Passed"]
    ].to_string(
        index=False,
        formatters={
            "Close": "${:.2f}".format,
            "EMA20": "${:.2f}".format,
            "RVOL": "{:.2f}x".format,
        },
    ))

    passed = table[table["Passed"]]

    print(f"\nStocks scanned successfully: {len(results)}")
    print(f"Potential setups: {len(passed)}")
    print("A passing signal is not a buy recommendation.")

    if errors:
        print(f"\nData warnings: {len(errors)}")
        for error in errors:
            print(error)

    table.to_csv("scanner_results.csv", index=False)
    print("\nSaved results to scanner_results.csv")


if __name__ == "__main__":
    main()
