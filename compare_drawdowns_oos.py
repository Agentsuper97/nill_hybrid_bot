import pandas as pd
import yfinance as yf

STARTING_CASH = 10_000.00
EQUITY_FILE = "spy_drawdown_oos_equity.csv"


def max_drawdown(values):
    values = values.astype(float)
    running_peak = values.cummax()
    return float((values / running_peak - 1).min())


# Load the strategy's saved daily equity curve.
equity = pd.read_csv(EQUITY_FILE, parse_dates=["Timestamp"])
equity["Date"] = pd.to_datetime(equity["Timestamp"], utc=True).dt.date
equity = (
    equity.sort_values("Timestamp")
    .groupby("Date", as_index=False)
    .last()
)

strategy = equity.set_index("Date")["Equity"].astype(float)

# Download adjusted SPY daily closing prices.
spy = yf.download(
    "SPY",
    period="5y",
    interval="1d",
    auto_adjust=True,
    progress=False,
)

if spy.empty:
    raise SystemExit("SPY download failed. Please try again.")

if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

spy.index = pd.to_datetime(spy.index, utc=True).date
spy = spy["Close"].dropna()

# Restrict both series to the strategy's recorded test dates.
common_dates = strategy.index.intersection(spy.index)

if len(common_dates) < 2:
    raise SystemExit("Not enough common dates to compare.")

strategy = strategy.loc[common_dates]
spy = spy.loc[common_dates]

# Use the same initial-capital baseline and drawdown formula.
strategy_curve = pd.concat([
    pd.Series([STARTING_CASH], index=[common_dates[0] - pd.Timedelta(days=1)]),
    strategy,
]).sort_index()

spy_curve = STARTING_CASH * spy / float(spy.iloc[0])
spy_curve.index = common_dates

spy_curve = pd.concat([
    pd.Series([STARTING_CASH], index=[common_dates[0] - pd.Timedelta(days=1)]),
    spy_curve,
]).sort_index()

print("\n=== ALIGNED DRAWDOWN COMPARISON ===")
print(f"Common trading dates: {len(common_dates)}")
print(f"Period: {common_dates[0]} to {common_dates[-1]}")
print(f"Strategy max drawdown: {max_drawdown(strategy_curve):.2%}")
print(f"SPY max drawdown:      {max_drawdown(spy_curve):.2%}")
print(f"Strategy final marked equity: ${strategy.iloc[-1]:,.2f}")
print(f"SPY final value:              ${spy_curve.iloc[-1]:,.2f}")
print("\nNote: SPY is buy-and-hold with adjusted closes.")
print("Strategy equity is marked to market; this is a historical")
print("comparison, not evidence of future or live performance.")
