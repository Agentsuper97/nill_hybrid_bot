from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRADES_PATH = ROOT / "research_lab" / "results" / "mean_reversion_trades_v01.csv"

trades = pd.read_csv(TRADES_PATH, parse_dates=["SignalTime", "EntryTime", "ExitTime"])

trades["Date"] = trades["EntryTime"].dt.date
trades["Hour"] = trades["EntryTime"].dt.hour
trades["Minute"] = trades["EntryTime"].dt.minute
trades["EntryTimeOnly"] = (
    trades["EntryTime"].dt.hour * 60
    + trades["EntryTime"].dt.minute
)

trades["ExtensionBucket"] = pd.cut(
    trades["EntryExtensionATR"],
    bins=[-np.inf, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, np.inf],
    labels=[
        "<=1.25",
        "1.25-1.50",
        "1.50-1.75",
        "1.75-2.00",
        "2.00-2.50",
        "2.50-3.00",
        ">3.00",
    ],
)

trades["RVOLBucket"] = pd.cut(
    trades["EntryRVOL"],
    bins=[-np.inf, 1.25, 1.50, 2.00, 3.00, np.inf],
    labels=[
        "<=1.25",
        "1.25-1.50",
        "1.50-2.00",
        "2.00-3.00",
        ">3.00",
    ],
)

trades["ATRBucket"] = pd.qcut(
    trades["EntryATR"],
    q=4,
    duplicates="drop",
)

trades["Month"] = trades["EntryTime"].dt.to_period("M").astype(str)

print("=" * 100)
print("MEAN REVERSION v0.1 — TRADE DIAGNOSTICS")
print("=" * 100)
print(f"Trades analysed: {len(trades)}")
print()

def show_group(title, group_col):
    print("-" * 100)
    print(title)

    result = (
        trades.groupby(group_col, observed=False)
        .agg(
            Trades=("NetPnL", "size"),
            Wins=("NetPnL", lambda x: (x > 0).sum()),
            WinRate=("NetPnL", lambda x: (x > 0).mean() * 100),
            NetPnL=("NetPnL", "sum"),
            AvgPnL=("NetPnL", "mean"),
            Costs=("EstimatedCosts", "sum"),
        )
    )

    result["ProfitFactor"] = (
        trades.groupby(group_col, observed=False)["NetPnL"]
        .apply(
            lambda x:
                x[x > 0].sum()
                / abs(x[x <= 0].sum())
                if abs(x[x <= 0].sum()) > 0
                else np.inf
        )
    )

    print(
        result.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )
    print()

show_group("BY EXTENSION SIZE", "ExtensionBucket")
show_group("BY RVOL", "RVOLBucket")
show_group("BY MONTH", "Month")
show_group("BY ENTRY TIME", "EntryTimeOnly")
show_group("BY SYMBOL", "Symbol")

print("-" * 100)
print("EXIT REASONS")

exit_summary = (
    trades.groupby("ExitReason")
    .agg(
        Trades=("NetPnL", "size"),
        Wins=("NetPnL", lambda x: (x > 0).sum()),
        WinRate=("NetPnL", lambda x: (x > 0).mean() * 100),
        NetPnL=("NetPnL", "sum"),
        AvgPnL=("NetPnL", "mean"),
    )
)

print(
    exit_summary.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)
print()

print("-" * 100)
print("WIN/LOSS SIZE")

print(
    trades.groupby(
        trades["NetPnL"] > 0
    )["NetPnL"]
    .agg(["count", "mean", "sum"])
    .rename(
        index={
            True: "Winning trades",
            False: "Losing trades",
        }
    )
    .to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)
print()

print("-" * 100)
print("TOP 10 WINNERS")

print(
    trades.nlargest(10, "NetPnL")[
        [
            "Symbol",
            "EntryTime",
            "ExitTime",
            "EntryPrice",
            "ExitPrice",
            "EntryExtensionATR",
            "EntryRVOL",
            "ExitReason",
            "NetPnL",
        ]
    ].to_string(index=False)
)
print()

print("-" * 100)
print("TOP 10 LOSERS")

print(
    trades.nsmallest(10, "NetPnL")[
        [
            "Symbol",
            "EntryTime",
            "ExitTime",
            "EntryPrice",
            "ExitPrice",
            "EntryExtensionATR",
            "EntryRVOL",
            "ExitReason",
            "NetPnL",
        ]
    ].to_string(index=False)
)
print()

print("=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)
