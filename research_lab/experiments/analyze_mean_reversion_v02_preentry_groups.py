from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "research_lab" / "results" / "mean_reversion_preentry_v02.csv"

df = pd.read_csv(INPUT)

print("=" * 100)
print("MEAN REVERSION v0.2 — PRE-ENTRY SUCCESS VS FAILURE")
print("=" * 100)
print("Research only. No strategy changes. No Alpaca. No orders.")
print(f"Diagnostic rows: {len(df)}")

# ---------------------------------------------------------------------
# Identify the next-bar outcome
# ---------------------------------------------------------------------
if "NextBarUp" not in df.columns:
    raise RuntimeError("Missing NextBarUp column.")

df["NextBarUp"] = df["NextBarUp"].astype(bool)

wins = df[df["NextBarUp"]].copy()
fails = df[~df["NextBarUp"]].copy()

print()
print("=" * 100)
print("NEXT-BAR GROUPS")
print("=" * 100)
print(f"Next-bar UP   : {len(wins)} trades")
print(f"Next-bar DOWN : {len(fails)} trades")

# ---------------------------------------------------------------------
# Candidate pre-entry features
# ---------------------------------------------------------------------
features = [
    "SignalRVOL",
    "SignalBodyFraction",
    "SignalBullishBodyFraction",
    "SignalCloseLocation",
    "SignalRangeATR",
    "SignalVWAPDistanceATR",
    "SignalEMADistanceATR",
    "EMASlopeATR",
    "Return1Pct",
    "Return2Pct",
    "Return3Pct",
    "Momentum3ATR",
    "SignalReclaimATR",
    "SignalLowVsExtensionATR",
    "ExtensionATR",
]

features = [c for c in features if c in df.columns]

rows = []

for col in features:
    up = pd.to_numeric(wins[col], errors="coerce").dropna()
    down = pd.to_numeric(fails[col], errors="coerce").dropna()

    if len(up) == 0 or len(down) == 0:
        continue

    up_mean = up.mean()
    down_mean = down.mean()

    pooled_std = np.sqrt(
        ((len(up) - 1) * up.var(ddof=1) +
         (len(down) - 1) * down.var(ddof=1))
        / max(len(up) + len(down) - 2, 1)
    )

    effect = (
        (up_mean - down_mean) / pooled_std
        if pooled_std > 0 else np.nan
    )

    rows.append({
        "Feature": col,
        "UpMean": up_mean,
        "DownMean": down_mean,
        "Difference": up_mean - down_mean,
        "EffectSize": effect,
    })

comparison = pd.DataFrame(rows)

if not comparison.empty:
    comparison["AbsEffect"] = comparison["EffectSize"].abs()
    comparison = comparison.sort_values("AbsEffect", ascending=False)

    print()
    print("=" * 100)
    print("PRE-ENTRY FEATURE DIFFERENCES")
    print("=" * 100)
    print(
        comparison[
            ["Feature", "UpMean", "DownMean", "Difference", "EffectSize"]
        ].to_string(index=False, float_format=lambda x: f"{x: .4f}")
    )

# ---------------------------------------------------------------------
# Simple combinations — deliberately broad, not parameter optimization
# ---------------------------------------------------------------------
print()
print("=" * 100)
print("SIMPLE COMBINATION DIAGNOSTICS")
print("=" * 100)

tests = {}

if "ExtensionATR" in df.columns:
    tests["Extreme extension >= 3 ATR"] = df["ExtensionATR"] >= 3.0

if "SignalEMADistanceATR" in df.columns:
    tests["Extreme EMA distance < -3 ATR"] = (
        df["SignalEMADistanceATR"] < -3.0
    )

if "Momentum3ATR" in df.columns:
    tests["Pre-signal momentum < -3 ATR"] = (
        df["Momentum3ATR"] < -3.0
    )

if "SignalReclaimATR" in df.columns:
    tests["Reclaim <= 0.5 ATR"] = (
        df["SignalReclaimATR"] <= 0.5
    )

if "SignalCloseLocation" in df.columns:
    tests["Close location <= 50%"] = (
        df["SignalCloseLocation"] <= 0.50
    )

if "SignalRVOL" in df.columns:
    tests["RVOL 1.5–3"] = (
        (df["SignalRVOL"] >= 1.5)
        & (df["SignalRVOL"] < 3.0)
    )

if "ExtensionATR" in df.columns and "SignalEMADistanceATR" in df.columns:
    tests["Extreme extension + extreme EMA distance"] = (
        (df["ExtensionATR"] >= 3.0)
        & (df["SignalEMADistanceATR"] < -3.0)
    )

if (
    "ExtensionATR" in df.columns
    and "Momentum3ATR" in df.columns
):
    tests["Extreme extension + strong downside momentum"] = (
        (df["ExtensionATR"] >= 3.0)
        & (df["Momentum3ATR"] < -3.0)
    )

if (
    "ExtensionATR" in df.columns
    and "SignalEMADistanceATR" in df.columns
    and "Momentum3ATR" in df.columns
):
    tests["Extreme extension + EMA distance + momentum"] = (
        (df["ExtensionATR"] >= 3.0)
        & (df["SignalEMADistanceATR"] < -3.0)
        & (df["Momentum3ATR"] < -3.0)
    )

results = []

for name, mask in tests.items():
    subset = df[mask].copy()
    n = len(subset)

    if n == 0:
        continue

    up_rate = subset["NextBarUp"].mean() * 100

    results.append({
        "Condition": name,
        "Trades": n,
        "NextBarUpPct": up_rate,
        "NextBarDownPct": 100 - up_rate,
    })

combo = pd.DataFrame(results)

if not combo.empty:
    combo = combo.sort_values(
        ["NextBarUpPct", "Trades"],
        ascending=[False, False],
    )

    print(
        combo.to_string(
            index=False,
            float_format=lambda x: f"{x: .2f}"
        )
    )

# ---------------------------------------------------------------------
# Minimum-sample warning
# ---------------------------------------------------------------------
print()
print("=" * 100)
print("RESEARCH WARNING")
print("=" * 100)
print(
    "These are diagnostics, not optimized trading rules. "
    "Small groups are especially unreliable."
)
print(
    "We are looking for broad, economically sensible differences "
    "that can survive validation/OOS testing."
)

output = ROOT / "research_lab" / "results" / "mean_reversion_preentry_groups_v02.csv"
comparison.to_csv(output, index=False)

print()
print("=" * 100)
print(f"Saved diagnostic: {output}")
print("=" * 100)
