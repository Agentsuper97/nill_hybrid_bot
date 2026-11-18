"""
PBD PULLBACK — VERSION 0.1

Research-only setup detector.
No broker connection. No orders.

Concept:
    PUSH -> BREAK -> DRIFT/PULLBACK -> CONFIRMATION

The rules are deliberately fixed before backtesting to reduce
data-mining and hindsight bias.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PBDConfig:
    # Structure
    breakout_lookback: int = 10
    push_lookback: int = 8
    pullback_bars_min: int = 1
    pullback_bars_max: int = 4

    # Push
    minimum_push_atr: float = 1.0

    # Pullback
    minimum_pullback_fraction: float = 0.10
    maximum_pullback_fraction: float = 0.50

    # Confirmation
    confirmation_close_above_break: bool = True

    # Indicators
    atr_length: int = 14
    ema_length: int = 20
    volume_length: int = 20
    minimum_rvol: float = 1.5

    # Risk
    stop_atr_multiple: float = 1.5
    target_r_multiple: float = 2.0


def calculate_indicators(df: pd.DataFrame, config: PBDConfig) -> pd.DataFrame:
    df = df.sort_index().copy()

    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.rolling(config.atr_length).mean()
    df["ema"] = df["close"].ewm(
        span=config.ema_length,
        adjust=False,
    ).mean()

    df["average_volume"] = (
        df["volume"]
        .shift(1)
        .rolling(config.volume_length)
        .mean()
    )

    df["rvol"] = (
        df["volume"]
        / df["average_volume"].replace(0, np.nan)
    )

    # Break level is the highest high of the preceding N bars.
    df["break_level"] = (
        df["high"]
        .shift(1)
        .rolling(config.breakout_lookback)
        .max()
    )

    # Push is measured from the lowest close in the recent push window.
    df["push_start"] = (
        df["close"]
        .shift(1)
        .rolling(config.push_lookback)
        .min()
    )

    df["push_size"] = df["close"] - df["push_start"]

    df["push_strength"] = (
        df["push_size"]
        / df["atr"].replace(0, np.nan)
    )

    return df


def detect_pbd_signal(
    df: pd.DataFrame,
    index: int,
    config: PBDConfig,
):
    """
    Evaluate one completed bar.

    Returns:
        dict with setup information, or None.
    """

    if index < max(
        config.breakout_lookback,
        config.push_lookback,
        config.atr_length,
        config.volume_length,
    ):
        return None

    row = df.iloc[index]

    if not np.isfinite(row["atr"]) or row["atr"] <= 0:
        return None

    if not np.isfinite(row["break_level"]):
        return None

    if not np.isfinite(row["push_strength"]):
        return None

    # ---------------------------------------------------------
    # 1. PUSH
    # ---------------------------------------------------------
    push_ok = row["push_strength"] >= config.minimum_push_atr

    if not push_ok:
        return None

    # ---------------------------------------------------------
    # 2. BREAK
    # ---------------------------------------------------------
    previous = df.iloc[index - 1]

    break_ok = (
        row["close"] > row["break_level"]
        and previous["close"] <= previous["break_level"]
    )

    if not break_ok:
        return None

    break_price = float(row["close"])

    # ---------------------------------------------------------
    # 3. DRIFT / PULLBACK
    # ---------------------------------------------------------
    future_end = min(
        len(df),
        index + 1 + config.pullback_bars_max,
    )

    future = df.iloc[index + 1:future_end]

    if len(future) < config.pullback_bars_min:
        return None

    # Look for a controlled retracement after the breakout.
    highest_after_break = float(
        future["high"].max()
    )

    lowest_after_break = float(
        future["low"].min()
    )

    pullback_distance = (
        break_price - lowest_after_break
    )

    push_distance = max(
        break_price - float(row["push_start"]),
        row["atr"],
    )

    pullback_fraction = max(
        0.0,
        pullback_distance / push_distance,
    )

    controlled_pullback = (
        config.minimum_pullback_fraction
        <= pullback_fraction
        <= config.maximum_pullback_fraction
    )

    if not controlled_pullback:
        return None

    # ---------------------------------------------------------
    # 4. CONFIRMATION
    # ---------------------------------------------------------
    confirmation = future.iloc[-1]

    if config.confirmation_close_above_break:
        confirmation_ok = (
            confirmation["close"] > break_price
        )
    else:
        confirmation_ok = True

    if not confirmation_ok:
        return None

    if not np.isfinite(confirmation["rvol"]):
        return None

    volume_ok = (
        confirmation["rvol"] >= config.minimum_rvol
    )

    if not volume_ok:
        return None

    if confirmation["close"] <= confirmation["ema"]:
        return None

    return {
        "signal_index": index,
        "confirmation_index": int(
            df.index.get_loc(confirmation.name)
        ),
        "break_price": break_price,
        "confirmation_close": float(confirmation["close"]),
        "atr": float(confirmation["atr"]),
        "rvol": float(confirmation["rvol"]),
        "push_strength": float(row["push_strength"]),
        "pullback_fraction": float(pullback_fraction),
        "stop_distance": float(
            confirmation["atr"] * config.stop_atr_multiple
        ),
        "target_distance": float(
            confirmation["atr"]
            * config.stop_atr_multiple
            * config.target_r_multiple
        ),
    }


if __name__ == "__main__":
    print("PBD Pullback v0.1 detector loaded.")
    print("Research only — no broker connection and no orders.")
    print("Rules are fixed before backtesting.")
