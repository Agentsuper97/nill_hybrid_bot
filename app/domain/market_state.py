from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class MarketState(BaseModel):
    state_id: str
    market_id: str
    timestamp: datetime

    regime: MarketRegime = MarketRegime.UNKNOWN
    regime_confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    trend_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    volatility_score: float | None = Field(default=None, ge=0.0, le=1.0)
    breadth_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    liquidity_score: float | None = Field(default=None, ge=0.0, le=1.0)

    leadership: list[str] = Field(default_factory=list)
    weakening_groups: list[str] = Field(default_factory=list)

    change_detected: bool = False
    change_reason: str | None = None

    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
