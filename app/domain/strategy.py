from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class StrategyType(str, Enum):
    TREND = "TREND"
    MOMENTUM = "MOMENTUM"
    BREAKOUT = "BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"
    COUNTER_TREND = "COUNTER_TREND"
    VALUE = "VALUE"
    GROWTH = "GROWTH"
    EVENT = "EVENT"
    MULTI_FACTOR = "MULTI_FACTOR"
    CUSTOM = "CUSTOM"


class StrategyStatus(str, Enum):
    RESEARCH = "RESEARCH"
    TESTING = "TESTING"
    VALIDATING = "VALIDATING"
    MONITORING = "MONITORING"
    PROMOTED = "PROMOTED"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


class Strategy(BaseModel):
    strategy_id: str
    name: str
    version: str

    strategy_type: StrategyType
    status: StrategyStatus = StrategyStatus.RESEARCH

    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    rules: dict[str, str] = Field(default_factory=dict)
    required_features: list[str] = Field(default_factory=list)
    preferred_regimes: list[str] = Field(default_factory=list)
    excluded_regimes: list[str] = Field(default_factory=list)

    validation_sample_size: int = Field(default=0, ge=0)
    validation_score: float | None = Field(default=None, ge=0.0)
    max_drawdown: float | None = Field(default=None, le=0.0)

    parent_strategy_id: str | None = None
    frozen: bool = False

    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
