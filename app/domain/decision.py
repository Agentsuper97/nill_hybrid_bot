from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DecisionState(str, Enum):
    DISCOVER = "DISCOVER"
    RESEARCH = "RESEARCH"
    WATCH = "WATCH"
    WAIT = "WAIT"
    TRADE_CANDIDATE = "TRADE_CANDIDATE"
    INVEST_CANDIDATE = "INVEST_CANDIDATE"
    ACTIONABLE = "ACTIONABLE"
    REJECT = "REJECT"
    NO_TRADE = "NO_TRADE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REGIME_UNCLEAR = "REGIME_UNCLEAR"
    THESIS_CONFLICT = "THESIS_CONFLICT"


class Decision(BaseModel):
    decision_id: str
    subject_id: str
    opportunity_id: str | None = None
    strategy_id: str | None = None

    created_at: datetime
    valid_until: datetime | None = None

    state: DecisionState

    conclusion: str
    rationale: str

    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    uncertainty: float = Field(ge=0.0, le=1.0, default=1.0)

    expected_upside: float | None = None
    expected_downside: float | None = None

    bull_case: str | None = None
    base_case: str | None = None
    bear_case: str | None = None

    invalidation: str | None = None
    next_research_action: str | None = None

    evidence_ids: list[str] = Field(default_factory=list)
    contradiction_ids: list[str] = Field(default_factory=list)

    portfolio_fit: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_budget: float | None = Field(default=None, ge=0.0, le=1.0)

    metadata: dict[str, str] = Field(default_factory=dict)
