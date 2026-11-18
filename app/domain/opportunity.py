from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class OpportunityState(str, Enum):
    DISCOVERED = "DISCOVERED"
    RESEARCHING = "RESEARCHING"
    WATCH = "WATCH"
    WAIT = "WAIT"
    INVEST_CANDIDATE = "INVEST_CANDIDATE"
    TRADE_CANDIDATE = "TRADE_CANDIDATE"
    ACTIONABLE = "ACTIONABLE"
    REJECTED = "REJECTED"
    NO_TRADE = "NO_TRADE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REGIME_UNCLEAR = "REGIME_UNCLEAR"
    THESIS_CONFLICT = "THESIS_CONFLICT"


class Opportunity(BaseModel):
    opportunity_id: str
    subject_id: str

    discovered_at: datetime
    updated_at: datetime

    state: OpportunityState = OpportunityState.DISCOVERED

    thesis: str | None = None
    why_now: str | None = None
    invalidation: str | None = None

    business_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    financial_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    growth_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    moat_score: float | None = Field(default=None, ge=0.0, le=1.0)
    valuation_score: float | None = Field(default=None, ge=0.0, le=1.0)

    market_regime_fit: float | None = Field(default=None, ge=0.0, le=1.0)
    structure_score: float | None = Field(default=None, ge=0.0, le=1.0)
    capital_flow_score: float | None = Field(default=None, ge=0.0, le=1.0)
    catalyst_score: float | None = Field(default=None, ge=0.0, le=1.0)
    strategy_fit: float | None = Field(default=None, ge=0.0, le=1.0)
    portfolio_fit: float | None = Field(default=None, ge=0.0, le=1.0)

    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)

    evidence_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)

    competing_explanations: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)

    metadata: dict[str, str] = Field(default_factory=dict)
