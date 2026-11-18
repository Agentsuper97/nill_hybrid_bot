from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class EventType(str, Enum):
    EARNINGS = "EARNINGS"
    GUIDANCE = "GUIDANCE"
    FILINGS = "FILINGS"
    MACRO = "MACRO"
    MANAGEMENT = "MANAGEMENT"
    CORPORATE_ACTION = "CORPORATE_ACTION"
    NEWS = "NEWS"
    MARKET = "MARKET"
    REGULATORY = "REGULATORY"
    OTHER = "OTHER"


class EventImpact(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class Event(BaseModel):
    event_id: str
    subject_id: str

    event_type: EventType
    occurred_at: datetime
    published_at: datetime | None = None

    title: str
    description: str | None = None

    impact: EventImpact = EventImpact.UNKNOWN
    impact_confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    evidence_ids: list[str] = Field(default_factory=list)
    related_subject_ids: list[str] = Field(default_factory=list)

    expected: str | None = None
    actual: str | None = None
    surprise: str | None = None

    materiality: float = Field(ge=0.0, le=1.0, default=0.0)
    metadata: dict[str, str] = Field(default_factory=dict)
