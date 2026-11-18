from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Company(BaseModel):
    company_id: str
    ticker: str
    name: str

    exchange: str | None = None
    country: str | None = None
    currency: str | None = None
    sector: str | None = None
    industry: str | None = None

    active: bool = True
    first_seen_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated_at: datetime = Field(default_factory=datetime.utcnow)

    identifiers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
