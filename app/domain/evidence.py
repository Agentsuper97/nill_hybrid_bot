from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceGrade(str, Enum):
    A = "A"  # Primary / directly observed
    B = "B"  # Derived measurement
    C = "C"  # Strong secondary evidence
    D = "D"  # Interpretation
    E = "E"  # Hypothesis / speculation


class EvidenceStatus(str, Enum):
    VALID = "VALID"
    DEGRADED = "DEGRADED"
    SUSPECT = "SUSPECT"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class Evidence(BaseModel):
    evidence_id: str
    subject_id: str
    metric: str
    value: Any
    unit: str | None = None

    period_end: datetime | None = None
    published_at: datetime | None = None
    available_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)

    source: str
    source_type: str
    grade: EvidenceGrade
    status: EvidenceStatus = EvidenceStatus.VALID

    calculation: str | None = None
    dependencies: list[str] = Field(default_factory=list)

    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    freshness_seconds: float | None = Field(default=None, ge=0.0)

    interpretation: str | None = None
    contradiction_ids: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)
