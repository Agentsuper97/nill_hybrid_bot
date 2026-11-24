from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class AvailabilityBasis(str, Enum):
    UNKNOWN = "unknown"
    ESTIMATED_POLICY = "estimated_policy"
    PROVIDER_REPORTED = "provider_reported"
    INDEPENDENTLY_VERIFIED = "independently_verified"


@dataclass(frozen=True)
class MarketEvidence:
    """Identity and provenance of one stored market observation."""

    observation_id: int
    symbol: str
    timeframe: str
    timestamp: datetime
    available_at: datetime
    retrieved_at: datetime
    source: str
    adjusted: bool
    availability_basis: AvailabilityBasis = AvailabilityBasis.UNKNOWN

    def __post_init__(self):
        try:
            basis = AvailabilityBasis(self.availability_basis)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid availability_basis") from exc

        object.__setattr__(self, "availability_basis", basis)

        if self.observation_id <= 0:
            raise ValueError("observation_id must be positive")

        for name in ("timestamp", "available_at", "retrieved_at"):
            value = getattr(self, name)

            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")

        if self.available_at < self.timestamp:
            raise ValueError("available_at cannot precede timestamp")
