from dataclasses import dataclass
from datetime import datetime


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

    def __post_init__(self):
        if self.observation_id <= 0:
            raise ValueError("observation_id must be positive")

        for name in ("timestamp", "available_at", "retrieved_at"):
            value = getattr(self, name)

            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")

        if self.available_at < self.timestamp:
            raise ValueError("available_at cannot precede timestamp")
