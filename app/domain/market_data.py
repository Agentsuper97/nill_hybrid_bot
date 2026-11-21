from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1wk"


class MarketBar(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    symbol: str = Field(min_length=1)
    timeframe: Timeframe

    timestamp: datetime
    available_at: datetime

    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    source: str = Field(min_length=1)
    adjusted: bool = False

    @model_validator(mode="after")
    def validate_bar(self):
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("High is inconsistent with OHLC prices")

        if self.low > min(self.open, self.close, self.high):
            raise ValueError("Low is inconsistent with OHLC prices")

        if self.timestamp.tzinfo is None:
            raise ValueError("Timestamp must include timezone")

        if self.available_at.tzinfo is None:
            raise ValueError("Availability time must include timezone")

        if self.available_at < self.timestamp:
            raise ValueError("Data cannot be available before its timestamp")

        duration = {
            Timeframe.M1: timedelta(minutes=1),
            Timeframe.M5: timedelta(minutes=5),
            Timeframe.M15: timedelta(minutes=15),
            Timeframe.M30: timedelta(minutes=30),
            Timeframe.H1: timedelta(hours=1),
            Timeframe.H4: timedelta(hours=4),
        }.get(self.timeframe)

        if duration and self.available_at < self.timestamp + duration:
            raise ValueError("Bar cannot be available before it closes")

        return self
