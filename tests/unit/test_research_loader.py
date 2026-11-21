from datetime import datetime, timedelta, timezone

from app.data.validated.research_loader import load_approved_market_data
from app.domain.market_data import MarketBar
from app.evidence.quality_gate import EvidenceDecision


class FakeMarketStore:
    def __init__(self, bars):
        self.bars = bars

    def load_bars(self, symbol):
        return self.bars


def make_bars(closes):
    start = datetime(2026, 11, 2, 21, 0, tzinfo=timezone.utc)
    bars = []

    for index, close in enumerate(closes):
        timestamp = start + timedelta(days=index)

        bars.append(
            MarketBar(
                symbol="NVDA",
                timeframe="1d",
                timestamp=timestamp,
                available_at=timestamp,
                open=close,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=1000000,
                source="test",
                adjusted=False,
            )
        )

    return bars


def test_valid_market_data_is_released():
    store = FakeMarketStore(make_bars([100, 101, 102]))

    result = load_approved_market_data("NVDA", store=store)

    assert result.gate.decision == EvidenceDecision.ALLOW
    assert result.gate.approved_for_research is True
    assert len(result.bars) == 3


def test_warning_market_data_is_withheld():
    store = FakeMarketStore(make_bars([100, 101, 85]))

    result = load_approved_market_data("NVDA", store=store)

    assert result.gate.decision == EvidenceDecision.REVIEW
    assert result.gate.approved_for_research is False
    assert result.bars == ()


def test_missing_market_data_is_withheld():
    store = FakeMarketStore([])

    result = load_approved_market_data("NVDA", store=store)

    assert result.gate.decision == EvidenceDecision.BLOCK
    assert result.gate.approved_for_research is False
    assert result.bars == ()
