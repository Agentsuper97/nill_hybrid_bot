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


def test_historical_loader_excludes_future_bars():
    from datetime import datetime, timezone

    class HistoricalStore:
        def load_observations_as_of(self, symbol, as_of):
            from app.evidence.provenance import MarketEvidence

            assert symbol == "NVDA"
            assert as_of == datetime(
                2026, 11, 5, 22, 0, tzinfo=timezone.utc
            )

            observations = []

            for index, bar in enumerate(make_bars([100, 101, 102]), start=1):
                evidence = MarketEvidence(
                    observation_id=index,
                    symbol=bar.symbol,
                    timeframe=bar.timeframe.value,
                    timestamp=bar.timestamp,
                    available_at=bar.available_at,
                    retrieved_at=bar.available_at,
                    source=bar.source,
                    adjusted=bar.adjusted,
                )
                observations.append((bar, evidence))

            return observations

        def load_bars(self, symbol):
            raise AssertionError(
                "Historical research must not use unrestricted loading"
            )

    result = load_approved_market_data(
        "NVDA",
        store=HistoricalStore(),
        as_of=datetime(2026, 11, 5, 22, 0, tzinfo=timezone.utc),
    )

    assert result.gate.decision == EvidenceDecision.ALLOW
    assert len(result.bars) == 3


def test_historical_loader_rejects_naive_datetime():
    import pytest

    store = FakeMarketStore(make_bars([100, 101]))

    with pytest.raises(ValueError, match="timezone-aware"):
        load_approved_market_data(
            "NVDA",
            store=store,
            as_of=datetime(2026, 11, 5, 22, 0),
        )


def test_historical_loader_releases_paired_evidence(tmp_path):
    from app.storage.market_store import MarketStore

    store = MarketStore(tmp_path / "approved_evidence.sqlite")
    bars = make_bars([100, 101, 102])

    assert store.save_bars(bars) == 3

    cutoff = datetime.now(timezone.utc) + timedelta(minutes=1)

    result = load_approved_market_data(
        "NVDA",
        store=store,
        as_of=cutoff,
    )

    assert result.gate.decision == EvidenceDecision.ALLOW
    assert len(result.bars) == 3
    assert len(result.evidence) == 3

    for bar, evidence in zip(result.bars, result.evidence):
        assert evidence.observation_id > 0
        assert evidence.symbol == bar.symbol
        assert evidence.timestamp == bar.timestamp
        assert evidence.available_at == bar.available_at
        assert evidence.source == bar.source
        assert evidence.retrieved_at <= cutoff


def test_historical_loader_withholds_rejected_evidence(tmp_path):
    from app.storage.market_store import MarketStore

    store = MarketStore(tmp_path / "rejected_evidence.sqlite")

    # The final candle falls sharply enough to trigger REVIEW.
    bars = make_bars([100, 101, 85])

    assert store.save_bars(bars) == 3

    cutoff = datetime.now(timezone.utc) + timedelta(minutes=1)

    result = load_approved_market_data(
        "NVDA",
        store=store,
        as_of=cutoff,
    )

    assert result.gate.decision == EvidenceDecision.REVIEW
    assert result.gate.approved_for_research is False
    assert result.bars == ()
    assert result.evidence == ()
