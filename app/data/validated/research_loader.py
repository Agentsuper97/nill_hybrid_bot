from dataclasses import dataclass

from app.domain.market_data import MarketBar
from app.storage.market_store import MarketStore
from app.data.validated.quality import check_market_bars
from app.evidence.quality_gate import (
    EvidenceGateResult,
    evaluate_market_data,
)


@dataclass(frozen=True)
class ResearchLoadResult:
    symbol: str
    bars: tuple[MarketBar, ...]
    gate: EvidenceGateResult


def load_approved_market_data(
    symbol: str,
    store: MarketStore | None = None,
) -> ResearchLoadResult:
    """Load market data only when it passes the evidence gate."""
    store = store if store is not None else MarketStore()

    bars = store.load_bars(symbol)
    report = check_market_bars(bars)
    gate = evaluate_market_data(report)

    if not gate.approved_for_research:
        return ResearchLoadResult(
            symbol=symbol.upper(),
            bars=(),
            gate=gate,
        )

    return ResearchLoadResult(
        symbol=symbol.upper(),
        bars=tuple(bars),
        gate=gate,
    )
