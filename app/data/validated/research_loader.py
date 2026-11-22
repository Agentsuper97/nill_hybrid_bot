from dataclasses import dataclass
from datetime import datetime, timezone

from app.domain.market_data import MarketBar
from app.evidence.provenance import MarketEvidence
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
    evidence: tuple[MarketEvidence, ...] = ()


def load_approved_market_data(
    symbol: str,
    store: MarketStore | None = None,
    as_of: datetime | None = None,
) -> ResearchLoadResult:
    """Load market data only when it passes the evidence gate."""
    store = store if store is not None else MarketStore()

    if as_of is None:
        as_of = datetime.now(timezone.utc)
    elif as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")

    observations = store.load_observations_as_of(symbol, as_of)
    bars = [bar for bar, _ in observations]
    evidence = tuple(record for _, record in observations)

    report = check_market_bars(bars)
    gate = evaluate_market_data(report)

    if not gate.approved_for_research:
        return ResearchLoadResult(
            symbol=symbol.upper(),
            bars=(),
            gate=gate,
            evidence=(),
        )

    return ResearchLoadResult(
        symbol=symbol.upper(),
        bars=tuple(bars),
        gate=gate,
        evidence=evidence,
    )
