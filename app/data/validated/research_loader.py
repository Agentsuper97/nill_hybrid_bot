from dataclasses import dataclass
from datetime import datetime, timezone

from app.domain.market_data import MarketBar
from app.evidence.provenance import MarketEvidence
from app.storage.market_store import MarketStore
from app.data.validated.quality import check_market_bars
from app.evidence.availability_gate import (
    AvailabilityGateResult,
    ResearchPolicy,
    evaluate_availability,
)
from app.evidence.quality_gate import (
    EvidenceDecision,
    EvidenceGateResult,
    evaluate_market_data,
)


@dataclass(frozen=True)
class ResearchLoadResult:
    symbol: str
    bars: tuple[MarketBar, ...]
    gate: EvidenceGateResult
    evidence: tuple[MarketEvidence, ...] = ()
    availability_gate: AvailabilityGateResult | None = None


def load_approved_market_data(
    symbol: str,
    store: MarketStore | None = None,
    as_of: datetime | None = None,
    *,
    policy: ResearchPolicy = ResearchPolicy.EXPLORATORY,
) -> ResearchLoadResult:
    """Release data only when quality and availability gates approve."""
    policy = ResearchPolicy(policy)
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

    availability_gate = evaluate_availability(
        tuple(record.availability_basis for record in evidence),
        policy=policy,
    )

    if any(
        bar.availability_basis != record.availability_basis
        for bar, record in observations
    ):
        availability_gate = AvailabilityGateResult(
            policy=policy,
            approved_for_research=False,
            reason="Bar and evidence availability classifications disagree.",
        )

    if gate.approved_for_research and not availability_gate.approved_for_research:
        gate = EvidenceGateResult(
            decision=EvidenceDecision.BLOCK,
            reason=availability_gate.reason,
            approved_for_research=False,
        )

    if not gate.approved_for_research:
        return ResearchLoadResult(
            symbol=symbol.upper(),
            bars=(),
            gate=gate,
            evidence=(),
            availability_gate=availability_gate,
        )

    return ResearchLoadResult(
        symbol=symbol.upper(),
        bars=tuple(bars),
        gate=gate,
        evidence=evidence,
        availability_gate=availability_gate,
    )
