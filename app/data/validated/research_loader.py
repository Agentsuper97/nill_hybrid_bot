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


def _provenance_error(
    symbol: str,
    observations: list[tuple[MarketBar, MarketEvidence]],
    as_of: datetime,
) -> str | None:
    """Validate observation identity and point-in-time consistency."""
    seen_ids: set[int] = set()

    for bar, record in observations:
        if bar.symbol.upper() != symbol.upper():
            return "Market observation does not match requested symbol."

        fields = (
            ("symbol", bar.symbol, record.symbol),
            ("timeframe", bar.timeframe.value, record.timeframe),
            ("timestamp", bar.timestamp, record.timestamp),
            ("available_at", bar.available_at, record.available_at),
            ("source", bar.source, record.source),
            ("adjusted", bar.adjusted, record.adjusted),
        )

        for field, bar_value, evidence_value in fields:
            if bar_value != evidence_value:
                return f"Bar and evidence disagree on {field}."

        if record.observation_id in seen_ids:
            return "Duplicate evidence observation identity."

        seen_ids.add(record.observation_id)

        if bar.available_at > as_of or record.available_at > as_of:
            return "Observation availability exceeds research cutoff."

        if record.retrieved_at > as_of:
            return "Evidence retrieval exceeds research cutoff."

    return None


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

    provenance_error = _provenance_error(
        symbol, observations, as_of
    )

    if provenance_error is not None:
        gate = EvidenceGateResult(
            decision=EvidenceDecision.BLOCK,
            reason=provenance_error,
            approved_for_research=False,
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
