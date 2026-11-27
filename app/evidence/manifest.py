"""Immutable audit records for market research evaluations."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from app.domain.market_data import MarketBar
from app.evidence.provenance import MarketEvidence
from app.evidence.availability_gate import (
    AvailabilityGateResult,
    ResearchPolicy,
)
from app.evidence.quality_gate import EvidenceGateResult
from app.evidence.fingerprint import (
    FINGERPRINT_SCHEMA_VERSION,
    fingerprint_observation,
)


MANIFEST_SCHEMA_VERSION = "research-manifest-v1"


@dataclass(frozen=True)
class ObservationFingerprint:
    """Evidence identity without raw market content."""

    observation_id: int
    fingerprint: str


@dataclass(frozen=True)
class ResearchEvidenceManifest:
    """Snapshot of evidence and research gate outcomes."""

    schema_version: str
    fingerprint_schema_version: str
    symbol: str
    as_of: datetime
    policy: ResearchPolicy
    observations: tuple[ObservationFingerprint, ...]
    quality_gate: EvidenceGateResult
    availability_gate: AvailabilityGateResult
    provenance_error: str | None
    final_gate: EvidenceGateResult


def build_research_manifest(
    *,
    symbol: str,
    as_of: datetime,
    observations: Iterable[tuple[MarketBar, MarketEvidence]],
    quality_gate: EvidenceGateResult,
    availability_gate: AvailabilityGateResult,
    provenance_error: str | None,
    final_gate: EvidenceGateResult,
) -> ResearchEvidenceManifest:
    """Record research outcomes without releasing raw market bars.

    This component records gate decisions. It does not independently
    authenticate observations or authorise their use.
    """

    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("Research symbol must not be empty.")

    if not isinstance(as_of, datetime):
        raise TypeError("Research cutoff must be a datetime.")

    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("Research cutoff must be timezone-aware.")

    if not isinstance(quality_gate, EvidenceGateResult):
        raise TypeError("Invalid quality gate result.")

    if not isinstance(availability_gate, AvailabilityGateResult):
        raise TypeError("Invalid availability gate result.")

    if not isinstance(final_gate, EvidenceGateResult):
        raise TypeError("Invalid final gate result.")

    entries = tuple(
        ObservationFingerprint(
            observation_id=evidence.observation_id,
            fingerprint=fingerprint_observation(bar, evidence),
        )
        for bar, evidence in observations
    )

    if final_gate.approved_for_research and (
        not quality_gate.approved_for_research
        or not availability_gate.approved_for_research
        or provenance_error is not None
        or not entries
    ):
        raise ValueError(
            "Research approval contradicts the recorded assessments."
        )

    return ResearchEvidenceManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fingerprint_schema_version=FINGERPRINT_SCHEMA_VERSION,
        symbol=symbol.upper(),
        as_of=as_of.astimezone(timezone.utc),
        policy=ResearchPolicy(availability_gate.policy),
        observations=entries,
        quality_gate=quality_gate,
        availability_gate=availability_gate,
        provenance_error=provenance_error,
        final_gate=final_gate,
    )
