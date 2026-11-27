from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.market_data import MarketBar
from app.evidence.provenance import (
    AvailabilityBasis,
    MarketEvidence,
)
from app.evidence.availability_gate import (
    ResearchPolicy,
    evaluate_availability,
)
from app.evidence.quality_gate import (
    EvidenceDecision,
    EvidenceGateResult,
)
from app.evidence.fingerprint import (
    FINGERPRINT_SCHEMA_VERSION,
    fingerprint_observation,
)
from app.evidence.manifest import build_research_manifest


START = datetime(2026, 11, 2, tzinfo=timezone.utc)
CUTOFF = START + timedelta(days=5)


def make_observation():
    bar = MarketBar(
        symbol="NVDA",
        timeframe="1d",
        timestamp=START,
        available_at=START + timedelta(days=1),
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1_000_000,
        source="synthetic_manifest_test",
        adjusted=False,
        availability_basis=AvailabilityBasis.ESTIMATED_POLICY,
    )

    evidence = MarketEvidence(
        observation_id=1,
        symbol=bar.symbol,
        timeframe=bar.timeframe.value,
        timestamp=bar.timestamp,
        available_at=bar.available_at,
        retrieved_at=START + timedelta(days=2),
        source=bar.source,
        adjusted=bar.adjusted,
        availability_basis=bar.availability_basis,
    )

    return bar, evidence


def make_manifest(**overrides):
    observation = make_observation()

    quality_gate = EvidenceGateResult(
        decision=EvidenceDecision.ALLOW,
        reason="Quality checks passed.",
        approved_for_research=True,
    )

    availability_gate = evaluate_availability(
        (AvailabilityBasis.ESTIMATED_POLICY,),
        policy=ResearchPolicy.EXPLORATORY,
    )

    arguments = {
        "symbol": "NVDA",
        "as_of": CUTOFF,
        "observations": (observation,),
        "quality_gate": quality_gate,
        "availability_gate": availability_gate,
        "provenance_error": None,
        "final_gate": quality_gate,
    }

    arguments.update(overrides)

    return build_research_manifest(**arguments)


def test_manifest_records_research_context():
    manifest = make_manifest()

    assert manifest.symbol == "NVDA"
    assert manifest.as_of == CUTOFF
    assert manifest.policy == ResearchPolicy.EXPLORATORY


def test_manifest_records_observation_fingerprint():
    bar, evidence = make_observation()
    manifest = make_manifest()

    assert len(manifest.observations) == 1

    entry = manifest.observations[0]

    assert entry.observation_id == evidence.observation_id
    assert entry.fingerprint == fingerprint_observation(
        bar, evidence
    )
    assert (
        manifest.fingerprint_schema_version
        == FINGERPRINT_SCHEMA_VERSION
    )


def test_manifest_records_gate_results_and_limitations():
    manifest = make_manifest()

    assert manifest.quality_gate.approved_for_research
    assert manifest.availability_gate.approved_for_research
    assert manifest.availability_gate.limitations
    assert manifest.final_gate.approved_for_research
    assert manifest.provenance_error is None


def test_rejected_manifest_preserves_original_quality_result():
    availability = evaluate_availability(
        (AvailabilityBasis.ESTIMATED_POLICY,),
        policy=ResearchPolicy.STRICT,
    )

    final_gate = EvidenceGateResult(
        decision=EvidenceDecision.BLOCK,
        reason=availability.reason,
        approved_for_research=False,
    )

    manifest = make_manifest(
        availability_gate=availability,
        final_gate=final_gate,
    )

    assert manifest.quality_gate.approved_for_research
    assert not manifest.availability_gate.approved_for_research
    assert not manifest.final_gate.approved_for_research
    assert manifest.policy == ResearchPolicy.STRICT
    assert len(manifest.observations) == 1


def test_manifest_records_provenance_rejection():
    reason = "Bar and evidence disagree on source."

    final_gate = EvidenceGateResult(
        decision=EvidenceDecision.BLOCK,
        reason=reason,
        approved_for_research=False,
    )

    bar, evidence = make_observation()
    evidence = replace(evidence, source="different_source")

    manifest = make_manifest(
        observations=((bar, evidence),),
        provenance_error=reason,
        final_gate=final_gate,
    )

    assert manifest.provenance_error == reason
    assert manifest.final_gate.reason == reason
    assert not manifest.final_gate.approved_for_research
    assert len(manifest.observations) == 1


def test_manifest_does_not_expose_raw_market_bars():
    manifest = make_manifest()
    entry = manifest.observations[0]

    assert not hasattr(manifest, "bars")
    assert not hasattr(entry, "bar")
    assert not hasattr(entry, "close")
    assert not hasattr(entry, "volume")


def test_manifest_is_immutable():
    manifest = make_manifest()

    with pytest.raises(FrozenInstanceError):
        manifest.symbol = "AMD"

    with pytest.raises(FrozenInstanceError):
        manifest.observations[0].fingerprint = "changed"


def test_manifest_rejects_naive_research_cutoff():
    with pytest.raises(ValueError):
        make_manifest(
            as_of=CUTOFF.replace(tzinfo=None)
        )
