from dataclasses import replace
from datetime import timedelta

import pytest

from app.data.validated.research_loader import (
    load_approved_market_data,
)
from app.evidence.availability_gate import ResearchPolicy
from app.evidence.provenance import AvailabilityBasis
from app.evidence.fingerprint import fingerprint_observation

from test_evidence_manifest import (
    CUTOFF,
    make_observation,
)


class FakeStore:
    def __init__(self, observations):
        self.observations = observations

    def load_observations_as_of(self, symbol, as_of):
        return list(self.observations)


def make_observations():
    bar, evidence = make_observation()
    observations = []

    for index in range(3):
        offset = timedelta(days=index)

        current_bar = bar.model_copy(
            update={
                "timestamp": bar.timestamp + offset,
                "available_at": bar.available_at + offset,
            }
        )

        current_evidence = replace(
            evidence,
            observation_id=index + 1,
            timestamp=current_bar.timestamp,
            available_at=current_bar.available_at,
            retrieved_at=evidence.retrieved_at + offset,
        )

        observations.append(
            (current_bar, current_evidence)
        )

    return observations


def load(observations, policy=ResearchPolicy.EXPLORATORY):
    return load_approved_market_data(
        "NVDA",
        store=FakeStore(observations),
        as_of=CUTOFF,
        policy=policy,
    )


def test_approved_research_contains_manifest():
    observations = make_observations()
    result = load(observations)

    assert result.gate.approved_for_research
    assert result.manifest is not None
    assert result.manifest.final_gate == result.gate
    assert len(result.bars) == 3
    assert len(result.manifest.observations) == 3


def test_manifest_records_exact_observation_fingerprints():
    observations = make_observations()
    result = load(observations)

    expected = tuple(
        fingerprint_observation(bar, evidence)
        for bar, evidence in observations
    )

    actual = tuple(
        entry.fingerprint
        for entry in result.manifest.observations
    )

    assert actual == expected


def test_manifest_preserves_research_context():
    result = load(make_observations())

    assert result.manifest.symbol == "NVDA"
    assert result.manifest.as_of == CUTOFF
    assert result.manifest.policy == ResearchPolicy.EXPLORATORY


def test_strict_rejection_preserves_audit_record():
    result = load(
        make_observations(),
        policy=ResearchPolicy.STRICT,
    )

    assert not result.gate.approved_for_research
    assert result.bars == ()
    assert result.evidence == ()
    assert result.manifest.quality_gate.approved_for_research
    assert not result.manifest.availability_gate.approved_for_research
    assert result.manifest.final_gate == result.gate
    assert len(result.manifest.observations) == 3


def test_manifest_preserves_availability_limitations():
    result = load(make_observations())

    assert result.manifest.availability_gate == result.availability_gate
    assert result.manifest.availability_gate.limitations


def test_empty_research_has_rejection_manifest():
    result = load([])

    assert not result.gate.approved_for_research
    assert result.bars == ()
    assert result.evidence == ()
    assert result.manifest.observations == ()
    assert result.manifest.final_gate == result.gate


def test_provenance_rejection_preserves_quality_assessment():
    observations = make_observations()
    bar, evidence = observations[0]

    observations[0] = (
        bar,
        replace(evidence, source="different_source"),
    )

    result = load(observations)

    assert not result.gate.approved_for_research
    assert result.bars == ()
    assert result.evidence == ()
    assert result.manifest.quality_gate.approved_for_research
    assert result.manifest.provenance_error is not None
    assert "source" in result.manifest.provenance_error
    assert len(result.manifest.observations) == 3


def test_duplicate_rejection_preserves_all_identities():
    observations = make_observations()
    bar, evidence = observations[1]

    observations[1] = (
        bar,
        replace(evidence, observation_id=1),
    )

    result = load(observations)

    assert not result.gate.approved_for_research
    assert result.bars == ()
    assert result.evidence == ()

    identities = tuple(
        entry.observation_id
        for entry in result.manifest.observations
    )

    assert identities == (1, 1, 3)


def test_unknown_availability_has_rejection_manifest():
    observations = make_observations()
    bar, evidence = observations[0]

    observations[0] = (
        bar.model_copy(
            update={
                "availability_basis": AvailabilityBasis.UNKNOWN,
            }
        ),
        replace(
            evidence,
            availability_basis=AvailabilityBasis.UNKNOWN,
        ),
    )

    result = load(observations)

    assert not result.gate.approved_for_research
    assert result.bars == ()
    assert result.evidence == ()
    assert not result.manifest.availability_gate.approved_for_research
    assert len(result.manifest.observations) == 3


def test_unfingerprintable_observation_fails_closed():
    observations = make_observations()
    bar, evidence = observations[0]

    observations[0] = (
        bar.model_copy(update={"volume": float("nan")}),
        evidence,
    )

    with pytest.raises(ValueError):
        load(observations)
