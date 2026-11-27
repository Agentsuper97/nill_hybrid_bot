from dataclasses import replace
from datetime import timedelta, timezone

import pytest

from app.evidence.manifest import MANIFEST_SCHEMA_VERSION
from app.evidence.availability_gate import ResearchPolicy
from app.evidence.quality_gate import EvidenceDecision

from test_evidence_manifest import (
    CUTOFF,
    make_manifest,
    make_observation,
)


@pytest.mark.parametrize(
    "assessment",
    ["quality", "availability", "provenance", "empty"],
)
def test_contradictory_approval_is_rejected(assessment):
    baseline = make_manifest()

    overrides = {
        "quality": {
            "quality_gate": replace(
                baseline.quality_gate,
                decision=EvidenceDecision.BLOCK,
                approved_for_research=False,
            ),
        },
        "availability": {
            "availability_gate": replace(
                baseline.availability_gate,
                approved_for_research=False,
            ),
        },
        "provenance": {
            "provenance_error": "Provenance mismatch.",
        },
        "empty": {
            "observations": (),
        },
    }

    with pytest.raises(ValueError, match="contradicts"):
        make_manifest(**overrides[assessment])


@pytest.mark.parametrize("symbol", ["", " ", None, 123])
def test_invalid_symbol_is_rejected(symbol):
    with pytest.raises(ValueError):
        make_manifest(symbol=symbol)


@pytest.mark.parametrize("cutoff", [None, "2026-11-07", 123])
def test_invalid_cutoff_type_is_rejected(cutoff):
    with pytest.raises(TypeError):
        make_manifest(as_of=cutoff)


@pytest.mark.parametrize(
    "field",
    ["quality_gate", "availability_gate", "final_gate"],
)
def test_invalid_gate_type_is_rejected(field):
    with pytest.raises(TypeError):
        make_manifest(**{field: None})


def test_manifest_schema_version():
    manifest = make_manifest()

    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION
    assert manifest.schema_version == "research-manifest-v1"


def test_equivalent_cutoffs_produce_identical_manifests():
    alternative = CUTOFF.astimezone(
        timezone(timedelta(hours=-5))
    )

    assert make_manifest(as_of=alternative) == make_manifest()


def test_symbol_is_normalised():
    assert make_manifest(symbol="nvda").symbol == "NVDA"


def test_repeated_evaluations_are_deterministic():
    assert make_manifest() == make_manifest()


def test_observation_order_is_preserved():
    bar, evidence = make_observation()
    second = replace(evidence, observation_id=2)

    manifest = make_manifest(
        observations=((bar, second), (bar, evidence))
    )

    assert tuple(
        entry.observation_id
        for entry in manifest.observations
    ) == (2, 1)


def test_rejected_duplicate_observations_are_preserved():
    baseline = make_manifest()
    observation = make_observation()

    manifest = make_manifest(
        observations=(observation, observation),
        provenance_error="Duplicate evidence observation identity.",
        final_gate=replace(
            baseline.final_gate,
            decision=EvidenceDecision.BLOCK,
            approved_for_research=False,
        ),
    )

    assert len(manifest.observations) == 2
    assert manifest.observations[0] == manifest.observations[1]
    assert not manifest.final_gate.approved_for_research


def test_empty_rejected_manifest_is_supported():
    baseline = make_manifest()

    blocked = replace(
        baseline.final_gate,
        decision=EvidenceDecision.BLOCK,
        approved_for_research=False,
    )

    manifest = make_manifest(
        observations=(),
        quality_gate=blocked,
        availability_gate=replace(
            baseline.availability_gate,
            approved_for_research=False,
        ),
        final_gate=blocked,
    )

    assert manifest.observations == ()
    assert not manifest.final_gate.approved_for_research


@pytest.mark.parametrize(
    "invalid",
    [float("nan"), float("inf"), float("-inf")],
)
def test_nonfinite_observation_cannot_be_manifested(invalid):
    bar, evidence = make_observation()

    corrupted = bar.model_copy(
        update={"close": invalid}
    )

    with pytest.raises(ValueError):
        make_manifest(
            observations=((corrupted, evidence),)
        )


def test_invalid_research_policy_is_rejected():
    baseline = make_manifest()

    with pytest.raises(ValueError):
        make_manifest(
            availability_gate=replace(
                baseline.availability_gate,
                policy="invalid_policy",
            )
        )
