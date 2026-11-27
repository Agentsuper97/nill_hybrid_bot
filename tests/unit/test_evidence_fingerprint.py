from dataclasses import replace
from datetime import datetime, timedelta, timezone
import re

import pytest

from app.domain.market_data import MarketBar
from app.evidence.provenance import (
    AvailabilityBasis,
    MarketEvidence,
)
from app.evidence.fingerprint import fingerprint_observation


START = datetime(2026, 11, 2, tzinfo=timezone.utc)


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
        source="synthetic_fingerprint_test",
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


def test_fingerprint_has_sha256_format():
    bar, evidence = make_observation()
    result = fingerprint_observation(bar, evidence)

    assert re.fullmatch(r"[0-9a-f]{64}", result)


def test_identical_observations_have_identical_fingerprints():
    first_bar, first_evidence = make_observation()
    second_bar, second_evidence = make_observation()

    assert fingerprint_observation(
        first_bar, first_evidence
    ) == fingerprint_observation(
        second_bar, second_evidence
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("open", 101.0),
        ("high", 106.0),
        ("low", 94.0),
        ("close", 103.0),
        ("volume", 1_000_001),
    ],
)
def test_changed_market_content_changes_fingerprint(field, value):
    bar, evidence = make_observation()

    updated = MarketBar.model_validate(
        {**bar.model_dump(), field: value}
    )

    assert fingerprint_observation(
        bar, evidence
    ) != fingerprint_observation(
        updated, evidence
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("observation_id", 2),
        ("retrieved_at", START + timedelta(days=3)),
    ],
)
def test_changed_evidence_identity_changes_fingerprint(field, value):
    bar, evidence = make_observation()
    updated = replace(evidence, **{field: value})

    assert fingerprint_observation(
        bar, evidence
    ) != fingerprint_observation(
        bar, updated
    )


def test_equivalent_timezones_have_identical_fingerprints():
    bar, evidence = make_observation()
    offset = timezone(timedelta(hours=-5))

    updated_bar = MarketBar.model_validate({
        **bar.model_dump(),
        "timestamp": bar.timestamp.astimezone(offset),
        "available_at": bar.available_at.astimezone(offset),
    })

    updated_evidence = replace(
        evidence,
        timestamp=evidence.timestamp.astimezone(offset),
        available_at=evidence.available_at.astimezone(offset),
        retrieved_at=evidence.retrieved_at.astimezone(offset),
    )

    assert fingerprint_observation(
        bar, evidence
    ) == fingerprint_observation(
        updated_bar, updated_evidence
    )
