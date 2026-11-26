from datetime import datetime, timedelta, timezone

from app.domain.market_data import MarketBar
from app.storage.market_store import MarketStore
from app.evidence.provenance import AvailabilityBasis
from app.evidence.availability_gate import (
    ResearchPolicy,
    evaluate_availability,
)
from app.evidence.quality_gate import EvidenceDecision
from app.data.validated.research_loader import (
    load_approved_market_data,
)


CLAIMED_VERIFIED = AvailabilityBasis.INDEPENDENTLY_VERIFIED


def test_strict_rejects_single_unproven_verification_claim():
    result = evaluate_availability(
        (CLAIMED_VERIFIED,),
        policy=ResearchPolicy.STRICT,
    )

    assert result.approved_for_research is False
    assert "authentication" in result.reason.lower()


def test_strict_rejects_multiple_unproven_verification_claims():
    result = evaluate_availability(
        (CLAIMED_VERIFIED,) * 3,
        policy=ResearchPolicy.STRICT,
    )

    assert result.approved_for_research is False
    assert "authentication" in result.reason.lower()


def test_exploratory_reports_unproven_verification_claim():
    result = evaluate_availability(
        (CLAIMED_VERIFIED,),
        policy=ResearchPolicy.EXPLORATORY,
    )

    assert result.approved_for_research is True
    assert any(
        "authentication" in limitation.lower()
        for limitation in result.limitations
    )


def test_mixed_exploratory_evidence_reports_authentication_limit():
    result = evaluate_availability(
        (
            CLAIMED_VERIFIED,
            AvailabilityBasis.ESTIMATED_POLICY,
            AvailabilityBasis.PROVIDER_REPORTED,
        ),
        policy=ResearchPolicy.EXPLORATORY,
    )

    assert result.approved_for_research is True
    assert any(
        "authentication" in limitation.lower()
        for limitation in result.limitations
    )
    assert any(
        "estimated" in limitation.lower()
        for limitation in result.limitations
    )
    assert any(
        "provider-reported" in limitation.lower()
        for limitation in result.limitations
    )


def make_claimed_verified_store(tmp_path):
    store = MarketStore(
        tmp_path / "authentication_test.sqlite"
    )

    start = datetime.now(timezone.utc) - timedelta(days=10)
    bars = []

    for index in range(3):
        timestamp = start + timedelta(days=index)

        bars.append(
            MarketBar(
                symbol="NVDA",
                timeframe="1d",
                timestamp=timestamp,
                available_at=timestamp + timedelta(days=1),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1_000_000,
                source="synthetic_authentication_test",
                adjusted=False,
                availability_basis=CLAIMED_VERIFIED,
            )
        )

    assert store.save_bars(bars) == len(bars)

    return store


def test_strict_loader_withholds_unauthenticated_evidence(tmp_path):
    store = make_claimed_verified_store(tmp_path)

    result = load_approved_market_data(
        "NVDA",
        store=store,
        policy=ResearchPolicy.STRICT,
    )

    assert result.gate.decision == EvidenceDecision.BLOCK
    assert result.gate.approved_for_research is False
    assert result.availability_gate.approved_for_research is False
    assert "authentication" in result.gate.reason.lower()
    assert result.bars == ()
    assert result.evidence == ()


def test_exploratory_loader_discloses_missing_authentication(tmp_path):
    store = make_claimed_verified_store(tmp_path)

    result = load_approved_market_data(
        "NVDA",
        store=store,
        policy=ResearchPolicy.EXPLORATORY,
    )

    assert result.gate.approved_for_research is True
    assert len(result.bars) == 3
    assert len(result.evidence) == 3
    assert any(
        "authentication" in limitation.lower()
        for limitation in result.availability_gate.limitations
    )
