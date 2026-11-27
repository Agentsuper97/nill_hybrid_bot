"""Deterministic fingerprints for market observations."""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum

from app.domain.market_data import MarketBar
from app.evidence.provenance import MarketEvidence


FINGERPRINT_SCHEMA_VERSION = "market-observation-v1"


def _normalise(value):
    """Convert supported values into canonical JSON data."""

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "Fingerprint timestamps must be timezone-aware."
            )

        return value.astimezone(timezone.utc).isoformat(
            timespec="microseconds"
        )

    if isinstance(value, Enum):
        return _normalise(value.value)

    if isinstance(value, dict):
        return {
            key: _normalise(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]

    if isinstance(value, float) and value == 0.0:
        return 0.0

    return value


def fingerprint_observation(
    bar: MarketBar,
    evidence: MarketEvidence,
) -> str:
    """Hash market content and provenance using SHA-256.

    A fingerprint identifies content. It does not establish
    authenticity, accuracy or historical availability.
    """

    if not isinstance(bar, MarketBar):
        raise TypeError("bar must be a MarketBar")

    if not isinstance(evidence, MarketEvidence):
        raise TypeError("evidence must be MarketEvidence")

    payload = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "market_bar": bar.model_dump(mode="python"),
        "evidence": asdict(evidence),
    }

    canonical = json.dumps(
        _normalise(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
