from dataclasses import dataclass, field
from enum import Enum

from app.domain.market_data import MarketBar


class QualityStatus(str, Enum):
    VALID = "valid"
    WARNING = "warning"
    SUSPECT = "suspect"
    INVALID = "invalid"


@dataclass
class QualityReport:
    symbol: str
    status: QualityStatus
    bars_checked: int
    warnings: list[str] = field(default_factory=list)


def check_market_bars(
    bars: list[MarketBar],
    large_move_threshold: float = 0.08,
) -> QualityReport:
    """Check validated market bars for basic historical anomalies."""

    if not bars:
        return QualityReport(
            symbol="UNKNOWN",
            status=QualityStatus.INVALID,
            bars_checked=0,
            warnings=["No market bars supplied"],
        )

    if not 0 < large_move_threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")

    symbol = bars[0].symbol
    warnings = []
    structural_problem = False

    previous = None
    seen_timestamps = set()

    for bar in bars:
        if bar.symbol != symbol:
            return QualityReport(
                symbol=symbol,
                status=QualityStatus.INVALID,
                bars_checked=len(bars),
                warnings=["Mixed symbols in market data"],
            )

        if bar.timestamp in seen_timestamps:
            structural_problem = True
            warnings.append(
                f"Duplicate timestamp: {bar.timestamp.isoformat()}"
            )

        seen_timestamps.add(bar.timestamp)

        if previous is not None:
            if bar.timestamp <= previous.timestamp:
                structural_problem = True
                warnings.append(
                    f"Out-of-order timestamp: {bar.timestamp.isoformat()}"
                )

            change = bar.close / previous.close - 1

            if abs(change) >= large_move_threshold:
                warnings.append(
                    f"Large price move: {bar.timestamp.date()} "
                    f"{change:+.2%}"
                )

        previous = bar

    return QualityReport(
        symbol=symbol,
        status=(
            QualityStatus.SUSPECT
            if structural_problem
            else QualityStatus.WARNING
            if warnings
            else QualityStatus.VALID
        ),
        bars_checked=len(bars),
        warnings=warnings,
    )
