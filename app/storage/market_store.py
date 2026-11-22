import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.domain.market_data import MarketBar


class MarketStore:
    def __init__(self, path="research_data/market.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _initialize(self):
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS market_bars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    source TEXT NOT NULL,
                    adjusted INTEGER NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    UNIQUE (
                        symbol, timeframe, timestamp,
                        source, adjusted, retrieved_at
                    )
                )
            """)

    def save_bars(self, bars: list[MarketBar]) -> int:
        if not bars:
            return 0

        retrieved_at = datetime.now(timezone.utc).isoformat()

        rows = [
            (
                bar.symbol,
                bar.timeframe.value,
                bar.timestamp.astimezone(timezone.utc).isoformat(),
                bar.available_at.astimezone(timezone.utc).isoformat(),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.source,
                int(bar.adjusted),
                retrieved_at,
            )
            for bar in bars
        ]

        with self._connect() as connection:
            connection.executemany("""
                INSERT OR IGNORE INTO market_bars (
                    symbol, timeframe, timestamp, available_at,
                    open, high, low, close, volume,
                    source, adjusted, retrieved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)

            return connection.total_changes

    def count_bars(self, symbol=None) -> int:
        with self._connect() as connection:
            if symbol is None:
                result = connection.execute(
                    "SELECT COUNT(*) FROM market_bars"
                ).fetchone()
            else:
                result = connection.execute(
                    "SELECT COUNT(*) FROM market_bars WHERE symbol = ?",
                    (symbol.upper(),),
                ).fetchone()

        return result[0]

    def load_bars(self, symbol: str, timeframe: str = "1d") -> list[MarketBar]:
        """Load the latest stored observation for each market candle."""
        symbol = symbol.strip().upper()

        if not symbol:
            raise ValueError("Symbol cannot be empty")

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute("""
                SELECT *
                FROM market_bars
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp ASC, retrieved_at DESC, id DESC
            """, (symbol, timeframe)).fetchall()

        bars = []
        seen = set()

        for row in rows:
            timestamp = row["timestamp"]

            if timestamp in seen:
                continue

            seen.add(timestamp)

            bars.append(MarketBar(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                available_at=datetime.fromisoformat(row["available_at"]),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                source=row["source"],
                adjusted=bool(row["adjusted"]),
            ))

        return bars

    def load_bars_as_of(
        self,
        symbol: str,
        as_of: datetime,
        timeframe: str = "1d",
    ) -> list[MarketBar]:
        """Return observations recorded and available by the cutoff."""
        symbol = symbol.strip().upper()

        if not symbol:
            raise ValueError("Symbol cannot be empty")

        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")

        cutoff = as_of.astimezone(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute("""
                SELECT *
                FROM market_bars
                WHERE symbol = ?
                  AND timeframe = ?
                  AND retrieved_at <= ?
                  AND available_at <= ?
                ORDER BY timestamp ASC, retrieved_at DESC, id DESC
            """, (symbol, timeframe, cutoff, cutoff)).fetchall()

        bars = []
        seen = set()

        for row in rows:
            timestamp = row["timestamp"]

            if timestamp in seen:
                continue

            seen.add(timestamp)

            bars.append(MarketBar(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                available_at=datetime.fromisoformat(row["available_at"]),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                source=row["source"],
                adjusted=bool(row["adjusted"]),
            ))

        return bars

    def load_evidence_as_of(
        self,
        symbol: str,
        as_of: datetime,
        timeframe: str = "1d",
    ):
        """Return provenance for the latest eligible version of each candle."""
        from app.evidence.provenance import MarketEvidence

        symbol = symbol.strip().upper()

        if not symbol:
            raise ValueError("Symbol cannot be empty")

        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")

        cutoff = as_of.astimezone(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("""
                SELECT *
                FROM market_bars
                WHERE symbol = ?
                  AND timeframe = ?
                  AND retrieved_at <= ?
                  AND available_at <= ?
                ORDER BY timestamp ASC, retrieved_at DESC, id DESC
            """, (symbol, timeframe, cutoff, cutoff)).fetchall()

        evidence = []
        seen = set()

        for row in rows:
            timestamp = row["timestamp"]

            if timestamp in seen:
                continue

            seen.add(timestamp)

            evidence.append(MarketEvidence(
                observation_id=row["id"],
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                available_at=datetime.fromisoformat(row["available_at"]),
                retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
                source=row["source"],
                adjusted=bool(row["adjusted"]),
            ))

        return evidence
