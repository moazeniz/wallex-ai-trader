"""
Episodic Memory

Long-term memory for storing and retrieving past trading experiences:
- Signal outcomes
- Successful patterns
- Market regime transitions
- Learning from past trades
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import sqlite3
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.vector_store import VectorStore


@dataclass
class Episode:
    """A trading episode (signal + context + outcome)."""
    id: Optional[int]
    symbol: str
    signal_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    technical_summary: str
    onchain_summary: str
    regime: str
    timestamp: datetime
    outcome: Optional[str] = None  # win/loss/pending
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    closed_at: Optional[datetime] = None
    lessons: Optional[str] = None


class EpisodicMemory:
    """
    Long-term episodic memory system.

    Stores complete trading episodes including:
    - Signal context
    - Analysis summaries
    - Outcomes
    - Lessons learned
    """

    def __init__(
        self,
        db_path: str = "data/episodic_memory.db",
        vector_store: Optional[VectorStore] = None,
    ):
        """
        Initialize episodic memory.

        Args:
            db_path: Path to SQLite database
            vector_store: Vector store for semantic search
        """
        self.db_path = db_path
        self.vector_store = vector_store or VectorStore()

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                confidence REAL NOT NULL,
                technical_summary TEXT,
                onchain_summary TEXT,
                regime TEXT,
                timestamp TEXT NOT NULL,
                outcome TEXT,
                exit_price REAL,
                pnl_pct REAL,
                closed_at TEXT,
                lessons TEXT,
                vector_id TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                from_regime TEXT NOT NULL,
                to_regime TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price_at_transition REAL,
                price_after_24h REAL,
                price_change_pct REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_occurrences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                signal_type TEXT,
                confidence REAL,
                outcome TEXT,
                pnl_pct REAL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_symbol ON episodes(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON episodes(outcome)
        """)

        conn.commit()
        conn.close()

    def store_episode(self, episode: Episode) -> int:
        """
        Store a new episode.

        Args:
            episode: Episode to store

        Returns:
            Episode ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create combined text for vector storage
        combined_text = f"""
        Symbol: {episode.symbol}
        Signal: {episode.signal_type} (confidence: {episode.confidence:.0%})
        Regime: {episode.regime}
        Technical: {episode.technical_summary}
        On-Chain: {episode.onchain_summary}
        """

        # Store in vector store
        vector_id = self.vector_store.add_signal(
            symbol=episode.symbol,
            signal_type=episode.signal_type,
            confidence=episode.confidence,
            reasoning=combined_text,
        )

        cursor.execute("""
            INSERT INTO episodes (
                symbol, signal_type, entry_price, stop_loss, take_profit,
                confidence, technical_summary, onchain_summary, regime,
                timestamp, outcome, exit_price, pnl_pct, closed_at, lessons, vector_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode.symbol,
            episode.signal_type,
            episode.entry_price,
            episode.stop_loss,
            episode.take_profit,
            episode.confidence,
            episode.technical_summary,
            episode.onchain_summary,
            episode.regime,
            episode.timestamp.isoformat(),
            episode.outcome,
            episode.exit_price,
            episode.pnl_pct,
            episode.closed_at.isoformat() if episode.closed_at else None,
            episode.lessons,
            vector_id,
        ))

        episode_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return episode_id

    def update_outcome(
        self,
        episode_id: int,
        outcome: str,
        exit_price: float,
        pnl_pct: float,
        lessons: Optional[str] = None,
    ) -> bool:
        """
        Update an episode's outcome.

        Args:
            episode_id: Episode ID
            outcome: win/loss
            exit_price: Exit price
            pnl_pct: Profit/loss percentage
            lessons: Lessons learned

        Returns:
            Success status
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE episodes
            SET outcome = ?, exit_price = ?, pnl_pct = ?,
                closed_at = ?, lessons = ?
            WHERE id = ?
        """, (
            outcome,
            exit_price,
            pnl_pct,
            datetime.now().isoformat(),
            lessons,
            episode_id,
        ))

        # Update vector store
        cursor.execute("SELECT vector_id FROM episodes WHERE id = ?", (episode_id,))
        row = cursor.fetchone()
        if row and row[0]:
            self.vector_store.update_signal_outcome(row[0], outcome, pnl_pct)

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    def get_episode(self, episode_id: int) -> Optional[Episode]:
        """Get an episode by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return Episode(
            id=row[0],
            symbol=row[1],
            signal_type=row[2],
            entry_price=row[3],
            stop_loss=row[4],
            take_profit=row[5],
            confidence=row[6],
            technical_summary=row[7],
            onchain_summary=row[8],
            regime=row[9],
            timestamp=datetime.fromisoformat(row[10]),
            outcome=row[11],
            exit_price=row[12],
            pnl_pct=row[13],
            closed_at=datetime.fromisoformat(row[14]) if row[14] else None,
            lessons=row[15],
        )

    def get_recent_episodes(
        self,
        symbol: Optional[str] = None,
        limit: int = 20,
    ) -> List[Episode]:
        """Get recent episodes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if symbol:
            cursor.execute("""
                SELECT * FROM episodes WHERE symbol = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (symbol, limit))
        else:
            cursor.execute("""
                SELECT * FROM episodes
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        episodes = []
        for row in rows:
            episodes.append(Episode(
                id=row[0],
                symbol=row[1],
                signal_type=row[2],
                entry_price=row[3],
                stop_loss=row[4],
                take_profit=row[5],
                confidence=row[6],
                technical_summary=row[7],
                onchain_summary=row[8],
                regime=row[9],
                timestamp=datetime.fromisoformat(row[10]),
                outcome=row[11],
                exit_price=row[12],
                pnl_pct=row[13],
                closed_at=datetime.fromisoformat(row[14]) if row[14] else None,
                lessons=row[15],
            ))

        return episodes

    def find_similar_episodes(
        self,
        current_analysis: str,
        symbol: Optional[str] = None,
        n_results: int = 5,
    ) -> List[Tuple[Episode, float]]:
        """
        Find episodes similar to current analysis.

        Args:
            current_analysis: Current analysis text
            symbol: Filter by symbol
            n_results: Number of results

        Returns:
            List of (Episode, similarity_score) tuples
        """
        # Search vector store
        similar = self.vector_store.get_similar_signals(
            symbol=symbol or "",
            current_analysis=current_analysis,
            n_results=n_results,
        )

        results = []
        for doc in similar:
            # Get episode from database
            episodes = self.get_recent_episodes(
                symbol=doc["metadata"].get("symbol"),
                limit=100,
            )

            # Find matching episode by timestamp
            for ep in episodes:
                if abs((ep.timestamp - datetime.fromisoformat(
                    doc["metadata"].get("timestamp", ep.timestamp.isoformat())
                )).total_seconds()) < 60:
                    distance = doc.get("distance", 0)
                    similarity = 1 / (1 + distance) if distance else 0.5
                    results.append((ep, similarity))
                    break

        return results

    def record_regime_transition(
        self,
        symbol: str,
        from_regime: str,
        to_regime: str,
        price: float,
    ) -> int:
        """Record a market regime transition."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO regime_transitions (
                symbol, from_regime, to_regime, timestamp, price_at_transition
            ) VALUES (?, ?, ?, ?, ?)
        """, (symbol, from_regime, to_regime, datetime.now().isoformat(), price))

        transition_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return transition_id

    def record_pattern(
        self,
        pattern_name: str,
        symbol: str,
        signal_type: str,
        confidence: float,
    ) -> int:
        """Record a pattern occurrence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO pattern_occurrences (
                pattern_name, symbol, timestamp, signal_type, confidence
            ) VALUES (?, ?, ?, ?, ?)
        """, (pattern_name, symbol, datetime.now().isoformat(), signal_type, confidence))

        pattern_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return pattern_id

    def update_pattern_outcome(
        self,
        pattern_id: int,
        outcome: str,
        pnl_pct: float,
    ) -> bool:
        """Update a pattern's outcome."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE pattern_occurrences
            SET outcome = ?, pnl_pct = ?
            WHERE id = ?
        """, (outcome, pnl_pct, pattern_id))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    def get_pattern_stats(
        self,
        pattern_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get pattern performance statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if pattern_name:
            cursor.execute("""
                SELECT
                    pattern_name,
                    COUNT(*) as occurrences,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                    AVG(pnl_pct) as avg_pnl
                FROM pattern_occurrences
                WHERE pattern_name = ? AND outcome IS NOT NULL
                GROUP BY pattern_name
            """, (pattern_name,))
        else:
            cursor.execute("""
                SELECT
                    pattern_name,
                    COUNT(*) as occurrences,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                    AVG(pnl_pct) as avg_pnl
                FROM pattern_occurrences
                WHERE outcome IS NOT NULL
                GROUP BY pattern_name
            """)

        rows = cursor.fetchall()
        conn.close()

        stats = {}
        for row in rows:
            total = row[2] + row[3]
            win_rate = row[2] / total if total > 0 else 0
            stats[row[0]] = {
                "occurrences": row[1],
                "wins": row[2],
                "losses": row[3],
                "win_rate": win_rate,
                "avg_pnl": row[4] or 0,
            }

        return stats

    def get_winning_contexts(
        self,
        symbol: Optional[str] = None,
        min_pnl: float = 0,
    ) -> List[Episode]:
        """Get episodes with positive outcomes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if symbol:
            cursor.execute("""
                SELECT * FROM episodes
                WHERE outcome = 'win' AND pnl_pct >= ? AND symbol = ?
                ORDER BY pnl_pct DESC
            """, (min_pnl, symbol))
        else:
            cursor.execute("""
                SELECT * FROM episodes
                WHERE outcome = 'win' AND pnl_pct >= ?
                ORDER BY pnl_pct DESC
            """, (min_pnl,))

        rows = cursor.fetchall()
        conn.close()

        return [
            Episode(
                id=row[0],
                symbol=row[1],
                signal_type=row[2],
                entry_price=row[3],
                stop_loss=row[4],
                take_profit=row[5],
                confidence=row[6],
                technical_summary=row[7],
                onchain_summary=row[8],
                regime=row[9],
                timestamp=datetime.fromisoformat(row[10]),
                outcome=row[11],
                exit_price=row[12],
                pnl_pct=row[13],
                closed_at=datetime.fromisoformat(row[14]) if row[14] else None,
                lessons=row[15],
            )
            for row in rows
        ]

    def get_lessons_learned(self, limit: int = 10) -> List[str]:
        """Get lessons learned from past episodes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT lessons FROM episodes
            WHERE lessons IS NOT NULL AND lessons != ''
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall episodic memory stats."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total episodes
        cursor.execute("SELECT COUNT(*) FROM episodes")
        total = cursor.fetchone()[0]

        # Win/loss stats
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                AVG(pnl_pct) as avg_pnl
            FROM episodes
            WHERE outcome IS NOT NULL
        """)
        row = cursor.fetchone()

        # Regime transitions
        cursor.execute("SELECT COUNT(*) FROM regime_transitions")
        transitions = cursor.fetchone()[0]

        # Pattern occurrences
        cursor.execute("SELECT COUNT(*) FROM pattern_occurrences")
        patterns = cursor.fetchone()[0]

        conn.close()

        completed = row[0] or 0
        wins = row[1] or 0
        losses = row[2] or 0
        avg_pnl = row[3] or 0

        return {
            "total_episodes": total,
            "completed_episodes": completed,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / completed if completed > 0 else 0,
            "avg_pnl": avg_pnl,
            "regime_transitions": transitions,
            "pattern_occurrences": patterns,
        }


def create_episodic_memory(
    db_path: str = "data/episodic_memory.db",
) -> EpisodicMemory:
    """Create an EpisodicMemory instance."""
    return EpisodicMemory(db_path=db_path)


if __name__ == "__main__":
    print("Testing EpisodicMemory...")

    memory = EpisodicMemory(db_path="data/test_episodic.db")

    # Test storing an episode
    print("\n1. Storing episode...")

    episode = Episode(
        id=None,
        symbol="BTC/USDT",
        signal_type="BUY",
        entry_price=42000.0,
        stop_loss=39900.0,
        take_profit=46200.0,
        confidence=0.72,
        technical_summary="RSI oversold at 28, price at 200 EMA support",
        onchain_summary="Whales accumulated 1,500 BTC in 24h",
        regime="neutral",
        timestamp=datetime.now(),
    )

    ep_id = memory.store_episode(episode)
    print(f"  Stored episode ID: {ep_id}")

    # Test updating outcome
    print("\n2. Updating outcome...")

    success = memory.update_outcome(
        episode_id=ep_id,
        outcome="win",
        exit_price=45000.0,
        pnl_pct=7.14,
        lessons="Whale accumulation + oversold RSI is a strong setup",
    )
    print(f"  Update success: {success}")

    # Test retrieving episode
    print("\n3. Retrieving episode...")

    retrieved = memory.get_episode(ep_id)
    if retrieved:
        print(f"  Symbol: {retrieved.symbol}")
        print(f"  Outcome: {retrieved.outcome}")
        print(f"  PnL: {retrieved.pnl_pct:.2f}%")

    # Test getting stats
    print("\n4. Getting stats...")

    stats = memory.get_stats()
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Win rate: {stats['win_rate']:.0%}")

    # Test regime transition
    print("\n5. Recording regime transition...")

    trans_id = memory.record_regime_transition(
        symbol="BTC/USDT",
        from_regime="neutral",
        to_regime="bullish",
        price=43000.0,
    )
    print(f"  Transition ID: {trans_id}")

    # Test pattern recording
    print("\n6. Recording pattern...")

    pattern_id = memory.record_pattern(
        pattern_name="whale_accumulation_reversal",
        symbol="BTC/USDT",
        signal_type="BUY",
        confidence=0.72,
    )
    print(f"  Pattern ID: {pattern_id}")

    memory.update_pattern_outcome(pattern_id, "win", 7.14)

    pattern_stats = memory.get_pattern_stats()
    print(f"  Pattern stats: {pattern_stats}")

    print("\nTest complete!")
