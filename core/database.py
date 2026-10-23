"""
Database Layer

SQLite-based persistence for signal history and performance tracking.
"""

import sqlite3
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import threading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import TradingSignal, SignalType, SignalStrength


@dataclass
class SignalRecord:
    """Database record for a signal."""
    id: int
    symbol: str
    signal_type: str
    strength: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    confidence: float
    reasoning: str
    technical_summary: str
    onchain_summary: str
    risk_assessment: str
    created_at: datetime
    outcome: Optional[str] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    closed_at: Optional[datetime] = None


class SignalDatabase:
    """
    SQLite database for signal persistence.

    Stores signal history and tracks outcomes for performance analysis.
    """

    def __init__(self, db_path: str = "data/signals.db"):
        """
        Initialize the database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strength TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                position_size_pct REAL NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT,
                technical_summary TEXT,
                onchain_summary TEXT,
                risk_assessment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                outcome TEXT,
                exit_price REAL,
                pnl_pct REAL,
                closed_at TIMESTAMP
            )
        """)

        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL UNIQUE,
                total_signals INTEGER DEFAULT 0,
                winning_signals INTEGER DEFAULT 0,
                losing_signals INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_win REAL DEFAULT 0,
                avg_loss REAL DEFAULT 0,
                best_trade REAL DEFAULT 0,
                worst_trade REAL DEFAULT 0
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_symbol
            ON signals(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_created
            ON signals(created_at)
        """)

        conn.commit()

    def save_signal(self, signal: TradingSignal) -> int:
        """
        Save a trading signal to database.

        Args:
            signal: TradingSignal to save

        Returns:
            Signal ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO signals (
                symbol, signal_type, strength, entry_price, stop_loss,
                take_profit, position_size_pct, confidence, reasoning,
                technical_summary, onchain_summary, risk_assessment, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.symbol,
            signal.signal_type.value,
            signal.strength.value,
            signal.entry_price,
            signal.stop_loss,
            signal.take_profit,
            signal.position_size_pct,
            signal.confidence,
            signal.reasoning,
            signal.technical_summary,
            signal.onchain_summary,
            signal.risk_assessment,
            signal.timestamp,
        ))

        conn.commit()
        return cursor.lastrowid

    def update_signal_outcome(
        self,
        signal_id: int,
        outcome: str,
        exit_price: float,
        pnl_pct: float,
    ) -> None:
        """
        Update signal with outcome.

        Args:
            signal_id: Signal ID
            outcome: "win", "loss", or "breakeven"
            exit_price: Exit price
            pnl_pct: PnL percentage
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE signals
            SET outcome = ?, exit_price = ?, pnl_pct = ?, closed_at = ?
            WHERE id = ?
        """, (outcome, exit_price, pnl_pct, datetime.now(), signal_id))

        conn.commit()

        # Update daily metrics
        self._update_daily_metrics()

    def get_signal(self, signal_id: int) -> Optional[SignalRecord]:
        """Get signal by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM signals WHERE id = ?", (signal_id,))
        row = cursor.fetchone()

        if row:
            return self._row_to_record(row)
        return None

    def get_recent_signals(
        self,
        limit: int = 50,
        symbol: Optional[str] = None,
    ) -> List[SignalRecord]:
        """Get recent signals."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if symbol:
            cursor.execute("""
                SELECT * FROM signals
                WHERE symbol = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (symbol, limit))
        else:
            cursor.execute("""
                SELECT * FROM signals
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_open_signals(self, symbol: Optional[str] = None) -> List[SignalRecord]:
        """Get signals without outcomes."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if symbol:
            cursor.execute("""
                SELECT * FROM signals
                WHERE outcome IS NULL AND symbol = ?
                ORDER BY created_at DESC
            """, (symbol,))
        else:
            cursor.execute("""
                SELECT * FROM signals
                WHERE outcome IS NULL
                ORDER BY created_at DESC
            """)

        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_signals_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[SignalRecord]:
        """Get signals within date range."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM signals
            WHERE created_at BETWEEN ? AND ?
            ORDER BY created_at DESC
        """, (start_date, end_date))

        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_performance_summary(
        self,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get performance summary for recent period.

        Args:
            days: Number of days to analyze

        Returns:
            Performance metrics dictionary
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        start_date = datetime.now() - timedelta(days=days)

        cursor.execute("""
            SELECT
                COUNT(*) as total_signals,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as winning,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losing,
                AVG(CASE WHEN outcome IS NOT NULL THEN pnl_pct ELSE NULL END) as avg_pnl,
                AVG(CASE WHEN outcome = 'win' THEN pnl_pct ELSE NULL END) as avg_win,
                AVG(CASE WHEN outcome = 'loss' THEN pnl_pct ELSE NULL END) as avg_loss,
                MAX(pnl_pct) as best_trade,
                MIN(pnl_pct) as worst_trade,
                SUM(pnl_pct) as total_pnl
            FROM signals
            WHERE created_at >= ? AND outcome IS NOT NULL
        """, (start_date,))

        row = cursor.fetchone()

        total = row['total_signals'] or 0
        winning = row['winning'] or 0
        losing = row['losing'] or 0

        return {
            "period_days": days,
            "total_signals": total,
            "winning_signals": winning,
            "losing_signals": losing,
            "open_signals": total - winning - losing,
            "win_rate": (winning / total * 100) if total > 0 else 0,
            "avg_pnl": row['avg_pnl'] or 0,
            "avg_win": row['avg_win'] or 0,
            "avg_loss": row['avg_loss'] or 0,
            "best_trade": row['best_trade'] or 0,
            "worst_trade": row['worst_trade'] or 0,
            "total_pnl": row['total_pnl'] or 0,
        }

    def get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """Get performance for a specific symbol."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                AVG(pnl_pct) as avg_pnl,
                SUM(pnl_pct) as total_pnl
            FROM signals
            WHERE symbol = ? AND outcome IS NOT NULL
        """, (symbol,))

        row = cursor.fetchone()
        total = row['total'] or 0
        wins = row['wins'] or 0

        return {
            "symbol": symbol,
            "total_signals": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": (wins / total * 100) if total > 0 else 0,
            "avg_pnl": row['avg_pnl'] or 0,
            "total_pnl": row['total_pnl'] or 0,
        }

    def _update_daily_metrics(self) -> None:
        """Update daily performance metrics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        today = datetime.now().date()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                SUM(pnl_pct) as total_pnl,
                AVG(CASE WHEN outcome = 'win' THEN pnl_pct END) as avg_win,
                AVG(CASE WHEN outcome = 'loss' THEN pnl_pct END) as avg_loss,
                MAX(pnl_pct) as best,
                MIN(pnl_pct) as worst
            FROM signals
            WHERE DATE(created_at) = ? AND outcome IS NOT NULL
        """, (today,))

        row = cursor.fetchone()
        total = row['total'] or 0
        wins = row['wins'] or 0

        cursor.execute("""
            INSERT OR REPLACE INTO performance_metrics
            (date, total_signals, winning_signals, losing_signals,
             total_pnl, win_rate, avg_win, avg_loss, best_trade, worst_trade)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            today,
            total,
            wins,
            row['losses'] or 0,
            row['total_pnl'] or 0,
            (wins / total * 100) if total > 0 else 0,
            row['avg_win'] or 0,
            row['avg_loss'] or 0,
            row['best'] or 0,
            row['worst'] or 0,
        ))

        conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> SignalRecord:
        """Convert database row to SignalRecord."""
        return SignalRecord(
            id=row['id'],
            symbol=row['symbol'],
            signal_type=row['signal_type'],
            strength=row['strength'],
            entry_price=row['entry_price'],
            stop_loss=row['stop_loss'],
            take_profit=row['take_profit'],
            position_size_pct=row['position_size_pct'],
            confidence=row['confidence'],
            reasoning=row['reasoning'],
            technical_summary=row['technical_summary'],
            onchain_summary=row['onchain_summary'],
            risk_assessment=row['risk_assessment'],
            created_at=row['created_at'],
            outcome=row['outcome'],
            exit_price=row['exit_price'],
            pnl_pct=row['pnl_pct'],
            closed_at=row['closed_at'],
        )

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')


def create_database(db_path: str = "data/signals.db") -> SignalDatabase:
    """Create a SignalDatabase instance."""
    return SignalDatabase(db_path=db_path)


if __name__ == "__main__":
    print("Testing SignalDatabase...")

    db = SignalDatabase("data/test_signals.db")

    # Create test signal
    signal = TradingSignal(
        symbol="BTC/USDT",
        signal_type=SignalType.BUY,
        strength=SignalStrength.MODERATE,
        entry_price=96500,
        stop_loss=91675,
        take_profit=104220,
        position_size_pct=1.5,
        confidence=0.72,
        reasoning="Test signal",
        technical_summary="Bullish trend",
        onchain_summary="Whale accumulation",
        risk_assessment="APPROVED",
        timestamp=datetime.now(),
    )

    # Save signal
    signal_id = db.save_signal(signal)
    print(f"\n1. Saved signal with ID: {signal_id}")

    # Retrieve signal
    record = db.get_signal(signal_id)
    print(f"2. Retrieved: {record.symbol} - {record.signal_type}")

    # Update outcome
    db.update_signal_outcome(signal_id, "win", 98500, 2.07)
    print(f"3. Updated outcome: win +2.07%")

    # Get performance
    perf = db.get_performance_summary(30)
    print(f"\n4. Performance Summary (30 days):")
    for key, value in perf.items():
        print(f"   {key}: {value}")

    db.close()
