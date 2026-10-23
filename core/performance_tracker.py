"""
Performance Tracker

Tracks and analyzes trading signal performance:
- Win rate calculation
- Profit/loss metrics
- Sharpe ratio
- Confidence calibration
- Strategy performance breakdown
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import SignalDatabase
from memory.episodic import EpisodicMemory


@dataclass
class PerformanceMetrics:
    """Performance metrics for a period."""
    period_start: datetime
    period_end: datetime
    total_signals: int
    completed_signals: int
    wins: int
    losses: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    best_trade: float
    worst_trade: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown: float
    avg_confidence: float
    confidence_accuracy: float  # How accurate confidence predictions are


class PerformanceTracker:
    """
    Tracks and analyzes signal performance.

    Provides detailed metrics for evaluating and improving
    the trading system.
    """

    def __init__(
        self,
        signal_db: Optional[SignalDatabase] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        db_path: str = "data/performance.db",
    ):
        """
        Initialize performance tracker.

        Args:
            signal_db: Signal database for trade data
            episodic_memory: Episodic memory for pattern data
            db_path: Path to performance database
        """
        self.signal_db = signal_db or SignalDatabase()
        self.episodic_memory = episodic_memory
        self.db_path = db_path

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize performance database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                total_signals INTEGER,
                completed_signals INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                total_pnl REAL,
                avg_pnl REAL,
                best_trade REAL,
                worst_trade REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                updated_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbol_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                period TEXT NOT NULL,
                total_signals INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                avg_pnl REAL,
                total_pnl REAL,
                updated_at TEXT,
                UNIQUE(symbol, period)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS confidence_calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                confidence_bucket TEXT NOT NULL,
                total_signals INTEGER,
                wins INTEGER,
                actual_win_rate REAL,
                expected_win_rate REAL,
                calibration_error REAL,
                updated_at TEXT,
                UNIQUE(confidence_bucket)
            )
        """)

        conn.commit()
        conn.close()

    def calculate_metrics(
        self,
        days: int = 30,
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics for a period.

        Args:
            days: Number of days to analyze

        Returns:
            PerformanceMetrics
        """
        summary = self.signal_db.get_performance_summary(days=days)

        # Get detailed PnL data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        conn = sqlite3.connect(self.signal_db.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT pnl_pct, confidence, outcome
            FROM signals
            WHERE outcome IN ('win', 'loss')
              AND timestamp >= ?
            ORDER BY timestamp
        """, (start_date.isoformat(),))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return PerformanceMetrics(
                period_start=start_date,
                period_end=end_date,
                total_signals=0,
                completed_signals=0,
                wins=0,
                losses=0,
                win_rate=0.0,
                avg_pnl=0.0,
                total_pnl=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                avg_confidence=0.0,
                confidence_accuracy=0.0,
            )

        pnls = [row[0] for row in rows if row[0] is not None]
        confidences = [row[1] for row in rows if row[1] is not None]
        outcomes = [row[2] for row in rows]

        # Calculate metrics
        wins = sum(1 for o in outcomes if o == 'win')
        losses = sum(1 for o in outcomes if o == 'loss')
        total = wins + losses

        win_rate = wins / total if total > 0 else 0
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        total_pnl = sum(pnls)
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0

        # Sharpe Ratio (annualized, assuming daily returns)
        if len(pnls) > 1:
            mean_return = sum(pnls) / len(pnls)
            std_return = math.sqrt(sum((p - mean_return) ** 2 for p in pnls) / len(pnls))
            sharpe_ratio = (mean_return / std_return) * math.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Profit Factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max Drawdown
        max_drawdown = self._calculate_max_drawdown(pnls)

        # Confidence accuracy
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        confidence_accuracy = self._calculate_confidence_accuracy(rows)

        return PerformanceMetrics(
            period_start=start_date,
            period_end=end_date,
            total_signals=summary.get("total_signals", 0),
            completed_signals=total,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            total_pnl=total_pnl,
            best_trade=best_trade,
            worst_trade=worst_trade,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            avg_confidence=avg_confidence,
            confidence_accuracy=confidence_accuracy,
        )

    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown from PnL series."""
        if not pnls:
            return 0.0

        # Calculate cumulative returns
        cumulative = []
        total = 0
        for pnl in pnls:
            total += pnl
            cumulative.append(total)

        # Calculate drawdown
        max_dd = 0
        peak = cumulative[0]

        for val in cumulative:
            if val > peak:
                peak = val
            dd = (peak - val)
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_confidence_accuracy(
        self,
        rows: List[Tuple],
    ) -> float:
        """
        Calculate how accurate confidence predictions are.

        High confidence should correlate with higher win rate.
        """
        if not rows:
            return 0.0

        # Group by confidence buckets
        buckets = {
            "0.5-0.6": {"wins": 0, "total": 0},
            "0.6-0.7": {"wins": 0, "total": 0},
            "0.7-0.8": {"wins": 0, "total": 0},
            "0.8-0.9": {"wins": 0, "total": 0},
            "0.9-1.0": {"wins": 0, "total": 0},
        }

        for pnl, confidence, outcome in rows:
            if confidence is None:
                continue

            bucket = None
            if 0.5 <= confidence < 0.6:
                bucket = "0.5-0.6"
            elif 0.6 <= confidence < 0.7:
                bucket = "0.6-0.7"
            elif 0.7 <= confidence < 0.8:
                bucket = "0.7-0.8"
            elif 0.8 <= confidence < 0.9:
                bucket = "0.8-0.9"
            elif 0.9 <= confidence <= 1.0:
                bucket = "0.9-1.0"

            if bucket:
                buckets[bucket]["total"] += 1
                if outcome == "win":
                    buckets[bucket]["wins"] += 1

        # Calculate calibration error
        expected_rates = {
            "0.5-0.6": 0.55,
            "0.6-0.7": 0.65,
            "0.7-0.8": 0.75,
            "0.8-0.9": 0.85,
            "0.9-1.0": 0.95,
        }

        total_error = 0
        total_weight = 0

        for bucket, data in buckets.items():
            if data["total"] > 0:
                actual_rate = data["wins"] / data["total"]
                expected_rate = expected_rates[bucket]
                error = abs(actual_rate - expected_rate)
                total_error += error * data["total"]
                total_weight += data["total"]

        if total_weight == 0:
            return 0.0

        avg_error = total_error / total_weight

        # Convert to accuracy (1 - error)
        return max(0, 1 - avg_error)

    def get_symbol_performance(
        self,
        symbol: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get performance for a specific symbol."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        conn = sqlite3.connect(self.signal_db.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                AVG(pnl_pct) as avg_pnl,
                SUM(pnl_pct) as total_pnl
            FROM signals
            WHERE symbol = ?
              AND outcome IN ('win', 'loss')
              AND timestamp >= ?
        """, (symbol, start_date.isoformat()))

        row = cursor.fetchone()
        conn.close()

        if not row or row[0] == 0:
            return {
                "symbol": symbol,
                "total_signals": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
            }

        wins = row[1] or 0
        losses = row[2] or 0
        total = wins + losses

        return {
            "symbol": symbol,
            "total_signals": row[0],
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total > 0 else 0,
            "avg_pnl": row[3] or 0,
            "total_pnl": row[4] or 0,
        }

    def get_strategy_breakdown(
        self,
        days: int = 30,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get performance breakdown by signal strength.

        Returns metrics for STRONG, MODERATE, WEAK signals.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        conn = sqlite3.connect(self.signal_db.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                strength,
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                AVG(pnl_pct) as avg_pnl
            FROM signals
            WHERE outcome IN ('win', 'loss')
              AND timestamp >= ?
            GROUP BY strength
        """, (start_date.isoformat(),))

        rows = cursor.fetchall()
        conn.close()

        breakdown = {}
        for row in rows:
            strength = row[0] or "unknown"
            wins = row[2] or 0
            losses = row[3] or 0
            total = wins + losses

            breakdown[strength] = {
                "total_signals": row[1],
                "wins": wins,
                "losses": losses,
                "win_rate": wins / total if total > 0 else 0,
                "avg_pnl": row[4] or 0,
            }

        return breakdown

    def get_regime_performance(
        self,
        days: int = 30,
    ) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by market regime."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        conn = sqlite3.connect(self.signal_db.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                regime,
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                AVG(pnl_pct) as avg_pnl
            FROM signals
            WHERE outcome IN ('win', 'loss')
              AND timestamp >= ?
            GROUP BY regime
        """, (start_date.isoformat(),))

        rows = cursor.fetchall()
        conn.close()

        breakdown = {}
        for row in rows:
            regime = row[0] or "unknown"
            wins = row[2] or 0
            losses = row[3] or 0
            total = wins + losses

            breakdown[regime] = {
                "total_signals": row[1],
                "wins": wins,
                "losses": losses,
                "win_rate": wins / total if total > 0 else 0,
                "avg_pnl": row[4] or 0,
            }

        return breakdown

    def update_confidence_calibration(self) -> Dict[str, Any]:
        """Update confidence calibration data."""
        conn = sqlite3.connect(self.signal_db.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT confidence, outcome
            FROM signals
            WHERE outcome IN ('win', 'loss')
              AND confidence IS NOT NULL
        """)

        rows = cursor.fetchall()
        conn.close()

        # Group by buckets
        buckets = {}
        bucket_ranges = [
            ("0.5-0.6", 0.5, 0.6, 0.55),
            ("0.6-0.7", 0.6, 0.7, 0.65),
            ("0.7-0.8", 0.7, 0.8, 0.75),
            ("0.8-0.9", 0.8, 0.9, 0.85),
            ("0.9-1.0", 0.9, 1.0, 0.95),
        ]

        for name, low, high, expected in bucket_ranges:
            buckets[name] = {
                "total": 0,
                "wins": 0,
                "expected": expected,
            }

        for confidence, outcome in rows:
            for name, low, high, expected in bucket_ranges:
                if low <= confidence < high or (high == 1.0 and confidence == 1.0):
                    buckets[name]["total"] += 1
                    if outcome == "win":
                        buckets[name]["wins"] += 1
                    break

        # Store calibration
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for name, data in buckets.items():
            actual_rate = data["wins"] / data["total"] if data["total"] > 0 else 0
            calibration_error = abs(actual_rate - data["expected"])

            cursor.execute("""
                INSERT OR REPLACE INTO confidence_calibration (
                    confidence_bucket, total_signals, wins,
                    actual_win_rate, expected_win_rate, calibration_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                data["total"],
                data["wins"],
                actual_rate,
                data["expected"],
                calibration_error,
                datetime.now().isoformat(),
            ))

        conn.commit()
        conn.close()

        return {
            name: {
                "total": data["total"],
                "wins": data["wins"],
                "actual_win_rate": data["wins"] / data["total"] if data["total"] > 0 else 0,
                "expected_win_rate": data["expected"],
            }
            for name, data in buckets.items()
        }

    def get_confidence_adjustment(
        self,
        current_confidence: float,
    ) -> float:
        """
        Get adjusted confidence based on historical calibration.

        If model tends to be overconfident, this will adjust down.
        If model tends to be underconfident, this will adjust up.

        Args:
            current_confidence: Current confidence value

        Returns:
            Adjusted confidence
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Find the appropriate bucket
        bucket = None
        if 0.5 <= current_confidence < 0.6:
            bucket = "0.5-0.6"
        elif 0.6 <= current_confidence < 0.7:
            bucket = "0.6-0.7"
        elif 0.7 <= current_confidence < 0.8:
            bucket = "0.7-0.8"
        elif 0.8 <= current_confidence < 0.9:
            bucket = "0.8-0.9"
        elif 0.9 <= current_confidence <= 1.0:
            bucket = "0.9-1.0"

        if not bucket:
            return current_confidence

        cursor.execute("""
            SELECT actual_win_rate, expected_win_rate, total_signals
            FROM confidence_calibration
            WHERE confidence_bucket = ?
        """, (bucket,))

        row = cursor.fetchone()
        conn.close()

        if not row or row[2] < 10:  # Need at least 10 samples
            return current_confidence

        actual_rate = row[0]
        expected_rate = row[1]

        # Adjust based on historical accuracy
        # If actual is lower than expected, adjust down
        adjustment_factor = actual_rate / expected_rate if expected_rate > 0 else 1

        adjusted = current_confidence * adjustment_factor

        # Clamp to valid range
        return max(0.5, min(0.99, adjusted))

    def generate_report(
        self,
        days: int = 30,
    ) -> str:
        """
        Generate a performance report.

        Args:
            days: Number of days to analyze

        Returns:
            Formatted report string
        """
        metrics = self.calculate_metrics(days=days)
        strategy = self.get_strategy_breakdown(days=days)
        regime = self.get_regime_performance(days=days)

        report = []
        report.append("=" * 50)
        report.append(f"PERFORMANCE REPORT ({days} days)")
        report.append("=" * 50)
        report.append("")

        # Overall metrics
        report.append("OVERALL PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Total Signals: {metrics.total_signals}")
        report.append(f"Completed: {metrics.completed_signals}")
        report.append(f"Win/Loss: {metrics.wins}/{metrics.losses}")
        report.append(f"Win Rate: {metrics.win_rate:.1%}")
        report.append(f"Avg PnL: {metrics.avg_pnl:.2f}%")
        report.append(f"Total PnL: {metrics.total_pnl:.2f}%")
        report.append(f"Best Trade: {metrics.best_trade:.2f}%")
        report.append(f"Worst Trade: {metrics.worst_trade:.2f}%")
        report.append(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        report.append(f"Profit Factor: {metrics.profit_factor:.2f}")
        report.append(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
        report.append(f"Avg Confidence: {metrics.avg_confidence:.1%}")
        report.append(f"Confidence Accuracy: {metrics.confidence_accuracy:.1%}")
        report.append("")

        # Strategy breakdown
        if strategy:
            report.append("BY SIGNAL STRENGTH")
            report.append("-" * 30)
            for strength, data in strategy.items():
                report.append(f"  {strength.upper()}: {data['win_rate']:.1%} win rate "
                             f"({data['wins']}/{data['losses']}) avg: {data['avg_pnl']:.2f}%")
            report.append("")

        # Regime breakdown
        if regime:
            report.append("BY MARKET REGIME")
            report.append("-" * 30)
            for reg, data in regime.items():
                report.append(f"  {reg}: {data['win_rate']:.1%} win rate "
                             f"({data['wins']}/{data['losses']})")
            report.append("")

        report.append("=" * 50)

        return "\n".join(report)

    def save_daily_metrics(self) -> None:
        """Save today's metrics to database."""
        today = datetime.now().date().isoformat()
        metrics = self.calculate_metrics(days=1)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO daily_metrics (
                date, total_signals, completed_signals, wins, losses,
                win_rate, total_pnl, avg_pnl, best_trade, worst_trade,
                sharpe_ratio, max_drawdown, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            today,
            metrics.total_signals,
            metrics.completed_signals,
            metrics.wins,
            metrics.losses,
            metrics.win_rate,
            metrics.total_pnl,
            metrics.avg_pnl,
            metrics.best_trade,
            metrics.worst_trade,
            metrics.sharpe_ratio,
            metrics.max_drawdown,
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()


def create_performance_tracker(
    signal_db: Optional[SignalDatabase] = None,
) -> PerformanceTracker:
    """Create a PerformanceTracker instance."""
    return PerformanceTracker(signal_db=signal_db)


if __name__ == "__main__":
    print("Testing PerformanceTracker...")

    tracker = PerformanceTracker()

    # Calculate metrics
    print("\n1. Calculating metrics...")
    metrics = tracker.calculate_metrics(days=30)

    print(f"  Total Signals: {metrics.total_signals}")
    print(f"  Win Rate: {metrics.win_rate:.1%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

    # Generate report
    print("\n2. Generating report...")
    report = tracker.generate_report(days=30)
    print(report)

    # Test confidence adjustment
    print("\n3. Testing confidence adjustment...")
    adjusted = tracker.get_confidence_adjustment(0.75)
    print(f"  Original: 0.75 -> Adjusted: {adjusted:.2f}")

    print("\nTest complete!")
