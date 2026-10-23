"""
Prediction Tracker

Stores predictions with timestamps for later evaluation.
Part of the learning/feedback mechanism.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

DB_PATH = "/opt/wallex/data/predictions.db"


@dataclass
class Prediction:
    """A single prediction record."""
    id: int
    symbol: str
    timestamp: datetime
    timeframe: str
    predicted_direction: str
    confidence: float
    price_at_prediction: float
    gru_score: float
    gru_direction: str
    technical_score: float
    regime_score: float
    actual_direction: Optional[str] = None
    price_at_evaluation: Optional[float] = None
    price_change_pct: Optional[float] = None
    was_correct: Optional[bool] = None
    evaluated_at: Optional[datetime] = None


class PredictionTracker:
    """
    Tracks predictions for later evaluation.

    Records each prediction with:
    - Symbol, timestamp, timeframe
    - Predicted direction and confidence
    - Individual model scores (GRU, Technical, Regime)
    - Price at prediction time

    Later, the FeedbackEvaluator compares predictions to actual outcomes.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                timeframe TEXT DEFAULT '4h',

                predicted_direction TEXT,
                confidence REAL,
                price_at_prediction REAL,

                gru_score REAL,
                gru_direction TEXT,
                technical_score REAL,
                regime_score REAL,

                actual_direction TEXT,
                price_at_evaluation REAL,
                price_change_pct REAL,
                was_correct INTEGER,
                evaluated_at DATETIME,

                UNIQUE(symbol, timestamp, timeframe)
            )
        """)

        # Model accuracy table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model_name TEXT NOT NULL,
                period TEXT DEFAULT '7d',
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                accuracy_pct REAL DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                updated_at DATETIME,
                UNIQUE(symbol, model_name, period)
            )
        """)

        # Confidence adjustments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS confidence_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model_name TEXT NOT NULL,
                adjustment_factor REAL DEFAULT 1.0,
                reason TEXT,
                updated_at DATETIME,
                UNIQUE(symbol, model_name)
            )
        """)

        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_symbol
            ON predictions(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_evaluated
            ON predictions(evaluated_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
            ON predictions(timestamp)
        """)

        conn.commit()
        conn.close()

    def record_prediction(
        self,
        symbol: str,
        predicted_direction: str,
        confidence: float,
        price: float,
        scores: Dict[str, Any],
        timeframe: str = "4h"
    ) -> bool:
        """
        Record a new prediction.

        Args:
            symbol: Trading pair (e.g., "AAVE/USDT")
            predicted_direction: "UP", "DOWN", or "NEUTRAL"
            confidence: Confidence score (0-100)
            price: Current price at prediction time
            scores: Dict with model scores {gru_score, gru_direction, technical_score, regime_score}
            timeframe: Evaluation timeframe (default 4h)

        Returns:
            True if recorded successfully, False if duplicate
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO predictions (
                    symbol, timestamp, timeframe,
                    predicted_direction, confidence, price_at_prediction,
                    gru_score, gru_direction, technical_score, regime_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                datetime.now().isoformat(),
                timeframe,
                predicted_direction,
                confidence,
                price,
                scores.get("gru_score", 50),
                scores.get("gru_direction", "neutral"),
                scores.get("technical_score", 50),
                scores.get("regime_score", 50)
            ))
            conn.commit()
            logger.debug(f"Recorded prediction: {symbol} {predicted_direction} ({confidence:.1f}%)")
            return True
        except sqlite3.IntegrityError:
            # Duplicate prediction (same symbol + timestamp + timeframe)
            return False
        finally:
            conn.close()

    def get_pending_evaluations(self, timeframe: str = "4h") -> List[Prediction]:
        """
        Get predictions that are ready for evaluation.

        Returns predictions where:
        - evaluated_at is NULL
        - timestamp + timeframe has passed
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Calculate cutoff time based on timeframe
        hours = int(timeframe.replace("h", ""))
        cutoff = datetime.now() - timedelta(hours=hours)

        cursor.execute("""
            SELECT id, symbol, timestamp, timeframe,
                   predicted_direction, confidence, price_at_prediction,
                   gru_score, gru_direction, technical_score, regime_score,
                   actual_direction, price_at_evaluation, price_change_pct,
                   was_correct, evaluated_at
            FROM predictions
            WHERE evaluated_at IS NULL
              AND timestamp < ?
            ORDER BY timestamp ASC
        """, (cutoff.isoformat(),))

        predictions = []
        for row in cursor.fetchall():
            predictions.append(Prediction(
                id=row[0],
                symbol=row[1],
                timestamp=datetime.fromisoformat(row[2]) if row[2] else None,
                timeframe=row[3],
                predicted_direction=row[4],
                confidence=row[5],
                price_at_prediction=row[6],
                gru_score=row[7],
                gru_direction=row[8],
                technical_score=row[9],
                regime_score=row[10],
                actual_direction=row[11],
                price_at_evaluation=row[12],
                price_change_pct=row[13],
                was_correct=bool(row[14]) if row[14] is not None else None,
                evaluated_at=datetime.fromisoformat(row[15]) if row[15] else None
            ))

        conn.close()
        return predictions

    def update_prediction_outcome(
        self,
        prediction_id: int,
        actual_direction: str,
        price_at_evaluation: float,
        price_change_pct: float,
        was_correct: bool
    ) -> None:
        """Update a prediction with its actual outcome."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE predictions
            SET actual_direction = ?,
                price_at_evaluation = ?,
                price_change_pct = ?,
                was_correct = ?,
                evaluated_at = ?
            WHERE id = ?
        """, (
            actual_direction,
            price_at_evaluation,
            price_change_pct,
            1 if was_correct else 0,
            datetime.now().isoformat(),
            prediction_id
        ))

        conn.commit()
        conn.close()

    def get_accuracy_stats(
        self,
        symbol: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get accuracy statistics.

        Args:
            symbol: Filter by symbol (None for all)
            days: Look back period

        Returns:
            Dict with accuracy metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = datetime.now() - timedelta(days=days)

        if symbol:
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(was_correct) as correct,
                    AVG(confidence) as avg_confidence,
                    AVG(CASE WHEN was_correct = 1 THEN confidence END) as avg_conf_correct,
                    AVG(CASE WHEN was_correct = 0 THEN confidence END) as avg_conf_wrong
                FROM predictions
                WHERE symbol = ?
                  AND evaluated_at IS NOT NULL
                  AND timestamp > ?
            """, (symbol, cutoff.isoformat()))
        else:
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(was_correct) as correct,
                    AVG(confidence) as avg_confidence,
                    AVG(CASE WHEN was_correct = 1 THEN confidence END) as avg_conf_correct,
                    AVG(CASE WHEN was_correct = 0 THEN confidence END) as avg_conf_wrong
                FROM predictions
                WHERE evaluated_at IS NOT NULL
                  AND timestamp > ?
            """, (cutoff.isoformat(),))

        row = cursor.fetchone()
        conn.close()

        total = row[0] or 0
        correct = row[1] or 0

        return {
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy_pct": (correct / total * 100) if total > 0 else 0,
            "avg_confidence": row[2] or 0,
            "avg_confidence_when_correct": row[3] or 0,
            "avg_confidence_when_wrong": row[4] or 0,
            "period_days": days
        }

    def get_model_accuracy(self, days: int = 7) -> Dict[str, Dict[str, float]]:
        """
        Get accuracy broken down by model.

        Evaluates how often each model's signal matched the actual direction.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = datetime.now() - timedelta(days=days)

        # For GRU model
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE
                    WHEN (gru_direction = 'up' AND actual_direction = 'UP')
                      OR (gru_direction = 'down' AND actual_direction = 'DOWN')
                    THEN 1 ELSE 0
                END) as correct,
                AVG(gru_score) as avg_score
            FROM predictions
            WHERE evaluated_at IS NOT NULL
              AND timestamp > ?
        """, (cutoff.isoformat(),))
        gru_row = cursor.fetchone()

        # For Technical model (score > 55 = UP, < 45 = DOWN)
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE
                    WHEN (technical_score > 55 AND actual_direction = 'UP')
                      OR (technical_score < 45 AND actual_direction = 'DOWN')
                      OR (technical_score BETWEEN 45 AND 55 AND actual_direction = 'NEUTRAL')
                    THEN 1 ELSE 0
                END) as correct,
                AVG(technical_score) as avg_score
            FROM predictions
            WHERE evaluated_at IS NOT NULL
              AND timestamp > ?
        """, (cutoff.isoformat(),))
        tech_row = cursor.fetchone()

        # Combined (predicted_direction)
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(was_correct) as correct,
                AVG(confidence) as avg_score
            FROM predictions
            WHERE evaluated_at IS NOT NULL
              AND timestamp > ?
        """, (cutoff.isoformat(),))
        combined_row = cursor.fetchone()

        conn.close()

        def calc_accuracy(row):
            total = row[0] or 0
            correct = row[1] or 0
            return {
                "total": total,
                "correct": correct,
                "accuracy_pct": (correct / total * 100) if total > 0 else 0,
                "avg_score": row[2] or 0
            }

        return {
            "gru": calc_accuracy(gru_row),
            "technical": calc_accuracy(tech_row),
            "combined": calc_accuracy(combined_row)
        }

    def get_token_accuracy(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Get accuracy broken down by token."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = datetime.now() - timedelta(days=days)

        cursor.execute("""
            SELECT
                symbol,
                COUNT(*) as total,
                SUM(was_correct) as correct,
                AVG(confidence) as avg_confidence
            FROM predictions
            WHERE evaluated_at IS NOT NULL
              AND timestamp > ?
            GROUP BY symbol
            HAVING total >= 3
            ORDER BY (correct * 1.0 / total) DESC
            LIMIT ?
        """, (cutoff.isoformat(), limit))

        results = []
        for row in cursor.fetchall():
            total = row[1]
            correct = row[2] or 0
            results.append({
                "symbol": row[0],
                "total": total,
                "correct": correct,
                "accuracy_pct": (correct / total * 100) if total > 0 else 0,
                "avg_confidence": row[3] or 0
            })

        conn.close()
        return results

    def get_confidence_adjustment(self, symbol: str, model: str = "combined") -> float:
        """Get confidence adjustment factor for a symbol/model."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT adjustment_factor
            FROM confidence_adjustments
            WHERE symbol = ? AND model_name = ?
        """, (symbol, model))

        row = cursor.fetchone()
        conn.close()

        return row[0] if row else 1.0

    def set_confidence_adjustment(
        self,
        symbol: str,
        model: str,
        factor: float,
        reason: str = ""
    ) -> None:
        """Set confidence adjustment factor for a symbol/model."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO confidence_adjustments (symbol, model_name, adjustment_factor, reason, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, model_name) DO UPDATE SET
                adjustment_factor = excluded.adjustment_factor,
                reason = excluded.reason,
                updated_at = excluded.updated_at
        """, (symbol, model, factor, reason, datetime.now().isoformat()))

        conn.commit()
        conn.close()


# Singleton instance
_tracker = None

def get_prediction_tracker() -> PredictionTracker:
    """Get the singleton prediction tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = PredictionTracker()
    return _tracker
