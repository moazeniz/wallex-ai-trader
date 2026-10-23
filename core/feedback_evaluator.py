"""
Feedback Evaluator

Evaluates predictions against actual outcomes.
Enhanced with automatic meta-learner training.
"""

import sys
sys.path.insert(0, "/opt/wallex")

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from core.prediction_tracker import get_prediction_tracker, Prediction
from data.exchange_client import ExchangeClient

# Import enhanced ensemble for dynamic weight updates
try:
    from ml.ensemble_enhanced import get_enhanced_ensemble
    ENHANCED_ENSEMBLE = True
except:
    ENHANCED_ENSEMBLE = False

logger = logging.getLogger(__name__)


class FeedbackEvaluator:
    """
    Evaluates predictions against actual price movements.

    Runs periodically to:
    1. Find predictions that have matured (timeframe expired)
    2. Get current price for each symbol
    3. Compare predicted vs actual direction
    4. Update prediction records with outcomes
    5. Feed accuracy back to ensemble for dynamic weights
    6. Automatically train meta-learner when enough data
    """

    # Direction thresholds
    UP_THRESHOLD = 0.005      # >0.5% up = UP
    DOWN_THRESHOLD = -0.005   # >0.5% down = DOWN

    def __init__(self):
        self.tracker = get_prediction_tracker()
        self.client = ExchangeClient()
        self._last_meta_train = None

    def evaluate_pending(self, timeframe: str = "4h") -> Dict[str, Any]:
        """
        Evaluate all pending predictions.

        Returns:
            Dict with evaluation summary
        """
        pending = self.tracker.get_pending_evaluations(timeframe)

        if not pending:
            return {
                "evaluated": 0,
                "correct": 0,
                "accuracy": 0,
                "message": "No pending predictions to evaluate"
            }

        evaluated = 0
        correct = 0
        errors = 0
        outcomes = []

        for prediction in pending:
            try:
                result = self._evaluate_single(prediction)
                if result:
                    evaluated += 1
                    if result["was_correct"]:
                        correct += 1
                    outcomes.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {prediction.symbol}: {e}")
                errors += 1

        accuracy = (correct / evaluated * 100) if evaluated > 0 else 0

        # Feed outcomes to enhanced ensemble
        if outcomes and ENHANCED_ENSEMBLE:
            self._feed_outcomes_to_ensemble(outcomes)
            # Try to train meta-learner
            self._try_train_meta_learner()

        summary = {
            "evaluated": evaluated,
            "correct": correct,
            "wrong": evaluated - correct,
            "errors": errors,
            "accuracy_pct": accuracy
        }

        if evaluated > 0:
            logger.info(
                f"[Learning] Evaluated {evaluated} predictions: "
                f"{correct}/{evaluated} correct ({accuracy:.1f}%)"
            )

        return summary

    def _feed_outcomes_to_ensemble(self, outcomes: List[Dict]):
        """Feed prediction outcomes to ensemble for learning."""
        try:
            ensemble = get_enhanced_ensemble()

            for outcome in outcomes:
                # Record outcome in ensemble
                ensemble.record_outcome(
                    symbol=outcome['symbol'],
                    actual_direction=outcome['actual'],
                    model_predictions=outcome.get('model_predictions')
                )

            logger.info(f"[Learning] Fed {len(outcomes)} outcomes to ensemble")

        except Exception as e:
            logger.warning(f"Could not feed outcomes to ensemble: {e}")

    def _try_train_meta_learner(self):
        """Attempt to train or retrain the meta-learner."""
        try:
            ensemble = get_enhanced_ensemble()
            result = ensemble.train_meta_learner()

            if result.get("trained"):
                logger.info(f"[Learning] Meta-learner training: {result}")
                self._last_meta_train = datetime.now()
            elif result.get("reason"):
                # Log only occasionally to avoid spam
                pass

        except Exception as e:
            logger.warning(f"Could not train meta-learner: {e}")

    def _evaluate_single(self, prediction: Prediction) -> Optional[Dict[str, Any]]:
        """Evaluate a single prediction."""
        # Get current price
        try:
            df = self.client.fetch_ohlcv(prediction.symbol, "1h", limit=1)
            if df is None or len(df) == 0:
                return None
            current_price = float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning(f"Could not get price for {prediction.symbol}: {e}")
            return None

        # Calculate price change
        price_change_pct = (current_price - prediction.price_at_prediction) / prediction.price_at_prediction

        # Determine actual direction
        if price_change_pct > self.UP_THRESHOLD:
            actual_direction = "UP"
        elif price_change_pct < self.DOWN_THRESHOLD:
            actual_direction = "DOWN"
        else:
            actual_direction = "NEUTRAL"

        # Check if prediction was correct
        if prediction.predicted_direction == "NEUTRAL":
            was_correct = abs(price_change_pct) < 0.02
        else:
            was_correct = prediction.predicted_direction == actual_direction

        # Update the prediction record
        self.tracker.update_prediction_outcome(
            prediction_id=prediction.id,
            actual_direction=actual_direction,
            price_at_evaluation=current_price,
            price_change_pct=price_change_pct * 100,
            was_correct=was_correct
        )

        # Get model predictions if available
        model_predictions = None
        if hasattr(prediction, 'model_predictions'):
            model_predictions = prediction.model_predictions

        result = {
            "symbol": prediction.symbol,
            "predicted": prediction.predicted_direction,
            "actual": actual_direction,
            "price_change_pct": price_change_pct * 100,
            "was_correct": was_correct,
            "confidence": prediction.confidence,
            "model_predictions": model_predictions
        }

        if prediction.confidence >= 70:
            status = "correct" if was_correct else "WRONG"
            logger.info(
                f"[Learning] {prediction.symbol}: "
                f"Predicted {prediction.predicted_direction} ({prediction.confidence:.0f}%) -> "
                f"Actual {actual_direction} ({price_change_pct*100:+.2f}%) = {status}"
            )

        return result

    def get_learning_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive learning summary."""
        overall = self.tracker.get_accuracy_stats(days=days)
        by_model = self.tracker.get_model_accuracy(days=days)
        by_token = self.tracker.get_token_accuracy(days=days)

        # Include ensemble info if available
        ensemble_stats = None
        if ENHANCED_ENSEMBLE:
            try:
                ensemble = get_enhanced_ensemble()
                ensemble_stats = ensemble.get_stats()
            except:
                pass

        return {
            "overall": overall,
            "by_model": by_model,
            "by_token": by_token,
            "period_days": days,
            "ensemble_stats": ensemble_stats
        }

    def update_confidence_adjustments(self, min_predictions: int = 10) -> List[Dict[str, Any]]:
        """Update confidence adjustments based on historical accuracy."""
        adjustments = []
        tokens = self.tracker.get_token_accuracy(days=7, limit=50)

        for token in tokens:
            if token["total"] < min_predictions:
                continue

            accuracy = token["accuracy_pct"]
            symbol = token["symbol"]

            if accuracy >= 65:
                factor = 1.0 + (accuracy - 50) / 100
                reason = f"High accuracy ({accuracy:.1f}%)"
            elif accuracy <= 35:
                factor = 1.0 - (50 - accuracy) / 100
                reason = f"Low accuracy ({accuracy:.1f}%)"
            else:
                factor = 1.0
                reason = "Normal accuracy"

            factor = max(0.5, min(1.5, factor))

            current = self.tracker.get_confidence_adjustment(symbol, "combined")
            if abs(factor - current) > 0.05:
                self.tracker.set_confidence_adjustment(symbol, "combined", factor, reason)
                adjustments.append({
                    "symbol": symbol,
                    "old_factor": current,
                    "new_factor": factor,
                    "reason": reason
                })
                logger.info(f"[Learning] Adjusted {symbol} confidence: {current:.2f} -> {factor:.2f} ({reason})")

        return adjustments

    def format_learning_report(self, days: int = 7) -> str:
        """Format a human-readable learning report."""
        summary = self.get_learning_summary(days)
        overall = summary["overall"]
        by_model = summary["by_model"]
        by_token = summary["by_token"]

        report = f"""*Learning System Report* ({days}d)

*Overall Accuracy*
Total Predictions: {overall['total_predictions']}
Correct: {overall['correct_predictions']}
Accuracy: {overall['accuracy_pct']:.1f}%
Avg Confidence: {overall['avg_confidence']:.1f}%

*Model Performance*
"""
        for model, stats in by_model.items():
            if stats["total"] > 0:
                report += f"  {model.upper()}: {stats['accuracy_pct']:.1f}% ({stats['total']} predictions)\n"

        # Add ensemble stats if available
        if summary.get('ensemble_stats'):
            stats = summary['ensemble_stats']

            # Dynamic weights
            weights = stats.get('current_weights', {})
            if weights:
                report += "\n*Dynamic Weights*\n"
                for model, weight in weights.items():
                    report += f"  {model}: {weight:.1%}\n"

            # Meta-learner status
            report += "\n*Meta-Learner*\n"
            if stats.get('meta_learner_trained'):
                report += f"  Status: ACTIVE\n"
                report += f"  Trained on: {stats.get('meta_learner_samples', 0)} samples\n"
                report += f"  Last trained: {stats.get('meta_learner_last_trained', 'Unknown')}\n"
            else:
                buffer = stats.get('prediction_buffer', {})
                report += f"  Status: Pending ({buffer.get('evaluated', 0)}/50 samples)\n"

        report += "\n*Top Tokens*\n"
        for token in by_token[:5]:
            emoji = "good" if token["accuracy_pct"] >= 60 else "low" if token["accuracy_pct"] < 40 else "-"
            report += f"  {token['symbol']}: {token['accuracy_pct']:.1f}% ({token['total']} pred) [{emoji}]\n"

        if by_token:
            worst = sorted(by_token, key=lambda x: x["accuracy_pct"])[:3]
            if worst and worst[0]["accuracy_pct"] < 45:
                report += "\n*Underperforming*\n"
                for token in worst:
                    if token["accuracy_pct"] < 45:
                        report += f"  {token['symbol']}: {token['accuracy_pct']:.1f}% (needs attention)\n"

        return report


# Singleton instance
_evaluator = None

def get_feedback_evaluator() -> FeedbackEvaluator:
    """Get the singleton feedback evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = FeedbackEvaluator()
    return _evaluator
