"""
Enhanced Ensemble Combiner

Improvements:
1. Dynamic weighting based on rolling accuracy
2. Confidence gating - only include confident models
3. Stacking meta-learner - learns when to trust which model
4. Focus on bullish predictions (GRU is better at UP)
5. Regime-aware model selection
6. Prediction buffer for automatic meta-learner training
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import sys
import logging

sys.path.insert(0, '/opt/wallex')

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class DynamicWeightTracker:
    """
    Tracks model accuracy over rolling windows and updates weights.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions: Dict[str, List[Dict]] = {}
        self.accuracy_cache: Dict[str, float] = {}
        self.last_update = datetime.now()

    def record_prediction(
        self,
        model_name: str,
        predicted: str,
        actual: str,
        confidence: float
    ):
        """Record a prediction outcome for a model."""
        if model_name not in self.predictions:
            self.predictions[model_name] = []

        correct = predicted == actual
        self.predictions[model_name].append({
            "predicted": predicted,
            "actual": actual,
            "correct": correct,
            "confidence": confidence,
            "timestamp": datetime.now()
        })

        # Keep only last N predictions
        if len(self.predictions[model_name]) > self.window_size:
            self.predictions[model_name] = self.predictions[model_name][-self.window_size:]

        # Invalidate cache
        self.accuracy_cache.pop(model_name, None)

    def get_accuracy(self, model_name: str, direction: str = None) -> float:
        """Get rolling accuracy for a model, optionally filtered by direction."""
        if model_name not in self.predictions:
            return 0.5

        preds = self.predictions[model_name]
        if not preds:
            return 0.5

        if direction:
            preds = [p for p in preds if p["predicted"] == direction]

        if not preds:
            return 0.5

        return sum(p["correct"] for p in preds) / len(preds)

    def get_dynamic_weights(self, models: List[str]) -> Dict[str, float]:
        """Calculate weights based on recent accuracy."""
        accuracies = {}
        for model in models:
            up_acc = self.get_accuracy(model, "UP")
            overall_acc = self.get_accuracy(model)
            accuracies[model] = 0.7 * up_acc + 0.3 * overall_acc

        total = sum(accuracies.values()) or 1
        weights = {m: acc / total for m, acc in accuracies.items()}

        return weights


class StackingMetaLearner:
    """
    Meta-learner that learns when to trust which model.
    """

    def __init__(self):
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.last_trained = None
        self.training_samples = 0

    def _prepare_features(
        self,
        model_predictions: Dict[str, Dict],
        regime: str = "unknown",
        volatility: float = 0.0
    ) -> np.ndarray:
        """Prepare features for meta-learner."""
        features = []

        # Model predictions and confidences
        for model_name in sorted(model_predictions.keys()):
            pred = model_predictions[model_name]
            features.append(pred.get("probability_up", 0.5))
            features.append(pred.get("probability_down", 0.5))
            features.append(pred.get("confidence", 0.0))

        # Agreement score
        directions = [p.get("direction") for p in model_predictions.values()]
        agreement = max(directions.count(d) for d in set(directions)) / len(directions) if directions else 0
        features.append(agreement)

        # Regime encoding (one-hot)
        regimes = ["trending_up", "trending_down", "ranging", "high_volatility", "sideways", "unknown"]
        for r in regimes:
            features.append(1.0 if regime == r else 0.0)

        # Volatility
        features.append(volatility)

        return np.array(features).reshape(1, -1)

    def train(
        self,
        training_data: List[Tuple[Dict[str, Dict], str, str, float]],
    ) -> Dict[str, Any]:
        """Train the meta-learner on historical predictions."""
        if not HAS_XGB:
            return {"error": "XGBoost not available"}

        X, y = [], []

        for model_preds, regime, actual, vol in training_data:
            try:
                features = self._prepare_features(model_preds, regime, vol)
                X.append(features.flatten())

                if actual == "UP":
                    y.append(1)
                elif actual == "DOWN":
                    y.append(0)
                else:
                    y.append(0)
            except Exception as e:
                continue

        X = np.array(X)
        y = np.array(y)

        if len(X) < 50:
            return {"error": f"Not enough data: {len(X)}", "samples": len(X)}

        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        # Train/test split
        split = int(len(X) * 0.8)
        self.model.fit(X[:split], y[:split])

        # Evaluate
        val_pred = self.model.predict(X[split:])
        val_acc = np.mean(val_pred == y[split:])

        self.is_trained = True
        self.last_trained = datetime.now()
        self.training_samples = len(X)

        logger.info(f"[MetaLearner] Trained on {len(X)} samples, validation accuracy: {val_acc:.1%}")

        return {
            "success": True,
            "train_samples": split,
            "val_samples": len(X) - split,
            "val_accuracy": val_acc
        }

    def predict(
        self,
        model_predictions: Dict[str, Dict],
        regime: str = "unknown",
        volatility: float = 0.0
    ) -> Tuple[float, float]:
        """Predict probability of UP using meta-learner."""
        if not self.is_trained or self.model is None:
            probs = [p.get("probability_up", 0.5) for p in model_predictions.values()]
            return np.mean(probs), 0.0

        features = self._prepare_features(model_predictions, regime, volatility)
        prob_up = self.model.predict_proba(features)[0, 1]
        confidence = abs(prob_up - 0.5) * 2

        return prob_up, confidence

    def save(self, path: str):
        """Save meta-learner."""
        data = {
            "model": self.model,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
            "last_trained": self.last_trained,
            "training_samples": self.training_samples
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load meta-learner."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.is_trained = data["is_trained"]
            self.feature_names = data.get("feature_names", [])
            self.last_trained = data.get("last_trained")
            self.training_samples = data.get("training_samples", 0)
        except:
            pass


class PredictionBuffer:
    """
    Stores recent predictions with full context for meta-learner training.
    """

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.predictions: List[Dict] = []

    def store(
        self,
        symbol: str,
        model_predictions: Dict[str, Dict],
        regime: str,
        volatility: float,
        ensemble_direction: str,
        timestamp: datetime = None
    ):
        """Store a prediction with full context."""
        self.predictions.append({
            "symbol": symbol,
            "model_predictions": model_predictions,
            "regime": regime,
            "volatility": volatility,
            "ensemble_direction": ensemble_direction,
            "timestamp": timestamp or datetime.now(),
            "actual_direction": None,  # Filled later when evaluated
            "evaluated": False
        })

        # Keep buffer size limited
        if len(self.predictions) > self.max_size:
            # Remove oldest evaluated predictions first
            evaluated = [p for p in self.predictions if p["evaluated"]]
            unevaluated = [p for p in self.predictions if not p["evaluated"]]

            if len(evaluated) > self.max_size // 2:
                evaluated = evaluated[-(self.max_size // 2):]

            self.predictions = evaluated + unevaluated
            self.predictions = self.predictions[-self.max_size:]

    def mark_evaluated(self, symbol: str, actual_direction: str, max_age_hours: int = 6):
        """Mark a prediction as evaluated with actual outcome."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        for pred in reversed(self.predictions):
            if pred["symbol"] == symbol and not pred["evaluated"]:
                if pred["timestamp"] > cutoff:
                    pred["actual_direction"] = actual_direction
                    pred["evaluated"] = True
                    return True
        return False

    def get_training_data(self) -> List[Tuple[Dict[str, Dict], str, str, float]]:
        """Get evaluated predictions formatted for meta-learner training."""
        training_data = []

        for pred in self.predictions:
            if pred["evaluated"] and pred["actual_direction"]:
                training_data.append((
                    pred["model_predictions"],
                    pred["regime"],
                    pred["actual_direction"],
                    pred["volatility"]
                ))

        return training_data

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        evaluated = sum(1 for p in self.predictions if p["evaluated"])
        return {
            "total": len(self.predictions),
            "evaluated": evaluated,
            "pending": len(self.predictions) - evaluated,
            "ready_for_training": evaluated >= 50
        }


class EnhancedEnsembleCombiner:
    """
    Enhanced ensemble with:
    - Dynamic weighting
    - Confidence gating
    - Stacking meta-learner with automatic training
    - Focus on bullish predictions
    - Prediction buffer for learning
    """

    def __init__(self, focus_bullish: bool = True):
        self.models: Dict[str, Any] = {}
        self.focus_bullish = focus_bullish

        # Dynamic weighting
        self.weight_tracker = DynamicWeightTracker()

        # Stacking meta-learner
        self.meta_learner = StackingMetaLearner()

        # Prediction buffer for meta-learner training
        self.prediction_buffer = PredictionBuffer(max_size=500)

        # Static fallback weights
        self.default_weights = {
            "gru": 0.50,
            "lstm": 0.25,
            "xgboost": 0.25,
        }

        # Confidence thresholds for gating
        self.min_confidence = 0.15
        self.bullish_boost = 1.3

        # Use dynamic weights after enough data
        self.min_predictions_for_dynamic = 50

        # Meta-learner training settings
        self.min_samples_for_training = 50
        self.retrain_interval_hours = 24

    def register_model(self, name: str, model: Any, weight: float = None):
        """Register a model with the ensemble."""
        self.models[name] = model
        if weight:
            self.default_weights[name] = weight

    def _should_include_model(self, pred: Dict, model_name: str) -> bool:
        """Confidence gating: decide if model should be included."""
        confidence = pred.get("confidence", 0)
        return confidence >= self.min_confidence

    def _get_weights(self) -> Dict[str, float]:
        """Get current weights (dynamic or static)."""
        total_preds = sum(len(p) for p in self.weight_tracker.predictions.values())

        if total_preds >= self.min_predictions_for_dynamic:
            return self.weight_tracker.get_dynamic_weights(list(self.models.keys()))
        else:
            return self.default_weights

    def predict(
        self,
        df: pd.DataFrame,
        token: str = None,
        df_4h: pd.DataFrame = None,
        df_1d: pd.DataFrame = None,
        btc_df: pd.DataFrame = None,
        regime: str = "unknown",
        use_meta_learner: bool = True
    ) -> Dict[str, Any]:
        """Get ensemble prediction with all enhancements."""

        predictions = {}
        included_models = []

        # Get predictions from each model
        for name, model in self.models.items():
            try:
                try:
                    pred = model.predict(df, symbol=token, df_4h=df_4h, df_1d=df_1d, btc_df=btc_df)
                except TypeError:
                    pred = model.predict(df, token=token) if token else model.predict(df)

                predictions[name] = pred

                if self._should_include_model(pred, name):
                    included_models.append(name)

            except Exception as e:
                print(f"Model {name} error: {e}")
                predictions[name] = {
                    "direction": "NEUTRAL",
                    "probability_up": 0.5,
                    "probability_down": 0.5,
                    "confidence": 0.0
                }

        if not included_models:
            included_models = list(self.models.keys())

        weights = self._get_weights()

        # Calculate volatility
        vol = df['close'].pct_change().std() if len(df) > 0 else 0.0

        # Use meta-learner if trained
        if use_meta_learner and self.meta_learner.is_trained:
            meta_prob_up, meta_confidence = self.meta_learner.predict(
                predictions, regime, vol
            )
        else:
            meta_prob_up = None

        # Weighted combination
        weighted_prob_up = 0
        weighted_prob_down = 0
        total_weight = 0

        for name in included_models:
            pred = predictions[name]
            weight = weights.get(name, 0.33)

            prob_up = pred.get("probability_up", 0.5)
            prob_down = pred.get("probability_down", 0.5)

            if self.focus_bullish and pred.get("direction") == "UP":
                weight *= self.bullish_boost

            weighted_prob_up += prob_up * weight
            weighted_prob_down += prob_down * weight
            total_weight += weight

        if total_weight > 0:
            ensemble_prob_up = weighted_prob_up / total_weight
            ensemble_prob_down = weighted_prob_down / total_weight
        else:
            ensemble_prob_up = 0.5
            ensemble_prob_down = 0.5

        # Blend with meta-learner if available
        if meta_prob_up is not None:
            ensemble_prob_up = 0.6 * meta_prob_up + 0.4 * ensemble_prob_up

        # Determine final direction
        if self.focus_bullish:
            up_threshold = 0.50
            down_threshold = 0.60
        else:
            up_threshold = 0.55
            down_threshold = 0.55

        if ensemble_prob_up > up_threshold:
            direction = "UP"
            confidence = (ensemble_prob_up - 0.5) * 2
        elif ensemble_prob_down > down_threshold:
            direction = "DOWN"
            confidence = (ensemble_prob_down - 0.5) * 2
        else:
            direction = "NEUTRAL"
            confidence = 0.0

        # Model agreement
        directions = [p.get("direction", "NEUTRAL") for p in predictions.values()]
        unique_directions = set(d for d in directions if d != "NEUTRAL")
        disagreement = len(unique_directions) > 1
        agreement_ratio = max(directions.count(d) for d in set(directions)) / len(directions) if directions else 0

        if disagreement and direction != "NEUTRAL":
            if not (direction == "UP" and agreement_ratio >= 0.5):
                confidence *= 0.7

        # Store prediction in buffer for future training
        if token:
            self.prediction_buffer.store(
                symbol=token,
                model_predictions=predictions,
                regime=regime,
                volatility=vol,
                ensemble_direction=direction
            )

        return {
            "direction": direction,
            "probability_up": ensemble_prob_up,
            "probability_down": ensemble_prob_down,
            "confidence": confidence,
            "model_predictions": {
                name: {
                    "direction": pred.get("direction"),
                    "probability_up": pred.get("probability_up"),
                    "probability_down": pred.get("probability_down"),
                    "confidence": pred.get("confidence"),
                    "included": name in included_models
                }
                for name, pred in predictions.items()
            },
            "weights_used": {k: weights.get(k, 0) for k in included_models},
            "disagreement": disagreement,
            "agreement_ratio": agreement_ratio,
            "models_included": included_models,
            "used_meta_learner": meta_prob_up is not None,
        }

    def record_outcome(
        self,
        symbol: str,
        actual_direction: str,
        model_predictions: Dict[str, Dict] = None
    ):
        """Record prediction outcome for learning."""
        # Update weight tracker if model predictions provided
        if model_predictions:
            for model_name, pred in model_predictions.items():
                self.weight_tracker.record_prediction(
                    model_name,
                    pred.get("direction", "NEUTRAL"),
                    actual_direction,
                    pred.get("confidence", 0)
                )

        # Mark in prediction buffer
        self.prediction_buffer.mark_evaluated(symbol, actual_direction)

    def train_meta_learner(self, force: bool = False) -> Dict[str, Any]:
        """Train or retrain the meta-learner if conditions are met."""
        buffer_stats = self.prediction_buffer.get_stats()

        # Check if we have enough data
        if buffer_stats["evaluated"] < self.min_samples_for_training:
            return {
                "trained": False,
                "reason": f"Not enough samples: {buffer_stats['evaluated']}/{self.min_samples_for_training}"
            }

        # Check if we need to retrain
        if not force and self.meta_learner.is_trained:
            if self.meta_learner.last_trained:
                hours_since = (datetime.now() - self.meta_learner.last_trained).total_seconds() / 3600
                if hours_since < self.retrain_interval_hours:
                    return {
                        "trained": False,
                        "reason": f"Recently trained {hours_since:.1f}h ago"
                    }

        # Get training data and train
        training_data = self.prediction_buffer.get_training_data()
        result = self.meta_learner.train(training_data)

        if result.get("success"):
            logger.info(f"[Ensemble] Meta-learner trained: {result}")
            # Save state
            self.save("/opt/wallex/models/ensemble_state.pkl")

        return {
            "trained": result.get("success", False),
            "result": result,
            "samples_used": len(training_data)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        return {
            "models": list(self.models.keys()),
            "current_weights": self._get_weights(),
            "default_weights": self.default_weights,
            "focus_bullish": self.focus_bullish,
            "min_confidence": self.min_confidence,
            "meta_learner_trained": self.meta_learner.is_trained,
            "meta_learner_last_trained": str(self.meta_learner.last_trained) if self.meta_learner.last_trained else None,
            "meta_learner_samples": self.meta_learner.training_samples,
            "prediction_buffer": self.prediction_buffer.get_stats(),
            "model_accuracies": {
                name: {
                    "overall": self.weight_tracker.get_accuracy(name),
                    "up": self.weight_tracker.get_accuracy(name, "UP"),
                    "down": self.weight_tracker.get_accuracy(name, "DOWN"),
                }
                for name in self.models
            }
        }

    def save(self, path: str):
        """Save ensemble state."""
        data = {
            "default_weights": self.default_weights,
            "focus_bullish": self.focus_bullish,
            "min_confidence": self.min_confidence,
            "weight_tracker_predictions": self.weight_tracker.predictions,
            "prediction_buffer": self.prediction_buffer.predictions,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

        # Save meta-learner separately
        meta_path = str(Path(path).parent / "meta_learner.pkl")
        self.meta_learner.save(meta_path)

    def load(self, path: str):
        """Load ensemble state."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.default_weights = data.get("default_weights", self.default_weights)
            self.focus_bullish = data.get("focus_bullish", True)
            self.min_confidence = data.get("min_confidence", 0.15)
            self.weight_tracker.predictions = data.get("weight_tracker_predictions", {})
            self.prediction_buffer.predictions = data.get("prediction_buffer", [])

            # Load meta-learner
            meta_path = str(Path(path).parent / "meta_learner.pkl")
            self.meta_learner.load(meta_path)

            if self.meta_learner.is_trained:
                logger.info(f"[Ensemble] Loaded trained meta-learner ({self.meta_learner.training_samples} samples)")
        except Exception as e:
            logger.warning(f"Could not load ensemble state: {e}")


# Singleton
_enhanced_ensemble = None

def get_enhanced_ensemble() -> EnhancedEnsembleCombiner:
    global _enhanced_ensemble
    if _enhanced_ensemble is None:
        _enhanced_ensemble = EnhancedEnsembleCombiner(focus_bullish=True)
        # Try to load saved state
        _enhanced_ensemble.load("/opt/wallex/models/ensemble_state.pkl")
    return _enhanced_ensemble
