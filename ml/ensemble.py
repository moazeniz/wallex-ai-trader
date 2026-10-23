"""
Ensemble Combiner

Combines predictions from multiple models:
- GRU (sequence-based neural network)
- LSTM (alternative sequence-based neural network)
- XGBoost (feature-based tree model)

Uses weighted averaging based on model accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import pickle


class EnsembleCombiner:
    """
    Combines predictions from multiple models.
    
    Features:
    - Weighted averaging based on accuracy
    - Disagreement detection
    - Confidence calibration
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.accuracy_history: Dict[str, List[float]] = {}
        
        # Default weights (updated based on performance)
        self.default_weights = {
            "gru": 0.35,
            "lstm": 0.30,
            "xgboost": 0.35,
        }
    
    def register_model(self, name: str, model: Any, weight: float = None) -> None:
        """Register a model with the ensemble."""
        self.models[name] = model
        self.weights[name] = weight if weight else self.default_weights.get(name, 0.33)
        self.accuracy_history[name] = []
    
    def predict(self, df: pd.DataFrame, token: str = None, df_4h: pd.DataFrame = None, df_1d: pd.DataFrame = None, btc_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Get ensemble prediction from all models."""
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                # Pass multi-timeframe data if model supports it
                try:
                    pred = model.predict(df, token=token, df_4h=df_4h, df_1d=df_1d, btc_df=btc_df)
                except TypeError:
                    # Model doesn't support multi-timeframe params, use basic call
                    pred = model.predict(df, token=token) if token else model.predict(df)
                predictions[name] = pred
                probabilities[name] = pred.get("probability_up", 0.5)
            except Exception as e:
                print(f"Model {name} error: {e}")
                predictions[name] = {"direction": "NEUTRAL", "probability_up": 0.5}
                probabilities[name] = 0.5
        
        # Weighted average of probabilities
        weighted_prob = 0
        total_weight = 0
        
        for name, prob in probabilities.items():
            weight = self.weights.get(name, 0.33)
            weighted_prob += prob * weight
            total_weight += weight
        
        ensemble_prob = weighted_prob / total_weight if total_weight > 0 else 0.5
        
        # Determine ensemble direction
        if ensemble_prob > 0.55:
            direction = "UP"
            confidence = (ensemble_prob - 0.5) * 2
        elif ensemble_prob < 0.45:
            direction = "DOWN"
            confidence = (0.5 - ensemble_prob) * 2
        else:
            direction = "NEUTRAL"
            confidence = 0.0
        
        # Check for model disagreement
        directions = [p.get("direction", "NEUTRAL") for p in predictions.values()]
        unique_directions = set(d for d in directions if d != "NEUTRAL")
        
        disagreement = len(unique_directions) > 1  # Some say UP, others say DOWN
        agreement_ratio = max(directions.count(d) for d in set(directions)) / len(directions)
        
        # If strong disagreement, reduce confidence
        if disagreement:
            confidence *= 0.5
            if agreement_ratio < 0.5:
                direction = "NEUTRAL"
                confidence = 0.0
        
        return {
            "direction": direction,
            "probability_up": ensemble_prob,
            "probability_down": 1 - ensemble_prob,
            "confidence": confidence,
            "model_predictions": {
                name: {
                    "direction": pred.get("direction"),
                    "probability_up": pred.get("probability_up"),
                }
                for name, pred in predictions.items()
            },
            "disagreement": disagreement,
            "agreement_ratio": agreement_ratio,
            "weights_used": dict(self.weights),
        }
    
    def update_weights_from_accuracy(self, accuracy_by_model: Dict[str, float]) -> None:
        """
        Update model weights based on their accuracy.
        Higher accuracy = higher weight.
        """
        # Track accuracy
        for name, acc in accuracy_by_model.items():
            if name in self.accuracy_history:
                self.accuracy_history[name].append(acc)
                # Keep last 10 readings
                self.accuracy_history[name] = self.accuracy_history[name][-10:]
        
        # Calculate average accuracy per model
        avg_accuracy = {}
        for name in self.models:
            if name in self.accuracy_history and self.accuracy_history[name]:
                avg_accuracy[name] = np.mean(self.accuracy_history[name])
            else:
                avg_accuracy[name] = 0.5  # Default
        
        # Convert to weights (normalize to sum to 1)
        total_acc = sum(avg_accuracy.values())
        if total_acc > 0:
            for name in self.models:
                self.weights[name] = avg_accuracy[name] / total_acc
        
        print(f"Updated weights: {self.weights}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        return {
            "models": list(self.models.keys()),
            "weights": dict(self.weights),
            "accuracy_history": {
                name: list(hist) for name, hist in self.accuracy_history.items()
            },
        }
    
    def save(self, path: str) -> None:
        """Save ensemble state (weights, not models)."""
        data = {
            "weights": self.weights,
            "accuracy_history": self.accuracy_history,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> "EnsembleCombiner":
        """Load ensemble state."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.weights = data.get("weights", self.default_weights)
            self.accuracy_history = data.get("accuracy_history", {})
        except:
            pass  # Use defaults if no saved state
        return self


# Singleton instance
_ensemble = None

def get_ensemble() -> EnsembleCombiner:
    global _ensemble
    if _ensemble is None:
        _ensemble = EnsembleCombiner()
    return _ensemble
