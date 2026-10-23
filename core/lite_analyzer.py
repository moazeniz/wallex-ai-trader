"""
Lite Analyzer - Ensemble Edition

Uses multiple ML models for prediction:
- GRU (sequence-based neural network)
- LSTM (alternative neural network)
- XGBoost (feature-based tree model)

Plus technical indicators and regime detection.
"""

import sys
sys.path.insert(0, "/opt/wallex")

from typing import Dict, Any, Optional
from datetime import datetime

from data.exchange_client import ExchangeClient
from ml.gru_enhanced import EnhancedDirectionPredictor as GRUPredictor
from ml.lstm_enhanced import EnhancedLSTMPredictor as LSTMPredictor
from ml.xgboost_enhanced import EnhancedXGBoostClassifier as XGBoostDirectionClassifier
from ml.ensemble_enhanced import EnhancedEnsembleCombiner as EnsembleCombiner
from ml.anomaly_detector import AnomalyDetector
from ml.regime_detector import RegimeDetector
from ml.pattern_matcher import PatternMatcher
from tools.technical_tools import TechnicalTools

from core.prediction_tracker import get_prediction_tracker


class LiteAnalyzer:
    """Ensemble-based analyzer with multiple ML models."""
    
    def __init__(self):
        self.client = ExchangeClient()
        self.tech_tools = TechnicalTools()
        
        # Initialize ensemble
        self.ensemble = EnsembleCombiner(focus_bullish=True)
        self._load_models()
        
        # Other ML models
        self.anomaly = AnomalyDetector()
        self.regime = RegimeDetector()
        self.pattern = PatternMatcher()
        self._ml_fitted = False
        
        # Prediction tracker
        try:
            self.tracker = get_prediction_tracker()
            self.tracking_enabled = True
        except Exception as e:
            print(f"Warning: Prediction tracking disabled: {e}")
            self.tracking_enabled = False
    
    def _load_models(self):
        """Load all ensemble models."""
        models_loaded = []
        
        # Load GRU
        try:
            gru = GRUPredictor()
            gru.load("/opt/wallex/models/gru_direction.pkl")
            self.ensemble.register_model("gru", gru, weight=0.35)
            models_loaded.append("GRU")
        except Exception as e:
            print(f"Warning: Could not load GRU: {e}")
        
        # Load LSTM
        try:
            lstm = LSTMPredictor()
            lstm.load("/opt/wallex/models/lstm_enhanced.pkl")
            self.ensemble.register_model("lstm", lstm, weight=0.30)
            models_loaded.append("LSTM")
        except Exception as e:
            print(f"Warning: Could not load LSTM: {e}")
        
        # Load XGBoost
        try:
            xgb = XGBoostDirectionClassifier()
            xgb.load("/opt/wallex/models/xgboost_enhanced.pkl")
            self.ensemble.register_model("xgboost", xgb, weight=0.35)
            models_loaded.append("XGBoost")
        except Exception as e:
            print(f"Warning: Could not load XGBoost: {e}")
        
        print(f"Ensemble models loaded: {models_loaded}")
    
    def _apply_confidence_adjustment(self, symbol: str, raw_confidence: float) -> float:
        """Apply learned confidence adjustment."""
        if not self.tracking_enabled:
            return raw_confidence
        try:
            factor = self.tracker.get_confidence_adjustment(symbol, "combined")
            return max(0, min(100, raw_confidence * factor))
        except:
            return raw_confidence
    
    def _get_volatility_penalty(self, regime_info: Dict[str, Any]) -> float:
        """Get confidence penalty based on volatility."""
        vol_regime = regime_info.get("volatility_regime", "medium")
        if vol_regime == "extreme":
            return 0.5
        elif vol_regime == "high":
            return 0.75
        return 1.0
    
    def _get_recent_streak(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        """Check recent prediction streak for circuit breaker."""
        if not self.tracking_enabled:
            return {"streak_type": "none", "streak_length": 0}
        try:
            import sqlite3
            conn = sqlite3.connect(self.tracker.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT was_correct FROM predictions
                WHERE symbol = ? AND was_correct IS NOT NULL
                ORDER BY timestamp DESC LIMIT ?
            """, (symbol, limit))
            results = [bool(row[0]) for row in cursor.fetchall()]
            conn.close()
            
            if not results:
                return {"streak_type": "none", "streak_length": 0}
            
            streak_value = results[0]
            streak_length = 1
            for r in results[1:]:
                if r == streak_value:
                    streak_length += 1
                else:
                    break
            
            return {
                "streak_type": "correct" if streak_value else "wrong",
                "streak_length": streak_length,
            }
        except:
            return {"streak_type": "none", "streak_length": 0}
    
    def analyze(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Run ensemble analysis on a symbol."""
        start = datetime.now()
        
        # Fetch primary data (1h)
        df = self.client.fetch_ohlcv(symbol, timeframe, limit=200, use_cache=False)
        
        if df is None or len(df) < 50:
            return {"error": "Insufficient data"}
        
        # Fetch multi-timeframe data for enhanced features
        df_4h = None
        df_1d = None
        btc_df = None
        try:
            df_4h = self.client.fetch_ohlcv(symbol, "4h", limit=100, use_cache=True)
            df_1d = self.client.fetch_ohlcv(symbol, "1d", limit=60, use_cache=True)
            if symbol != "BTC/USDT":
                btc_df = self.client.fetch_ohlcv("BTC/USDT", timeframe, limit=200, use_cache=True)
        except Exception as e:
            pass  # Continue with available data
        
        price = float(df["close"].iloc[-1])
        change_1h = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100
        change_24h = (df["close"].iloc[-1] / df["close"].iloc[-24] - 1) * 100 if len(df) >= 24 else 0
        
        # Technical indicators
        indicators = self.tech_tools.get_indicators(symbol, timeframe)
        
        # Fit other ML models if needed
        if not self._ml_fitted and len(df) >= 100:
            self.anomaly.fit(df)
            self.regime.fit(df)
            self.pattern.fit(df.iloc[:-20])
            self._ml_fitted = True
        
        # ENSEMBLE PREDICTION (main prediction)
        ensemble_result = self.ensemble.predict(df, token=symbol, df_4h=df_4h, df_1d=df_1d, btc_df=btc_df)
        
        # ML analysis
        anomaly_score = 0
        regime_info = {}
        pattern_info = {}
        
        if self._ml_fitted:
            try:
                anomaly_data = self.anomaly.get_latest_anomaly_score(df)
                anomaly_score = anomaly_data.get("combined_anomaly_score", 0)
            except:
                pass
            try:
                regime_info = self.regime.get_regime_summary(df)
            except:
                pass
            try:
                pattern_info = self.pattern.get_pattern_summary(df)
            except:
                pass
        
        # Technical score
        tech_score = 50
        if indicators and "indicators" in indicators:
            ind = indicators["indicators"]
            rsi = ind.get("rsi", {}).get("value", 50)
            if rsi < 30:
                tech_score += 20
            elif rsi > 70:
                tech_score -= 20
            macd = ind.get("macd", {})
            if macd.get("interpretation") == "bullish":
                tech_score += 15
            elif macd.get("interpretation") == "bearish":
                tech_score -= 15
        
        # Combine scores
        ensemble_prob = ensemble_result.get("probability_up", 0.5)
        ensemble_score = ensemble_prob * 100
        
        # Apply volatility penalty
        volatility_penalty = self._get_volatility_penalty(regime_info)
        ensemble_score_adjusted = 50 + (ensemble_score - 50) * volatility_penalty
        
        # Weighted combination: 60% ensemble, 25% technical, 15% regime/pattern
        regime_score = regime_info.get("bias_score", 50)
        pattern_score = pattern_info.get("bullish_probability", 0.5) * 100
        
        total_score = (
            ensemble_score_adjusted * 0.60 +
            tech_score * 0.25 +
            (regime_score * 0.10 + pattern_score * 0.05)
        )
        
        # Circuit breakers
        streak_info = self._get_recent_streak(symbol, 5)
        circuit_breaker = streak_info["streak_type"] == "wrong" and streak_info["streak_length"] >= 3
        
        # Model disagreement already handled in ensemble
        model_disagreement = ensemble_result.get("disagreement", False)
        
        force_neutral = False
        force_reason = None
        
        if circuit_breaker:
            force_neutral = True
            force_reason = f"losing_streak_{streak_info['streak_length']}"
        elif model_disagreement and ensemble_result.get("agreement_ratio", 1) < 0.5:
            force_neutral = True
            force_reason = "strong_model_disagreement"
        
        # Apply confidence adjustment
        total_score = self._apply_confidence_adjustment(symbol, total_score)
        
        # Determine signal
        if force_neutral:
            signal = "HOLD"
            strength = f"forced_neutral ({force_reason})"
            total_score = 50
        elif total_score >= 60:
            signal = "BUY"
            strength = "strong" if total_score >= 70 else "moderate"
        elif total_score <= 40:
            signal = "SELL"
            strength = "strong" if total_score <= 30 else "moderate"
        else:
            signal = "HOLD"
            strength = "neutral"
        
        elapsed = (datetime.now() - start).total_seconds()
        
        # Record prediction
        if self.tracking_enabled:
            try:
                if force_neutral:
                    predicted_direction = "NEUTRAL"
                elif total_score >= 55:
                    predicted_direction = "UP"
                elif total_score <= 45:
                    predicted_direction = "DOWN"
                else:
                    predicted_direction = "NEUTRAL"
                
                # Get individual model predictions for tracking
                model_preds = ensemble_result.get("model_predictions", {})
                
                self.tracker.record_prediction(
                    symbol=symbol,
                    predicted_direction=predicted_direction,
                    confidence=total_score,
                    price=price,
                    scores={
                        "gru_score": model_preds.get("gru", {}).get("probability_up", 0.5) * 100,
                        "gru_direction": model_preds.get("gru", {}).get("direction", "NEUTRAL"),
                        "lstm_score": model_preds.get("lstm", {}).get("probability_up", 0.5) * 100,
                        "xgboost_score": model_preds.get("xgboost", {}).get("probability_up", 0.5) * 100,
                        "technical_score": tech_score,
                        "regime_score": regime_score,
                        "ensemble_score": ensemble_score,
                    },
                    timeframe="4h"
                )
            except:
                pass
        
        return {
            "symbol": symbol,
            "price": price,
            "change_1h": change_1h,
            "change_24h": change_24h,
            "signal": signal,
            "strength": strength,
            "confidence": total_score,
            "ensemble": {
                "direction": ensemble_result.get("direction"),
                "probability_up": ensemble_result.get("probability_up"),
                "models": ensemble_result.get("model_predictions", {}),
                "disagreement": model_disagreement,
                "agreement_ratio": ensemble_result.get("agreement_ratio"),
            },
            "scores": {
                "ensemble": round(ensemble_score_adjusted, 1),
                "technical": round(tech_score, 1),
                "regime": round(regime_score, 1),
                "pattern": round(pattern_score, 1),
            },
            "regime": regime_info.get("market_regime", "unknown"),
            "volatility": regime_info.get("volatility_regime", "unknown"),
            "anomaly_score": round(anomaly_score, 2),
            "analysis_time": round(elapsed, 1),
            "adjustments": {
                "volatility_penalty": volatility_penalty,
                "circuit_breaker": circuit_breaker,
                "force_reason": force_reason,
            }
        }


# Singleton
_analyzer = None

def get_lite_analyzer() -> LiteAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = LiteAnalyzer()
    return _analyzer


if __name__ == "__main__":
    analyzer = LiteAnalyzer()
    result = analyzer.analyze("BTC/USDT")
    
    print("\n" + "=" * 60)
    print(f"ENSEMBLE ANALYSIS: {result['symbol']}")
    print("=" * 60)
    print(f"Price: ${result['price']:,.2f} ({result['change_24h']:+.1f}% 24h)")
    print(f"Signal: {result['signal']} ({result['strength']}) - {result['confidence']:.0f}%")
    
    print(f"\nEnsemble: {result['ensemble']['direction']} (prob UP: {result['ensemble']['probability_up']:.1%})")
    print(f"Model Predictions:")
    for name, pred in result['ensemble']['models'].items():
        print(f"  {name.upper()}: {pred['direction']} ({pred['probability_up']:.1%})")
    
    print(f"\nScores: {result['scores']}")
    print(f"Regime: {result['regime']} | Volatility: {result['volatility']}")
    print(f"Model Agreement: {result['ensemble']['agreement_ratio']:.0%}")
    print(f"\nAnalysis time: {result['analysis_time']}s")
