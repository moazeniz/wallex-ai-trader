"""
XGBoost Direction Classifier

Tree-based model that uses engineered features directly.
Complements GRU by using a completely different approach.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class XGBoostDirectionClassifier:
    """
    XGBoost-based classifier for price direction.
    
    Uses engineered features (not sequences) for prediction.
    Good at finding non-linear relationships between indicators.
    """
    
    def __init__(
        self,
        lookback_periods: List[int] = [1, 4, 12, 24],  # Hours to look back
        forecast_horizon: int = 4,  # Predict 4 hours ahead
        threshold: float = 0.005,  # 0.5% minimum move
    ):
        self.lookback_periods = lookback_periods
        self.forecast_horizon = forecast_horizon
        self.threshold = threshold
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.scalers: Dict[str, StandardScaler] = {}
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from OHLCV data."""
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        high = df['high'] if 'high' in df.columns else close
        low = df['low'] if 'low' in df.columns else close
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        
        # Returns at different lookback periods
        for period in self.lookback_periods:
            features[f'return_{period}h'] = close.pct_change(period)
        
        # Volatility at different windows
        for window in [6, 12, 24]:
            features[f'volatility_{window}h'] = close.pct_change().rolling(window).std()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        features['macd'] = (ema_12 - ema_26) / close
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands position
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['bb_position'] = (close - sma_20) / (2 * std_20 + 1e-10)
        
        # Price relative to moving averages
        for period in [10, 20, 50]:
            sma = close.rolling(period).mean()
            features[f'price_vs_sma_{period}'] = (close - sma) / (sma + 1e-10)
        
        # Volume features
        vol_ma = volume.rolling(20).mean()
        features['volume_ratio'] = volume / (vol_ma + 1e-10)
        features['volume_trend'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-10)
        
        # High-Low range
        features['hl_range'] = (high - low) / (close + 1e-10)
        
        # Momentum
        features['momentum_12'] = close / close.shift(12) - 1
        features['momentum_24'] = close / close.shift(24) - 1
        
        # Trend strength (ADX-like)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        features['atr_ratio'] = tr.rolling(14).mean() / (close + 1e-10)
        
        # Recent highs/lows
        features['dist_from_high_24'] = (close - high.rolling(24).max()) / (close + 1e-10)
        features['dist_from_low_24'] = (close - low.rolling(24).min()) / (close + 1e-10)
        
        self.feature_names = features.columns.tolist()
        
        return features.fillna(0)
    
    def _prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare features and labels for training."""
        features = self._engineer_features(df)
        
        # Create labels: 1 if price goes up by threshold, else 0
        future_return = df['close'].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)
        labels = (future_return > self.threshold).astype(int)
        
        # Remove NaN rows
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        
        return features[valid_idx].values, labels[valid_idx].values
    
    def fit(
        self,
        df: pd.DataFrame,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Train the XGBoost classifier."""
        
        if xgb is None:
            return {"error": "XGBoost not installed"}
        
        # Check for multi-token data
        if 'token' in df.columns:
            tokens = df['token'].unique()
            all_X, all_y = [], []
            
            for token in tokens:
                token_df = df[df['token'] == token].copy()
                if len(token_df) < 100:
                    continue
                
                X, y = self._prepare_data(token_df)
                if len(X) > 0:
                    # Per-token scaling
                    self.scalers[token] = StandardScaler()
                    X_scaled = self.scalers[token].fit_transform(X)
                    all_X.append(X_scaled)
                    all_y.append(y)
                    print(f"  {token}: {len(X)} samples")
            
            X = np.vstack(all_X)
            y = np.concatenate(all_y)
        else:
            X, y = self._prepare_data(df)
            X = self.scaler.fit_transform(X)
        
        print(f"Total samples: {len(X)}, Class distribution: {np.mean(y):.2%} UP")
        
        # Chronological split (no shuffle for time series!)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=20,
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        # Evaluate
        val_pred = self.model.predict(X_val)
        val_acc = np.mean(val_pred == y_val)
        
        # Feature importance
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "validation_accuracy": val_acc,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "best_iteration": self.model.best_iteration,
            "top_features": top_features,
        }
    
    def predict(self, df: pd.DataFrame, token: str = None) -> Dict[str, Any]:
        """Predict direction for current market state."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        features = self._engineer_features(df)
        
        if len(features) < 1:
            return {
                "direction": "NEUTRAL",
                "probability_up": 0.5,
                "confidence": 0.0,
            }
        
        # Get last row of features
        X = features.iloc[-1:].values
        
        # Scale
        if token and token in self.scalers:
            X = self.scalers[token].transform(X)
        else:
            X = self.scaler.transform(X)
        
        # Predict
        prob_up = self.model.predict_proba(X)[0, 1]
        
        # Determine direction (using same thresholds as GRU)
        if prob_up > 0.55:
            direction = "UP"
            confidence = (prob_up - 0.5) * 2
        elif prob_up < 0.45:
            direction = "DOWN"
            confidence = (0.5 - prob_up) * 2
        else:
            direction = "NEUTRAL"
            confidence = 0.0
        
        return {
            "direction": direction,
            "probability_up": float(prob_up),
            "probability_down": float(1 - prob_up),
            "confidence": float(confidence),
        }
    
    def save(self, path: str) -> None:
        """Save model."""
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "scalers": self.scalers,
            "feature_names": self.feature_names,
            "config": {
                "lookback_periods": self.lookback_periods,
                "forecast_horizon": self.forecast_horizon,
                "threshold": self.threshold,
            }
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> "XGBoostDirectionClassifier":
        """Load model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.scalers = data.get("scalers", {})
        self.feature_names = data["feature_names"]
        self.lookback_periods = data["config"]["lookback_periods"]
        self.forecast_horizon = data["config"]["forecast_horizon"]
        self.threshold = data["config"]["threshold"]
        
        return self
