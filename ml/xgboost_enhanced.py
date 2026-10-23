"""
Enhanced XGBoost Direction Classifier

Uses all 50 enhanced features:
- Price features (normalized, returns, volatility)
- Technical indicators (RSI, MACD, BB, etc.)
- Market microstructure (funding, OI)
- Multi-timeframe (1h, 4h, 1d trends)
- Cross-asset (BTC correlation, dominance)
- Regime indicators

Multi-class output: UP/DOWN/NEUTRAL
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.preprocessing import RobustScaler
import pickle
from pathlib import Path

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class EnhancedXGBoostClassifier:
    """
    Enhanced XGBoost classifier using 50 features.

    Uses EnhancedFeatureExtractor for all features.
    Multi-class output: UP (0), DOWN (1), NEUTRAL (2)
    """

    CLASS_UP = 0
    CLASS_DOWN = 1
    CLASS_NEUTRAL = 2

    def __init__(
        self,
        forecast_horizon: int = 4,
        threshold: float = 0.005,
    ):
        self.forecast_horizon = forecast_horizon
        self.threshold = threshold

        self.model = None
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.scalers: Dict[str, RobustScaler] = {}
        self._feature_extractor = None

    @property
    def feature_extractor(self):
        if self._feature_extractor is None:
            try:
                from enhanced_features import get_feature_extractor
                self._feature_extractor = get_feature_extractor()
            except ImportError:
                try:
                    from ml.enhanced_features import get_feature_extractor
                    self._feature_extractor = get_feature_extractor()
                except:
                    self._feature_extractor = None
        return self._feature_extractor

    def _prepare_features(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        df_4h: pd.DataFrame = None,
        df_1d: pd.DataFrame = None,
        btc_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Extract enhanced features using EnhancedFeatureExtractor."""
        df = df.reset_index(drop=True)

        if self.feature_extractor is not None:
            features = self.feature_extractor.extract_all_features(
                df=df,
                symbol=symbol,
                df_4h=df_4h,
                df_1d=df_1d,
                btc_df=btc_df,
                include_funding=True
            )
            self.feature_names = features.columns.tolist()
        else:
            features = self._prepare_features_basic(df)

        return features

    def _prepare_features_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic feature extraction fallback."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df.get('high', close)
        low = df.get('low', close)
        volume = df.get('volume', pd.Series(1, index=df.index))

        # Returns
        for period in [1, 4, 12, 24]:
            features[f'return_{period}h'] = close.pct_change(period)

        # Volatility
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

        # Bollinger Bands
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['bb_position'] = (close - sma_20) / (2 * std_20 + 1e-10)

        # Price vs SMAs
        for period in [10, 20, 50]:
            sma = close.rolling(period).mean()
            features[f'price_vs_sma_{period}'] = (close - sma) / (sma + 1e-10)

        # Volume
        vol_ma = volume.rolling(20).mean()
        features['volume_ratio'] = volume / (vol_ma + 1e-10)
        features['volume_trend'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-10)

        # Range
        features['hl_range'] = (high - low) / (close + 1e-10)

        # Momentum
        features['momentum_12'] = close / close.shift(12) - 1
        features['momentum_24'] = close / close.shift(24) - 1

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        features['atr_ratio'] = tr.rolling(14).mean() / (close + 1e-10)

        # Distance from highs/lows
        features['dist_from_high_24'] = (close - high.rolling(24).max()) / (close + 1e-10)
        features['dist_from_low_24'] = (close - low.rolling(24).min()) / (close + 1e-10)

        self.feature_names = features.columns.tolist()
        return features.fillna(0)

    def _prepare_data(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        df_4h: pd.DataFrame = None,
        df_1d: pd.DataFrame = None,
        btc_df: pd.DataFrame = None
    ) -> tuple:
        """Prepare features and labels for training."""
        features = self._prepare_features(df, symbol, df_4h, df_1d, btc_df)

        # Ensure features and df are aligned
        df = df.reset_index(drop=True)
        features = features.reset_index(drop=True)

        # Create multi-class labels
        future_return = df['close'].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)

        labels = np.full(len(features), self.CLASS_NEUTRAL)
        labels[future_return > self.threshold] = self.CLASS_UP
        labels[future_return < -self.threshold] = self.CLASS_DOWN

        # Remove NaN rows
        valid_idx = ~(features.isna().any(axis=1) | future_return.isna())
        valid_idx_array = valid_idx.values

        return features.loc[valid_idx].values, labels[valid_idx_array]

    def fit(
        self,
        df: pd.DataFrame,
        df_4h: pd.DataFrame = None,
        df_1d: pd.DataFrame = None,
        btc_df: pd.DataFrame = None,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Train the enhanced XGBoost classifier."""

        if xgb is None:
            return {"error": "XGBoost not installed"}

        print("Preparing enhanced features for XGBoost...")

        # Handle multi-token data
        if 'token' in df.columns:
            tokens = df['token'].unique()
            all_X, all_y = [], []

            for token in tokens:
                token_df = df[df['token'] == token].copy()
                if len(token_df) < 100:
                    continue

                X, y = self._prepare_data(token_df, symbol=token)
                if len(X) > 0:
                    self.scalers[token] = RobustScaler()
                    X_scaled = self.scalers[token].fit_transform(X)
                    all_X.append(X_scaled)
                    all_y.append(y)

                    up_pct = (y == self.CLASS_UP).mean()
                    down_pct = (y == self.CLASS_DOWN).mean()
                    neutral_pct = (y == self.CLASS_NEUTRAL).mean()
                    print(f"  {token}: {len(X)} samples, UP={up_pct:.1%}, DOWN={down_pct:.1%}, NEUTRAL={neutral_pct:.1%}")

            X = np.vstack(all_X)
            y = np.concatenate(all_y)
        else:
            X, y = self._prepare_data(df, df_4h=df_4h, df_1d=df_1d, btc_df=btc_df)
            X = self.scaler.fit_transform(X)

        up_pct = (y == self.CLASS_UP).mean()
        down_pct = (y == self.CLASS_DOWN).mean()
        neutral_pct = (y == self.CLASS_NEUTRAL).mean()
        print(f"\nTotal: {len(X)} samples, {len(self.feature_names)} features")
        print(f"Class distribution: UP={up_pct:.1%}, DOWN={down_pct:.1%}, NEUTRAL={neutral_pct:.1%}")

        # Chronological split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Ensure labels are integers
        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)

        # Debug: Check data
        print(f"\nTraining data check:")
        print(f"  X_train: shape={X_train.shape}, dtype={X_train.dtype}")
        print(f"  X_train stats: min={X_train.min():.4f}, max={X_train.max():.4f}, mean={X_train.mean():.4f}")
        print(f"  y_train: shape={y_train.shape}, dtype={y_train.dtype}, unique={np.unique(y_train)}")
        print(f"  y_train distribution: {np.bincount(y_train, minlength=3)}")

        # Train XGBoost multi-class classifier
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            objective='multi:softprob',  # Use softprob for predict_proba
            num_class=3,
            eval_metric='mlogloss',
            early_stopping_rounds=30,
            tree_method='hist',
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True,
        )

        # Evaluate
        val_pred = self.model.predict(X_val)
        val_acc = np.mean(val_pred == y_val)

        # Directional accuracy (UP and DOWN only)
        directional_mask = (y_val != self.CLASS_NEUTRAL)
        if directional_mask.sum() > 0:
            directional_acc = np.mean(val_pred[directional_mask] == y_val[directional_mask])
        else:
            directional_acc = 0

        # UP accuracy
        up_mask = (y_val == self.CLASS_UP)
        up_acc = np.mean(val_pred[up_mask] == y_val[up_mask]) if up_mask.sum() > 0 else 0

        # Feature importance
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        print(f"\nValidation Accuracy: {val_acc:.3f}")
        print(f"Directional Accuracy: {directional_acc:.3f}")
        print(f"UP Accuracy: {up_acc:.3f}")
        print(f"\nTop 10 features:")
        for feat, imp in top_features:
            print(f"  {feat}: {imp:.4f}")

        return {
            "validation_accuracy": val_acc,
            "directional_accuracy": directional_acc,
            "up_accuracy": up_acc,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "best_iteration": self.model.best_iteration,
            "feature_count": len(self.feature_names),
            "top_features": top_features,
            "class_distribution": {
                "up": up_pct,
                "down": down_pct,
                "neutral": neutral_pct
            }
        }

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        token: str = None,
        df_4h: pd.DataFrame = None,
        df_1d: pd.DataFrame = None,
        btc_df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Predict direction with probabilities."""
        if token and not symbol:
            symbol = token

        if self.model is None:
            raise RuntimeError("Model not trained")

        features = self._prepare_features(df, symbol, df_4h, df_1d, btc_df)

        if len(features) < 1:
            return {
                "direction": "NEUTRAL",
                "probability_up": 0.33,
                "probability_down": 0.33,
                "probability_neutral": 0.34,
                "confidence": 0.0,
            }

        X = features.iloc[-1:].values

        if symbol and symbol in self.scalers:
            X = self.scalers[symbol].transform(X)
        else:
            X = self.scaler.transform(X)

        probs = self.model.predict_proba(X)[0]
        prob_up = probs[self.CLASS_UP]
        prob_down = probs[self.CLASS_DOWN]
        prob_neutral = probs[self.CLASS_NEUTRAL]

        pred_class = np.argmax(probs)

        if pred_class == self.CLASS_UP:
            direction = "UP"
            confidence = prob_up - max(prob_down, prob_neutral)
        elif pred_class == self.CLASS_DOWN:
            direction = "DOWN"
            confidence = prob_down - max(prob_up, prob_neutral)
        else:
            direction = "NEUTRAL"
            confidence = 0.0

        if max(prob_up, prob_down) < 0.4:
            direction = "NEUTRAL"
            confidence = 0.0

        return {
            "direction": direction,
            "probability_up": float(prob_up),
            "probability_down": float(prob_down),
            "probability_neutral": float(prob_neutral),
            "confidence": float(max(0, confidence)),
            "forecast_horizon": self.forecast_horizon,
        }

    def save(self, path: str) -> None:
        """Save model."""
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "scalers": self.scalers,
            "feature_names": self.feature_names,
            "config": {
                "forecast_horizon": self.forecast_horizon,
                "threshold": self.threshold,
            }
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved XGBoost model to {path}")

    def load(self, path: str) -> "EnhancedXGBoostClassifier":
        """Load model."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.scalers = data.get("scalers", {})
        self.feature_names = data["feature_names"]
        self.forecast_horizon = data["config"]["forecast_horizon"]
        self.threshold = data["config"].get("threshold", 0.005)

        return self
