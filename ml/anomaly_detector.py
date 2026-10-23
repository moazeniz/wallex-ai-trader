"""
Anomaly Detector

Uses Isolation Forest for detecting unusual market conditions:
- Price anomalies
- Volume anomalies
- On-chain anomalies
- Multi-feature anomaly scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from datetime import datetime


class AnomalyDetector:
    """
    Anomaly detection using Isolation Forest.

    Identifies unusual market conditions that may signal
    trading opportunities or risks.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_features: float = 1.0,
        random_state: int = 42,
    ):
        """
        Initialize the anomaly detector.

        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees in the forest
            max_features: Features to draw for each tree
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

        # Models for different types of anomalies
        self._price_model: Optional[IsolationForest] = None
        self._volume_model: Optional[IsolationForest] = None
        self._combined_model: Optional[IsolationForest] = None

        # Scalers
        self._price_scaler: Optional[StandardScaler] = None
        self._volume_scaler: Optional[StandardScaler] = None
        self._combined_scaler: Optional[StandardScaler] = None

        # Feature columns
        self._price_features: List[str] = []
        self._volume_features: List[str] = []
        self._combined_features: List[str] = []

        self._fitted = False

    def _create_model(self) -> IsolationForest:
        """Create a new Isolation Forest model."""
        return IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )

    # =========================================================================
    # Feature Extraction
    # =========================================================================

    def _get_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract price-based features for anomaly detection."""
        features = pd.DataFrame(index=df.index)

        # Returns at different windows
        for window in [1, 5, 10, 20]:
            features[f"return_{window}"] = df["close"].pct_change(window)

        # Volatility
        features["volatility_5"] = df["close"].pct_change().rolling(5).std()
        features["volatility_20"] = df["close"].pct_change().rolling(20).std()

        # High-low range
        features["hl_range"] = (df["high"] - df["low"]) / df["close"]

        # Gap from previous close
        features["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # Distance from moving averages
        if "sma_20" in df.columns:
            features["dist_sma_20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
        else:
            features["dist_sma_20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).mean()

        # RSI extremes
        if "rsi_14" in df.columns:
            features["rsi_extreme"] = (df["rsi_14"] - 50).abs() / 50

        self._price_features = features.columns.tolist()
        return features.dropna()

    def _get_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volume-based features for anomaly detection."""
        features = pd.DataFrame(index=df.index)

        # Volume ratios
        vol_ma_20 = df["volume"].rolling(20).mean()
        features["volume_ratio"] = df["volume"] / vol_ma_20

        # Volume change
        features["volume_change"] = df["volume"].pct_change()
        features["volume_change_5"] = df["volume"].pct_change(5)

        # Volume trend
        features["volume_trend"] = df["volume"].rolling(5).mean() / vol_ma_20

        # Volume volatility
        features["volume_std"] = df["volume"].rolling(10).std() / vol_ma_20

        # Price-volume divergence
        price_up = (df["close"] > df["close"].shift(1)).astype(int)
        volume_up = (df["volume"] > df["volume"].shift(1)).astype(int)
        features["pv_divergence"] = (price_up != volume_up).astype(int)

        self._volume_features = features.columns.tolist()
        return features.dropna()

    def _get_combined_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract combined features for overall anomaly detection."""
        price_features = self._get_price_features(df)
        volume_features = self._get_volume_features(df)

        # Merge on index
        combined = price_features.join(volume_features, how="inner")

        self._combined_features = combined.columns.tolist()
        return combined.dropna()

    # =========================================================================
    # Model Training
    # =========================================================================

    def fit(
        self,
        df: pd.DataFrame,
        fit_price: bool = True,
        fit_volume: bool = True,
        fit_combined: bool = True,
    ) -> "AnomalyDetector":
        """
        Fit the anomaly detection models.

        Args:
            df: DataFrame with OHLCV data
            fit_price: Whether to fit price anomaly model
            fit_volume: Whether to fit volume anomaly model
            fit_combined: Whether to fit combined model

        Returns:
            Self for chaining
        """
        if fit_price:
            price_features = self._get_price_features(df)
            self._price_scaler = StandardScaler()
            scaled = self._price_scaler.fit_transform(price_features)
            self._price_model = self._create_model()
            self._price_model.fit(scaled)

        if fit_volume:
            volume_features = self._get_volume_features(df)
            self._volume_scaler = StandardScaler()
            scaled = self._volume_scaler.fit_transform(volume_features)
            self._volume_model = self._create_model()
            self._volume_model.fit(scaled)

        if fit_combined:
            combined_features = self._get_combined_features(df)
            self._combined_scaler = StandardScaler()
            scaled = self._combined_scaler.fit_transform(combined_features)
            self._combined_model = self._create_model()
            self._combined_model.fit(scaled)

        self._fitted = True
        return self

    # =========================================================================
    # Anomaly Detection
    # =========================================================================

    def detect_price_anomalies(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect price anomalies.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with anomaly labels and scores
        """
        if self._price_model is None:
            raise RuntimeError("Price model not fitted")

        features = self._get_price_features(df)
        scaled = self._price_scaler.transform(features)

        # Predict (-1 for anomaly, 1 for normal)
        predictions = self._price_model.predict(scaled)
        scores = self._price_model.decision_function(scaled)

        result = pd.DataFrame(index=features.index)
        result["is_price_anomaly"] = (predictions == -1).astype(int)
        result["price_anomaly_score"] = -scores  # Higher = more anomalous
        result["price_anomaly_score"] = (result["price_anomaly_score"] - result["price_anomaly_score"].min()) / \
                                         (result["price_anomaly_score"].max() - result["price_anomaly_score"].min())

        return result

    def detect_volume_anomalies(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect volume anomalies.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with anomaly labels and scores
        """
        if self._volume_model is None:
            raise RuntimeError("Volume model not fitted")

        features = self._get_volume_features(df)
        scaled = self._volume_scaler.transform(features)

        predictions = self._volume_model.predict(scaled)
        scores = self._volume_model.decision_function(scaled)

        result = pd.DataFrame(index=features.index)
        result["is_volume_anomaly"] = (predictions == -1).astype(int)
        result["volume_anomaly_score"] = -scores
        result["volume_anomaly_score"] = (result["volume_anomaly_score"] - result["volume_anomaly_score"].min()) / \
                                          (result["volume_anomaly_score"].max() - result["volume_anomaly_score"].min())

        return result

    def detect_combined_anomalies(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect combined price/volume anomalies.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with anomaly labels and scores
        """
        if self._combined_model is None:
            raise RuntimeError("Combined model not fitted")

        features = self._get_combined_features(df)
        scaled = self._combined_scaler.transform(features)

        predictions = self._combined_model.predict(scaled)
        scores = self._combined_model.decision_function(scaled)

        result = pd.DataFrame(index=features.index)
        result["is_anomaly"] = (predictions == -1).astype(int)
        result["anomaly_score"] = -scores
        result["anomaly_score"] = (result["anomaly_score"] - result["anomaly_score"].min()) / \
                                   (result["anomaly_score"].max() - result["anomaly_score"].min())

        return result

    def detect_all_anomalies(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run all anomaly detection models.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all anomaly results
        """
        results = []

        if self._price_model:
            results.append(self.detect_price_anomalies(df))

        if self._volume_model:
            results.append(self.detect_volume_anomalies(df))

        if self._combined_model:
            results.append(self.detect_combined_anomalies(df))

        if not results:
            raise RuntimeError("No models fitted")

        # Merge all results
        merged = results[0]
        for r in results[1:]:
            merged = merged.join(r, how="inner")

        return merged

    def get_latest_anomaly_score(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Get anomaly scores for the latest data point.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with latest anomaly scores
        """
        anomalies = self.detect_all_anomalies(df)
        last_row = anomalies.iloc[-1]

        return {
            "price_anomaly_score": float(last_row.get("price_anomaly_score", 0)),
            "volume_anomaly_score": float(last_row.get("volume_anomaly_score", 0)),
            "combined_anomaly_score": float(last_row.get("anomaly_score", 0)),
            "is_anomaly": bool(last_row.get("is_anomaly", 0)),
            "is_price_anomaly": bool(last_row.get("is_price_anomaly", 0)),
            "is_volume_anomaly": bool(last_row.get("is_volume_anomaly", 0)),
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str) -> None:
        """Save models to disk."""
        data = {
            "price_model": self._price_model,
            "volume_model": self._volume_model,
            "combined_model": self._combined_model,
            "price_scaler": self._price_scaler,
            "volume_scaler": self._volume_scaler,
            "combined_scaler": self._combined_scaler,
            "price_features": self._price_features,
            "volume_features": self._volume_features,
            "combined_features": self._combined_features,
            "config": {
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "max_features": self.max_features,
            },
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> "AnomalyDetector":
        """Load models from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._price_model = data["price_model"]
        self._volume_model = data["volume_model"]
        self._combined_model = data["combined_model"]
        self._price_scaler = data["price_scaler"]
        self._volume_scaler = data["volume_scaler"]
        self._combined_scaler = data["combined_scaler"]
        self._price_features = data["price_features"]
        self._volume_features = data["volume_features"]
        self._combined_features = data["combined_features"]
        self._fitted = True

        return self


def create_anomaly_detector(
    contamination: float = 0.1,
    n_estimators: int = 100,
) -> AnomalyDetector:
    """Create an AnomalyDetector instance."""
    return AnomalyDetector(
        contamination=contamination,
        n_estimators=n_estimators,
    )


if __name__ == "__main__":
    print("Testing AnomalyDetector...")

    # Create sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1H")

    # Normal price movement
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    # Add some anomalies
    close[100] += 10  # Price spike
    close[200] -= 8   # Price crash
    close[300] += 15  # Large spike

    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, n).astype(float)

    # Add volume anomalies
    volume[150] *= 10
    volume[250] *= 8

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    # Test detector
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(df)

    anomalies = detector.detect_all_anomalies(df)

    print(f"\nTotal anomalies detected: {anomalies['is_anomaly'].sum()}")
    print(f"Price anomalies: {anomalies['is_price_anomaly'].sum()}")
    print(f"Volume anomalies: {anomalies['is_volume_anomaly'].sum()}")

    # Get latest
    latest = detector.get_latest_anomaly_score(df)
    print(f"\nLatest anomaly score: {latest['combined_anomaly_score']:.3f}")
