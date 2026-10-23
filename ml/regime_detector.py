"""
Regime Detector

Detects market regimes (states) using multiple methods:
- Bull/Bear/Sideways classification
- Volatility regimes (Low/Medium/High)
- Trend strength regimes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path


class MarketRegime(Enum):
    """Market regime classifications."""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    SIDEWAYS = "sideways"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"


class VolatilityRegime(Enum):
    """Volatility regime classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RegimeState:
    """Current regime state."""
    market_regime: MarketRegime
    volatility_regime: VolatilityRegime
    trend_strength: float  # 0-100
    regime_confidence: float  # 0-1
    regime_duration: int  # Bars in current regime
    features: Dict[str, float]


class RegimeDetector:
    """
    Market regime detector.

    Classifies market conditions into distinct regimes
    for adaptive trading strategies.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        lookback: int = 20,
        random_state: int = 42,
    ):
        """
        Initialize the regime detector.

        Args:
            n_regimes: Number of market regimes to detect
            lookback: Lookback period for feature calculation
            random_state: Random seed
        """
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.random_state = random_state

        self._market_model: Optional[GaussianMixture] = None
        self._volatility_model: Optional[GaussianMixture] = None
        self._scaler: Optional[StandardScaler] = None

        self._regime_history: List[int] = []
        self._fitted = False

    # =========================================================================
    # Feature Engineering
    # =========================================================================

    def _calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for regime detection."""
        features = pd.DataFrame(index=df.index)

        # Returns at multiple horizons
        features['return_1'] = df['close'].pct_change(1)
        features['return_5'] = df['close'].pct_change(5)
        features['return_20'] = df['close'].pct_change(self.lookback)

        # Volatility
        features['volatility'] = features['return_1'].rolling(self.lookback).std()
        features['volatility_change'] = features['volatility'].pct_change(5)

        # Trend indicators
        sma_short = df['close'].rolling(self.lookback // 2).mean()
        sma_long = df['close'].rolling(self.lookback).mean()
        features['trend'] = (sma_short - sma_long) / sma_long

        # Price position
        rolling_high = df['high'].rolling(self.lookback).max()
        rolling_low = df['low'].rolling(self.lookback).min()
        features['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low)

        # Volume trend
        if 'volume' in df.columns:
            vol_ma = df['volume'].rolling(self.lookback).mean()
            features['volume_trend'] = df['volume'] / vol_ma

        # Higher highs / lower lows
        features['higher_high'] = (df['high'] > df['high'].shift(1)).rolling(5).sum() / 5
        features['lower_low'] = (df['low'] < df['low'].shift(1)).rolling(5).sum() / 5

        return features.dropna()

    def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for volatility regime detection."""
        features = pd.DataFrame(index=df.index)

        # Historical volatility
        returns = df['close'].pct_change()
        features['realized_vol'] = returns.rolling(self.lookback).std() * np.sqrt(252)

        # ATR-based volatility
        if all(col in df.columns for col in ['high', 'low', 'close']):
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            ], axis=1).max(axis=1)
            features['atr_pct'] = tr.rolling(self.lookback).mean() / df['close']

        # Volatility of volatility
        features['vol_of_vol'] = features['realized_vol'].rolling(self.lookback).std()

        # Range expansion
        daily_range = (df['high'] - df['low']) / df['close']
        features['range_ratio'] = daily_range / daily_range.rolling(self.lookback).mean()

        return features.dropna()

    # =========================================================================
    # Model Training
    # =========================================================================

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """
        Fit the regime detection models.

        Args:
            df: OHLCV DataFrame

        Returns:
            Self for chaining
        """
        # Market regime model
        market_features = self._calculate_regime_features(df)
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(market_features)

        self._market_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=self.random_state,
            n_init=5,
        )
        self._market_model.fit(scaled)

        # Volatility regime model
        vol_features = self._calculate_volatility_features(df)
        vol_scaler = StandardScaler()
        vol_scaled = vol_scaler.fit_transform(vol_features)

        self._volatility_model = GaussianMixture(
            n_components=4,  # Low, Medium, High, Extreme
            covariance_type='full',
            random_state=self.random_state,
            n_init=5,
        )
        self._volatility_model.fit(vol_scaled)

        self._fitted = True
        return self

    # =========================================================================
    # Regime Detection
    # =========================================================================

    def detect_market_regime(self, df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime.

        Args:
            df: OHLCV DataFrame

        Returns:
            Tuple of (regime, confidence)
        """
        # Rule-based detection (more interpretable)
        recent = df.tail(self.lookback)

        # Calculate trend metrics
        returns = recent['close'].pct_change().mean() * 100
        volatility = recent['close'].pct_change().std() * 100

        # Moving average trend
        if len(df) >= 50:
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            current_price = df['close'].iloc[-1]

            above_20 = current_price > sma_20
            above_50 = current_price > sma_50
            ma_bullish = sma_20 > sma_50
        else:
            above_20 = above_50 = ma_bullish = True

        # Determine regime
        if returns > 0.5 and above_20 and above_50 and ma_bullish:
            regime = MarketRegime.STRONG_BULL
            confidence = min(1.0, returns / 1.0)
        elif returns > 0.1 and (above_20 or above_50):
            regime = MarketRegime.BULL
            confidence = 0.7
        elif returns < -0.5 and not above_20 and not above_50:
            regime = MarketRegime.STRONG_BEAR
            confidence = min(1.0, abs(returns) / 1.0)
        elif returns < -0.1 and not (above_20 and above_50):
            regime = MarketRegime.BEAR
            confidence = 0.7
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 1.0 - abs(returns) / 0.5

        return regime, max(0.3, min(1.0, confidence))

    def detect_volatility_regime(self, df: pd.DataFrame) -> Tuple[VolatilityRegime, float]:
        """
        Detect current volatility regime.

        Args:
            df: OHLCV DataFrame

        Returns:
            Tuple of (regime, confidence)
        """
        # Calculate current volatility
        returns = df['close'].pct_change()
        current_vol = returns.tail(self.lookback).std() * np.sqrt(252) * 100

        # Historical percentiles
        historical_vol = returns.rolling(self.lookback).std() * np.sqrt(252) * 100
        vol_percentile = (historical_vol < current_vol).mean()

        # Classify
        if vol_percentile < 0.25:
            regime = VolatilityRegime.LOW
            confidence = 1.0 - vol_percentile * 2
        elif vol_percentile < 0.50:
            regime = VolatilityRegime.MEDIUM
            confidence = 0.7
        elif vol_percentile < 0.75:
            regime = VolatilityRegime.HIGH
            confidence = 0.7 + (vol_percentile - 0.5) * 0.6
        else:
            regime = VolatilityRegime.EXTREME
            confidence = 0.8 + (vol_percentile - 0.75) * 0.8

        return regime, max(0.3, min(1.0, confidence))

    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength score (0-100).

        Args:
            df: OHLCV DataFrame

        Returns:
            Trend strength score
        """
        score = 50  # Neutral starting point

        # Price vs MAs
        if len(df) >= 200:
            current = df['close'].iloc[-1]
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            sma_200 = df['close'].rolling(200).mean().iloc[-1]

            if current > sma_20:
                score += 10
            if current > sma_50:
                score += 10
            if current > sma_200:
                score += 10
            if sma_20 > sma_50:
                score += 10
            if sma_50 > sma_200:
                score += 10

            if current < sma_20:
                score -= 10
            if current < sma_50:
                score -= 10
            if current < sma_200:
                score -= 10

        # Recent momentum
        if len(df) >= 20:
            returns_5 = df['close'].pct_change(5).iloc[-1] * 100
            returns_20 = df['close'].pct_change(20).iloc[-1] * 100

            score += min(15, max(-15, returns_5 * 3))
            score += min(10, max(-10, returns_20 * 1))

        # ADX if available
        if 'adx_14' in df.columns:
            adx = df['adx_14'].iloc[-1]
            if adx > 25:
                score += 5
            if adx > 40:
                score += 5

        return max(0, min(100, score))

    def get_current_regime(self, df: pd.DataFrame) -> RegimeState:
        """
        Get complete current regime state.

        Args:
            df: OHLCV DataFrame

        Returns:
            RegimeState with all regime information
        """
        market_regime, market_conf = self.detect_market_regime(df)
        vol_regime, vol_conf = self.detect_volatility_regime(df)
        trend_strength = self.calculate_trend_strength(df)

        # Calculate regime duration
        duration = self._calculate_regime_duration(df, market_regime)

        # Aggregate features
        features = {
            'return_5d': df['close'].pct_change(5).iloc[-1] * 100 if len(df) >= 5 else 0,
            'return_20d': df['close'].pct_change(20).iloc[-1] * 100 if len(df) >= 20 else 0,
            'volatility': df['close'].pct_change().tail(20).std() * np.sqrt(252) * 100,
        }

        return RegimeState(
            market_regime=market_regime,
            volatility_regime=vol_regime,
            trend_strength=trend_strength,
            regime_confidence=(market_conf + vol_conf) / 2,
            regime_duration=duration,
            features=features,
        )

    def _calculate_regime_duration(
        self,
        df: pd.DataFrame,
        current_regime: MarketRegime,
    ) -> int:
        """Calculate how long current regime has been active."""
        if len(df) < self.lookback * 2:
            return 1

        duration = 0
        for i in range(len(df) - self.lookback, 0, -1):
            subset = df.iloc[i:i + self.lookback]
            regime, _ = self.detect_market_regime(subset)
            if regime == current_regime:
                duration += 1
            else:
                break

        return max(1, duration)

    def get_regime_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of regime analysis.

        Args:
            df: OHLCV DataFrame

        Returns:
            Dictionary with regime summary
        """
        state = self.get_current_regime(df)

        # Determine trading bias
        if state.market_regime in [MarketRegime.STRONG_BULL, MarketRegime.BULL]:
            bias = "bullish"
            bias_score = 70 + state.trend_strength * 0.3
        elif state.market_regime in [MarketRegime.STRONG_BEAR, MarketRegime.BEAR]:
            bias = "bearish"
            bias_score = 30 - state.trend_strength * 0.3
        else:
            bias = "neutral"
            bias_score = 50

        # Volatility assessment
        if state.volatility_regime == VolatilityRegime.EXTREME:
            vol_warning = "EXTREME - Reduce position sizes"
        elif state.volatility_regime == VolatilityRegime.HIGH:
            vol_warning = "HIGH - Use wider stops"
        else:
            vol_warning = None

        return {
            "market_regime": state.market_regime.value,
            "volatility_regime": state.volatility_regime.value,
            "trend_strength": state.trend_strength,
            "regime_confidence": state.regime_confidence,
            "regime_duration": state.regime_duration,
            "trading_bias": bias,
            "bias_score": max(0, min(100, bias_score)),
            "volatility_warning": vol_warning,
            "features": state.features,
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str) -> None:
        """Save models to disk."""
        data = {
            "market_model": self._market_model,
            "volatility_model": self._volatility_model,
            "scaler": self._scaler,
            "config": {
                "n_regimes": self.n_regimes,
                "lookback": self.lookback,
            },
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> "RegimeDetector":
        """Load models from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._market_model = data["market_model"]
        self._volatility_model = data["volatility_model"]
        self._scaler = data["scaler"]
        self.n_regimes = data["config"]["n_regimes"]
        self.lookback = data["config"]["lookback"]
        self._fitted = True

        return self


def create_regime_detector(
    n_regimes: int = 3,
    lookback: int = 20,
) -> RegimeDetector:
    """Create a RegimeDetector instance."""
    return RegimeDetector(n_regimes=n_regimes, lookback=lookback)


if __name__ == "__main__":
    print("Testing RegimeDetector...")

    # Create sample data with different regimes
    np.random.seed(42)
    n = 500

    # Bull market (0-150)
    bull = 100 + np.cumsum(np.random.randn(150) * 0.5 + 0.1)

    # Sideways (150-300)
    sideways = bull[-1] + np.cumsum(np.random.randn(150) * 0.3)

    # Bear market (300-400)
    bear = sideways[-1] + np.cumsum(np.random.randn(100) * 0.5 - 0.15)

    # Recovery (400-500)
    recovery = bear[-1] + np.cumsum(np.random.randn(100) * 0.4 + 0.08)

    close = np.concatenate([bull, sideways, bear, recovery])
    dates = pd.date_range("2024-01-01", periods=n, freq="1H")

    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    # Test detector
    detector = RegimeDetector(lookback=20)
    detector.fit(df)

    # Get current regime
    summary = detector.get_regime_summary(df)
    print(f"\nCurrent Market Regime: {summary['market_regime']}")
    print(f"Volatility Regime: {summary['volatility_regime']}")
    print(f"Trend Strength: {summary['trend_strength']:.1f}")
    print(f"Trading Bias: {summary['trading_bias']} ({summary['bias_score']:.1f})")

    if summary['volatility_warning']:
        print(f"Warning: {summary['volatility_warning']}")
