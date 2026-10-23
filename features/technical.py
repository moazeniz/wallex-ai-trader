"""
Technical Analysis Features

Calculates technical indicators for trading analysis:
- Momentum indicators (RSI, MACD, Stochastic)
- Trend indicators (SMA, EMA, ADX)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, MFI, VWAP)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import pandas_ta as ta


class TechnicalFeatures:
    """
    Technical analysis feature generator.

    Uses pandas-ta for indicator calculations with additional
    custom features for trading signals.
    """

    def __init__(self):
        """Initialize the technical features generator."""
        pass

    # =========================================================================
    # Momentum Indicators
    # =========================================================================

    def add_rsi(
        self,
        df: pd.DataFrame,
        length: int = 14,
        col: str = "close",
    ) -> pd.DataFrame:
        """Add RSI (Relative Strength Index)."""
        df[f"rsi_{length}"] = ta.rsi(df[col], length=length)
        return df

    def add_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        col: str = "close",
    ) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)."""
        macd = ta.macd(df[col], fast=fast, slow=slow, signal=signal)
        df["macd"] = macd[f"MACD_{fast}_{slow}_{signal}"]
        df["macd_signal"] = macd[f"MACDs_{fast}_{slow}_{signal}"]
        df["macd_hist"] = macd[f"MACDh_{fast}_{slow}_{signal}"]
        return df

    def add_stochastic(
        self,
        df: pd.DataFrame,
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3,
    ) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=k, d=d, smooth_k=smooth_k)
        df["stoch_k"] = stoch[f"STOCHk_{k}_{d}_{smooth_k}"]
        df["stoch_d"] = stoch[f"STOCHd_{k}_{d}_{smooth_k}"]
        return df

    def add_williams_r(
        self,
        df: pd.DataFrame,
        length: int = 14,
    ) -> pd.DataFrame:
        """Add Williams %R."""
        df[f"willr_{length}"] = ta.willr(df["high"], df["low"], df["close"], length=length)
        return df

    def add_cci(
        self,
        df: pd.DataFrame,
        length: int = 20,
    ) -> pd.DataFrame:
        """Add Commodity Channel Index."""
        df[f"cci_{length}"] = ta.cci(df["high"], df["low"], df["close"], length=length)
        return df

    def add_roc(
        self,
        df: pd.DataFrame,
        length: int = 10,
        col: str = "close",
    ) -> pd.DataFrame:
        """Add Rate of Change."""
        df[f"roc_{length}"] = ta.roc(df[col], length=length)
        return df

    # =========================================================================
    # Trend Indicators
    # =========================================================================

    def add_sma(
        self,
        df: pd.DataFrame,
        lengths: List[int] = [20, 50, 200],
        col: str = "close",
    ) -> pd.DataFrame:
        """Add Simple Moving Averages."""
        for length in lengths:
            df[f"sma_{length}"] = ta.sma(df[col], length=length)
        return df

    def add_ema(
        self,
        df: pd.DataFrame,
        lengths: List[int] = [9, 21, 55],
        col: str = "close",
    ) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        for length in lengths:
            df[f"ema_{length}"] = ta.ema(df[col], length=length)
        return df

    def add_adx(
        self,
        df: pd.DataFrame,
        length: int = 14,
    ) -> pd.DataFrame:
        """Add Average Directional Index."""
        adx = ta.adx(df["high"], df["low"], df["close"], length=length)
        df[f"adx_{length}"] = adx[f"ADX_{length}"]
        df[f"dmp_{length}"] = adx[f"DMP_{length}"]
        df[f"dmn_{length}"] = adx[f"DMN_{length}"]
        return df

    def add_supertrend(
        self,
        df: pd.DataFrame,
        length: int = 10,
        multiplier: float = 3.0,
    ) -> pd.DataFrame:
        """Add Supertrend indicator."""
        st = ta.supertrend(df["high"], df["low"], df["close"], length=length, multiplier=multiplier)
        df[f"supertrend_{length}"] = st[f"SUPERT_{length}_{multiplier}"]
        df[f"supertrend_dir_{length}"] = st[f"SUPERTd_{length}_{multiplier}"]
        return df

    # =========================================================================
    # Volatility Indicators
    # =========================================================================

    def add_bollinger_bands(
        self,
        df: pd.DataFrame,
        length: int = 20,
        std: float = 2.0,
        col: str = "close",
    ) -> pd.DataFrame:
        """Add Bollinger Bands."""
        bb = ta.bbands(df[col], length=length, std=std)
        df[f"bb_upper_{length}"] = bb[f"BBU_{length}_{std}"]
        df[f"bb_mid_{length}"] = bb[f"BBM_{length}_{std}"]
        df[f"bb_lower_{length}"] = bb[f"BBL_{length}_{std}"]
        df[f"bb_width_{length}"] = bb[f"BBB_{length}_{std}"]
        df[f"bb_pct_{length}"] = bb[f"BBP_{length}_{std}"]
        return df

    def add_atr(
        self,
        df: pd.DataFrame,
        length: int = 14,
    ) -> pd.DataFrame:
        """Add Average True Range."""
        df[f"atr_{length}"] = ta.atr(df["high"], df["low"], df["close"], length=length)
        return df

    def add_keltner_channels(
        self,
        df: pd.DataFrame,
        length: int = 20,
        multiplier: float = 2.0,
    ) -> pd.DataFrame:
        """Add Keltner Channels."""
        kc = ta.kc(df["high"], df["low"], df["close"], length=length, scalar=multiplier)
        df[f"kc_upper_{length}"] = kc[f"KCUe_{length}_{multiplier}"]
        df[f"kc_mid_{length}"] = kc[f"KCBe_{length}_{multiplier}"]
        df[f"kc_lower_{length}"] = kc[f"KCLe_{length}_{multiplier}"]
        return df

    def add_donchian(
        self,
        df: pd.DataFrame,
        length: int = 20,
    ) -> pd.DataFrame:
        """Add Donchian Channels."""
        dc = ta.donchian(df["high"], df["low"], lower_length=length, upper_length=length)
        df[f"dc_upper_{length}"] = dc[f"DCU_{length}_{length}"]
        df[f"dc_mid_{length}"] = dc[f"DCM_{length}_{length}"]
        df[f"dc_lower_{length}"] = dc[f"DCL_{length}_{length}"]
        return df

    # =========================================================================
    # Volume Indicators
    # =========================================================================

    def add_obv(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add On-Balance Volume."""
        df["obv"] = ta.obv(df["close"], df["volume"])
        return df

    def add_mfi(
        self,
        df: pd.DataFrame,
        length: int = 14,
    ) -> pd.DataFrame:
        """Add Money Flow Index."""
        df[f"mfi_{length}"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=length)
        return df

    def add_vwap(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add Volume Weighted Average Price."""
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        return df

    def add_ad(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add Accumulation/Distribution Line."""
        df["ad"] = ta.ad(df["high"], df["low"], df["close"], df["volume"])
        return df

    def add_cmf(
        self,
        df: pd.DataFrame,
        length: int = 20,
    ) -> pd.DataFrame:
        """Add Chaikin Money Flow."""
        df[f"cmf_{length}"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=length)
        return df

    # =========================================================================
    # Support/Resistance
    # =========================================================================

    def add_pivot_points(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add pivot points (traditional)."""
        # Calculate pivot points from previous candle
        df["pivot"] = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
        df["r1"] = 2 * df["pivot"] - df["low"].shift(1)
        df["s1"] = 2 * df["pivot"] - df["high"].shift(1)
        df["r2"] = df["pivot"] + (df["high"].shift(1) - df["low"].shift(1))
        df["s2"] = df["pivot"] - (df["high"].shift(1) - df["low"].shift(1))
        return df

    def find_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 20,
        num_levels: int = 3,
    ) -> Tuple[List[float], List[float]]:
        """
        Find support and resistance levels.

        Args:
            df: OHLCV DataFrame
            window: Window for finding local extrema
            num_levels: Number of levels to return

        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        highs = df["high"].values
        lows = df["low"].values

        # Find local maxima (resistance)
        resistance = []
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i - window : i + window + 1]):
                resistance.append(highs[i])

        # Find local minima (support)
        support = []
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i - window : i + window + 1]):
                support.append(lows[i])

        # Get most significant levels
        current_price = df["close"].iloc[-1]

        resistance = sorted(set([r for r in resistance if r > current_price]))[:num_levels]
        support = sorted(set([s for s in support if s < current_price]), reverse=True)[:num_levels]

        return support, resistance

    # =========================================================================
    # Custom Features
    # =========================================================================

    def add_price_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["return_10"] = df["close"].pct_change(10)
        df["return_20"] = df["close"].pct_change(20)

        # Log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # High-Low range
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]

        # Close position in range
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

        # Gap
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        return df

    def add_volume_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving averages
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # Volume trend
        df["volume_trend"] = df["volume"].pct_change(5)

        # Up/Down volume
        df["up_volume"] = np.where(df["close"] > df["open"], df["volume"], 0)
        df["down_volume"] = np.where(df["close"] < df["open"], df["volume"], 0)
        df["volume_imbalance"] = (df["up_volume"] - df["down_volume"]) / df["volume"]

        return df

    def add_trend_strength(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add trend strength features."""
        # Price vs moving averages
        if "sma_20" not in df.columns:
            df["sma_20"] = ta.sma(df["close"], length=20)
        if "sma_50" not in df.columns:
            df["sma_50"] = ta.sma(df["close"], length=50)
        if "sma_200" not in df.columns:
            df["sma_200"] = ta.sma(df["close"], length=200)

        df["above_sma_20"] = (df["close"] > df["sma_20"]).astype(int)
        df["above_sma_50"] = (df["close"] > df["sma_50"]).astype(int)
        df["above_sma_200"] = (df["close"] > df["sma_200"]).astype(int)

        # Trend score
        df["trend_score"] = df["above_sma_20"] + df["above_sma_50"] + df["above_sma_200"]

        # Distance from MAs
        df["dist_sma_20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
        df["dist_sma_50"] = (df["close"] - df["sma_50"]) / df["sma_50"]

        return df

    # =========================================================================
    # All Features
    # =========================================================================

    def add_all_features(
        self,
        df: pd.DataFrame,
        include_pivot: bool = True,
    ) -> pd.DataFrame:
        """
        Add all technical features to the DataFrame.

        Args:
            df: OHLCV DataFrame
            include_pivot: Whether to include pivot points

        Returns:
            DataFrame with all features added
        """
        # Make a copy
        df = df.copy()

        # Momentum
        self.add_rsi(df)
        self.add_macd(df)
        self.add_stochastic(df)
        self.add_williams_r(df)
        self.add_cci(df)
        self.add_roc(df)

        # Trend
        self.add_sma(df)
        self.add_ema(df)
        self.add_adx(df)

        # Volatility
        self.add_bollinger_bands(df)
        self.add_atr(df)

        # Volume
        self.add_obv(df)
        self.add_mfi(df)
        self.add_vwap(df)
        self.add_ad(df)
        self.add_cmf(df)

        # Pivot points
        if include_pivot:
            self.add_pivot_points(df)

        # Custom
        self.add_price_features(df)
        self.add_volume_features(df)
        self.add_trend_strength(df)

        return df


def create_technical_features() -> TechnicalFeatures:
    """Create a TechnicalFeatures instance."""
    return TechnicalFeatures()


if __name__ == "__main__":
    # Test technical features
    print("Testing TechnicalFeatures...")

    # Create sample data
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1H")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    # Add features
    tf = TechnicalFeatures()
    df = tf.add_all_features(df)

    print(f"Features added: {len(df.columns)}")
    print(f"Sample columns: {list(df.columns[:10])}")
    print(f"\nLast RSI: {df['rsi_14'].iloc[-1]:.2f}")
    print(f"Last MACD: {df['macd'].iloc[-1]:.4f}")
