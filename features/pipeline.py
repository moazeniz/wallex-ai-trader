"""
Feature Pipeline

Orchestrates feature generation from multiple sources:
- Technical indicators from price data
- On-chain features from blockchain data
- Combined feature sets for ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.exchange_client import ExchangeClient
from data.onchain_client import OnChainClient
from features.technical import TechnicalFeatures
from features.onchain import OnChainFeatures
from features.preprocessor import FeaturePreprocessor


@dataclass
class FeatureSet:
    """Container for a complete feature set."""
    symbol: str
    timestamp: datetime
    technical: pd.DataFrame
    onchain: Dict[str, float]
    combined: pd.DataFrame
    metadata: Dict[str, Any]


class FeaturePipeline:
    """
    Unified feature generation pipeline.

    Combines technical and on-chain features for trading analysis.
    """

    def __init__(
        self,
        exchange: str = "binance",
        chains: Optional[List[str]] = None,
    ):
        """
        Initialize the feature pipeline.

        Args:
            exchange: Exchange for price data
            chains: Chains for on-chain data
        """
        self.exchange = exchange
        self.chains = chains or ["ethereum"]

        # Initialize components
        self.exchange_client = ExchangeClient(exchange=exchange)
        self.technical = TechnicalFeatures()
        self.onchain = OnChainFeatures(chains=chains)
        self.preprocessor = FeaturePreprocessor()

        # Cache
        self._cache: Dict[str, FeatureSet] = {}

    def generate_features(
        self,
        symbol: str,
        token_address: Optional[str] = None,
        chain: str = "ethereum",
        timeframe: str = "1h",
        limit: int = 500,
        include_onchain: bool = True,
    ) -> FeatureSet:
        """
        Generate complete feature set for a token.

        Args:
            symbol: Trading pair (e.g., 'AAVE/USDT')
            token_address: Token contract address for on-chain data
            chain: Chain for on-chain data
            timeframe: Timeframe for price data
            limit: Number of candles
            include_onchain: Whether to include on-chain features

        Returns:
            FeatureSet with all features
        """
        timestamp = datetime.now()

        # Get price data
        df = self.exchange_client.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )

        # Generate technical features
        df_technical = self.technical.add_all_features(df)

        # Generate on-chain features
        onchain_features = {}
        if include_onchain and token_address:
            try:
                onchain_features = self.onchain.get_all_features(
                    token_address=token_address,
                    chain=chain,
                )
            except Exception as e:
                print(f"Warning: Could not get on-chain features: {e}")

        # Combine features
        df_combined = self._combine_features(df_technical, onchain_features)

        # Create feature set
        feature_set = FeatureSet(
            symbol=symbol,
            timestamp=timestamp,
            technical=df_technical,
            onchain=onchain_features,
            combined=df_combined,
            metadata={
                "timeframe": timeframe,
                "limit": limit,
                "chain": chain,
                "token_address": token_address,
            },
        )

        # Cache
        cache_key = f"{symbol}_{timeframe}"
        self._cache[cache_key] = feature_set

        return feature_set

    def _combine_features(
        self,
        technical: pd.DataFrame,
        onchain: Dict[str, float],
    ) -> pd.DataFrame:
        """Combine technical and on-chain features."""
        df = technical.copy()

        # Add on-chain features as constants (same for all rows)
        for key, value in onchain.items():
            df[f"oc_{key}"] = value

        return df

    def generate_multi_timeframe(
        self,
        symbol: str,
        token_address: Optional[str] = None,
        chain: str = "ethereum",
        timeframes: Optional[List[str]] = None,
    ) -> Dict[str, FeatureSet]:
        """
        Generate features for multiple timeframes.

        Args:
            symbol: Trading pair
            token_address: Token contract address
            chain: Chain for on-chain data
            timeframes: List of timeframes

        Returns:
            Dictionary mapping timeframe to FeatureSet
        """
        if timeframes is None:
            timeframes = ["15m", "1h", "4h"]

        results = {}
        for tf in timeframes:
            # Only include on-chain for the first timeframe (to avoid redundant calls)
            include_onchain = (tf == timeframes[0])

            results[tf] = self.generate_features(
                symbol=symbol,
                token_address=token_address,
                chain=chain,
                timeframe=tf,
                include_onchain=include_onchain,
            )

        return results

    def get_latest_features(
        self,
        symbol: str,
        token_address: Optional[str] = None,
        chain: str = "ethereum",
        timeframe: str = "1h",
    ) -> Dict[str, float]:
        """
        Get the latest feature values (last row).

        Args:
            symbol: Trading pair
            token_address: Token contract address
            chain: Chain for on-chain data
            timeframe: Timeframe

        Returns:
            Dictionary of latest feature values
        """
        feature_set = self.generate_features(
            symbol=symbol,
            token_address=token_address,
            chain=chain,
            timeframe=timeframe,
        )

        # Get last row
        last_row = feature_set.combined.iloc[-1]

        # Convert to dict, handling NaN
        features = {}
        for col, val in last_row.items():
            if pd.isna(val):
                features[col] = 0.0
            elif isinstance(val, (int, float, np.integer, np.floating)):
                features[col] = float(val)
            else:
                features[col] = val

        return features

    def get_feature_summary(
        self,
        symbol: str,
        token_address: Optional[str] = None,
        chain: str = "ethereum",
    ) -> Dict[str, Any]:
        """
        Get a summary of key features for quick analysis.

        Args:
            symbol: Trading pair
            token_address: Token contract address
            chain: Chain

        Returns:
            Dictionary with key metrics summary
        """
        features = self.get_latest_features(
            symbol=symbol,
            token_address=token_address,
            chain=chain,
        )

        summary = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),

            # Price metrics
            "price": features.get("close", 0),
            "return_1h": features.get("return_1", 0) * 100,
            "return_24h": features.get("return_20", 0) * 100,  # Approximate

            # Momentum
            "rsi_14": features.get("rsi_14", 50),
            "macd": features.get("macd", 0),
            "macd_signal": features.get("macd_signal", 0),

            # Trend
            "trend_score": features.get("trend_score", 0),
            "above_sma_200": features.get("above_sma_200", 0),

            # Volatility
            "atr_14": features.get("atr_14", 0),
            "bb_pct": features.get("bb_pct_20", 0.5),

            # Volume
            "volume_ratio": features.get("volume_ratio", 1),
            "mfi": features.get("mfi_14", 50),

            # On-chain (if available)
            "whale_accumulating": features.get("oc_whale_whale_accumulating", 0),
            "exchange_flow": features.get("oc_flow_is_accumulating", 0),
            "onchain_score": features.get("oc_overall_onchain_score", 50),
            "onchain_sentiment": features.get("oc_onchain_sentiment", "neutral"),

            # Signals
            "signals": self._generate_signals(features),
        }

        return summary

    def _generate_signals(
        self,
        features: Dict[str, float],
    ) -> Dict[str, str]:
        """Generate trading signals from features."""
        signals = {}

        # RSI signals
        rsi = features.get("rsi_14", 50)
        if rsi < 30:
            signals["rsi"] = "oversold"
        elif rsi > 70:
            signals["rsi"] = "overbought"
        else:
            signals["rsi"] = "neutral"

        # MACD signals
        macd = features.get("macd", 0)
        macd_signal = features.get("macd_signal", 0)
        if macd > macd_signal:
            signals["macd"] = "bullish"
        else:
            signals["macd"] = "bearish"

        # Trend signals
        trend_score = features.get("trend_score", 0)
        if trend_score >= 3:
            signals["trend"] = "strong_uptrend"
        elif trend_score >= 2:
            signals["trend"] = "uptrend"
        elif trend_score <= 0:
            signals["trend"] = "downtrend"
        else:
            signals["trend"] = "neutral"

        # Bollinger Band signals
        bb_pct = features.get("bb_pct_20", 0.5)
        if bb_pct < 0.2:
            signals["bb"] = "near_lower"
        elif bb_pct > 0.8:
            signals["bb"] = "near_upper"
        else:
            signals["bb"] = "middle"

        # Volume signals
        volume_ratio = features.get("volume_ratio", 1)
        if volume_ratio > 2:
            signals["volume"] = "high_volume"
        elif volume_ratio < 0.5:
            signals["volume"] = "low_volume"
        else:
            signals["volume"] = "normal"

        # On-chain signals
        whale_acc = features.get("oc_whale_whale_accumulating", 0)
        flow_acc = features.get("oc_flow_is_accumulating", 0)
        if whale_acc and flow_acc:
            signals["onchain"] = "strong_accumulation"
        elif whale_acc or flow_acc:
            signals["onchain"] = "accumulation"
        else:
            signals["onchain"] = "distribution"

        return signals

    def get_cached(self, symbol: str, timeframe: str = "1h") -> Optional[FeatureSet]:
        """Get cached feature set if available."""
        cache_key = f"{symbol}_{timeframe}"
        return self._cache.get(cache_key)


def create_feature_pipeline(
    exchange: str = "binance",
    chains: Optional[List[str]] = None,
) -> FeaturePipeline:
    """Create a FeaturePipeline instance."""
    return FeaturePipeline(exchange=exchange, chains=chains)


if __name__ == "__main__":
    print("Testing FeaturePipeline...")

    pipeline = FeaturePipeline(exchange="binance", chains=["ethereum"])

    # Test with BTC (no on-chain since it's not ERC-20)
    print("\nGenerating features for BTC/USDT...")
    features = pipeline.generate_features(
        symbol="BTC/USDT",
        timeframe="1h",
        limit=100,
        include_onchain=False,
    )

    print(f"Technical features shape: {features.technical.shape}")
    print(f"Combined features shape: {features.combined.shape}")

    # Get summary
    summary = pipeline.get_feature_summary("BTC/USDT")
    print(f"\nSummary:")
    print(f"  Price: ${summary['price']:,.2f}")
    print(f"  RSI: {summary['rsi_14']:.1f} ({summary['signals']['rsi']})")
    print(f"  Trend: {summary['signals']['trend']}")
