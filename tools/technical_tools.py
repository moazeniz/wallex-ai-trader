"""
Technical Analysis Tools

LangChain-compatible tools for technical analysis.
These tools are used by the Technical Analyst Agent.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
from pydantic import BaseModel, Field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.exchange_client import ExchangeClient
from features.technical import TechnicalFeatures


class FetchOHLCVInput(BaseModel):
    """Input for fetching OHLCV data."""
    symbol: str = Field(description="Trading pair symbol (e.g., 'BTC/USDT', 'AAVE/USDT')")
    timeframe: str = Field(default="1h", description="Timeframe (e.g., '15m', '1h', '4h', '1d')")
    limit: int = Field(default=100, description="Number of candles to fetch")


class GetIndicatorsInput(BaseModel):
    """Input for getting technical indicators."""
    symbol: str = Field(description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Timeframe")
    indicators: List[str] = Field(
        default=["rsi", "macd", "bb"],
        description="List of indicators to calculate"
    )


class FindLevelsInput(BaseModel):
    """Input for finding support/resistance levels."""
    symbol: str = Field(description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Timeframe")
    num_levels: int = Field(default=3, description="Number of levels to find")


class TechnicalTools:
    """
    Technical analysis tools for LLM agents.

    Provides tools for fetching market data and calculating indicators.
    """

    def __init__(self, exchange: str = "binance"):
        """Initialize technical tools."""
        self.exchange_client = ExchangeClient(exchange=exchange)
        self.technical = TechnicalFeatures()
        self._cache: Dict[str, Any] = {}

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Fetch OHLCV (candlestick) data.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h')
            limit: Number of candles

        Returns:
            Dictionary with OHLCV data and summary
        """
        try:
            df = self.exchange_client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )

            # Cache for later use
            cache_key = f"{symbol}_{timeframe}"
            self._cache[cache_key] = df

            # Calculate summary
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest

            change_1 = (latest['close'] - prev['close']) / prev['close'] * 100
            change_24 = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100

            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": len(df),
                "latest": {
                    "open": float(latest['open']),
                    "high": float(latest['high']),
                    "low": float(latest['low']),
                    "close": float(latest['close']),
                    "volume": float(latest['volume']),
                },
                "summary": {
                    "current_price": float(latest['close']),
                    "change_1_period": round(change_1, 2),
                    "change_total": round(change_24, 2),
                    "high_period": float(df['high'].max()),
                    "low_period": float(df['low'].min()),
                    "avg_volume": float(df['volume'].mean()),
                },
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_indicators(
        self,
        symbol: str,
        timeframe: str = "1h",
        indicators: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate technical indicators.

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            indicators: List of indicator names

        Returns:
            Dictionary with indicator values
        """
        indicators = indicators or ["rsi", "macd", "bb", "sma", "atr"]

        try:
            # Get cached data or fetch
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self._cache:
                df = self._cache[cache_key]
            else:
                df = self.exchange_client.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=200,
                )
                self._cache[cache_key] = df

            # Add all features
            df = self.technical.add_all_features(df)

            # Get latest values
            latest = df.iloc[-1]
            result = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "indicators": {},
            }

            # Extract requested indicators
            if "rsi" in indicators:
                rsi = latest.get('rsi_14', None)
                result["indicators"]["rsi"] = {
                    "value": float(rsi) if pd.notna(rsi) else None,
                    "interpretation": self._interpret_rsi(rsi),
                }

            if "macd" in indicators:
                macd = latest.get('macd', None)
                macd_signal = latest.get('macd_signal', None)
                macd_hist = latest.get('macd_hist', None)
                result["indicators"]["macd"] = {
                    "macd": float(macd) if pd.notna(macd) else None,
                    "signal": float(macd_signal) if pd.notna(macd_signal) else None,
                    "histogram": float(macd_hist) if pd.notna(macd_hist) else None,
                    "interpretation": self._interpret_macd(macd, macd_signal),
                }

            if "bb" in indicators:
                bb_upper = latest.get('bb_upper_20', None)
                bb_mid = latest.get('bb_mid_20', None)
                bb_lower = latest.get('bb_lower_20', None)
                bb_pct = latest.get('bb_pct_20', None)
                result["indicators"]["bollinger_bands"] = {
                    "upper": float(bb_upper) if pd.notna(bb_upper) else None,
                    "middle": float(bb_mid) if pd.notna(bb_mid) else None,
                    "lower": float(bb_lower) if pd.notna(bb_lower) else None,
                    "percent_b": float(bb_pct) if pd.notna(bb_pct) else None,
                    "interpretation": self._interpret_bb(bb_pct),
                }

            if "sma" in indicators:
                sma_20 = latest.get('sma_20', None)
                sma_50 = latest.get('sma_50', None)
                sma_200 = latest.get('sma_200', None)
                close = latest['close']
                result["indicators"]["sma"] = {
                    "sma_20": float(sma_20) if pd.notna(sma_20) else None,
                    "sma_50": float(sma_50) if pd.notna(sma_50) else None,
                    "sma_200": float(sma_200) if pd.notna(sma_200) else None,
                    "price_vs_sma_20": "above" if close > sma_20 else "below" if pd.notna(sma_20) else None,
                    "price_vs_sma_50": "above" if close > sma_50 else "below" if pd.notna(sma_50) else None,
                    "price_vs_sma_200": "above" if close > sma_200 else "below" if pd.notna(sma_200) else None,
                }

            if "atr" in indicators:
                atr = latest.get('atr_14', None)
                close = latest['close']
                result["indicators"]["atr"] = {
                    "value": float(atr) if pd.notna(atr) else None,
                    "percent": float(atr / close * 100) if pd.notna(atr) else None,
                    "interpretation": self._interpret_atr(atr, close),
                }

            if "stochastic" in indicators:
                stoch_k = latest.get('stoch_k', None)
                stoch_d = latest.get('stoch_d', None)
                result["indicators"]["stochastic"] = {
                    "k": float(stoch_k) if pd.notna(stoch_k) else None,
                    "d": float(stoch_d) if pd.notna(stoch_d) else None,
                    "interpretation": self._interpret_stochastic(stoch_k, stoch_d),
                }

            if "adx" in indicators:
                adx = latest.get('adx_14', None)
                result["indicators"]["adx"] = {
                    "value": float(adx) if pd.notna(adx) else None,
                    "interpretation": self._interpret_adx(adx),
                }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def find_support_resistance(
        self,
        symbol: str,
        timeframe: str = "1h",
        num_levels: int = 3,
    ) -> Dict[str, Any]:
        """
        Find support and resistance levels.

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            num_levels: Number of levels to find

        Returns:
            Dictionary with support and resistance levels
        """
        try:
            # Get cached data or fetch
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self._cache:
                df = self._cache[cache_key]
            else:
                df = self.exchange_client.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=200,
                )
                self._cache[cache_key] = df

            support, resistance = self.technical.find_support_resistance(
                df, window=20, num_levels=num_levels
            )

            current_price = float(df['close'].iloc[-1])

            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": current_price,
                "support_levels": [round(s, 4) for s in support],
                "resistance_levels": [round(r, 4) for r in resistance],
                "nearest_support": round(support[0], 4) if support else None,
                "nearest_resistance": round(resistance[0], 4) if resistance else None,
                "support_distance_pct": round((current_price - support[0]) / current_price * 100, 2) if support else None,
                "resistance_distance_pct": round((resistance[0] - current_price) / current_price * 100, 2) if resistance else None,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_trend_summary(
        self,
        symbol: str,
        timeframe: str = "1h",
    ) -> Dict[str, Any]:
        """
        Get a comprehensive trend summary.

        Args:
            symbol: Trading pair
            timeframe: Timeframe

        Returns:
            Dictionary with trend analysis
        """
        try:
            # Fetch with features
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self._cache:
                df = self._cache[cache_key]
            else:
                df = self.exchange_client.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=200,
                )
                self._cache[cache_key] = df

            df = self.technical.add_all_features(df)
            latest = df.iloc[-1]

            # Trend score
            trend_score = latest.get('trend_score', 1.5)
            if trend_score >= 3:
                trend = "strong_uptrend"
            elif trend_score >= 2:
                trend = "uptrend"
            elif trend_score <= 0:
                trend = "downtrend"
            else:
                trend = "sideways"

            # Momentum
            rsi = latest.get('rsi_14', 50)
            if rsi > 70:
                momentum = "overbought"
            elif rsi < 30:
                momentum = "oversold"
            else:
                momentum = "neutral"

            # Volatility
            atr_pct = latest.get('atr_14', 0) / latest['close'] * 100 if latest['close'] > 0 else 0
            if atr_pct > 5:
                volatility = "high"
            elif atr_pct > 2:
                volatility = "medium"
            else:
                volatility = "low"

            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": float(latest['close']),
                "trend": trend,
                "trend_score": float(trend_score),
                "momentum": momentum,
                "rsi": float(rsi) if pd.notna(rsi) else None,
                "volatility": volatility,
                "atr_percent": round(atr_pct, 2),
                "above_sma_20": bool(latest.get('above_sma_20', 0)),
                "above_sma_50": bool(latest.get('above_sma_50', 0)),
                "above_sma_200": bool(latest.get('above_sma_200', 0)),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    # Interpretation helpers
    def _interpret_rsi(self, rsi: Optional[float]) -> str:
        if rsi is None or pd.isna(rsi):
            return "unknown"
        if rsi > 70:
            return "overbought - potential reversal or pullback"
        elif rsi > 60:
            return "bullish momentum"
        elif rsi < 30:
            return "oversold - potential bounce"
        elif rsi < 40:
            return "bearish momentum"
        return "neutral"

    def _interpret_macd(self, macd: Optional[float], signal: Optional[float]) -> str:
        if macd is None or signal is None:
            return "unknown"
        if pd.isna(macd) or pd.isna(signal):
            return "unknown"
        if macd > signal and macd > 0:
            return "bullish - above signal and zero line"
        elif macd > signal:
            return "bullish crossover"
        elif macd < signal and macd < 0:
            return "bearish - below signal and zero line"
        else:
            return "bearish crossover"

    def _interpret_bb(self, bb_pct: Optional[float]) -> str:
        if bb_pct is None or pd.isna(bb_pct):
            return "unknown"
        if bb_pct > 1:
            return "above upper band - overbought/breakout"
        elif bb_pct > 0.8:
            return "near upper band - potential resistance"
        elif bb_pct < 0:
            return "below lower band - oversold/breakdown"
        elif bb_pct < 0.2:
            return "near lower band - potential support"
        return "middle of bands - neutral"

    def _interpret_atr(self, atr: Optional[float], close: float) -> str:
        if atr is None or pd.isna(atr):
            return "unknown"
        pct = atr / close * 100
        if pct > 5:
            return "high volatility - use wider stops"
        elif pct > 2:
            return "moderate volatility"
        return "low volatility - tighter stops possible"

    def _interpret_stochastic(self, k: Optional[float], d: Optional[float]) -> str:
        if k is None or d is None:
            return "unknown"
        if pd.isna(k) or pd.isna(d):
            return "unknown"
        if k > 80 and d > 80:
            return "overbought"
        elif k < 20 and d < 20:
            return "oversold"
        elif k > d:
            return "bullish momentum"
        return "bearish momentum"

    def _interpret_adx(self, adx: Optional[float]) -> str:
        if adx is None or pd.isna(adx):
            return "unknown"
        if adx > 40:
            return "very strong trend"
        elif adx > 25:
            return "strong trend"
        elif adx > 20:
            return "developing trend"
        return "weak/no trend - ranging market"


def create_technical_tools(exchange: str = "binance") -> TechnicalTools:
    """Create TechnicalTools instance."""
    return TechnicalTools(exchange=exchange)


if __name__ == "__main__":
    print("Testing TechnicalTools...")

    tools = TechnicalTools(exchange="binance")

    # Test OHLCV fetch
    print("\n1. Fetching OHLCV for BTC/USDT...")
    result = tools.fetch_ohlcv("BTC/USDT", "1h", 50)
    if result["success"]:
        print(f"   Current price: ${result['summary']['current_price']:,.2f}")
        print(f"   Change: {result['summary']['change_1_period']:.2f}%")

    # Test indicators
    print("\n2. Getting indicators...")
    result = tools.get_indicators("BTC/USDT", "1h", ["rsi", "macd", "bb"])
    if result["success"]:
        print(f"   RSI: {result['indicators']['rsi']['value']:.1f} - {result['indicators']['rsi']['interpretation']}")
        print(f"   MACD: {result['indicators']['macd']['interpretation']}")

    # Test support/resistance
    print("\n3. Finding S/R levels...")
    result = tools.find_support_resistance("BTC/USDT", "1h")
    if result["success"]:
        print(f"   Support levels: {result['support_levels']}")
        print(f"   Resistance levels: {result['resistance_levels']}")

    # Test trend summary
    print("\n4. Getting trend summary...")
    result = tools.get_trend_summary("BTC/USDT", "1h")
    if result["success"]:
        print(f"   Trend: {result['trend']}")
        print(f"   Momentum: {result['momentum']}")
        print(f"   Volatility: {result['volatility']}")
