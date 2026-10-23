"""
CoinGlass API Client

Provides access to cryptocurrency derivatives data:
- Funding rates
- Open interest
- Liquidations
- Long/short ratios
- Exchange data

Rate Limits:
- Paid plans only (starting at $35/month)
- Rate limit specified in API-KEY-MAX-LIMIT header
"""

import time
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import get_config

logger = logging.getLogger(__name__)


class CoinGlassClient:
    """
    Client for CoinGlass API.

    Provides derivatives market data including funding rates,
    open interest, liquidations, and more.
    """

    BASE_URL = "https://open-api.coinglass.com/public/v2"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGlass client.

        Args:
            api_key: API key (uses config if not provided)
        """
        config = get_config()
        self.api_key = api_key or config.market_data.coinglass_api_key
        self.rate_limit = config.market_data.coinglass_rate_limit
        self._last_request = 0

        if not self.api_key:
            logger.warning("CoinGlass API key not configured")

        self.headers = {
            "coinglassSecret": self.api_key,
            "Accept": "application/json",
        }

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make API request."""
        if not self.api_key:
            raise ValueError("CoinGlass API key not configured")

        self._rate_limit_wait()

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data.get("success", False):
                error_msg = data.get("msg", "Unknown error")
                raise Exception(f"CoinGlass API Error: {error_msg}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGlass request failed: {e}")
            raise

    def get_funding_rates(
        self,
        symbol: str = "BTC",
    ) -> List[Dict]:
        """
        Get current funding rates across exchanges.

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')

        Returns:
            List of funding rate data by exchange
        """
        params = {"symbol": symbol}
        data = self._request("funding", params)
        return data.get("data", [])

    def get_funding_rate_chart(
        self,
        symbol: str = "BTC",
        time_type: str = "h8",  # h1, h4, h8, h12
    ) -> List[Dict]:
        """
        Get historical funding rate chart data.

        Args:
            symbol: Trading symbol
            time_type: Time interval (h1, h4, h8, h12)

        Returns:
            Historical funding rate data
        """
        params = {"symbol": symbol, "time_type": time_type}
        data = self._request("funding_chart", params)
        return data.get("data", [])

    def get_open_interest(
        self,
        symbol: str = "BTC",
    ) -> Dict[str, Any]:
        """
        Get open interest data across exchanges.

        Args:
            symbol: Trading symbol

        Returns:
            Open interest data
        """
        params = {"symbol": symbol}
        data = self._request("open_interest", params)
        return data.get("data", {})

    def get_open_interest_chart(
        self,
        symbol: str = "BTC",
        interval: str = "0",  # 0=5min, 1=15min, 2=30min, 3=1h, 4=4h, 5=12h
    ) -> List[Dict]:
        """
        Get historical open interest data.

        Args:
            symbol: Trading symbol
            interval: Time interval (0=5m, 1=15m, 2=30m, 3=1h, 4=4h, 5=12h)

        Returns:
            Historical open interest data
        """
        params = {"symbol": symbol, "interval": interval}
        data = self._request("open_interest_history", params)
        return data.get("data", [])

    def get_liquidations(
        self,
        symbol: str = "BTC",
    ) -> Dict[str, Any]:
        """
        Get liquidation data.

        Args:
            symbol: Trading symbol

        Returns:
            Liquidation data
        """
        params = {"symbol": symbol}
        data = self._request("liquidation_symbol", params)
        return data.get("data", {})

    def get_liquidation_chart(
        self,
        symbol: str = "BTC",
        time_type: str = "h1",  # m5, m15, m30, h1, h4, h12, h24
    ) -> List[Dict]:
        """
        Get historical liquidation data.

        Args:
            symbol: Trading symbol
            time_type: Time interval

        Returns:
            Historical liquidation data
        """
        params = {"symbol": symbol, "time_type": time_type}
        data = self._request("liquidation_chart", params)
        return data.get("data", [])

    def get_long_short_ratio(
        self,
        symbol: str = "BTC",
    ) -> Dict[str, Any]:
        """
        Get long/short ratio data.

        Args:
            symbol: Trading symbol

        Returns:
            Long/short ratio data
        """
        params = {"symbol": symbol}
        data = self._request("long_short", params)
        return data.get("data", {})

    def get_exchange_data(self) -> List[Dict]:
        """
        Get data for all exchanges.

        Returns:
            List of exchange data
        """
        data = self._request("exchange_list")
        return data.get("data", [])

    def get_market_summary(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get comprehensive derivatives market summary.

        Args:
            symbol: Trading symbol

        Returns:
            Summary with funding, OI, liquidations, ratios
        """
        summary = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Get funding rates
            funding = self.get_funding_rates(symbol)
            if funding:
                avg_funding = sum(f.get("rate", 0) for f in funding) / len(funding)
                summary["avg_funding_rate"] = avg_funding
                summary["funding_by_exchange"] = {
                    f.get("exchange"): f.get("rate") for f in funding
                }
        except Exception as e:
            logger.warning(f"Failed to get funding rates: {e}")

        try:
            # Get open interest
            oi = self.get_open_interest(symbol)
            summary["total_open_interest"] = oi.get("openInterest")
            summary["oi_change_24h"] = oi.get("h24Change")
        except Exception as e:
            logger.warning(f"Failed to get open interest: {e}")

        try:
            # Get long/short ratio
            ls_ratio = self.get_long_short_ratio(symbol)
            summary["long_short_ratio"] = ls_ratio.get("longRate")
        except Exception as e:
            logger.warning(f"Failed to get long/short ratio: {e}")

        return summary

    def get_funding_signal(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get trading signal based on funding rate analysis.

        High positive funding = overleveraged longs (bearish signal)
        High negative funding = overleveraged shorts (bullish signal)

        Args:
            symbol: Trading symbol

        Returns:
            Signal analysis
        """
        funding = self.get_funding_rates(symbol)
        if not funding:
            return {"signal": "neutral", "confidence": 0}

        avg_rate = sum(f.get("rate", 0) for f in funding) / len(funding)

        # Thresholds (funding rate is typically 0.01% = 0.0001)
        if avg_rate > 0.0005:  # Very positive (0.05%+)
            signal = "bearish"
            confidence = min(abs(avg_rate) / 0.001, 0.9)
        elif avg_rate < -0.0005:  # Very negative (-0.05%+)
            signal = "bullish"
            confidence = min(abs(avg_rate) / 0.001, 0.9)
        elif avg_rate > 0.0002:  # Moderately positive
            signal = "slightly_bearish"
            confidence = 0.3
        elif avg_rate < -0.0002:  # Moderately negative
            signal = "slightly_bullish"
            confidence = 0.3
        else:
            signal = "neutral"
            confidence = 0

        return {
            "signal": signal,
            "confidence": confidence,
            "avg_funding_rate": avg_rate,
            "interpretation": f"{'Longs' if avg_rate > 0 else 'Shorts'} paying {'shorts' if avg_rate > 0 else 'longs'}",
        }


def create_coinglass_client(
    api_key: Optional[str] = None,
) -> CoinGlassClient:
    """Create a CoinGlassClient instance."""
    return CoinGlassClient(api_key=api_key)


if __name__ == "__main__":
    print("Testing CoinGlass Client...")

    client = CoinGlassClient()

    if not client.api_key:
        print("Error: COINGLASS_API_KEY not set")
        exit(1)

    # Test funding rates
    print("\n1. Getting BTC funding rates...")
    try:
        funding = client.get_funding_rates("BTC")
        print(f"  Found {len(funding)} exchanges")
        for f in funding[:3]:
            print(f"  {f.get('exchange')}: {f.get('rate', 0)*100:.4f}%")
    except Exception as e:
        print(f"  Error: {e}")

    # Test open interest
    print("\n2. Getting BTC open interest...")
    try:
        oi = client.get_open_interest("BTC")
        print(f"  Total OI: ${oi.get('openInterest', 0):,.0f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test funding signal
    print("\n3. Getting funding signal...")
    try:
        signal = client.get_funding_signal("BTC")
        print(f"  Signal: {signal['signal']}")
        print(f"  Avg Funding: {signal['avg_funding_rate']*100:.4f}%")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nTest complete!")
