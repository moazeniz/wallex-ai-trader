"""
CoinMarketCap API Client

Provides access to cryptocurrency market data:
- Price quotes
- Market cap rankings
- Historical data
- Global metrics

Rate Limits (Free Tier):
- 10,000 calls/month
- ~30 calls/minute
- Credit-based system (1 credit per ~100 data points)
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


class CoinMarketCapClient:
    """
    Client for CoinMarketCap API.

    Free tier: 10,000 calls/month, ~30 calls/min
    """

    BASE_URL = "https://pro-api.coinmarketcap.com/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinMarketCap client.

        Args:
            api_key: API key (uses config if not provided)
        """
        config = get_config()
        self.api_key = api_key or config.market_data.coinmarketcap_api_key
        self.rate_limit = config.market_data.coinmarketcap_rate_limit
        self._last_request = 0

        if not self.api_key:
            logger.warning("CoinMarketCap API key not configured")

        self.headers = {
            "X-CMC_PRO_API_KEY": self.api_key,
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
            raise ValueError("CoinMarketCap API key not configured")

        self._rate_limit_wait()

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status", {}).get("error_code", 0) != 0:
                error_msg = data.get("status", {}).get("error_message", "Unknown error")
                raise Exception(f"CMC API Error: {error_msg}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"CoinMarketCap request failed: {e}")
            raise

    def get_listings(
        self,
        start: int = 1,
        limit: int = 100,
        sort: str = "market_cap",
        sort_dir: str = "desc",
    ) -> List[Dict]:
        """
        Get latest cryptocurrency listings.

        Args:
            start: Start rank (1-based)
            limit: Number of results (max 5000)
            sort: Sort field (market_cap, price, etc.)
            sort_dir: Sort direction (asc/desc)

        Returns:
            List of cryptocurrency data
        """
        params = {
            "start": start,
            "limit": limit,
            "sort": sort,
            "sort_dir": sort_dir,
            "convert": "USD",
        }

        data = self._request("cryptocurrency/listings/latest", params)
        return data.get("data", [])

    def get_quotes(
        self,
        symbols: List[str],
    ) -> Dict[str, Dict]:
        """
        Get latest quotes for specific symbols.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])

        Returns:
            Dict mapping symbol to quote data
        """
        params = {
            "symbol": ",".join(symbols),
            "convert": "USD",
        }

        data = self._request("cryptocurrency/quotes/latest", params)
        return data.get("data", {})

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote for a single symbol."""
        quotes = self.get_quotes([symbol])
        return quotes.get(symbol)

    def get_global_metrics(self) -> Dict[str, Any]:
        """
        Get global cryptocurrency market metrics.

        Returns:
            Global market data (total market cap, BTC dominance, etc.)
        """
        data = self._request("global-metrics/quotes/latest")
        return data.get("data", {})

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current USD price for a symbol."""
        quote = self.get_quote(symbol)
        if quote:
            return quote.get("quote", {}).get("USD", {}).get("price")
        return None

    def get_market_cap(self, symbol: str) -> Optional[float]:
        """Get market cap for a symbol."""
        quote = self.get_quote(symbol)
        if quote:
            return quote.get("quote", {}).get("USD", {}).get("market_cap")
        return None

    def get_volume_24h(self, symbol: str) -> Optional[float]:
        """Get 24h trading volume for a symbol."""
        quote = self.get_quote(symbol)
        if quote:
            return quote.get("quote", {}).get("USD", {}).get("volume_24h")
        return None

    def get_percent_change(
        self,
        symbol: str,
        period: str = "24h",
    ) -> Optional[float]:
        """
        Get percent change for a period.

        Args:
            symbol: Token symbol
            period: 1h, 24h, 7d, 30d

        Returns:
            Percent change
        """
        quote = self.get_quote(symbol)
        if quote:
            key = f"percent_change_{period}"
            return quote.get("quote", {}).get("USD", {}).get(key)
        return None

    def get_top_gainers(self, limit: int = 10) -> List[Dict]:
        """Get top gaining cryptocurrencies (24h)."""
        listings = self.get_listings(limit=100, sort="percent_change_24h", sort_dir="desc")
        return listings[:limit]

    def get_top_losers(self, limit: int = 10) -> List[Dict]:
        """Get top losing cryptocurrencies (24h)."""
        listings = self.get_listings(limit=100, sort="percent_change_24h", sort_dir="asc")
        return listings[:limit]

    def get_market_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive market summary.

        Returns:
            Dict with global metrics and top movers
        """
        global_metrics = self.get_global_metrics()

        return {
            "total_market_cap": global_metrics.get("quote", {}).get("USD", {}).get("total_market_cap"),
            "total_volume_24h": global_metrics.get("quote", {}).get("USD", {}).get("total_volume_24h"),
            "btc_dominance": global_metrics.get("btc_dominance"),
            "eth_dominance": global_metrics.get("eth_dominance"),
            "active_cryptocurrencies": global_metrics.get("active_cryptocurrencies"),
            "active_exchanges": global_metrics.get("active_exchanges"),
            "timestamp": datetime.now().isoformat(),
        }


def create_coinmarketcap_client(
    api_key: Optional[str] = None,
) -> CoinMarketCapClient:
    """Create a CoinMarketCapClient instance."""
    return CoinMarketCapClient(api_key=api_key)


if __name__ == "__main__":
    print("Testing CoinMarketCap Client...")

    client = CoinMarketCapClient()

    if not client.api_key:
        print("Error: COINMARKETCAP_API_KEY not set")
        exit(1)

    # Test get price
    print("\n1. Getting BTC price...")
    price = client.get_price("BTC")
    if price:
        print(f"  BTC: ${price:,.2f}")

    # Test get quotes
    print("\n2. Getting multiple quotes...")
    quotes = client.get_quotes(["BTC", "ETH", "SOL"])
    for symbol, data in quotes.items():
        price = data.get("quote", {}).get("USD", {}).get("price", 0)
        print(f"  {symbol}: ${price:,.2f}")

    # Test global metrics
    print("\n3. Getting global metrics...")
    summary = client.get_market_summary()
    print(f"  Total Market Cap: ${summary['total_market_cap']:,.0f}")
    print(f"  BTC Dominance: {summary['btc_dominance']:.1f}%")

    print("\nTest complete!")
