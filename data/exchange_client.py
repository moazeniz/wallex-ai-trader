"""
Exchange Client - Unified interface for cryptocurrency exchanges

Provides a unified interface for fetching market data from multiple exchanges
using the CCXT library. Supports:
- OHLCV (candlestick) data
- Order book depth
- Ticker information
- Multiple timeframes
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import ccxt
import ccxt.async_support as ccxt_async

from .cache import DataCache, get_cache


# Supported exchanges
ExchangeType = Literal["binance", "bybit", "coinbase"]

# Standard timeframes
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]


# Forex symbols (routed to ForexClient)
FOREX_SYMBOLS = ["USD/EUR", "EUR/USD", "USD/GBP", "GBP/USD", "USD/JPY", "JPY/USD"]

def is_forex_symbol(symbol: str) -> bool:
    """Check if a symbol is a forex pair."""
    return symbol in FOREX_SYMBOLS


@dataclass
class OHLCVData:
    """Container for OHLCV data with metadata."""
    symbol: str
    timeframe: str
    exchange: str
    data: pd.DataFrame
    last_updated: datetime

    def __repr__(self) -> str:
        return (
            f"OHLCVData(symbol={self.symbol}, timeframe={self.timeframe}, "
            f"exchange={self.exchange}, rows={len(self.data)})"
        )


class ExchangeClient:
    """
    Unified exchange client for cryptocurrency market data.

    Provides a consistent interface across multiple exchanges with:
    - Automatic rate limiting
    - Response caching
    - Error handling and retries
    - Async support
    """

    def __init__(
        self,
        exchange: ExchangeType = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        cache: Optional[DataCache] = None,
        rate_limit_ms: int = 100,
    ):
        """
        Initialize the exchange client.

        Args:
            exchange: Exchange to connect to ('binance', 'bybit', 'coinbase')
            api_key: API key (optional for public data)
            api_secret: API secret (optional for public data)
            cache: DataCache instance for caching responses
            rate_limit_ms: Minimum milliseconds between requests
        """
        self.exchange_id = exchange
        self.api_key = api_key
        self.api_secret = api_secret
        self.cache = cache or get_cache()
        self.rate_limit_ms = rate_limit_ms
        self._last_request_time = 0

        # Initialize synchronous client
        self._exchange = self._create_exchange(exchange, api_key, api_secret)

        # Async client (created lazily)
        self._async_exchange: Optional[ccxt_async.Exchange] = None

    def _create_exchange(
        self,
        exchange_id: str,
        api_key: Optional[str],
        api_secret: Optional[str]
    ) -> ccxt.Exchange:
        """Create a CCXT exchange instance."""
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unsupported exchange: {exchange_id}")

        config = {
            "enableRateLimit": True,
            "rateLimit": self.rate_limit_ms,
        }

        if api_key and api_secret:
            config["apiKey"] = api_key
            config["secret"] = api_secret

        return exchange_class(config)

    def _create_async_exchange(self) -> ccxt_async.Exchange:
        """Create an async CCXT exchange instance."""
        exchange_class = getattr(ccxt_async, self.exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unsupported exchange: {self.exchange_id}")

        config = {
            "enableRateLimit": True,
            "rateLimit": self.rate_limit_ms,
        }

        if self.api_key and self.api_secret:
            config["apiKey"] = self.api_key
            config["secret"] = self.api_secret

        return exchange_class(config)

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = (time.time() * 1000) - self._last_request_time
        if elapsed < self.rate_limit_ms:
            time.sleep((self.rate_limit_ms - elapsed) / 1000)
        self._last_request_time = time.time() * 1000

    # =========================================================================
    # OHLCV Data
    # =========================================================================

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        since: Optional[int] = None,
        use_cache: bool = True,
        cache_ttl: int = 60,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch (max depends on exchange)
            since: Start timestamp in milliseconds (optional)
            use_cache: Whether to use cached data
            cache_ttl: Cache time-to-live in seconds

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """

        # Route forex symbols to ForexClient
        if is_forex_symbol(symbol):
            try:
                from .forex_client import get_forex_client
                forex = get_forex_client()
                df = forex.fetch_ohlcv(symbol, timeframe, limit)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Forex fetch failed for {symbol}: {e}")
                return pd.DataFrame()

        # Check cache first
        if use_cache:
            cached = self.cache.get_ohlcv(
                self.exchange_id, symbol, timeframe, cache_ttl
            )
            if cached is not None:
                return cached

        # Rate limit
        self._rate_limit()

        try:
            # Fetch from exchange
            ohlcv = self._exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
                since=since
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Cache the result
            if use_cache:
                self.cache.set_ohlcv(
                    self.exchange_id, symbol, timeframe, df, cache_ttl
                )

            return df

        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error fetching OHLCV: {e}")
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error fetching OHLCV: {e}")

    def fetch_ohlcv_extended(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 30,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch extended OHLCV data spanning multiple API calls.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days of historical data
            use_cache: Whether to use cache

        Returns:
            DataFrame with extended historical data
        """
        # Calculate timeframe in milliseconds
        tf_map = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
            "1w": 7 * 24 * 60 * 60 * 1000,
        }

        tf_ms = tf_map.get(timeframe, 60 * 60 * 1000)
        now = int(time.time() * 1000)
        start = now - (days * 24 * 60 * 60 * 1000)

        all_data = []
        current = start
        limit = 1000  # Max per request for most exchanges

        while current < now:
            df = self.fetch_ohlcv(
                symbol,
                timeframe,
                limit=limit,
                since=current,
                use_cache=False  # Don't cache partial fetches
            )

            if df.empty:
                break

            all_data.append(df)

            # Move to next batch
            last_ts = int(df.index[-1].timestamp() * 1000)
            current = last_ts + tf_ms

            # Small delay to respect rate limits
            time.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        # Combine all data
        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined.sort_index(inplace=True)

        # Cache the combined result
        if use_cache:
            self.cache.set_ohlcv(
                self.exchange_id, symbol, timeframe, combined, 300
            )

        return combined

    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        limit: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple timeframes.

        Args:
            symbol: Trading pair
            timeframes: List of timeframes (defaults to ['15m', '1h', '4h', '1d'])
            limit: Number of candles per timeframe

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        if timeframes is None:
            timeframes = ["15m", "1h", "4h", "1d"]

        result = {}
        for tf in timeframes:
            result[tf] = self.fetch_ohlcv(symbol, tf, limit)

        return result

    # =========================================================================
    # Order Book
    # =========================================================================

    def fetch_orderbook(
        self,
        symbol: str,
        limit: int = 20,
        use_cache: bool = True,
        cache_ttl: int = 10,
    ) -> Dict[str, Any]:
        """
        Fetch order book depth.

        Args:
            symbol: Trading pair
            limit: Number of levels per side
            use_cache: Whether to use cache
            cache_ttl: Cache TTL in seconds

        Returns:
            Dictionary with 'bids', 'asks', 'timestamp', and computed metrics
        """
        # Check cache
        if use_cache:
            cached = self.cache.get_orderbook(self.exchange_id, symbol, cache_ttl)
            if cached is not None:
                return cached

        self._rate_limit()

        try:
            orderbook = self._exchange.fetch_order_book(symbol, limit=limit)

            # Compute additional metrics
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])

            result = {
                "symbol": symbol,
                "exchange": self.exchange_id,
                "timestamp": orderbook.get("timestamp"),
                "bids": bids,
                "asks": asks,
                "bid_price": bids[0][0] if bids else None,
                "ask_price": asks[0][0] if asks else None,
                "spread": (asks[0][0] - bids[0][0]) if (bids and asks) else None,
                "spread_pct": (
                    (asks[0][0] - bids[0][0]) / bids[0][0] * 100
                    if (bids and asks) else None
                ),
                "bid_volume": sum(b[1] for b in bids),
                "ask_volume": sum(a[1] for a in asks),
            }

            # Order flow imbalance
            total_vol = result["bid_volume"] + result["ask_volume"]
            if total_vol > 0:
                result["imbalance"] = (
                    (result["bid_volume"] - result["ask_volume"]) / total_vol
                )
            else:
                result["imbalance"] = 0

            if use_cache:
                self.cache.set_orderbook(self.exchange_id, symbol, result, cache_ttl)

            return result

        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error fetching orderbook: {e}")
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error fetching orderbook: {e}")

    # =========================================================================
    # Ticker
    # =========================================================================

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker information.

        Args:
            symbol: Trading pair

        Returns:
            Dictionary with price, volume, and change information
        """
        self._rate_limit()

        try:
            ticker = self._exchange.fetch_ticker(symbol)

            return {
                "symbol": symbol,
                "exchange": self.exchange_id,
                "timestamp": ticker.get("timestamp"),
                "last": ticker.get("last"),
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask"),
                "high": ticker.get("high"),
                "low": ticker.get("low"),
                "open": ticker.get("open"),
                "close": ticker.get("close"),
                "volume": ticker.get("baseVolume"),
                "quote_volume": ticker.get("quoteVolume"),
                "change": ticker.get("change"),
                "change_pct": ticker.get("percentage"),
                "vwap": ticker.get("vwap"),
            }

        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error fetching ticker: {e}")
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"Exchange error fetching ticker: {e}")

    def fetch_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch tickers for multiple symbols.

        Args:
            symbols: List of trading pairs

        Returns:
            Dictionary mapping symbol to ticker data
        """
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.fetch_ticker(symbol)
            except Exception as e:
                print(f"Error fetching ticker for {symbol}: {e}")
                result[symbol] = None
        return result

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_markets(self) -> Dict[str, Any]:
        """Get all available markets on the exchange."""
        self._rate_limit()
        return self._exchange.load_markets()

    def get_symbols(self, quote: str = "USDT") -> List[str]:
        """
        Get all symbols with a specific quote currency.

        Args:
            quote: Quote currency (e.g., 'USDT', 'BTC')

        Returns:
            List of trading pair symbols
        """
        markets = self.get_markets()
        return [
            symbol for symbol, market in markets.items()
            if market.get("quote") == quote and market.get("active", True)
        ]

    def check_connection(self) -> bool:
        """Test connection to the exchange."""
        try:
            self._exchange.fetch_time()
            return True
        except Exception:
            return False

    # =========================================================================
    # Async Methods
    # =========================================================================

    async def _get_async_exchange(self) -> ccxt_async.Exchange:
        """Get or create async exchange instance."""
        if self._async_exchange is None:
            self._async_exchange = self._create_async_exchange()
        return self._async_exchange

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """Async version of fetch_ohlcv."""
        exchange = await self._get_async_exchange()

        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            return df
        finally:
            pass  # Keep connection open for reuse

    async def fetch_multiple_async(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        limit: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple symbols concurrently."""
        exchange = await self._get_async_exchange()

        async def fetch_one(symbol: str) -> tuple:
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                return symbol, df
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                return symbol, pd.DataFrame()

        tasks = [fetch_one(s) for s in symbols]
        results = await asyncio.gather(*tasks)

        return dict(results)

    async def close_async(self) -> None:
        """Close async exchange connection."""
        if self._async_exchange is not None:
            await self._async_exchange.close()
            self._async_exchange = None

    def close(self) -> None:
        """Close all connections."""
        # Sync client doesn't need explicit closing
        pass


# Factory function for creating clients
def create_exchange_client(
    exchange: ExchangeType = "binance",
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> ExchangeClient:
    """
    Create an exchange client with configuration from environment.

    Args:
        exchange: Exchange name
        api_key: Optional API key (uses config if not provided)
        api_secret: Optional API secret (uses config if not provided)

    Returns:
        Configured ExchangeClient instance
    """
    # Try to import config
    try:
        from settings import get_config
        config = get_config()

        if api_key is None:
            if exchange == "binance":
                api_key = config.exchange.binance_api_key
                api_secret = config.exchange.binance_api_secret
            elif exchange == "bybit":
                api_key = config.exchange.bybit_api_key
                api_secret = config.exchange.bybit_api_secret
            elif exchange == "coinbase":
                api_key = config.exchange.coinbase_api_key
                api_secret = config.exchange.coinbase_api_secret

    except ImportError:
        pass  # Config not available, use provided credentials

    return ExchangeClient(
        exchange=exchange,
        api_key=api_key,
        api_secret=api_secret,
    )


if __name__ == "__main__":
    # Test the exchange client
    print("Testing ExchangeClient...")

    client = ExchangeClient(exchange="binance")

    # Test connection
    print(f"Connection OK: {client.check_connection()}")

    # Test OHLCV fetch
    print("\nFetching BTC/USDT 1h OHLCV...")
    df = client.fetch_ohlcv("BTC/USDT", "1h", limit=10)
    print(df.tail())

    # Test ticker
    print("\nFetching BTC/USDT ticker...")
    ticker = client.fetch_ticker("BTC/USDT")
    print(f"Price: ${ticker['last']:,.2f}")
    print(f"24h Change: {ticker['change_pct']:.2f}%")

    # Test orderbook
    print("\nFetching BTC/USDT orderbook...")
    ob = client.fetch_orderbook("BTC/USDT", limit=5)
    print(f"Spread: ${ob['spread']:.2f} ({ob['spread_pct']:.4f}%)")
    print(f"Imbalance: {ob['imbalance']:.2f}")

    print("\nAll tests passed!")
