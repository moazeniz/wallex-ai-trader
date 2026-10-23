"""
Unit Tests for Exchange Client and Cache

Run with: pytest tests/test_exchange.py -v
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cache import DataCache, CacheEntry
from data.exchange_client import ExchangeClient, create_exchange_client


# =============================================================================
# Cache Tests
# =============================================================================

class TestDataCache:
    """Tests for the DataCache class."""

    def test_cache_initialization(self, tmp_path):
        """Test cache initializes correctly."""
        cache = DataCache(cache_dir=str(tmp_path), enabled=True)

        assert cache.enabled
        assert cache.cache_dir.exists()
        assert (cache.cache_dir / "ohlcv").exists()
        assert (cache.cache_dir / "orderbook").exists()

    def test_cache_disabled(self):
        """Test cache operations when disabled."""
        cache = DataCache(enabled=False)

        cache.set_memory("test", {"data": 1}, ttl=60)
        assert cache.get_memory("test") is None

    def test_memory_cache_set_get(self):
        """Test setting and getting from memory cache."""
        cache = DataCache(enabled=True)

        test_data = {"price": 50000, "volume": 100}
        cache.set_memory("btc_price", test_data, ttl=60)

        retrieved = cache.get_memory("btc_price")
        assert retrieved == test_data

    def test_memory_cache_expiration(self):
        """Test that expired cache entries return None."""
        cache = DataCache(enabled=True)

        # Set with very short TTL
        cache.set_memory("short_lived", {"data": 1}, ttl=0)

        # Should be expired immediately
        import time
        time.sleep(0.1)
        assert cache.get_memory("short_lived") is None

    def test_ohlcv_cache(self, tmp_path):
        """Test OHLCV data caching."""
        cache = DataCache(cache_dir=str(tmp_path), enabled=True)

        # Create test DataFrame
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [99, 100, 101],
            "close": [104, 105, 106],
            "volume": [1000, 1100, 1200]
        })

        # Cache it
        cache.set_ohlcv("binance", "BTC/USDT", "1h", df, ttl=60)

        # Retrieve it
        cached_df = cache.get_ohlcv("binance", "BTC/USDT", "1h", ttl=60)

        assert cached_df is not None
        assert len(cached_df) == 3
        assert list(cached_df.columns) == ["open", "high", "low", "close", "volume"]

    def test_cache_key_generation(self):
        """Test that different parameters generate different keys."""
        cache = DataCache(enabled=True)

        key1 = cache._generate_key("ohlcv", "binance", "BTC/USDT", "1h")
        key2 = cache._generate_key("ohlcv", "binance", "ETH/USDT", "1h")
        key3 = cache._generate_key("ohlcv", "binance", "BTC/USDT", "4h")

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_cache_stats(self, tmp_path):
        """Test cache statistics."""
        cache = DataCache(cache_dir=str(tmp_path), enabled=True)

        cache.set_memory("key1", {"data": 1}, ttl=60)
        cache.set_memory("key2", {"data": 2}, ttl=60)

        stats = cache.get_stats()

        assert stats["enabled"] is True
        assert stats["memory_entries"] == 2

    def test_clear_memory(self):
        """Test clearing memory cache."""
        cache = DataCache(enabled=True)

        cache.set_memory("key1", {"data": 1}, ttl=60)
        cache.set_memory("key2", {"data": 2}, ttl=60)

        cache.clear_memory()

        assert cache.get_memory("key1") is None
        assert cache.get_memory("key2") is None


# =============================================================================
# Exchange Client Tests
# =============================================================================

class TestExchangeClient:
    """Tests for the ExchangeClient class."""

    @pytest.fixture
    def mock_ccxt(self):
        """Create a mock CCXT exchange."""
        with patch("data.exchange_client.ccxt") as mock:
            mock_exchange = MagicMock()
            mock.binance.return_value = mock_exchange
            yield mock_exchange

    def test_client_initialization(self, mock_ccxt):
        """Test client initializes with default settings."""
        client = ExchangeClient(exchange="binance")

        assert client.exchange_id == "binance"
        assert client.cache is not None

    def test_fetch_ohlcv(self, mock_ccxt):
        """Test fetching OHLCV data."""
        # Mock response
        mock_ccxt.fetch_ohlcv.return_value = [
            [1609459200000, 29000, 29500, 28900, 29300, 1000],
            [1609462800000, 29300, 29800, 29200, 29700, 1100],
            [1609466400000, 29700, 30000, 29600, 29900, 1200],
        ]

        client = ExchangeClient(exchange="binance")
        client._exchange = mock_ccxt

        df = client.fetch_ohlcv("BTC/USDT", "1h", limit=3, use_cache=False)

        assert len(df) == 3
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_fetch_ohlcv_with_cache(self, mock_ccxt, tmp_path):
        """Test that caching works for OHLCV data."""
        mock_ccxt.fetch_ohlcv.return_value = [
            [1609459200000, 29000, 29500, 28900, 29300, 1000],
        ]

        cache = DataCache(cache_dir=str(tmp_path), enabled=True)
        client = ExchangeClient(exchange="binance", cache=cache)
        client._exchange = mock_ccxt

        # First call - should hit API
        df1 = client.fetch_ohlcv("BTC/USDT", "1h", limit=1, use_cache=True)

        # Second call - should hit cache
        df2 = client.fetch_ohlcv("BTC/USDT", "1h", limit=1, use_cache=True)

        # API should only be called once
        assert mock_ccxt.fetch_ohlcv.call_count == 1
        assert df1.equals(df2)

    def test_fetch_ticker(self, mock_ccxt):
        """Test fetching ticker data."""
        mock_ccxt.fetch_ticker.return_value = {
            "timestamp": 1609459200000,
            "last": 29500,
            "bid": 29490,
            "ask": 29510,
            "high": 30000,
            "low": 29000,
            "open": 29200,
            "close": 29500,
            "baseVolume": 10000,
            "quoteVolume": 295000000,
            "change": 300,
            "percentage": 1.03,
            "vwap": 29450,
        }

        client = ExchangeClient(exchange="binance")
        client._exchange = mock_ccxt

        ticker = client.fetch_ticker("BTC/USDT")

        assert ticker["symbol"] == "BTC/USDT"
        assert ticker["last"] == 29500
        assert ticker["bid"] == 29490
        assert ticker["ask"] == 29510

    def test_fetch_orderbook(self, mock_ccxt):
        """Test fetching order book."""
        mock_ccxt.fetch_order_book.return_value = {
            "timestamp": 1609459200000,
            "bids": [[29490, 10], [29480, 20], [29470, 30]],
            "asks": [[29510, 15], [29520, 25], [29530, 35]],
        }

        client = ExchangeClient(exchange="binance")
        client._exchange = mock_ccxt

        ob = client.fetch_orderbook("BTC/USDT", limit=3, use_cache=False)

        assert ob["symbol"] == "BTC/USDT"
        assert ob["bid_price"] == 29490
        assert ob["ask_price"] == 29510
        assert ob["spread"] == 20
        assert len(ob["bids"]) == 3
        assert len(ob["asks"]) == 3

    def test_orderbook_imbalance(self, mock_ccxt):
        """Test order book imbalance calculation."""
        # More bid volume than ask volume
        mock_ccxt.fetch_order_book.return_value = {
            "timestamp": 1609459200000,
            "bids": [[29490, 100]],
            "asks": [[29510, 50]],
        }

        client = ExchangeClient(exchange="binance")
        client._exchange = mock_ccxt

        ob = client.fetch_orderbook("BTC/USDT", use_cache=False)

        # Imbalance should be positive (more bids)
        assert ob["imbalance"] > 0
        assert ob["bid_volume"] == 100
        assert ob["ask_volume"] == 50

    def test_check_connection(self, mock_ccxt):
        """Test connection check."""
        mock_ccxt.fetch_time.return_value = 1609459200000

        client = ExchangeClient(exchange="binance")
        client._exchange = mock_ccxt

        assert client.check_connection() is True

    def test_check_connection_failure(self, mock_ccxt):
        """Test connection check failure."""
        mock_ccxt.fetch_time.side_effect = Exception("Connection failed")

        client = ExchangeClient(exchange="binance")
        client._exchange = mock_ccxt

        assert client.check_connection() is False

    def test_multi_timeframe_fetch(self, mock_ccxt):
        """Test fetching multiple timeframes."""
        mock_ccxt.fetch_ohlcv.return_value = [
            [1609459200000, 29000, 29500, 28900, 29300, 1000],
        ]

        client = ExchangeClient(exchange="binance")
        client._exchange = mock_ccxt

        result = client.fetch_multi_timeframe(
            "BTC/USDT",
            timeframes=["1h", "4h"],
            limit=1
        )

        assert "1h" in result
        assert "4h" in result
        assert mock_ccxt.fetch_ohlcv.call_count == 2


# =============================================================================
# Integration Tests (require real API - skip by default)
# =============================================================================

@pytest.mark.skip(reason="Integration test - requires API connection")
class TestExchangeClientIntegration:
    """Integration tests that require real API connection."""

    def test_real_connection(self):
        """Test real connection to Binance."""
        client = ExchangeClient(exchange="binance")
        assert client.check_connection() is True

    def test_real_ohlcv_fetch(self):
        """Test real OHLCV fetch."""
        client = ExchangeClient(exchange="binance")
        df = client.fetch_ohlcv("BTC/USDT", "1h", limit=10)

        assert len(df) == 10
        assert df["close"].iloc[-1] > 0

    def test_real_ticker(self):
        """Test real ticker fetch."""
        client = ExchangeClient(exchange="binance")
        ticker = client.fetch_ticker("BTC/USDT")

        assert ticker["last"] > 0
        assert ticker["volume"] > 0


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
