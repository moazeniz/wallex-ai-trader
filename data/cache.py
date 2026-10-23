"""
Data Caching Layer

Provides caching for API responses to:
- Reduce API calls and avoid rate limits
- Improve response times for repeated queries
- Store data locally for offline analysis
"""

import os
import json
import time
import hashlib
import pickle
from typing import Any, Optional, Dict
from pathlib import Path
from dataclasses import dataclass
import pandas as pd


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    data: Any
    timestamp: float
    ttl: int  # Time to live in seconds

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > self.ttl


class DataCache:
    """
    File-based cache for API responses.

    Supports caching of:
    - OHLCV data (pandas DataFrames)
    - Order book snapshots
    - On-chain data
    - Any JSON-serializable data
    """

    def __init__(self, cache_dir: Optional[str] = None, enabled: bool = True):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory for cache files. Defaults to .cache in project root.
            enabled: Whether caching is enabled.
        """
        self.enabled = enabled

        if cache_dir is None:
            # Default to .cache directory in project root
            project_root = Path(__file__).parent.parent
            self.cache_dir = project_root / ".cache"
        else:
            self.cache_dir = Path(cache_dir)

        # Create cache directory if it doesn't exist
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "ohlcv").mkdir(exist_ok=True)
            (self.cache_dir / "orderbook").mkdir(exist_ok=True)
            (self.cache_dir / "onchain").mkdir(exist_ok=True)
            (self.cache_dir / "misc").mkdir(exist_ok=True)

        # In-memory cache for frequently accessed data
        self._memory_cache: Dict[str, CacheEntry] = {}

    def _generate_key(self, *args) -> str:
        """Generate a unique cache key from arguments."""
        key_str = ":".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, category: str, key: str, extension: str = "pkl") -> Path:
        """Get the file path for a cache entry."""
        return self.cache_dir / category / f"{key}.{extension}"

    # =========================================================================
    # Memory Cache Operations
    # =========================================================================

    def get_memory(self, key: str) -> Optional[Any]:
        """Get data from in-memory cache."""
        if not self.enabled:
            return None

        entry = self._memory_cache.get(key)
        if entry is None:
            return None

        if entry.is_expired():
            del self._memory_cache[key]
            return None

        return entry.data

    def set_memory(self, key: str, data: Any, ttl: int = 60) -> None:
        """Set data in in-memory cache."""
        if not self.enabled:
            return

        self._memory_cache[key] = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl=ttl
        )

    # =========================================================================
    # File Cache Operations
    # =========================================================================

    def get_file(self, category: str, key: str) -> Optional[Any]:
        """Get data from file cache."""
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(category, key)
        meta_path = self._get_cache_path(category, f"{key}_meta", "json")

        if not cache_path.exists() or not meta_path.exists():
            return None

        # Check metadata for expiration
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            if time.time() - meta["timestamp"] > meta["ttl"]:
                # Expired, remove files
                cache_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return None

            # Load cached data
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        except Exception:
            return None

    def set_file(self, category: str, key: str, data: Any, ttl: int = 3600) -> None:
        """Set data in file cache."""
        if not self.enabled:
            return

        cache_path = self._get_cache_path(category, key)
        meta_path = self._get_cache_path(category, f"{key}_meta", "json")

        try:
            # Save data
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

            # Save metadata
            with open(meta_path, "w") as f:
                json.dump({
                    "timestamp": time.time(),
                    "ttl": ttl
                }, f)

        except Exception as e:
            print(f"Cache write error: {e}")

    # =========================================================================
    # OHLCV Cache Operations
    # =========================================================================

    def get_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        ttl: int = 60
    ) -> Optional[pd.DataFrame]:
        """
        Get cached OHLCV data.

        Args:
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h')
            ttl: Time to live in seconds

        Returns:
            Cached DataFrame or None if not found/expired
        """
        key = self._generate_key("ohlcv", exchange, symbol, timeframe)

        # Try memory cache first
        data = self.get_memory(key)
        if data is not None:
            return data

        # Try file cache
        data = self.get_file("ohlcv", key)
        if data is not None:
            # Populate memory cache
            self.set_memory(key, data, ttl)
            return data

        return None

    def set_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        ttl: int = 60
    ) -> None:
        """
        Cache OHLCV data.

        Args:
            exchange: Exchange name
            symbol: Trading pair
            timeframe: Candle timeframe
            data: OHLCV DataFrame
            ttl: Time to live in seconds
        """
        key = self._generate_key("ohlcv", exchange, symbol, timeframe)

        # Set in both caches
        self.set_memory(key, data, ttl)
        self.set_file("ohlcv", key, data, ttl)

    # =========================================================================
    # Order Book Cache Operations
    # =========================================================================

    def get_orderbook(
        self,
        exchange: str,
        symbol: str,
        ttl: int = 10
    ) -> Optional[Dict]:
        """Get cached order book."""
        key = self._generate_key("orderbook", exchange, symbol)
        return self.get_memory(key)  # Order books only in memory (short TTL)

    def set_orderbook(
        self,
        exchange: str,
        symbol: str,
        data: Dict,
        ttl: int = 10
    ) -> None:
        """Cache order book."""
        key = self._generate_key("orderbook", exchange, symbol)
        self.set_memory(key, data, ttl)

    # =========================================================================
    # On-Chain Cache Operations
    # =========================================================================

    def get_onchain(
        self,
        chain: str,
        data_type: str,
        address: str,
        ttl: int = 300
    ) -> Optional[Any]:
        """Get cached on-chain data."""
        key = self._generate_key("onchain", chain, data_type, address)

        # Try memory first
        data = self.get_memory(key)
        if data is not None:
            return data

        # Try file cache
        data = self.get_file("onchain", key)
        if data is not None:
            self.set_memory(key, data, ttl)
            return data

        return None

    def set_onchain(
        self,
        chain: str,
        data_type: str,
        address: str,
        data: Any,
        ttl: int = 300
    ) -> None:
        """Cache on-chain data."""
        key = self._generate_key("onchain", chain, data_type, address)
        self.set_memory(key, data, ttl)
        self.set_file("onchain", key, data, ttl)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def clear_memory(self) -> None:
        """Clear all in-memory cache."""
        self._memory_cache.clear()

    def clear_expired(self) -> int:
        """Remove expired entries from memory cache. Returns count removed."""
        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._memory_cache[key]
        return len(expired_keys)

    def clear_all(self) -> None:
        """Clear all caches (memory and file)."""
        self.clear_memory()

        if self.enabled:
            import shutil
            for subdir in ["ohlcv", "orderbook", "onchain", "misc"]:
                subdir_path = self.cache_dir / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path)
                    subdir_path.mkdir()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        stats = {
            "enabled": self.enabled,
            "memory_entries": len(self._memory_cache),
            "memory_expired": sum(
                1 for e in self._memory_cache.values() if e.is_expired()
            ),
        }

        if self.enabled:
            for subdir in ["ohlcv", "orderbook", "onchain", "misc"]:
                subdir_path = self.cache_dir / subdir
                if subdir_path.exists():
                    # Count .pkl files (exclude _meta.json)
                    stats[f"file_{subdir}"] = len(list(subdir_path.glob("*.pkl")))

        return stats


# Global cache instance
_cache_instance: Optional[DataCache] = None


def get_cache() -> DataCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DataCache()
    return _cache_instance


if __name__ == "__main__":
    # Test cache functionality
    cache = DataCache()

    # Test memory cache
    cache.set_memory("test_key", {"data": "test"}, ttl=5)
    print(f"Memory get: {cache.get_memory('test_key')}")

    # Test OHLCV cache
    import pandas as pd
    df = pd.DataFrame({
        "open": [100, 101, 102],
        "high": [105, 106, 107],
        "low": [99, 100, 101],
        "close": [104, 105, 106],
        "volume": [1000, 1100, 1200]
    })

    cache.set_ohlcv("binance", "BTC/USDT", "1h", df, ttl=60)
    cached_df = cache.get_ohlcv("binance", "BTC/USDT", "1h")
    print(f"OHLCV cached: {cached_df is not None}")

    # Stats
    print(f"Cache stats: {cache.get_stats()}")
