"""
Multi-Timeframe Data Fetcher

Fetches OHLCV data at multiple timeframes for enhanced analysis:
- 1h: Short-term signals (primary)
- 4h: Medium-term trend
- 1d: Long-term trend

Provides aligned data for multi-timeframe feature extraction.
"""

import sys
sys.path.insert(0, '/opt/wallex')

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import time

from data.exchange_client import ExchangeClient
from settings import TRACKED_TOKENS


class MultiTimeframeFetcher:
    """
    Fetches and caches multi-timeframe OHLCV data.

    Supports:
    - 1h, 4h, 1d timeframes
    - Caching to reduce API calls
    - BTC data for cross-asset features
    """

    def __init__(self, exchange: str = "binance"):
        self.exchange = exchange
        self.client = ExchangeClient(exchange=exchange)

        # Cache
        self._cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl = 300  # 5 minutes

        # BTC data cache for cross-asset features
        self._btc_cache: Dict[str, pd.DataFrame] = {}

    def _get_cache_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        if cache_key not in self._cache_timestamps:
            return False
        age = (datetime.now() - self._cache_timestamps[cache_key]).total_seconds()
        return age < self._cache_ttl

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 200,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a single timeframe."""
        cache_key = self._get_cache_key(symbol, timeframe)

        if use_cache and self._is_cache_valid(cache_key):
            return self._cache.get(cache_key)

        try:
            df = self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

            if df is not None and len(df) > 0:
                self._cache[cache_key] = df
                self._cache_timestamps[cache_key] = datetime.now()

            return df
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return None

    def fetch_multi_timeframe(
        self,
        symbol: str,
        include_btc: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all timeframes.

        Returns:
            Dict with keys: '1h', '4h', '1d', 'btc_1h' (optional)
        """
        result = {}

        # 1h data (primary, more candles)
        result['1h'] = self.fetch_ohlcv(symbol, '1h', limit=200)

        # 4h data
        result['4h'] = self.fetch_ohlcv(symbol, '4h', limit=100)

        # 1d data
        result['1d'] = self.fetch_ohlcv(symbol, '1d', limit=60)

        # BTC data for cross-asset features
        if include_btc and 'BTC' not in symbol:
            result['btc_1h'] = self.fetch_ohlcv('BTC/USDT', '1h', limit=200)
        elif 'BTC' in symbol:
            result['btc_1h'] = result['1h']  # Same data

        return result

    def fetch_all_tokens(
        self,
        tokens: List[str] = None,
        include_btc: bool = True
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch multi-timeframe data for all tokens.

        Returns:
            Dict[symbol] -> Dict[timeframe] -> DataFrame
        """
        if tokens is None:
            tokens = [v['symbol'] for v in TRACKED_TOKENS.values()]

        all_data = {}

        # Fetch BTC first (used for cross-asset features)
        btc_data = None
        if include_btc:
            btc_data = self.fetch_ohlcv('BTC/USDT', '1h', limit=200)
            self._btc_cache['1h'] = btc_data

        for symbol in tokens:
            try:
                data = self.fetch_multi_timeframe(symbol, include_btc=False)

                # Add BTC data
                if btc_data is not None and 'BTC' not in symbol:
                    data['btc_1h'] = btc_data

                all_data[symbol] = data

            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                all_data[symbol] = {'error': str(e)}

        return all_data

    def get_aligned_data(
        self,
        symbol: str,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame = None,
        df_1d: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Align multi-timeframe data to the 1h timeframe.

        Higher timeframes are forward-filled to match 1h index.
        """
        if df_1h is None or len(df_1h) == 0:
            return None, None, None

        # 1h is primary
        aligned_1h = df_1h.copy()

        # Align 4h to 1h
        aligned_4h = None
        if df_4h is not None and len(df_4h) > 0:
            aligned_4h = df_4h.copy()

        # Align 1d to 1h
        aligned_1d = None
        if df_1d is not None and len(df_1d) > 0:
            aligned_1d = df_1d.copy()

        return aligned_1h, aligned_4h, aligned_1d


class HistoricalMultiframeFetcher:
    """
    Fetches historical multi-timeframe data for training.
    Uses pagination to get full year of data.
    """

    def __init__(self, exchange: str = "binance"):
        try:
            import ccxt
            self.exchange = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"}
            })
        except ImportError:
            self.exchange = None

    def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        days: int = 365
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data with pagination."""
        if self.exchange is None:
            return None

        all_candles = []

        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        tf_ms = {
            "1h": 3600000,
            "4h": 14400000,
            "1d": 86400000,
        }

        candle_ms = tf_ms.get(timeframe, 3600000)
        limit = 1000
        current_start = start_time

        while current_start < end_time:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_start,
                    limit=limit
                )

                if not candles:
                    break

                all_candles.extend(candles)
                current_start = candles[-1][0] + candle_ms

                time.sleep(0.1)  # Rate limiting

                if len(all_candles) > 20000:  # Safety limit
                    break

            except Exception as e:
                print(f"Error: {e}")
                break

        if not all_candles:
            return None

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        return df

    def fetch_all_timeframes(
        self,
        symbol: str,
        days: int = 365,
        output_dir: str = "/opt/wallex/data/historical"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all timeframes."""
        result = {}

        for tf in ['1h', '4h', '1d']:
            print(f"  Fetching {symbol} {tf}...")
            df = self.fetch_historical(symbol, tf, days)

            if df is not None:
                result[tf] = df

                # Save to file
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    filename = f"{symbol.replace('/', '_')}_{tf}_{days}d.csv"
                    df.to_csv(Path(output_dir) / filename, index=False)
                    print(f"    Saved {len(df)} candles")

        return result


# Singleton
_fetcher = None

def get_multiframe_fetcher() -> MultiTimeframeFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = MultiTimeframeFetcher()
    return _fetcher


if __name__ == "__main__":
    print("Testing Multi-Timeframe Fetcher...")

    fetcher = MultiTimeframeFetcher()

    # Test single symbol
    data = fetcher.fetch_multi_timeframe("BTC/USDT")

    print("\nFetched data:")
    for tf, df in data.items():
        if df is not None:
            print(f"  {tf}: {len(df)} candles")
        else:
            print(f"  {tf}: No data")
