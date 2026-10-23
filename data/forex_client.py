"""
Forex Client for Frankfurter API

Fetches USD/EUR exchange rates from the free Frankfurter API.
No API key required.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ForexClient:
    """Client for fetching forex data from Frankfurter API."""
    
    BASE_URL = "https://api.frankfurter.dev/v1"
    
    def __init__(self):
        self.session = requests.Session()
        self._cache = {}
        self._cache_time = None
        self._cache_duration = 3600  # Cache for 1 hour (data updates daily anyway)
    
    def get_latest_rate(self, base: str = "USD", symbol: str = "EUR") -> Optional[Dict[str, Any]]:
        """Get latest exchange rate."""
        try:
            url = f"{self.BASE_URL}/latest?base={base}&symbols={symbol}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            rate = data.get("rates", {}).get(symbol)
            if rate:
                return {
                    "base": base,
                    "symbol": symbol,
                    "rate": rate,
                    "date": data.get("date"),
                    "pair": f"{base}/{symbol}"
                }
            return None
        except Exception as e:
            logger.error(f"Forex API error: {e}")
            return None
    
    def get_historical_rates(self, base: str = "USD", symbol: str = "EUR", days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical rates for technical analysis."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.BASE_URL}/{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}?base={base}&symbols={symbol}"
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            rates = data.get("rates", {})
            if not rates:
                return None
            
            # Convert to DataFrame
            records = []
            for date_str, rate_dict in rates.items():
                records.append({
                    "timestamp": pd.to_datetime(date_str),
                    "open": rate_dict.get(symbol),
                    "high": rate_dict.get(symbol),  # Frankfurter only has close
                    "low": rate_dict.get(symbol),
                    "close": rate_dict.get(symbol),
                    "volume": 0  # No volume for forex from this API
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Simulate OHLC from daily closes (approximate high/low as +/- 0.5%)
            if len(df) > 1:
                df["high"] = df["close"] * 1.005
                df["low"] = df["close"] * 0.995
                df["open"] = df["close"].shift(1).fillna(df["close"])
            
            return df
            
        except Exception as e:
            logger.error(f"Forex historical API error: {e}")
            return None
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 50) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for forex pair.
        Compatible with exchange_client interface.
        
        Symbol format: USD/EUR or EUR/USD
        """
        try:
            parts = symbol.replace("/USDT", "").replace("USDT", "").split("/")
            if len(parts) == 2:
                base, quote = parts
            elif symbol in ["USD", "EUR"]:
                # Single currency - compare to the other
                if symbol == "USD":
                    base, quote = "USD", "EUR"
                else:
                    base, quote = "EUR", "USD"
            else:
                logger.warning(f"Invalid forex symbol: {symbol}")
                return None
            
            df = self.get_historical_rates(base=base, symbol=quote, days=limit)
            return df
            
        except Exception as e:
            logger.error(f"Forex fetch_ohlcv error: {e}")
            return None
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a forex pair."""
        try:
            parts = symbol.replace("/USDT", "").replace("USDT", "").split("/")
            if len(parts) == 2:
                base, quote = parts
            elif symbol in ["USD", "EUR"]:
                if symbol == "USD":
                    base, quote = "USD", "EUR"
                else:
                    base, quote = "EUR", "USD"
            else:
                return None
            
            result = self.get_latest_rate(base=base, symbol=quote)
            if result:
                return result["rate"]
            return None
        except Exception as e:
            logger.error(f"Forex get_price error: {e}")
            return None


# Singleton instance
_forex_client = None

def get_forex_client() -> ForexClient:
    global _forex_client
    if _forex_client is None:
        _forex_client = ForexClient()
    return _forex_client


if __name__ == "__main__":
    # Test the client
    client = ForexClient()
    
    print("Testing Frankfurter API...")
    
    # Test latest rate
    rate = client.get_latest_rate("USD", "EUR")
    print(f"USD/EUR: {rate}")
    
    rate = client.get_latest_rate("EUR", "USD")
    print(f"EUR/USD: {rate}")
    
    # Test historical
    df = client.get_historical_rates("USD", "EUR", days=7)
    print(f"\nHistorical data (7 days):")
    print(df)
