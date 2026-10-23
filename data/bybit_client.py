"""
Bybit API Client

Provides free access to derivatives data:
- Funding rates
- Open interest
- Long/short ratios

No API key required for public market data endpoints.
"""

import time
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Symbols known to not have perpetual contracts on Bybit
UNSUPPORTED_SYMBOLS = {'LUNC', 'USD', 'EUR'}


class BybitDataClient:
    """
    Client for Bybit public API.

    Provides derivatives market data including funding rates,
    open interest, and long/short ratios - all FREE.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        """Initialize Bybit client (no API key needed)."""
        self.rate_limit = 0.1
        self._last_request = 0
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._unsupported_symbols = set(UNSUPPORTED_SYMBOLS)

    def _rate_limit_wait(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()

    def _is_supported(self, symbol: str) -> bool:
        """Check if symbol is likely supported on Bybit derivatives."""
        return symbol.upper() not in self._unsupported_symbols

    def _request(self, endpoint: str, params: Optional[Dict] = None, symbol: str = None) -> Dict[str, Any]:
        self._rate_limit_wait()
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("retCode") != 0:
                # Don't log warnings for known unsupported symbols
                ret_msg = data.get("retMsg", "")
                is_invalid_symbol = "symbol" in ret_msg.lower() and ("invalid" in ret_msg.lower() or "not support" in ret_msg.lower())
                if is_invalid_symbol and symbol:
                    # Cache this as unsupported for future calls
                    self._unsupported_symbols.add(symbol.upper())
                elif not is_invalid_symbol:
                    logger.warning(f"Bybit API error: {ret_msg}")
                return {}
            return data.get("result", {})
        except requests.exceptions.RequestException as e:
            logger.error(f"Bybit request failed: {e}")
            return {}

    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        if not self._is_supported(symbol):
            return {"symbol": symbol, "rate": 0.0, "timestamp": 0}
        bybit_symbol = f"{symbol.upper()}USDT"
        result = self._request("/v5/market/funding/history", {
            "category": "linear", "symbol": bybit_symbol, "limit": 1
        }, symbol=symbol)
        if result and result.get("list"):
            funding = result["list"][0]
            return {"symbol": symbol, "rate": float(funding.get("fundingRate", 0)), "timestamp": int(funding.get("fundingRateTimestamp", 0))}
        return {"symbol": symbol, "rate": 0.0, "timestamp": 0}

    def get_funding_history(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        if not self._is_supported(symbol):
            return []
        bybit_symbol = f"{symbol.upper()}USDT"
        result = self._request("/v5/market/funding/history", {
            "category": "linear", "symbol": bybit_symbol, "limit": min(limit, 200)
        }, symbol=symbol)
        if result and result.get("list"):
            return [{"rate": float(f.get("fundingRate", 0)), "timestamp": int(f.get("fundingRateTimestamp", 0))} for f in result["list"]]
        return []

    def get_open_interest(self, symbol: str) -> Dict[str, Any]:
        if not self._is_supported(symbol):
            return {"symbol": symbol, "openInterest": 0, "change_1h": 0, "change_24h": 0, "timestamp": 0}
        bybit_symbol = f"{symbol.upper()}USDT"
        result = self._request("/v5/market/open-interest", {
            "category": "linear", "symbol": bybit_symbol, "intervalTime": "1h", "limit": 25
        }, symbol=symbol)
        if result and result.get("list") and len(result["list"]) > 0:
            oi_list = result["list"]
            current_oi = float(oi_list[0].get("openInterest", 0))
            oi_1h_ago = float(oi_list[1].get("openInterest", current_oi)) if len(oi_list) > 1 else current_oi
            oi_24h_ago = float(oi_list[-1].get("openInterest", current_oi)) if len(oi_list) >= 24 else current_oi
            change_1h = ((current_oi - oi_1h_ago) / oi_1h_ago * 100) if oi_1h_ago > 0 else 0
            change_24h = ((current_oi - oi_24h_ago) / oi_24h_ago * 100) if oi_24h_ago > 0 else 0
            return {"symbol": symbol, "openInterest": current_oi, "change_1h": change_1h, "change_24h": change_24h, "timestamp": int(oi_list[0].get("timestamp", 0))}
        return {"symbol": symbol, "openInterest": 0, "change_1h": 0, "change_24h": 0, "timestamp": 0}

    def get_long_short_ratio(self, symbol: str) -> Dict[str, Any]:
        if not self._is_supported(symbol):
            return {"symbol": symbol, "buyRatio": 0.5, "sellRatio": 0.5, "longShortRatio": 1.0, "timestamp": 0}
        bybit_symbol = f"{symbol.upper()}USDT"
        result = self._request("/v5/market/account-ratio", {
            "category": "linear", "symbol": bybit_symbol, "period": "1h", "limit": 1
        }, symbol=symbol)
        if result and result.get("list"):
            ratio = result["list"][0]
            buy = float(ratio.get("buyRatio", 0.5))
            sell = float(ratio.get("sellRatio", 0.5))
            return {"symbol": symbol, "buyRatio": buy, "sellRatio": sell, "longShortRatio": buy / max(sell, 0.001), "timestamp": int(ratio.get("timestamp", 0))}
        return {"symbol": symbol, "buyRatio": 0.5, "sellRatio": 0.5, "longShortRatio": 1.0, "timestamp": 0}

    def get_all_derivatives_data(self, symbol: str) -> Dict[str, Any]:
        if not self._is_supported(symbol):
            return {
                "symbol": symbol,
                "funding_rate": 0.0,
                "funding_ma": 0.0,
                "funding_zscore": 0.0,
                "oi_change_1h": 0.0,
                "oi_change_24h": 0.0,
                "open_interest": 0.0,
                "long_ratio": 0.5,
                "short_ratio": 0.5,
                "long_short_ratio": 1.0,
            }
        funding = self.get_funding_rate(symbol)
        funding_history = self.get_funding_history(symbol, limit=50)
        if funding_history:
            rates = [f["rate"] for f in funding_history]
            funding_ma = np.mean(rates)
            funding_std = np.std(rates) if len(rates) > 1 else 1e-10
            funding_zscore = (funding["rate"] - funding_ma) / max(funding_std, 1e-10)
        else:
            funding_ma = funding["rate"]
            funding_zscore = 0.0
        oi = self.get_open_interest(symbol)
        ls_ratio = self.get_long_short_ratio(symbol)
        return {
            "symbol": symbol,
            "funding_rate": funding["rate"],
            "funding_ma": funding_ma,
            "funding_zscore": funding_zscore,
            "oi_change_1h": oi["change_1h"] / 100,
            "oi_change_24h": oi["change_24h"] / 100,
            "open_interest": oi["openInterest"],
            "long_ratio": ls_ratio["buyRatio"],
            "short_ratio": ls_ratio["sellRatio"],
            "long_short_ratio": ls_ratio["longShortRatio"],
        }


_client = None

def get_bybit_client() -> BybitDataClient:
    global _client
    if _client is None:
        _client = BybitDataClient()
    return _client
