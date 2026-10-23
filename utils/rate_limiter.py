"""
Rate Limiter with Instant Key Rotation

Instantly switches to next API key on rate limit - no waiting.
"""

import time
from typing import List, Optional, Callable, Any, Tuple
from collections import defaultdict
from threading import Lock
import logging

logger = logging.getLogger(__name__)


class InstantKeyRotator:
    """Instantly rotates through API keys on any error."""

    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("At least one API key required")
        self.keys = keys
        self.current_index = 0
        self.lock = Lock()
        logger.info(f"InstantKeyRotator initialized with {len(keys)} keys")

    def get_key(self) -> Tuple[str, int]:
        """Get current key and its index."""
        with self.lock:
            key = self.keys[self.current_index]
            idx = self.current_index
            return key, idx

    def next_key(self) -> Tuple[str, int]:
        """Move to next key immediately."""
        with self.lock:
            self.current_index = (self.current_index + 1) % len(self.keys)
            key = self.keys[self.current_index]
            idx = self.current_index
            logger.debug(f"Switched to key index {idx}")
            return key, idx

    def execute_with_rotation(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function, rotating through ALL keys on rate limit errors.
        No waiting - instant switch to next key.
        """
        start_index = self.current_index
        attempts = 0
        last_error = None

        while attempts < len(self.keys):
            key, idx = self.get_key()
            try:
                # Pass the current key to the function
                return func(api_key=key, *args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                last_error = e

                # Check if rate limit error
                if "429" in str(e) or "rate" in error_str or "quota" in error_str or "exhausted" in error_str:
                    logger.info(f"Key {idx} rate limited, switching instantly...")
                    self.next_key()
                    attempts += 1

                    # Check if we have cycled through all keys
                    if self.current_index == start_index:
                        logger.warning(f"All {len(self.keys)} keys exhausted")
                        raise Exception(f"All {len(self.keys)} API keys rate limited") from e
                else:
                    # Non-rate-limit error, raise immediately
                    raise e

        raise last_error


class GeminiKeyManager:
    """Manages Gemini API keys with instant rotation."""

    def __init__(self, api_keys: List[str]):
        self.rotator = InstantKeyRotator(api_keys)
        self.total_keys = len(api_keys)

    def get_key(self) -> str:
        """Get current API key."""
        key, _ = self.rotator.get_key()
        return key

    def rotate(self) -> str:
        """Force rotation to next key."""
        key, _ = self.rotator.next_key()
        return key

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with automatic key rotation on rate limits."""
        return self.rotator.execute_with_rotation(func, *args, **kwargs)


class RPCEndpointManager:
    """Manages RPC endpoints with instant rotation."""

    FREE_RPCS = {
        "ethereum": [
            "https://ethereum.publicnode.com",
            "https://1rpc.io/eth",
            "https://eth.drpc.org",
            "https://endpoints.omniatech.io/v1/eth/mainnet/public",
        ],
        "arbitrum": [
            "https://arb1.arbitrum.io/rpc",
            "https://rpc.ankr.com/arbitrum",
            "https://arbitrum.publicnode.com",
            "https://1rpc.io/arb",
        ],
        "polygon": [
            "https://polygon-rpc.com",
            "https://rpc.ankr.com/polygon",
            "https://polygon.publicnode.com",
            "https://1rpc.io/matic",
        ],
        "base": [
            "https://mainnet.base.org",
            "https://rpc.ankr.com/base",
            "https://base.publicnode.com",
            "https://1rpc.io/base",
        ],
    }

    def __init__(self, chain: str = "ethereum"):
        self.chain = chain
        self.endpoints = self.FREE_RPCS.get(chain, self.FREE_RPCS["ethereum"])
        self.rotator = InstantKeyRotator(self.endpoints)

    def get_endpoint(self) -> str:
        """Get current RPC endpoint."""
        endpoint, _ = self.rotator.get_key()
        return endpoint

    def rotate(self) -> str:
        """Force rotation to next endpoint."""
        endpoint, _ = self.rotator.next_key()
        return endpoint


# Global instances
_gemini_manager: Optional[GeminiKeyManager] = None
_rpc_managers: dict = {}


def get_gemini_manager() -> GeminiKeyManager:
    """Get or create the global Gemini key manager."""
    global _gemini_manager
    if _gemini_manager is None:
        from settings import load_config
        config = load_config()
        keys = config.llm.gemini_api_keys or []
        if not keys:
            raise ValueError("No Gemini API keys configured")
        _gemini_manager = GeminiKeyManager(keys)
    return _gemini_manager


def get_rpc_manager(chain: str = "ethereum") -> RPCEndpointManager:
    """Get or create an RPC manager for a chain."""
    global _rpc_managers
    if chain not in _rpc_managers:
        _rpc_managers[chain] = RPCEndpointManager(chain)
    return _rpc_managers[chain]


def get_next_gemini_key() -> str:
    """Convenience: get next Gemini API key."""
    return get_gemini_manager().get_key()


def rotate_gemini_key() -> str:
    """Convenience: rotate to next Gemini key."""
    return get_gemini_manager().rotate()
