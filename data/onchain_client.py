"""
On-Chain Data Client

Provides access to blockchain data via Web3:
- Token balances and holders
- Transfer events and large transactions
- DEX volume and liquidity
- Exchange inflow/outflow tracking
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from web3 import Web3
from web3.exceptions import ContractLogicError
import pandas as pd

from .cache import DataCache, get_cache

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.chains import get_chain_config, ERC20_ABI, get_exchange_wallets, ChainInfo
from utils.rate_limiter import get_rpc_manager
from config.tokens import get_token_config, TokenInfo


@dataclass
class TransferEvent:
    """Represents a token transfer event."""
    tx_hash: str
    block_number: int
    timestamp: Optional[int]
    from_address: str
    to_address: str
    value: float  # In token units (not wei)
    value_usd: Optional[float]
    is_exchange_inflow: bool
    is_exchange_outflow: bool


@dataclass
class HolderInfo:
    """Information about a token holder."""
    address: str
    balance: float
    percentage: float
    is_exchange: bool
    is_contract: bool
    label: Optional[str] = None


class OnChainClient:
    """
    Client for fetching on-chain data from EVM chains.

    Supports:
    - Token balance queries
    - Transfer event monitoring
    - Whale tracking
    - Exchange flow analysis
    """

    def __init__(
        self,
        chain: str = "ethereum",
        cache: Optional[DataCache] = None,
    ):
        """
        Initialize the on-chain client.

        Args:
            chain: Chain to connect to ('ethereum', 'arbitrum', 'base', 'polygon')
            cache: Optional cache instance
        """
        self.chain = chain.lower()
        self.chain_config = get_chain_config(chain)
        self.cache = cache or get_cache()

        # Initialize RPC manager for endpoint rotation
        self.rpc_manager = get_rpc_manager(chain)

        # Initialize Web3 with rotatable endpoint
        self._init_web3()

        # Load exchange wallets
        self.exchange_wallets = get_exchange_wallets(chain)
        self._exchange_addresses = set(
            addr.lower() for addr in self.exchange_wallets.values()
        )

        # Contract cache
        self._contracts: Dict[str, Any] = {}

    def _init_web3(self):
        """Initialize or reinitialize Web3 with current RPC endpoint."""
        endpoint = self.rpc_manager.get_endpoint()
        self.w3 = Web3(Web3.HTTPProvider(endpoint, request_kwargs={"timeout": 30}))
        # Clear contract cache on RPC change
        self._contracts = {}

    def _rotate_rpc(self):
        """Rotate to next RPC endpoint and reinitialize Web3."""
        new_endpoint = self.rpc_manager.rotate()
        print(f"Rotating RPC to: {new_endpoint[:40]}...")
        self._init_web3()

    def _call_with_retry(self, func, max_retries=None):
        """
        Execute RPC call with automatic rotation on 429 errors.

        Args:
            func: Callable to execute
            max_retries: Max number of retries (defaults to number of endpoints)
        """
        if max_retries is None:
            max_retries = len(self.rpc_manager.endpoints)

        last_error = None
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "429" in error_str or "Too Many Requests" in error_str or "rate" in error_str.lower():
                    print(f"RPC rate limited (attempt {attempt + 1}/{max_retries}), rotating endpoint...")
                    self._rotate_rpc()
                    continue
                # Non-rate-limit error, raise immediately
                raise

        raise RuntimeError(f"All {max_retries} RPC endpoints exhausted") from last_error

    def is_connected(self) -> bool:
        """Check if connected to the RPC node."""
        try:
            return self.w3.is_connected()
        except Exception:
            return False

    def _get_contract(self, token_address: str) -> Any:
        """Get or create a contract instance."""
        address_lower = token_address.lower()
        if address_lower not in self._contracts:
            checksum = Web3.to_checksum_address(token_address)
            self._contracts[address_lower] = self.w3.eth.contract(
                address=checksum,
                abi=ERC20_ABI
            )
        return self._contracts[address_lower]

    def _is_exchange_address(self, address: str) -> bool:
        """Check if an address belongs to a known exchange."""
        return address.lower() in self._exchange_addresses

    def _is_contract(self, address: str) -> bool:
        """Check if an address is a contract."""
        try:
            code = self.w3.eth.get_code(Web3.to_checksum_address(address))
            return len(code) > 0
        except Exception:
            return False

    # =========================================================================
    # Token Data
    # =========================================================================

    def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get basic token information.

        Args:
            token_address: Token contract address

        Returns:
            Dictionary with symbol, decimals, total supply
        """
        contract = self._get_contract(token_address)

        try:
            symbol = self._call_with_retry(lambda: contract.functions.symbol().call())
            decimals = self._call_with_retry(lambda: contract.functions.decimals().call())
            total_supply_raw = self._call_with_retry(lambda: contract.functions.totalSupply().call())
            total_supply = total_supply_raw / (10 ** decimals)

            return {
                "address": token_address,
                "symbol": symbol,
                "decimals": decimals,
                "total_supply": total_supply,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get token info: {e}")

    def get_balance(
        self,
        token_address: str,
        wallet_address: str,
        decimals: Optional[int] = None,
    ) -> float:
        """
        Get token balance for a wallet.

        Args:
            token_address: Token contract address
            wallet_address: Wallet to check
            decimals: Token decimals (fetched if not provided)

        Returns:
            Balance in token units
        """
        contract = self._get_contract(token_address)
        checksum_wallet = Web3.to_checksum_address(wallet_address)

        try:
            if decimals is None:
                decimals = self._call_with_retry(lambda: contract.functions.decimals().call())

            balance_raw = self._call_with_retry(lambda: contract.functions.balanceOf(checksum_wallet).call())
            return balance_raw / (10 ** decimals)
        except Exception as e:
            raise RuntimeError(f"Failed to get balance: {e}")

    def get_top_holders(
        self,
        token_address: str,
        limit: int = 100,
        use_cache: bool = True,
        cache_ttl: int = 3600,
    ) -> List[HolderInfo]:
        """
        Get top token holders.

        Note: This requires indexing or an external API for accurate results.
        This implementation provides a simplified version using recent transfers.

        Args:
            token_address: Token contract address
            limit: Number of top holders to return
            use_cache: Whether to use cache
            cache_ttl: Cache TTL in seconds

        Returns:
            List of HolderInfo objects
        """
        # Check cache
        if use_cache:
            cached = self.cache.get_onchain(
                self.chain, "top_holders", token_address, cache_ttl
            )
            if cached is not None:
                return cached

        # Get token info
        token_info = self.get_token_info(token_address)
        decimals = token_info["decimals"]
        total_supply = token_info["total_supply"]

        # Get recent transfer events to find active holders
        # Note: This is a simplified approach - production would use an indexer
        contract = self._get_contract(token_address)

        try:
            # Get transfers from last ~1000 blocks using web3.py 7.x API
            current_block = self._call_with_retry(lambda: self.w3.eth.block_number)
            from_block = max(0, current_block - 1000)

            # Use contract.events.Transfer.get_logs() (web3.py 7.x API)
            events = self._call_with_retry(
                lambda: contract.events.Transfer.get_logs(
                    from_block=from_block,
                    to_block="latest"
                )
            )

            # Collect unique addresses
            addresses = set()
            for event in events:
                addresses.add(event["args"]["from"])
                addresses.add(event["args"]["to"])

            # Remove zero address
            addresses.discard("0x0000000000000000000000000000000000000000")

            # Get balances
            holders = []
            for addr in list(addresses)[:limit * 2]:  # Check more than needed
                try:
                    balance = self.get_balance(token_address, addr, decimals)
                    if balance > 0:
                        holders.append(HolderInfo(
                            address=addr,
                            balance=balance,
                            percentage=(balance / total_supply) * 100 if total_supply > 0 else 0,
                            is_exchange=self._is_exchange_address(addr),
                            is_contract=self._is_contract(addr),
                            label=self._get_address_label(addr),
                        ))
                except Exception:
                    continue

            # Sort by balance and limit
            holders.sort(key=lambda x: x.balance, reverse=True)
            holders = holders[:limit]

            # Cache results
            if use_cache:
                self.cache.set_onchain(
                    self.chain, "top_holders", token_address, holders, cache_ttl
                )

            return holders

        except Exception as e:
            raise RuntimeError(f"Failed to get top holders: {e}")

    def _get_address_label(self, address: str) -> Optional[str]:
        """Get label for known addresses."""
        addr_lower = address.lower()
        for label, known_addr in self.exchange_wallets.items():
            if known_addr.lower() == addr_lower:
                return label
        return None

    # =========================================================================
    # Transfer Events
    # =========================================================================

    def get_recent_transfers(
        self,
        token_address: str,
        blocks: int = 1000,
        min_value_usd: float = 0,
        use_cache: bool = True,
        cache_ttl: int = 300,
    ) -> List[TransferEvent]:
        """
        Get recent transfer events for a token.

        Args:
            token_address: Token contract address
            blocks: Number of blocks to look back
            min_value_usd: Minimum transfer value in USD
            use_cache: Whether to use cache
            cache_ttl: Cache TTL in seconds

        Returns:
            List of TransferEvent objects
        """
        cache_key = f"{token_address}_{blocks}_{min_value_usd}"
        if use_cache:
            cached = self.cache.get_onchain(
                self.chain, "transfers", cache_key, cache_ttl
            )
            if cached is not None:
                return cached

        contract = self._get_contract(token_address)
        token_info = self.get_token_info(token_address)
        decimals = token_info["decimals"]

        current_block = self._call_with_retry(lambda: self.w3.eth.block_number)
        from_block = max(0, current_block - blocks)

        try:
            # Use contract.events.Transfer.get_logs() (web3.py 7.x API)
            events = self._call_with_retry(
                lambda: contract.events.Transfer.get_logs(
                    from_block=from_block,
                    to_block="latest"
                )
            )

            transfers = []
            for event in events:
                value = event["args"]["value"] / (10 ** decimals)
                from_addr = event["args"]["from"]
                to_addr = event["args"]["to"]

                transfer = TransferEvent(
                    tx_hash=event["transactionHash"].hex(),
                    block_number=event["blockNumber"],
                    timestamp=None,  # Would need block lookup for timestamp
                    from_address=from_addr,
                    to_address=to_addr,
                    value=value,
                    value_usd=None,  # Would need price data
                    is_exchange_inflow=self._is_exchange_address(to_addr),
                    is_exchange_outflow=self._is_exchange_address(from_addr),
                )
                transfers.append(transfer)

            if use_cache:
                self.cache.set_onchain(
                    self.chain, "transfers", cache_key, transfers, cache_ttl
                )

            return transfers

        except Exception as e:
            raise RuntimeError(f"Failed to get transfers: {e}")

    def get_large_transfers(
        self,
        token_address: str,
        min_value: float,
        blocks: int = 5000,
    ) -> List[TransferEvent]:
        """
        Get large transfers (whale movements).

        Args:
            token_address: Token contract address
            min_value: Minimum transfer value in tokens
            blocks: Number of blocks to look back

        Returns:
            List of large TransferEvent objects
        """
        all_transfers = self.get_recent_transfers(
            token_address, blocks, use_cache=False
        )
        return [t for t in all_transfers if t.value >= min_value]

    # =========================================================================
    # Exchange Flow Analysis
    # =========================================================================

    def get_exchange_flow(
        self,
        token_address: str,
        blocks: int = 5000,
    ) -> Dict[str, Any]:
        """
        Calculate exchange inflow/outflow for a token.

        Args:
            token_address: Token contract address
            blocks: Number of blocks to analyze

        Returns:
            Dictionary with inflow, outflow, and net flow
        """
        transfers = self.get_recent_transfers(token_address, blocks, use_cache=False)

        inflow = sum(t.value for t in transfers if t.is_exchange_inflow)
        outflow = sum(t.value for t in transfers if t.is_exchange_outflow)
        net_flow = outflow - inflow  # Positive = leaving exchanges (bullish)

        inflow_count = sum(1 for t in transfers if t.is_exchange_inflow)
        outflow_count = sum(1 for t in transfers if t.is_exchange_outflow)

        return {
            "token_address": token_address,
            "chain": self.chain,
            "blocks_analyzed": blocks,
            "inflow": inflow,
            "outflow": outflow,
            "net_flow": net_flow,
            "inflow_count": inflow_count,
            "outflow_count": outflow_count,
            "flow_ratio": outflow / inflow if inflow > 0 else float("inf"),
            "is_accumulating": net_flow > 0,
        }

    # =========================================================================
    # Whale Analysis
    # =========================================================================

    def get_whale_activity(
        self,
        token_address: str,
        whale_threshold_pct: float = 0.1,
        blocks: int = 5000,
    ) -> Dict[str, Any]:
        """
        Analyze whale activity for a token.

        Args:
            token_address: Token contract address
            whale_threshold_pct: Minimum % of supply to be considered whale
            blocks: Number of blocks to analyze

        Returns:
            Dictionary with whale activity metrics
        """
        # Get top holders
        holders = self.get_top_holders(token_address, limit=50)

        # Filter whales
        whales = [h for h in holders if h.percentage >= whale_threshold_pct]

        # Get recent transfers
        transfers = self.get_recent_transfers(token_address, blocks, use_cache=False)

        # Analyze whale transactions
        whale_addresses = {w.address.lower() for w in whales}
        whale_transfers = [
            t for t in transfers
            if t.from_address.lower() in whale_addresses
            or t.to_address.lower() in whale_addresses
        ]

        whale_buying = sum(
            t.value for t in whale_transfers
            if t.to_address.lower() in whale_addresses
        )
        whale_selling = sum(
            t.value for t in whale_transfers
            if t.from_address.lower() in whale_addresses
        )

        return {
            "token_address": token_address,
            "chain": self.chain,
            "whale_count": len(whales),
            "whale_holdings_pct": sum(w.percentage for w in whales),
            "whale_transfers": len(whale_transfers),
            "whale_buying": whale_buying,
            "whale_selling": whale_selling,
            "whale_net": whale_buying - whale_selling,
            "is_whale_accumulating": whale_buying > whale_selling,
            "top_whale": whales[0].address if whales else None,
            "top_whale_pct": whales[0].percentage if whales else 0,
        }

    # =========================================================================
    # Holder Distribution
    # =========================================================================

    def get_holder_distribution(
        self,
        token_address: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze holder distribution (concentration metrics).

        Args:
            token_address: Token contract address
            use_cache: Whether to use cache

        Returns:
            Dictionary with distribution metrics
        """
        holders = self.get_top_holders(token_address, limit=100, use_cache=use_cache)

        if not holders:
            return {
                "token_address": token_address,
                "holder_count": 0,
                "top_10_pct": 0,
                "top_50_pct": 0,
                "exchange_holdings_pct": 0,
            }

        # Calculate metrics
        top_10_pct = sum(h.percentage for h in holders[:10])
        top_50_pct = sum(h.percentage for h in holders[:50])
        top_100_pct = sum(h.percentage for h in holders)

        exchange_holdings = sum(h.percentage for h in holders if h.is_exchange)
        contract_holdings = sum(h.percentage for h in holders if h.is_contract)

        # Gini coefficient (simplified)
        percentages = [h.percentage for h in holders]
        n = len(percentages)
        if n > 0 and sum(percentages) > 0:
            sorted_pcts = sorted(percentages)
            cumulative = sum((i + 1) * p for i, p in enumerate(sorted_pcts))
            gini = (2 * cumulative) / (n * sum(percentages)) - (n + 1) / n
        else:
            gini = 0

        return {
            "token_address": token_address,
            "chain": self.chain,
            "holder_count": len(holders),
            "top_10_pct": top_10_pct,
            "top_50_pct": top_50_pct,
            "top_100_pct": top_100_pct,
            "exchange_holdings_pct": exchange_holdings,
            "contract_holdings_pct": contract_holdings,
            "gini_coefficient": gini,
            "is_concentrated": top_10_pct > 50,
        }


def create_onchain_client(chain: str = "ethereum") -> OnChainClient:
    """Create an on-chain client for the specified chain."""
    return OnChainClient(chain=chain)


if __name__ == "__main__":
    # Test the on-chain client
    print("Testing OnChainClient...")

    client = OnChainClient(chain="ethereum")
    print(f"Connected: {client.is_connected()}")

    if client.is_connected():
        # Test with AAVE token
        aave_address = "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9"

        print("\nGetting AAVE token info...")
        info = client.get_token_info(aave_address)
        print(f"Symbol: {info['symbol']}, Supply: {info['total_supply']:,.0f}")

        print("\nGetting exchange flow...")
        flow = client.get_exchange_flow(aave_address, blocks=100)
        print(f"Net flow: {flow['net_flow']:,.2f} ({('Accumulating' if flow['is_accumulating'] else 'Distributing')})")
