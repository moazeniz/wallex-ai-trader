"""
Etherscan API Client (V2)

Provides access to Ethereum and multichain blockchain data:
- Account balances and transactions
- Token transfers
- Contract information
- Gas prices
- Block data

Supports 60+ EVM chains via chainid parameter.

Rate Limits (Free Tier):
- 5 calls/second
- 100,000 calls/day
- 1,000 records per call
"""

import time
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import get_config

logger = logging.getLogger(__name__)


# Chain IDs for Etherscan V2 API
CHAIN_IDS = {
    "ethereum": 1,
    "eth": 1,
    "arbitrum": 42161,
    "arb": 42161,
    "base": 8453,
    "polygon": 137,
    "matic": 137,
    "optimism": 10,
    "op": 10,
    "bsc": 56,
    "avalanche": 43114,
    "fantom": 250,
    "linea": 59144,
    "scroll": 534352,
    "zksync": 324,
}


class EtherscanClient:
    """
    Client for Etherscan API V2.

    Free tier: 5 calls/sec, 100,000 calls/day
    Supports 60+ EVM chains with single API key.
    """

    BASE_URL = "https://api.etherscan.io/v2/api"

    def __init__(self, api_key: Optional[str] = None, chain: str = "ethereum"):
        """
        Initialize Etherscan client.

        Args:
            api_key: API key (uses config if not provided)
            chain: Chain name (ethereum, arbitrum, base, polygon, etc.)
        """
        config = get_config()
        self.api_key = api_key or config.market_data.etherscan_api_key
        self.rate_limit = config.market_data.etherscan_rate_limit
        self._last_request = 0
        self.chain = chain.lower()
        self.chain_id = CHAIN_IDS.get(self.chain, 1)  # Default to Ethereum

        if not self.api_key:
            logger.warning("Etherscan API key not configured")

    def set_chain(self, chain: str) -> None:
        """Switch to a different chain."""
        self.chain = chain.lower()
        self.chain_id = CHAIN_IDS.get(self.chain, 1)

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limits (5 calls/sec)."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()

    def _request(
        self,
        module: str,
        action: str,
        params: Optional[Dict] = None,
        chain_id: Optional[int] = None,
    ) -> Any:
        """Make API request to Etherscan V2."""
        if not self.api_key:
            raise ValueError("Etherscan API key not configured")

        self._rate_limit_wait()

        # Use provided chain_id or default to instance's chain_id
        effective_chain_id = chain_id if chain_id is not None else self.chain_id

        request_params = {
            "chainid": effective_chain_id,
            "module": module,
            "action": action,
            "apikey": self.api_key,
            **(params or {}),
        }

        try:
            response = requests.get(self.BASE_URL, params=request_params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # V2 API returns status "1" for success, "0" for error
            if data.get("status") == "0" and data.get("message") not in ["No transactions found", "No records found"]:
                error_msg = data.get("result", data.get("message", "Unknown error"))
                raise Exception(f"Etherscan API Error: {error_msg}")

            return data.get("result")

        except requests.exceptions.RequestException as e:
            logger.error(f"Etherscan request failed: {e}")
            raise

    def _wei_to_eth(self, wei: str) -> float:
        """Convert Wei to ETH."""
        return float(Decimal(wei) / Decimal(10**18))

    # =========================================================================
    # Account Methods
    # =========================================================================

    def get_eth_balance(self, address: str) -> float:
        """
        Get ETH balance for an address.

        Args:
            address: Ethereum address

        Returns:
            Balance in ETH
        """
        result = self._request("account", "balance", {"address": address, "tag": "latest"})
        return self._wei_to_eth(result)

    def get_eth_balances(self, addresses: List[str]) -> Dict[str, float]:
        """
        Get ETH balances for multiple addresses.

        Args:
            addresses: List of addresses (max 20)

        Returns:
            Dict mapping address to balance
        """
        if len(addresses) > 20:
            raise ValueError("Max 20 addresses per request")

        result = self._request(
            "account",
            "balancemulti",
            {"address": ",".join(addresses), "tag": "latest"},
        )

        return {item["account"]: self._wei_to_eth(item["balance"]) for item in result}

    def get_transactions(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100,
        sort: str = "desc",
    ) -> List[Dict]:
        """
        Get transactions for an address.

        Args:
            address: Ethereum address
            start_block: Start block number
            end_block: End block number
            page: Page number
            offset: Results per page (max 10000)
            sort: Sort order (asc/desc)

        Returns:
            List of transactions
        """
        result = self._request(
            "account",
            "txlist",
            {
                "address": address,
                "startblock": start_block,
                "endblock": end_block,
                "page": page,
                "offset": min(offset, 10000),
                "sort": sort,
            },
        )

        return result if isinstance(result, list) else []

    def get_token_transfers(
        self,
        address: Optional[str] = None,
        contract_address: Optional[str] = None,
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100,
        sort: str = "desc",
    ) -> List[Dict]:
        """
        Get ERC-20 token transfers.

        Args:
            address: Filter by address
            contract_address: Filter by token contract
            start_block: Start block
            end_block: End block
            page: Page number
            offset: Results per page
            sort: Sort order

        Returns:
            List of token transfers
        """
        params = {
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": min(offset, 10000),
            "sort": sort,
        }

        if address:
            params["address"] = address
        if contract_address:
            params["contractaddress"] = contract_address

        result = self._request("account", "tokentx", params)
        return result if isinstance(result, list) else []

    # =========================================================================
    # Token Methods
    # =========================================================================

    def get_token_balance(
        self,
        contract_address: str,
        address: str,
    ) -> int:
        """
        Get token balance for an address.

        Args:
            contract_address: Token contract address
            address: Holder address

        Returns:
            Token balance (raw, without decimals)
        """
        result = self._request(
            "account",
            "tokenbalance",
            {"contractaddress": contract_address, "address": address, "tag": "latest"},
        )
        return int(result)

    def get_token_info(self, contract_address: str) -> Dict[str, Any]:
        """
        Get token information.

        Args:
            contract_address: Token contract address

        Returns:
            Token info (name, symbol, decimals, etc.)
        """
        # This uses the token supply endpoint
        result = self._request(
            "stats",
            "tokensupply",
            {"contractaddress": contract_address},
        )

        return {
            "contract_address": contract_address,
            "total_supply": int(result) if result else 0,
        }

    # =========================================================================
    # Contract Methods
    # =========================================================================

    def get_contract_abi(self, address: str) -> List[Dict]:
        """
        Get contract ABI.

        Args:
            address: Contract address

        Returns:
            Contract ABI
        """
        import json
        result = self._request("contract", "getabi", {"address": address})
        return json.loads(result) if result else []

    def get_contract_source(self, address: str) -> Dict[str, Any]:
        """
        Get contract source code.

        Args:
            address: Contract address

        Returns:
            Source code and metadata
        """
        result = self._request("contract", "getsourcecode", {"address": address})
        return result[0] if result else {}

    # =========================================================================
    # Gas Methods
    # =========================================================================

    def get_gas_price(self) -> Dict[str, float]:
        """
        Get current gas prices.

        Returns:
            Gas prices in Gwei (safe, propose, fast)
        """
        result = self._request("gastracker", "gasoracle")
        return {
            "safe": float(result.get("SafeGasPrice", 0)),
            "propose": float(result.get("ProposeGasPrice", 0)),
            "fast": float(result.get("FastGasPrice", 0)),
            "base_fee": float(result.get("suggestBaseFee", 0)),
        }

    # =========================================================================
    # Block Methods
    # =========================================================================

    def get_block_number(self) -> int:
        """Get latest block number."""
        result = self._request("proxy", "eth_blockNumber")
        return int(result, 16)

    def get_block_by_timestamp(
        self,
        timestamp: int,
        closest: str = "before",
    ) -> int:
        """
        Get block number closest to timestamp.

        Args:
            timestamp: Unix timestamp
            closest: 'before' or 'after'

        Returns:
            Block number
        """
        result = self._request(
            "block",
            "getblocknobytime",
            {"timestamp": timestamp, "closest": closest},
        )
        return int(result)

    # =========================================================================
    # Analytics Methods
    # =========================================================================

    def get_whale_transactions(
        self,
        contract_address: str,
        min_value: float = 100000,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get large token transfers (whale activity).

        Args:
            contract_address: Token contract
            min_value: Minimum transfer value (in tokens)
            limit: Max results

        Returns:
            List of large transfers
        """
        transfers = self.get_token_transfers(
            contract_address=contract_address,
            offset=min(limit * 2, 1000),  # Get more to filter
            sort="desc",
        )

        # Filter by value (note: values are in raw token units)
        # Would need token decimals to convert properly
        large_transfers = []
        for tx in transfers:
            try:
                value = int(tx.get("value", 0))
                if value >= min_value * (10**18):  # Assume 18 decimals
                    large_transfers.append(tx)
                    if len(large_transfers) >= limit:
                        break
            except (ValueError, TypeError):
                continue

        return large_transfers

    def get_top_token_holders(
        self,
        contract_address: str,
    ) -> List[Dict]:
        """
        Note: This endpoint requires Etherscan Pro.
        Returns empty list for free tier.
        """
        logger.info("Top token holders requires Etherscan Pro subscription")
        return []

    def analyze_address_activity(
        self,
        address: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Analyze recent activity for an address.

        Args:
            address: Ethereum address
            days: Days to analyze

        Returns:
            Activity summary
        """
        # Get recent transactions
        txs = self.get_transactions(address, offset=500, sort="desc")

        if not txs:
            return {
                "address": address,
                "tx_count": 0,
                "is_active": False,
            }

        # Calculate cutoff timestamp
        cutoff = int(datetime.now().timestamp()) - (days * 86400)

        # Filter recent transactions
        recent_txs = [
            tx for tx in txs
            if int(tx.get("timeStamp", 0)) > cutoff
        ]

        # Calculate stats
        total_eth_in = sum(
            self._wei_to_eth(tx.get("value", "0"))
            for tx in recent_txs
            if tx.get("to", "").lower() == address.lower()
        )

        total_eth_out = sum(
            self._wei_to_eth(tx.get("value", "0"))
            for tx in recent_txs
            if tx.get("from", "").lower() == address.lower()
        )

        return {
            "address": address,
            "tx_count_total": len(txs),
            "tx_count_recent": len(recent_txs),
            "eth_received": total_eth_in,
            "eth_sent": total_eth_out,
            "net_flow": total_eth_in - total_eth_out,
            "is_active": len(recent_txs) > 0,
            "days_analyzed": days,
        }


def create_etherscan_client(
    api_key: Optional[str] = None,
    chain: str = "ethereum",
) -> EtherscanClient:
    """Create an EtherscanClient instance."""
    return EtherscanClient(api_key=api_key, chain=chain)


if __name__ == "__main__":
    print("Testing Etherscan Client (V2 API)...")

    client = EtherscanClient()

    if not client.api_key:
        print("Error: ETHERSCAN_API_KEY not set")
        exit(1)

    print(f"Using chain: {client.chain} (chainid: {client.chain_id})")

    # Test address (Vitalik's wallet)
    test_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

    # Test ETH balance
    print("\n1. Getting ETH balance...")
    try:
        balance = client.get_eth_balance(test_address)
        print(f"  Balance: {balance:.4f} ETH")
    except Exception as e:
        print(f"  Error: {e}")

    # Test gas prices
    print("\n2. Getting gas prices...")
    try:
        gas = client.get_gas_price()
        print(f"  Safe: {gas['safe']} Gwei")
        print(f"  Propose: {gas['propose']} Gwei")
        print(f"  Fast: {gas['fast']} Gwei")
    except Exception as e:
        print(f"  Error: {e}")

    # Test recent transactions
    print("\n3. Getting recent transactions...")
    try:
        txs = client.get_transactions(test_address, offset=5)
        print(f"  Found {len(txs)} transactions")
        if txs:
            latest = txs[0]
            value = client._wei_to_eth(latest.get("value", "0"))
            print(f"  Latest: {value:.4f} ETH at block {latest.get('blockNumber')}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test block number
    print("\n4. Getting block number...")
    try:
        block = client.get_block_number()
        print(f"  Current block: {block:,}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test multichain - Arbitrum
    print("\n5. Testing Arbitrum chain...")
    try:
        client.set_chain("arbitrum")
        print(f"  Switched to: {client.chain} (chainid: {client.chain_id})")
        arb_block = client.get_block_number()
        print(f"  Arbitrum block: {arb_block:,}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nTest complete!")
