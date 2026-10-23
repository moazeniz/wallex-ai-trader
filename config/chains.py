"""
Blockchain Chain Configurations

Contains RPC endpoints, chain IDs, and configuration for supported EVM chains.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ChainInfo:
    """Configuration for a blockchain network."""
    name: str
    chain_id: int
    rpc_url: str
    explorer_url: str
    native_token: str
    block_time: float  # Average block time in seconds

    # DEX configurations
    uniswap_v3_factory: Optional[str] = None
    uniswap_v2_router: Optional[str] = None


# Chain configurations
CHAIN_CONFIG: Dict[str, ChainInfo] = {
    "ethereum": ChainInfo(
        name="Ethereum Mainnet",
        chain_id=1,
        rpc_url=os.getenv("ETH_RPC_URL", "https://eth.llamarpc.com"),
        explorer_url="https://etherscan.io",
        native_token="ETH",
        block_time=12.0,
        uniswap_v3_factory="0x1F98431c8aD98523631AE4a59f267346ea31F984",
        uniswap_v2_router="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
    ),
    "arbitrum": ChainInfo(
        name="Arbitrum One",
        chain_id=42161,
        rpc_url=os.getenv("ARB_RPC_URL", "https://arb1.arbitrum.io/rpc"),
        explorer_url="https://arbiscan.io",
        native_token="ETH",
        block_time=0.25,
        uniswap_v3_factory="0x1F98431c8aD98523631AE4a59f267346ea31F984",
    ),
    "base": ChainInfo(
        name="Base",
        chain_id=8453,
        rpc_url=os.getenv("BASE_RPC_URL", "https://mainnet.base.org"),
        explorer_url="https://basescan.org",
        native_token="ETH",
        block_time=2.0,
        uniswap_v3_factory="0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
    ),
    "polygon": ChainInfo(
        name="Polygon",
        chain_id=137,
        rpc_url=os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
        explorer_url="https://polygonscan.com",
        native_token="MATIC",
        block_time=2.0,
        uniswap_v3_factory="0x1F98431c8aD98523631AE4a59f267346ea31F984",
    ),
}

# Known exchange wallet addresses (for flow tracking)
EXCHANGE_WALLETS: Dict[str, Dict[str, str]] = {
    "ethereum": {
        "binance_1": "0x28C6c06298d514Db089934071355E5743bf21d60",
        "binance_2": "0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549",
        "coinbase_1": "0x71660c4005BA85c37ccec55d0C4493E66Fe775d3",
        "coinbase_2": "0x503828976D22510aad0201ac7EC88293211D23Da",
        "kraken_1": "0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2",
        "ftx_1": "0x2FAF487A4414Fe77e2327F0bf4AE2a264a776AD2",
        "okx_1": "0x6cC5F688a315f3dC28A7781717a9A798a59fDA7b",
        "kucoin_1": "0xf16E9B0D03470827A95CDfd0Cb8a8A3b46969B91",
    },
    "arbitrum": {
        "binance_1": "0xB38e8c17e38363aF6EbdCb3dAE12e0243582891D",
    },
    "polygon": {
        "binance_1": "0xe7804c37c13166fF0b37F5aE0BB07A3aEbb6e245",
    },
}


def get_chain_config(chain: str) -> ChainInfo:
    """Get configuration for a specific chain."""
    chain_lower = chain.lower()

    # Handle aliases
    aliases = {
        "eth": "ethereum",
        "arb": "arbitrum",
        "matic": "polygon",
    }
    chain_lower = aliases.get(chain_lower, chain_lower)

    if chain_lower not in CHAIN_CONFIG:
        raise ValueError(f"Unsupported chain: {chain}")

    return CHAIN_CONFIG[chain_lower]


def get_exchange_wallets(chain: str) -> Dict[str, str]:
    """Get known exchange wallets for a chain."""
    chain_lower = chain.lower()
    aliases = {"eth": "ethereum", "arb": "arbitrum", "matic": "polygon"}
    chain_lower = aliases.get(chain_lower, chain_lower)

    return EXCHANGE_WALLETS.get(chain_lower, {})


# ERC-20 ABI (minimal for balance/transfer tracking)
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "from", "type": "address"},
            {"indexed": True, "name": "to", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"},
        ],
        "name": "Transfer",
        "type": "event",
    },
]
