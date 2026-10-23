"""
Data Layer - Exchange and On-Chain Data Fetching

This module handles all data ingestion:
- Exchange APIs (Binance, Bybit, Coinbase) via CCXT
- On-chain data (Ethereum, Arbitrum, Base, Polygon) via Web3
- Market data APIs (CoinMarketCap, CoinGlass, Dune, Etherscan)
- Caching layer for rate limit management
"""

from .exchange_client import ExchangeClient
from .cache import DataCache
from .coinmarketcap_client import CoinMarketCapClient, create_coinmarketcap_client
from .coinglass_client import CoinGlassClient, create_coinglass_client
from .dune_client import DuneClient, create_dune_client
from .etherscan_client import EtherscanClient, create_etherscan_client

__all__ = [
    # Exchange
    "ExchangeClient",
    "DataCache",
    # Market Data
    "CoinMarketCapClient",
    "create_coinmarketcap_client",
    "CoinGlassClient",
    "create_coinglass_client",
    "DuneClient",
    "create_dune_client",
    "EtherscanClient",
    "create_etherscan_client",
]
