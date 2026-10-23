"""
Token Configurations

Contains contract addresses and metadata for tracked tokens.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class TokenInfo:
    """Configuration for a tracked token."""
    symbol: str
    name: str
    chain: str
    address: str
    decimals: int
    trading_pair: str  # e.g., "AAVE/USDT"
    priority: str  # "high", "medium", "low"
    category: str  # "defi", "l2", "infrastructure"

    # Optional metadata
    coingecko_id: Optional[str] = None
    website: Optional[str] = None


# Token configurations
TOKEN_CONFIG: Dict[str, TokenInfo] = {
    # DeFi Blue-Chips (Ethereum)
    "AAVE": TokenInfo(
        symbol="AAVE",
        name="Aave",
        chain="ethereum",
        address="0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
        decimals=18,
        trading_pair="AAVE/USDT",
        priority="high",
        category="defi",
        coingecko_id="aave",
    ),
    "UNI": TokenInfo(
        symbol="UNI",
        name="Uniswap",
        chain="ethereum",
        address="0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        decimals=18,
        trading_pair="UNI/USDT",
        priority="high",
        category="defi",
        coingecko_id="uniswap",
    ),
    "LINK": TokenInfo(
        symbol="LINK",
        name="Chainlink",
        chain="ethereum",
        address="0x514910771AF9Ca656af840dff83E8264EcF986CA",
        decimals=18,
        trading_pair="LINK/USDT",
        priority="high",
        category="infrastructure",
        coingecko_id="chainlink",
    ),
    "MKR": TokenInfo(
        symbol="MKR",
        name="Maker",
        chain="ethereum",
        address="0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2",
        decimals=18,
        trading_pair="MKR/USDT",
        priority="high",
        category="defi",
        coingecko_id="maker",
    ),
    "SNX": TokenInfo(
        symbol="SNX",
        name="Synthetix",
        chain="ethereum",
        address="0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F",
        decimals=18,
        trading_pair="SNX/USDT",
        priority="medium",
        category="defi",
        coingecko_id="havven",
    ),
    "CRV": TokenInfo(
        symbol="CRV",
        name="Curve DAO",
        chain="ethereum",
        address="0xD533a949740bb3306d119CC777fa900bA034cd52",
        decimals=18,
        trading_pair="CRV/USDT",
        priority="medium",
        category="defi",
        coingecko_id="curve-dao-token",
    ),
    "LDO": TokenInfo(
        symbol="LDO",
        name="Lido DAO",
        chain="ethereum",
        address="0x5A98FcBEA516Cf06857215779Fd812CA3beF1B32",
        decimals=18,
        trading_pair="LDO/USDT",
        priority="medium",
        category="defi",
        coingecko_id="lido-dao",
    ),
    "COMP": TokenInfo(
        symbol="COMP",
        name="Compound",
        chain="ethereum",
        address="0xc00e94Cb662C3520282E6f5717214004A7f26888",
        decimals=18,
        trading_pair="COMP/USDT",
        priority="medium",
        category="defi",
        coingecko_id="compound-governance-token",
    ),
    "SUSHI": TokenInfo(
        symbol="SUSHI",
        name="SushiSwap",
        chain="ethereum",
        address="0x6B3595068778DD592e39A122f4f5a5cF09C90fE2",
        decimals=18,
        trading_pair="SUSHI/USDT",
        priority="medium",
        category="defi",
        coingecko_id="sushi",
    ),
    "1INCH": TokenInfo(
        symbol="1INCH",
        name="1inch",
        chain="ethereum",
        address="0x111111111117dC0aa78b770fA6A738034120C302",
        decimals=18,
        trading_pair="1INCH/USDT",
        priority="medium",
        category="defi",
        coingecko_id="1inch",
    ),

    # Layer 2 Tokens
    "ARB": TokenInfo(
        symbol="ARB",
        name="Arbitrum",
        chain="arbitrum",
        address="0x912CE59144191C1204E64559FE8253a0e49E6548",
        decimals=18,
        trading_pair="ARB/USDT",
        priority="high",
        category="l2",
        coingecko_id="arbitrum",
    ),
    "OP": TokenInfo(
        symbol="OP",
        name="Optimism",
        chain="ethereum",
        address="0x4200000000000000000000000000000000000042",
        decimals=18,
        trading_pair="OP/USDT",
        priority="high",
        category="l2",
        coingecko_id="optimism",
    ),
    "MATIC": TokenInfo(
        symbol="MATIC",
        name="Polygon",
        chain="ethereum",
        address="0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0",
        decimals=18,
        trading_pair="MATIC/USDT",
        priority="high",
        category="l2",
        coingecko_id="matic-network",
    ),

    # Perpetuals & DeFi (Volatile)
    "DYDX": TokenInfo(
        symbol="DYDX",
        name="dYdX",
        chain="ethereum",
        address="0x92D6C1e31e14520e676a687F0a93788B716BEff5",
        decimals=18,
        trading_pair="DYDX/USDT",
        priority="high",
        category="defi",
        coingecko_id="dydx",
    ),
    "GMX": TokenInfo(
        symbol="GMX",
        name="GMX",
        chain="arbitrum",
        address="0xfc5A1A6EB076a2C7aD06eD22C90d7E710E35ad0a",
        decimals=18,
        trading_pair="GMX/USDT",
        priority="high",
        category="defi",
        coingecko_id="gmx",
    ),
    "PENDLE": TokenInfo(
        symbol="PENDLE",
        name="Pendle",
        chain="ethereum",
        address="0x808507121B80c02388fAd14726482e061B8da827",
        decimals=18,
        trading_pair="PENDLE/USDT",
        priority="high",
        category="defi",
        coingecko_id="pendle",
    ),
    "FXS": TokenInfo(
        symbol="FXS",
        name="Frax Share",
        chain="ethereum",
        address="0x3432B6A60D23Ca0dFCa7761B7ab56459D9C964D0",
        decimals=18,
        trading_pair="FXS/USDT",
        priority="medium",
        category="defi",
        coingecko_id="frax-share",
    ),
    "ENS": TokenInfo(
        symbol="ENS",
        name="Ethereum Name Service",
        chain="ethereum",
        address="0xC18360217D8F7Ab5e7c516566761Ea12Ce7F9D72",
        decimals=18,
        trading_pair="ENS/USDT",
        priority="medium",
        category="infrastructure",
        coingecko_id="ethereum-name-service",
    ),

    # Gaming & Metaverse (High Volatility)
    "IMX": TokenInfo(
        symbol="IMX",
        name="Immutable X",
        chain="ethereum",
        address="0xF57e7e7C23978C3cAEC3C3548E3D615c346e79fF",
        decimals=18,
        trading_pair="IMX/USDT",
        priority="high",
        category="gaming",
        coingecko_id="immutable-x",
    ),
    "GALA": TokenInfo(
        symbol="GALA",
        name="Gala",
        chain="ethereum",
        address="0xd1d2Eb1B1e90B638588728b4130137D262C87cae",
        decimals=8,
        trading_pair="GALA/USDT",
        priority="medium",
        category="gaming",
        coingecko_id="gala",
    ),
    "SAND": TokenInfo(
        symbol="SAND",
        name="The Sandbox",
        chain="ethereum",
        address="0x3845badAde8e6dFF049820680d1F14bD3903a5d0",
        decimals=18,
        trading_pair="SAND/USDT",
        priority="medium",
        category="metaverse",
        coingecko_id="the-sandbox",
    ),
    "MANA": TokenInfo(
        symbol="MANA",
        name="Decentraland",
        chain="ethereum",
        address="0x0F5D2fB29fb7d3CFeE444a200298f468908cC942",
        decimals=18,
        trading_pair="MANA/USDT",
        priority="medium",
        category="metaverse",
        coingecko_id="decentraland",
    ),

    # AI Tokens (Very Volatile)
    "FET": TokenInfo(
        symbol="FET",
        name="Fetch.ai",
        chain="ethereum",
        address="0xaea46A60368A7bD060eec7DF8CBa43b7EF41Ad85",
        decimals=18,
        trading_pair="FET/USDT",
        priority="high",
        category="ai",
        coingecko_id="fetch-ai",
    ),
    "RNDR": TokenInfo(
        symbol="RNDR",
        name="Render",
        chain="ethereum",
        address="0x6De037ef9aD2725EB40118Bb1702EBb27e4Aeb24",
        decimals=18,
        trading_pair="RNDR/USDT",
        priority="high",
        category="ai",
        coingecko_id="render-token",
    ),
    "AGIX": TokenInfo(
        symbol="AGIX",
        name="SingularityNET",
        chain="ethereum",
        address="0x5B7533812759B45C2B44C19e320ba2cD2681b542",
        decimals=8,
        trading_pair="AGIX/USDT",
        priority="medium",
        category="ai",
        coingecko_id="singularitynet",
    ),
    "WLD": TokenInfo(
        symbol="WLD",
        name="Worldcoin",
        chain="ethereum",
        address="0x163f8C2467924be0ae7B5347228CABF260318753",
        decimals=18,
        trading_pair="WLD/USDT",
        priority="high",
        category="ai",
        coingecko_id="worldcoin-wld",
    ),

    # New L1s (High Volatility)
    "INJ": TokenInfo(
        symbol="INJ",
        name="Injective",
        chain="ethereum",
        address="0xe28b3B32B6c345A34Ff64674606124Dd5Aceca30",
        decimals=18,
        trading_pair="INJ/USDT",
        priority="high",
        category="l1",
        coingecko_id="injective-protocol",
    ),
    "SEI": TokenInfo(
        symbol="SEI",
        name="Sei",
        chain="sei",
        address="native",
        decimals=18,
        trading_pair="SEI/USDT",
        priority="high",
        category="l1",
        coingecko_id="sei-network",
    ),
    "SUI": TokenInfo(
        symbol="SUI",
        name="Sui",
        chain="sui",
        address="native",
        decimals=9,
        trading_pair="SUI/USDT",
        priority="high",
        category="l1",
        coingecko_id="sui",
    ),
    "APT": TokenInfo(
        symbol="APT",
        name="Aptos",
        chain="aptos",
        address="native",
        decimals=8,
        trading_pair="APT/USDT",
        priority="high",
        category="l1",
        coingecko_id="aptos",
    ),
    "TIA": TokenInfo(
        symbol="TIA",
        name="Celestia",
        chain="celestia",
        address="native",
        decimals=6,
        trading_pair="TIA/USDT",
        priority="high",
        category="l1",
        coingecko_id="celestia",
    ),

    # NFT & Others
    "BLUR": TokenInfo(
        symbol="BLUR",
        name="Blur",
        chain="ethereum",
        address="0x5283D291DBCF85356A21bA090E6db59121208b44",
        decimals=18,
        trading_pair="BLUR/USDT",
        priority="medium",
        category="nft",
        coingecko_id="blur",
    ),
}


def get_token_config(symbol: str) -> TokenInfo:
    """Get configuration for a specific token."""
    symbol_upper = symbol.upper()
    if symbol_upper not in TOKEN_CONFIG:
        raise ValueError(f"Unknown token: {symbol}")
    return TOKEN_CONFIG[symbol_upper]


def get_tokens_by_priority(priority: str) -> List[TokenInfo]:
    """Get all tokens with a specific priority."""
    return [t for t in TOKEN_CONFIG.values() if t.priority == priority]


def get_tokens_by_chain(chain: str) -> List[TokenInfo]:
    """Get all tokens on a specific chain."""
    chain_lower = chain.lower()
    return [t for t in TOKEN_CONFIG.values() if t.chain.lower() == chain_lower]


def get_tokens_by_category(category: str) -> List[TokenInfo]:
    """Get all tokens in a specific category."""
    return [t for t in TOKEN_CONFIG.values() if t.category == category]


def get_all_trading_pairs() -> List[str]:
    """Get all trading pairs for tracked tokens."""
    return [t.trading_pair for t in TOKEN_CONFIG.values()]
