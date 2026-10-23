"""
CryptoAlpha Configuration Management
Centralized configuration for all system components.
Loads settings from environment variables with sensible defaults.
"""
import os
from typing import List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
@dataclass
class ExchangeConfig:
    """Exchange API configuration."""
    # Binance (Primary)
    binance_api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    # Bybit (Secondary)
    bybit_api_key: str = field(default_factory=lambda: os.getenv("BYBIT_API_KEY", ""))
    bybit_api_secret: str = field(default_factory=lambda: os.getenv("BYBIT_API_SECRET", ""))
    # Coinbase (Secondary)
    coinbase_api_key: str = field(default_factory=lambda: os.getenv("COINBASE_API_KEY", ""))
    coinbase_api_secret: str = field(default_factory=lambda: os.getenv("COINBASE_API_SECRET", ""))
    # Default exchange
    default_exchange: str = "binance"
    # Rate limiting
    rate_limit_ms: int = 100  # Minimum ms between requests
@dataclass
class MarketDataConfig:
    """Market data API configuration."""
    # CoinMarketCap - Free: 10,000 calls/month, ~30 calls/min
    coinmarketcap_api_key: str = field(default_factory=lambda: os.getenv("COINMARKETCAP_API_KEY", ""))
    coinmarketcap_rate_limit: float = 2.0  # seconds between requests (30/min)
    # CoinGlass - Derivatives data (paid: $35/month min)
    coinglass_api_key: str = field(default_factory=lambda: os.getenv("COINGLASS_API_KEY", ""))
    coinglass_rate_limit: float = 1.0  # seconds between requests
    # Dune Analytics - Free: 2,500 credits/month
    dune_api_key: str = field(default_factory=lambda: os.getenv("DUNE_API_KEY", ""))
    dune_rate_limit: float = 1.0  # seconds between requests
    # Etherscan - Free: 5 calls/sec, 100,000 calls/day
    etherscan_api_key: str = field(default_factory=lambda: os.getenv("ETHERSCAN_API_KEY", ""))
    etherscan_rate_limit: float = 0.2  # seconds between requests (5/sec)
@dataclass
class OnChainConfig:
    """On-chain RPC configuration."""
    # RPC endpoints for each chain
    eth_rpc_url: str = field(default_factory=lambda: os.getenv("ETH_RPC_URL", "https://eth.llamarpc.com"))
    arb_rpc_url: str = field(default_factory=lambda: os.getenv("ARB_RPC_URL", "https://arb1.arbitrum.io/rpc"))
    base_rpc_url: str = field(default_factory=lambda: os.getenv("BASE_RPC_URL", "https://mainnet.base.org"))
    polygon_rpc_url: str = field(default_factory=lambda: os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"))
    # Alchemy API key (optional, for premium RPCs)
    alchemy_api_key: str = field(default_factory=lambda: os.getenv("ALCHEMY_API_KEY", ""))
    # Request settings
    request_timeout: int = 30  # seconds
    max_retries: int = 3
    def get_rpc_url(self, chain: str) -> str:
        """Get RPC URL for a specific chain."""
        chain_map = {
            "ethereum": self.eth_rpc_url,
            "eth": self.eth_rpc_url,
            "arbitrum": self.arb_rpc_url,
            "arb": self.arb_rpc_url,
            "base": self.base_rpc_url,
            "polygon": self.polygon_rpc_url,
            "matic": self.polygon_rpc_url,
        }
        return chain_map.get(chain.lower(), self.eth_rpc_url)
@dataclass
class LLMConfig:
    """LLM (Gemini) configuration."""
    # API keys (comma-separated for multiple keys)
    gemini_api_keys: List[str] = field(default_factory=lambda: [
        k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()
    ])
    # Single key fallback
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    # Model settings - load from env or use default
    model_name: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_output_tokens: int = 8192
    # Key rotation tracking
    _current_key_index: int = field(default=0, repr=False)
    def get_api_key(self, index: int = None) -> str:
        """Get API key by index (for rotation)."""
        if index is None:
            index = self._current_key_index
        if self.gemini_api_keys:
            return self.gemini_api_keys[index % len(self.gemini_api_keys)]
        return self.google_api_key
    def rotate_key(self) -> str:
        """Get next API key in rotation."""
        if self.gemini_api_keys:
            self._current_key_index = (self._current_key_index + 1) % len(self.gemini_api_keys)
            return self.gemini_api_keys[self._current_key_index]
        return self.google_api_key
    def get_random_key(self) -> str:
        """Get a random API key."""
        import random
        if self.gemini_api_keys:
            return random.choice(self.gemini_api_keys)
        return self.google_api_key
    @property
    def key_count(self) -> int:
        """Get the number of available API keys."""
        return len(self.gemini_api_keys) if self.gemini_api_keys else (1 if self.google_api_key else 0)
@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    # Alert settings
    enable_alerts: bool = True
    alert_on_signal: bool = True
    alert_on_error: bool = True
@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Position sizing
    max_position_size: float = 0.02  # 2% of portfolio per trade
    max_open_positions: int = 5
    max_correlation: float = 0.7
    # Stop-loss
    default_stop_loss: float = 0.05  # 5%
    volatility_adjusted_sl: bool = True
    # Signal thresholds
    min_confidence: float = 0.65  # 65% minimum confidence
    min_rr_ratio: float = 1.5  # Minimum risk/reward ratio
    cooldown_hours: int = 4  # Hours between signals on same asset
    # Drawdown limits
    max_daily_drawdown: float = 0.03  # 3%
    max_total_drawdown: float = 0.10  # 10%
@dataclass
class AnalysisConfig:
    """Analysis and monitoring configuration."""
    # Timing
    analysis_interval_minutes: int = 60
    # Technical analysis timeframes
    timeframes: List[str] = field(default_factory=lambda: ["15m", "1h", "4h", "1d"])
    # Data retention
    ohlcv_1m_retention_days: int = 7
    ohlcv_1h_retention_days: int = 90
    ohlcv_4h_retention_days: int = 180
    # On-chain settings
    whale_threshold_usd: float = 100000  # $100k for large transfer detection
    top_holders_count: int = 100
@dataclass
class CacheConfig:
    """Caching configuration."""
    # Cache directory
    cache_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".cache"
    ))
    # TTL (Time To Live) in seconds
    ohlcv_ttl: int = 60  # 1 minute
    orderbook_ttl: int = 10  # 10 seconds
    onchain_ttl: int = 300  # 5 minutes
    # Enable/disable
    enable_cache: bool = True
@dataclass
class DatabaseConfig:
    """Database configuration."""
    # SQLite for signal history
    sqlite_path: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "signals.db"
    ))
    # ChromaDB for vector memory
    chromadb_path: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "chromadb"
    ))
@dataclass
class AppConfig:
    """Main application configuration."""
    # Environment
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    # Sub-configurations
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    onchain: OnChainConfig = field(default_factory=OnChainConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        # Check Gemini API keys (required for LLM agents)
        if not self.llm.gemini_api_keys and not self.llm.google_api_key:
            issues.append("No Gemini API keys configured")
        # Check Telegram bot token (required for alerts)
        if not self.telegram.bot_token:
            issues.append("TELEGRAM_BOT_TOKEN not set")
        # Check market data APIs (at least one should be configured)
        has_market_data = any([
            self.market_data.coinmarketcap_api_key,
            self.market_data.coinglass_api_key,
            self.market_data.etherscan_api_key,
        ])
        if not has_market_data:
            issues.append("No market data API keys configured")
        return issues
    def get_available_data_sources(self) -> List[str]:
        """Get list of available data sources based on configured API keys."""
        sources = []
        if self.market_data.coinmarketcap_api_key:
            sources.append("coinmarketcap")
        if self.market_data.coinglass_api_key:
            sources.append("coinglass")
        if self.market_data.etherscan_api_key:
            sources.append("etherscan")
        if self.market_data.dune_api_key:
            sources.append("dune")
        if self.exchange.binance_api_key:
            sources.append("binance")
        if self.exchange.bybit_api_key:
            sources.append("bybit")
        return sources
# Global configuration instance
config = AppConfig()
# Token configurations
TRACKED_TOKENS = {
    # Major Cryptocurrencies
    "BTC": {"symbol": "BTC/USDT", "chain": "bitcoin", "address": "native", "priority": "high", "type": "crypto"},
    "ETH": {"symbol": "ETH/USDT", "chain": "ethereum", "address": "native", "priority": "high", "type": "crypto"},
    "XRP": {"symbol": "XRP/USDT", "chain": "ripple", "address": "native", "priority": "high", "type": "crypto"},
    "LUNC": {"symbol": "LUNC/USDT", "chain": "terra", "address": "native", "priority": "high", "type": "crypto"},
    "PAXG": {"symbol": "PAXG/USDT", "chain": "ethereum", "address": "0x45804880De22913dAFE09f4980848ECE6EcbAf78", "priority": "high", "type": "crypto"},

    # Forex pairs (via Frankfurter API)
}
def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config
def load_config() -> AppConfig:
    """Alias for get_config() - for compatibility."""
    return config
if __name__ == "__main__":
    # Test configuration loading
    cfg = get_config()
    print(f"Environment: {cfg.environment}")
    print(f"Debug: {cfg.debug}")
    print(f"Log Level: {cfg.log_level}")
    issues = cfg.validate()
    if issues:
        print("\nConfiguration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration OK!")
