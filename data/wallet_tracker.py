"""
Wallet Tracker

Monitors whale wallets and tracks their activity patterns.
Provides alerts for significant movements.
"""

import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd

from .onchain_client import OnChainClient, TransferEvent, HolderInfo
from .cache import DataCache, get_cache


@dataclass
class WalletProfile:
    """Profile of a tracked wallet."""
    address: str
    label: Optional[str]
    first_seen: datetime
    last_active: datetime
    total_bought: float
    total_sold: float
    current_balance: float
    transaction_count: int
    is_whale: bool
    is_smart_money: bool  # Based on historical performance
    tags: List[str] = field(default_factory=list)


@dataclass
class WalletAlert:
    """Alert for significant wallet activity."""
    timestamp: datetime
    wallet_address: str
    wallet_label: Optional[str]
    alert_type: str  # "large_buy", "large_sell", "new_position", "exit_position"
    token_symbol: str
    amount: float
    amount_usd: Optional[float]
    significance: str  # "low", "medium", "high", "critical"
    details: Dict[str, Any] = field(default_factory=dict)


class WalletTracker:
    """
    Tracks whale wallets and monitors their activity.

    Features:
    - Monitor known whale addresses
    - Detect new whale positions
    - Track accumulation/distribution patterns
    - Generate alerts for significant activity
    """

    def __init__(
        self,
        chains: Optional[List[str]] = None,
        cache: Optional[DataCache] = None,
    ):
        """
        Initialize the wallet tracker.

        Args:
            chains: List of chains to monitor
            cache: Optional cache instance
        """
        self.chains = chains or ["ethereum", "arbitrum", "base", "polygon"]
        self.cache = cache or get_cache()

        # Initialize clients for each chain
        self.clients: Dict[str, OnChainClient] = {}
        for chain in self.chains:
            try:
                self.clients[chain] = OnChainClient(chain=chain, cache=self.cache)
            except Exception as e:
                print(f"Warning: Could not initialize client for {chain}: {e}")

        # Tracked wallets
        self.tracked_wallets: Dict[str, WalletProfile] = {}

        # Alert thresholds
        self.large_transfer_threshold = 100000  # USD
        self.whale_threshold_pct = 0.1  # % of supply

        # Alert history
        self.alerts: List[WalletAlert] = []

    def add_wallet(
        self,
        address: str,
        label: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Add a wallet to tracking."""
        address_lower = address.lower()

        if address_lower not in self.tracked_wallets:
            self.tracked_wallets[address_lower] = WalletProfile(
                address=address,
                label=label,
                first_seen=datetime.now(),
                last_active=datetime.now(),
                total_bought=0,
                total_sold=0,
                current_balance=0,
                transaction_count=0,
                is_whale=False,
                is_smart_money=False,
                tags=tags or [],
            )

    def remove_wallet(self, address: str) -> None:
        """Remove a wallet from tracking."""
        self.tracked_wallets.pop(address.lower(), None)

    def get_tracked_wallets(self) -> List[WalletProfile]:
        """Get all tracked wallets."""
        return list(self.tracked_wallets.values())

    # =========================================================================
    # Whale Detection
    # =========================================================================

    def discover_whales(
        self,
        token_address: str,
        chain: str = "ethereum",
        min_holdings_pct: float = 0.1,
    ) -> List[HolderInfo]:
        """
        Discover whale wallets for a token.

        Args:
            token_address: Token contract address
            chain: Chain to query
            min_holdings_pct: Minimum % of supply to be whale

        Returns:
            List of whale HolderInfo objects
        """
        if chain not in self.clients:
            raise ValueError(f"Chain not initialized: {chain}")

        client = self.clients[chain]
        holders = client.get_top_holders(token_address, limit=100)

        whales = [h for h in holders if h.percentage >= min_holdings_pct]

        # Auto-track discovered whales
        for whale in whales:
            if whale.address.lower() not in self.tracked_wallets:
                self.add_wallet(
                    address=whale.address,
                    label=whale.label,
                    tags=["whale", "auto_discovered"],
                )
                self.tracked_wallets[whale.address.lower()].is_whale = True

        return whales

    # =========================================================================
    # Activity Monitoring
    # =========================================================================

    def check_wallet_activity(
        self,
        token_address: str,
        chain: str = "ethereum",
        blocks: int = 1000,
    ) -> List[WalletAlert]:
        """
        Check for significant activity from tracked wallets.

        Args:
            token_address: Token to check
            chain: Chain to query
            blocks: Number of blocks to analyze

        Returns:
            List of new alerts
        """
        if chain not in self.clients:
            return []

        client = self.clients[chain]
        transfers = client.get_recent_transfers(token_address, blocks, use_cache=False)

        new_alerts = []
        tracked_addresses = set(self.tracked_wallets.keys())

        for transfer in transfers:
            from_addr = transfer.from_address.lower()
            to_addr = transfer.to_address.lower()

            # Check if this involves a tracked wallet
            if from_addr in tracked_addresses:
                alert = self._create_alert(
                    transfer, "sell", self.tracked_wallets[from_addr]
                )
                if alert:
                    new_alerts.append(alert)
                    self._update_wallet_stats(from_addr, transfer, "sell")

            if to_addr in tracked_addresses:
                alert = self._create_alert(
                    transfer, "buy", self.tracked_wallets[to_addr]
                )
                if alert:
                    new_alerts.append(alert)
                    self._update_wallet_stats(to_addr, transfer, "buy")

        self.alerts.extend(new_alerts)
        return new_alerts

    def _create_alert(
        self,
        transfer: TransferEvent,
        direction: str,
        wallet: WalletProfile,
    ) -> Optional[WalletAlert]:
        """Create an alert for a transfer if significant."""
        # Determine significance
        # This is simplified - would use USD value in production
        if transfer.value < 1000:
            return None

        if transfer.value >= 100000:
            significance = "critical"
        elif transfer.value >= 10000:
            significance = "high"
        elif transfer.value >= 1000:
            significance = "medium"
        else:
            significance = "low"

        alert_type = f"large_{direction}"

        return WalletAlert(
            timestamp=datetime.now(),
            wallet_address=wallet.address,
            wallet_label=wallet.label,
            alert_type=alert_type,
            token_symbol="UNKNOWN",  # Would need to look up
            amount=transfer.value,
            amount_usd=transfer.value_usd,
            significance=significance,
            details={
                "tx_hash": transfer.tx_hash,
                "block": transfer.block_number,
                "is_exchange_flow": transfer.is_exchange_inflow or transfer.is_exchange_outflow,
            },
        )

    def _update_wallet_stats(
        self,
        address: str,
        transfer: TransferEvent,
        direction: str,
    ) -> None:
        """Update wallet statistics after a transfer."""
        wallet = self.tracked_wallets.get(address)
        if not wallet:
            return

        wallet.last_active = datetime.now()
        wallet.transaction_count += 1

        if direction == "buy":
            wallet.total_bought += transfer.value
        else:
            wallet.total_sold += transfer.value

    # =========================================================================
    # Smart Money Analysis
    # =========================================================================

    def analyze_smart_money(
        self,
        token_address: str,
        chain: str = "ethereum",
    ) -> Dict[str, Any]:
        """
        Analyze smart money behavior for a token.

        Args:
            token_address: Token to analyze
            chain: Chain to query

        Returns:
            Dictionary with smart money metrics
        """
        if chain not in self.clients:
            return {}

        client = self.clients[chain]

        # Get whale activity
        whale_activity = client.get_whale_activity(
            token_address,
            whale_threshold_pct=self.whale_threshold_pct,
        )

        # Get exchange flow
        exchange_flow = client.get_exchange_flow(token_address)

        # Determine sentiment
        whale_sentiment = "bullish" if whale_activity["is_whale_accumulating"] else "bearish"
        exchange_sentiment = "bullish" if exchange_flow["is_accumulating"] else "bearish"

        # Overall smart money sentiment
        if whale_sentiment == exchange_sentiment:
            overall_sentiment = whale_sentiment
            confidence = "high"
        else:
            overall_sentiment = "neutral"
            confidence = "low"

        return {
            "token_address": token_address,
            "chain": chain,
            "whale_activity": whale_activity,
            "exchange_flow": exchange_flow,
            "whale_sentiment": whale_sentiment,
            "exchange_sentiment": exchange_sentiment,
            "overall_sentiment": overall_sentiment,
            "confidence": confidence,
            "signals": {
                "whale_accumulating": whale_activity["is_whale_accumulating"],
                "exchange_outflow": exchange_flow["is_accumulating"],
                "concentrated_holdings": whale_activity["whale_holdings_pct"] > 30,
            },
        }

    # =========================================================================
    # Aggregated Metrics
    # =========================================================================

    def get_whale_score(
        self,
        token_address: str,
        chain: str = "ethereum",
    ) -> float:
        """
        Calculate a whale activity score (0-100).

        Args:
            token_address: Token to analyze
            chain: Chain to query

        Returns:
            Whale score from 0 (bearish) to 100 (bullish)
        """
        smart_money = self.analyze_smart_money(token_address, chain)

        if not smart_money:
            return 50  # Neutral if no data

        score = 50  # Start neutral

        whale = smart_money.get("whale_activity", {})
        flow = smart_money.get("exchange_flow", {})

        # Whale accumulation (+/- 20 points)
        if whale.get("is_whale_accumulating"):
            score += 20
        else:
            score -= 20

        # Exchange outflow (+/- 15 points)
        if flow.get("is_accumulating"):
            score += 15
        else:
            score -= 15

        # Whale transaction volume (+/- 10 points)
        whale_net = whale.get("whale_net", 0)
        if whale_net > 0:
            score += min(10, whale_net / 1000)
        else:
            score -= min(10, abs(whale_net) / 1000)

        # Flow ratio (+/- 5 points)
        flow_ratio = flow.get("flow_ratio", 1)
        if flow_ratio > 1.5:
            score += 5
        elif flow_ratio < 0.67:
            score -= 5

        return max(0, min(100, score))

    def get_recent_alerts(
        self,
        hours: int = 24,
        min_significance: str = "medium",
    ) -> List[WalletAlert]:
        """Get recent alerts filtered by time and significance."""
        cutoff = datetime.now() - timedelta(hours=hours)

        significance_order = ["low", "medium", "high", "critical"]
        min_idx = significance_order.index(min_significance)

        return [
            a for a in self.alerts
            if a.timestamp >= cutoff
            and significance_order.index(a.significance) >= min_idx
        ]


def create_wallet_tracker(chains: Optional[List[str]] = None) -> WalletTracker:
    """Create a wallet tracker for the specified chains."""
    return WalletTracker(chains=chains)


if __name__ == "__main__":
    print("Testing WalletTracker...")

    tracker = WalletTracker(chains=["ethereum"])

    # Add a known whale wallet
    tracker.add_wallet(
        "0x28C6c06298d514Db089934071355E5743bf21d60",
        label="Binance Hot Wallet",
        tags=["exchange", "binance"],
    )

    print(f"Tracked wallets: {len(tracker.get_tracked_wallets())}")

    # Test smart money analysis
    aave_address = "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9"
    score = tracker.get_whale_score(aave_address)
    print(f"AAVE Whale Score: {score:.1f}/100")
