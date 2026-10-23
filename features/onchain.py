"""
On-Chain Features

Generates features from on-chain data:
- Whale activity metrics
- Exchange flow features
- Holder distribution features
- Network activity metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.onchain_client import OnChainClient
from data.wallet_tracker import WalletTracker


class OnChainFeatures:
    """
    On-chain feature generator.

    Generates features from blockchain data for trading signals.
    """

    def __init__(
        self,
        chains: Optional[List[str]] = None,
    ):
        """
        Initialize the on-chain features generator.

        Args:
            chains: List of chains to use
        """
        self.chains = chains or ["ethereum"]
        self.clients: Dict[str, OnChainClient] = {}
        self.wallet_tracker: Optional[WalletTracker] = None

        # Initialize clients
        for chain in self.chains:
            try:
                self.clients[chain] = OnChainClient(chain=chain)
            except Exception as e:
                print(f"Warning: Could not initialize {chain} client: {e}")

    def _get_client(self, chain: str) -> OnChainClient:
        """Get client for a chain."""
        if chain not in self.clients:
            raise ValueError(f"Chain not initialized: {chain}")
        return self.clients[chain]

    # =========================================================================
    # Whale Features
    # =========================================================================

    def get_whale_features(
        self,
        token_address: str,
        chain: str = "ethereum",
    ) -> Dict[str, float]:
        """
        Generate whale-related features.

        Args:
            token_address: Token contract address
            chain: Chain to query

        Returns:
            Dictionary of whale features
        """
        client = self._get_client(chain)

        try:
            whale_activity = client.get_whale_activity(token_address)
            distribution = client.get_holder_distribution(token_address)

            features = {
                # Whale holdings
                "whale_count": whale_activity.get("whale_count", 0),
                "whale_holdings_pct": whale_activity.get("whale_holdings_pct", 0),

                # Whale activity
                "whale_buying": whale_activity.get("whale_buying", 0),
                "whale_selling": whale_activity.get("whale_selling", 0),
                "whale_net": whale_activity.get("whale_net", 0),
                "whale_accumulating": 1 if whale_activity.get("is_whale_accumulating") else 0,

                # Top whale
                "top_whale_pct": whale_activity.get("top_whale_pct", 0),

                # Distribution
                "top_10_holdings_pct": distribution.get("top_10_pct", 0),
                "top_50_holdings_pct": distribution.get("top_50_pct", 0),
                "gini_coefficient": distribution.get("gini_coefficient", 0),
                "is_concentrated": 1 if distribution.get("is_concentrated") else 0,
            }

            return features

        except Exception as e:
            print(f"Error getting whale features: {e}")
            return self._get_empty_whale_features()

    def _get_empty_whale_features(self) -> Dict[str, float]:
        """Return empty whale features."""
        return {
            "whale_count": 0,
            "whale_holdings_pct": 0,
            "whale_buying": 0,
            "whale_selling": 0,
            "whale_net": 0,
            "whale_accumulating": 0,
            "top_whale_pct": 0,
            "top_10_holdings_pct": 0,
            "top_50_holdings_pct": 0,
            "gini_coefficient": 0,
            "is_concentrated": 0,
        }

    # =========================================================================
    # Exchange Flow Features
    # =========================================================================

    def get_exchange_flow_features(
        self,
        token_address: str,
        chain: str = "ethereum",
    ) -> Dict[str, float]:
        """
        Generate exchange flow features.

        Args:
            token_address: Token contract address
            chain: Chain to query

        Returns:
            Dictionary of exchange flow features
        """
        client = self._get_client(chain)

        try:
            flow = client.get_exchange_flow(token_address)

            features = {
                # Raw flows
                "exchange_inflow": flow.get("inflow", 0),
                "exchange_outflow": flow.get("outflow", 0),
                "exchange_net_flow": flow.get("net_flow", 0),

                # Counts
                "inflow_count": flow.get("inflow_count", 0),
                "outflow_count": flow.get("outflow_count", 0),

                # Ratios
                "flow_ratio": min(flow.get("flow_ratio", 1), 10),  # Cap at 10
                "is_accumulating": 1 if flow.get("is_accumulating") else 0,
            }

            # Normalized features
            total_flow = features["exchange_inflow"] + features["exchange_outflow"]
            if total_flow > 0:
                features["inflow_pct"] = features["exchange_inflow"] / total_flow
                features["outflow_pct"] = features["exchange_outflow"] / total_flow
            else:
                features["inflow_pct"] = 0.5
                features["outflow_pct"] = 0.5

            return features

        except Exception as e:
            print(f"Error getting exchange flow features: {e}")
            return self._get_empty_flow_features()

    def _get_empty_flow_features(self) -> Dict[str, float]:
        """Return empty flow features."""
        return {
            "exchange_inflow": 0,
            "exchange_outflow": 0,
            "exchange_net_flow": 0,
            "inflow_count": 0,
            "outflow_count": 0,
            "flow_ratio": 1,
            "is_accumulating": 0,
            "inflow_pct": 0.5,
            "outflow_pct": 0.5,
        }

    # =========================================================================
    # Holder Distribution Features
    # =========================================================================

    def get_distribution_features(
        self,
        token_address: str,
        chain: str = "ethereum",
    ) -> Dict[str, float]:
        """
        Generate holder distribution features.

        Args:
            token_address: Token contract address
            chain: Chain to query

        Returns:
            Dictionary of distribution features
        """
        client = self._get_client(chain)

        try:
            dist = client.get_holder_distribution(token_address)

            features = {
                "holder_count": dist.get("holder_count", 0),
                "top_10_pct": dist.get("top_10_pct", 0),
                "top_50_pct": dist.get("top_50_pct", 0),
                "top_100_pct": dist.get("top_100_pct", 0),
                "exchange_holdings_pct": dist.get("exchange_holdings_pct", 0),
                "contract_holdings_pct": dist.get("contract_holdings_pct", 0),
                "gini": dist.get("gini_coefficient", 0),
                "concentrated": 1 if dist.get("is_concentrated") else 0,
            }

            # Derived features
            features["retail_holdings_pct"] = max(0, 100 - features["top_100_pct"])
            features["non_exchange_top_10"] = features["top_10_pct"] - features["exchange_holdings_pct"]

            return features

        except Exception as e:
            print(f"Error getting distribution features: {e}")
            return self._get_empty_distribution_features()

    def _get_empty_distribution_features(self) -> Dict[str, float]:
        """Return empty distribution features."""
        return {
            "holder_count": 0,
            "top_10_pct": 0,
            "top_50_pct": 0,
            "top_100_pct": 0,
            "exchange_holdings_pct": 0,
            "contract_holdings_pct": 0,
            "gini": 0,
            "concentrated": 0,
            "retail_holdings_pct": 0,
            "non_exchange_top_10": 0,
        }

    # =========================================================================
    # Combined Score Features
    # =========================================================================

    def get_onchain_score(
        self,
        token_address: str,
        chain: str = "ethereum",
    ) -> Dict[str, float]:
        """
        Calculate on-chain health scores.

        Args:
            token_address: Token contract address
            chain: Chain to query

        Returns:
            Dictionary with various scores
        """
        whale = self.get_whale_features(token_address, chain)
        flow = self.get_exchange_flow_features(token_address, chain)
        dist = self.get_distribution_features(token_address, chain)

        # Calculate scores (0-100)

        # Accumulation score
        accumulation_score = 50
        if whale["whale_accumulating"]:
            accumulation_score += 20
        if flow["is_accumulating"]:
            accumulation_score += 20
        if whale["whale_net"] > 0:
            accumulation_score += min(10, whale["whale_net"] / 1000)

        # Distribution health score
        dist_score = 50
        if dist["gini"] < 0.5:
            dist_score += 20
        if dist["top_10_pct"] < 30:
            dist_score += 15
        if dist["exchange_holdings_pct"] < 20:
            dist_score += 15

        # Flow score
        flow_score = 50
        if flow["flow_ratio"] > 1.5:
            flow_score += 25
        elif flow["flow_ratio"] > 1.2:
            flow_score += 15
        elif flow["flow_ratio"] < 0.8:
            flow_score -= 15
        elif flow["flow_ratio"] < 0.5:
            flow_score -= 25

        # Overall score
        overall_score = (accumulation_score * 0.4 +
                         dist_score * 0.3 +
                         flow_score * 0.3)

        return {
            "accumulation_score": max(0, min(100, accumulation_score)),
            "distribution_score": max(0, min(100, dist_score)),
            "flow_score": max(0, min(100, flow_score)),
            "overall_onchain_score": max(0, min(100, overall_score)),

            # Sentiment
            "onchain_sentiment": (
                "bullish" if overall_score >= 60 else
                "bearish" if overall_score <= 40 else
                "neutral"
            ),
        }

    # =========================================================================
    # All Features
    # =========================================================================

    def get_all_features(
        self,
        token_address: str,
        chain: str = "ethereum",
    ) -> Dict[str, float]:
        """
        Get all on-chain features for a token.

        Args:
            token_address: Token contract address
            chain: Chain to query

        Returns:
            Dictionary with all features
        """
        features = {}

        # Whale features
        whale = self.get_whale_features(token_address, chain)
        features.update({f"whale_{k}": v for k, v in whale.items()})

        # Exchange flow features
        flow = self.get_exchange_flow_features(token_address, chain)
        features.update({f"flow_{k}": v for k, v in flow.items()})

        # Distribution features
        dist = self.get_distribution_features(token_address, chain)
        features.update({f"dist_{k}": v for k, v in dist.items()})

        # Score features
        scores = self.get_onchain_score(token_address, chain)
        features.update(scores)

        return features

    def features_to_dataframe(
        self,
        features: Dict[str, float],
    ) -> pd.DataFrame:
        """Convert features dict to single-row DataFrame."""
        return pd.DataFrame([features])


def create_onchain_features(chains: Optional[List[str]] = None) -> OnChainFeatures:
    """Create an OnChainFeatures instance."""
    return OnChainFeatures(chains=chains)


if __name__ == "__main__":
    print("Testing OnChainFeatures...")

    ocf = OnChainFeatures(chains=["ethereum"])

    # Test with AAVE
    aave_address = "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9"

    print("\nGetting all features for AAVE...")
    features = ocf.get_all_features(aave_address)

    print(f"\nTotal features: {len(features)}")
    print(f"Overall score: {features.get('overall_onchain_score', 0):.1f}")
    print(f"Sentiment: {features.get('onchain_sentiment', 'unknown')}")
