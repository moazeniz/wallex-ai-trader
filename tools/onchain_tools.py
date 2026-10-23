"""
On-Chain Analysis Tools

LangChain-compatible tools for on-chain analysis.
These tools are used by the On-Chain Analyst Agent.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.onchain_client import OnChainClient
from data.wallet_tracker import WalletTracker
from features.onchain import OnChainFeatures


class GetWhaleActivityInput(BaseModel):
    """Input for getting whale activity."""
    token_address: str = Field(description="Token contract address")
    chain: str = Field(default="ethereum", description="Chain name (ethereum, arbitrum, base, polygon)")


class GetExchangeFlowInput(BaseModel):
    """Input for getting exchange flow data."""
    token_address: str = Field(description="Token contract address")
    chain: str = Field(default="ethereum", description="Chain name")


class GetHolderDistributionInput(BaseModel):
    """Input for getting holder distribution."""
    token_address: str = Field(description="Token contract address")
    chain: str = Field(default="ethereum", description="Chain name")


class OnChainTools:
    """
    On-chain analysis tools for LLM agents.

    Provides tools for querying blockchain data and wallet analysis.
    """

    def __init__(self, chains: Optional[List[str]] = None):
        """Initialize on-chain tools."""
        self.chains = chains or ["ethereum"]
        self.onchain_features = OnChainFeatures(chains=self.chains)
        self.wallet_tracker = WalletTracker(chains=self.chains)

    def get_whale_activity(
        self,
        token_address: str,
        chain: str = "ethereum",
    ) -> Dict[str, Any]:
        """
        Get whale wallet activity for a token.

        Args:
            token_address: Token contract address
            chain: Blockchain name

        Returns:
            Dictionary with whale activity data
        """
        try:
            features = self.onchain_features.get_whale_features(token_address, chain)

            # Interpret results
            whale_behavior = "neutral"
            if features.get("whale_accumulating", 0):
                whale_behavior = "accumulating"
            elif features.get("whale_net", 0) < 0:
                whale_behavior = "distributing"

            concentration_risk = "low"
            top_10_pct = features.get("top_10_holdings_pct", 0)
            if top_10_pct > 50:
                concentration_risk = "high"
            elif top_10_pct > 30:
                concentration_risk = "medium"

            return {
                "success": True,
                "token_address": token_address,
                "chain": chain,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "whale_count": int(features.get("whale_count", 0)),
                    "whale_holdings_pct": round(features.get("whale_holdings_pct", 0), 2),
                    "whale_buying": round(features.get("whale_buying", 0), 2),
                    "whale_selling": round(features.get("whale_selling", 0), 2),
                    "whale_net": round(features.get("whale_net", 0), 2),
                    "top_whale_pct": round(features.get("top_whale_pct", 0), 2),
                },
                "analysis": {
                    "whale_behavior": whale_behavior,
                    "is_accumulating": bool(features.get("whale_accumulating", 0)),
                    "concentration_risk": concentration_risk,
                    "top_10_holdings_pct": round(top_10_pct, 2),
                    "gini_coefficient": round(features.get("gini_coefficient", 0), 3),
                },
                "interpretation": self._interpret_whale_activity(features),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_exchange_flow(
        self,
        token_address: str,
        chain: str = "ethereum",
    ) -> Dict[str, Any]:
        """
        Get exchange inflow/outflow data.

        Args:
            token_address: Token contract address
            chain: Blockchain name

        Returns:
            Dictionary with exchange flow data
        """
        try:
            features = self.onchain_features.get_exchange_flow_features(token_address, chain)

            # Interpret flow
            net_flow = features.get("exchange_net_flow", 0)
            if net_flow > 0:
                flow_direction = "inflow"
                flow_interpretation = "More tokens moving TO exchanges - potential selling pressure"
            elif net_flow < 0:
                flow_direction = "outflow"
                flow_interpretation = "More tokens moving FROM exchanges - potential accumulation"
            else:
                flow_direction = "balanced"
                flow_interpretation = "Balanced flow between exchanges and wallets"

            flow_ratio = features.get("flow_ratio", 1)
            if flow_ratio > 2:
                flow_strength = "strong_outflow"
            elif flow_ratio > 1.2:
                flow_strength = "moderate_outflow"
            elif flow_ratio < 0.5:
                flow_strength = "strong_inflow"
            elif flow_ratio < 0.8:
                flow_strength = "moderate_inflow"
            else:
                flow_strength = "balanced"

            return {
                "success": True,
                "token_address": token_address,
                "chain": chain,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "inflow": round(features.get("exchange_inflow", 0), 2),
                    "outflow": round(features.get("exchange_outflow", 0), 2),
                    "net_flow": round(net_flow, 2),
                    "flow_ratio": round(flow_ratio, 2),
                    "inflow_count": int(features.get("inflow_count", 0)),
                    "outflow_count": int(features.get("outflow_count", 0)),
                },
                "analysis": {
                    "flow_direction": flow_direction,
                    "flow_strength": flow_strength,
                    "is_accumulating": bool(features.get("is_accumulating", 0)),
                    "inflow_pct": round(features.get("inflow_pct", 0.5) * 100, 1),
                    "outflow_pct": round(features.get("outflow_pct", 0.5) * 100, 1),
                },
                "interpretation": flow_interpretation,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_holder_distribution(
        self,
        token_address: str,
        chain: str = "ethereum",
    ) -> Dict[str, Any]:
        """
        Get token holder distribution analysis.

        Args:
            token_address: Token contract address
            chain: Blockchain name

        Returns:
            Dictionary with holder distribution data
        """
        try:
            features = self.onchain_features.get_distribution_features(token_address, chain)

            # Analyze distribution health
            gini = features.get("gini", 0)
            top_10_pct = features.get("top_10_pct", 0)
            exchange_pct = features.get("exchange_holdings_pct", 0)

            # Distribution health score (0-100)
            health_score = 50
            if gini < 0.5:
                health_score += 20
            if top_10_pct < 30:
                health_score += 15
            if exchange_pct < 20:
                health_score += 15

            if gini > 0.8:
                health_score -= 20
            if top_10_pct > 60:
                health_score -= 20

            health_score = max(0, min(100, health_score))

            if health_score >= 70:
                distribution_health = "healthy"
            elif health_score >= 50:
                distribution_health = "moderate"
            else:
                distribution_health = "concentrated"

            return {
                "success": True,
                "token_address": token_address,
                "chain": chain,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "holder_count": int(features.get("holder_count", 0)),
                    "top_10_pct": round(top_10_pct, 2),
                    "top_50_pct": round(features.get("top_50_pct", 0), 2),
                    "top_100_pct": round(features.get("top_100_pct", 0), 2),
                    "exchange_holdings_pct": round(exchange_pct, 2),
                    "retail_holdings_pct": round(features.get("retail_holdings_pct", 0), 2),
                    "gini_coefficient": round(gini, 3),
                },
                "analysis": {
                    "distribution_health": distribution_health,
                    "health_score": health_score,
                    "is_concentrated": bool(features.get("concentrated", 0)),
                    "whale_dominated": top_10_pct > 50,
                    "exchange_heavy": exchange_pct > 30,
                },
                "interpretation": self._interpret_distribution(features),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_onchain_score(
        self,
        token_address: str,
        chain: str = "ethereum",
    ) -> Dict[str, Any]:
        """
        Get comprehensive on-chain health score.

        Args:
            token_address: Token contract address
            chain: Blockchain name

        Returns:
            Dictionary with overall on-chain analysis
        """
        try:
            scores = self.onchain_features.get_onchain_score(token_address, chain)
            all_features = self.onchain_features.get_all_features(token_address, chain)

            overall_score = scores.get("overall_onchain_score", 50)
            sentiment = scores.get("onchain_sentiment", "neutral")

            # Trading bias
            if overall_score >= 65:
                trading_bias = "bullish"
                bias_confidence = min(100, (overall_score - 50) * 2)
            elif overall_score <= 35:
                trading_bias = "bearish"
                bias_confidence = min(100, (50 - overall_score) * 2)
            else:
                trading_bias = "neutral"
                bias_confidence = 100 - abs(overall_score - 50) * 2

            return {
                "success": True,
                "token_address": token_address,
                "chain": chain,
                "timestamp": datetime.now().isoformat(),
                "scores": {
                    "overall": round(overall_score, 1),
                    "accumulation": round(scores.get("accumulation_score", 50), 1),
                    "distribution_health": round(scores.get("distribution_score", 50), 1),
                    "flow": round(scores.get("flow_score", 50), 1),
                },
                "analysis": {
                    "sentiment": sentiment,
                    "trading_bias": trading_bias,
                    "bias_confidence": round(bias_confidence, 1),
                    "whale_accumulating": bool(all_features.get("whale_whale_accumulating", 0)),
                    "exchange_accumulating": bool(all_features.get("flow_is_accumulating", 0)),
                },
                "summary": self._generate_onchain_summary(scores, all_features),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_smart_money_analysis(
        self,
        token_address: str,
        chain: str = "ethereum",
    ) -> Dict[str, Any]:
        """
        Analyze smart money wallet behavior.

        Args:
            token_address: Token contract address
            chain: Blockchain name

        Returns:
            Dictionary with smart money analysis
        """
        try:
            analysis = self.wallet_tracker.analyze_smart_money(token_address, chain)

            return {
                "success": True,
                "token_address": token_address,
                "chain": chain,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "smart_wallets_analyzed": analysis.get("analyzed_count", 0),
                    "avg_holding_days": round(analysis.get("avg_holding_days", 0), 1),
                    "buy_pressure": round(analysis.get("buy_pressure", 0) * 100, 1),
                    "sell_pressure": round(analysis.get("sell_pressure", 0) * 100, 1),
                },
                "analysis": {
                    "dominant_action": analysis.get("dominant_action", "neutral"),
                    "confidence": round(analysis.get("confidence", 0) * 100, 1),
                    "trend": analysis.get("trend", "neutral"),
                },
                "interpretation": f"Smart money is primarily {analysis.get('dominant_action', 'neutral')} "
                                 f"with {analysis.get('confidence', 0)*100:.0f}% confidence",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def _interpret_whale_activity(self, features: Dict[str, float]) -> str:
        """Generate whale activity interpretation."""
        if features.get("whale_accumulating", 0):
            return "Whales are actively accumulating - bullish signal"
        elif features.get("whale_net", 0) < -1000:
            return "Whales are distributing significant amounts - bearish signal"
        elif features.get("whale_net", 0) > 1000:
            return "Positive whale net flow - slightly bullish"
        else:
            return "Whale activity is relatively neutral"

    def _interpret_distribution(self, features: Dict[str, float]) -> str:
        """Generate distribution interpretation."""
        gini = features.get("gini", 0)
        top_10 = features.get("top_10_pct", 0)

        if gini > 0.8 and top_10 > 50:
            return "Highly concentrated distribution - high manipulation risk"
        elif gini > 0.6 or top_10 > 40:
            return "Moderately concentrated - some whale influence expected"
        else:
            return "Well distributed token - lower manipulation risk"

    def _generate_onchain_summary(
        self,
        scores: Dict[str, float],
        features: Dict[str, float],
    ) -> str:
        """Generate overall on-chain summary."""
        parts = []

        overall = scores.get("overall_onchain_score", 50)
        if overall >= 65:
            parts.append("On-chain metrics are bullish")
        elif overall <= 35:
            parts.append("On-chain metrics are bearish")
        else:
            parts.append("On-chain metrics are neutral")

        if features.get("whale_whale_accumulating", 0):
            parts.append("whales accumulating")
        if features.get("flow_is_accumulating", 0):
            parts.append("exchange outflow dominant")

        return ". ".join(parts) + "."


def create_onchain_tools(chains: Optional[List[str]] = None) -> OnChainTools:
    """Create OnChainTools instance."""
    return OnChainTools(chains=chains)


if __name__ == "__main__":
    print("Testing OnChainTools...")

    tools = OnChainTools(chains=["ethereum"])

    # AAVE token address
    aave_address = "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9"

    # Test whale activity
    print("\n1. Getting whale activity for AAVE...")
    result = tools.get_whale_activity(aave_address, "ethereum")
    if result["success"]:
        print(f"   Whale behavior: {result['analysis']['whale_behavior']}")
        print(f"   Interpretation: {result['interpretation']}")

    # Test exchange flow
    print("\n2. Getting exchange flow...")
    result = tools.get_exchange_flow(aave_address, "ethereum")
    if result["success"]:
        print(f"   Flow direction: {result['analysis']['flow_direction']}")
        print(f"   Interpretation: {result['interpretation']}")

    # Test holder distribution
    print("\n3. Getting holder distribution...")
    result = tools.get_holder_distribution(aave_address, "ethereum")
    if result["success"]:
        print(f"   Distribution health: {result['analysis']['distribution_health']}")
        print(f"   Health score: {result['analysis']['health_score']}")

    # Test overall score
    print("\n4. Getting on-chain score...")
    result = tools.get_onchain_score(aave_address, "ethereum")
    if result["success"]:
        print(f"   Overall score: {result['scores']['overall']}")
        print(f"   Sentiment: {result['analysis']['sentiment']}")
        print(f"   Summary: {result['summary']}")
