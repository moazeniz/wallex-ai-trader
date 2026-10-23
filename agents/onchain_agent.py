"""
On-Chain Analyst Agent

LangGraph agent that performs on-chain analysis using LLM reasoning.
Uses on-chain tools to gather blockchain data and interpret patterns.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import AgentState, OnChainAnalysis
from tools.onchain_tools import OnChainTools
from utils.prompts import ONCHAIN_ANALYST_PROMPT
from settings import load_config
from utils.rate_limiter import get_gemini_manager


class OnChainAnalystAgent:
    """
    On-Chain Analyst Agent.

    Analyzes blockchain data and wallet behavior using LLM reasoning.
    """

    def __init__(
        self,
        chains: Optional[List[str]] = None,
        model_name: str = "gemini-3-flash-preview",
    ):
        """
        Initialize the on-chain analyst agent.

        Args:
            chains: List of chains to analyze
            model_name: LLM model to use
        """
        self.chains = chains or ["ethereum"]
        self.model_name = model_name
        self.tools = OnChainTools(chains=self.chains)

        # Initialize with key rotation
        self.config = load_config()
        self.key_manager = get_gemini_manager()

    def analyze(
        self,
        token_address: str,
        chain: str = "ethereum",
    ) -> OnChainAnalysis:
        """
        Perform on-chain analysis for a token.

        Args:
            token_address: Token contract address
            chain: Blockchain to analyze

        Returns:
            OnChainAnalysis result
        """
        timestamp = datetime.now()

        # Gather data using tools
        whale_data = self.tools.get_whale_activity(token_address, chain)
        flow_data = self.tools.get_exchange_flow(token_address, chain)
        distribution_data = self.tools.get_holder_distribution(token_address, chain)
        score_data = self.tools.get_onchain_score(token_address, chain)

        # Check for errors
        if not all([
            whale_data.get("success"),
            flow_data.get("success"),
            distribution_data.get("success"),
            score_data.get("success")
        ]):
            return self._create_error_analysis(token_address, chain, timestamp)

        # Prepare context for LLM
        context = self._prepare_context(whale_data, flow_data, distribution_data, score_data)

        # Get LLM interpretation
        interpretation = self._get_llm_interpretation(token_address, chain, context)

        # Build analysis result
        return self._build_analysis(
            token_address=token_address,
            chain=chain,
            timestamp=timestamp,
            whale_data=whale_data,
            flow_data=flow_data,
            distribution_data=distribution_data,
            score_data=score_data,
            interpretation=interpretation,
        )

    def _prepare_context(
        self,
        whale: Dict,
        flow: Dict,
        distribution: Dict,
        score: Dict,
    ) -> str:
        """Prepare context string for LLM."""
        context = f"""
## Whale Activity
- Whale Count: {whale['metrics'].get('whale_count', 0)}
- Whale Holdings: {whale['metrics'].get('whale_holdings_pct', 0)}%
- Whale Net Flow: {whale['metrics'].get('whale_net', 0)}
- Whale Behavior: {whale['analysis'].get('whale_behavior', 'unknown')}
- Is Accumulating: {whale['analysis'].get('is_accumulating', False)}
- Interpretation: {whale.get('interpretation', '')}

## Exchange Flow
- Inflow: {flow['metrics'].get('inflow', 0)}
- Outflow: {flow['metrics'].get('outflow', 0)}
- Net Flow: {flow['metrics'].get('net_flow', 0)}
- Flow Ratio (out/in): {flow['metrics'].get('flow_ratio', 1)}
- Flow Direction: {flow['analysis'].get('flow_direction', 'unknown')}
- Flow Strength: {flow['analysis'].get('flow_strength', 'unknown')}
- Interpretation: {flow.get('interpretation', '')}

## Holder Distribution
- Holder Count: {distribution['metrics'].get('holder_count', 0)}
- Top 10 Holdings: {distribution['metrics'].get('top_10_pct', 0)}%
- Top 50 Holdings: {distribution['metrics'].get('top_50_pct', 0)}%
- Exchange Holdings: {distribution['metrics'].get('exchange_holdings_pct', 0)}%
- Gini Coefficient: {distribution['metrics'].get('gini_coefficient', 0)}
- Distribution Health: {distribution['analysis'].get('distribution_health', 'unknown')}
- Health Score: {distribution['analysis'].get('health_score', 0)}

## Overall Scores
- Overall Score: {score['scores'].get('overall', 50)}/100
- Accumulation Score: {score['scores'].get('accumulation', 50)}/100
- Distribution Score: {score['scores'].get('distribution_health', 50)}/100
- Flow Score: {score['scores'].get('flow', 50)}/100
- Sentiment: {score['analysis'].get('sentiment', 'neutral')}
"""
        return context

    def _get_llm_with_key(self, api_key):
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=4096,
            max_retries=0,  # Disable internal retries - we handle rotation ourselves
            timeout=30,  # Fast timeout
        )

    def _get_llm_interpretation(
        self,
        token_address: str,
        chain: str,
        context: str,
    ) -> Dict[str, Any]:
        """Get LLM interpretation of on-chain data."""
        prompt = f"""{ONCHAIN_ANALYST_PROMPT}

## Current Analysis: Token on {chain}
Address: {token_address}

{context}

## Your Task
Based on the on-chain data above, provide:
1. Whale activity assessment (accumulating/distributing/neutral)
2. Exchange flow assessment (inflow/outflow/balanced)
3. Holder trend assessment (growing/shrinking/stable)
4. Key alerts (list any concerning or notable patterns)
5. Confidence score (0-100)
6. Brief summary (2-3 sentences)

Respond in JSON format:
{{
    "whale_activity": "accumulating|distributing|neutral",
    "exchange_flow": "inflow|outflow|balanced",
    "holder_trend": "growing|shrinking|stable",
    "alerts": ["alert1", "alert2", ...],
    "confidence": 75,
    "summary": "Brief summary here"
}}
"""

        # Retry loop with key rotation (matching technical_agent.py pattern)
        for attempt in range(min(getattr(self.key_manager, "total_keys", 200), 50)):
            api_key = self.key_manager.get_key()
            try:
                llm = self._get_llm_with_key(api_key)
                response = llm.invoke([HumanMessage(content=prompt)])
                resp = response.content

                # Robust handling of Gemini response (handles list/object types)
                try:
                    if isinstance(resp, list):
                        if len(resp) > 0:
                            item = resp[0]
                            if hasattr(item, "text"):
                                resp = item.text
                            elif hasattr(item, "content"):
                                resp = item.content
                            else:
                                resp = str(item)
                        else:
                            resp = ""
                    elif hasattr(resp, "text"):
                        resp = resp.text
                    resp = str(resp) if resp else ""
                except Exception as parse_err:
                    print(f"Response parsing error: {parse_err}, type={type(resp)}")
                    resp = str(resp)

                # Parse JSON from response
                start = resp.find("{")
                end = resp.rfind("}") + 1
                if start >= 0 and end > start:
                    try:
                        return json.loads(resp[start:end])
                    except json.JSONDecodeError:
                        pass
                break
            except Exception as e:
                error_str = str(e).lower()
                if "429" in str(e) or "rate" in error_str or "quota" in error_str or "exhausted" in error_str:
                    print(f"Key {attempt+1} rate limited, switching...")
                    self.key_manager.rotate()
                    continue
                print(f"LLM interpretation error: {e}")
                break

        # Default response on failure
        return {
            "whale_activity": "neutral",
            "exchange_flow": "balanced",
            "holder_trend": "stable",
            "alerts": [],
            "confidence": 50,
            "summary": "Unable to generate interpretation"
        }

    def _build_analysis(
        self,
        token_address: str,
        chain: str,
        timestamp: datetime,
        whale_data: Dict,
        flow_data: Dict,
        distribution_data: Dict,
        score_data: Dict,
        interpretation: Dict,
    ) -> OnChainAnalysis:
        """Build OnChainAnalysis result."""
        # Extract key metrics
        key_metrics = {
            "whale_count": whale_data["metrics"].get("whale_count", 0),
            "whale_holdings_pct": whale_data["metrics"].get("whale_holdings_pct", 0),
            "whale_net": whale_data["metrics"].get("whale_net", 0),
            "exchange_inflow": flow_data["metrics"].get("inflow", 0),
            "exchange_outflow": flow_data["metrics"].get("outflow", 0),
            "flow_ratio": flow_data["metrics"].get("flow_ratio", 1),
            "holder_count": distribution_data["metrics"].get("holder_count", 0),
            "top_10_pct": distribution_data["metrics"].get("top_10_pct", 0),
            "gini": distribution_data["metrics"].get("gini_coefficient", 0),
            "overall_score": score_data["scores"].get("overall", 50),
        }

        return OnChainAnalysis(
            token_address=token_address,
            chain=chain,
            timestamp=timestamp,
            whale_activity=interpretation.get("whale_activity", "neutral"),
            exchange_flow=interpretation.get("exchange_flow", "balanced"),
            holder_trend=interpretation.get("holder_trend", "stable"),
            key_metrics=key_metrics,
            alerts=interpretation.get("alerts", []),
            summary=interpretation.get("summary", "No summary available"),
            confidence=interpretation.get("confidence", 50) / 100,
        )

    def _create_error_analysis(
        self,
        token_address: str,
        chain: str,
        timestamp: datetime,
    ) -> OnChainAnalysis:
        """Create error analysis result."""
        return OnChainAnalysis(
            token_address=token_address,
            chain=chain,
            timestamp=timestamp,
            whale_activity="unknown",
            exchange_flow="unknown",
            holder_trend="unknown",
            key_metrics={},
            alerts=["Error fetching on-chain data"],
            summary="Unable to perform on-chain analysis due to data errors",
            confidence=0,
        )

    def __call__(self, state: AgentState) -> AgentState:
        """
        Run agent as part of LangGraph workflow.

        Args:
            state: Current agent state

        Returns:
            Updated state with on-chain analysis
        """
        token_address = state.get("token_address")
        chain = state.get("chain", "ethereum")

        if not token_address:
            # No token address provided, skip on-chain analysis
            state["onchain_analysis"] = None
            return state

        start_time = datetime.now()
        analysis = self.analyze(token_address, chain)
        end_time = datetime.now()

        # Update state
        state["onchain_analysis"] = analysis
        state["timing"] = state.get("timing", {})
        state["timing"]["onchain_analysis"] = (end_time - start_time).total_seconds()

        # Add to messages
        state["messages"] = state.get("messages", [])
        state["messages"].append(
            AIMessage(content=f"On-Chain Analysis Complete: {analysis.summary}")
        )

        return state


def create_onchain_agent(
    chains: Optional[List[str]] = None,
    model_name: str = "gemini-3-flash-preview",
) -> OnChainAnalystAgent:
    """Create an OnChainAnalystAgent instance."""
    return OnChainAnalystAgent(chains=chains, model_name=model_name)


if __name__ == "__main__":
    print("Testing OnChainAnalystAgent...")

    agent = OnChainAnalystAgent(chains=["ethereum"])

    # AAVE token address
    aave_address = "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9"

    # Test analysis
    print(f"\nAnalyzing AAVE token...")
    analysis = agent.analyze(aave_address, "ethereum")

    print(f"\nResults:")
    print(f"  Token: {analysis.token_address[:10]}...")
    print(f"  Chain: {analysis.chain}")
    print(f"  Whale Activity: {analysis.whale_activity}")
    print(f"  Exchange Flow: {analysis.exchange_flow}")
    print(f"  Holder Trend: {analysis.holder_trend}")
    print(f"  Confidence: {analysis.confidence:.0%}")
    print(f"  Alerts: {analysis.alerts}")
    print(f"  Summary: {analysis.summary}")
