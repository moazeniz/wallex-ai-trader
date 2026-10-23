"""
Technical Analyst Agent

LangGraph agent that performs technical analysis using LLM reasoning.
Uses technical analysis tools to gather data and interpret patterns.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import AgentState, TechnicalAnalysis
from tools.technical_tools import TechnicalTools
from utils.prompts import TECHNICAL_ANALYST_PROMPT
from settings import load_config
from utils.rate_limiter import get_gemini_manager


class TechnicalAnalystAgent:
    """
    Technical Analyst Agent.

    Analyzes price action and technical indicators using LLM reasoning.
    """

    def __init__(
        self,
        exchange: str = "binance",
        model_name: str = "gemini-3-flash-preview",
    ):
        """
        Initialize the technical analyst agent.

        Args:
            exchange: Exchange for price data
            model_name: LLM model to use
        """
        self.exchange = exchange
        self.model_name = model_name
        self.tools = TechnicalTools(exchange=exchange)

        # Initialize with key rotation
        self.config = load_config()
        self.key_manager = get_gemini_manager()

    def analyze(
        self,
        symbol: str,
        timeframe: str = "1h",
    ) -> TechnicalAnalysis:
        """
        Perform technical analysis on a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe for analysis

        Returns:
            TechnicalAnalysis result
        """
        timestamp = datetime.now()

        # Gather data using tools
        ohlcv_data = self.tools.fetch_ohlcv(symbol, timeframe, 200)
        indicator_data = self.tools.get_indicators(
            symbol, timeframe,
            ["rsi", "macd", "bb", "sma", "atr", "stochastic", "adx"]
        )
        levels_data = self.tools.find_support_resistance(symbol, timeframe)
        trend_data = self.tools.get_trend_summary(symbol, timeframe)

        # Check for errors
        if not all([
            ohlcv_data.get("success"),
            indicator_data.get("success"),
            levels_data.get("success"),
            trend_data.get("success")
        ]):
            return self._create_error_analysis(symbol, timeframe, timestamp)

        # Prepare context for LLM
        context = self._prepare_context(ohlcv_data, indicator_data, levels_data, trend_data)

        # Get LLM interpretation
        interpretation = self._get_llm_interpretation(symbol, timeframe, context)

        # Build analysis result
        return self._build_analysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            ohlcv_data=ohlcv_data,
            indicator_data=indicator_data,
            levels_data=levels_data,
            trend_data=trend_data,
            interpretation=interpretation,
        )

    def _prepare_context(
        self,
        ohlcv: Dict,
        indicators: Dict,
        levels: Dict,
        trend: Dict,
    ) -> str:
        """Prepare context string for LLM."""
        context = f"""
## Price Data
- Current Price: ${ohlcv['summary']['current_price']:,.4f}
- Change (1 period): {ohlcv['summary']['change_1_period']}%
- Period High: ${ohlcv['summary']['high_period']:,.4f}
- Period Low: ${ohlcv['summary']['low_period']:,.4f}

## Technical Indicators
"""
        # Add indicators
        for name, data in indicators.get("indicators", {}).items():
            if isinstance(data, dict):
                if "value" in data:
                    context += f"- {name.upper()}: {data['value']}"
                    if "interpretation" in data:
                        context += f" ({data['interpretation']})"
                    context += "\n"
                elif "macd" in data:
                    context += f"- MACD: {data.get('macd', 'N/A')} | Signal: {data.get('signal', 'N/A')} ({data.get('interpretation', '')})\n"

        context += f"""
## Support/Resistance
- Nearest Support: ${levels.get('nearest_support', 'N/A')} ({levels.get('support_distance_pct', 0)}% away)
- Nearest Resistance: ${levels.get('nearest_resistance', 'N/A')} ({levels.get('resistance_distance_pct', 0)}% away)
- Support Levels: {levels.get('support_levels', [])}
- Resistance Levels: {levels.get('resistance_levels', [])}

## Trend Summary
- Overall Trend: {trend.get('trend', 'unknown')}
- Trend Score: {trend.get('trend_score', 'N/A')}
- Momentum: {trend.get('momentum', 'unknown')}
- Volatility: {trend.get('volatility', 'unknown')}
- Above SMA 20/50/200: {trend.get('above_sma_20', 'N/A')}/{trend.get('above_sma_50', 'N/A')}/{trend.get('above_sma_200', 'N/A')}
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
        symbol: str,
        timeframe: str,
        context: str,
    ) -> Dict[str, Any]:
        """Get LLM interpretation of technical data."""
        prompt = f"""{TECHNICAL_ANALYST_PROMPT}

## Current Analysis: {symbol} ({timeframe})

{context}

## Your Task
Based on the technical data above, provide:
1. Overall trend assessment (bullish/bearish/neutral)
2. Key signals (list the most important 3-5 signals)
3. Confidence score (0-100)
4. Brief summary (2-3 sentences)

Respond in JSON format:
{{
    "trend": "bullish|bearish|neutral",
    "momentum": "overbought|oversold|neutral",
    "volatility": "low|medium|high",
    "signals": ["signal1", "signal2", ...],
    "confidence": 75,
    "summary": "Brief summary here"
}}
"""

        for attempt in range(min(getattr(self.key_manager, "total_keys", 200), 50)):
            api_key = self.key_manager.get_key()
            try:
                llm = self._get_llm_with_key(api_key)
                response = llm.invoke([HumanMessage(content=prompt)])
                resp = response.content
                # Robust handling of Gemini response
                try:
                    if isinstance(resp, list):
                        if len(resp) > 0:
                            item = resp[0]
                            if hasattr(item, 'text'):
                                resp = item.text
                            elif hasattr(item, 'content'):
                                resp = item.content
                            else:
                                resp = str(item)
                        else:
                            resp = ""
                    elif hasattr(resp, 'text'):
                        resp = resp.text
                    resp = str(resp) if resp else ""
                except Exception as parse_err:
                    print(f"Response parsing error: {parse_err}, type={type(resp)}")
                    resp = str(resp)
                start = resp.find("{")
                end = resp.rfind("}") + 1
                if start >= 0 and end > start:
                    try:
                        return json.loads(resp[start:end])
                    except:
                        pass
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower():
                    print(f"Key {attempt+1} rate limited, switching...")
                    self.key_manager.rotate()
                    continue
                print(f"LLM error: {e}")
                break
        return {
            "trend": "neutral",
            "momentum": "neutral",
            "volatility": "medium",
            "signals": [],
            "confidence": 50,
            "summary": "Unable to generate interpretation"
        }

    def _build_analysis(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        ohlcv_data: Dict,
        indicator_data: Dict,
        levels_data: Dict,
        trend_data: Dict,
        interpretation: Dict,
    ) -> TechnicalAnalysis:
        """Build TechnicalAnalysis result."""
        # Extract key indicators
        key_indicators = {}
        for name, data in indicator_data.get("indicators", {}).items():
            if isinstance(data, dict):
                if "value" in data and data["value"] is not None:
                    key_indicators[name] = data["value"]
                elif "macd" in data and data["macd"] is not None:
                    key_indicators["macd"] = data["macd"]
                    key_indicators["macd_signal"] = data.get("signal")

        key_indicators["current_price"] = ohlcv_data["summary"]["current_price"]
        key_indicators["trend_score"] = trend_data.get("trend_score", 0)

        return TechnicalAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            trend=interpretation.get("trend", "neutral"),
            momentum=interpretation.get("momentum", "neutral"),
            volatility=interpretation.get("volatility", "medium"),
            support_levels=levels_data.get("support_levels", []),
            resistance_levels=levels_data.get("resistance_levels", []),
            key_indicators=key_indicators,
            signals=interpretation.get("signals", []),
            summary=interpretation.get("summary", "No summary available"),
            confidence=interpretation.get("confidence", 50) / 100,
        )

    def _create_error_analysis(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
    ) -> TechnicalAnalysis:
        """Create error analysis result."""
        return TechnicalAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            trend="unknown",
            momentum="unknown",
            volatility="unknown",
            support_levels=[],
            resistance_levels=[],
            key_indicators={},
            signals=["Error fetching data"],
            summary="Unable to perform technical analysis due to data errors",
            confidence=0,
        )

    def __call__(self, state: AgentState) -> AgentState:
        """
        Run agent as part of LangGraph workflow.

        Args:
            state: Current agent state

        Returns:
            Updated state with technical analysis
        """
        symbol = state.get("symbol", "BTC/USDT")
        timeframe = state.get("timeframe", "1h")

        start_time = datetime.now()
        analysis = self.analyze(symbol, timeframe)
        end_time = datetime.now()

        # Update state
        state["technical_analysis"] = analysis
        state["timing"] = state.get("timing", {})
        state["timing"]["technical_analysis"] = (end_time - start_time).total_seconds()

        # Add to messages
        state["messages"] = state.get("messages", [])
        state["messages"].append(
            AIMessage(content=f"Technical Analysis Complete: {analysis.summary}")
        )

        return state


def create_technical_agent(
    exchange: str = "binance",
    model_name: str = "gemini-3-flash-preview",
) -> TechnicalAnalystAgent:
    """Create a TechnicalAnalystAgent instance."""
    return TechnicalAnalystAgent(exchange=exchange, model_name=model_name)


if __name__ == "__main__":
    print("Testing TechnicalAnalystAgent...")

    agent = TechnicalAnalystAgent(exchange="binance")

    # Test analysis
    print("\nAnalyzing BTC/USDT...")
    analysis = agent.analyze("BTC/USDT", "1h")

    print(f"\nResults:")
    print(f"  Symbol: {analysis.symbol}")
    print(f"  Trend: {analysis.trend}")
    print(f"  Momentum: {analysis.momentum}")
    print(f"  Volatility: {analysis.volatility}")
    print(f"  Confidence: {analysis.confidence:.0%}")
    print(f"  Support: {analysis.support_levels[:2]}")
    print(f"  Resistance: {analysis.resistance_levels[:2]}")
    print(f"  Summary: {analysis.summary}")
