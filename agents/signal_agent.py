"""
Signal Aggregator Agent

Aggregates signals from multiple sources into a unified trading recommendation.
Combines technical, on-chain, and quantitative signals with proper weighting.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import (
    AgentState,
    TradingSignal,
    SignalType,
    SignalStrength,
    TechnicalAnalysis,
    OnChainAnalysis,
)
from utils.prompts import SIGNAL_AGGREGATOR_PROMPT
from settings import load_config


@dataclass
class SignalComponent:
    """A component signal from one analysis source."""
    source: str
    signal: str  # bullish, bearish, neutral
    score: float  # 0-100
    confidence: float  # 0-1
    weight: float  # 0-1
    reasoning: str


class SignalAggregator:
    """
    Signal aggregation agent.

    Combines multiple analysis sources into a unified signal.
    """

    # Default weights for signal sources
    DEFAULT_WEIGHTS = {
        "technical": 0.35,
        "onchain": 0.35,
        "quantitative": 0.30,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_confidence: float = 0.65,
        model_name: str = "gemini-3-flash-preview",
    ):
        """
        Initialize the signal aggregator.

        Args:
            weights: Custom weights for signal sources
            min_confidence: Minimum confidence for actionable signals
            model_name: LLM model for reasoning
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.min_confidence = min_confidence
        self.model_name = model_name

        # Initialize LLM (optional, for enhanced reasoning)
        config = load_config()
        api_key = config.llm.gemini_api_keys[0] if config.llm.gemini_api_keys else None

        if api_key:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2,
                max_output_tokens=2048,
                max_retries=0,
                timeout=30,
            )
        else:
            self.llm = None

    def aggregate(
        self,
        symbol: str,
        latest_price: float,
        technical: Optional[TechnicalAnalysis] = None,
        onchain: Optional[OnChainAnalysis] = None,
        regime_state: Optional[Dict] = None,
        pattern_analysis: Optional[Dict] = None,
        anomaly_scores: Optional[Dict] = None,
    ) -> TradingSignal:
        """
        Aggregate all signals into a final recommendation.

        Args:
            symbol: Trading pair
            latest_price: Current price
            technical: Technical analysis result
            onchain: On-chain analysis result
            regime_state: Market regime state
            pattern_analysis: Pattern matching results
            anomaly_scores: Anomaly detection scores

        Returns:
            TradingSignal with final recommendation
        """
        # Collect component signals
        components = []

        # Technical signal
        if technical:
            tech_component = self._extract_technical_signal(technical)
            components.append(tech_component)

        # On-chain signal
        if onchain:
            onchain_component = self._extract_onchain_signal(onchain)
            components.append(onchain_component)

        # Quantitative signal
        quant_component = self._extract_quant_signal(regime_state, pattern_analysis)
        components.append(quant_component)

        # Calculate weighted score
        total_score = self._calculate_weighted_score(components)

        # Determine signal type
        signal_type, strength = self._determine_signal(total_score)

        # Calculate confidence
        confidence = self._calculate_confidence(components, anomaly_scores)

        # Calculate trade parameters
        entry, stop_loss, take_profit = self._calculate_trade_params(
            latest_price, signal_type, technical
        )

        # Calculate position size
        position_size = self._calculate_position_size(confidence, anomaly_scores)

        # Build summaries
        tech_summary = technical.summary if technical else "No technical analysis"
        onchain_summary = onchain.summary if onchain else "No on-chain analysis"

        # Build reasoning
        reasoning = self._build_reasoning(components, total_score)

        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size,
            confidence=confidence,
            reasoning=reasoning,
            technical_summary=tech_summary,
            onchain_summary=onchain_summary,
            risk_assessment="Pending",
            timestamp=datetime.now(),
        )

    def _extract_technical_signal(self, technical: TechnicalAnalysis) -> SignalComponent:
        """Extract signal from technical analysis."""
        trend = technical.trend.lower()

        if trend in ["bullish", "strong_uptrend"]:
            signal = "bullish"
            base_score = 70 if trend == "bullish" else 85
        elif trend in ["bearish", "strong_downtrend", "downtrend"]:
            signal = "bearish"
            base_score = 30 if trend == "bearish" else 15
        else:
            signal = "neutral"
            base_score = 50

        # Adjust for momentum
        momentum = technical.momentum.lower()
        if momentum == "overbought":
            base_score -= 10
        elif momentum == "oversold":
            base_score += 10

        score = max(0, min(100, base_score * technical.confidence))

        return SignalComponent(
            source="technical",
            signal=signal,
            score=score,
            confidence=technical.confidence,
            weight=self.weights.get("technical", 0.35),
            reasoning=f"{technical.trend} trend, {technical.momentum} momentum",
        )

    def _extract_onchain_signal(self, onchain: OnChainAnalysis) -> SignalComponent:
        """Extract signal from on-chain analysis."""
        whale = onchain.whale_activity.lower()
        flow = onchain.exchange_flow.lower()

        score = 50

        # Whale activity
        if whale == "accumulating":
            score += 20
            signal = "bullish"
        elif whale == "distributing":
            score -= 20
            signal = "bearish"
        else:
            signal = "neutral"

        # Exchange flow
        if flow == "outflow":
            score += 15
        elif flow == "inflow":
            score -= 15

        score = max(0, min(100, score * onchain.confidence))

        return SignalComponent(
            source="onchain",
            signal=signal,
            score=score,
            confidence=onchain.confidence,
            weight=self.weights.get("onchain", 0.35),
            reasoning=f"Whales {whale}, exchange {flow}",
        )

    def _extract_quant_signal(
        self,
        regime: Optional[Dict],
        patterns: Optional[Dict],
    ) -> SignalComponent:
        """Extract signal from quantitative analysis."""
        score = 50
        confidence = 0.5

        if patterns:
            bullish_prob = patterns.get("bullish_probability", 0.5)
            score = bullish_prob * 100
            confidence = patterns.get("confidence", 0.5)

        if regime:
            bias_score = regime.get("bias_score", 50)
            score = (score + bias_score) / 2

            market_regime = regime.get("market_regime", "sideways")
            if "bull" in market_regime:
                score += 10
            elif "bear" in market_regime:
                score -= 10

        if score >= 60:
            signal = "bullish"
        elif score <= 40:
            signal = "bearish"
        else:
            signal = "neutral"

        return SignalComponent(
            source="quantitative",
            signal=signal,
            score=max(0, min(100, score)),
            confidence=confidence,
            weight=self.weights.get("quantitative", 0.30),
            reasoning=f"Pattern probability {score:.0f}%",
        )

    def _calculate_weighted_score(self, components: List[SignalComponent]) -> float:
        """Calculate weighted average score."""
        if not components:
            return 50

        total_weight = sum(c.weight for c in components)
        if total_weight == 0:
            return 50

        weighted_sum = sum(c.score * c.weight * c.confidence for c in components)
        weight_sum = sum(c.weight * c.confidence for c in components)

        return weighted_sum / weight_sum if weight_sum > 0 else 50

    def _determine_signal(self, score: float) -> tuple:
        """Determine signal type and strength from score."""
        if score >= 75:
            return SignalType.BUY, SignalStrength.STRONG
        elif score >= 65:
            return SignalType.BUY, SignalStrength.MODERATE
        elif score >= 55:
            return SignalType.BUY, SignalStrength.WEAK
        elif score <= 25:
            return SignalType.SELL, SignalStrength.STRONG
        elif score <= 35:
            return SignalType.SELL, SignalStrength.MODERATE
        elif score <= 45:
            return SignalType.SELL, SignalStrength.WEAK
        else:
            return SignalType.HOLD, SignalStrength.WEAK

    def _calculate_confidence(
        self,
        components: List[SignalComponent],
        anomaly_scores: Optional[Dict],
    ) -> float:
        """Calculate overall confidence."""
        if not components:
            return 0.5

        # Average confidence from components
        avg_confidence = sum(c.confidence for c in components) / len(components)

        # Reduce confidence if anomaly detected
        if anomaly_scores:
            anomaly = anomaly_scores.get("combined_anomaly_score", 0)
            if anomaly > 0.8:
                avg_confidence *= 0.7
            elif anomaly > 0.5:
                avg_confidence *= 0.85

        # Check signal agreement
        signals = [c.signal for c in components]
        bullish_count = signals.count("bullish")
        bearish_count = signals.count("bearish")

        agreement = max(bullish_count, bearish_count) / len(signals)
        avg_confidence *= (0.7 + 0.3 * agreement)

        return max(0.3, min(1.0, avg_confidence))

    def _calculate_trade_params(
        self,
        price: float,
        signal_type: SignalType,
        technical: Optional[TechnicalAnalysis],
    ) -> tuple:
        """Calculate entry, stop-loss, and take-profit."""
        # Default ATR-based calculation (5% SL, 10% TP)
        sl_pct = 0.05
        tp_pct = 0.10

        # Use ATR if available
        if technical and technical.key_indicators.get("atr"):
            atr = technical.key_indicators["atr"]
            atr_pct = atr / price
            sl_pct = min(0.05, max(0.02, atr_pct * 2))
            tp_pct = sl_pct * 2  # 2:1 R:R minimum

        if signal_type == SignalType.BUY:
            entry = price
            stop_loss = price * (1 - sl_pct)
            take_profit = price * (1 + tp_pct)
        elif signal_type == SignalType.SELL:
            entry = price
            stop_loss = price * (1 + sl_pct)
            take_profit = price * (1 - tp_pct)
        else:
            entry = price
            stop_loss = price
            take_profit = price

        return entry, stop_loss, take_profit

    def _calculate_position_size(
        self,
        confidence: float,
        anomaly_scores: Optional[Dict],
    ) -> float:
        """Calculate position size as percentage."""
        # Base size on confidence (max 2%)
        base_size = confidence * 2

        # Reduce for high anomaly
        if anomaly_scores:
            anomaly = anomaly_scores.get("combined_anomaly_score", 0)
            if anomaly > 0.7:
                base_size *= 0.5

        return max(0.5, min(2.0, base_size))

    def _build_reasoning(
        self,
        components: List[SignalComponent],
        total_score: float,
    ) -> str:
        """Build reasoning string."""
        parts = [f"Weighted Score: {total_score:.0f}/100"]

        for c in components:
            parts.append(f"{c.source.title()}: {c.signal} ({c.score:.0f}, {c.confidence:.0%})")

        return " | ".join(parts)

    def __call__(self, state: AgentState) -> AgentState:
        """Run agent as part of LangGraph workflow."""
        signal = self.aggregate(
            symbol=state.get("symbol", "BTC/USDT"),
            latest_price=state.get("latest_price", 0),
            technical=state.get("technical_analysis"),
            onchain=state.get("onchain_analysis"),
            regime_state=state.get("regime_state"),
            pattern_analysis=state.get("pattern_analysis"),
            anomaly_scores=state.get("anomaly_scores"),
        )

        state["aggregated_signal"] = signal
        state["messages"] = state.get("messages", [])
        state["messages"].append(
            AIMessage(content=f"Signal: {signal.signal_type.value.upper()} ({signal.confidence:.0%})")
        )

        return state


def create_signal_aggregator(
    weights: Optional[Dict[str, float]] = None,
    min_confidence: float = 0.65,
) -> SignalAggregator:
    """Create a SignalAggregator instance."""
    return SignalAggregator(weights=weights, min_confidence=min_confidence)


if __name__ == "__main__":
    print("Testing SignalAggregator...")

    aggregator = SignalAggregator()

    # Create mock analysis results
    technical = TechnicalAnalysis(
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp=datetime.now(),
        trend="bullish",
        momentum="neutral",
        volatility="medium",
        support_levels=[95000, 94000],
        resistance_levels=[98000, 100000],
        key_indicators={"rsi": 55, "atr": 500},
        signals=["Above SMA 200", "MACD bullish"],
        summary="Bullish trend with neutral momentum",
        confidence=0.75,
    )

    onchain = OnChainAnalysis(
        token_address="0x123",
        chain="ethereum",
        timestamp=datetime.now(),
        whale_activity="accumulating",
        exchange_flow="outflow",
        holder_trend="growing",
        key_metrics={},
        alerts=[],
        summary="Whales accumulating, outflow from exchanges",
        confidence=0.70,
    )

    regime = {
        "market_regime": "bull",
        "bias_score": 65,
    }

    patterns = {
        "bullish_probability": 0.65,
        "confidence": 0.6,
    }

    # Aggregate signals
    signal = aggregator.aggregate(
        symbol="BTC/USDT",
        latest_price=96500,
        technical=technical,
        onchain=onchain,
        regime_state=regime,
        pattern_analysis=patterns,
    )

    print(f"\nResults:")
    print(f"  Signal: {signal.signal_type.value.upper()}")
    print(f"  Strength: {signal.strength.value}")
    print(f"  Confidence: {signal.confidence:.0%}")
    print(f"  Entry: ${signal.entry_price:,.2f}")
    print(f"  Stop Loss: ${signal.stop_loss:,.2f}")
    print(f"  Take Profit: ${signal.take_profit:,.2f}")
    print(f"  Position Size: {signal.position_size_pct:.1f}%")
    print(f"  Reasoning: {signal.reasoning}")
