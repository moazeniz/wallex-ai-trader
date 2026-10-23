"""
Agent Orchestrator

LangGraph-based orchestrator for the multi-agent trading system.
Manages workflow between Technical, On-Chain, Strategy, and Risk agents.
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
import time

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import (
    AgentState,
    create_initial_state,
    TechnicalAnalysis,
    OnChainAnalysis,
    TradingSignal,
    SignalType,
    SignalStrength,
)
from agents.technical_agent import TechnicalAnalystAgent
from agents.onchain_agent import OnChainAnalystAgent
from ml.anomaly_detector import AnomalyDetector
from ml.regime_detector import RegimeDetector
from ml.pattern_matcher import PatternMatcher
from ml.gru_classifier import DirectionPredictor
from data.exchange_client import ExchangeClient


class TradingOrchestrator:
    """
    Multi-agent trading system orchestrator.

    Coordinates analysis from multiple agents and ML models
    to generate trading signals.
    """

    def __init__(
        self,
        exchange: str = "binance",
        chains: Optional[List[str]] = None,
        model_name: str = "gemini-3-flash-preview",
    ):
        """
        Initialize the orchestrator.

        Args:
            exchange: Exchange for price data
            chains: Chains for on-chain data
            model_name: LLM model for agents
        """
        self.exchange = exchange
        self.chains = chains or ["ethereum"]
        self.model_name = model_name

        # Initialize agents
        self.technical_agent = TechnicalAnalystAgent(
            exchange=exchange,
            model_name=model_name,
        )
        self.onchain_agent = OnChainAnalystAgent(
            chains=self.chains,
            model_name=model_name,
        )

        # Initialize ML models
        self.anomaly_detector = AnomalyDetector()
        self.regime_detector = RegimeDetector()
        self.pattern_matcher = PatternMatcher()

        # Initialize data client
        self.exchange_client = ExchangeClient(exchange=exchange)

        # GRU direction predictor
        self.gru_predictor = None
        self._init_gru()
        
        # ML models fitted flag
        self._ml_fitted = False
    
    def _init_gru(self):
        """Initialize GRU direction predictor."""
        try:
            self.gru_predictor = DirectionPredictor()
            self.gru_predictor.load("/opt/wallex/models/gru_direction.pkl")
            print("GRU direction predictor loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load GRU predictor: {e}")
            self.gru_predictor = None

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create graph with AgentState
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("fetch_data", self._fetch_data_node)
        workflow.add_node("technical_analysis", self._technical_node)
        workflow.add_node("onchain_analysis", self._onchain_node)
        workflow.add_node("ml_analysis", self._ml_node)
        workflow.add_node("aggregate_signals", self._aggregate_node)
        workflow.add_node("risk_check", self._risk_node)

        # Set entry point
        workflow.set_entry_point("fetch_data")

        # Add edges
        workflow.add_edge("fetch_data", "technical_analysis")
        workflow.add_edge("technical_analysis", "onchain_analysis")
        workflow.add_edge("onchain_analysis", "ml_analysis")
        workflow.add_edge("ml_analysis", "aggregate_signals")
        workflow.add_edge("aggregate_signals", "risk_check")
        workflow.add_edge("risk_check", END)

        return workflow.compile()

    def _fetch_data_node(self, state: AgentState) -> AgentState:
        """Fetch market data."""
        start = time.time()

        symbol = state.get("symbol", "BTC/USDT")
        timeframe = state.get("timeframe", "1h")

        try:
            df = self.exchange_client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=500,
            )

            # Fit ML models if not already fitted
            if not self._ml_fitted and len(df) >= 200:
                self.anomaly_detector.fit(df)
                self.regime_detector.fit(df)
                self.pattern_matcher.fit(df.iloc[:-50])
                self._ml_fitted = True

            state["latest_price"] = float(df["close"].iloc[-1])
            state["features"] = {
                "open": float(df["open"].iloc[-1]),
                "high": float(df["high"].iloc[-1]),
                "low": float(df["low"].iloc[-1]),
                "close": float(df["close"].iloc[-1]),
                "volume": float(df["volume"].iloc[-1]),
            }

        except Exception as e:
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Data fetch error: {e}")

        state["timing"] = state.get("timing", {})
        state["timing"]["fetch_data"] = time.time() - start
        return state

    def _technical_node(self, state: AgentState) -> AgentState:
        """Run technical analysis."""
        return self.technical_agent(state)

    def _onchain_node(self, state: AgentState) -> AgentState:
        """Run on-chain analysis."""
        if state.get("token_address"):
            return self.onchain_agent(state)
        return state

    def _ml_node(self, state: AgentState) -> AgentState:
        """Run ML model analysis."""
        start = time.time()

        symbol = state.get("symbol", "BTC/USDT")
        timeframe = state.get("timeframe", "1h")

        try:
            df = self.exchange_client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=200,
            )

            # Anomaly detection
            if self._ml_fitted:
                anomaly_scores = self.anomaly_detector.get_latest_anomaly_score(df)
                state["anomaly_scores"] = anomaly_scores

                # Regime detection
                regime_summary = self.regime_detector.get_regime_summary(df)
                state["regime_state"] = regime_summary

                # Pattern matching
                pattern_summary = self.pattern_matcher.get_pattern_summary(df)
                state["pattern_analysis"] = pattern_summary
                
                # GRU direction prediction
                if self.gru_predictor is not None:
                    try:
                        gru_prediction = self.gru_predictor.predict(df)
                        state["gru_prediction"] = gru_prediction
                    except Exception as e:
                        state["gru_prediction"] = {
                            "direction": "neutral",
                            "probability_up": 0.5,
                            "confidence": 0.0,
                            "error": str(e)
                        }
                else:
                    state["gru_prediction"] = {
                        "direction": "neutral",
                        "probability_up": 0.5,
                        "confidence": 0.0,
                    }

        except Exception as e:
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"ML analysis error: {e}")
            state["anomaly_scores"] = {}
            state["regime_state"] = {}
            state["pattern_analysis"] = {}

        state["timing"] = state.get("timing", {})
        state["timing"]["ml_analysis"] = time.time() - start
        return state

    def _aggregate_node(self, state: AgentState) -> AgentState:
        """Aggregate all signals into final recommendation."""
        start = time.time()

        symbol = state.get("symbol", "BTC/USDT")
        latest_price = state.get("latest_price", 0)

        # Get analysis results
        technical = state.get("technical_analysis")
        onchain = state.get("onchain_analysis")
        regime = state.get("regime_state", {})
        patterns = state.get("pattern_analysis", {})

        # Calculate signal components
        signals = []

        # Technical signal (30% weight)
        tech_score = 50
        if technical:
            if technical.trend == "bullish":
                tech_score = 70
            elif technical.trend == "bearish":
                tech_score = 30
            elif technical.trend in ["strong_uptrend"]:
                tech_score = 85
            tech_score *= technical.confidence
        signals.append(("technical", tech_score, 0.30))

        # On-chain signal (30% weight)
        onchain_score = 50
        if onchain:
            if onchain.whale_activity == "accumulating":
                onchain_score = 70
            elif onchain.whale_activity == "distributing":
                onchain_score = 30
            if onchain.exchange_flow == "outflow":
                onchain_score += 10
            elif onchain.exchange_flow == "inflow":
                onchain_score -= 10
            onchain_score *= onchain.confidence
        signals.append(("onchain", onchain_score, 0.30))

        # GRU direction signal (25% weight)
        gru_pred = state.get("gru_prediction", {})
        gru_score = 50
        if gru_pred:
            prob_up = gru_pred.get("probability_up", 0.5)
            gru_score = prob_up * 100
        signals.append(("gru_direction", gru_score, 0.25))

        # Quantitative signal (15% weight)
        quant_score = 50
        if patterns:
            bullish_prob = patterns.get("bullish_probability", 0.5)
            quant_score = bullish_prob * 100
        if regime:
            bias_score = regime.get("bias_score", 50)
            quant_score = (quant_score + bias_score) / 2
        signals.append(("quantitative", quant_score, 0.15))

        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in signals)

        # Determine signal
        if total_score >= 65:
            signal_type = SignalType.BUY
            strength = SignalStrength.STRONG if total_score >= 75 else SignalStrength.MODERATE
        elif total_score <= 35:
            signal_type = SignalType.SELL
            strength = SignalStrength.STRONG if total_score <= 25 else SignalStrength.MODERATE
        else:
            signal_type = SignalType.HOLD
            strength = SignalStrength.WEAK

        # Calculate entry, SL, TP
        if signal_type == SignalType.BUY:
            entry = latest_price
            stop_loss = entry * 0.95  # 5% SL
            take_profit = entry * 1.10  # 10% TP
        elif signal_type == SignalType.SELL:
            entry = latest_price
            stop_loss = entry * 1.05  # 5% SL
            take_profit = entry * 0.90  # 10% TP
        else:
            entry = latest_price
            stop_loss = entry
            take_profit = entry

        # Calculate position size (max 2%)
        confidence = total_score / 100
        position_size = min(2.0, confidence * 2)

        # Build summaries
        tech_summary = technical.summary if technical else "No technical analysis"
        onchain_summary = onchain.summary if onchain else "No on-chain analysis"

        # Create signal
        trading_signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size,
            confidence=confidence,
            reasoning=f"Technical: {tech_score:.0f}, OnChain: {onchain_score:.0f}, GRU: {gru_score:.0f}, Quant: {quant_score:.0f}",
            technical_summary=tech_summary,
            onchain_summary=onchain_summary,
            risk_assessment="Pending risk review",
            timestamp=datetime.now(),
        )

        state["aggregated_signal"] = trading_signal
        state["timing"]["aggregate"] = time.time() - start
        return state

    def _risk_node(self, state: AgentState) -> AgentState:
        """Perform risk assessment on the signal."""
        start = time.time()

        signal = state.get("aggregated_signal")
        anomaly_scores = state.get("anomaly_scores", {})
        regime = state.get("regime_state", {})

        if not signal:
            state["risk_approved"] = False
            state["risk_notes"] = "No signal to evaluate"
            return state

        # Risk checks
        approved = True
        notes = []

        # Check 1: Minimum confidence
        if signal.confidence < 0.65:
            if signal.signal_type != SignalType.HOLD:
                approved = False
                notes.append(f"Confidence too low ({signal.confidence:.0%} < 65%)")

        # Check 2: Risk/Reward ratio
        if signal.signal_type == SignalType.BUY:
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)
        else:
            risk = abs(signal.stop_loss - signal.entry_price)
            reward = abs(signal.entry_price - signal.take_profit)

        rr_ratio = reward / risk if risk > 0 else 0
        if rr_ratio < 1.5 and signal.signal_type != SignalType.HOLD:
            approved = False
            notes.append(f"R:R ratio too low ({rr_ratio:.1f} < 1.5)")

        # Check 3: Anomaly detection
        anomaly_score = anomaly_scores.get("combined_anomaly_score", 0)
        if anomaly_score > 0.9:
            approved = False
            notes.append(f"Extreme anomaly detected ({anomaly_score:.2f})")

        # Check 4: Volatility regime
        vol_regime = regime.get("volatility_regime", "medium")
        if vol_regime == "extreme":
            notes.append("EXTREME volatility - reduce position size")
            signal.position_size_pct = min(signal.position_size_pct, 1.0)

        # Update signal with risk assessment
        if approved:
            signal.risk_assessment = "APPROVED: " + ("; ".join(notes) if notes else "All checks passed")
        else:
            signal.risk_assessment = "REJECTED: " + "; ".join(notes)

        state["risk_approved"] = approved
        state["risk_notes"] = "; ".join(notes) if notes else "All checks passed"
        state["aggregated_signal"] = signal
        state["timing"]["risk_check"] = time.time() - start

        return state

    def analyze(
        self,
        symbol: str,
        token_address: Optional[str] = None,
        chain: str = "ethereum",
        timeframe: str = "1h",
    ) -> AgentState:
        """
        Run complete analysis pipeline.

        Args:
            symbol: Trading pair (e.g., 'AAVE/USDT')
            token_address: Token contract address for on-chain
            chain: Blockchain for on-chain data
            timeframe: Chart timeframe

        Returns:
            Final AgentState with all analysis
        """
        # Create initial state
        state = create_initial_state(
            symbol=symbol,
            token_address=token_address,
            chain=chain,
            timeframe=timeframe,
        )

        # Run the graph
        result = self.graph.invoke(state)

        return result

    def get_signal(
        self,
        symbol: str,
        token_address: Optional[str] = None,
        chain: str = "ethereum",
        timeframe: str = "1h",
    ) -> Optional[TradingSignal]:
        """
        Get trading signal for a symbol.

        Args:
            symbol: Trading pair
            token_address: Token contract address
            chain: Blockchain
            timeframe: Timeframe

        Returns:
            TradingSignal if approved, None otherwise
        """
        result = self.analyze(symbol, token_address, chain, timeframe)

        if result.get("risk_approved"):
            return result.get("aggregated_signal")
        return None


def create_orchestrator(
    exchange: str = "binance",
    chains: Optional[List[str]] = None,
) -> TradingOrchestrator:
    """Create a TradingOrchestrator instance."""
    return TradingOrchestrator(exchange=exchange, chains=chains)


if __name__ == "__main__":
    print("Testing TradingOrchestrator...")

    orchestrator = TradingOrchestrator(
        exchange="binance",
        chains=["ethereum"],
    )

    # Test with BTC (no on-chain)
    print("\n1. Analyzing BTC/USDT (no on-chain)...")
    result = orchestrator.analyze(
        symbol="BTC/USDT",
        timeframe="1h",
    )

    print(f"\nResults:")
    print(f"  Symbol: {result.get('symbol')}")
    print(f"  Latest Price: ${result.get('latest_price', 0):,.2f}")

    if result.get("technical_analysis"):
        ta = result["technical_analysis"]
        print(f"  Technical: {ta.trend} (confidence: {ta.confidence:.0%})")

    if result.get("regime_state"):
        regime = result["regime_state"]
        print(f"  Regime: {regime.get('market_regime')} | Volatility: {regime.get('volatility_regime')}")

    if result.get("aggregated_signal"):
        signal = result["aggregated_signal"]
        print(f"  Signal: {signal.signal_type.value.upper()} ({signal.strength.value})")
        print(f"  Confidence: {signal.confidence:.0%}")
        print(f"  Entry: ${signal.entry_price:,.2f}")
        print(f"  Stop Loss: ${signal.stop_loss:,.2f}")
        print(f"  Take Profit: ${signal.take_profit:,.2f}")

    print(f"  Risk Approved: {result.get('risk_approved')}")
    print(f"  Risk Notes: {result.get('risk_notes')}")

    # Print timing
    timing = result.get("timing", {})
    print(f"\nTiming:")
    for step, seconds in timing.items():
        print(f"  {step}: {seconds:.2f}s")
    print(f"  Total: {sum(timing.values()):.2f}s")
