"""
Risk Agent

Risk management agent with VETO power over trading signals.
Enforces risk parameters and portfolio constraints.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import AIMessage

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import (
    AgentState,
    TradingSignal,
    SignalType,
    SignalStrength,
)


class RiskDecision(Enum):
    """Risk decision types."""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    decision: RiskDecision
    original_signal: TradingSignal
    modified_signal: Optional[TradingSignal]
    checks_passed: List[str]
    checks_failed: List[str]
    warnings: List[str]
    notes: str


class RiskAgent:
    """
    Risk management agent with VETO power.

    Evaluates signals against risk parameters and
    portfolio constraints.
    """

    def __init__(
        self,
        max_position_size: float = 0.02,  # 2%
        max_stop_loss: float = 0.05,      # 5%
        min_rr_ratio: float = 1.5,
        min_confidence: float = 0.65,
        max_open_positions: int = 5,
        cooldown_hours: int = 4,
        max_daily_drawdown: float = 0.03,  # 3%
        max_anomaly_score: float = 0.9,
    ):
        """
        Initialize the risk agent.

        Args:
            max_position_size: Maximum position size as decimal
            max_stop_loss: Maximum stop-loss percentage
            min_rr_ratio: Minimum risk/reward ratio
            min_confidence: Minimum signal confidence
            max_open_positions: Maximum concurrent positions
            cooldown_hours: Hours between same-asset signals
            max_daily_drawdown: Maximum daily drawdown
            max_anomaly_score: Maximum anomaly score to trade
        """
        self.max_position_size = max_position_size
        self.max_stop_loss = max_stop_loss
        self.min_rr_ratio = min_rr_ratio
        self.min_confidence = min_confidence
        self.max_open_positions = max_open_positions
        self.cooldown_hours = cooldown_hours
        self.max_daily_drawdown = max_daily_drawdown
        self.max_anomaly_score = max_anomaly_score

        # Track recent signals
        self._signal_history: Dict[str, datetime] = {}
        self._open_positions: Dict[str, datetime] = {}
        self._daily_pnl: float = 0

    def assess(
        self,
        signal: TradingSignal,
        anomaly_scores: Optional[Dict] = None,
        regime_state: Optional[Dict] = None,
        current_positions: Optional[Dict] = None,
    ) -> RiskAssessment:
        """
        Assess a trading signal against risk parameters.

        Args:
            signal: Trading signal to evaluate
            anomaly_scores: Anomaly detection scores
            regime_state: Current market regime
            current_positions: Current open positions

        Returns:
            RiskAssessment with decision
        """
        checks_passed = []
        checks_failed = []
        warnings = []

        # Skip HOLD signals
        if signal.signal_type == SignalType.HOLD:
            return RiskAssessment(
                decision=RiskDecision.APPROVED,
                original_signal=signal,
                modified_signal=None,
                checks_passed=["HOLD signal - no risk checks needed"],
                checks_failed=[],
                warnings=[],
                notes="HOLD signals pass automatically",
            )

        # Check 1: Minimum confidence
        if signal.confidence < self.min_confidence:
            checks_failed.append(
                f"Confidence too low ({signal.confidence:.0%} < {self.min_confidence:.0%})"
            )
        else:
            checks_passed.append(f"Confidence OK ({signal.confidence:.0%})")

        # Check 2: Risk/Reward ratio
        rr_ratio = self._calculate_rr_ratio(signal)
        if rr_ratio < self.min_rr_ratio:
            checks_failed.append(
                f"R:R ratio too low ({rr_ratio:.2f} < {self.min_rr_ratio})"
            )
        else:
            checks_passed.append(f"R:R ratio OK ({rr_ratio:.2f})")

        # Check 3: Stop-loss size
        sl_pct = self._calculate_sl_percentage(signal)
        if sl_pct > self.max_stop_loss * 100:
            checks_failed.append(
                f"Stop-loss too wide ({sl_pct:.1f}% > {self.max_stop_loss*100:.1f}%)"
            )
        else:
            checks_passed.append(f"Stop-loss OK ({sl_pct:.1f}%)")

        # Check 4: Position size
        if signal.position_size_pct > self.max_position_size * 100:
            warnings.append(
                f"Position size reduced from {signal.position_size_pct:.1f}% to {self.max_position_size*100:.1f}%"
            )
        else:
            checks_passed.append(f"Position size OK ({signal.position_size_pct:.1f}%)")

        # Check 5: Anomaly detection
        if anomaly_scores:
            anomaly = anomaly_scores.get("combined_anomaly_score", 0)
            if anomaly > self.max_anomaly_score:
                checks_failed.append(
                    f"Extreme anomaly detected ({anomaly:.2f} > {self.max_anomaly_score})"
                )
            elif anomaly > 0.7:
                warnings.append(f"Elevated anomaly score ({anomaly:.2f})")
            else:
                checks_passed.append(f"Anomaly OK ({anomaly:.2f})")

        # Check 6: Volatility regime
        if regime_state:
            vol_regime = regime_state.get("volatility_regime", "medium")
            if vol_regime == "extreme":
                warnings.append("EXTREME volatility - position size reduced")
            elif vol_regime == "high":
                warnings.append("HIGH volatility - use caution")
            else:
                checks_passed.append(f"Volatility OK ({vol_regime})")

        # Check 7: Cooldown period
        symbol = signal.symbol
        if symbol in self._signal_history:
            last_signal = self._signal_history[symbol]
            hours_since = (datetime.now() - last_signal).total_seconds() / 3600
            if hours_since < self.cooldown_hours:
                checks_failed.append(
                    f"Cooldown active ({hours_since:.1f}h < {self.cooldown_hours}h)"
                )
            else:
                checks_passed.append("Cooldown OK")
        else:
            checks_passed.append("No prior signals")

        # Check 8: Maximum positions (if tracking)
        if current_positions and len(current_positions) >= self.max_open_positions:
            checks_failed.append(
                f"Max positions reached ({len(current_positions)} >= {self.max_open_positions})"
            )

        # Make decision
        if checks_failed:
            decision = RiskDecision.REJECTED
            notes = "Signal REJECTED: " + "; ".join(checks_failed)
        elif warnings:
            decision = RiskDecision.MODIFIED
            notes = "Signal MODIFIED: " + "; ".join(warnings)
        else:
            decision = RiskDecision.APPROVED
            notes = "Signal APPROVED: All checks passed"

        # Create modified signal if needed
        modified_signal = None
        if decision in [RiskDecision.APPROVED, RiskDecision.MODIFIED]:
            modified_signal = self._create_modified_signal(
                signal, regime_state, warnings
            )
            # Update history
            self._signal_history[symbol] = datetime.now()

        return RiskAssessment(
            decision=decision,
            original_signal=signal,
            modified_signal=modified_signal,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            notes=notes,
        )

    def _calculate_rr_ratio(self, signal: TradingSignal) -> float:
        """Calculate risk/reward ratio."""
        if signal.signal_type == SignalType.BUY:
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)
        elif signal.signal_type == SignalType.SELL:
            risk = abs(signal.stop_loss - signal.entry_price)
            reward = abs(signal.entry_price - signal.take_profit)
        else:
            return 0

        return reward / risk if risk > 0 else 0

    def _calculate_sl_percentage(self, signal: TradingSignal) -> float:
        """Calculate stop-loss percentage."""
        if signal.signal_type == SignalType.BUY:
            return abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
        elif signal.signal_type == SignalType.SELL:
            return abs(signal.stop_loss - signal.entry_price) / signal.entry_price * 100
        return 0

    def _create_modified_signal(
        self,
        signal: TradingSignal,
        regime_state: Optional[Dict],
        warnings: List[str],
    ) -> TradingSignal:
        """Create a modified signal with adjusted parameters."""
        # Start with original values
        position_size = signal.position_size_pct

        # Cap position size
        position_size = min(position_size, self.max_position_size * 100)

        # Reduce for volatility
        if regime_state:
            vol_regime = regime_state.get("volatility_regime", "medium")
            if vol_regime == "extreme":
                position_size *= 0.5
            elif vol_regime == "high":
                position_size *= 0.75

        # Build risk assessment text
        risk_text = f"APPROVED - Position: {position_size:.1f}%"
        if warnings:
            risk_text += f" | Warnings: {len(warnings)}"

        return TradingSignal(
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            strength=signal.strength,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size_pct=position_size,
            confidence=signal.confidence,
            reasoning=signal.reasoning,
            technical_summary=signal.technical_summary,
            onchain_summary=signal.onchain_summary,
            risk_assessment=risk_text,
            timestamp=signal.timestamp,
        )

    def clear_history(self) -> None:
        """Clear signal history."""
        self._signal_history.clear()
        self._open_positions.clear()
        self._daily_pnl = 0

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk state summary."""
        return {
            "open_positions": len(self._open_positions),
            "max_positions": self.max_open_positions,
            "daily_pnl": self._daily_pnl,
            "max_daily_drawdown": self.max_daily_drawdown,
            "recent_signals": len(self._signal_history),
            "parameters": {
                "max_position_size": f"{self.max_position_size*100:.1f}%",
                "max_stop_loss": f"{self.max_stop_loss*100:.1f}%",
                "min_rr_ratio": self.min_rr_ratio,
                "min_confidence": f"{self.min_confidence*100:.0f}%",
                "cooldown_hours": self.cooldown_hours,
            },
        }

    def __call__(self, state: AgentState) -> AgentState:
        """Run agent as part of LangGraph workflow."""
        signal = state.get("aggregated_signal")

        if not signal:
            state["risk_approved"] = False
            state["risk_notes"] = "No signal to evaluate"
            return state

        # Assess signal
        assessment = self.assess(
            signal=signal,
            anomaly_scores=state.get("anomaly_scores"),
            regime_state=state.get("regime_state"),
        )

        # Update state
        state["risk_approved"] = assessment.decision in [
            RiskDecision.APPROVED,
            RiskDecision.MODIFIED,
        ]
        state["risk_notes"] = assessment.notes

        # Use modified signal if available
        if assessment.modified_signal:
            state["aggregated_signal"] = assessment.modified_signal

        # Add to messages
        state["messages"] = state.get("messages", [])
        state["messages"].append(
            AIMessage(content=f"Risk: {assessment.decision.value.upper()} - {assessment.notes}")
        )

        return state


def create_risk_agent(
    max_position_size: float = 0.02,
    max_stop_loss: float = 0.05,
    min_rr_ratio: float = 1.5,
    min_confidence: float = 0.65,
) -> RiskAgent:
    """Create a RiskAgent instance."""
    return RiskAgent(
        max_position_size=max_position_size,
        max_stop_loss=max_stop_loss,
        min_rr_ratio=min_rr_ratio,
        min_confidence=min_confidence,
    )


if __name__ == "__main__":
    print("Testing RiskAgent...")

    risk = RiskAgent()

    # Test 1: Good signal
    print("\n1. Testing good signal...")
    good_signal = TradingSignal(
        symbol="BTC/USDT",
        signal_type=SignalType.BUY,
        strength=SignalStrength.MODERATE,
        entry_price=96500,
        stop_loss=91675,  # 5% SL
        take_profit=104220,  # 8% TP, R:R = 1.6
        position_size_pct=1.5,
        confidence=0.72,
        reasoning="Strong technical + on-chain",
        technical_summary="Bullish trend",
        onchain_summary="Whale accumulation",
        risk_assessment="Pending",
        timestamp=datetime.now(),
    )

    result = risk.assess(good_signal)
    print(f"   Decision: {result.decision.value}")
    print(f"   Notes: {result.notes}")
    print(f"   Passed: {result.checks_passed}")
    print(f"   Failed: {result.checks_failed}")

    # Test 2: Bad signal (low confidence)
    print("\n2. Testing low confidence signal...")
    bad_signal = TradingSignal(
        symbol="ETH/USDT",
        signal_type=SignalType.BUY,
        strength=SignalStrength.WEAK,
        entry_price=3500,
        stop_loss=3325,
        take_profit=3675,
        position_size_pct=2.0,
        confidence=0.55,  # Too low
        reasoning="Weak signals",
        technical_summary="Neutral",
        onchain_summary="Mixed",
        risk_assessment="Pending",
        timestamp=datetime.now(),
    )

    result = risk.assess(bad_signal)
    print(f"   Decision: {result.decision.value}")
    print(f"   Notes: {result.notes}")

    # Test 3: Signal during anomaly
    print("\n3. Testing signal with high anomaly...")
    result = risk.assess(
        good_signal,
        anomaly_scores={"combined_anomaly_score": 0.95},
    )
    print(f"   Decision: {result.decision.value}")
    print(f"   Notes: {result.notes}")

    # Print risk summary
    print("\n4. Risk Summary:")
    summary = risk.get_risk_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
