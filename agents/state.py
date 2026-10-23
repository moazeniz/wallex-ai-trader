"""
Agent State Definitions

Defines the state structure for the LangGraph-based multi-agent system.
Uses TypedDict for type-safe state management.
"""

from typing import Dict, List, Optional, Any, Annotated, Sequence
from typing_extensions import TypedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import operator


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class SignalStrength(Enum):
    """Signal strength levels."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


@dataclass
class TechnicalAnalysis:
    """Technical analysis result."""
    symbol: str
    timeframe: str
    timestamp: datetime
    trend: str  # bullish, bearish, neutral
    momentum: str  # overbought, oversold, neutral
    volatility: str  # low, medium, high
    support_levels: List[float]
    resistance_levels: List[float]
    key_indicators: Dict[str, float]
    signals: List[str]
    summary: str
    confidence: float


@dataclass
class OnChainAnalysis:
    """On-chain analysis result."""
    token_address: str
    chain: str
    timestamp: datetime
    whale_activity: str  # accumulating, distributing, neutral
    exchange_flow: str  # inflow, outflow, balanced
    holder_trend: str  # growing, shrinking, stable
    key_metrics: Dict[str, float]
    alerts: List[str]
    summary: str
    confidence: float


@dataclass
class StrategyResult:
    """Strategy backtesting result."""
    strategy_name: str
    code: str
    backtest_metrics: Dict[str, float]
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    is_profitable: bool
    validation_notes: str


@dataclass
class TradingSignal:
    """Final trading signal."""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    confidence: float
    reasoning: str
    technical_summary: str
    onchain_summary: str
    risk_assessment: str
    timestamp: datetime


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, preferring values from b."""
    result = a.copy()
    result.update(b)
    return result


class AgentState(TypedDict, total=False):
    """
    Main state for the multi-agent system.

    This state is passed between all agents in the LangGraph workflow.
    """
    # Input
    symbol: str
    token_address: Optional[str]
    chain: str
    timeframe: str

    # Messages for LLM context
    messages: Annotated[Sequence[Any], operator.add]

    # Analysis results
    technical_analysis: Optional[TechnicalAnalysis]
    onchain_analysis: Optional[OnChainAnalysis]
    strategy_result: Optional[StrategyResult]

    # ML model outputs
    anomaly_scores: Dict[str, float]
    regime_state: Dict[str, Any]
    pattern_analysis: Dict[str, Any]
    whale_profiles: Dict[str, Any]

    # Feature data
    features: Dict[str, float]
    latest_price: float

    # Signal generation
    raw_signals: List[Dict[str, Any]]
    aggregated_signal: Optional[TradingSignal]

    # Risk assessment
    risk_approved: bool
    risk_notes: str
    position_size: float

    # Metadata
    timestamp: datetime
    errors: List[str]
    timing: Annotated[Dict[str, float], merge_dicts]


class AnalysisRequest(TypedDict):
    """Request for token analysis."""
    symbol: str
    token_address: Optional[str]
    chain: str
    timeframe: str
    include_onchain: bool
    include_backtest: bool


class SignalOutput(TypedDict):
    """Output signal structure."""
    symbol: str
    signal: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    reasoning: str
    timestamp: str


def create_initial_state(
    symbol: str,
    token_address: Optional[str] = None,
    chain: str = "ethereum",
    timeframe: str = "1h",
) -> AgentState:
    """
    Create initial agent state for analysis.

    Args:
        symbol: Trading pair (e.g., 'AAVE/USDT')
        token_address: Token contract address
        chain: Blockchain for on-chain data
        timeframe: Chart timeframe

    Returns:
        Initial AgentState
    """
    return AgentState(
        symbol=symbol,
        token_address=token_address,
        chain=chain,
        timeframe=timeframe,
        messages=[],
        technical_analysis=None,
        onchain_analysis=None,
        strategy_result=None,
        anomaly_scores={},
        regime_state={},
        pattern_analysis={},
        whale_profiles={},
        features={},
        latest_price=0.0,
        raw_signals=[],
        aggregated_signal=None,
        risk_approved=False,
        risk_notes="",
        position_size=0.0,
        timestamp=datetime.now(),
        errors=[],
        timing={},
    )


def format_signal_for_telegram(signal: TradingSignal) -> str:
    """
    Format a trading signal for Telegram delivery.

    Args:
        signal: TradingSignal to format

    Returns:
        Formatted message string
    """
    # Signal emoji
    if signal.signal_type == SignalType.BUY:
        emoji = "🟢"
        action = "BUY"
    elif signal.signal_type == SignalType.SELL:
        emoji = "🔴"
        action = "SELL"
    else:
        emoji = "⚪"
        action = "HOLD"

    # Strength indicator
    strength_stars = {
        SignalStrength.STRONG: "⭐⭐⭐",
        SignalStrength.MODERATE: "⭐⭐",
        SignalStrength.WEAK: "⭐",
    }
    stars = strength_stars.get(signal.strength, "⭐")

    # Calculate risk/reward
    if signal.signal_type == SignalType.BUY:
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
    else:
        risk = abs(signal.stop_loss - signal.entry_price)
        reward = abs(signal.entry_price - signal.take_profit)

    rr_ratio = reward / risk if risk > 0 else 0

    message = f"""
{emoji} **{action} SIGNAL: {signal.symbol}** {stars}

**Confidence:** {signal.confidence:.0%}
**Entry:** ${signal.entry_price:,.4f}
**Stop Loss:** ${signal.stop_loss:,.4f} ({abs((signal.stop_loss - signal.entry_price) / signal.entry_price * 100):.1f}%)
**Take Profit:** ${signal.take_profit:,.4f} ({abs((signal.take_profit - signal.entry_price) / signal.entry_price * 100):.1f}%)
**Risk/Reward:** {rr_ratio:.1f}:1
**Position Size:** {signal.position_size_pct:.1f}%

---
**Technical:** {signal.technical_summary}

**On-Chain:** {signal.onchain_summary}

**Risk:** {signal.risk_assessment}

---
_{signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}_
"""
    return message.strip()
