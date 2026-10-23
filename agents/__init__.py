"""
LLM Agent Layer

This module contains the LangGraph-based multi-agent system:
- Technical Analyst Agent
- On-Chain Analyst Agent
- Strategy Agent (code generation)
- Signal Aggregator Agent
- Risk Agent
- Orchestrator (LangGraph state machine)

Note: Uses lazy loading to avoid importing LLM dependencies for lite modules.
"""

# Import state definitions directly (they don't require LLM)
from .state import (
    AgentState,
    TechnicalAnalysis,
    OnChainAnalysis,
    StrategyResult,
    TradingSignal,
    SignalType,
    SignalStrength,
    create_initial_state,
    format_signal_for_telegram,
)


def __getattr__(name):
    """Lazy import for LLM-dependent modules."""
    if name == "TechnicalAnalystAgent":
        from .technical_agent import TechnicalAnalystAgent
        return TechnicalAnalystAgent
    elif name == "create_technical_agent":
        from .technical_agent import create_technical_agent
        return create_technical_agent
    elif name == "OnChainAnalystAgent":
        from .onchain_agent import OnChainAnalystAgent
        return OnChainAnalystAgent
    elif name == "create_onchain_agent":
        from .onchain_agent import create_onchain_agent
        return create_onchain_agent
    elif name == "StrategyAgent":
        from .strategy_agent import StrategyAgent
        return StrategyAgent
    elif name == "create_strategy_agent":
        from .strategy_agent import create_strategy_agent
        return create_strategy_agent
    elif name == "SignalAggregator":
        from .signal_agent import SignalAggregator
        return SignalAggregator
    elif name == "create_signal_aggregator":
        from .signal_agent import create_signal_aggregator
        return create_signal_aggregator
    elif name == "RiskAgent":
        from .risk_agent import RiskAgent
        return RiskAgent
    elif name == "create_risk_agent":
        from .risk_agent import create_risk_agent
        return create_risk_agent
    elif name == "RiskAssessment":
        from .risk_agent import RiskAssessment
        return RiskAssessment
    elif name == "RiskDecision":
        from .risk_agent import RiskDecision
        return RiskDecision
    elif name == "TradingOrchestrator":
        from .orchestrator import TradingOrchestrator
        return TradingOrchestrator
    elif name == "create_orchestrator":
        from .orchestrator import create_orchestrator
        return create_orchestrator
    else:
        raise AttributeError(f"module 'agents' has no attribute '{name}'")


__all__ = [
    # State (loaded eagerly - no LLM deps)
    "AgentState",
    "TechnicalAnalysis",
    "OnChainAnalysis",
    "StrategyResult",
    "TradingSignal",
    "SignalType",
    "SignalStrength",
    "create_initial_state",
    "format_signal_for_telegram",
    # Agents (lazy loaded - require LLM)
    "TechnicalAnalystAgent",
    "create_technical_agent",
    "OnChainAnalystAgent",
    "create_onchain_agent",
    "StrategyAgent",
    "create_strategy_agent",
    "SignalAggregator",
    "create_signal_aggregator",
    "RiskAgent",
    "create_risk_agent",
    "RiskAssessment",
    "RiskDecision",
    # Orchestrator (lazy loaded)
    "TradingOrchestrator",
    "create_orchestrator",
]
