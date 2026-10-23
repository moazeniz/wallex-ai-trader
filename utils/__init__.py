"""
Utilities Layer

This module contains shared utilities:
- Logging configuration
- Agent prompts
- Helper functions
"""

from utils.prompts import (
    TECHNICAL_ANALYST_PROMPT,
    ONCHAIN_ANALYST_PROMPT,
    STRATEGY_AGENT_PROMPT,
    SIGNAL_AGGREGATOR_PROMPT,
    RISK_AGENT_PROMPT,
    ORCHESTRATOR_PROMPT,
    get_technical_prompt,
    get_onchain_prompt,
    get_strategy_prompt,
    get_aggregator_prompt,
    get_risk_prompt,
)
from utils.logger import (
    StructuredLogger,
    TimingContext,
    get_logger,
    configure_logging,
)

__all__ = [
    # Prompts
    "TECHNICAL_ANALYST_PROMPT",
    "ONCHAIN_ANALYST_PROMPT",
    "STRATEGY_AGENT_PROMPT",
    "SIGNAL_AGGREGATOR_PROMPT",
    "RISK_AGENT_PROMPT",
    "ORCHESTRATOR_PROMPT",
    "get_technical_prompt",
    "get_onchain_prompt",
    "get_strategy_prompt",
    "get_aggregator_prompt",
    "get_risk_prompt",
    # Logging
    "StructuredLogger",
    "TimingContext",
    "get_logger",
    "configure_logging",
]
