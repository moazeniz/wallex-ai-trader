"""
Strategy Agent

LLM agent that generates and validates trading strategies.
Uses code generation to create backtestable strategies.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import AgentState, StrategyResult
from tools.python_executor import PythonExecutor
from tools.backtest_tools import BacktestEngine
from utils.prompts import STRATEGY_AGENT_PROMPT
from settings import load_config
from data.exchange_client import ExchangeClient


class StrategyAgent:
    """
    Strategy generation and validation agent.

    Generates Python trading strategies and backtests them.
    """

    def __init__(
        self,
        exchange: str = "binance",
        model_name: str = "gemini-3-flash-preview",
    ):
        """
        Initialize the strategy agent.

        Args:
            exchange: Exchange for data
            model_name: LLM model to use
        """
        self.exchange = exchange
        self.model_name = model_name

        # Initialize components
        self.executor = PythonExecutor(timeout=30)
        self.backtest_engine = BacktestEngine(initial_capital=10000)
        self.exchange_client = ExchangeClient(exchange=exchange)

        # Initialize LLM
        config = load_config()
        api_key = config.llm.gemini_api_keys[0] if config.llm.gemini_api_keys else None

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.4,
            max_output_tokens=4096,
            max_retries=0,  # Disable internal retries
            timeout=30,
        )

    def generate_strategy(
        self,
        symbol: str,
        analysis_context: str,
        strategy_type: str = "momentum",
    ) -> StrategyResult:
        """
        Generate a trading strategy based on analysis.

        Args:
            symbol: Trading pair
            analysis_context: Context from other agents
            strategy_type: Type of strategy to generate

        Returns:
            StrategyResult with code and backtest
        """
        # Generate strategy code
        code = self._generate_strategy_code(symbol, analysis_context, strategy_type)

        # Fetch data for backtest
        df = self.exchange_client.fetch_ohlcv(
            symbol=symbol,
            timeframe="1h",
            limit=500,
        )

        # Execute and backtest
        result = self._execute_and_backtest(code, df, strategy_type)

        return result

    def _generate_strategy_code(
        self,
        symbol: str,
        analysis_context: str,
        strategy_type: str,
    ) -> str:
        """Generate Python strategy code using LLM."""
        prompt = f"""{STRATEGY_AGENT_PROMPT}

## Task: Generate a {strategy_type} trading strategy for {symbol}

## Analysis Context:
{analysis_context}

## Requirements:
1. Use pandas DataFrame 'df' with columns: open, high, low, close, volume
2. Create 'entries' (boolean Series) for buy signals
3. Create 'exits' (boolean Series) for sell signals
4. Use only allowed imports: pandas (pd), numpy (np)
5. Keep the strategy simple and interpretable

## Output Format:
Provide ONLY the Python code, no explanations:

```python
# Strategy: [name]
# Entry: [brief description]
# Exit: [brief description]

# Calculate indicators
...

# Generate signals
entries = ...
exits = ...

# Store for backtest
result = {{'entries': entries, 'exits': exits}}
```
"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Extract code block
            if "```python" in content:
                start = content.find("```python") + 9
                end = content.find("```", start)
                code = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                code = content[start:end].strip()
            else:
                code = content.strip()

            return code

        except Exception as e:
            # Default simple strategy
            return self._get_default_strategy(strategy_type)

    def _get_default_strategy(self, strategy_type: str) -> str:
        """Get a default strategy template."""
        if strategy_type == "momentum":
            return """
# Strategy: RSI Momentum
# Entry: RSI < 30 (oversold)
# Exit: RSI > 70 (overbought)

# Calculate RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))

# Generate signals
entries = rsi < 30
exits = rsi > 70

result = {'entries': entries, 'exits': exits}
"""
        elif strategy_type == "trend":
            return """
# Strategy: MA Crossover
# Entry: Fast MA crosses above Slow MA
# Exit: Fast MA crosses below Slow MA

fast_ma = df['close'].rolling(10).mean()
slow_ma = df['close'].rolling(30).mean()

entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

result = {'entries': entries, 'exits': exits}
"""
        else:
            return """
# Strategy: Simple Breakout
# Entry: Price breaks above 20-period high
# Exit: Price breaks below 20-period low

high_20 = df['high'].rolling(20).max()
low_20 = df['low'].rolling(20).min()

entries = df['close'] > high_20.shift(1)
exits = df['close'] < low_20.shift(1)

result = {'entries': entries, 'exits': exits}
"""

    def _execute_and_backtest(
        self,
        code: str,
        df: "pd.DataFrame",
        strategy_name: str,
    ) -> StrategyResult:
        """Execute strategy code and run backtest."""
        # Execute code
        exec_result = self.executor.execute_strategy(code, df)

        if not exec_result.success:
            return StrategyResult(
                strategy_name=strategy_name,
                code=code,
                backtest_metrics={},
                win_rate=0,
                sharpe_ratio=0,
                max_drawdown=100,
                total_return=0,
                is_profitable=False,
                validation_notes=f"Execution failed: {exec_result.error}",
            )

        # Get signals
        signals = exec_result.output
        if not signals or "entries" not in signals or "exits" not in signals:
            return StrategyResult(
                strategy_name=strategy_name,
                code=code,
                backtest_metrics={},
                win_rate=0,
                sharpe_ratio=0,
                max_drawdown=100,
                total_return=0,
                is_profitable=False,
                validation_notes="Strategy did not return valid signals",
            )

        entries = signals["entries"]
        exits = signals["exits"]

        # Run backtest
        try:
            backtest = self.backtest_engine.run_backtest(
                df=df,
                entries=entries,
                exits=exits,
                strategy_name=strategy_name,
            )

            # Validate strategy
            validation_notes = self._validate_strategy(backtest)

            return StrategyResult(
                strategy_name=strategy_name,
                code=code,
                backtest_metrics={
                    "total_return_pct": backtest.total_return_pct,
                    "sharpe_ratio": backtest.sharpe_ratio,
                    "max_drawdown": backtest.max_drawdown,
                    "total_trades": backtest.total_trades,
                    "profit_factor": backtest.profit_factor,
                },
                win_rate=backtest.win_rate,
                sharpe_ratio=backtest.sharpe_ratio,
                max_drawdown=backtest.max_drawdown,
                total_return=backtest.total_return_pct,
                is_profitable=backtest.is_profitable,
                validation_notes=validation_notes,
            )

        except Exception as e:
            return StrategyResult(
                strategy_name=strategy_name,
                code=code,
                backtest_metrics={},
                win_rate=0,
                sharpe_ratio=0,
                max_drawdown=100,
                total_return=0,
                is_profitable=False,
                validation_notes=f"Backtest failed: {e}",
            )

    def _validate_strategy(self, backtest) -> str:
        """Validate strategy quality."""
        notes = []

        # Check minimum trades
        if backtest.total_trades < 5:
            notes.append("Low trade count - may not be statistically significant")

        # Check win rate
        if backtest.win_rate < 40:
            notes.append(f"Low win rate ({backtest.win_rate:.1f}%)")
        elif backtest.win_rate > 70:
            notes.append(f"High win rate ({backtest.win_rate:.1f}%) - verify no lookahead bias")

        # Check Sharpe ratio
        if backtest.sharpe_ratio < 0.5:
            notes.append("Low risk-adjusted returns")
        elif backtest.sharpe_ratio > 3:
            notes.append("Unusually high Sharpe - verify calculations")

        # Check drawdown
        if backtest.max_drawdown > 20:
            notes.append(f"High max drawdown ({backtest.max_drawdown:.1f}%)")

        if not notes:
            notes.append("Strategy passed basic validation")

        return "; ".join(notes)

    def __call__(self, state: AgentState) -> AgentState:
        """Run agent as part of LangGraph workflow."""
        symbol = state.get("symbol", "BTC/USDT")

        # Build context from other analyses
        context_parts = []

        if state.get("technical_analysis"):
            ta = state["technical_analysis"]
            context_parts.append(f"Technical: {ta.trend} trend, {ta.momentum} momentum")

        if state.get("onchain_analysis"):
            oc = state["onchain_analysis"]
            context_parts.append(f"On-Chain: {oc.whale_activity}, {oc.exchange_flow}")

        if state.get("regime_state"):
            regime = state["regime_state"]
            context_parts.append(f"Regime: {regime.get('market_regime', 'unknown')}")

        context = "\n".join(context_parts) if context_parts else "No prior analysis available"

        # Generate strategy
        start_time = datetime.now()
        result = self.generate_strategy(symbol, context, "momentum")
        end_time = datetime.now()

        # Update state
        state["strategy_result"] = result
        state["timing"] = state.get("timing", {})
        state["timing"]["strategy_generation"] = (end_time - start_time).total_seconds()

        state["messages"] = state.get("messages", [])
        state["messages"].append(
            AIMessage(content=f"Strategy Generated: {result.validation_notes}")
        )

        return state


def create_strategy_agent(
    exchange: str = "binance",
) -> StrategyAgent:
    """Create a StrategyAgent instance."""
    return StrategyAgent(exchange=exchange)


if __name__ == "__main__":
    print("Testing StrategyAgent...")

    agent = StrategyAgent(exchange="binance")

    # Test strategy generation
    print("\nGenerating momentum strategy for BTC/USDT...")
    context = """
    Technical: Bullish trend, neutral momentum, RSI at 55
    On-Chain: Whale accumulation detected
    Regime: Sideways market with low volatility
    """

    result = agent.generate_strategy("BTC/USDT", context, "momentum")

    print(f"\nResults:")
    print(f"  Strategy: {result.strategy_name}")
    print(f"  Win Rate: {result.win_rate:.1f}%")
    print(f"  Sharpe: {result.sharpe_ratio:.2f}")
    print(f"  Max DD: {result.max_drawdown:.1f}%")
    print(f"  Return: {result.total_return:.1f}%")
    print(f"  Profitable: {result.is_profitable}")
    print(f"  Notes: {result.validation_notes}")
