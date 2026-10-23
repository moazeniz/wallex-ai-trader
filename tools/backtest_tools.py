"""
Backtesting Tools

Provides backtesting capabilities using vectorbt.
Used by Strategy Agent to validate trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False
    print("Warning: vectorbt not installed. Backtesting will use simplified methods.")


@dataclass
class BacktestResult:
    """Result of a backtest."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_trade_duration: float
    is_profitable: bool
    metrics: Dict[str, float]


class BacktestEngine:
    """
    Backtesting engine for trading strategies.

    Uses vectorbt for fast vectorized backtesting when available,
    falls back to simple simulation otherwise.
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,   # 0.05%
    ):
        """
        Initialize the backtest engine.

        Args:
            initial_capital: Starting capital
            commission: Commission per trade (as decimal)
            slippage: Slippage per trade (as decimal)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run_backtest(
        self,
        df: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        strategy_name: str = "Strategy",
        position_size: float = 1.0,
    ) -> BacktestResult:
        """
        Run a backtest with entry/exit signals.

        Args:
            df: OHLCV DataFrame
            entries: Boolean series of entry signals
            exits: Boolean series of exit signals
            strategy_name: Name for the strategy
            position_size: Position size as fraction of capital

        Returns:
            BacktestResult with metrics
        """
        if HAS_VBT:
            return self._run_vbt_backtest(df, entries, exits, strategy_name, position_size)
        else:
            return self._run_simple_backtest(df, entries, exits, strategy_name, position_size)

    def _run_vbt_backtest(
        self,
        df: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        strategy_name: str,
        position_size: float,
    ) -> BacktestResult:
        """Run backtest using vectorbt."""
        price = df["close"]

        # Create portfolio
        pf = vbt.Portfolio.from_signals(
            close=price,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            size=position_size,
            fees=self.commission,
            slippage=self.slippage,
        )

        # Extract metrics
        stats = pf.stats()

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=df.index[0] if hasattr(df.index[0], 'to_pydatetime') else datetime.now(),
            end_date=df.index[-1] if hasattr(df.index[-1], 'to_pydatetime') else datetime.now(),
            initial_capital=self.initial_capital,
            final_value=float(stats.get("End Value", self.initial_capital)),
            total_return=float(stats.get("Total Return", 0)),
            total_return_pct=float(stats.get("Total Return [%]", 0)),
            sharpe_ratio=float(stats.get("Sharpe Ratio", 0)),
            sortino_ratio=float(stats.get("Sortino Ratio", 0)),
            max_drawdown=float(stats.get("Max Drawdown [%]", 0)),
            win_rate=float(stats.get("Win Rate [%]", 0)),
            profit_factor=float(stats.get("Profit Factor", 0)),
            total_trades=int(stats.get("Total Trades", 0)),
            winning_trades=int(stats.get("Total Winning Trades", 0)),
            losing_trades=int(stats.get("Total Losing Trades", 0)),
            avg_win=float(stats.get("Avg Winning Trade [%]", 0)),
            avg_loss=float(stats.get("Avg Losing Trade [%]", 0)),
            best_trade=float(stats.get("Best Trade [%]", 0)),
            worst_trade=float(stats.get("Worst Trade [%]", 0)),
            avg_trade_duration=float(stats.get("Avg Winning Trade Duration", 0)),
            is_profitable=float(stats.get("Total Return", 0)) > 0,
            metrics=stats.to_dict() if hasattr(stats, "to_dict") else {},
        )

    def _run_simple_backtest(
        self,
        df: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        strategy_name: str,
        position_size: float,
    ) -> BacktestResult:
        """Run simple backtest without vectorbt."""
        price = df["close"].values
        entries = entries.values if hasattr(entries, 'values') else entries
        exits = exits.values if hasattr(exits, 'values') else exits

        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []

        for i in range(len(price)):
            if entries[i] and position == 0:
                # Enter position
                position = (capital * position_size) / price[i]
                entry_price = price[i]
                capital -= position * price[i] * (1 + self.commission + self.slippage)

            elif exits[i] and position > 0:
                # Exit position
                exit_value = position * price[i] * (1 - self.commission - self.slippage)
                pnl = exit_value - (position * entry_price)
                pnl_pct = (price[i] - entry_price) / entry_price * 100
                trades.append(pnl_pct)
                capital += exit_value
                position = 0

        # Close any remaining position
        if position > 0:
            exit_value = position * price[-1] * (1 - self.commission - self.slippage)
            pnl_pct = (price[-1] - entry_price) / entry_price * 100
            trades.append(pnl_pct)
            capital += exit_value

        # Calculate metrics
        trades = np.array(trades) if trades else np.array([0])
        winning_trades = trades[trades > 0]
        losing_trades = trades[trades < 0]

        total_return = capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        # Sharpe ratio (simplified)
        if len(trades) > 1 and trades.std() > 0:
            sharpe = (trades.mean() / trades.std()) * np.sqrt(252)
        else:
            sharpe = 0

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=df.index[0] if hasattr(df.index[0], 'to_pydatetime') else datetime.now(),
            end_date=df.index[-1] if hasattr(df.index[-1], 'to_pydatetime') else datetime.now(),
            initial_capital=self.initial_capital,
            final_value=capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sharpe * 0.8,  # Approximation
            max_drawdown=abs(trades.min()) if len(trades) > 0 else 0,
            win_rate=(len(winning_trades) / len(trades) * 100) if len(trades) > 0 else 0,
            profit_factor=abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 and losing_trades.sum() != 0 else 0,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=winning_trades.mean() if len(winning_trades) > 0 else 0,
            avg_loss=losing_trades.mean() if len(losing_trades) > 0 else 0,
            best_trade=trades.max() if len(trades) > 0 else 0,
            worst_trade=trades.min() if len(trades) > 0 else 0,
            avg_trade_duration=0,
            is_profitable=total_return > 0,
            metrics={},
        )

    def run_rsi_strategy(
        self,
        df: pd.DataFrame,
        rsi_period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
    ) -> BacktestResult:
        """
        Run a simple RSI strategy backtest.

        Args:
            df: OHLCV DataFrame
            rsi_period: RSI calculation period
            oversold: RSI oversold threshold
            overbought: RSI overbought threshold

        Returns:
            BacktestResult
        """
        # Calculate RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        entries = rsi < oversold
        exits = rsi > overbought

        return self.run_backtest(df, entries, exits, f"RSI({rsi_period})", 1.0)

    def run_ma_crossover(
        self,
        df: pd.DataFrame,
        fast_period: int = 10,
        slow_period: int = 30,
    ) -> BacktestResult:
        """
        Run a moving average crossover strategy.

        Args:
            df: OHLCV DataFrame
            fast_period: Fast MA period
            slow_period: Slow MA period

        Returns:
            BacktestResult
        """
        fast_ma = df["close"].rolling(fast_period).mean()
        slow_ma = df["close"].rolling(slow_period).mean()

        # Crossover signals
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        return self.run_backtest(df, entries, exits, f"MA({fast_period}/{slow_period})", 1.0)

    def format_result(self, result: BacktestResult) -> str:
        """Format backtest result as readable string."""
        return f"""
=== Backtest Results: {result.strategy_name} ===
Period: {result.start_date} to {result.end_date}

Performance:
  Initial Capital: ${result.initial_capital:,.2f}
  Final Value: ${result.final_value:,.2f}
  Total Return: ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)

Risk Metrics:
  Sharpe Ratio: {result.sharpe_ratio:.2f}
  Sortino Ratio: {result.sortino_ratio:.2f}
  Max Drawdown: {result.max_drawdown:.2f}%

Trade Statistics:
  Total Trades: {result.total_trades}
  Win Rate: {result.win_rate:.1f}%
  Profit Factor: {result.profit_factor:.2f}

  Winning Trades: {result.winning_trades}
  Losing Trades: {result.losing_trades}
  Avg Win: {result.avg_win:.2f}%
  Avg Loss: {result.avg_loss:.2f}%

  Best Trade: {result.best_trade:.2f}%
  Worst Trade: {result.worst_trade:.2f}%

Profitable: {'YES' if result.is_profitable else 'NO'}
"""


def create_backtest_engine(
    initial_capital: float = 10000,
    commission: float = 0.001,
) -> BacktestEngine:
    """Create a BacktestEngine instance."""
    return BacktestEngine(initial_capital=initial_capital, commission=commission)


if __name__ == "__main__":
    print("Testing BacktestEngine...")

    # Create sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1H")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    engine = BacktestEngine(initial_capital=10000)

    # Test RSI strategy
    print("\n1. Testing RSI Strategy...")
    result = engine.run_rsi_strategy(df, rsi_period=14, oversold=30, overbought=70)
    print(engine.format_result(result))

    # Test MA crossover
    print("\n2. Testing MA Crossover Strategy...")
    result = engine.run_ma_crossover(df, fast_period=10, slow_period=30)
    print(engine.format_result(result))
