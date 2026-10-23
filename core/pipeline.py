"""
Analysis Pipeline

Main pipeline for continuous token analysis and signal generation.
Orchestrates the entire analysis workflow.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import threading
import queue

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.signal_generator import SignalGenerator, SignalResult
from agents.state import TradingSignal, SignalType
from config.tokens import TOKEN_CONFIG


@dataclass
class AnalysisJob:
    """A scheduled analysis job."""
    symbol: str
    token_address: Optional[str]
    chain: str
    timeframe: str
    priority: int = 1
    last_run: Optional[datetime] = None


class AnalysisPipeline:
    """
    Main analysis pipeline for continuous monitoring.

    Schedules and runs analysis jobs for multiple tokens.
    """

    def __init__(
        self,
        exchange: str = "binance",
        chains: Optional[List[str]] = None,
        interval_minutes: int = 15,
        on_signal: Optional[Callable[[SignalResult], None]] = None,
    ):
        """
        Initialize the analysis pipeline.

        Args:
            exchange: Exchange for price data
            chains: Chains for on-chain data
            interval_minutes: Minutes between analysis runs
            on_signal: Callback for new signals
        """
        self.exchange = exchange
        self.chains = chains or ["ethereum"]
        self.interval_minutes = interval_minutes
        self.on_signal = on_signal

        # Initialize signal generator
        self.generator = SignalGenerator(
            exchange=exchange,
            chains=self.chains,
        )

        # Job queue
        self.jobs: List[AnalysisJob] = []
        self.results: queue.Queue = queue.Queue()

        # Control
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add_token(
        self,
        symbol: str,
        token_address: Optional[str] = None,
        chain: str = "ethereum",
        timeframe: str = "1h",
        priority: int = 1,
    ) -> None:
        """
        Add a token to the analysis pipeline.

        Args:
            symbol: Trading pair (e.g., 'AAVE/USDT')
            token_address: Token contract address
            chain: Blockchain for on-chain data
            timeframe: Chart timeframe
            priority: Analysis priority (1=high, 3=low)
        """
        job = AnalysisJob(
            symbol=symbol,
            token_address=token_address,
            chain=chain,
            timeframe=timeframe,
            priority=priority,
        )
        self.jobs.append(job)

    def add_tokens_from_config(self) -> None:
        """Add all tokens from TOKEN_CONFIG."""
        for symbol, config in TOKEN_CONFIG.items():
            self.add_token(
                symbol=f"{symbol}/USDT",
                token_address=config.address,
                chain=config.chain,
                priority=1 if config.priority == "high" else 2,
            )

    def remove_token(self, symbol: str) -> None:
        """Remove a token from the pipeline."""
        self.jobs = [j for j in self.jobs if j.symbol != symbol]

    def run_once(self) -> List[SignalResult]:
        """
        Run analysis for all tokens once.

        Returns:
            List of SignalResults
        """
        results = []

        # Sort by priority
        sorted_jobs = sorted(self.jobs, key=lambda j: j.priority)

        for job in sorted_jobs:
            try:
                result = self.generator.generate_signal(
                    symbol=job.symbol,
                    token_address=job.token_address,
                    chain=job.chain,
                    timeframe=job.timeframe,
                )
                results.append(result)

                # Callback for signals
                if result.approved and result.signal and self.on_signal:
                    self.on_signal(result)

                job.last_run = datetime.now()

            except Exception as e:
                print(f"Error analyzing {job.symbol}: {e}")

        return results

    def start(self) -> None:
        """Start continuous analysis loop."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the analysis loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        """Main analysis loop."""
        while self._running:
            try:
                # Run analysis
                results = self.run_once()

                # Put results in queue
                for result in results:
                    self.results.put(result)

                # Log summary
                actionable = [
                    r for r in results
                    if r.approved and r.signal and r.signal.signal_type != SignalType.HOLD
                ]
                print(f"[{datetime.now().strftime('%H:%M')}] "
                      f"Analyzed {len(results)} tokens, {len(actionable)} actionable signals")

            except Exception as e:
                print(f"Pipeline error: {e}")

            # Wait for next interval
            time.sleep(self.interval_minutes * 60)

    def get_results(self, timeout: float = 0) -> List[SignalResult]:
        """Get results from queue."""
        results = []
        try:
            while True:
                result = self.results.get(block=timeout > 0, timeout=timeout)
                results.append(result)
        except queue.Empty:
            pass
        return results

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "running": self._running,
            "tokens_tracked": len(self.jobs),
            "interval_minutes": self.interval_minutes,
            "jobs": [
                {
                    "symbol": j.symbol,
                    "chain": j.chain,
                    "priority": j.priority,
                    "last_run": j.last_run.isoformat() if j.last_run else None,
                }
                for j in self.jobs
            ],
        }

    def get_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get signal performance."""
        return self.generator.get_performance(days)


class QuickAnalyzer:
    """
    Quick analysis interface for single token analysis.

    Provides a simple way to analyze individual tokens without
    running the full pipeline.
    """

    def __init__(
        self,
        exchange: str = "binance",
        chains: Optional[List[str]] = None,
    ):
        """Initialize quick analyzer."""
        self.generator = SignalGenerator(
            exchange=exchange,
            chains=chains or ["ethereum"],
            save_signals=False,
        )

    def analyze(
        self,
        symbol: str,
        token_address: Optional[str] = None,
        chain: str = "ethereum",
        timeframe: str = "1h",
    ) -> SignalResult:
        """
        Analyze a single token.

        Args:
            symbol: Trading pair
            token_address: Token contract address
            chain: Blockchain
            timeframe: Chart timeframe

        Returns:
            SignalResult
        """
        return self.generator.generate_signal(
            symbol=symbol,
            token_address=token_address,
            chain=chain,
            timeframe=timeframe,
        )

    def quick_scan(
        self,
        symbols: List[str],
        timeframe: str = "1h",
    ) -> List[SignalResult]:
        """
        Quick scan multiple symbols without on-chain data.

        Args:
            symbols: List of trading pairs
            timeframe: Chart timeframe

        Returns:
            List of SignalResults
        """
        results = []
        for symbol in symbols:
            result = self.analyze(symbol, timeframe=timeframe)
            results.append(result)
        return results


def create_pipeline(
    exchange: str = "binance",
    chains: Optional[List[str]] = None,
    interval_minutes: int = 15,
) -> AnalysisPipeline:
    """Create an AnalysisPipeline instance."""
    return AnalysisPipeline(
        exchange=exchange,
        chains=chains,
        interval_minutes=interval_minutes,
    )


if __name__ == "__main__":
    print("Testing AnalysisPipeline...")

    # Create pipeline
    pipeline = AnalysisPipeline(
        exchange="binance",
        chains=["ethereum"],
        interval_minutes=15,
    )

    # Add test tokens
    pipeline.add_token("BTC/USDT", priority=1)
    pipeline.add_token("ETH/USDT", priority=1)

    # Check status
    status = pipeline.get_status()
    print(f"\nPipeline Status:")
    print(f"  Tokens tracked: {status['tokens_tracked']}")

    # Run once
    print("\nRunning single analysis cycle...")
    results = pipeline.run_once()

    print(f"\nResults:")
    for result in results:
        signal_type = result.signal.signal_type.value if result.signal else "NONE"
        approved = "✓" if result.approved else "✗"
        print(f"  {result.symbol}: {signal_type.upper()} {approved} "
              f"(regime: {result.regime}, time: {result.analysis_time:.1f}s)")

    # Test quick analyzer
    print("\n--- Quick Analyzer ---")
    quick = QuickAnalyzer(exchange="binance")
    result = quick.analyze("SOL/USDT")
    print(f"SOL/USDT: {result.signal.signal_type.value if result.signal else 'NO SIGNAL'}")
