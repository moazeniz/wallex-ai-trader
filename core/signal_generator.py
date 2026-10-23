"""
Signal Generator

High-level interface for generating trading signals.
Wraps the multi-agent system for easy signal generation.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import TradingOrchestrator
from agents.state import TradingSignal, SignalType, format_signal_for_telegram
from core.database import SignalDatabase


@dataclass
class SignalResult:
    """Complete signal generation result."""
    signal: Optional[TradingSignal]
    approved: bool
    symbol: str
    analysis_time: float
    technical_summary: str
    onchain_summary: str
    regime: str
    anomaly_score: float
    pattern_probability: float
    errors: List[str]
    signal_id: Optional[int] = None


class SignalGenerator:
    """
    Trading signal generator.

    Provides a simplified interface to the multi-agent system
    for generating and storing trading signals.
    """

    def __init__(
        self,
        exchange: str = "binance",
        chains: Optional[List[str]] = None,
        db_path: str = "data/signals.db",
        save_signals: bool = True,
    ):
        """
        Initialize the signal generator.

        Args:
            exchange: Exchange for price data
            chains: Chains for on-chain data
            db_path: Path to signal database
            save_signals: Whether to save signals to database
        """
        self.exchange = exchange
        self.chains = chains or ["ethereum"]
        self.save_signals = save_signals

        # Initialize components
        self.orchestrator = TradingOrchestrator(
            exchange=exchange,
            chains=self.chains,
        )

        if save_signals:
            self.database = SignalDatabase(db_path)
        else:
            self.database = None

    def generate_signal(
        self,
        symbol: str,
        token_address: Optional[str] = None,
        chain: str = "ethereum",
        timeframe: str = "1h",
    ) -> SignalResult:
        """
        Generate a trading signal for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            token_address: Token contract address for on-chain
            chain: Blockchain for on-chain data
            timeframe: Chart timeframe

        Returns:
            SignalResult with signal and metadata
        """
        start_time = datetime.now()

        # Run analysis
        state = self.orchestrator.analyze(
            symbol=symbol,
            token_address=token_address,
            chain=chain,
            timeframe=timeframe,
        )

        # Calculate total analysis time
        timing = state.get("timing", {})
        analysis_time = sum(timing.values())

        # Extract results
        signal = state.get("aggregated_signal")
        approved = state.get("risk_approved", False)
        errors = state.get("errors", [])

        # Technical summary
        tech = state.get("technical_analysis")
        tech_summary = tech.summary if tech else "No technical analysis"

        # On-chain summary
        onchain = state.get("onchain_analysis")
        onchain_summary = onchain.summary if onchain else "No on-chain analysis"

        # Regime
        regime = state.get("regime_state", {})
        regime_str = regime.get("market_regime", "unknown")

        # Anomaly
        anomaly = state.get("anomaly_scores", {})
        anomaly_score = anomaly.get("combined_anomaly_score", 0)

        # Pattern
        pattern = state.get("pattern_analysis", {})
        pattern_prob = pattern.get("bullish_probability", 0.5)

        # Save to database if approved
        signal_id = None
        if signal and approved and self.save_signals and self.database:
            signal_id = self.database.save_signal(signal)

        return SignalResult(
            signal=signal if approved else None,
            approved=approved,
            symbol=symbol,
            analysis_time=analysis_time,
            technical_summary=tech_summary,
            onchain_summary=onchain_summary,
            regime=regime_str,
            anomaly_score=anomaly_score,
            pattern_probability=pattern_prob,
            errors=errors,
            signal_id=signal_id,
        )

    def generate_signals_batch(
        self,
        tokens: List[Dict[str, Any]],
    ) -> List[SignalResult]:
        """
        Generate signals for multiple tokens.

        Args:
            tokens: List of token configs with 'symbol', 'token_address', 'chain'

        Returns:
            List of SignalResults
        """
        results = []

        for token in tokens:
            result = self.generate_signal(
                symbol=token.get("symbol"),
                token_address=token.get("token_address"),
                chain=token.get("chain", "ethereum"),
                timeframe=token.get("timeframe", "1h"),
            )
            results.append(result)

        return results

    def format_signal(self, result: SignalResult) -> str:
        """
        Format signal result for display.

        Args:
            result: SignalResult to format

        Returns:
            Formatted string
        """
        if not result.signal:
            return f"""
❌ No Signal for {result.symbol}

Analysis Time: {result.analysis_time:.1f}s
Regime: {result.regime}
Anomaly Score: {result.anomaly_score:.2f}

Technical: {result.technical_summary}
On-Chain: {result.onchain_summary}
"""

        return format_signal_for_telegram(result.signal)

    def get_actionable_signals(
        self,
        results: List[SignalResult],
    ) -> List[SignalResult]:
        """Filter for actionable (approved, non-HOLD) signals."""
        return [
            r for r in results
            if r.approved and r.signal and r.signal.signal_type != SignalType.HOLD
        ]

    def get_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get signal performance summary."""
        if self.database:
            return self.database.get_performance_summary(days)
        return {}

    def update_signal_outcome(
        self,
        signal_id: int,
        outcome: str,
        exit_price: float,
        pnl_pct: float,
    ) -> None:
        """Update signal outcome in database."""
        if self.database:
            self.database.update_signal_outcome(
                signal_id, outcome, exit_price, pnl_pct
            )

    def close(self) -> None:
        """Close database connection."""
        if self.database:
            self.database.close()


def create_signal_generator(
    exchange: str = "binance",
    chains: Optional[List[str]] = None,
) -> SignalGenerator:
    """Create a SignalGenerator instance."""
    return SignalGenerator(exchange=exchange, chains=chains)


if __name__ == "__main__":
    print("Testing SignalGenerator...")

    generator = SignalGenerator(
        exchange="binance",
        chains=["ethereum"],
        db_path="data/test_signals.db",
    )

    # Generate signal for BTC
    print("\n1. Generating signal for BTC/USDT...")
    result = generator.generate_signal("BTC/USDT", timeframe="1h")

    print(f"\nResults:")
    print(f"  Symbol: {result.symbol}")
    print(f"  Approved: {result.approved}")
    print(f"  Analysis Time: {result.analysis_time:.1f}s")
    print(f"  Regime: {result.regime}")
    print(f"  Anomaly Score: {result.anomaly_score:.2f}")

    if result.signal:
        print(f"\n  Signal: {result.signal.signal_type.value.upper()}")
        print(f"  Confidence: {result.signal.confidence:.0%}")
        print(f"  Entry: ${result.signal.entry_price:,.2f}")
        print(f"  Signal ID: {result.signal_id}")

    # Format for display
    print("\n2. Formatted Signal:")
    print(generator.format_signal(result))

    generator.close()
