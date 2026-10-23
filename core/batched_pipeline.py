"""
Batched Pipeline

Runs the batched analyzer on a schedule.
Uses only 1 Gemini request per cycle instead of 64.

Target: ~100 requests/day (96 cycles × 1 request)
"""

import sys
import os
import time
import threading
from datetime import datetime
from typing import Optional, List, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.batched_analyzer import BatchedAnalyzer, TokenSignal, get_batched_analyzer
from core.virtual_investor import get_virtual_investor
from core.prediction_tracker import get_prediction_tracker


class BatchedPipeline:
    """
    Pipeline that runs batched analysis every 15 minutes.

    Only makes 1 Gemini API call per cycle for all tokens.
    """

    def __init__(
        self,
        interval_minutes: int = 15,
        on_signal: Optional[Callable[[TokenSignal], None]] = None,
    ):
        self.interval_minutes = interval_minutes
        self.on_signal = on_signal

        # Components
        self.analyzer = None  # Lazy load
        self.investor = None  # Lazy load
        self.tracker = None   # Lazy load

        # Control
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self.last_run: Optional[datetime] = None
        self.total_cycles = 0
        self.total_signals = 0

    def _get_analyzer(self) -> BatchedAnalyzer:
        """Lazy load analyzer."""
        if self.analyzer is None:
            self.analyzer = get_batched_analyzer()
        return self.analyzer

    def _get_investor(self):
        """Lazy load virtual investor."""
        if self.investor is None:
            self.investor = get_virtual_investor()
        return self.investor

    def _get_tracker(self):
        """Lazy load prediction tracker."""
        if self.tracker is None:
            self.tracker = get_prediction_tracker()
        return self.tracker

    def _run_cycle(self):
        """Run one analysis cycle."""
        cycle_start = datetime.now()
        print(f"\n[{cycle_start.strftime('%H:%M:%S')}] Starting batched analysis cycle...")

        try:
            # Run batched analysis (1 Gemini call for all tokens)
            analyzer = self._get_analyzer()
            signals = analyzer.analyze_all()

            # Process signals
            actionable = 0
            for sig in signals:
                # Record prediction for learning system
                try:
                    tracker = self._get_tracker()
                    direction = "UP" if sig.signal == "BUY" else ("DOWN" if sig.signal == "SELL" else "NEUTRAL")
                    tracker.record_prediction(
                        symbol=sig.symbol,
                        predicted_direction=direction,
                        confidence=sig.confidence,
                        price=sig.entry_price,
                        scores={"gemini": sig.confidence, "reasoning": sig.reasoning},
                        timeframe="4h"
                    )
                except Exception as e:
                    print(f"  Warning: Could not record prediction: {e}")

                # Process with virtual investor
                if sig.signal in ['BUY', 'SELL'] and sig.confidence >= 65:
                    actionable += 1
                    try:
                        investor = self._get_investor()
                        msg = investor.process_signal(
                            {
                                "symbol": sig.symbol,
                                "signal": sig.signal,
                                "confidence": sig.confidence,
                            },
                            sig.entry_price
                        )
                        if msg:
                            print(f"  {msg}")

                        # Callback
                        if self.on_signal:
                            self.on_signal(sig)

                    except Exception as e:
                        print(f"  Error processing {sig.symbol}: {e}")

            # Update stats
            self.last_run = cycle_start
            self.total_cycles += 1
            self.total_signals += actionable

            # Summary
            cycle_time = (datetime.now() - cycle_start).total_seconds()
            investor = self._get_investor()
            balance = investor.get_balance() if investor else 10000

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle complete: "
                  f"{len(signals)} analyzed, {actionable} actionable | "
                  f"Balance: ${balance:,.2f} | Time: {cycle_time:.1f}s")

        except Exception as e:
            print(f"[ERROR] Cycle failed: {e}")
            import traceback
            traceback.print_exc()

    def _run_loop(self):
        """Background loop."""
        print(f"[BatchedPipeline] Started (interval: {self.interval_minutes} min)")
        print(f"[BatchedPipeline] API calls: 1 per cycle = ~{96 // (60 // self.interval_minutes * 24)} per day")

        while self._running:
            try:
                self._run_cycle()
            except Exception as e:
                print(f"[BatchedPipeline] Error in cycle: {e}")

            # Wait for next cycle
            if self._running:
                sleep_seconds = self.interval_minutes * 60
                print(f"[BatchedPipeline] Next cycle in {self.interval_minutes} minutes...")

                # Sleep in small chunks to allow quick shutdown
                for _ in range(sleep_seconds):
                    if not self._running:
                        break
                    time.sleep(1)

    def start(self):
        """Start the pipeline."""
        if self._running:
            print("[BatchedPipeline] Already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[BatchedPipeline] Started")

    def stop(self):
        """Stop the pipeline."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[BatchedPipeline] Stopped")

    def get_status(self) -> dict:
        """Get pipeline status."""
        return {
            "running": self._running,
            "interval_minutes": self.interval_minutes,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "total_cycles": self.total_cycles,
            "total_signals": self.total_signals,
            "estimated_daily_requests": 96,  # 1 per 15 min
        }


# Singleton
_batched_pipeline = None

def get_batched_pipeline() -> BatchedPipeline:
    global _batched_pipeline
    if _batched_pipeline is None:
        _batched_pipeline = BatchedPipeline()
    return _batched_pipeline


if __name__ == "__main__":
    print("=" * 60)
    print("BATCHED PIPELINE")
    print("Efficient Gemini usage: 1 request per cycle")
    print("=" * 60)

    pipeline = BatchedPipeline()

    try:
        pipeline.start()

        # Keep running
        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        print("\nShutting down...")
        pipeline.stop()
