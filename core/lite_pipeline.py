"""
Lite Pipeline - Uses LiteAnalyzer instead of full LLM-based pipeline

Runs every interval and sends signals to Telegram when actionable.
Integrates with VirtualInvestor for paper trading.

Enhanced with:
- Daily summary scheduler (sends at 00:00 UTC)
- Automatic meta-learner training
- Persistent subscriptions integration

NOTE: This module uses direct imports to avoid pulling in LLM dependencies.
"""

import sys
sys.path.insert(0, "/opt/wallex")

import threading
import queue
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# Direct import of lite_analyzer (doesn't go through core/__init__.py)
from core.lite_analyzer import LiteAnalyzer
from core.feedback_evaluator import get_feedback_evaluator
from settings import TRACKED_TOKENS

logger = logging.getLogger(__name__)


@dataclass
class LiteSignal:
    symbol: str
    signal: str
    strength: str
    confidence: float
    price: float
    change_24h: float
    gru_direction: str
    gru_prob_up: float
    regime: str
    timestamp: datetime


# Import virtual investor directly to avoid import chain issues
def _get_virtual_investor():
    """Lazy load virtual investor to avoid import issues."""
    try:
        from core.virtual_investor import get_virtual_investor
        return get_virtual_investor()
    except Exception as e:
        print(f"Warning: Could not load virtual investor: {e}")
        return None


class LitePipeline:
    """Lightweight analysis pipeline without LLM dependencies."""

    def __init__(self, interval_minutes: int = 60):
        self.interval_minutes = interval_minutes
        self.analyzer = LiteAnalyzer()
        self.results_queue = queue.Queue()
        self._running = False
        self._thread = None

        # Virtual investor for paper trading (lazy loaded)
        self._virtual_investor = None

        # Track last run times
        self.last_runs: Dict[str, datetime] = {}
        self.last_signals: Dict[str, LiteSignal] = {}

        # Trade notifications queue
        self.trade_notifications: queue.Queue = queue.Queue()

        # Feedback evaluator for learning
        try:
            self._evaluator = get_feedback_evaluator()
            self._learning_enabled = True
        except Exception as e:
            print(f"Warning: Learning system disabled: {e}")
            self._learning_enabled = False

        # Daily summary tracking
        self._last_daily_summary = None
        self._daily_summary_hour = 0  # Send at 00:00 UTC

    @property
    def virtual_investor(self):
        """Lazy-load virtual investor."""
        if self._virtual_investor is None:
            self._virtual_investor = _get_virtual_investor()
        return self._virtual_investor

    def run_once(self) -> List[LiteSignal]:
        """Run analysis on all tracked tokens."""
        signals = []
        prices = {}  # For checking exit conditions

        for token, config in TRACKED_TOKENS.items():
            symbol = config["symbol"]
            try:
                result = self.analyzer.analyze(symbol)

                if "error" in result:
                    continue

                self.last_runs[token] = datetime.now()
                prices[symbol] = result["price"]

                signal = LiteSignal(
                    symbol=symbol,
                    signal=result["signal"],
                    strength=result["strength"],
                    confidence=result["confidence"],
                    price=result["price"],
                    change_24h=result["change_24h"],
                    gru_direction=result.get("ensemble", {}).get("direction", "NEUTRAL"),
                    gru_prob_up=result.get("ensemble", {}).get("probability_up", 0.5),
                    regime=result["regime"],
                    timestamp=datetime.now(),
                )

                self.last_signals[token] = signal
                signals.append(signal)

                # Process through virtual investor if available
                if self.virtual_investor:
                    trade_msg = self.virtual_investor.process_signal(
                        {
                            "symbol": symbol,
                            "signal": result["signal"],
                            "confidence": result["confidence"],
                        },
                        result["price"]
                    )

                    if trade_msg:
                        self.trade_notifications.put(trade_msg)

            except Exception as e:
                print(f"Error analyzing {token}: {e}")

        # Check exit conditions for all open positions
        if self.virtual_investor:
            exit_msgs = self.virtual_investor.check_all_positions(prices)
            for msg in exit_msgs:
                self.trade_notifications.put(msg)

        return signals

    def get_actionable(self, signals: List[LiteSignal]) -> List[LiteSignal]:
        """Filter for actionable signals (BUY/SELL with good confidence)."""
        return [s for s in signals if s.signal != "HOLD" and s.confidence >= 55]

    def start(self):
        """Start the pipeline loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"Lite Pipeline started (interval: {self.interval_minutes} min)")

    def stop(self):
        """Stop the pipeline."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _should_send_daily_summary(self) -> bool:
        """Check if it's time to send daily summary."""
        now = datetime.utcnow()

        # Check if it's the right hour
        if now.hour != self._daily_summary_hour:
            return False

        # Check if we already sent today
        if self._last_daily_summary:
            if self._last_daily_summary.date() == now.date():
                return False

        return True

    def _get_daily_summary_data(self) -> Dict[str, Any]:
        """Get data for daily summary."""
        summary = {}

        # Virtual investor performance
        if self.virtual_investor:
            try:
                perf = self.virtual_investor.get_performance()
                summary.update(perf)
            except:
                pass

        # Learning system stats
        if self._learning_enabled:
            try:
                learning = self._evaluator.get_learning_summary(days=1)
                summary["learning"] = learning
            except:
                pass

        # Ensemble stats
        try:
            ensemble_stats = self.analyzer.ensemble.get_stats()
            summary["ensemble"] = ensemble_stats
        except:
            pass

        return summary

    def _format_daily_summary(self, data: Dict[str, Any]) -> str:
        """Format daily summary message."""
        now = datetime.utcnow()

        msg = f"""📅 *Daily Summary* - {now.strftime('%Y-%m-%d')}

*Virtual Investor*
Balance: ${data.get('current_balance', 10000):,.2f}
P&L Today: ${data.get('total_pnl', 0):+,.2f} ({data.get('total_return_pct', 0):+.1f}%)
Open Positions: {data.get('open_positions', 0)}
Total Trades: {data.get('total_trades', 0)}
Win Rate: {data.get('win_rate', 0):.1f}%
"""

        # Learning stats
        if 'learning' in data:
            learning = data['learning']
            overall = learning.get('overall', {})
            msg += f"""
*Learning System*
Predictions (24h): {overall.get('total_predictions', 0)}
Accuracy: {overall.get('accuracy_pct', 0):.1f}%
"""

        # Ensemble stats
        if 'ensemble' in data:
            ensemble = data['ensemble']
            if ensemble.get('meta_learner_trained'):
                msg += f"""
*Meta-Learner*
Status: Active ✅
Samples: {ensemble.get('meta_learner_samples', 0)}
"""
            else:
                buffer = ensemble.get('prediction_buffer', {})
                msg += f"""
*Meta-Learner*
Status: Training ({buffer.get('evaluated', 0)}/50)
"""

        msg += "\n_Automated report from CryptoAlpha_"

        return msg

    def _run_loop(self):
        """Main loop."""
        while self._running:
            try:
                signals = self.run_once()
                actionable = self.get_actionable(signals)

                for sig in actionable:
                    self.results_queue.put(sig)

                # Log virtual investor status if available
                if self.virtual_investor:
                    try:
                        perf = self.virtual_investor.get_performance()
                        print(f"[{datetime.now().strftime('%H:%M')}] "
                              f"Lite: {len(signals)} tokens, {len(actionable)} actionable | "
                              f"Virtual: ${perf['current_balance']:,.0f} ({perf['total_return_pct']:+.1f}%)")
                    except:
                        print(f"[{datetime.now().strftime('%H:%M')}] "
                              f"Lite: {len(signals)} tokens, {len(actionable)} actionable")
                else:
                    print(f"[{datetime.now().strftime('%H:%M')}] "
                          f"Lite: {len(signals)} tokens, {len(actionable)} actionable")

                # Run feedback evaluation to update learning metrics
                if self._learning_enabled:
                    try:
                        eval_result = self._evaluator.evaluate_pending()
                        if eval_result["evaluated"] > 0:
                            print(f"[Learning] Evaluated {eval_result['evaluated']} predictions: "
                                  f"{eval_result['accuracy_pct']:.1f}% accuracy")
                    except Exception as e:
                        pass  # Don't fail pipeline if evaluation fails

                # Check if we should send daily summary
                if self._should_send_daily_summary():
                    try:
                        summary_data = self._get_daily_summary_data()
                        summary_msg = self._format_daily_summary(summary_data)
                        self.trade_notifications.put(("DAILY_SUMMARY", summary_msg))
                        self._last_daily_summary = datetime.utcnow()
                        print(f"[{datetime.now().strftime('%H:%M')}] Daily summary queued")
                    except Exception as e:
                        logger.error(f"Failed to generate daily summary: {e}")

            except Exception as e:
                print(f"Lite pipeline error: {e}")

            # Sleep until next aligned time (1 min after the hour)
            now = datetime.now()
            minutes_until_next_hour = 60 - now.minute
            if minutes_until_next_hour == 60:
                minutes_until_next_hour = 0
            # Wait until :01 of next hour
            seconds_to_wait = (minutes_until_next_hour * 60) + (60 - now.second) + 60  # +60 for :01
            if seconds_to_wait > 3660:  # More than 61 min means we just passed :01
                seconds_to_wait = seconds_to_wait - 3600
            time.sleep(max(60, seconds_to_wait))  # Min 1 min between runs

    def get_learning_status(self) -> Dict[str, Any]:
        """Get learning system status."""
        if not self._learning_enabled:
            return {"enabled": False, "message": "Learning system disabled"}

        try:
            return self._evaluator.get_learning_summary(days=7)
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status with real data."""
        jobs = []
        for token in list(TRACKED_TOKENS.keys())[:10]:
            last_run = self.last_runs.get(token)
            last_signal = self.last_signals.get(token)

            jobs.append({
                "symbol": f"{token}/USDT",
                "priority": 1,
                "last_run": last_run.strftime("%H:%M") if last_run else None,
                "signal": last_signal.signal if last_signal else None,
            })

        status = {
            "running": self._running,
            "tokens_tracked": len(TRACKED_TOKENS),
            "interval_minutes": self.interval_minutes,
            "jobs": jobs,
            "last_daily_summary": self._last_daily_summary.isoformat() if self._last_daily_summary else None,
        }

        # Include virtual investor stats if available
        if self.virtual_investor:
            try:
                perf = self.virtual_investor.get_performance()
                status["virtual_investor"] = {
                    "balance": perf["current_balance"],
                    "pnl": perf["total_pnl"],
                    "return_pct": perf["total_return_pct"],
                    "open_positions": perf["open_positions"],
                    "total_trades": perf["total_trades"],
                    "win_rate": perf["win_rate"],
                }
            except:
                pass

        return status

    def get_trade_notifications(self) -> List[Any]:
        """Get pending trade notifications (for Telegram)."""
        notifications = []
        while not self.trade_notifications.empty():
            try:
                notifications.append(self.trade_notifications.get_nowait())
            except queue.Empty:
                break
        return notifications


# Singleton
_pipeline = None


def get_lite_pipeline() -> LitePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LitePipeline()
    return _pipeline
