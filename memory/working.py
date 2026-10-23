"""
Working Memory

Short-term memory for current session state:
- Active analyses
- Open positions
- Recent signals
- Current market context
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import deque
import threading
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import TradingSignal, SignalType


@dataclass
class MarketContext:
    """Current market context."""
    symbol: str
    latest_price: float
    regime: str
    volatility: str
    trend: str
    last_update: datetime
    anomaly_score: float = 0.0
    whale_activity: str = "neutral"
    exchange_flow: str = "neutral"


@dataclass
class OpenPosition:
    """An open position being tracked."""
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    signal_id: Optional[int] = None
    current_price: Optional[float] = None
    unrealized_pnl_pct: float = 0.0


@dataclass
class RecentAlert:
    """A recent alert sent."""
    symbol: str
    signal_type: str
    timestamp: datetime
    confidence: float
    delivered: bool = True


class WorkingMemory:
    """
    Working memory for current session state.

    Maintains short-term context including:
    - Market contexts for tracked symbols
    - Open positions
    - Recent alerts and signals
    - Cooldowns and rate limits
    """

    def __init__(
        self,
        max_recent_signals: int = 50,
        max_recent_alerts: int = 100,
        cooldown_hours: int = 4,
    ):
        """
        Initialize working memory.

        Args:
            max_recent_signals: Max signals to keep in memory
            max_recent_alerts: Max alerts to keep in memory
            cooldown_hours: Hours between signals on same asset
        """
        self.max_recent_signals = max_recent_signals
        self.max_recent_alerts = max_recent_alerts
        self.cooldown_hours = cooldown_hours

        # Market contexts by symbol
        self._contexts: Dict[str, MarketContext] = {}

        # Open positions by symbol
        self._positions: Dict[str, OpenPosition] = {}

        # Recent signals (FIFO queue)
        self._recent_signals: deque = deque(maxlen=max_recent_signals)

        # Recent alerts
        self._recent_alerts: deque = deque(maxlen=max_recent_alerts)

        # Signal cooldowns (symbol -> last signal time)
        self._cooldowns: Dict[str, datetime] = {}

        # Active analysis locks
        self._analysis_locks: Set[str] = set()

        # Thread lock for concurrent access
        self._lock = threading.Lock()

        # Session start time
        self._session_start = datetime.now()

        # Counters
        self._signals_generated = 0
        self._signals_approved = 0
        self._signals_rejected = 0
        self._alerts_sent = 0

    # =========================================================================
    # Market Context
    # =========================================================================

    def update_context(
        self,
        symbol: str,
        price: float,
        regime: str = "unknown",
        volatility: str = "normal",
        trend: str = "neutral",
        anomaly_score: float = 0.0,
        whale_activity: str = "neutral",
        exchange_flow: str = "neutral",
    ) -> None:
        """Update market context for a symbol."""
        with self._lock:
            self._contexts[symbol] = MarketContext(
                symbol=symbol,
                latest_price=price,
                regime=regime,
                volatility=volatility,
                trend=trend,
                last_update=datetime.now(),
                anomaly_score=anomaly_score,
                whale_activity=whale_activity,
                exchange_flow=exchange_flow,
            )

    def get_context(self, symbol: str) -> Optional[MarketContext]:
        """Get market context for a symbol."""
        with self._lock:
            return self._contexts.get(symbol)

    def get_all_contexts(self) -> Dict[str, MarketContext]:
        """Get all market contexts."""
        with self._lock:
            return dict(self._contexts)

    # =========================================================================
    # Open Positions
    # =========================================================================

    def add_position(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        signal_id: Optional[int] = None,
    ) -> None:
        """Add an open position."""
        with self._lock:
            self._positions[symbol] = OpenPosition(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.now(),
                signal_id=signal_id,
            )

    def update_position_price(self, symbol: str, current_price: float) -> None:
        """Update current price for a position."""
        with self._lock:
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.current_price = current_price
                pos.unrealized_pnl_pct = (
                    (current_price - pos.entry_price) / pos.entry_price * 100
                )

    def close_position(self, symbol: str) -> Optional[OpenPosition]:
        """Close and remove a position."""
        with self._lock:
            return self._positions.pop(symbol, None)

    def get_position(self, symbol: str) -> Optional[OpenPosition]:
        """Get an open position."""
        with self._lock:
            return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, OpenPosition]:
        """Get all open positions."""
        with self._lock:
            return dict(self._positions)

    def get_open_position_count(self) -> int:
        """Get number of open positions."""
        with self._lock:
            return len(self._positions)

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if stop loss is hit."""
        with self._lock:
            if symbol not in self._positions:
                return False
            pos = self._positions[symbol]
            return current_price <= pos.stop_loss

    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """Check if take profit is hit."""
        with self._lock:
            if symbol not in self._positions:
                return False
            pos = self._positions[symbol]
            return current_price >= pos.take_profit

    # =========================================================================
    # Recent Signals
    # =========================================================================

    def add_signal(self, signal: TradingSignal, approved: bool) -> None:
        """Add a signal to recent memory."""
        with self._lock:
            self._recent_signals.append({
                "signal": signal,
                "approved": approved,
                "timestamp": datetime.now(),
            })

            self._signals_generated += 1
            if approved:
                self._signals_approved += 1
            else:
                self._signals_rejected += 1

    def get_recent_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent signals."""
        with self._lock:
            signals = list(self._recent_signals)

            if symbol:
                signals = [s for s in signals if s["signal"].symbol == symbol]

            return signals[-limit:]

    # =========================================================================
    # Alerts
    # =========================================================================

    def add_alert(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        delivered: bool = True,
    ) -> None:
        """Record an alert."""
        with self._lock:
            self._recent_alerts.append(RecentAlert(
                symbol=symbol,
                signal_type=signal_type,
                timestamp=datetime.now(),
                confidence=confidence,
                delivered=delivered,
            ))

            if delivered:
                self._alerts_sent += 1

    def get_recent_alerts(self, limit: int = 20) -> List[RecentAlert]:
        """Get recent alerts."""
        with self._lock:
            return list(self._recent_alerts)[-limit:]

    # =========================================================================
    # Cooldowns
    # =========================================================================

    def set_cooldown(self, symbol: str) -> None:
        """Set cooldown for a symbol."""
        with self._lock:
            self._cooldowns[symbol] = datetime.now()

    def check_cooldown(self, symbol: str) -> bool:
        """
        Check if symbol is in cooldown.

        Returns:
            True if in cooldown (should NOT signal), False if clear
        """
        with self._lock:
            if symbol not in self._cooldowns:
                return False

            last_signal = self._cooldowns[symbol]
            cooldown_end = last_signal + timedelta(hours=self.cooldown_hours)

            return datetime.now() < cooldown_end

    def get_cooldown_remaining(self, symbol: str) -> Optional[timedelta]:
        """Get remaining cooldown time."""
        with self._lock:
            if symbol not in self._cooldowns:
                return None

            last_signal = self._cooldowns[symbol]
            cooldown_end = last_signal + timedelta(hours=self.cooldown_hours)
            remaining = cooldown_end - datetime.now()

            return remaining if remaining.total_seconds() > 0 else None

    def clear_cooldown(self, symbol: str) -> None:
        """Clear cooldown for a symbol."""
        with self._lock:
            self._cooldowns.pop(symbol, None)

    # =========================================================================
    # Analysis Locks
    # =========================================================================

    def acquire_analysis_lock(self, symbol: str) -> bool:
        """
        Try to acquire analysis lock for a symbol.

        Returns:
            True if lock acquired, False if already locked
        """
        with self._lock:
            if symbol in self._analysis_locks:
                return False
            self._analysis_locks.add(symbol)
            return True

    def release_analysis_lock(self, symbol: str) -> None:
        """Release analysis lock for a symbol."""
        with self._lock:
            self._analysis_locks.discard(symbol)

    def is_analysis_locked(self, symbol: str) -> bool:
        """Check if symbol is being analyzed."""
        with self._lock:
            return symbol in self._analysis_locks

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        with self._lock:
            session_duration = datetime.now() - self._session_start

            return {
                "session_start": self._session_start.isoformat(),
                "session_duration_hours": session_duration.total_seconds() / 3600,
                "tracked_symbols": len(self._contexts),
                "open_positions": len(self._positions),
                "signals_generated": self._signals_generated,
                "signals_approved": self._signals_approved,
                "signals_rejected": self._signals_rejected,
                "approval_rate": (
                    self._signals_approved / self._signals_generated
                    if self._signals_generated > 0 else 0
                ),
                "alerts_sent": self._alerts_sent,
                "active_cooldowns": sum(
                    1 for s in self._cooldowns
                    if self.check_cooldown(s)
                ),
                "active_analyses": len(self._analysis_locks),
            }

    def get_state_summary(self) -> Dict[str, Any]:
        """Get full state summary."""
        with self._lock:
            return {
                "contexts": {
                    s: asdict(c) for s, c in self._contexts.items()
                },
                "positions": {
                    s: asdict(p) for s, p in self._positions.items()
                },
                "recent_signals": [
                    {
                        "symbol": s["signal"].symbol,
                        "type": s["signal"].signal_type.value,
                        "approved": s["approved"],
                        "timestamp": s["timestamp"].isoformat(),
                    }
                    for s in list(self._recent_signals)[-10:]
                ],
                "stats": self.get_session_stats(),
            }

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup_stale_data(self, max_age_hours: int = 24) -> int:
        """
        Clean up stale data.

        Args:
            max_age_hours: Maximum age for data

        Returns:
            Number of items cleaned
        """
        with self._lock:
            cleaned = 0
            cutoff = datetime.now() - timedelta(hours=max_age_hours)

            # Clean old contexts
            stale_symbols = [
                s for s, c in self._contexts.items()
                if c.last_update < cutoff
            ]
            for s in stale_symbols:
                del self._contexts[s]
                cleaned += 1

            # Clean old cooldowns
            expired_cooldowns = [
                s for s, t in self._cooldowns.items()
                if t < cutoff
            ]
            for s in expired_cooldowns:
                del self._cooldowns[s]
                cleaned += 1

            return cleaned

    def reset(self) -> None:
        """Reset working memory."""
        with self._lock:
            self._contexts.clear()
            self._positions.clear()
            self._recent_signals.clear()
            self._recent_alerts.clear()
            self._cooldowns.clear()
            self._analysis_locks.clear()
            self._session_start = datetime.now()
            self._signals_generated = 0
            self._signals_approved = 0
            self._signals_rejected = 0
            self._alerts_sent = 0


# Global working memory instance
_working_memory: Optional[WorkingMemory] = None


def get_working_memory() -> WorkingMemory:
    """Get the global working memory instance."""
    global _working_memory
    if _working_memory is None:
        _working_memory = WorkingMemory()
    return _working_memory


def reset_working_memory() -> None:
    """Reset the global working memory."""
    global _working_memory
    if _working_memory:
        _working_memory.reset()
    _working_memory = WorkingMemory()


if __name__ == "__main__":
    print("Testing WorkingMemory...")

    memory = WorkingMemory(cooldown_hours=1)

    # Test market context
    print("\n1. Testing market context...")

    memory.update_context(
        symbol="BTC/USDT",
        price=42000.0,
        regime="neutral",
        volatility="normal",
        trend="sideways",
        whale_activity="accumulating",
    )

    ctx = memory.get_context("BTC/USDT")
    if ctx:
        print(f"  BTC context: {ctx.latest_price}, {ctx.regime}, {ctx.whale_activity}")

    # Test positions
    print("\n2. Testing positions...")

    memory.add_position(
        symbol="ETH/USDT",
        entry_price=2200.0,
        stop_loss=2090.0,
        take_profit=2420.0,
    )

    memory.update_position_price("ETH/USDT", 2300.0)

    pos = memory.get_position("ETH/USDT")
    if pos:
        print(f"  ETH position: entry={pos.entry_price}, pnl={pos.unrealized_pnl_pct:.2f}%")

    print(f"  Open positions: {memory.get_open_position_count()}")

    # Test cooldowns
    print("\n3. Testing cooldowns...")

    memory.set_cooldown("BTC/USDT")
    print(f"  BTC in cooldown: {memory.check_cooldown('BTC/USDT')}")
    print(f"  ETH in cooldown: {memory.check_cooldown('ETH/USDT')}")

    remaining = memory.get_cooldown_remaining("BTC/USDT")
    if remaining:
        print(f"  BTC cooldown remaining: {remaining}")

    # Test analysis locks
    print("\n4. Testing analysis locks...")

    acquired = memory.acquire_analysis_lock("SOL/USDT")
    print(f"  SOL lock acquired: {acquired}")
    print(f"  SOL is locked: {memory.is_analysis_locked('SOL/USDT')}")

    memory.release_analysis_lock("SOL/USDT")
    print(f"  SOL is locked after release: {memory.is_analysis_locked('SOL/USDT')}")

    # Test signals
    print("\n5. Testing signals...")

    from agents.state import TradingSignal, SignalType, SignalStrength

    signal = TradingSignal(
        symbol="BTC/USDT",
        signal_type=SignalType.BUY,
        strength=SignalStrength.MODERATE,
        entry_price=42000.0,
        stop_loss=39900.0,
        take_profit=46200.0,
        position_size_pct=1.5,
        confidence=0.72,
        reasoning="Test signal",
        timestamp=datetime.now(),
    )

    memory.add_signal(signal, approved=True)
    recent = memory.get_recent_signals(limit=5)
    print(f"  Recent signals: {len(recent)}")

    # Test alerts
    print("\n6. Testing alerts...")

    memory.add_alert(
        symbol="BTC/USDT",
        signal_type="BUY",
        confidence=0.72,
    )

    alerts = memory.get_recent_alerts()
    print(f"  Recent alerts: {len(alerts)}")

    # Test stats
    print("\n7. Session stats...")

    stats = memory.get_session_stats()
    print(f"  Signals generated: {stats['signals_generated']}")
    print(f"  Signals approved: {stats['signals_approved']}")
    print(f"  Approval rate: {stats['approval_rate']:.0%}")

    print("\nTest complete!")
