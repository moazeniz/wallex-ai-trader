"""
Alert System

Handles sending alerts to subscribed users.
Enhanced with:
- SQLite persistence for subscriptions
- Auto-load on startup
- Daily summary scheduling
"""

import asyncio
import sqlite3
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import json
import logging

from telegram import Bot
from telegram.error import TelegramError

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Try to import these, but don't fail if not available
try:
    from agents.state import TradingSignal, SignalType, SignalStrength
    from bot.formatters import format_signal_message
    from core.signal_generator import SignalResult
except ImportError:
    TradingSignal = None
    format_signal_message = None
    SignalResult = None

DB_PATH = "/opt/wallex/data/subscriptions.db"


@dataclass
class AlertSubscription:
    """User alert subscription."""
    user_id: int
    chat_id: int
    enabled: bool
    min_confidence: float
    signal_types: List[str]
    min_strength: str
    symbols: Optional[List[str]]  # None = all symbols


class AlertManager:
    """
    Alert management system with persistence.

    Handles sending trading signals to subscribed users.
    Subscriptions are saved to SQLite and restored on restart.
    """

    def __init__(self, bot_token: str, db_path: str = DB_PATH):
        """
        Initialize the alert manager.

        Args:
            bot_token: Telegram bot token
            db_path: Path to SQLite database for persistence
        """
        self.bot_token = bot_token
        self.db_path = db_path
        self.bot: Optional[Bot] = None
        self.subscriptions: Dict[int, AlertSubscription] = {}
        self._initialized = False

        # Initialize database and load existing subscriptions
        self._init_db()
        self._load_subscriptions()

    def _init_db(self):
        """Initialize the subscriptions database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                user_id INTEGER PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                enabled INTEGER DEFAULT 1,
                min_confidence REAL DEFAULT 0.65,
                signal_types TEXT DEFAULT '["BUY", "SELL"]',
                min_strength TEXT DEFAULT 'WEAK',
                symbols TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"[Alerts] Database initialized at {self.db_path}")

    def _load_subscriptions(self):
        """Load subscriptions from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM subscriptions")
            rows = cursor.fetchall()

            for row in rows:
                user_id, chat_id, enabled, min_conf, signal_types_json, min_strength, symbols_json, _, _ = row

                signal_types = json.loads(signal_types_json) if signal_types_json else ["BUY", "SELL"]
                symbols = json.loads(symbols_json) if symbols_json else None

                self.subscriptions[user_id] = AlertSubscription(
                    user_id=user_id,
                    chat_id=chat_id,
                    enabled=bool(enabled),
                    min_confidence=min_conf,
                    signal_types=signal_types,
                    min_strength=min_strength,
                    symbols=symbols,
                )

            conn.close()
            logger.info(f"[Alerts] Loaded {len(self.subscriptions)} subscriptions from database")

        except Exception as e:
            logger.error(f"[Alerts] Failed to load subscriptions: {e}")

    def _save_subscription(self, sub: AlertSubscription):
        """Save a subscription to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO subscriptions
                (user_id, chat_id, enabled, min_confidence, signal_types, min_strength, symbols, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sub.user_id,
                sub.chat_id,
                1 if sub.enabled else 0,
                sub.min_confidence,
                json.dumps(sub.signal_types),
                sub.min_strength,
                json.dumps(sub.symbols) if sub.symbols else None,
                datetime.now().isoformat(),
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"[Alerts] Failed to save subscription: {e}")

    def _delete_subscription(self, user_id: int):
        """Delete a subscription from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM subscriptions WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[Alerts] Failed to delete subscription: {e}")

    async def initialize(self) -> None:
        """Initialize the Telegram bot."""
        if not self._initialized:
            self.bot = Bot(token=self.bot_token)
            self._initialized = True

    def subscribe(
        self,
        user_id: int,
        chat_id: int,
        min_confidence: float = 0.65,
        signal_types: Optional[List[str]] = None,
        min_strength: str = "WEAK",
        symbols: Optional[List[str]] = None,
    ) -> None:
        """
        Subscribe a user to alerts (persisted to database).
        """
        sub = AlertSubscription(
            user_id=user_id,
            chat_id=chat_id,
            enabled=True,
            min_confidence=min_confidence,
            signal_types=signal_types or ["BUY", "SELL"],
            min_strength=min_strength,
            symbols=symbols,
        )

        self.subscriptions[user_id] = sub
        self._save_subscription(sub)
        logger.info(f"[Alerts] User {user_id} subscribed (chat_id: {chat_id})")

    def unsubscribe(self, user_id: int) -> None:
        """Unsubscribe a user from alerts."""
        if user_id in self.subscriptions:
            del self.subscriptions[user_id]
            self._delete_subscription(user_id)
            logger.info(f"[Alerts] User {user_id} unsubscribed")

    def toggle_alerts(self, user_id: int, enabled: bool) -> None:
        """Toggle alerts for a user."""
        if user_id in self.subscriptions:
            self.subscriptions[user_id].enabled = enabled
            self._save_subscription(self.subscriptions[user_id])

    def update_filters(
        self,
        user_id: int,
        min_confidence: Optional[float] = None,
        signal_types: Optional[List[str]] = None,
        min_strength: Optional[str] = None,
        symbols: Optional[List[str]] = None,
    ) -> None:
        """Update alert filters for a user."""
        if user_id not in self.subscriptions:
            return

        sub = self.subscriptions[user_id]

        if min_confidence is not None:
            sub.min_confidence = min_confidence
        if signal_types is not None:
            sub.signal_types = signal_types
        if min_strength is not None:
            sub.min_strength = min_strength
        if symbols is not None:
            sub.symbols = symbols

        self._save_subscription(sub)

    def _should_send_alert(
        self,
        subscription: AlertSubscription,
        signal,
    ) -> bool:
        """Check if alert should be sent to this subscription."""
        if not subscription.enabled:
            return False

        # Check confidence
        if hasattr(signal, 'confidence') and signal.confidence < subscription.min_confidence:
            return False

        # Check signal type
        if hasattr(signal, 'signal_type'):
            signal_type = signal.signal_type.value.upper() if hasattr(signal.signal_type, 'value') else str(signal.signal_type).upper()
            if signal_type not in subscription.signal_types:
                return False

        # Check strength
        if hasattr(signal, 'strength'):
            strength_order = {"WEAK": 0, "MODERATE": 1, "STRONG": 2}
            min_strength_val = strength_order.get(subscription.min_strength.upper(), 0)
            signal_strength = signal.strength.value.upper() if hasattr(signal.strength, 'value') else str(signal.strength).upper()
            signal_strength_val = strength_order.get(signal_strength, 0)
            if signal_strength_val < min_strength_val:
                return False

        # Check symbols
        if subscription.symbols and hasattr(signal, 'symbol'):
            if signal.symbol not in subscription.symbols:
                return False

        return True

    async def send_signal_alert(
        self,
        signal,
    ) -> int:
        """Send a signal alert to all eligible subscribers."""
        if not self._initialized:
            await self.initialize()

        if format_signal_message:
            message = format_signal_message(signal)
        else:
            message = f"*Signal Alert*\n\n{signal}"

        sent_count = 0

        for user_id, subscription in self.subscriptions.items():
            if self._should_send_alert(subscription, signal):
                try:
                    await self.bot.send_message(
                        chat_id=subscription.chat_id,
                        text=message,
                        parse_mode="Markdown",
                    )
                    sent_count += 1
                except TelegramError as e:
                    logger.error(f"Failed to send alert to {user_id}: {e}")

        return sent_count

    async def send_result_alert(self, result) -> int:
        """Send a signal result alert."""
        if not hasattr(result, 'signal') or not result.signal:
            return 0
        if hasattr(result, 'approved') and not result.approved:
            return 0

        return await self.send_signal_alert(result.signal)

    async def broadcast_message(
        self,
        message: str,
        parse_mode: str = "Markdown",
    ) -> int:
        """Broadcast a message to all subscribers."""
        if not self._initialized:
            await self.initialize()

        sent_count = 0

        for user_id, subscription in self.subscriptions.items():
            if subscription.enabled:
                try:
                    await self.bot.send_message(
                        chat_id=subscription.chat_id,
                        text=message,
                        parse_mode=parse_mode,
                    )
                    sent_count += 1
                except TelegramError as e:
                    logger.error(f"Failed to broadcast to {user_id}: {e}")

        return sent_count

    async def send_daily_summary(
        self,
        performance: Dict[str, Any],
    ) -> int:
        """Send daily performance summary."""
        try:
            from bot.formatters import format_performance_summary
            perf_text = format_performance_summary(performance)
        except:
            # Fallback formatting
            perf_text = f"""Balance: ${performance.get('current_balance', 0):,.2f}
P&L: ${performance.get('total_pnl', 0):+,.2f} ({performance.get('total_return_pct', 0):+.1f}%)
Trades: {performance.get('total_trades', 0)}
Win Rate: {performance.get('win_rate', 0):.1f}%"""

        message = f"📅 *Daily Summary*\n\n{perf_text}"

        return await self.broadcast_message(message)

    def get_subscriber_count(self) -> int:
        """Get number of active subscribers."""
        return sum(1 for s in self.subscriptions.values() if s.enabled)

    def get_subscription(self, user_id: int) -> Optional[AlertSubscription]:
        """Get subscription for a user."""
        return self.subscriptions.get(user_id)

    def get_all_chat_ids(self) -> Set[int]:
        """Get all chat IDs for notifications."""
        return {sub.chat_id for sub in self.subscriptions.values() if sub.enabled}


def create_alert_manager(bot_token: str) -> AlertManager:
    """Create an AlertManager instance."""
    return AlertManager(bot_token=bot_token)
