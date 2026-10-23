"""
Telegram Bot

Main Telegram bot for CryptoAlpha trading signals.
Enhanced with:
- Persistent subscriptions
- Daily summary notifications
- Auto-registration for alerts
"""

import asyncio
from typing import Optional, List, Set
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.handlers import BotHandlers, create_handlers
from bot.alerts import AlertManager, create_alert_manager
from settings import load_config


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Global bot instance for handler access
_bot_instance = None


class CryptoAlphaBot:
    """
    Main Telegram bot for CryptoAlpha.

    Provides trading signals and analysis via Telegram.
    Uses lite analyzer to avoid LLM rate limits.
    """

    def __init__(
        self,
        bot_token: str,
        exchange: str = "binance",
        chains: Optional[List[str]] = None,
    ):
        """
        Initialize the bot.

        Args:
            bot_token: Telegram bot token
            exchange: Exchange for price data
            chains: Chains for on-chain data
        """
        global _bot_instance
        _bot_instance = self

        self.bot_token = bot_token
        self.exchange = exchange
        self.chains = chains or ["ethereum"]

        # Initialize alert manager with persistence
        self.alert_manager = create_alert_manager(bot_token)

        # Initialize handlers
        self.handlers = create_handlers()
        self.handlers.bot_instance = self  # Give handlers access to bot

        # Telegram application
        self.application: Optional[Application] = None

        # Pipeline for background analysis
        self.pipeline = None

        # Chat IDs for notifications (also loaded from alert_manager)
        self.notification_chat_ids: Set[int] = self.alert_manager.get_all_chat_ids()

        logger.info(f"[Bot] Initialized with {len(self.notification_chat_ids)} saved subscriptions")

    def setup_handlers(self) -> None:
        """Set up command handlers."""
        if not self.application:
            return

        # Add command handlers
        self.application.add_handler(
            CommandHandler("start", self.handlers.start_command)
        )
        self.application.add_handler(
            CommandHandler("help", self.handlers.help_command)
        )
        self.application.add_handler(
            CommandHandler("status", self.handlers.status_command)
        )
        self.application.add_handler(
            CommandHandler("analyze", self.handlers.analyze_command)
        )
        self.application.add_handler(
            CommandHandler("signals", self.handlers.signals_command)
        )
        self.application.add_handler(
            CommandHandler("performance", self.handlers.performance_command)
        )
        self.application.add_handler(
            CommandHandler("settings", self.handlers.settings_command)
        )
        self.application.add_handler(
            CommandHandler("alerts", self.handlers.alerts_command)
        )

        # Learning system commands
        self.application.add_handler(
            CommandHandler("accuracy", self.handlers.accuracy_command)
        )
        self.application.add_handler(
            CommandHandler("ensemble", self.handlers.ensemble_command)
        )

        # Virtual Investor commands
        self.application.add_handler(
            CommandHandler("portfolio", self.handlers.portfolio_command)
        )
        self.application.add_handler(
            CommandHandler("vperformance", self.handlers.vperformance_command)
        )
        self.application.add_handler(
            CommandHandler("vtrades", self.handlers.vtrades_command)
        )

        # Unsubscribe command
        self.application.add_handler(
            CommandHandler("unsubscribe", self.handlers.unsubscribe_command)
        )

        # Error handler
        self.application.add_error_handler(self.handlers.error_handler)

        # Unknown command handler
        self.application.add_handler(
            MessageHandler(filters.COMMAND, self.handlers.unknown_command)
        )

    async def send_trade_notifications(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Check for trade notifications and send them."""
        if not self.pipeline:
            return

        # Update notification chat IDs from alert manager
        self.notification_chat_ids = self.alert_manager.get_all_chat_ids()

        if not self.notification_chat_ids:
            return

        try:
            notifications = self.pipeline.get_trade_notifications()

            for notification in notifications:
                # Check if it's a daily summary
                if isinstance(notification, tuple) and notification[0] == "DAILY_SUMMARY":
                    message = notification[1]
                else:
                    message = f"*Virtual Investor Trade*\n\n{notification}"

                # Send to all subscribed chats
                for chat_id in list(self.notification_chat_ids):
                    try:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=message,
                            parse_mode="Markdown",
                        )
                    except Exception as e:
                        logger.error(f"Failed to send to {chat_id}: {e}")
                        # Don't remove from set - might be temporary error

        except Exception as e:
            logger.error(f"Error sending notifications: {e}")

    async def run_async(self) -> None:
        """Run the bot asynchronously."""
        # Build application
        self.application = Application.builder().token(self.bot_token).build()

        # Setup handlers
        self.setup_handlers()

        # Initialize alert manager
        await self.alert_manager.initialize()

        # Get pipeline reference
        try:
            from core.lite_pipeline import get_lite_pipeline
            self.pipeline = get_lite_pipeline()
        except Exception as e:
            logger.warning(f"Could not get pipeline: {e}")

        # Setup job queue for periodic notifications
        if self.application.job_queue:
            # Check for notifications every 30 seconds
            self.application.job_queue.run_repeating(
                self.send_trade_notifications,
                interval=30,
                first=10,
            )

        # Start polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()

        logger.info(f"[Bot] Started with {len(self.notification_chat_ids)} subscribers")

    async def stop_async(self) -> None:
        """Stop the bot asynchronously."""
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()

    def run(self) -> None:
        """Run the bot (blocking)."""
        # Build application
        self.application = Application.builder().token(self.bot_token).build()

        # Setup handlers
        self.setup_handlers()

        # Get pipeline reference
        try:
            from core.lite_pipeline import get_lite_pipeline
            self.pipeline = get_lite_pipeline()
        except Exception as e:
            logger.warning(f"Could not get pipeline: {e}")

        # Setup job queue for periodic notifications
        if self.application.job_queue:
            self.application.job_queue.run_repeating(
                self.send_trade_notifications,
                interval=30,
                first=10,
            )

        # Run polling
        self.application.run_polling()


def create_bot(
    bot_token: str,
    exchange: str = "binance",
    chains: Optional[List[str]] = None,
) -> CryptoAlphaBot:
    """Create a CryptoAlphaBot instance."""
    return CryptoAlphaBot(
        bot_token=bot_token,
        exchange=exchange,
        chains=chains,
    )
