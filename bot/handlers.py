"""
Telegram Bot Handlers

Command handlers for the Telegram bot.
Enhanced with auto-registration for notifications.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from telegram import Update
from telegram.ext import ContextTypes

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.tokens import TOKEN_CONFIG


def _escape_md(text):
    """Escape Markdown special characters to prevent parsing errors."""
    if not isinstance(text, str):
        text = str(text)
    # Escape underscores which are common in variable names
    text = text.replace('_', '\\_')
    return text


def _get_lite_analyzer():
    """Lazy load lite analyzer."""
    from core.lite_analyzer import get_lite_analyzer
    return get_lite_analyzer()


def _get_lite_pipeline():
    """Lazy load lite pipeline."""
    from core.lite_pipeline import get_lite_pipeline
    return get_lite_pipeline()


def _get_virtual_investor():
    """Lazy load virtual investor."""
    try:
        from core.virtual_investor import get_virtual_investor
        return get_virtual_investor()
    except Exception as e:
        print(f"Warning: Could not load virtual investor: {e}")
        return None


def _get_feedback_evaluator():
    """Lazy load feedback evaluator."""
    try:
        from core.feedback_evaluator import get_feedback_evaluator
        return get_feedback_evaluator()
    except Exception as e:
        print(f"Warning: Could not load feedback evaluator: {e}")
        return None


def _get_alert_manager():
    """Lazy load alert manager from bot instance."""
    try:
        from bot.telegram_bot import _bot_instance
        if _bot_instance and hasattr(_bot_instance, 'alert_manager'):
            return _bot_instance.alert_manager
    except:
        pass
    return None


def format_status_message(status: Dict[str, Any]) -> str:
    """Format status message for Telegram."""
    running = "Running" if status.get("running") else "Stopped"
    tokens = status.get("tokens_tracked", 0)
    interval = status.get("interval_minutes", 15)

    message = f"""*Pipeline Status*

Status: {running}
Tokens: {tokens}
Interval: {interval} min

*Recent Jobs*
"""
    for job in status.get("jobs", [])[:10]:
        symbol = job.get("symbol", "?")
        last_run = job.get("last_run", "-")
        signal = job.get("signal", "-")
        message += f"{symbol}: {signal} ({last_run})\n"

    # Add virtual investor status if available
    vi = status.get("virtual_investor")
    if vi:
        message += f"""
*Virtual Investor*
Balance: ${vi.get('balance', 0):,.2f}
P&L: ${vi.get('pnl', 0):+,.2f} ({vi.get('return_pct', 0):+.1f}%)
Positions: {vi.get('open_positions', 0)}
Trades: {vi.get('total_trades', 0)}
"""

    return message


class BotHandlers:
    """
    Telegram bot command handlers.

    Handles all user commands and interactions.
    Auto-registers users for notifications on /start.
    """

    def __init__(
        self,
        analyzer=None,
        database=None,
    ):
        """
        Initialize handlers.

        Args:
            analyzer: Quick analyzer for on-demand analysis (optional, uses lite)
            database: Signal database for history (optional)
        """
        self.analyzer = analyzer
        self.database = database
        self.user_settings: Dict[int, Dict[str, Any]] = {}
        self.bot_instance = None  # Set by telegram_bot.py

    def _get_user_settings(self, user_id: int) -> Dict[str, Any]:
        """Get settings for a user."""
        if user_id not in self.user_settings:
            self.user_settings[user_id] = {
                "alerts_enabled": True,
                "min_confidence": 65,
                "signal_types": ["BUY", "SELL"],
                "min_strength": "WEAK",
            }
        return self.user_settings[user_id]

    async def start_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /start command - auto-registers user for notifications."""
        user = update.effective_user
        chat_id = update.effective_chat.id

        # Auto-register for notifications
        registered = False
        if self.bot_instance:
            # Register in notification chat IDs
            if hasattr(self.bot_instance, 'notification_chat_ids'):
                self.bot_instance.notification_chat_ids.add(chat_id)

            # Register in alert manager (persisted)
            if hasattr(self.bot_instance, 'alert_manager') and self.bot_instance.alert_manager:
                self.bot_instance.alert_manager.subscribe(
                    user_id=user.id,
                    chat_id=chat_id,
                    min_confidence=0.65,
                    signal_types=["BUY", "SELL"],
                    min_strength="WEAK",
                )
                registered = True
                print(f"[Bot] Registered user {user.id} (chat {chat_id}) for notifications")

        message = f"""*Welcome to CryptoAlpha, {user.first_name}!* 🚀

I'm your AI-powered crypto trading assistant. I analyze DeFi and L2 tokens using:
• Technical Analysis
• ML Pattern Recognition (GRU + LSTM + XGBoost)
• Dynamic Ensemble Learning

*Virtual Investor Active* 💰
Paper trading with $10,000 virtual credits.

{"✅ *You are registered for trade notifications!*" if registered else ""}

*Daily Summary*
You'll receive a daily performance report at 00:00 UTC.

Use /help to see available commands.
"""

        await update.message.reply_text(message, parse_mode="Markdown")

    async def help_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /help command."""
        help_text = """*CryptoAlpha Commands*

*Analysis*
/analyze <TOKEN> - Analyze a token (e.g. /analyze AAVE)
/status - Show pipeline status

*Virtual Investor*
/portfolio - View open positions
/vperformance - Paper trading stats
/vtrades - Recent virtual trades

*Learning System*
/accuracy - Model accuracy stats
/ensemble - Ensemble & meta-learner status

*Settings*
/alerts on|off - Toggle alerts
/unsubscribe - Stop all notifications

/help - Show this message
"""
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def status_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /status command - shows real pipeline data."""
        try:
            pipeline = _get_lite_pipeline()
            status = pipeline.get_status()
        except:
            status = {
                "running": True,
                "tokens_tracked": len(TOKEN_CONFIG),
                "interval_minutes": 15,
                "jobs": [
                    {"symbol": f"{sym}/USDT", "priority": 1, "last_run": None}
                    for sym in list(TOKEN_CONFIG.keys())[:10]
                ],
            }

        await update.message.reply_text(
            format_status_message(status),
            parse_mode="Markdown",
        )

    async def analyze_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /analyze <TOKEN> command - uses fast lite analyzer."""
        args = context.args
        if not args:
            await update.message.reply_text(
                "Please specify a token. Example: /analyze AAVE",
                parse_mode="Markdown",
            )
            return

        token = args[0].upper()
        symbol = f"{token}/USDT"

        status_msg = await update.message.reply_text(
            f"Analyzing {symbol}...",
            parse_mode="Markdown",
        )

        try:
            # Use lite analyzer (fast, no LLM)
            lite = _get_lite_analyzer()
            result = lite.analyze(symbol)

            if "error" in result:
                await status_msg.edit_text(f"Error: {result['error']}")
                return

            # Format result - handle both old and new formats
            ensemble = result.get("ensemble", result.get("gru", {}))
            direction = ensemble.get("direction", "NEUTRAL").upper()
            prob_up = ensemble.get("probability_up", 0.5) * 100

            # Get individual model predictions if available
            model_preds = ensemble.get("models", {})

            message = f"""*{result['symbol']}* Analysis

Price: ${result['price']:,.2f} ({result['change_24h']:+.1f}% 24h)

*Ensemble Prediction*
Direction: {direction}
Probability UP: {prob_up:.1f}%
"""

            # Add individual model predictions if available
            if model_preds:
                message += "\n*Model Breakdown*\n"
                for model, pred in model_preds.items():
                    m_dir = pred.get("direction", "?")
                    m_prob = pred.get("probability_up", 0.5) * 100
                    message += f"  {model.upper()}: {m_dir} ({m_prob:.0f}%)\n"

            message += f"""
*Signal*
Action: {result['signal']} ({_escape_md(result['strength'])})
Confidence: {result['confidence']:.0f}%

*Scores*
- Ensemble: {result['scores'].get('ensemble', result['scores'].get('GRU', 50)):.0f}
- Technical: {result['scores'].get('technical', result['scores'].get('Technical', 50)):.0f}
- Regime: {result['scores'].get('regime', result['scores'].get('Regime', 50)):.0f}

Market: {_escape_md(result['regime'])} | Vol: {_escape_md(result['volatility'])}
Time: {result['analysis_time']:.1f}s"""

            await status_msg.edit_text(message, parse_mode="Markdown")

        except Exception as e:
            await status_msg.edit_text(
                f"Error analyzing {symbol}: {str(e)}",
                parse_mode="Markdown",
            )

    async def ensemble_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /ensemble command - show ensemble and meta-learner status."""
        try:
            lite = _get_lite_analyzer()
            stats = lite.ensemble.get_stats()

            # Format weights
            weights_str = "\n".join([f"  {m}: {w:.1%}" for m, w in stats.get('current_weights', {}).items()])

            # Meta-learner status
            if stats.get('meta_learner_trained'):
                ml_status = f"✅ Active ({stats.get('meta_learner_samples', 0)} samples)"
                ml_trained = stats.get('meta_learner_last_trained', 'Unknown')
            else:
                buffer = stats.get('prediction_buffer', {})
                ml_status = f"⏳ Training ({buffer.get('evaluated', 0)}/50 samples)"
                ml_trained = "Not yet"

            # Model accuracies
            acc_str = ""
            for model, acc in stats.get('model_accuracies', {}).items():
                acc_str += f"  {model}: {acc['overall']:.1%} (UP: {acc['up']:.1%})\n"

            message = f"""*Ensemble Status*

*Current Weights*
{weights_str}

*Meta-Learner*
Status: {ml_status}
Last Trained: {ml_trained}

*Model Accuracies (rolling)*
{acc_str}
*Settings*
Focus Bullish: {'Yes' if stats.get('focus_bullish') else 'No'}
Min Confidence Gate: {stats.get('min_confidence', 0.15):.0%}
"""

            await update.message.reply_text(message, parse_mode="Markdown")

        except Exception as e:
            await update.message.reply_text(
                f"Error loading ensemble stats: {str(e)}",
                parse_mode="Markdown",
            )

    async def portfolio_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /portfolio command - show virtual portfolio."""
        try:
            investor = _get_virtual_investor()
            if not investor:
                await update.message.reply_text(
                    "Virtual Investor not available.",
                    parse_mode="Markdown",
                )
                return

            positions = investor.get_open_positions()
            balance = investor.get_balance()
            available = investor.get_available_balance()
            perf = investor.get_performance()

            message = f"""*Virtual Portfolio* 💰

Balance: ${balance:,.2f}
Available: ${available:,.2f}
P&L: ${perf['total_pnl']:+,.2f} ({perf['total_return_pct']:+.1f}%)

*Open Positions* ({len(positions)})
"""
            if positions:
                for pos in positions:
                    pnl_target = pos.take_profit / pos.entry_price - 1
                    pnl_stop = 1 - pos.stop_loss / pos.entry_price
                    value = pos.entry_price * pos.quantity
                    message += f"""
{pos.symbol}
  Entry: ${pos.entry_price:,.2f}
  Qty: {pos.quantity:.6f} (${value:,.2f})
  SL: ${pos.stop_loss:,.2f} (-{pnl_stop*100:.1f}%)
  TP: ${pos.take_profit:,.2f} (+{pnl_target*100:.1f}%)
  Conf: {pos.signal_confidence:.0f}%
"""
            else:
                message += "\n_No open positions_"

            await update.message.reply_text(message, parse_mode="Markdown")

        except Exception as e:
            await update.message.reply_text(
                f"Error loading portfolio: {str(e)}",
                parse_mode="Markdown",
            )

    async def vperformance_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /vperformance command - show virtual investor performance."""
        try:
            investor = _get_virtual_investor()
            if not investor:
                await update.message.reply_text(
                    "Virtual Investor not available.",
                    parse_mode="Markdown",
                )
                return

            perf = investor.get_performance()

            message = f"""*Virtual Investor Performance* 📊

*Account*
Initial: ${perf['initial_balance']:,.2f}
Current: ${perf['current_balance']:,.2f}
P&L: ${perf['total_pnl']:+,.2f} ({perf['total_return_pct']:+.1f}%)

*Trading Stats*
Total Trades: {perf['total_trades']}
Win Rate: {perf['win_rate']:.1f}%
Avg P&L/Trade: ${perf['avg_pnl_per_trade']:,.2f}
Max Drawdown: {perf['max_drawdown']:.1f}%

*Current*
Open Positions: {perf['open_positions']}

_Paper trading - no real money_
"""
            await update.message.reply_text(message, parse_mode="Markdown")

        except Exception as e:
            await update.message.reply_text(
                f"Error loading performance: {str(e)}",
                parse_mode="Markdown",
            )

    async def vtrades_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /vtrades command - show recent virtual trades."""
        try:
            investor = _get_virtual_investor()
            if not investor:
                await update.message.reply_text(
                    "Virtual Investor not available.",
                    parse_mode="Markdown",
                )
                return

            trades = investor.get_recent_trades(limit=10)

            message = "*Recent Virtual Trades*\n\n"

            if not trades:
                message += "_No trades yet. Waiting for signals..._"
            else:
                for trade in trades:
                    if trade.exit_reason == "take_profit":
                        exit_emoji = "✅ TP"
                    elif trade.exit_reason == "stop_loss":
                        exit_emoji = "❌ SL"
                    else:
                        exit_emoji = "📤 SIG"

                    exit_time = trade.exit_time.strftime("%m/%d %H:%M") if isinstance(trade.exit_time, datetime) else str(trade.exit_time)[:10]

                    message += f"""{trade.symbol}
  ${trade.pnl:+,.2f} ({trade.pnl_pct:+.1f}%) [{exit_emoji}]
  ${trade.entry_price:,.2f} → ${trade.exit_price:,.2f}
  {exit_time}

"""

            await update.message.reply_text(message, parse_mode="Markdown")

        except Exception as e:
            await update.message.reply_text(
                f"Error loading trades: {str(e)}",
                parse_mode="Markdown",
            )

    async def signals_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /signals command."""
        await update.message.reply_text(
            "Use /analyze <TOKEN> to analyze a specific token.",
            parse_mode="Markdown",
        )

    async def performance_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /performance command."""
        # Redirect to virtual investor performance
        await self.vperformance_command(update, context)

    async def settings_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /settings command."""
        user_id = update.effective_user.id
        settings = self._get_user_settings(user_id)

        status = "enabled" if settings["alerts_enabled"] else "disabled"
        message = f"""*Settings*

Alerts: {status}
Min Confidence: {settings['min_confidence']}%

Use /alerts on|off to toggle alerts.
Use /unsubscribe to stop all notifications.
"""
        await update.message.reply_text(message, parse_mode="Markdown")

    async def alerts_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /alerts command."""
        user_id = update.effective_user.id
        settings = self._get_user_settings(user_id)

        args = context.args
        if args:
            action = args[0].lower()
            if action == "on":
                settings["alerts_enabled"] = True
                # Update in alert manager
                if self.bot_instance and hasattr(self.bot_instance, 'alert_manager'):
                    self.bot_instance.alert_manager.toggle_alerts(user_id, True)
                await update.message.reply_text(
                    "✅ Alerts *enabled*. You will receive trade notifications.",
                    parse_mode="Markdown",
                )
            elif action == "off":
                settings["alerts_enabled"] = False
                if self.bot_instance and hasattr(self.bot_instance, 'alert_manager'):
                    self.bot_instance.alert_manager.toggle_alerts(user_id, False)
                await update.message.reply_text(
                    "🔕 Alerts *disabled*. You will not receive trade notifications.",
                    parse_mode="Markdown",
                )
            else:
                await update.message.reply_text(
                    "Use `/alerts on` or `/alerts off`",
                    parse_mode="Markdown",
                )
        else:
            status = "enabled" if settings["alerts_enabled"] else "disabled"
            await update.message.reply_text(
                f"Alerts: *{status}*. Use `/alerts on` or `/alerts off` to toggle.",
                parse_mode="Markdown",
            )

    async def unsubscribe_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /unsubscribe command - remove from all notifications."""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        if self.bot_instance:
            # Remove from notification chat IDs
            if hasattr(self.bot_instance, 'notification_chat_ids'):
                self.bot_instance.notification_chat_ids.discard(chat_id)

            # Remove from alert manager
            if hasattr(self.bot_instance, 'alert_manager') and self.bot_instance.alert_manager:
                self.bot_instance.alert_manager.unsubscribe(user_id)

        await update.message.reply_text(
            "✅ You have been unsubscribed from all notifications.\n\nUse /start to re-subscribe.",
            parse_mode="Markdown",
        )

    async def accuracy_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /accuracy command - show learning system stats."""
        try:
            evaluator = _get_feedback_evaluator()
            if not evaluator:
                await update.message.reply_text(
                    "Learning system not available.",
                    parse_mode="Markdown",
                )
                return

            report = evaluator.format_learning_report(days=7)
            await update.message.reply_text(report, parse_mode="Markdown")

        except Exception as e:
            await update.message.reply_text(
                f"Error loading accuracy stats: {str(e)}",
                parse_mode="Markdown",
            )

    async def unknown_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle unknown commands."""
        await update.message.reply_text(
            "Unknown command. Use /help to see available commands.",
            parse_mode="Markdown",
        )

    async def error_handler(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle errors."""
        print(f"Error: {context.error}")

        if update and update.message:
            await update.message.reply_text(
                "An error occurred. Please try again later.",
                parse_mode="Markdown",
            )


def create_handlers(
    analyzer=None,
    database=None,
) -> BotHandlers:
    """Create BotHandlers instance."""
    return BotHandlers(analyzer=analyzer, database=database)
