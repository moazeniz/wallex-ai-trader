"""
Message Formatters

Telegram message formatting utilities.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import TradingSignal, SignalType, SignalStrength
from core.signal_generator import SignalResult


def format_signal_message(signal: TradingSignal) -> str:
    """Format a trading signal for Telegram."""
    # Signal emoji
    if signal.signal_type == SignalType.BUY:
        emoji = "🟢"
        action = "BUY"
    elif signal.signal_type == SignalType.SELL:
        emoji = "🔴"
        action = "SELL"
    else:
        emoji = "⚪"
        action = "HOLD"

    # Strength indicator
    strength_map = {
        SignalStrength.STRONG: "⭐⭐⭐",
        SignalStrength.MODERATE: "⭐⭐",
        SignalStrength.WEAK: "⭐",
    }
    stars = strength_map.get(signal.strength, "⭐")

    # Calculate percentages
    if signal.signal_type == SignalType.BUY:
        sl_pct = (signal.stop_loss - signal.entry_price) / signal.entry_price * 100
        tp_pct = (signal.take_profit - signal.entry_price) / signal.entry_price * 100
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
    else:
        sl_pct = (signal.stop_loss - signal.entry_price) / signal.entry_price * 100
        tp_pct = (signal.take_profit - signal.entry_price) / signal.entry_price * 100
        risk = abs(signal.stop_loss - signal.entry_price)
        reward = abs(signal.entry_price - signal.take_profit)

    rr_ratio = reward / risk if risk > 0 else 0

    message = f"""
{emoji} *{action} SIGNAL: {signal.symbol}* {stars}

*Confidence:* {signal.confidence:.0%}
*Entry:* ${signal.entry_price:,.4f}
*Stop Loss:* ${signal.stop_loss:,.4f} ({sl_pct:+.1f}%)
*Take Profit:* ${signal.take_profit:,.4f} ({tp_pct:+.1f}%)
*Risk/Reward:* {rr_ratio:.1f}:1
*Position Size:* {signal.position_size_pct:.1f}%

───────────────
*Technical:* {signal.technical_summary}

*On-Chain:* {signal.onchain_summary}

*Risk:* {signal.risk_assessment}
───────────────
_{signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}_
"""
    return message.strip()


def format_analysis_result(result: SignalResult) -> str:
    """Format analysis result for Telegram."""
    if result.signal and result.approved:
        return format_signal_message(result.signal)

    # No signal or not approved
    status_emoji = "⚠️" if not result.approved else "📊"

    message = f"""
{status_emoji} *Analysis: {result.symbol}*

*Status:* {'Signal Rejected' if not result.approved else 'No Action'}
*Analysis Time:* {result.analysis_time:.1f}s

*Technical:* {result.technical_summary}
*On-Chain:* {result.onchain_summary}

*Market Regime:* {result.regime}
*Anomaly Score:* {result.anomaly_score:.2f}
*Pattern Probability:* {result.pattern_probability:.0%}
"""

    if result.errors:
        message += f"\n*Errors:* {', '.join(result.errors)}"

    return message.strip()


def format_performance_summary(performance: Dict[str, Any]) -> str:
    """Format performance summary for Telegram."""
    win_rate = performance.get("win_rate", 0)
    total_pnl = performance.get("total_pnl", 0)

    # Performance emoji
    if total_pnl > 10:
        perf_emoji = "🚀"
    elif total_pnl > 0:
        perf_emoji = "📈"
    elif total_pnl < -10:
        perf_emoji = "📉"
    else:
        perf_emoji = "➡️"

    message = f"""
{perf_emoji} *Performance Summary ({performance.get('period_days', 30)} Days)*

*Total Signals:* {performance.get('total_signals', 0)}
*Winning:* {performance.get('winning_signals', 0)}
*Losing:* {performance.get('losing_signals', 0)}
*Win Rate:* {win_rate:.1f}%

*Total PnL:* {total_pnl:+.1f}%
*Avg Win:* {performance.get('avg_win', 0):+.1f}%
*Avg Loss:* {performance.get('avg_loss', 0):.1f}%

*Best Trade:* {performance.get('best_trade', 0):+.1f}%
*Worst Trade:* {performance.get('worst_trade', 0):.1f}%
"""
    return message.strip()


def format_status_message(status: Dict[str, Any]) -> str:
    """Format system status for Telegram."""
    running = status.get("running", False)
    status_emoji = "🟢" if running else "🔴"

    message = f"""
{status_emoji} *System Status*

*Status:* {'Running' if running else 'Stopped'}
*Tokens Tracked:* {status.get('tokens_tracked', 0)}
*Update Interval:* {status.get('interval_minutes', 15)} minutes

*Tracked Tokens:*
"""

    for job in status.get("jobs", [])[:10]:
        last_run = job.get("last_run", "Never")
        if last_run and last_run != "Never":
            last_run = last_run[:16]  # Truncate timestamp
        message += f"• {job['symbol']} (P{job['priority']}) - {last_run}\n"

    return message.strip()


def format_help_message() -> str:
    """Format help message for Telegram."""
    return """
🤖 *CryptoAlpha Trading Bot*

*Available Commands:*

/start - Initialize bot
/status - System status
/analyze <TOKEN> - Analyze a specific token
/signals - Recent signals
/performance - Performance metrics
/settings - Bot settings
/alerts - Toggle alerts
/help - This message

*Examples:*
• `/analyze BTC` - Analyze Bitcoin
• `/analyze AAVE` - Analyze AAVE with on-chain

*Signal Format:*
🟢 BUY / 🔴 SELL / ⚪ HOLD
⭐⭐⭐ Strong / ⭐⭐ Moderate / ⭐ Weak

_Use responsibly. Not financial advice._
"""


def format_token_list(tokens: List[Dict[str, Any]]) -> str:
    """Format token list for Telegram."""
    message = "*📊 Tracked Tokens*\n\n"

    for token in tokens:
        symbol = token.get("symbol", "Unknown")
        chain = token.get("chain", "ethereum")
        priority = token.get("priority", 2)

        priority_emoji = "🔥" if priority == 1 else "📌" if priority == 2 else "📎"
        message += f"{priority_emoji} {symbol} ({chain})\n"

    return message.strip()


def format_alert_settings(enabled: bool, filters: Dict[str, Any]) -> str:
    """Format alert settings for Telegram."""
    status_emoji = "🔔" if enabled else "🔕"

    message = f"""
{status_emoji} *Alert Settings*

*Alerts:* {'Enabled' if enabled else 'Disabled'}

*Filters:*
• Min Confidence: {filters.get('min_confidence', 65)}%
• Signal Types: {', '.join(filters.get('signal_types', ['BUY', 'SELL']))}
• Min Strength: {filters.get('min_strength', 'WEAK')}

Use /alerts on/off to toggle alerts.
"""
    return message.strip()


def escape_markdown(text: str) -> str:
    """Escape Markdown special characters."""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text
