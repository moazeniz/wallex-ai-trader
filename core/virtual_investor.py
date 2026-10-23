"""
Virtual Investor Agent - Paper Trading Simulator

Trades based on system signals with virtual money to validate performance.
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An open position."""
    id: int
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    signal_confidence: float


@dataclass
class Trade:
    """A completed trade."""
    id: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    exit_reason: str  # "signal", "stop_loss", "take_profit"


class VirtualInvestor:
    """Paper trading agent that trades based on system signals."""

    STOP_LOSS_PCT = 0.05      # -5%
    TAKE_PROFIT_PCT = 0.10    # +10%
    MIN_CONFIDENCE = 65.0     # Minimum signal confidence to trade
    MIN_POSITION_PCT = 0.05   # 5% of balance minimum
    MAX_POSITION_PCT = 0.20   # 20% of balance maximum

    def __init__(
        self,
        initial_balance: float = 10000.0,
        db_path: str = "/opt/wallex/data/virtual_investor.db"
    ):
        self.initial_balance = initial_balance
        self.db_path = db_path
        self.positions: Dict[str, Position] = {}

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_state()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account (
                id INTEGER PRIMARY KEY,
                balance REAL NOT NULL,
                initial_balance REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                signal_confidence REAL NOT NULL,
                status TEXT DEFAULT 'open'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP NOT NULL,
                pnl REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                exit_reason TEXT NOT NULL
            )
        """)

        # Create balance history table for drawdown calculation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS balance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                balance REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _load_state(self):
        """Load account state and open positions from DB."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if account exists, create if not
        cursor.execute("SELECT balance FROM account WHERE id = 1")
        row = cursor.fetchone()

        if row is None:
            cursor.execute(
                "INSERT INTO account (id, balance, initial_balance) VALUES (1, ?, ?)",
                (self.initial_balance, self.initial_balance)
            )
            conn.commit()
            # Record initial balance
            cursor.execute(
                "INSERT INTO balance_history (balance) VALUES (?)",
                (self.initial_balance,)
            )
            conn.commit()

        # Load open positions
        cursor.execute("""
            SELECT id, symbol, side, entry_price, quantity, entry_time,
                   stop_loss, take_profit, signal_confidence
            FROM positions WHERE status = 'open'
        """)

        for row in cursor.fetchall():
            pos = Position(
                id=row[0],
                symbol=row[1],
                side=row[2],
                entry_price=row[3],
                quantity=row[4],
                entry_time=datetime.fromisoformat(row[5]) if isinstance(row[5], str) else row[5],
                stop_loss=row[6],
                take_profit=row[7],
                signal_confidence=row[8],
            )
            self.positions[pos.symbol] = pos

        conn.close()
        logger.info(f"Loaded {len(self.positions)} open positions")

    def get_balance(self) -> float:
        """Get current balance."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT balance FROM account WHERE id = 1")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else self.initial_balance

    def _update_balance(self, new_balance: float):
        """Update account balance."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE account SET balance = ? WHERE id = 1", (new_balance,))
        # Record in history
        cursor.execute("INSERT INTO balance_history (balance) VALUES (?)", (new_balance,))
        conn.commit()
        conn.close()

    def get_available_balance(self) -> float:
        """Get balance not tied up in positions."""
        total_in_positions = sum(
            pos.entry_price * pos.quantity
            for pos in self.positions.values()
        )
        return self.get_balance() - total_in_positions

    def calculate_position_size(self, confidence: float) -> float:
        """
        Calculate position size based on confidence (5-20% of available balance).

        confidence 65% -> 5%
        confidence 100% -> 20%
        """
        conf_range = 100 - self.MIN_CONFIDENCE
        conf_normalized = min(1.0, (confidence - self.MIN_CONFIDENCE) / conf_range)
        position_pct = self.MIN_POSITION_PCT + conf_normalized * (self.MAX_POSITION_PCT - self.MIN_POSITION_PCT)

        available = self.get_available_balance()
        return available * position_pct

    def process_signal(self, signal: Dict[str, Any], current_price: float) -> Optional[str]:
        """
        Process a trading signal.

        Args:
            signal: Dict with 'symbol', 'signal' (BUY/SELL/HOLD), 'confidence' (0-100)
            current_price: Current market price

        Returns:
            Action message or None
        """
        symbol = signal.get("symbol", "")
        signal_type = signal.get("signal", "HOLD")
        confidence = signal.get("confidence", 0)

        messages = []

        # First check exit conditions for existing position
        if symbol in self.positions:
            exit_msg = self._check_exit_conditions(symbol, current_price)
            if exit_msg:
                messages.append(exit_msg)

        # If still have position after exit check, handle SELL signal
        if symbol in self.positions and signal_type == "SELL":
            msg = self._close_position(symbol, current_price, "signal")
            if msg:
                messages.append(msg)

        # Open new position on BUY with sufficient confidence
        elif symbol not in self.positions and signal_type == "BUY" and confidence >= self.MIN_CONFIDENCE:
            msg = self._open_position(symbol, current_price, confidence)
            if msg:
                messages.append(msg)

        return " | ".join(messages) if messages else None

    def _open_position(self, symbol: str, price: float, confidence: float) -> Optional[str]:
        """Open a new long position."""
        position_value = self.calculate_position_size(confidence)

        if position_value < 10:  # Minimum $10 position
            logger.warning(f"Position size too small for {symbol}: ${position_value:.2f}")
            return None

        quantity = position_value / price
        stop_loss = price * (1 - self.STOP_LOSS_PCT)
        take_profit = price * (1 + self.TAKE_PROFIT_PCT)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now()
        cursor.execute("""
            INSERT INTO positions
            (symbol, side, entry_price, quantity, entry_time, stop_loss, take_profit, signal_confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """, (symbol, "long", price, quantity, now.isoformat(), stop_loss, take_profit, confidence))

        position_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Track in memory
        self.positions[symbol] = Position(
            id=position_id,
            symbol=symbol,
            side="long",
            entry_price=price,
            quantity=quantity,
            entry_time=now,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_confidence=confidence,
        )

        msg = f"OPENED {symbol} @ ${price:,.2f} (qty: {quantity:.6f}, SL: ${stop_loss:,.2f}, TP: ${take_profit:,.2f})"
        logger.info(msg)
        return msg

    def _close_position(self, symbol: str, price: float, reason: str) -> Optional[str]:
        """Close an open position."""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        pnl = (price - position.entry_price) * position.quantity
        pnl_pct = (price / position.entry_price - 1) * 100

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now()

        # Mark position as closed
        cursor.execute(
            "UPDATE positions SET status = 'closed' WHERE id = ?",
            (position.id,)
        )

        # Record trade
        cursor.execute("""
            INSERT INTO trades
            (symbol, side, entry_price, exit_price, quantity, entry_time, exit_time, pnl, pnl_pct, exit_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, position.side, position.entry_price, price,
            position.quantity, position.entry_time.isoformat(), now.isoformat(),
            pnl, pnl_pct, reason
        ))

        # Update balance
        current_balance = self.get_balance()
        new_balance = current_balance + pnl
        cursor.execute("UPDATE account SET balance = ? WHERE id = 1", (new_balance,))
        cursor.execute("INSERT INTO balance_history (balance) VALUES (?)", (new_balance,))

        conn.commit()
        conn.close()

        # Remove from memory
        del self.positions[symbol]

        emoji = "+" if pnl >= 0 else ""
        msg = f"CLOSED {symbol} @ ${price:,.2f} ({reason}) | P&L: {emoji}${pnl:,.2f} ({pnl_pct:+.1f}%)"
        logger.info(msg)
        return msg

    def _check_exit_conditions(self, symbol: str, current_price: float) -> Optional[str]:
        """Check and execute stop-loss or take-profit."""
        position = self.positions.get(symbol)
        if not position:
            return None

        if current_price <= position.stop_loss:
            return self._close_position(symbol, position.stop_loss, "stop_loss")
        elif current_price >= position.take_profit:
            return self._close_position(symbol, position.take_profit, "take_profit")

        return None

    def check_all_positions(self, prices: Dict[str, float]) -> List[str]:
        """
        Check all open positions against current prices.

        Args:
            prices: Dict mapping symbol to current price

        Returns:
            List of action messages
        """
        messages = []
        symbols = list(self.positions.keys())

        for symbol in symbols:
            if symbol in prices:
                msg = self._check_exit_conditions(symbol, prices[symbol])
                if msg:
                    messages.append(msg)

        return messages

    def get_open_positions(self) -> List[Position]:
        """Get list of open positions."""
        return list(self.positions.values())

    def get_recent_trades(self, limit: int = 10) -> List[Trade]:
        """Get recent completed trades."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, symbol, side, entry_price, exit_price, quantity,
                   entry_time, exit_time, pnl, pnl_pct, exit_reason
            FROM trades
            ORDER BY exit_time DESC
            LIMIT ?
        """, (limit,))

        trades = []
        for row in cursor.fetchall():
            trades.append(Trade(
                id=row[0],
                symbol=row[1],
                side=row[2],
                entry_price=row[3],
                exit_price=row[4],
                quantity=row[5],
                entry_time=datetime.fromisoformat(row[6]) if isinstance(row[6], str) else row[6],
                exit_time=datetime.fromisoformat(row[7]) if isinstance(row[7], str) else row[7],
                pnl=row[8],
                pnl_pct=row[9],
                exit_reason=row[10],
            ))

        conn.close()
        return trades

    def _count_trades(self) -> int:
        """Count total completed trades."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def _calculate_win_rate(self) -> float:
        """Calculate win rate (% of profitable trades)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0")
        wins = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM trades")
        total = cursor.fetchone()[0]

        conn.close()
        return (wins / total * 100) if total > 0 else 0.0

    def _calculate_avg_pnl(self) -> float:
        """Calculate average P&L per trade."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT AVG(pnl) FROM trades")
        avg = cursor.fetchone()[0]
        conn.close()
        return avg if avg else 0.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT balance FROM balance_history ORDER BY timestamp")
        balances = [row[0] for row in cursor.fetchall()]
        conn.close()

        if len(balances) < 2:
            return 0.0

        peak = balances[0]
        max_dd = 0.0

        for balance in balances:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def get_performance(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        current_balance = self.get_balance()

        # Calculate unrealized P&L for open positions (would need current prices)
        unrealized_pnl = 0.0  # Will be updated when we have prices

        return {
            "initial_balance": self.initial_balance,
            "current_balance": current_balance,
            "total_pnl": current_balance - self.initial_balance,
            "total_return_pct": ((current_balance / self.initial_balance) - 1) * 100,
            "open_positions": len(self.positions),
            "total_trades": self._count_trades(),
            "win_rate": self._calculate_win_rate(),
            "avg_pnl_per_trade": self._calculate_avg_pnl(),
            "max_drawdown": self._calculate_max_drawdown(),
        }

    def reset(self):
        """Reset the virtual investor (clear all data)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM trades")
        cursor.execute("DELETE FROM positions")
        cursor.execute("DELETE FROM balance_history")
        cursor.execute("UPDATE account SET balance = ? WHERE id = 1", (self.initial_balance,))
        cursor.execute("INSERT INTO balance_history (balance) VALUES (?)", (self.initial_balance,))

        conn.commit()
        conn.close()

        self.positions = {}
        logger.info(f"Virtual investor reset to ${self.initial_balance:,.2f}")


# Singleton instance
_investor: Optional[VirtualInvestor] = None


def get_virtual_investor() -> VirtualInvestor:
    """Get or create the singleton virtual investor."""
    global _investor
    if _investor is None:
        _investor = VirtualInvestor()
    return _investor
