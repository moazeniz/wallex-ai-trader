"""Paper trading account: TMN cash, spot positions, fills against live Wallex depth, real fee schedule.

Each strategy owns one account (one SQLite file). Every order, fill and decision is appended to a
hash-chained JSONL log so a reviewer can verify that nothing was edited after the fact.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

# Wallex public fee schedule, level 1 (GET https://api.wallex.ir/v2/structured-fee, fetched 2026-10-22).
TAKER_FEE = {"TMN": 0.0030, "USDT": 0.0013}
MAKER_FEE = {"TMN": 0.0025, "USDT": 0.0010}
MIN_ORDER_TMN = 100_000  # organizer-stated minimum (API metadata says 50,000; use the stricter value)


def quote_of(symbol: str) -> str:
    return "USDT" if symbol.endswith("USDT") else "TMN"


@dataclass
class Position:
    symbol: str
    qty: float
    cost_tmn: float          # total TMN paid including entry fee
    entry_price: float
    opened_at: float
    stop: float | None = None
    target: float | None = None
    expires_at: float | None = None
    peak_price: float = 0.0  # for trailing stops
    meta: dict = field(default_factory=dict)


def _mirror(path: Path, record: dict):
    """Structured-log copy of a hash-chained record: decision.* (account decisions/fills), order.* (live order
    lifecycle from live.jsonl), engine.* and watchdog.*; errors carry their stack."""
    from . import obs
    kind = str(record.get("type", "record"))
    prefix = {"decisions": "decision", "live": "order", "engine": "engine"}.get(path.stem, path.stem)
    level = ("error" if "error" in kind or "crash" in kind else
             "warn" if any(w in kind for w in ("refused", "halt", "pause", "take_over", "ambiguous", "reconcile")) and record.get("differences", True) else "info")
    if not obs.enabled(level):
        return
    fields = {k: v for k, v in record.items() if k not in ("input", "trace", "ts", "type")}
    obs.log(f"{prefix}.{kind}", level, kind, account=path.parent.name if prefix in ("decision", "order") else None,
            stack=record.get("trace"), **{("rec_" + k if k in ("msg", "level", "event", "ts") else k): v for k, v in fields.items()})


class DecisionLog:
    """Append-only JSONL; each line carries sha256(previous_hash + canonical record)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.head = "0" * 64
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    self.head = json.loads(line)["hash"]

    def append(self, record: dict) -> str:
        body = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        digest = hashlib.sha256((self.head + body).encode()).hexdigest()
        with self.path.open("a") as f:
            f.write(json.dumps({"prev": self.head, "hash": digest, "record": json.loads(body)}, ensure_ascii=False) + "\n")
        self.head = digest
        _mirror(self.path, record)
        return digest

    @staticmethod
    def verify(path: Path) -> int:
        head, count = "0" * 64, 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            body = json.dumps(row["record"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            if row["prev"] != head or hashlib.sha256((head + body).encode()).hexdigest() != row["hash"]:
                raise ValueError(f"decision log broken at line {count + 1}")
            head, count = row["hash"], count + 1
        return count


class PaperAccount:
    def __init__(self, name: str, root: Path, initial_tmn: float = 5_000_000):
        self.name = name
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.dir / "account.sqlite", check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS positions(symbol TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS fills(id INTEGER PRIMARY KEY, ts REAL, symbol TEXT, side TEXT, qty REAL,
                price REAL, notional_tmn REAL, fee_tmn REAL, liquidity TEXT, pnl_tmn REAL, reason TEXT);
            CREATE TABLE IF NOT EXISTS equity(ts REAL PRIMARY KEY, equity_tmn REAL, cash_tmn REAL, exposure_tmn REAL);
        """)
        if self._get("initial") is None:
            for k, v in {"initial": initial_tmn, "cash": initial_tmn, "peak": initial_tmn, "max_dd": 0.0,
                         "fees": 0.0, "realized": 0.0, "created_at": time.time()}.items():
                self._set(k, v)
            self.db.commit()
        self.log = DecisionLog(self.dir / "decisions.jsonl")

    # ---- persistence helpers
    def _get(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, json.dumps(value)))

    @property
    def cash(self) -> float:
        return self._get("cash")

    @property
    def initial(self) -> float:
        return self._get("initial")

    def positions(self) -> dict[str, Position]:
        return {s: Position(**json.loads(v)) for s, v in self.db.execute("SELECT symbol, value FROM positions")}

    def save_position(self, p: Position):
        self.db.execute("INSERT OR REPLACE INTO positions VALUES(?,?)", (p.symbol, json.dumps(p.__dict__)))
        self.db.commit()

    # ---- valuation
    def equity(self, marks: dict[str, float]) -> tuple[float, float]:
        """Liquidation value in TMN: each position marked at its current best bid (USDT pairs via USDTTMN bid)."""
        exposure = 0.0
        for p in self.positions().values():
            mark = marks.get(p.symbol)
            exposure += p.qty * (mark if mark else p.entry_price)
        return self.cash + exposure, exposure

    def record_equity(self, marks: dict[str, float], now: float | None = None) -> dict:
        now = now or time.time()
        eq, exposure = self.equity(marks)
        peak = max(self._get("peak"), eq)
        dd = (peak - eq) / peak if peak else 0.0
        self._set("peak", peak)
        self._set("max_dd", max(self._get("max_dd"), dd))
        self.db.execute("INSERT OR REPLACE INTO equity VALUES(?,?,?,?)", (now, eq, self.cash, exposure))
        self.db.commit()
        return {"equity_tmn": eq, "cash_tmn": self.cash, "exposure_tmn": exposure, "drawdown": dd,
                "max_drawdown": self._get("max_dd"), "peak_tmn": peak}

    def summary(self, marks: dict[str, float]) -> dict:
        eq, exposure = self.equity(marks)
        fills = self.db.execute("SELECT COUNT(*), COALESCE(SUM(notional_tmn),0) FROM fills").fetchone()
        closed = self.db.execute("SELECT COUNT(*), COALESCE(SUM(pnl_tmn>0),0) FROM fills WHERE side='SELL'").fetchone()
        return {"name": self.name, "equity_tmn": round(eq), "return_pct": round(100 * (eq / self.initial - 1), 3),
                "cash_tmn": round(self.cash), "exposure_tmn": round(exposure), "fees_tmn": round(self._get("fees")),
                "realized_tmn": round(self._get("realized")), "max_drawdown_pct": round(100 * self._get("max_dd"), 3),
                "fills": fills[0], "volume_tmn": round(fills[1]), "closed_trades": closed[0], "winning_trades": closed[1],
                "open_positions": {s: {"qty": p.qty, "entry": p.entry_price, "mark": marks.get(s), "opened_at": p.opened_at}
                                   for s, p in self.positions().items()}}

    # ---- execution
    @staticmethod
    def walk_book(levels: list[tuple[float, float]], side: str, budget_quote: float | None, qty: float | None):
        """Consume price levels (asks for BUY, bids for SELL). Returns (filled_qty, quote_amount)."""
        filled = spent = 0.0
        for price, size in levels:
            if side == "BUY":
                take = min(size, (budget_quote - spent) / price)
            else:
                take = min(size, qty - filled)
            if take <= 0:
                break
            filled += take
            spent += take * price
        return filled, spent

    def book_fill(self, symbol: str, side: str, qty: float, price: float, usdt_tmn: float, liquidity: str,
                  reason: str = "", now: float | None = None, position_fields: dict | None = None,
                  fee_tmn: float | None = None, fee_in_base: bool = False) -> dict | None:
        """Apply one fill (qty at average price, in the market's quote currency) to cash and positions.

        fee_tmn overrides the schedule fee with the fee actually charged (live fills). fee_in_base means the exchange
        took a BUY fee from the coin received: we hold qty minus the fee and pay only the notional in cash.
        """
        now = now or time.time()
        quote = quote_of(symbol)
        to_tmn = 1.0 if quote == "TMN" else usdt_tmn
        fee_rate = (TAKER_FEE if liquidity == "taker" else MAKER_FEE)[quote]
        positions = self.positions()
        if side == "SELL":
            pos = positions.get(symbol)
            if not pos:
                return None
            qty = min(qty, pos.qty)
        if qty <= 0 or price <= 0:
            return None
        notional = qty * price * to_tmn
        fee = notional * fee_rate if fee_tmn is None else fee_tmn
        fee_rate = fee / notional if notional else fee_rate
        pnl = None
        if side == "BUY":
            held = qty - (fee / (price * to_tmn) if fee_in_base else 0.0)
            paid = notional + (0.0 if fee_in_base else fee)
            if paid > self.cash + 1e-6 and fee_tmn is None:
                return None  # a live fill is booked even if it overdraws the internal ledger; reconciliation corrects
            self._set("cash", self.cash - paid)
            pos = positions.get(symbol)
            if pos:
                pos.entry_price = (pos.entry_price * pos.qty + price * held) / (pos.qty + held)
                pos.cost_tmn += paid
                pos.qty += held
            else:
                pos = Position(symbol, held, paid, price, now, peak_price=price)
            for k, v in (position_fields or {}).items():
                setattr(pos, k, v)
            self.db.execute("INSERT OR REPLACE INTO positions VALUES(?,?)", (symbol, json.dumps(pos.__dict__)))
        else:
            basis = pos.cost_tmn * qty / pos.qty
            pnl = notional - fee - basis
            self._set("cash", self.cash + notional - fee)
            self._set("realized", self._get("realized") + pnl)
            pos.qty -= qty
            pos.cost_tmn -= basis
            if pos.qty * price * to_tmn < 1000:  # closed (sub-1000-TMN remainder is dust)
                self.db.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
            else:
                self.db.execute("INSERT OR REPLACE INTO positions VALUES(?,?)", (symbol, json.dumps(pos.__dict__)))
        self._set("fees", self._get("fees") + fee)
        self.db.execute("INSERT INTO fills(ts,symbol,side,qty,price,notional_tmn,fee_tmn,liquidity,pnl_tmn,reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (now, symbol, side, qty, price, notional, fee, liquidity, pnl, reason))
        self.db.commit()
        fill = {"type": "fill", "ts": now, "symbol": symbol, "side": side, "qty": qty, "price": price, "notional_tmn": notional,
                "fee_tmn": fee, "fee_rate": fee_rate, "liquidity": liquidity, "pnl_tmn": pnl, "reason": reason,
                "cash_after_tmn": self.cash}
        self.log.append(fill)
        return fill

    def market_order(self, symbol: str, side: str, book: dict, usdt_tmn: float, *, budget_tmn: float | None = None,
                     qty: float | None = None, max_impact: float = 0.004, reason: str = "", now: float | None = None,
                     position_fields: dict | None = None) -> dict | None:
        """Taker fill walking a live depth snapshot {'asks': [(p,q)...], 'bids': [...]} (best first).

        BUY spends at most budget_tmn (fee included); SELL sells qty (default: whole position).
        Levels further than max_impact from the best price are not used (no invented liquidity).
        """
        quote = quote_of(symbol)
        to_tmn = 1.0 if quote == "TMN" else usdt_tmn
        levels = book["asks"] if side == "BUY" else book["bids"]
        if not levels:
            return None
        best = levels[0][0]
        bound = best * (1 + max_impact) if side == "BUY" else best * (1 - max_impact)
        usable = [(p, q) for p, q in levels if (p <= bound if side == "BUY" else p >= bound)]
        if side == "BUY":
            budget_tmn = min(budget_tmn or 0, self.cash)
            if budget_tmn < MIN_ORDER_TMN:
                return None
            filled, spent_q = self.walk_book(usable, "BUY", budget_tmn / (1 + TAKER_FEE[quote]) / to_tmn, None)
        else:
            pos = self.positions().get(symbol)
            if not pos:
                return None
            want = min(qty or pos.qty, pos.qty)
            filled, spent_q = self.walk_book(usable, "SELL", None, want)
            if spent_q * to_tmn < MIN_ORDER_TMN:
                return None  # Wallex rejects orders under the minimum, whole-position dust included
        if filled <= 0:
            return None
        if side == "BUY" and spent_q * to_tmn < MIN_ORDER_TMN:
            return None
        fill = self.book_fill(symbol, side, filled, spent_q / filled, usdt_tmn, "taker", reason, now, position_fields)
        if fill:
            fill["book_best"] = best
            fill["partial"] = side == "BUY" and fill["notional_tmn"] + fill["fee_tmn"] < budget_tmn * 0.98
        return fill
