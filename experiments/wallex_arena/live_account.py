"""A PaperAccount whose orders go to Wallex through LiveExecutor.

Same interface the engine and strategies use, so a strategy proven in the arena runs unchanged on the real account:
market orders become LIMIT orders bounded by max_impact from the best price (unfilled remainder cancelled), resting
orders become real LIMIT orders, and only fills reported by Wallex are booked, at the fee Wallex actually charged.
The internal ledger is reconciled against exchange balances every few minutes; differences are logged and the
exchange quantity is adopted.
"""
from __future__ import annotations

import time

from .account import MAKER_FEE, MIN_ORDER_TMN, TAKER_FEE, PaperAccount
from .live import AmbiguousOrder, LiveExecutor, LiveRefused, WallexAPIError


class LiveAccount(PaperAccount):
    live = True
    # Wallex allows 20 order placements per 10 s; the client sleeps when that is reached, which would stall the engine.
    # Resting (non-urgent) orders may use at most this many slots per window, leaving the rest for protective sells.
    RESTING_ORDER_BUDGET = 12

    def __init__(self, name: str, root, executor: LiveExecutor, initial_tmn: float = 5_000_000):
        super().__init__(name, root, initial_tmn)
        self.ex = executor
        self.order_fields: dict[str, dict] = {}   # client_id -> position fields to apply when a resting BUY fills
        self.reconciled_at = 0.0

    # ---- fills reported by Wallex
    def _book(self, fill: dict, usdt_tmn: float, position_fields: dict | None = None) -> dict | None:
        base = self.ex.markets.get(fill["symbol"], {}).get("base") or fill["symbol"][:-3]
        fee_in_base = fill.get("fee_asset") == base and fill["side"] == "BUY"
        fee_tmn = fill.get("fee_tmn")
        if fee_tmn is None:  # fee asset not reported: book the schedule fee (a real fill is never dropped)
            fee_tmn = fill["qty"] * fill["price"] * (MAKER_FEE if fill.get("liquidity") == "maker" else TAKER_FEE)["TMN"]
        return self.book_fill(fill["symbol"], fill["side"], fill["qty"], fill["price"], usdt_tmn,
                              fill.get("liquidity") or "taker", reason=fill.get("reason", ""), now=fill.get("ts"),
                              position_fields=position_fields, fee_tmn=fee_tmn, fee_in_base=fee_in_base)

    def _book_all(self, fills: list[dict], usdt_tmn: float) -> list[dict]:
        booked = []
        for f in fills:
            fields = self.order_fields.get(f["client_id"]) if f["side"] == "BUY" else None
            b = self._book(f, usdt_tmn, fields)
            if b:
                b["client_id"] = f["client_id"]
                booked.append(b)
        return booked

    def _refused(self, op: str, symbol: str, exc: Exception) -> None:
        self.log.append({"ts": time.time(), "type": "live_refused", "op": op, "symbol": symbol,
                         "error": type(exc).__name__, "message": str(exc)[:300]})
        return None

    # ---- PaperAccount interface
    def market_order(self, symbol: str, side: str, book: dict, usdt_tmn: float, *, budget_tmn: float | None = None,
                     qty: float | None = None, max_impact: float = 0.004, reason: str = "", now: float | None = None,
                     position_fields: dict | None = None) -> dict | None:
        try:
            if side == "BUY":
                budget = min(budget_tmn or 0, self.cash)
                if budget < MIN_ORDER_TMN:
                    return None
                res = self.ex.market_buy(symbol, budget, book["asks"][0][0] * (1 + max_impact), reason)
            else:
                pos = self.positions().get(symbol)
                if not pos:
                    return None
                res = self.ex.market_sell(symbol, min(qty or pos.qty, pos.qty), book["bids"][0][0] * (1 - max_impact), reason)
        except (LiveRefused, WallexAPIError, AmbiguousOrder, OSError) as exc:
            return self._refused("market_" + side.lower(), symbol, exc)
        fills = res.get("fills", [])
        if side == "BUY" and fills:
            cid = fills[0]["client_id"]
            self.order_fields[cid] = position_fields or {}
        booked = self._book_all(fills, usdt_tmn)
        if not booked:
            return None
        qty_total = sum(b["qty"] for b in booked)
        notional = sum(b["notional_tmn"] for b in booked)
        return {"type": "fill", "symbol": symbol, "side": side, "qty": qty_total, "price": notional / qty_total if qty_total else 0,
                "notional_tmn": notional, "fee_tmn": sum(b["fee_tmn"] for b in booked), "liquidity": "taker",
                "pnl_tmn": sum((b["pnl_tmn"] or 0) for b in booked) if side == "SELL" else None, "reason": reason,
                "live": True, "client_id": booked[0]["client_id"], "book_best": book["asks" if side == "BUY" else "bids"][0][0],
                "partial": side == "BUY" and notional < (budget_tmn or 0) * 0.98}

    def _resting_budget_left(self) -> bool:
        api = getattr(self.ex, "api", None)
        posts = getattr(api, "_posts", None)
        if posts is None:
            return True
        window = 10.0
        now = time.time()
        recent = sum(1 for t in posts if now - t <= window)
        return recent < self.RESTING_ORDER_BUDGET

    def place_resting(self, symbol: str, side: str, price: float, qty: float, reason: str, fields: dict) -> str | None:
        if not self._resting_budget_left():
            self.log.append({"ts": time.time(), "type": "live_skipped", "op": "place_" + side.lower(), "symbol": symbol,
                             "why": "resting_order_budget"})
            return None  # the strategy re-quotes on a later tick; protective sells keep their slots
        try:
            res = self.ex.place_limit(symbol, side, price, qty, reason)
        except (LiveRefused, WallexAPIError, AmbiguousOrder, OSError) as exc:
            return self._refused("place_" + side.lower(), symbol, exc)
        cid = res["order"]["client_id"] if "order" in res else res["result"]["clientOrderId"]
        if side == "BUY":
            self.order_fields[cid] = fields or {}
        return cid

    def cancel_resting(self, client_id: str, usdt_tmn: float, reason: str) -> list[dict]:
        try:
            res = self.ex.cancel(client_id, reason)
        except (LiveRefused, WallexAPIError, OSError) as exc:
            self._refused("cancel", client_id, exc)
            return []
        return self._book_all(res.get("fills", []), usdt_tmn)

    def sync(self, usdt_tmn: float) -> tuple[list[dict], set[str]]:
        """Book new fills of any order; return (fills, client_ids that are no longer active)."""
        booked = self._book_all(self.ex.sync(), usdt_tmn)
        inactive = {cid for cid, o in self.ex.orders.items() if not (o.active or o.unknown)}
        return booked, inactive

    def reconcile(self, marks: dict[str, float], now: float, every: float = 300) -> dict | None:
        """Compare the ledger with exchange balances; adopt exchange quantities and TMN cash, and log differences."""
        if now - self.reconciled_at < every or self.ex.dry_run:
            return None
        self.reconciled_at = now
        try:
            balances = self.ex.balances()
        except (LiveRefused, WallexAPIError, OSError) as exc:
            return self._refused("balances", "", exc)
        diffs = {}
        tmn = balances.get("TMN", {}).get("total")
        if tmn is not None and abs(tmn - self.cash) > 1000:
            diffs["TMN"] = {"ledger": self.cash, "exchange": tmn}
            self._set("cash", tmn)
        positions = self.positions()
        for sym, pos in positions.items():
            base = self.ex.markets.get(sym, {}).get("base") or sym[:-3]
            held = balances.get(base, {}).get("total")
            if held is not None and abs(held - pos.qty) > max(1e-12, pos.qty * 0.005):
                diffs[sym] = {"ledger": pos.qty, "exchange": held}
                if held * (marks.get(sym) or pos.entry_price) < 1000:
                    self.db.execute("DELETE FROM positions WHERE symbol=?", (sym,))
                else:
                    pos.cost_tmn *= held / pos.qty
                    pos.qty = held
                    self.save_position(pos)
        self.db.commit()
        record = {"ts": now, "type": "reconcile", "differences": diffs,
                  "exchange_assets": {k: v for k, v in balances.items() if v.get("total")}}
        self.log.append(record)
        return record
