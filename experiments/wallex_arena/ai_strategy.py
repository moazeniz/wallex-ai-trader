"""AI-led strategy: Claude decides entries, exits and sizes; the engine enforces hard risk limits.

mode="solo":   the model trades from the dashboard alone.
mode="hybrid": additionally, rule-based lag signals (Wallex ask far below global fair value) are put in front of
               the model, which approves or rejects each one and sets its size. A new signal triggers an early call.
"""
from __future__ import annotations

import gzip
import hashlib
import math
import time
from pathlib import Path

from .ai import MAKER_SYSTEM, Budget, ClaudeTrader, MakerDecision, build_dashboard
from .strategies import FairValueMaker, Strategy

MAX_BUY_FRACTION = 0.35


class AITrader(Strategy):
    def __init__(self, name: str, account_dir: Path, key_path: Path, news, schema=None, system_extra: str = "", **params):
        super().__init__(name, **params)
        extra = {"schema": schema} if schema else {}
        self.trader = ClaudeTrader(params["model"], key_path, Budget(account_dir / "llm-budget.sqlite", params["daily_usd"]),
                                   effort=params.get("effort", "medium"), style=params.get("style", "patient"),
                                   system_extra=system_extra, **extra)
        self.news = news
        self.notes_path = account_dir / "ai_notes.txt"
        if self.notes_path.exists():
            self.trader.notes = self.notes_path.read_text()[:1500]
        self.inputs_dir = account_dir / "ai_inputs"
        self.inputs_dir.mkdir(exist_ok=True)
        self.last_input: dict = {}
        self.review: list[dict] = []
        self.reviewed_at: dict[str, float] = {}

    def lag_signals(self, ctx) -> list[dict]:
        if self.params.get("mode") != "hybrid":
            return []
        out = []
        held = ctx.account.positions()
        for sym in ctx.feed.focus:
            book = ctx.feed.books.get(sym)
            if sym == "USDTTMN" or sym in held or not book or ctx.now - book["ts"] > 15 or book["spread"] > 0.012:
                continue
            if ctx.feed.meta.get(sym, {}).get("vol24_tmn", 0) < self.params.get("min_vol24_tmn", 3e9):
                continue  # thin books gap and stay gapped (IMX lag trades lost repeatedly on 2026-10-22)
            dev = ctx.feed.deviation(sym, "ask")
            if dev is not None and dev < -self.params["threshold"]:
                bars = ctx.feed.bars.get(sym)
                out.append({"symbol": sym, "gap_ask_pct": round(100 * dev, 2), "spread_pct": round(100 * book["spread"], 2),
                            "global_chg_15m_pct": round(100 * (bars.fair_change(15) or 0), 2) if bars else None,
                            "wallex_chg_15m_pct": round(100 * (bars.change(15) or 0), 2) if bars else None,
                            "rule": "buy at market, exit at global fair value; historical mean +0.37%/trade at gap<-1.2%"})
        return out

    def decide(self, ctx) -> list[dict]:
        actions = self._manage_exits(ctx)
        result = self.trader.take()
        if result is not None:
            ctx.log({"type": "ai_decision", "model": self.trader.model, "input": self.last_input, **result})
            if result.get("ok"):
                self.notes_path.write_text(self.trader.notes)
                actions += self._translate(ctx, result["decision"])
        signals = self.lag_signals(ctx)
        fresh = [s for s in signals if ctx.now - self.reviewed_at.get(s["symbol"], 0) > 600]
        due = ctx.now - self.trader.last_call >= self.params["interval_s"]
        early = fresh and ctx.now - self.trader.last_call >= 45
        if (due or early) and not self.trader.busy:
            try:
                news = self.news.recent(30) if self.news else []
            except Exception:
                news = []
            self.review = signals
            for s in signals:
                self.reviewed_at[s["symbol"]] = ctx.now
            extra = {"basket": self.basket_view(ctx)} if self.params.get("universe") else None
            dashboard = build_dashboard(ctx, news, self.trader.notes, signals, extra)
            digest = hashlib.sha256(dashboard.encode()).hexdigest()
            path = self.inputs_dir / f"{int(ctx.now)}-{digest[:12]}.json.gz"
            if self.trader.start(dashboard):
                path.write_bytes(gzip.compress(dashboard.encode()))
                self.last_input = {"sha256": digest, "file": str(path.relative_to(path.parents[2]))}
        return actions

    def basket_view(self, ctx) -> list[dict]:
        """Per allocatable asset: TMN price, spread, multi-horizon TMN returns from hourly candles, daily volatility and
        the account's current weight. Used by macro allocator accounts."""
        out, now = [], ctx.now
        positions = ctx.account.positions()
        for sym in self.params["universe"]:
            book = ctx.feed.books.get(sym)
            row = {"symbol": sym, "mid": book["mid"] if book else None,
                   "spread_pct": round(100 * book["spread"], 3) if book else None}
            try:
                closes = [(c["t"], c["c"]) for c in ctx.feed.candles(sym, "60", days=4) if c["c"] > 0]
            except Exception:
                closes = []
            last = book["mid"] if book else (closes[-1][1] if closes else None)
            for label, hours in (("chg_1h_pct", 1), ("chg_4h_pct", 4), ("chg_24h_pct", 24), ("chg_48h_pct", 48), ("chg_72h_pct", 72)):
                ref = [c for t, c in closes if t <= now - hours * 3600]
                row[label] = round(100 * (last / ref[-1] - 1), 2) if ref and last else None
            if len(closes) > 6:
                rets = [math.log(b / a) for (_, a), (_, b) in zip(closes, closes[1:]) if a > 0 and b > 0]
                mean = sum(rets) / len(rets)
                row["daily_vol_pct"] = round(100 * math.sqrt(24 * sum((r - mean) ** 2 for r in rets) / len(rets)), 2)
            pos = positions.get(sym)
            row["weight_pct"] = round(100 * pos.qty * (book["bid"] if book else pos.entry_price) / ctx.equity, 1) if pos else 0.0
            if sym == "USDTTMN":
                row["market_open"] = not ctx.feed.usdt_halted()
            out.append(row)
        return out

    def _manage_exits(self, ctx) -> list[dict]:
        """Keep a resting take-profit sell for each position (fair-value exit for approved lag trades)."""
        actions = []
        orders = {o.symbol: o for o in ctx.orders if o.side == "SELL"}
        for sym, pos in ctx.account.positions().items():
            book = ctx.feed.books.get(sym)
            if not book:
                continue
            if pos.meta.get("exit") == "fair":
                fair, prem = ctx.feed.fair(sym), ctx.feed.premium(sym)
                if not fair or prem is None:
                    continue
                target = fair * math.exp(prem) * 1.002
            elif pos.target:
                target = pos.target
            else:
                continue
            if target <= book["bid"]:
                if sym in orders:
                    actions.append({"op": "cancel", "order_id": orders[sym].id, "reason": "target_at_bid"})
                actions.append({"op": "market_sell", "symbol": sym, "qty": None, "reason": "take_profit_reached_at_bid",
                                "signal": {"target": target, "bid": book["bid"]}})
                continue
            price = ctx.feed.round_price(sym, max(target, book["bid"] * 1.0001), down=False)
            o = orders.get(sym)
            if o and abs(o.price / price - 1) < 0.001 and abs(o.qty - pos.qty) / pos.qty < 0.01:
                continue
            if o:
                actions.append({"op": "cancel", "order_id": o.id, "reason": "reprice_take_profit"})
            actions.append({"op": "place", "side": "SELL", "symbol": sym, "price": price, "qty": pos.qty, "reason": "take_profit_limit",
                            "signal": {"target": target}})
        return actions

    def _translate(self, ctx, d: dict) -> list[dict]:
        actions, positions, now = [], ctx.account.positions(), ctx.now
        bought = set()  # one entry per market per decision (an action and a signal approval can name the same market)
        tradable = set(self.params["universe"]) if self.params.get("universe") else set(ctx.feed.focus)
        for a in d.get("actions", []):
            sym = str(a.get("symbol", "")).upper()
            if sym not in tradable:
                ctx.log({"type": "ai_action_rejected", "action": a, "why": "symbol_not_tradable"})
                continue
            if a["action"] == "SELL":
                if sym in positions:
                    frac = min(1.0, max(0.0, float(a["size_fraction"])))
                    actions.append({"op": "cancel_symbol", "symbol": sym, "reason": "ai_sell"})
                    actions.append({"op": "market_sell", "symbol": sym, "qty": positions[sym].qty * frac if frac < 0.98 else None,
                                    "reason": "ai: " + a["reason"][:300], "signal": a})
                continue
            if sym in positions or sym in bought or not ctx.entries_allowed:
                continue
            frac = min(MAX_BUY_FRACTION, max(0.0, float(a["size_fraction"])))
            stop = min(0.08, max(0.01, float(a["stop_loss_pct"])))
            take = min(0.15, max(0.005, float(a["take_profit_pct"])))
            hold = min(1440, max(15, int(a["max_hold_minutes"])))
            book = ctx.feed.books.get(sym)
            if not book or frac * ctx.equity < 100_000:
                continue
            bought.add(sym)
            actions.append({"op": "market_buy", "symbol": sym, "budget_tmn": frac * ctx.equity, "stop_pct": stop,
                            "fields": {"stop": book["ask"] * (1 - stop), "target": book["ask"] * (1 + take), "expires_at": now + hold * 60,
                                       "meta": {"source": "ai", "confidence": a.get("confidence")}},
                            "reason": "ai: " + a["reason"][:300], "signal": a})
        for r in d.get("signal_reviews", []):
            sym = str(r.get("symbol", "")).upper()
            if not r.get("approve") or sym not in tradable or sym in positions or sym in bought or not ctx.entries_allowed:
                continue
            bought.add(sym)
            dev = ctx.feed.deviation(sym, "ask")
            if dev is None or dev > -self.params["threshold"] / 2:
                ctx.log({"type": "ai_approval_expired", "symbol": sym, "gap_now": dev})
                continue
            book = ctx.feed.books[sym]
            frac = min(MAX_BUY_FRACTION, max(0.0, float(r["size_fraction"])))
            actions.append({"op": "market_buy", "symbol": sym, "budget_tmn": frac * ctx.equity, "stop_pct": 0.03,
                            "fields": {"stop": book["ask"] * 0.97, "expires_at": now + 120 * 60, "meta": {"source": "ai_approved_lag", "exit": "fair"}},
                            "reason": "ai approved lag signal: " + r["reason"][:300], "signal": {"gap_ask": dev, **r}})
        return actions

    def describe(self) -> dict:
        return {"class": "AITrader", "params": self.params, "llm": self.trader.budget.total()}


class AIMaker(AITrader):
    """Claude manages a market-making engine: it chooses markets, quote width and size, and can flatten inventory.
    The FairValueMaker engine executes the plan every tick."""

    MAX_MARKETS = 8

    def __init__(self, name: str, account_dir: Path, key_path: Path, news, **params):
        super().__init__(name, account_dir, key_path, news, schema=MakerDecision, system_extra=MAKER_SYSTEM, **params)
        self.maker = FairValueMaker(name, **params["maker"])
        self.plan_path = account_dir / "maker_plan.json"
        if self.plan_path.exists():
            import json
            self.maker.plan = json.loads(self.plan_path.read_text())

    def maker_stats(self, ctx) -> dict:
        rows = ctx.account.db.execute("SELECT ts, symbol, side, notional_tmn, pnl_tmn FROM fills WHERE ts > ?", (ctx.now - 6 * 3600,)).fetchall()
        stats = {}
        for ts, sym, side, notional, pnl in rows:
            s = stats.setdefault(sym, {"buys": 0, "sells": 0, "realized_pnl_tmn": 0.0})
            s["buys" if side == "BUY" else "sells"] += 1
            s["realized_pnl_tmn"] = round(s["realized_pnl_tmn"] + (pnl or 0))
        for sym, pos in ctx.account.positions().items():
            stats.setdefault(sym, {})["inventory_held_min"] = round((ctx.now - pos.opened_at) / 60)
        return stats

    def decide(self, ctx) -> list[dict]:
        import json
        actions = []
        result = self.trader.take()
        if result is not None:
            ctx.log({"type": "ai_decision", "model": self.trader.model, "input": self.last_input, **result})
            if result.get("ok"):
                self.notes_path.write_text(self.trader.notes)
                d = result["decision"]
                tradable = set(ctx.feed.focus) - {"USDTTMN"}
                g = self.params.get("guardrail", {})
                lo, hi = g.get("margin_min", 0.002), g.get("margin_max", 0.02)
                plan, dropped = {}, {}
                for item in d.get("plan", [])[: self.MAX_MARKETS]:
                    sym = str(item.get("symbol", "")).upper()
                    if sym not in tradable:
                        continue
                    book = ctx.feed.books.get(sym)
                    if g and book and not g.get("spread_min", 0) <= book["spread"] <= g.get("spread_max", 1):
                        dropped[sym] = round(book["spread"], 5)   # outside the spread band where the engine has earned
                        continue
                    plan[sym] = {"margin": min(hi, max(lo, float(item["margin_pct"]) / 100)),
                                 "size_frac": min(g.get("size_max", 0.25), max(0.03, float(item["size_fraction"])))}
                if dropped:
                    ctx.log({"type": "maker_plan_guardrail", "dropped_spread": dropped})
                self.maker.plan = plan
                self.plan_path.write_text(json.dumps(plan))
                ctx.log({"type": "maker_plan", "plan": plan})
                for sym in d.get("flatten", []):
                    sym = str(sym).upper()
                    if sym in ctx.account.positions():
                        actions.append({"op": "cancel_symbol", "symbol": sym, "reason": "ai_flatten"})
                        actions.append({"op": "market_sell", "symbol": sym, "qty": None, "reason": "ai flatten (risk)", "signal": {}})
        actions += self.maker.decide(ctx)
        if ctx.now - self.trader.last_call >= self.params["interval_s"] and not self.trader.busy:
            try:
                news = self.news.recent(30) if self.news else []
            except Exception:
                news = []
            extra = {"current_maker_plan": self.maker.plan, "maker_stats_last_6h": self.maker_stats(ctx)}
            dashboard = build_dashboard(ctx, news, self.trader.notes, [], extra)
            digest = hashlib.sha256(dashboard.encode()).hexdigest()
            path = self.inputs_dir / f"{int(ctx.now)}-{digest[:12]}.json.gz"
            if self.trader.start(dashboard):
                path.write_bytes(gzip.compress(dashboard.encode()))
                self.last_input = {"sha256": digest, "file": str(path.relative_to(path.parents[2]))}
        return actions
