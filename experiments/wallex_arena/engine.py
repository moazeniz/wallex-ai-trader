"""Arena engine: one shared live feed, many strategies, each with its own paper account.

Every tick (default 8 s):
  1. poll live Wallex depth + trades for the focus markets and Binance quotes
  2. per account: fill resting maker orders that real trades crossed; enforce stops, time-outs, drawdown
     limits and the final TMN liquidation; ask the strategy for actions; validate and execute them
  3. record equity every minute; write status.json (leaderboard) every 30 s
"""
from __future__ import annotations

import calendar
import json
import math
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from types import SimpleNamespace

from .account import PaperAccount, MIN_ORDER_TMN, DecisionLog
from .feed import MarketFeed, EXCLUDED


@dataclass
class RestingOrder:
    id: str
    symbol: str
    side: str
    price: float
    qty: float
    placed_at: float
    reason: str
    fields: dict = field(default_factory=dict)
    filled: float = 0.0
    checked_until: float = 0.0


@dataclass
class Risk:
    max_position_frac: float = 0.35
    max_exposure: float = 0.95
    risk_per_trade: float = 0.01      # equity lost if a stop is hit (position = risk / (stop + round-trip cost))
    default_stop: float = 0.03
    dd_pause: float = 0.05            # drawdown from peak that pauses new entries for pause_minutes
    pause_minutes: int = 60
    dd_halt: float = 0.10             # drawdown that liquidates and stops new entries for the rest of the run
    fast_guard: bool = True           # pull bids / block buys in a market whose global price is falling fast


class Ctx:
    def __init__(self, engine, name, now):
        self.engine, self.name, self.now = engine, name, now
        self.feed = engine.feed
        self.account: PaperAccount = engine.accounts[name]
        self.orders = list(engine.orders[name].values())
        self.marks = engine.marks()
        self.equity = self.account.equity(self.marks)[0]
        self.liquidate_at = engine.liquidate_at
        self.entries_allowed = engine.entries_allowed(name, now)

    def log(self, record: dict):
        self.account.log.append({"ts": self.now, **record})


EXTRA_FEEDS = (
    ("The Block", "https://www.theblock.co/rss.xml", "theblock.co", "crypto"),
    ("Arzdigital", "https://arzdigital.com/feed/", "arzdigital.com", "crypto"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories", "dowjones.io", "macro"),
    ("Investing.com", "https://www.investing.com/rss/news_1.rss", "investing.com", "macro"),
    ("Donya-e-Eqtesad", "https://donya-e-eqtesad.com/rss", "donya-e-eqtesad.com", "iran_economy"),
    ("ISNA Economy", "https://www.isna.ir/rss/tp/34", "isna.ir", "iran_economy"),
    ("Khabaronline Economy", "https://www.khabaronline.ir/rss/tp/5", "khabaronline.ir", "iran_economy"),
    ("Mehr Economy", "https://www.mehrnews.com/rss/tp/25", "mehrnews.com", "iran_economy"),
)
TOPIC_QUOTA = {"crypto": 8, "iran_economy": 8, "macro": 6, "forex": 3, "politics": 5}


class NewsFeed:
    """Background refresh of the multi-topic RSS store: crypto (EN/FA), central banks and global macro,
    geopolitics, and Iranian economic news (rial, sanctions, oil). Headlines are selected per topic so the
    model always sees a balanced mix; news text is passed to the model as untrusted data."""

    def __init__(self, path: Path):
        from experiments.wallex_agent import news as news_module
        known = {f[0] for f in news_module.FEEDS}
        news_module.FEEDS = tuple(news_module.FEEDS) + tuple(f[:3] for f in EXTRA_FEEDS if f[0] not in known)
        news_module.FEED_TOPICS.update({f[0]: f[3] for f in EXTRA_FEEDS})
        news_module.FEED_TOPICS["IRNA Economy"] = "iran_economy"
        self.store = news_module.NewsStore(path)
        self.refreshed_at = 0.0
        self.status = {}

    def maybe_refresh(self):
        if time.time() - self.refreshed_at > 600:
            self.refreshed_at = time.time()
            threading.Thread(target=self._refresh, daemon=True).start()

    def _refresh(self):
        try:
            result = self.store.refresh()
            self.status = {"status": result.get("status"), "inserted": result.get("inserted"),
                           "feeds_ok": sum(f["status"] == "ok" for f in result.get("feeds", [])), "feeds": len(result.get("feeds", []))}
        except Exception as exc:
            self.status = {"status": "error", "error": type(exc).__name__}

    def recent(self, n: int = 30, hours: float = 24) -> list[dict]:
        """Latest version of each story seen in the last `hours`, newest first, filled per topic quota."""
        now = time.time()
        with self.store._db() as db:
            rows = db.execute("""SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY source,url ORDER BY first_seen_at DESC) r
                                 FROM news_versions WHERE first_seen_at <= ?) WHERE r = 1 AND available_at <= ? AND published_at >= ?
                                 ORDER BY available_at DESC""", (now, now, now - hours * 3600)).fetchall()
        picked, seen, used = [], set(), {}
        for row in rows:
            topic = row["topic"]
            if row["title_key"] in seen or used.get(topic, 0) >= TOPIC_QUOTA.get(topic, 3):
                continue
            seen.add(row["title_key"]); used[topic] = used.get(topic, 0) + 1
            picked.append({k: row[k] for k in ("id", "source", "title", "summary", "published_at", "available_at", "topic")})
            if len(picked) >= n:
                break
        return picked


def challenge_times(config: dict) -> tuple[float | None, float | None, float | None]:
    """(challenge end, flatten start, no-new-entries from) as unix times.

    challenge_end_utc: official end of trading (only realized TMN counts); flatten_minutes (default 60) before it the
    engine cancels everything and sells all coins to TMN; no_new_entries_minutes (default 120) before it no new
    positions are opened. Legacy configs with only liquidate_at_utc flatten at that time and stop entries 1 h before."""
    parse = lambda s: calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))
    if config.get("challenge_end_utc"):
        end = parse(config["challenge_end_utc"])
        return end, end - 60 * float(config.get("flatten_minutes", 60)), end - 60 * float(config.get("no_new_entries_minutes", 120))
    if config.get("liquidate_at_utc"):
        t = parse(config["liquidate_at_utc"])
        return t, t, t - 3600
    return None, None, None


class Engine:
    def __init__(self, config: dict, root: Path, key_path: Path):
        self.config = config
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.feed = MarketFeed(focus_tmn=config.get("focus_markets", 28), websocket=config.get("websocket", False))
        self.tick_seconds = config.get("tick_seconds", 8)
        self.challenge_end, self.liquidate_at, self.entries_stop_at = challenge_times(config)
        self.news = NewsFeed(self.root / "news.sqlite")
        self.accounts: dict[str, PaperAccount] = {}
        self.strategies = {}
        self.risk: dict[str, Risk] = {}
        self.orders: dict[str, dict[str, RestingOrder]] = {}
        self.state: dict[str, dict] = {}
        self.engine_log = DecisionLog(self.root / "engine.jsonl")
        from . import strategies as rules
        from .ai_strategy import AIMaker, AITrader
        for spec in config["accounts"]:
            name = spec["name"]
            if spec.get("live"):
                acct = self._live_account(name, spec, config)
            else:
                acct = PaperAccount(name, self.root / "accounts", spec.get("initial_tmn", 5_000_000))
            self.accounts[name] = acct
            if spec["class"] in ("AITrader", "AIMaker"):
                cls = AITrader if spec["class"] == "AITrader" else AIMaker
                self.strategies[name] = cls(name, acct.dir, key_path, self.news, **spec["params"])
            else:
                self.strategies[name] = getattr(rules, spec["class"])(name, **spec["params"])
            self.risk[name] = Risk(**spec.get("risk", {}))
            self.orders[name] = self._load_orders(acct)
            self.state[name] = json.loads((acct.dir / "state.json").read_text()) if (acct.dir / "state.json").exists() else {"paused_until": 0, "halted": False}
            acct.log.append({"ts": time.time(), "type": "engine_start", "strategy": self.strategies[name].describe(),
                             "risk": asdict(self.risk[name]), "liquidate_at": self.liquidate_at})
        self.last_equity = 0.0
        self.last_status = 0.0
        self.started_at = time.time()

    def _live_account(self, name, spec, config):
        """Real Wallex account (or dry run). Orders still require WALLEX_LIVE_ENABLED=1 and no LIVE_DISABLED file."""
        from .live import KILL_FILE, LiveExecutor, WallexPrivateAPI
        from .live_account import LiveAccount
        self.feed.refresh_meta(force=True)
        dry = spec["live"] == "dry_run"
        api = None if dry else WallexPrivateAPI.from_env_file(Path(config["wallex_env_file"]))
        directory = self.root / "accounts" / name
        directory.mkdir(parents=True, exist_ok=True)
        executor = LiveExecutor(api, self.feed.meta, max_order_tmn=spec.get("max_order_tmn", 2_000_000), dry_run=dry,
                                kill_file=Path(config.get("kill_file", KILL_FILE)), log_path=directory / "live.jsonl",
                                state_path=directory / "live_orders.json")
        return LiveAccount(name, self.root / "accounts", executor, spec.get("initial_tmn", 5_000_000))

    # ---- persistence
    def _load_orders(self, acct) -> dict:
        path = acct.dir / "orders.json"
        if not path.exists():
            return {}
        return {o["id"]: RestingOrder(**o) for o in json.loads(path.read_text())}

    def _save(self, name):
        acct = self.accounts[name]
        (acct.dir / "orders.json").write_text(json.dumps([asdict(o) for o in self.orders[name].values()]))
        (acct.dir / "state.json").write_text(json.dumps(self.state[name]))

    # ---- helpers
    def marks(self) -> dict[str, float]:
        return {s: b["bid"] for s, b in self.feed.books.items()}

    def entries_allowed(self, name, now) -> bool:
        st = self.state[name]
        if st.get("halted") or now < st.get("paused_until", 0):
            return False
        if self.entries_stop_at and now > self.entries_stop_at:
            return False
        return True

    def challenge_phase(self, now) -> str:
        if not self.liquidate_at:
            return "no_end_configured"
        if now >= (self.challenge_end or self.liquidate_at):
            return "ended"
        if now >= self.liquidate_at:
            return "flattening"
        if self.entries_stop_at and now > self.entries_stop_at:
            return "no_new_entries"
        return "trading"

    def held_symbols(self) -> list[str]:
        return sorted({s for a in self.accounts.values() for s in a.positions()})

    # ---- main loop
    def run(self):
        self.feed.refresh_meta(force=True)
        self.feed.poll(extra=self.held_symbols())
        self.feed.seed_bars()
        self.engine_log.append({"ts": time.time(), "type": "seeded", "focus": self.feed.focus, "bars": {s: len(b.minutes) for s, b in self.feed.bars.items()}})
        while True:
            started = time.time()
            try:
                self.tick()
            except Exception as exc:
                self.engine_log.append({"ts": time.time(), "type": "tick_error", "error": type(exc).__name__, "trace": traceback.format_exc()[-1500:]})
            time.sleep(max(1.0, self.tick_seconds - (time.time() - started)))

    def tick(self):
        self.news.maybe_refresh()
        # With the WebSocket streams, books and global prices are live; full REST snapshots run every few ticks as backup.
        self._ticks = getattr(self, "_ticks", 0) + 1
        rest_every = self.config.get("rest_every_ticks", 4 if self.feed.stream else 1)
        rest = self._ticks % rest_every == 1 or rest_every == 1
        self._rest_polls = getattr(self, "_rest_polls", 0) + rest
        self.feed.poll(extra=self.held_symbols(), with_trades=rest and self._rest_polls % 2 == 1, rest=rest)
        now = time.time()
        self._fast_guard(now)
        for name in self.accounts:
            try:
                self._account_tick(name, now)
            except Exception as exc:
                self.accounts[name].log.append({"ts": now, "type": "account_error", "error": type(exc).__name__, "trace": traceback.format_exc()[-1500:]})
        if now - self.last_equity >= 60:
            marks = self.marks()
            for acct in self.accounts.values():
                acct.record_equity(marks, now)
            self.last_equity = now
        if now - self.last_status >= 30:
            self.write_status(now)
            self.last_status = now

    def _account_tick(self, name, now):
        acct, risk, st = self.accounts[name], self.risk[name], self.state[name]
        self._process_maker_fills(name, now)
        marks = self.marks()
        eq = acct.equity(marks)[0]
        peak = max(acct._get("peak"), eq)
        dd = (peak - eq) / peak
        if dd >= risk.dd_halt and not st.get("halted"):
            st["halted"] = True
            acct.log.append({"ts": now, "type": "risk_halt", "drawdown": dd})
            self._liquidate(name, now, "drawdown_halt")
        elif dd >= risk.dd_pause and now >= st.get("paused_until", 0):
            st["paused_until"] = now + risk.pause_minutes * 60
            acct.log.append({"ts": now, "type": "risk_pause", "drawdown": dd, "until": st["paused_until"]})
        if self.liquidate_at and now >= self.liquidate_at:
            self._flatten(name, now)
            self._save(name)
            return
        stop_pending = self._protective_exits(name, now)
        ctx = Ctx(self, name, now)
        for action in self.strategies[name].decide(ctx):
            sym = action.get("symbol") or getattr(self.orders[name].get(action.get("order_id", "")), "symbol", None)
            if sym in stop_pending:
                continue  # a stop exit owns this market until it fills or falls back to a market sell
            self._execute(name, action, ctx)
        self._save(name)

    def _protective_exits(self, name, now) -> set:
        """Stops and time-outs, checked every tick on the live best bid. Returns markets owned by a pending stop exit."""
        acct = self.accounts[name]
        stop_pending = set()
        for sym, pos in acct.positions().items():
            book = self.feed.books.get(sym)
            if not book or now - book["ts"] > 30:
                continue
            why = None
            until = (pos.meta or {}).get("stop_exit_until")
            if until:
                if now < until:
                    stop_pending.add(sym)
                    continue
                why = "stop_loss_limit_expired"
            elif pos.stop and book["bid"] <= pos.stop:
                # Confirm against a fresh book: a single snapshot can show a momentary wick in a thin book.
                fresh = self._fresh_book(sym)
                fresh_bid = fresh["bids"][0][0] if fresh and fresh.get("bids") else None
                if fresh_bid is None or fresh_bid > pos.stop:
                    acct.log.append({"ts": now, "type": "stop_not_confirmed", "symbol": sym, "stop": pos.stop,
                                     "snapshot_bid": book["bid"], "fresh_bid": fresh_bid})
                    continue
                anchor = self._anchor(sym)
                if anchor and fresh_bid < anchor * (1 - self.STOP_OVERSHOOT):
                    # Wallex has overshot the global price: selling into the thin book sells the wick. Rest a limit
                    # sell just under fair value instead; fall back to a market sell after STOP_LIMIT_SECONDS.
                    self._cancel_symbol(name, sym, "stop_loss_limit")
                    price = self.feed.round_price(sym, max(fresh_bid * 1.0005, anchor * (1 - self.STOP_LIMIT_DISCOUNT)), down=False)
                    pos.meta = dict(pos.meta or {}, stop_exit_until=now + self.STOP_LIMIT_SECONDS)
                    acct.save_position(pos)
                    self._execute(name, {"op": "place", "side": "SELL", "symbol": sym, "price": price, "qty": pos.qty,
                                         "reason": "stop_loss_limit_near_fair",
                                         "signal": {"bid": fresh_bid, "fair_anchor": anchor, "stop": pos.stop}},
                                  SimpleNamespace(now=now, entries_allowed=False))
                    stop_pending.add(sym)
                    continue
                why = "stop_loss"
            elif pos.expires_at and now >= pos.expires_at:
                why = "max_hold_time"
            if why:
                self._cancel_symbol(name, sym, why)
                self._market_sell(name, sym, None, why, {"bid": book["bid"], "stop": pos.stop, "expires_at": pos.expires_at})
        return stop_pending

    GUARD_DROP = 0.004             # global price down this much from its 60-second high ...
    GUARD_SECONDS = 60
    GUARD_HOLD_SECONDS = 120       # ... pulls resting bids in that market and blocks new buys for this long

    def _fast_guard(self, now):
        """Adverse-selection guard: when a coin's global price is falling fast, Wallex is about to follow, so resting
        bids there would be filled at stale prices. Cancel them in every guarded account and block buys briefly."""
        self.guard_until = getattr(self, "guard_until", {})
        for sym in {o.symbol for orders in self.orders.values() for o in orders.values() if o.side == "BUY"} | set(self.feed.focus):
            drop = self.feed.global_drop(sym, self.GUARD_SECONDS) if hasattr(self.feed, "global_drop") else None
            if drop is not None and drop < -self.GUARD_DROP:
                first = now >= self.guard_until.get(sym, 0)
                self.guard_until[sym] = now + self.GUARD_HOLD_SECONDS
                for name, orders in self.orders.items():
                    if not self.risk[name].fast_guard:
                        continue
                    for oid, o in list(orders.items()):
                        if o.symbol == sym and o.side == "BUY":
                            self._cancel_order(name, oid, "fast_guard_global_drop")
                            self.accounts[name].log.append({"ts": now, "type": "guard_pull", "symbol": sym, "global_drop": drop})
                if first:
                    self.engine_log.append({"ts": now, "type": "guard", "symbol": sym, "global_drop": drop})

    def _guarded(self, name, sym, now) -> bool:
        return self.risk[name].fast_guard and now < getattr(self, "guard_until", {}).get(sym, 0)

    STOP_OVERSHOOT = 0.01          # bid this far below fair value means Wallex overshot the global move
    STOP_LIMIT_DISCOUNT = 0.005    # stop-limit price: fair value minus this
    STOP_LIMIT_SECONDS = 180

    def _anchor(self, sym):
        fair, prem = self.feed.fair(sym), self.feed.premium(sym)
        return fair * math.exp(prem) if fair and prem is not None else None

    # ---- order handling
    def _process_maker_fills(self, name, now):
        acct = self.accounts[name]
        if getattr(acct, "live", False):
            fills, inactive = acct.sync(self.feed.usdt_tmn())
            for oid in list(self.orders[name]):
                if oid in inactive:
                    del self.orders[name][oid]
            acct.reconcile(self.marks(), now)
            return
        for oid, o in list(self.orders[name].items()):
            book = self.feed.books.get(o.symbol)
            since = max(o.placed_at + 2, o.checked_until)
            trades = self.feed.trades_since(o.symbol, since)
            if o.side == "SELL":
                through = sum(q for ts, p, q, _ in trades if p > o.price)
                crossed = bool(book and now - book["ts"] < 30 and book["bid"] >= o.price)
            else:
                through = sum(q for ts, p, q, _ in trades if p < o.price)
                crossed = bool(book and now - book["ts"] < 30 and book["ask"] <= o.price)
            if trades:
                o.checked_until = max(t[0] for t in trades)
            remaining = o.qty - o.filled
            qty = remaining if crossed else min(remaining, through)
            if qty <= 0:
                continue
            if o.side == "SELL":
                held = acct.positions().get(o.symbol)
                qty = min(qty, held.qty if held else 0)
                if qty <= 0:
                    del self.orders[name][oid]
                    continue
            fill = acct.book_fill(o.symbol, o.side, qty, o.price, self.feed.usdt_tmn(), "maker",
                                  reason=o.reason + (" (book crossed)" if crossed else " (trades printed through)"), now=now,
                                  position_fields=o.fields if o.side == "BUY" else None)
            if fill:
                o.filled += qty
            if not fill or o.filled >= o.qty * 0.999 or (o.qty - o.filled) * o.price < MIN_ORDER_TMN / 2:
                del self.orders[name][oid]

    def _cancel_order(self, name, oid, reason):
        o = self.orders[name].pop(oid, None)
        acct = self.accounts[name]
        if o and getattr(acct, "live", False):
            acct.cancel_resting(oid, self.feed.usdt_tmn(), reason)
        return o

    def _cancel_symbol(self, name, sym, reason):
        for oid, o in list(self.orders[name].items()):
            if o.symbol == sym:
                self._cancel_order(name, oid, reason)
                self.accounts[name].log.append({"ts": time.time(), "type": "cancel", "order": asdict(o), "reason": reason})

    FLATTEN_RETRY_SECONDS = 30

    def _flatten(self, name, now):
        """End-of-challenge settlement in TMN (only realized TMN counts). Every tick inside the flatten window: cancel all
        resting orders, then sell each remaining coin with a bounded, immediate-or-cancel order (live: LIMIT + cancel the
        unfilled rest, so nothing duplicate is left resting), at most once per FLATTEN_RETRY_SECONDS per market until flat.
        Positions worth less than the 100k TMN minimum order cannot be sold and are reported as dust."""
        acct, st = self.accounts[name], self.state[name]
        last = st.setdefault("flatten_last", {})
        cancelled = []
        for oid in list(self.orders[name]):
            o = self._cancel_order(name, oid, "final_tmn_settlement")
            if o:
                cancelled.append({"order_id": oid, "symbol": o.symbol, "side": o.side})
        marks = self.marks()
        sells, waiting, dust = [], [], []
        for sym, pos in acct.positions().items():
            value = pos.qty * (marks.get(sym) or pos.entry_price)
            if value < MIN_ORDER_TMN:
                dust.append({"symbol": sym, "value_tmn": round(value)})
                continue
            if now - last.get(sym, 0) < self.FLATTEN_RETRY_SECONDS:
                waiting.append(sym)
                continue
            last[sym] = now
            fill = self._market_sell(name, sym, None, "final_tmn_settlement", {"phase": "flatten"})
            sells.append({"symbol": sym, "value_tmn": round(value), "result": "filled" if fill else "not_filled"})
        remaining = sorted(acct.positions())
        dust_syms = {d["symbol"] for d in dust}
        flat = not [s for s in remaining if s not in dust_syms]
        if cancelled or sells or flat != st.get("flat"):
            acct.log.append({"ts": now, "type": "flatten", "phase": self.challenge_phase(now), "challenge_end": self.challenge_end,
                             "cancelled": cancelled, "sells": sells, "waiting_retry": waiting, "dust_below_min_order": dust,
                             "remaining": remaining, "flat": flat})
        st["flat"] = flat
        return flat

    def _liquidate(self, name, now, reason):
        for oid in list(self.orders[name]):
            self._cancel_order(name, oid, reason)
        for sym in list(self.accounts[name].positions()):
            self._market_sell(name, sym, None, reason, {})

    def _fresh_book(self, sym):
        try:
            _, asks, bids, ts = self.feed._depth(sym)
            if asks and bids:
                return {"asks": asks, "bids": bids, "bid": bids[0][0], "ask": asks[0][0], "ts": ts}
        except Exception:
            pass
        return self.feed.books.get(sym)

    def _market_sell(self, name, sym, qty, reason, signal):
        acct = self.accounts[name]
        if sym == "USDTTMN" and self.feed.usdt_halted():
            acct.log.append({"ts": time.time(), "type": "deferred", "op": "market_sell", "symbol": sym, "reason": reason,
                             "why": "usdttmn_market_closed"})
            return None
        book = self._fresh_book(sym)
        if not book:
            return None
        fill = acct.market_order(sym, "SELL", book, self.feed.usdt_tmn(), qty=qty, max_impact=0.01, reason=reason)
        acct.log.append({"ts": time.time(), "type": "decision", "op": "market_sell", "symbol": sym, "reason": reason,
                         "signal": signal, "result": "filled" if fill else "not_filled"})
        return fill

    def _execute(self, name, a, ctx):
        acct, risk, now = self.accounts[name], self.risk[name], ctx.now
        op, sym = a["op"], a.get("symbol")
        if op == "cancel":
            o = self._cancel_order(name, a["order_id"], a.get("reason", ""))
            if o:
                acct.log.append({"ts": now, "type": "cancel", "order": asdict(o), "reason": a.get("reason")})
            return
        if op == "cancel_symbol":
            self._cancel_symbol(name, sym, a.get("reason", ""))
            return
        if sym in EXCLUDED or not sym.endswith("TMN"):
            acct.log.append({"ts": now, "type": "rejected", "action": a, "why": "market_not_allowed"})
            return
        if sym == "USDTTMN" and self.feed.usdt_halted():
            acct.log.append({"ts": now, "type": "rejected", "action": {k: v for k, v in a.items() if k != "signal"}, "why": "usdttmn_market_closed"})
            return
        if op == "market_sell":
            self._cancel_symbol(name, sym, "before_market_sell")
            self._market_sell(name, sym, a.get("qty"), a.get("reason", ""), a.get("signal"))
            return
        if op in ("market_buy", "place") and (a.get("side", "BUY") == "BUY") and not ctx.entries_allowed:
            return
        if op in ("market_buy", "place") and a.get("side", "BUY") == "BUY" and self._guarded(name, sym, now):
            return
        marks = self.marks()
        equity, exposure = acct.equity(marks)
        if op == "market_buy" or (op == "place" and a["side"] == "BUY"):
            stop_pct = a.get("stop_pct") or (1 - a.get("fields", {}).get("stop", 0) / (a.get("price") or self.feed.books[sym]["ask"])
                                             if a.get("fields", {}).get("stop") else risk.default_stop)
            committed = sum(o.qty * o.price for o in self.orders[name].values() if o.side == "BUY")
            cap = min(risk.max_position_frac * equity,
                      max(0.0, risk.max_exposure * equity - exposure - committed),
                      risk.risk_per_trade * equity / (max(stop_pct, 0.005) + 0.007),
                      acct.cash - committed)
            budget = min(a.get("budget_tmn") or a["qty"] * a["price"], cap)
            if budget < MIN_ORDER_TMN:
                acct.log.append({"ts": now, "type": "skipped", "action": {k: v for k, v in a.items() if k != "signal"},
                                 "why": "below_min_order_after_risk_caps", "cap_tmn": cap})
                return
            if op == "market_buy":
                book = self._fresh_book(sym)
                fill = acct.market_order(sym, "BUY", book, self.feed.usdt_tmn(), budget_tmn=budget, max_impact=0.004,
                                         reason=a.get("reason", ""), position_fields=self._fields_from_fill(a, book))
                acct.log.append({"ts": now, "type": "decision", "op": "market_buy", "symbol": sym, "budget_tmn": budget,
                                 "reason": a.get("reason"), "signal": a.get("signal"), "result": "filled" if fill else "not_filled"})
                if fill:
                    self._rebase_fields(acct, sym, a, fill["price"], book["bids"][0][0] if book["bids"] else fill["price"], stop_pct)
                return
            qty = budget / a["price"]
        else:  # resting SELL
            pos = acct.positions().get(sym)
            if not pos:
                return
            qty = min(a["qty"], pos.qty)
            if qty * a["price"] < MIN_ORDER_TMN:
                acct.log.append({"ts": now, "type": "skipped", "action": {k: v for k, v in a.items() if k != "signal"},
                                 "why": "sell_below_exchange_minimum"})
                return
        oid = uuid.uuid4().hex[:12]
        if getattr(acct, "live", False):
            oid = acct.place_resting(sym, a["side"], a["price"], qty, a.get("reason", ""), a.get("fields", {}))
            if not oid:
                return
        o = RestingOrder(oid, sym, a["side"], a["price"], qty, now, a.get("reason", ""), a.get("fields", {}))
        self.orders[name][o.id] = o
        acct.log.append({"ts": now, "type": "place", "order": asdict(o), "signal": a.get("signal")})

    @staticmethod
    def _fields_from_fill(a, book):
        f = dict(a.get("fields", {}))
        f.setdefault("meta", {})
        return f

    @staticmethod
    def _rebase_fields(acct, sym, a, price, entry_bid, stop_pct):
        """Re-anchor stop and target to the actual fill. The stop is measured from the best bid at entry (the price a
        stop would sell at), so the spread paid on entry does not count toward the stop distance."""
        pos = acct.positions().get(sym)
        if not pos:
            return
        if stop_pct:
            pos.stop = min(price, entry_bid) * (1 - stop_pct)
        if pos.target and a.get("signal", {}).get("take_profit_pct"):
            pos.target = price * (1 + float(a["signal"]["take_profit_pct"]))
        acct.save_position(pos)

    # ---- reporting
    def _stream_health(self):
        stream = getattr(self.feed, "stream", None)
        if not stream:
            return None
        h = stream.health()
        return {k: v for k, v in h.items() if k not in ("per_symbol_age", "symbol_ages", "ages")}

    def write_status(self, now):
        marks = self.marks()
        board = []
        for name, acct in self.accounts.items():
            s = acct.summary(marks)
            s.update({"strategy": self.strategies[name].describe(), "open_orders": len(self.orders[name]),
                      "halted": self.state[name].get("halted", False), "paused": now < self.state[name].get("paused_until", 0)})
            board.append(s)
        board.sort(key=lambda r: -r["return_pct"])
        first = min(a._get("created_at") for a in self.accounts.values())
        status = {"updated_at": now, "started_at": first, "engine_started_at": self.started_at,
                  "usdt_tmn_halted": self.feed.usdt_halted(), "usdt_rate_ratio": self.feed.rate_ratio,
                  "stream": self._stream_health(),
                  "global_stream": self.feed.global_stream.health() if getattr(self.feed, "global_stream", None) else None,
                  "guarded": sorted(s for s, u in getattr(self, "guard_until", {}).items() if u > now), "liquidate_at": self.liquidate_at, "mode": "paper",
                  "challenge": {"end": self.challenge_end, "flatten_start": self.liquidate_at, "entries_stop": self.entries_stop_at,
                                "phase": self.challenge_phase(now), "flat": {n: self.state[n].get("flat") for n in self.accounts}},
                  "focus": self.feed.focus, "usdt_tmn": self.feed.usdt_tmn(), "feed_errors_recent": list(self.feed.errors)[-10:],
                  "news": self.news.status, "leaderboard": board}
        tmp = self.root / "status.json.tmp"
        tmp.write_text(json.dumps(status, ensure_ascii=False, default=str, indent=1))
        tmp.replace(self.root / "status.json")
