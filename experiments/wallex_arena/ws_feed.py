"""Real-time Wallex market data over the v2 (Centrifugo) WebSocket, written into a MarketFeed.

Channels per symbol: orderbook:<SYM>@buyDepth, orderbook:<SYM>@sellDepth, public:<SYM>@trade.
Observed live (2026-10-22):
- frames may carry several JSON messages separated by newlines; the server pings with "{}" (reply "{}");
- depth: {"push":{"channel":"orderbook:BTCTMN@buyDepth","pub":{"data":{"message":[{"price":2.2e10,"quantity":0.004,
  "sum":9.9e7}, ...]}}}} -- a full snapshot of one side, best first, up to 50 levels, numbers;
- trade: {"push":{"channel":"public:XRPTMN@trade","pub":{"data":{"isBuyOrder":true,"price":"358001","quantity":"276.4",
  "timestamp":"2026-10-22T12:15:28Z"}}}} -- same fields as REST /v1/trades (second resolution, UTC);
- more than 64 channels on one connection -> {"id":N,"error":{"code":106,"message":"limit exceeded"}}.

Books: only symbols that already have a book (seeded by REST) are updated; the dict object is replaced, never mutated,
and no new keys are added from this thread, so readers iterating feed.books are safe. A side message replaces that
side only; the other side is kept with its own timestamp. book["ts"] is the older of the two side receipt times, so a
book never looks fresher than its stalest side, and REST overwrites any book whose sides were not both confirmed by
the stream after the REST request started. Trades carry the exchange timestamp (as REST does) so the dedup key
(ts, price, qty, buyer_is_taker) matches REST rows exactly.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import threading
import time
from collections import deque
from datetime import datetime

try:  # websockets >= 13
    from websockets.asyncio.client import connect as _ws_connect
    _HEADERS_KW = "additional_headers"
except ImportError:  # pragma: no cover - legacy websockets
    from websockets import connect as _ws_connect
    _HEADERS_KW = "extra_headers"

WS_URL = "wss://api.wallex.ir/connection/websocket"
MAX_CHANNELS_PER_CONN = 64
SYMBOLS_PER_CONN = MAX_CHANNELS_PER_CONN // 3  # buyDepth, sellDepth, trade
MAX_LEVELS = 40


def _num(value) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def channels_for(symbol: str) -> list[str]:
    return [f"orderbook:{symbol}@buyDepth", f"orderbook:{symbol}@sellDepth", f"public:{symbol}@trade"]


def split_channel(channel: str) -> tuple[str, str] | None:
    """'orderbook:BTCTMN@buyDepth' -> ('BTCTMN', 'buyDepth')."""
    try:
        sym, kind = channel.split(":", 1)[1].split("@", 1)
    except (IndexError, ValueError):
        return None
    return sym, kind


def parse_depth(data, bids: bool) -> list[tuple[float, float]]:
    """One side of the book, best first: bids descending, asks ascending."""
    rows = data.get("message") if isinstance(data, dict) else data
    levels = []
    for row in rows or ():
        if isinstance(row, dict):
            p, q = _num(row.get("price")), _num(row.get("quantity"))
        else:
            p, q = _num(row[0]), _num(row[1])
        if p > 0 and q > 0:
            levels.append((p, q))
    levels.sort(reverse=bids)
    return levels[:MAX_LEVELS]


def parse_trades(data) -> list[tuple[float, float, float, bool]]:
    """(exchange ts, price, qty, buyer_is_taker) exactly as MarketFeed._trades builds REST rows."""
    items = data if isinstance(data, list) else [data]
    out = []
    for t in items:
        if not isinstance(t, dict) or "timestamp" not in t:
            continue
        ts = datetime.fromisoformat(str(t["timestamp"]).replace("Z", "+00:00")).timestamp()
        out.append((ts, _num(t.get("price")), _num(t.get("quantity")), bool(t.get("isBuyOrder"))))
    return out


def apply_side(feed, symbol: str, side: str, levels: list[tuple[float, float]], ts: float) -> bool:
    """Replace one side of feed.books[symbol]; keep the other side. Returns True when a book was published."""
    if not levels:
        return False
    other = "asks" if side == "bids" else "bids"
    with feed._lock:
        cur = feed.books.get(symbol)
        if cur is None:  # first book comes from REST; never add keys from this thread
            return False
        other_ts = cur.get(other + "_ts", cur.get("ts", 0.0))
        best = levels[0][0]
        # the new side is authoritative: opposite levels that would cross it cannot still be resting
        if side == "bids":
            kept = [lv for lv in cur.get("asks", ()) if lv[0] > best]
            bids, asks, bids_ts, asks_ts = levels, kept, ts, other_ts
        else:
            kept = [lv for lv in cur.get("bids", ()) if lv[0] < best]
            bids, asks, bids_ts, asks_ts = kept, levels, other_ts, ts
        if not bids or not asks or not 0 < bids[0][0] < asks[0][0]:
            return False
        mid = (asks[0][0] + bids[0][0]) / 2
        feed.books[symbol] = {"symbol": symbol, "bid": bids[0][0], "ask": asks[0][0], "mid": mid,
                              "spread": (asks[0][0] - bids[0][0]) / mid, "asks": list(asks[:MAX_LEVELS]),
                              "bids": list(bids[:MAX_LEVELS]), "ts": min(bids_ts, asks_ts),
                              "bids_ts": bids_ts, "asks_ts": asks_ts, "src": "ws"}
        return True


class _Conn:
    def __init__(self, idx: int):
        self.idx = idx
        self.symbols: set[str] = set()
        self.subscribed: set[str] = set()
        self.pending: dict[int, tuple[str, str]] = {}
        self.ws = None
        self.task = None
        self.connected = False
        self.closed = False
        self.connected_at = 0.0
        self.last_msg = 0.0
        self._id = 0

    def next_id(self) -> int:
        self._id += 1
        return self._id


class WallexStream:
    """Background-thread WebSocket client that keeps feed.books / feed.trades current for a symbol set."""

    def __init__(self, feed, url: str = WS_URL, symbols_per_conn: int = SYMBOLS_PER_CONN, idle_timeout: float = 60.0):
        self.feed = feed
        self.url = url
        self.symbols_per_conn = max(1, min(symbols_per_conn, SYMBOLS_PER_CONN))
        self.idle_timeout = idle_timeout
        self.conns: list[_Conn] = []
        self._desired: set[str] = set()
        self._guard = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._next_conn = 0
        self.last_msg: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.messages = 0
        self.book_updates = 0
        self.trade_rows = 0
        self.reconnects = 0
        self.parse_errors = 0
        self.errors: deque = deque(maxlen=100)

    # ---- public API (any thread)
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stopping = False
        self._thread = threading.Thread(target=self._thread_main, name="wallex-ws", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        self._stopping = True
        loop = self._loop
        if loop and loop.is_running() and self._wake is not None:
            loop.call_soon_threadsafe(self._wake.set)
        if self._thread:
            self._thread.join(timeout)

    def set_symbols(self, symbols) -> bool:
        """Desired symbol set; channels are (un)subscribed on live connections. Returns True if it changed."""
        new = {s for s in symbols if s}
        with self._guard:
            if new == self._desired:
                return False
            self._desired = new
        loop = self._loop
        if loop and loop.is_running() and self._wake is not None:
            loop.call_soon_threadsafe(self._wake.set)
        return True

    def health(self) -> dict:
        now = time.time()
        with self._guard:
            desired = sorted(self._desired)
        conns = list(self.conns)
        ages = sorted(now - self.last_msg[s] for s in desired if s in self.last_msg)
        return {
            "connections": len(conns),
            "connected": sum(1 for c in conns if c.connected),
            "channels": sum(len(c.subscribed) for c in conns),
            "symbols": len(desired),
            "silent_symbols": [s for s in desired if s not in self.last_msg],
            "last_message_age": round(now - max((c.last_msg for c in conns), default=0.0), 3) if conns else None,
            "symbol_age_median": round(ages[len(ages) // 2], 3) if ages else None,
            "symbol_age_max": round(ages[-1], 3) if ages else None,
            "symbol_ages": {s: round(now - self.last_msg[s], 3) for s in desired if s in self.last_msg},
            "reconnects": self.reconnects,
            "messages": self.messages,
            "book_updates": self.book_updates,
            "trade_rows": self.trade_rows,
            "parse_errors": self.parse_errors,
            "recent_errors": list(self.errors)[-5:],
        }

    # ---- message handling (pure apart from writing into the feed; used directly by tests)
    def on_frame(self, raw, conn: _Conn | None = None, recv_ts: float | None = None) -> int:
        """Handle one WebSocket frame (possibly several newline-separated messages). Returns pings to answer."""
        recv_ts = recv_ts or time.time()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        pings = 0
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.parse_errors += 1
                continue
            if not msg:
                pings += 1
                if conn:
                    conn.last_msg = recv_ts
                continue
            try:
                self.on_message(msg, conn, recv_ts)
            except Exception as exc:  # a malformed message must never kill the connection
                self.parse_errors += 1
                self.errors.append((recv_ts, "parse", type(exc).__name__, str(exc)[:120]))
        return pings

    def on_message(self, msg: dict, conn: _Conn | None, recv_ts: float):
        if conn:
            conn.last_msg = recv_ts
        push = msg.get("push")
        if push is None:
            self._on_reply(msg, conn, recv_ts)
            return
        pub = push.get("pub")
        parts = split_channel(push.get("channel", ""))
        if not pub or not parts:
            return
        sym, kind = parts
        data = pub.get("data")
        self.messages += 1
        self.last_msg[sym] = recv_ts
        self.counts[sym] = self.counts.get(sym, 0) + 1
        if kind == "buyDepth":
            self.book_updates += apply_side(self.feed, sym, "bids", parse_depth(data, True), recv_ts)
        elif kind == "sellDepth":
            self.book_updates += apply_side(self.feed, sym, "asks", parse_depth(data, False), recv_ts)
        elif kind == "trade":
            rows = parse_trades(data)
            if rows:
                self.trade_rows += len(rows)
                self.feed._add_trades(sym, rows, keep_sorted=True)

    def _on_reply(self, msg: dict, conn: _Conn | None, recv_ts: float):
        if "disconnect" in msg:
            self.errors.append((recv_ts, conn.idx if conn else None, "disconnect", str(msg["disconnect"])[:120]))
            return
        if conn is None or "id" not in msg:
            return
        op = conn.pending.pop(msg["id"], None)
        if "error" in msg:
            err = msg["error"] or {}
            self.errors.append((recv_ts, conn.idx, err.get("code"), f"{err.get('message')} {op[1] if op else ''}".strip()))
            if op and op[0] == "subscribe":
                conn.subscribed.discard(op[1])
            return
        if op and op[0] == "subscribe":
            conn.subscribed.add(op[1])
        elif op and op[0] == "unsubscribe":
            conn.subscribed.discard(op[1])

    # ---- symbol -> connection assignment
    def _assign(self) -> tuple[list[tuple[_Conn, str]], list[tuple[_Conn, str]]]:
        """Move conn.symbols towards the desired set. Returns (added, removed) (conn, symbol) pairs."""
        with self._guard:
            desired = set(self._desired)
        added, removed = [], []
        for c in self.conns:
            for s in sorted(c.symbols - desired):
                c.symbols.discard(s)
                removed.append((c, s))
        current = {s for c in self.conns for s in c.symbols}
        for s in sorted(desired - current):
            c = next((c for c in self.conns if not c.closed and len(c.symbols) < self.symbols_per_conn), None)
            if c is None:
                c = _Conn(self._next_conn)
                self._next_conn += 1
                self.conns.append(c)
            c.symbols.add(s)
            added.append((c, s))
        return added, removed

    async def _reconcile(self):
        added, removed = self._assign()
        for c, s in removed:
            if c.connected:
                await self._send_channels(c, s, "unsubscribe")
        for c, s in added:
            if c.connected:
                await self._send_channels(c, s, "subscribe")
        for c in list(self.conns):
            if not c.symbols:
                c.closed = True
                self.conns.remove(c)
                if c.ws is not None:
                    try:
                        await c.ws.close()
                    except Exception:
                        pass
                if c.task:
                    c.task.cancel()
            elif c.task is None and self._loop is not None:
                c.task = asyncio.ensure_future(self._run_conn(c))

    async def _send_channels(self, c: _Conn, symbol: str, op: str):
        for ch in channels_for(symbol):
            i = c.next_id()
            c.pending[i] = (op, ch)
            if op == "unsubscribe":
                c.subscribed.discard(ch)
            try:
                await c.ws.send(json.dumps({"id": i, op: {"channel": ch}}))
            except Exception as exc:
                self.errors.append((time.time(), c.idx, op, type(exc).__name__))
                return

    # ---- connection loop
    async def _run_conn(self, c: _Conn):
        backoff = 1.0
        while not c.closed and not self._stopping:
            started = time.time()
            try:
                async with _ws_connect(self.url, open_timeout=15, close_timeout=2, max_size=2 ** 22,
                                       **{_HEADERS_KW: {"User-Agent": "Mozilla/5.0"}}) as ws:
                    c.ws, c.pending, c.subscribed, c._id = ws, {}, set(), 0
                    await ws.send(json.dumps({"id": c.next_id(), "connect": {}}))
                    c.connected, c.connected_at = True, time.time()
                    for s in sorted(c.symbols):
                        if s in c.symbols:  # may be removed while we await
                            await self._send_channels(c, s, "subscribe")
                    while not c.closed and not self._stopping:
                        raw = await asyncio.wait_for(ws.recv(), timeout=self.idle_timeout)
                        for _ in range(self.on_frame(raw, c)):
                            await ws.send("{}")
            except asyncio.CancelledError:
                break
            except Exception as exc:  # disconnects, handshake 5xx, idle timeouts
                self.errors.append((time.time(), c.idx, "conn", f"{type(exc).__name__}: {str(exc)[:100]}"))
            finally:
                c.connected, c.ws = False, None
                c.subscribed = set()
            if c.closed or self._stopping:
                break
            self.reconnects += 1
            if time.time() - started > 60:
                backoff = 1.0
            await asyncio.sleep(backoff * (0.75 + random.random() / 2))
            backoff = min(backoff * 2, 30.0)

    def _thread_main(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._main())
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            finally:
                self._loop = None
                loop.close()

    async def _main(self):
        self._wake = asyncio.Event()
        while not self._stopping:
            try:
                await self._reconcile()
            except Exception as exc:
                self.errors.append((time.time(), None, "reconcile", type(exc).__name__))
            await self._wake.wait()
            self._wake.clear()
        for c in list(self.conns):
            c.closed = True
            if c.ws is not None:
                try:
                    await c.ws.close()
                except Exception:
                    pass
            if c.task:
                c.task.cancel()
