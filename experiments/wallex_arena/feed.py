"""Shared live market feed for all arena strategies.

- Live top-of-book and depth from /v1/depth for a focus universe (the cached /v1/markets quotes matched
  live depth only 9% of the time in our measurement, so they are used for metadata and volume only).
- Recent trades from /v1/trades, used to decide whether a resting (maker) order would have filled.
- Binance book tickers; fair value of a TMN market = Binance mid x Wallex USDTTMN mid.
- Rolling 1-minute bars of Wallex mid and fair value (seeded from history at start-up).
- Optional (websocket=True): a WallexStream (ws_feed.py) pushes depth and trades into the same books/trades in real
  time; REST polling continues as the seed and fallback but never overwrites a book the stream updated more recently.

Public read-only endpoints only.
"""
from __future__ import annotations

import json
import bisect
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

EXCLUDED = set(json.loads((Path(__file__).parent / "excluded_markets.json").read_text()))
WALLEX = "https://api.wallex.ir"
# Global price REST source. Binance by default; WALLEX_GLOBAL_REST can point at a Binance-compatible API (e.g.
# https://api.mexc.com) where Binance is geo-blocked (Iran-hosted runtimes). The Binance stream runs only with Binance.
BINANCE = os.environ.get("WALLEX_GLOBAL_REST", "https://api.binance.com").rstrip("/")
BAR_MINUTES = 2 * 24 * 60
# Liquid coins whose Wallex/Binance price ratio reveals the USDT/TMN rate Wallex traders are using. USDTTMN itself
# stops trading every night 23:30-11:30 Tehran (20:00-08:00 UTC, measured: zero volume) while coin markets continue.
RATE_REFERENCE = ("BTCTMN", "ETHTMN", "PEPETMN", "DOGETMN", "XRPTMN", "SOLTMN", "TRXTMN", "BNBTMN")
USDT_HALT_AFTER_SECONDS = 20 * 60


def _upstream(url: str) -> str:
    host = urllib.parse.urlsplit(url).netloc
    return "wallex.public" if host.endswith("wallex.ir") else "global.rest" if host == urllib.parse.urlsplit(BINANCE).netloc else host


def http_json(url: str, params: dict | None = None, timeout: float = 10, retries: int = 2):
    """Public GET with the shared retry helper (backoff + jitter, circuit breaker per upstream, external.call logs)."""
    from .resilience import HTTPStatusError, call
    endpoint = urllib.parse.urlsplit(url).path
    if params:
        url += "?" + urllib.parse.urlencode(params)

    def once():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise HTTPStatusError(e.code, endpoint, e.headers.get("Retry-After") if e.headers else None) from None
    return call(_upstream(url), once, max_attempts=retries + 1, base=0.5, cap=8.0, endpoint=endpoint,
                retry_if=lambda e: not isinstance(e, HTTPStatusError) or e.status in (408, 429) or e.status >= 500)


def num(value) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else 0.0
    except (TypeError, ValueError):
        return 0.0


class Bars:
    """Rolling 1-minute closes of Wallex mid and fair value, carried forward through gaps."""

    def __init__(self):
        self.minutes: deque[int] = deque(maxlen=BAR_MINUTES)
        self.mid: deque[float] = deque(maxlen=BAR_MINUTES)
        self.fair: deque[float] = deque(maxlen=BAR_MINUTES)

    def update(self, minute: int, mid: float, fair: float):
        if self.minutes and minute < self.minutes[-1]:
            return
        if self.minutes and self.minutes[-1] == minute:
            self.mid[-1] = mid
            if fair:
                self.fair[-1] = fair
            return
        if self.minutes:
            for m in range(self.minutes[-1] + 1, min(minute, self.minutes[-1] + BAR_MINUTES)):
                self.minutes.append(m); self.mid.append(self.mid[-1]); self.fair.append(self.fair[-1])
        self.minutes.append(minute); self.mid.append(mid)
        self.fair.append(fair or (self.fair[-1] if self.fair else 0.0))

    def change(self, k: int) -> float | None:
        """Wallex mid log-return over the last k minutes."""
        if len(self.mid) <= k or self.mid[-1 - k] <= 0:
            return None
        return math.log(self.mid[-1] / self.mid[-1 - k])

    def fair_change(self, k: int) -> float | None:
        if len(self.fair) <= k or self.fair[-1 - k] <= 0 or self.fair[-1] <= 0:
            return None
        return math.log(self.fair[-1] / self.fair[-1 - k])

    def premium(self, window: int = 1440) -> float | None:
        """Median ln(mid/fair) over the trailing window: the market's normal premium to global prices."""
        pairs = [math.log(m / f) for m, f in zip(list(self.mid)[-window:], list(self.fair)[-window:]) if m > 0 and f > 0]
        if len(pairs) < 120:
            return None
        pairs.sort()
        return pairs[len(pairs) // 2]


class MarketFeed:
    def __init__(self, focus_tmn: int = 28, websocket: bool = False):
        self.focus_tmn = focus_tmn
        self.meta: dict[str, dict] = {}
        self.focus: list[str] = []
        self.books: dict[str, dict] = {}
        self.binance: dict[str, tuple[float, float]] = {}
        self.trades: dict[str, deque] = {}
        self._trade_keys: dict[str, set] = {}
        self.bars: dict[str, Bars] = {}
        self.meta_at = 0.0
        self.polled_at = 0.0
        self.errors: deque = deque(maxlen=200)
        self._candles: dict[tuple[str, str], tuple[float, list]] = {}
        self._premium: dict[str, tuple[float, float | None]] = {}
        self.pool = ThreadPoolExecutor(max_workers=6)
        self._tick = 0
        self.implied: deque = deque(maxlen=720)   # (ts, implied USDT/TMN from coin markets)
        self.rate_ratio: float | None = None      # USDTTMN mid / implied rate, learned while USDTTMN trades
        self._lock = threading.Lock()             # guards books/trades writes shared with the WebSocket thread
        self.rest_skipped = 0                     # REST depth results dropped because the stream was newer
        self.stream = None
        self.global_hist: dict[str, deque] = {}       # pair -> (ts, price) for the fast drop guard
        self._binance_rest_ts = 0.0
        self.global_stream = None
        if websocket:
            from .ws_feed import WallexStream
            from .binance_ws import BinanceStream
            self.stream = WallexStream(self)
            self.stream.start()
            if "binance.com" in BINANCE:
                self.global_stream = BinanceStream(self)
                self.global_stream.start()

    def close(self):
        if self.stream:
            self.stream.stop()

    # ---- metadata and universe
    def refresh_meta(self, force: bool = False):
        if not force and time.time() - self.meta_at < 600:
            return
        result = http_json(WALLEX + "/v1/markets")["result"]["symbols"]
        for sym, m in result.items():
            s = m.get("stats") or {}
            self.meta[sym] = {"symbol": sym, "base": m["baseAsset"], "quote": m["quoteAsset"],
                              "price_decimals": int(num(m.get("tickSize"))), "qty_decimals": int(num(m.get("stepSize"))),
                              "min_qty": num(m.get("minQty")), "min_notional": num(m.get("minNotional")),
                              "vol24_tmn": num(s.get("24h_tmnVolume")) or num(s.get("24h_quoteVolume")),
                              "ch24": num(s.get("24h_ch")), "ch7d": num(s.get("7d_ch")), "fa_name": m.get("faName")}
        ranked = sorted((s for s, m in self.meta.items() if m["quote"] == "TMN" and s not in EXCLUDED and s != "USDTTMN"),
                        key=lambda s: -self.meta[s]["vol24_tmn"])
        focus = ranked[: self.focus_tmn]
        for keep in list(self.books):  # never drop a symbol that may hold a position; strategies handle that via extra
            if keep not in focus and keep in self.meta and keep != "USDTTMN":
                focus.append(keep)
        self.focus = sorted(set(focus)) + ["USDTTMN"]
        self.meta_at = time.time()

    def binance_pair(self, symbol: str) -> str | None:
        m = self.meta.get(symbol)
        if not m:
            return None
        pair = m["base"] + "USDT"
        return pair if pair in self.binance else None

    # ---- live polling
    def _depth(self, sym: str):
        r = http_json(WALLEX + "/v1/depth", {"symbol": sym}, timeout=6, retries=1)["result"]
        asks = sorted((num(x["price"]), num(x["quantity"])) for x in r["ask"])
        bids = sorted(((num(x["price"]), num(x["quantity"])) for x in r["bid"]), reverse=True)
        return sym, asks, bids, time.time()

    def _trades(self, sym: str):
        rows = http_json(WALLEX + "/v1/trades", {"symbol": sym}, timeout=6, retries=1)["result"]["latestTrades"]
        out = []
        for t in rows:
            ts = datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00")).timestamp()
            out.append((ts, num(t["price"]), num(t["quantity"]), bool(t["isBuyOrder"])))
        return sym, out

    def on_global_price(self, pair: str, price: float, ts: float):
        """A global (Binance) price update; REST polls and the Binance stream both feed this."""
        if price <= 0:
            return
        with self._lock:
            self.binance[pair] = self._blend(pair, price) if pair in self.binance else (price, price)
            self.global_hist.setdefault(pair, deque(maxlen=400)).append((ts, price))

    def _blend(self, pair, price):
        # keep the last REST spread around the streamed last price
        bid, ask = self.binance[pair]
        half = (ask - bid) / 2 if ask > bid > 0 else 0.0
        return (price - half, price + half)

    def fair_smooth(self, symbol: str, seconds: float = 30) -> float | None:
        """Fair value from the median global price of the last `seconds`: quoting uses this so per-second jitter in the
        Binance stream does not move resting orders; the drop guard keeps using raw prices."""
        pair = self.binance_pair(symbol)
        hist = self.global_hist.get(pair) if pair else None
        u = self.usdt_tmn()
        if not hist or u <= 0:
            return self.fair(symbol)
        now = time.time()
        window = sorted(p for ts, p in list(hist) if now - ts <= seconds)
        if not window:
            return self.fair(symbol)
        return window[len(window) // 2] * u

    def global_drop(self, symbol: str, seconds: float = 60) -> float | None:
        """Global price now vs its highest point in the last `seconds` (negative = falling)."""
        pair = self.binance_pair(symbol)
        hist = self.global_hist.get(pair) if pair else None
        if not hist:
            return None
        now = time.time()
        window = [p for ts, p in list(hist) if now - ts <= seconds]
        if len(window) < 2:
            return None
        return window[-1] / max(window) - 1

    def poll(self, extra: list[str] | None = None, with_trades: bool = True, rest: bool = True):
        self.refresh_meta()
        symbols = sorted(set(self.focus) | set(extra or []))
        if self.stream:
            self.stream.set_symbols(symbols)
        if self.global_stream:
            self.global_stream.set_pairs([p for p in (self.binance_pair(s) for s in symbols) if p])
        if not rest:
            now = time.time()
            self._update_implied_rate(now)
            minute = int(now // 60)
            for sym, b in list(self.books.items()):
                if sym != "USDTTMN" and now - b["ts"] < 30:
                    self.bars.setdefault(sym, Bars()).update(minute, b["mid"], self.fair(sym) or 0.0)
            self.polled_at = now
            self._tick += 1
            return

        def safe(fn, s):
            try:
                return fn(s)
            except Exception as exc:
                self.errors.append((time.time(), fn.__name__, s, type(exc).__name__))
                return None

        def timed_depth(s):
            started = time.time()
            item = safe(self._depth, s)
            return item and (*item, started)

        books = list(self.pool.map(timed_depth, symbols))
        try:
            now = time.time()
            for t in http_json(BINANCE + "/api/v3/ticker/bookTicker", timeout=6, retries=1):
                if t["symbol"].endswith("USDT"):
                    bid, ask = num(t["bidPrice"]), num(t["askPrice"])
                    with self._lock:
                        self.binance[t["symbol"]] = (bid, ask)
                        if bid > 0 and (not self.global_stream or t["symbol"] not in self.global_stream.pairs):
                            self.global_hist.setdefault(t["symbol"], deque(maxlen=400)).append((now, (bid + ask) / 2))
            self._binance_rest_ts = now
        except Exception as exc:
            self.errors.append((time.time(), "binance", "", type(exc).__name__))
        for item in books:
            if not item:
                continue
            sym, asks, bids, ts, started = item
            if asks and bids and 0 < bids[0][0] < asks[0][0]:
                mid = (asks[0][0] + bids[0][0]) / 2
                with self._lock:
                    cur = self.books.get(sym)
                    # both sides confirmed by the stream after this request was sent: the stream is newer
                    if self.stream and cur and cur.get("src") == "ws" and cur["ts"] >= started:
                        self.rest_skipped += 1
                        continue
                    self.books[sym] = {"symbol": sym, "bid": bids[0][0], "ask": asks[0][0], "mid": mid,
                                       "spread": (asks[0][0] - bids[0][0]) / mid, "asks": asks[:40], "bids": bids[:40], "ts": ts}
        if with_trades:
            for item in self.pool.map(lambda s: safe(self._trades, s), symbols):
                if not item:
                    continue
                sym, rows = item
                self._add_trades(sym, rows, keep_sorted=self.stream is not None)
        now = time.time()
        self._update_implied_rate(now)
        minute = int(now // 60)
        for sym, b in self.books.items():
            if sym != "USDTTMN" and now - b["ts"] < 30:
                self.bars.setdefault(sym, Bars()).update(minute, b["mid"], self.fair(sym) or 0.0)
        self.polled_at = now
        self._tick += 1

    def _add_trades(self, sym: str, rows, keep_sorted: bool = False):
        """Append unseen (ts, price, qty, buyer_is_taker) rows. keep_sorted inserts late rows (e.g. a REST row older
        than trades the stream already delivered) in ts order; the deque is replaced, never mutated out of order."""
        with self._lock:
            q = self.trades.setdefault(sym, deque(maxlen=400)); keys = self._trade_keys.setdefault(sym, set())
            for row in sorted(rows):
                if row in keys:
                    continue
                keys.add(row)
                if keep_sorted and q and row[0] < q[-1][0]:
                    items = list(q)
                    bisect.insort(items, row)
                    q = self.trades[sym] = deque(items, maxlen=400)
                else:
                    q.append(row)
            if len(keys) > 2000:
                self._trade_keys[sym] = set(q)

    # ---- derived values
    def _update_implied_rate(self, now: float):
        ratios = []
        for sym in RATE_REFERENCE:
            book, pair = self.books.get(sym), self.binance_pair(sym)
            if book and pair and now - book["ts"] < 30 and self.binance[pair][0] > 0:
                ratios.append(book["mid"] / ((self.binance[pair][0] + self.binance[pair][1]) / 2))
        if len(ratios) >= 4:
            ratios.sort()
            implied = ratios[len(ratios) // 2]
            self.implied.append((now, implied))
            u = self.books.get("USDTTMN")
            if u and not self.usdt_halted() and now - u["ts"] < 30:
                r = u["mid"] / implied
                self.rate_ratio = r if self.rate_ratio is None else 0.98 * self.rate_ratio + 0.02 * r

    def usdt_halted(self) -> bool:
        """True when USDTTMN has printed no trade for 20 minutes (nightly halt or exchange pause)."""
        trades = self.trades.get("USDTTMN")
        if not trades:
            return False
        return time.time() - trades[-1][0] > USDT_HALT_AFTER_SECONDS

    def usdt_tmn(self) -> float:
        """USDT/TMN for fair values: the live USDTTMN mid while it trades; during the halt, the 60-minute median of
        the rate implied by liquid coin markets, scaled by the USDTTMN/implied ratio learned while it traded."""
        b = self.books.get("USDTTMN")
        if b and not self.usdt_halted() and time.time() - b["ts"] < 60:
            return b["mid"]
        recent = sorted(v for ts, v in self.implied if time.time() - ts < 3600)
        if recent and self.rate_ratio:
            return recent[len(recent) // 2] * self.rate_ratio
        return b["mid"] if b else 0.0

    def fair(self, symbol: str) -> float | None:
        pair = self.binance_pair(symbol)
        u = self.usdt_tmn()
        if not pair or u <= 0:
            return None
        bid, ask = self.binance[pair]
        return (bid + ask) / 2 * u if bid > 0 else None

    def premium(self, symbol: str) -> float | None:
        """Cached (60 s) trailing-24h median of ln(mid/fair)."""
        cached = self._premium.get(symbol)
        if cached and time.time() - cached[0] < 60:
            return cached[1]
        bars = self.bars.get(symbol)
        value = bars.premium() if bars else None
        self._premium[symbol] = (time.time(), value)
        return value

    def deviation(self, symbol: str, side: str = "mid") -> float | None:
        """ln(price/fair) minus the normal premium. side='ask' measures what a buyer pays now."""
        book, fair, prem = self.books.get(symbol), self.fair(symbol), self.premium(symbol)
        if not book or not fair or prem is None:
            return None
        return math.log(book[side] / fair) - prem

    def trades_since(self, symbol: str, since: float) -> list[tuple]:
        return [t for t in list(self.trades.get(symbol, ())) if t[0] > since]  # list(): snapshot vs the WS thread

    def round_price(self, symbol: str, price: float, down: bool) -> float:
        d = self.meta.get(symbol, {}).get("price_decimals", 0)
        f = 10 ** d
        return (math.floor(price * f) if down else math.ceil(price * f)) / f

    def candles(self, symbol: str, resolution: str = "60", days: float = 7, ttl: float = 600) -> list[dict]:
        key = (symbol, resolution)
        cached = self._candles.get(key)
        if cached and time.time() - cached[0] < ttl:
            return cached[1]
        end = int(time.time())
        data = http_json(WALLEX + "/v1/udf/history", {"symbol": symbol, "resolution": resolution, "from": int(end - days * 86400), "to": end})
        rows = [] if data.get("s") != "ok" else [
            {"t": int(t), "o": num(o), "h": num(h), "l": num(l), "c": num(c), "v": num(v)}
            for t, o, h, l, c, v in zip(*(data[k] for k in "tohlcv"))]
        self._candles[key] = (time.time(), rows)
        return rows

    def seed_bars(self, hours: int = 26):
        """Backfill 1-minute bars from Wallex trade closes x Binance klines so premiums exist at start-up."""
        end = int(time.time()) // 60 * 60
        start = end - hours * 3600

        def wallex_closes(sym):
            data = http_json(WALLEX + "/v1/udf/history", {"symbol": sym, "resolution": "1", "from": start, "to": end - 1}, timeout=20)
            return {int(t) // 60: num(c) for t, c in zip(data.get("t", []), data.get("c", []))} if data.get("s") == "ok" else {}

        def binance_closes(pair):
            out, cursor = {}, start * 1000
            while cursor < end * 1000:
                rows = http_json(BINANCE + "/api/v3/klines", {"symbol": pair, "interval": "1m", "startTime": cursor,
                                                              "endTime": end * 1000 - 1, "limit": 1000}, timeout=20)
                if not rows:
                    break
                out.update({int(r[0]) // 60000: num(r[4]) for r in rows})
                cursor = int(rows[-1][0]) + 60000
            return out

        usdt = wallex_closes("USDTTMN")

        def job(sym):
            try:
                w = wallex_closes(sym)
                pair = self.binance_pair(sym)
                b = binance_closes(pair) if pair else {}
                bars, lw, lu, lb = Bars(), None, None, None
                for minute in range(start // 60, end // 60):
                    lw, lu, lb = w.get(minute, lw), usdt.get(minute, lu), b.get(minute, lb)
                    if lw:
                        bars.update(minute, lw, lb * lu if (lb and lu) else 0.0)
                return sym, bars
            except Exception as exc:
                self.errors.append((time.time(), "seed", sym, type(exc).__name__))
                return sym, None

        for sym, bars in self.pool.map(job, [s for s in self.focus if s != "USDTTMN"]):
            if bars and bars.minutes:
                self.bars[sym] = bars
