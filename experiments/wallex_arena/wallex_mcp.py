"""Wallex MCP client (official server https://api.wallex.ir/mcp, Streamable HTTP, JSON-RPC 2.0) for market data.

Read-only by construction:
- no API key is ever sent (the public tools need none);
- only the public market-data tools in ALLOWED_TOOLS can be called. The server also exposes account, order, margin
  and credit tools, including STATE-CHANGING ones, and every one of those is refused before any network call.
Every call goes through the shared retry helper (`wallex.mcp` circuit breaker) and is logged as `mcp.call`. When the
MCP server is unavailable (circuit open, network error, 5xx/429), the same data is fetched from Wallex public REST
(source="rest") so the AI and the dashboard keep working. Each call is also appended to `mcp_calls.jsonl` for the
dashboard.

The high-level helpers (candles / order_book / recent_trades / market_info) return compact, size-capped JSON that is
safe to hand to an LLM. AI_TOOLS describes them as tools for the AI's research step (see ai.py).
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from . import obs
from .resilience import HTTPStatusError, call

DEFAULT_URL = "https://api.wallex.ir/mcp"
REST = "https://api.wallex.ir"
PROTOCOL = "2025-06-18"
ALLOWED_TOOLS = {"wallex_basic_candle_history", "wallex_basic_get_market_depth", "wallex_basic_get_latest_trades",
                 "wallex_basic_get_markets"}
# The server accepts resolutions 1, 60 and D (it answers 15 and 240 with 1-minute / 1-hour bars and rejects 5, 30, W),
# so we fetch the nearest base series and resample to what was asked.
RESOLUTIONS = {"1": (1, "1"), "5": (5, "1"), "15": (15, "1"), "30": (30, "1"), "60": (60, "60"), "240": (240, "60"), "D": (1440, "D")}
MAX_BARS = 120


class MCPToolError(Exception):
    """The tool ran and reported an error (e.g. invalid symbol): not retried, no REST fallback."""


def _http_transport(url: str, body: bytes, headers: dict, timeout: float) -> tuple[int, bytes, dict]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers or {})


def _parse_rpc(raw: bytes, content_type: str) -> dict:
    text = raw.decode(errors="replace")
    if "text/event-stream" in (content_type or "") or text.lstrip().startswith(("event:", "data:")):
        datas = [l[5:].strip() for l in text.splitlines() if l.startswith("data:")]
        text = datas[-1] if datas else "{}"
    return json.loads(text or "{}")


class WallexMCP:
    def __init__(self, url: str | None = None, transport: Callable | None = None, rest: Callable | None = None,
                 log_path: Path | None = None, min_interval: float = 0.25, timeout: float = 20.0, fallback: bool = True):
        self.url = url or os.environ.get("WALLEX_MCP_URL") or DEFAULT_URL
        self.transport = transport or _http_transport
        self.rest = rest                       # callable(path, params) -> json; default: feed.http_json
        log_path = log_path or os.environ.get("WALLEX_MCP_LOG")
        self.log_path = Path(log_path) if log_path else None
        self.min_interval, self.timeout, self.fallback = min_interval, timeout, fallback
        self._lock = threading.Lock()
        self._last = 0.0
        self._id = 0
        self._markets_cache: tuple[float, dict] = (0.0, {})
        self.stats = {"mcp": 0, "rest_fallback": 0, "errors": 0}

    # ---- low level
    def _rpc(self, method: str, params: dict) -> dict:
        with self._lock:                        # the server allows ~5 requests per second
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self._id += 1
            rid = self._id
        body = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": PROTOCOL, "User-Agent": "meowmeow-wallex-mcp/1"}
        if obs.context().get("request_id"):
            headers["X-Request-ID"] = obs.context()["request_id"]
        status, raw, h = self.transport(self.url, body, headers, self.timeout)
        if not 200 <= status < 300:
            raise HTTPStatusError(status, raw[:200].decode(errors="replace"), (h or {}).get("Retry-After"))
        msg = _parse_rpc(raw, (h or {}).get("Content-Type") or (h or {}).get("content-type", ""))
        if "error" in msg:
            raise MCPToolError(str(msg["error"])[:300])
        return msg.get("result") or {}

    def list_tools(self) -> list[dict]:
        res = call("wallex.mcp", lambda: self._rpc("tools/list", {}), max_attempts=2, base=0.5, cap=4.0, endpoint="tools/list")
        return [{"name": t["name"], "description": (t.get("description") or "")[:200], "allowed": t["name"] in ALLOWED_TOOLS}
                for t in res.get("tools", [])]

    def call_tool(self, name: str, args: dict) -> tuple[object, str]:
        """Run one allow-listed tool. Returns (parsed JSON payload, source) where source is "mcp" or "rest"."""
        if name not in ALLOWED_TOOLS:
            obs.log("mcp.call", "warn", "refused: not an allowed read-only tool", tool=name, status="refused")
            raise PermissionError(f"MCP tool {name!r} is not allowed (read-only public market data only)")
        t0 = time.perf_counter()
        status, source, err, data = "ok", "mcp", None, None
        try:
            def once():
                res = self._rpc("tools/call", {"name": name, "arguments": args})
                text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
                payload = json.loads(text) if text.strip().startswith(("{", "[")) else {"text": text}
                if res.get("isError") or (isinstance(payload, dict) and payload.get("success") is False):
                    raise MCPToolError(text[:300])
                return payload
            data = call("wallex.mcp", once, max_attempts=2, base=0.5, cap=4.0, endpoint="tools/call:" + name,
                        retry_if=lambda e: not isinstance(e, MCPToolError) and (not isinstance(e, HTTPStatusError) or e.status in (408, 429) or e.status >= 500))
            self.stats["mcp"] += 1
        except MCPToolError as exc:
            status, err = "tool_error", str(exc)[:200]
            self.stats["errors"] += 1
            self._record(name, args, status, source, t0, err)
            raise
        except (HTTPStatusError, OSError, ValueError) as exc:   # OSError covers timeouts, connection errors, open circuit
            if not self.fallback:
                status, err = "error", f"{type(exc).__name__}: {exc}"[:200]
                self.stats["errors"] += 1
                self._record(name, args, status, source, t0, err)
                raise
            status, source, err = "fallback", "rest", f"{type(exc).__name__}: {exc}"[:200]
            try:
                data = self._rest_equivalent(name, args)
                self.stats["rest_fallback"] += 1
            except Exception as exc2:
                status, err = "error", f"mcp: {err}; rest: {type(exc2).__name__}: {exc2}"[:300]
                self.stats["errors"] += 1
                self._record(name, args, status, source, t0, err)
                raise
        self._record(name, args, status, source, t0, err, data)
        return data, source

    def _rest_equivalent(self, name: str, args: dict):
        get = self.rest
        if get is None:
            from .feed import http_json
            get = lambda path, params: http_json(REST + path, params, timeout=10, retries=1)
        if name == "wallex_basic_candle_history":
            return get("/v1/udf/history", {"symbol": args["symbol"], "resolution": args["resolution"], "from": args["from"], "to": args["to"]})
        if name == "wallex_basic_get_market_depth":
            return get("/v1/depth", {"symbol": args["symbol"]})
        if name == "wallex_basic_get_latest_trades":
            return get("/v1/trades", {"symbol": args["symbol"]})
        if name == "wallex_basic_get_markets":
            return get("/v1/markets", {})
        raise PermissionError(name)

    def _record(self, tool, args, status, source, t0, err=None, data=None):
        ms = int((time.perf_counter() - t0) * 1000)
        size = len(json.dumps(data, default=str)) if data is not None and tool != "wallex_basic_get_markets" else None
        obs.log("mcp.call", "info" if status == "ok" else "warn", f"{tool} {status}", tool=tool, server=self.url, source=source,
                status=status, duration_ms=ms, args={k: v for k, v in args.items()}, error=err, bytes=size)
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                if self.log_path.exists() and self.log_path.stat().st_size > 2_000_000:
                    self.log_path.replace(self.log_path.with_suffix(".old.jsonl"))
                with open(self.log_path, "a") as f:
                    f.write(json.dumps({"ts": time.time(), "tool": tool, "args": args, "status": status, "source": source,
                                        "duration_ms": ms, "error": err, "request_id": obs.context().get("request_id")}) + "\n")
            except OSError:
                pass

    # ---- compact helpers (what the AI and the dashboard use)
    def candles(self, symbol: str, resolution: str = "60", lookback_hours: float = 24) -> dict:
        symbol, resolution = symbol.upper(), str(resolution).upper()
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {sorted(RESOLUTIONS)}")
        minutes, base = RESOLUTIONS[resolution]
        lookback_hours = max(1.0, min(float(lookback_hours), 24 * 60))
        now = int(time.time()) // 60 * 60
        raw, source = self.call_tool("wallex_basic_candle_history", {"symbol": symbol, "resolution": base,
                                                                     "from": now - int(lookback_hours * 3600), "to": now})
        if not isinstance(raw, dict) or raw.get("s") != "ok":
            return {"symbol": symbol, "resolution": resolution, "bars": [], "source": source, "status": (raw or {}).get("s") if isinstance(raw, dict) else "bad"}
        rows = [(int(t), float(o), float(h), float(l), float(c), float(v))
                for t, o, h, l, c, v in zip(raw["t"], raw["o"], raw["h"], raw["l"], raw["c"], raw["v"])]
        bars = resample(rows, minutes * 60)[-MAX_BARS:]
        out = {"symbol": symbol, "resolution": resolution, "source": source, "columns": ["t", "o", "h", "l", "c", "v"], "bars": bars}
        if bars:
            out.update(first_open=bars[0][1], last_close=bars[-1][4], high=max(b[2] for b in bars), low=min(b[3] for b in bars),
                       change_pct=round(100 * (bars[-1][4] / bars[0][1] - 1), 3) if bars[0][1] else None)
        return out

    def order_book(self, symbol: str, levels: int = 10) -> dict:
        symbol, levels = symbol.upper(), max(1, min(int(levels), 20))
        raw, source = self.call_tool("wallex_basic_get_market_depth", {"symbol": symbol})
        r = raw.get("result", raw) if isinstance(raw, dict) else {}
        asks = sorted((float(x["price"]), float(x["quantity"])) for x in r.get("ask", []))[:levels]
        bids = sorted(((float(x["price"]), float(x["quantity"])) for x in r.get("bid", [])), reverse=True)[:levels]
        out = {"symbol": symbol, "source": source, "asks": asks, "bids": bids}
        if asks and bids:
            mid = (asks[0][0] + bids[0][0]) / 2
            out.update(best_bid=bids[0][0], best_ask=asks[0][0], spread_pct=round(100 * (asks[0][0] - bids[0][0]) / mid, 4),
                       bid_tmn_within_1pct=round(sum(p * q for p, q in bids if p >= mid * 0.99)),
                       ask_tmn_within_1pct=round(sum(p * q for p, q in asks if p <= mid * 1.01)))
        return out

    def recent_trades(self, symbol: str, limit: int = 30) -> dict:
        symbol, limit = symbol.upper(), max(1, min(int(limit), 50))
        raw, source = self.call_tool("wallex_basic_get_latest_trades", {"symbol": symbol})
        rows = (raw.get("result", raw) if isinstance(raw, dict) else {}).get("latestTrades", [])[:limit]
        trades = [[t["timestamp"], float(t["price"]), float(t["quantity"]), "buy" if t.get("isBuyOrder") else "sell"] for t in rows]
        buy = sum(p * q for _, p, q, s in trades if s == "buy")
        sell = sum(p * q for _, p, q, s in trades if s == "sell")
        return {"symbol": symbol, "source": source, "columns": ["time", "price", "qty", "taker_side"], "trades": trades,
                "buy_tmn": round(buy), "sell_tmn": round(sell)}

    def market_info(self, symbol: str) -> dict:
        symbol = symbol.upper()
        ts, cache = self._markets_cache
        source = "cache"
        if time.time() - ts > 300 or not cache:
            raw, source = self.call_tool("wallex_basic_get_markets", {})
            cache = _index_markets(raw)
            self._markets_cache = (time.time(), cache)
        m = cache.get(symbol)
        if not m:
            raise ValueError(f"unknown market {symbol}")
        return {"symbol": symbol, "source": source, **m}


def _index_markets(raw) -> dict:
    out = {}
    r = raw.get("result", raw) if isinstance(raw, dict) else {}
    if isinstance(r.get("markets"), list):          # MCP: basic-services markets
        for m in r["markets"]:
            fp = m.get("fair_price") or {}
            out[m["symbol"]] = {"price": _f(m.get("price")), "change_24h_pct": m.get("change_24h"), "change_7d_pct": m.get("change_7D"),
                                "high_24h": _f(m.get("highest_24h_price")), "low_24h": _f(m.get("lowest_24h_price")),
                                "quote_volume_24h_tmn": m.get("quote_volume_24h"), "wallex_fair_bid": _f(fp.get("bid")),
                                "wallex_fair_ask": _f(fp.get("ask")), "zero_fee": m.get("is_zero_fee"), "price_precision": m.get("price_precision")}
    elif isinstance(r.get("symbols"), dict):        # REST fallback: /v1/markets
        for sym, m in r["symbols"].items():
            s = m.get("stats") or {}
            out[sym] = {"price": _f(s.get("lastPrice")), "change_24h_pct": _f(s.get("24h_ch")), "change_7d_pct": _f(s.get("7d_ch")),
                        "high_24h": _f(s.get("24h_highPrice")), "low_24h": _f(s.get("24h_lowPrice")),
                        "quote_volume_24h_tmn": _f(s.get("24h_quoteVolume")), "best_bid": _f(s.get("bidPrice")), "best_ask": _f(s.get("askPrice"))}
    return out


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def resample(rows: list[tuple], step: int) -> list[list]:
    """Aggregate (t, o, h, l, c, v) rows into `step`-second buckets aligned to the epoch."""
    out: dict[int, list] = {}
    for t, o, h, l, c, v in sorted(rows):
        k = t - t % step
        b = out.get(k)
        if b is None:
            out[k] = [k, o, h, l, c, v]
        else:
            b[2], b[3], b[4], b[5] = max(b[2], h), min(b[3], l), c, b[5] + v
    return [[t, o, h, l, c, round(v, 8)] for t, o, h, l, c, v in sorted(out.values())]


# ---- tools offered to the AI during its research step
AI_TOOLS = [
    {"name": "get_candles", "description": "OHLCV candles for one Wallex TMN market from the Wallex MCP server (resolution in minutes: "
     "1, 5, 15, 30, 60, 240 or D). Returns at most 120 bars plus change_pct, high and low.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string", "description": "e.g. SUITMN"},
                      "resolution": {"type": "string", "enum": sorted(RESOLUTIONS)}, "lookback_hours": {"type": "number", "minimum": 1, "maximum": 720}},
                      "required": ["symbol", "resolution", "lookback_hours"], "additionalProperties": False}},
    {"name": "get_order_book", "description": "Current Wallex order book (top levels), spread and TMN depth within 1% of mid.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}, "levels": {"type": "integer", "minimum": 1, "maximum": 20}},
                      "required": ["symbol", "levels"], "additionalProperties": False}},
    {"name": "get_recent_trades", "description": "Latest Wallex trades in one market with taker side and buy/sell TMN totals.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                      "required": ["symbol", "limit"], "additionalProperties": False}},
    {"name": "get_market_info", "description": "Wallex 24h/7d change, 24h high/low, 24h TMN volume and Wallex's own fair price for one market.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"], "additionalProperties": False}},
]


def run_ai_tool(client: WallexMCP, name: str, args: dict, max_chars: int = 6000) -> dict:
    """Dispatch one AI tool call; errors are returned to the model as data, never raised into the decision loop."""
    try:
        if name == "get_candles":
            out = client.candles(args["symbol"], args.get("resolution", "60"), args.get("lookback_hours", 24))
        elif name == "get_order_book":
            out = client.order_book(args["symbol"], args.get("levels", 10))
        elif name == "get_recent_trades":
            out = client.recent_trades(args["symbol"], args.get("limit", 30))
        elif name == "get_market_info":
            out = client.market_info(args["symbol"])
        else:
            return {"error": f"unknown tool {name}"}
    except (KeyError, TypeError, ValueError, PermissionError, MCPToolError, OSError, HTTPStatusError) as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    text = json.dumps(out, default=str)
    while len(text) > max_chars and out.get("bars"):
        out["bars"] = out["bars"][len(out["bars"]) // 2:]
        out["truncated"] = True
        text = json.dumps(out, default=str)
    return out
