"""Live Wallex order client for the arena. DISABLED unless a human turns it on.

Two layers:
  WallexPrivateAPI  thin authenticated REST client (stdlib urllib, injectable transport)
  LiveExecutor      paper-account-like intents (market_buy / market_sell / place_limit / cancel / sync)
                    with hard safety checks: kill switch, market whitelist, 100k TMN minimum, max notional,
                    price bounds, decimal rounding, and a dry-run mode that never touches the network.

Order placement is enabled only when env WALLEX_LIVE_ENABLED == "1" AND the file
/var/lib/wallex-arena/LIVE_DISABLED does not exist. Docs used: research/live-api/ (see LIVE.md).
"""
from __future__ import annotations

import http.client
import json
import os
import secrets
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Callable

from .account import MIN_ORDER_TMN, TAKER_FEE, DecisionLog
from .feed import EXCLUDED, num
from . import obs
from .resilience import call

BASE_URL = "https://api.wallex.ir"
KILL_FILE = Path(os.environ.get("WALLEX_KILL_FILE") or "/var/lib/wallex-arena/LIVE_DISABLED")
ENABLE_ENV = "WALLEX_LIVE_ENABLED"
# Header name as declared in the trade-credit swagger (trade-credit-swagger.yaml: `sub-account-client-id`).
# The prose page trade-credit-connect-wallet spells it `sub_account_client_id`; HTTP names are case-insensitive
# but '_' vs '-' is not, so it is configurable.
SUB_ACCOUNT_HEADER = "sub-account-client-id"
ORDER_RATE = (20, 10.0)  # spot-create-order: at most 20 create requests per 10 s
TERMINAL = {"FILLED", "CANCELED", "CANCELLED", "FINISHED", "EXPIRED", "REJECTED"}

# transport(method, url, headers, body, timeout) -> (http_status, response_bytes).
# Raises OSError (incl. socket.timeout / URLError) when no HTTP response was received.
Transport = Callable[[str, str, dict, "bytes | None", float], "tuple[int, bytes]"]


class LiveRefused(Exception):
    """An order was refused locally (kill switch, whitelist, limits); nothing was sent."""


class WallexAPIError(Exception):
    """Wallex answered with an HTTP error or success=false."""

    def __init__(self, status: int, message: str, code=None, result=None):
        super().__init__(f"HTTP {status}: {message}")
        self.status, self.message, self.code, self.result = status, message, code, result


class AmbiguousOrder(Exception):
    """A create-order POST got no definitive answer and a lookup by client_id could not settle it."""

    def __init__(self, client_id: str, cause: str):
        super().__init__(f"order {client_id} state unknown: {cause}")
        self.client_id, self.cause = client_id, cause


def urllib_transport(method: str, url: str, headers: dict, body: bytes | None, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:  # an HTTP answer, even if an error one
        return exc.code, exc.read() or b""
    except http.client.HTTPException as exc:  # e.g. IncompleteRead: we cannot know what the server did
        raise ConnectionError(f"{type(exc).__name__}") from exc


def new_client_id(prefix: str = "ARENA") -> str:
    """Unique id using only the characters the docs allow (A-Z, 0-9, underscore)."""
    return f"{prefix}_{int(time.time() * 1000)}_{secrets.token_hex(4).upper()}"


class WallexPrivateAPI:
    """Authenticated Wallex REST calls. The key is kept private: never logged, never in repr or redacted views.

    GETs are retried on network errors; POST/DELETE are never retried automatically.
    """

    def __init__(self, api_key: str | None, *, sub_account_id: str | None = None,
                 sub_account_header: str = SUB_ACCOUNT_HEADER, sub_account_on_orders: bool = True,
                 transport: Transport | None = None, base_url: str = BASE_URL, timeout: float = 10.0,
                 get_retries: int = 2, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time):
        self._key = api_key
        self.sub_account_id = sub_account_id
        self.sub_account_header = sub_account_header
        self.sub_account_on_orders = sub_account_on_orders  # spot-intro: credit-account spot calls send the key AND this header
        self.transport = transport or urllib_transport
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.get_retries = get_retries
        self.sleep, self.clock = sleep, clock
        self._posts: deque[float] = deque()

    def __repr__(self) -> str:
        return f"WallexPrivateAPI(base_url={self.base_url!r}, key={'set' if self._key else 'missing'}, sub_account={'set' if self.sub_account_id else 'none'})"

    @classmethod
    def from_env_file(cls, path: Path | str = "/opt/wallex/private/wallex.env", **kw) -> "WallexPrivateAPI":
        """Read WALLEX_API_KEY / WALLEX_SUBACCOUNT_ID from a KEY=VALUE file without echoing anything."""
        values = {}
        for line in Path(path).read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip("'\"")
        return cls(values.get("WALLEX_API_KEY") or None, sub_account_id=values.get("WALLEX_SUBACCOUNT_ID") or None, **kw)

    # ---- request building
    def build(self, method: str, path: str, *, query: dict | None = None, body: dict | None = None,
              sub_account: bool = False) -> dict:
        """The exact request that would be sent, with the key redacted (safe to log)."""
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        headers = {"Accept": "application/json", "User-Agent": "wallex-arena-live/1"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if sub_account and self.sub_account_id:
            headers[self.sub_account_header] = self.sub_account_id
        headers["x-api-key"] = "***"
        return {"method": method, "url": url, "headers": headers, "body": body}

    def _send(self, req: dict) -> dict:
        if not self._key:
            raise LiveRefused("no API key configured")
        headers = dict(req["headers"], **{"x-api-key": self._key})
        data = json.dumps(req["body"], separators=(",", ":")).encode() if req["body"] is not None else None
        is_get = req["method"] == "GET"
        endpoint = req["method"] + " " + urllib.parse.urlsplit(req["url"]).path
        # Reads retry transient network errors (backoff + jitter, circuit breaker "wallex.private.read"). Order
        # placement and cancels are never re-sent: create_order settles ambiguity by client_id lookup instead.
        status, raw = call("wallex.private.read" if is_get else "wallex.private.write",
                           lambda: self.transport(req["method"], req["url"], headers, data, self.timeout),
                           max_attempts=1 + (self.get_retries if is_get else 0), base=0.5, cap=4.0, endpoint=endpoint,
                           retry_if=lambda e: isinstance(e, OSError), use_breaker=is_get, sleep=self.sleep)
        obs.log("external.call", "info" if 200 <= status < 300 else "warn", endpoint, upstream="wallex.private", endpoint=endpoint, status=status,
                client_order_id=(req["body"] or {}).get("client_id") or None)
        try:
            payload = json.loads(raw.decode() or "{}")
        except ValueError:
            payload = {"message": raw[:200].decode(errors="replace")}
        if not 200 <= status < 300 or payload.get("success") is False:
            raise WallexAPIError(status, str(payload.get("message", "")), payload.get("code"), payload.get("result"))
        return payload

    def _throttle_post(self):
        limit, window = ORDER_RATE
        now = self.clock()
        while self._posts and now - self._posts[0] > window:
            self._posts.popleft()
        if len(self._posts) >= limit:
            self.sleep(window - (now - self._posts[0]) + 0.05)
        self._posts.append(self.clock())

    # ---- spot orders
    def order_request(self, symbol: str, side: str, type_: str, price: str | None, quantity: str,
                      client_id: str, stop_price: str | None = None) -> dict:
        # POST /v1/account/orders  (docs/spot-create-order, docs/generated/post-account-orders)
        body = {"symbol": symbol, "side": side, "type": type_, "quantity": quantity, "client_id": client_id}
        if price is not None:
            body["price"] = price
        if stop_price is not None:
            body["stop_Price"] = stop_price  # sic: the docs spell it stop_Price
        return self.build("POST", "/v1/account/orders", body=body, sub_account=self.sub_account_on_orders)

    def create_order(self, symbol: str, side: str, type_: str, price: str | None, quantity: str,
                     client_id: str, stop_price: str | None = None, lookup_attempts: int = 3) -> dict:
        """Place an order (expects HTTP 201). Never re-POSTs: an ambiguous outcome (timeout, connection drop,
        5xx/408) is resolved by GET-ing the order by client_id; if that cannot settle it, AmbiguousOrder."""
        req = self.order_request(symbol, side, type_, price, quantity, client_id, stop_price)
        self._throttle_post()
        try:
            result = self._send(req).get("result")
            if isinstance(result, dict) and result.get("clientOrderId"):
                return result
            cause = "2xx without an order in the body"
        except WallexAPIError as exc:
            if exc.status < 500 and exc.status != 408:
                raise  # definitive rejection: the order does not exist
            cause = f"HTTP {exc.status}"
        except (OSError, socket.timeout) as exc:
            cause = type(exc).__name__
        for attempt in range(lookup_attempts):
            self.sleep(1.0 + attempt)
            try:
                return self.get_order(client_id)
            except WallexAPIError as exc:
                if exc.status != 404:
                    break
            except OSError:
                pass
        raise AmbiguousOrder(client_id, cause)

    def cancel_order(self, client_id: str, via: str = "path") -> dict:
        """Cancel by clientOrderId (the docs offer no cancel by exchange order id)."""
        if via == "path":
            # DELETE /v1/account/orders/{client_id}  (docs/spot-cancel-order, docs/generated/delete-orders-client)
            req = self.build("DELETE", "/v1/account/orders/" + urllib.parse.quote(client_id, safe=""),
                             sub_account=self.sub_account_on_orders)
        else:
            # DELETE /v1/account/orders  body {"clientOrderId"}  (docs/generated/delete-account-orders)
            req = self.build("DELETE", "/v1/account/orders", body={"clientOrderId": client_id},
                             sub_account=self.sub_account_on_orders)
        return self._send(req)["result"]

    def get_order(self, client_id: str) -> dict:
        # GET /v1/account/orders/{client_id}  (docs/spot-single-order-detail, docs/generated/get-orders-client)
        req = self.build("GET", "/v1/account/orders/" + urllib.parse.quote(client_id, safe=""),
                         sub_account=self.sub_account_on_orders)
        return self._send(req)["result"]

    def open_orders(self, symbol: str | None = None, page: int | None = None, per_page: int | None = None) -> list[dict]:
        # GET /v1/account/openOrders?symbol&page&per_page  (docs/spot-active-orders, docs/generated/get-open-order)
        # result is an object of arrays (keyed by symbol in practice); flatten it.
        req = self.build("GET", "/v1/account/openOrders", query={"symbol": symbol, "page": page, "per_page": per_page},
                         sub_account=self.sub_account_on_orders)
        result = self._send(req).get("result") or {}
        if isinstance(result, list):
            return result
        return [o for rows in result.values() if isinstance(rows, list) for o in rows]

    # ---- account
    def balances(self) -> dict:
        # GET /v1/account/balances -> result.balances{ASSET: {asset, value, locked, ...}}  (docs/generated/get-account-balances)
        return self._send(self.build("GET", "/v1/account/balances", sub_account=self.sub_account_on_orders))["result"]

    def balances_detail(self) -> dict:
        # GET /v1/account/balances-detail, header sub-account-client-id -> result{ASSET: {symbol,total,freeze,available}}
        # (docs/generated/sub-account-balances-details, trade-credit-swagger.yaml operationId subAccountBalancesDetails)
        return self._send(self.build("GET", "/v1/account/balances-detail", sub_account=True))["result"]

    def account_fee(self) -> dict:
        # GET /v1/account/fee -> result{SYMBOL: {makerFeeRate, takerFeeRate, recent_days_sum}}
        # (docs/basic-services-account-fee, docs/generated/get-account-fees)
        return self._send(self.build("GET", "/v1/account/fee", sub_account=self.sub_account_on_orders))["result"]


# ---- rounding (Decimal so the strings sent are exact)
def round_dec(value, decimals: int, up: bool) -> Decimal:
    step = Decimal(1).scaleb(-int(decimals))
    return Decimal(str(value)).quantize(step, rounding=ROUND_CEILING if up else ROUND_FLOOR)


def fmt(value: Decimal, decimals: int) -> str:
    return f"{value:.{int(decimals)}f}"


@dataclass
class LiveOrder:
    client_id: str
    symbol: str
    side: str
    price: str
    qty: str
    kind: str                  # "market" (bounded aggressive LIMIT) or "limit"
    placed_at: float
    reason: str = ""
    status: str = "PENDING"
    active: bool = True
    executed_qty: float = 0.0
    executed_sum: float = 0.0
    fee: float = 0.0
    fills_seen: int = 0
    unknown: bool = False      # POST was ambiguous and the lookup has not found it yet
    last_sync: float = 0.0
    fills: list = field(default_factory=list)


class LiveExecutor:
    """Same intents as PaperAccount, executed on Wallex with LIMIT orders bounded by the caller's price limits.

    `markets` maps symbol -> {"quote", "base", "price_decimals", "qty_decimals", "min_qty", "min_notional"},
    i.e. MarketFeed.meta (tickSize/stepSize from /v1/markets are decimal places).
    """

    def __init__(self, api: WallexPrivateAPI | None, markets: dict[str, dict], *, max_order_tmn: float = 2_000_000,
                 min_order_tmn: float = MIN_ORDER_TMN, allowed: set[str] | None = None, dry_run: bool = False,
                 kill_file: Path = KILL_FILE, env: dict | None = None, log_path: Path | None = None,
                 state_path: Path | None = None, ioc_wait: float = 1.5, client_prefix: str = "ARENA",
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time):
        self.api = api
        self.markets = markets
        self.max_order_tmn = max_order_tmn
        self.min_order_tmn = max(min_order_tmn, MIN_ORDER_TMN)
        self.allowed = allowed
        self.dry_run = dry_run
        self.kill_file = Path(kill_file)
        self.env = os.environ if env is None else env
        self.log = DecisionLog(Path(log_path)) if log_path else None
        self.state_path = Path(state_path) if state_path else None
        self.ioc_wait = ioc_wait
        self.client_prefix = client_prefix
        self.sleep, self.clock = sleep, clock
        self.orders: dict[str, LiveOrder] = {}
        if self.state_path and self.state_path.exists():
            self.orders = {k: LiveOrder(**v) for k, v in json.loads(self.state_path.read_text()).items()}
        # dry-run builds requests with a key-less client, so it can never send anything
        self._builder = WallexPrivateAPI(None, sub_account_id=api.sub_account_id if api else None,
                                         sub_account_header=api.sub_account_header if api else SUB_ACCOUNT_HEADER,
                                         sub_account_on_orders=api.sub_account_on_orders if api else False)

    # ---- safety
    def enabled(self) -> tuple[bool, str]:
        if self.kill_file.exists():
            return False, f"kill switch file {self.kill_file} exists"
        if self.env.get(ENABLE_ENV) != "1":
            return False, f"{ENABLE_ENV} is not '1'"
        if self.api is None:
            return False, "no API client"
        return True, "enabled"

    def _gate(self):
        """Dry-run never sends, so it is allowed while disabled; real orders need the switches."""
        if self.dry_run:
            return
        ok, why = self.enabled()
        if not ok:
            raise LiveRefused(why)

    def _market(self, symbol: str) -> dict:
        m = self.markets.get(symbol)
        if not symbol.endswith("TMN") or symbol in EXCLUDED or m is None or m.get("quote", "TMN") != "TMN":
            raise LiveRefused(f"{symbol} is not an allowed TMN market")
        if self.allowed is not None and symbol not in self.allowed:
            raise LiveRefused(f"{symbol} is not in this executor's whitelist")
        return m

    def _check_size(self, m: dict, price: Decimal, qty: Decimal, symbol: str):
        notional = float(price * qty)
        if qty <= 0 or price <= 0:
            raise LiveRefused(f"{symbol}: quantity/price round to zero")
        if notional < self.min_order_tmn or notional < num(m.get("min_notional")):
            raise LiveRefused(f"{symbol}: notional {notional:,.0f} TMN below minimum {self.min_order_tmn:,.0f}")
        if notional > self.max_order_tmn:
            raise LiveRefused(f"{symbol}: notional {notional:,.0f} TMN above per-order max {self.max_order_tmn:,.0f}")
        if float(qty) < num(m.get("min_qty")):
            raise LiveRefused(f"{symbol}: quantity {qty} below market minQty {m.get('min_qty')}")

    # ---- intents
    def market_buy(self, symbol: str, budget_tmn: float, max_price: float, reason: str = "") -> dict:
        """Spend at most budget_tmn (taker fee included) via a LIMIT BUY at max_price (rounded down to the tick,
        so the bound is never exceeded; an on-tick ask <= max_price stays marketable). Remainder is cancelled."""
        self._gate()
        m = self._market(symbol)
        if max_price <= 0 or budget_tmn <= 0:
            raise LiveRefused("budget and max_price must be positive")
        price = round_dec(max_price, m["price_decimals"], up=False)
        if price <= 0:
            raise LiveRefused(f"{symbol}: max_price rounds to zero")
        qty = round_dec(Decimal(str(budget_tmn)) / (1 + Decimal(str(TAKER_FEE["TMN"]))) / price, m["qty_decimals"], up=False)
        self._check_size(m, price, qty, symbol)
        return self._place(symbol, "BUY", price, qty, m, "market", reason)

    def market_sell(self, symbol: str, qty: float, min_price: float, reason: str = "") -> dict:
        """Sell qty (rounded down to the step) via a LIMIT SELL at min_price rounded up to the tick."""
        self._gate()
        m = self._market(symbol)
        if min_price <= 0 or qty <= 0:
            raise LiveRefused("qty and min_price must be positive")
        price = round_dec(min_price, m["price_decimals"], up=True)
        q = round_dec(qty, m["qty_decimals"], up=False)
        self._check_size(m, price, q, symbol)
        return self._place(symbol, "SELL", price, q, m, "market", reason)

    def place_limit(self, symbol: str, side: str, price: float, qty: float, reason: str = "") -> dict:
        """Resting LIMIT order. Buy price rounds down, sell price rounds up (never worse than asked); qty rounds down."""
        self._gate()
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise LiveRefused(f"bad side {side}")
        m = self._market(symbol)
        p = round_dec(price, m["price_decimals"], up=(side == "SELL"))
        q = round_dec(qty, m["qty_decimals"], up=False)
        self._check_size(m, p, q, symbol)
        return self._place(symbol, side, p, q, m, "limit", reason)

    def cancel(self, order_id: str, reason: str = "") -> dict:
        """Cancel by client_id (our order_id). Returns the final order state."""
        self._gate()
        if self.dry_run:
            req = self._builder.build("DELETE", "/v1/account/orders/" + order_id, sub_account=self._builder.sub_account_on_orders)
            self._record({"type": "live_dry_run", "request": req, "reason": reason})
            if order_id in self.orders:
                self.orders[order_id].active, self.orders[order_id].status = False, "CANCELED"
                self._save()
            return {"dry_run": True, "request": req, "result": {"clientOrderId": order_id, "status": "CANCELED", "active": False}}
        try:
            result = self.api.cancel_order(order_id)
        except WallexAPIError as exc:
            # e.g. already filled/cancelled: read the true state instead of guessing
            self._record({"type": "live_cancel_error", "client_id": order_id, "status": exc.status, "message": exc.message})
            result = self.api.get_order(order_id)
        fills = self._apply(order_id, result)
        self._record({"type": "live_cancel", "client_id": order_id, "reason": reason, "status": result.get("status")})
        return {"result": result, "fills": fills}

    def sync(self) -> list[dict]:
        """Poll every non-final order; return the new fills (price, qty, fee) since the last sync."""
        if self.dry_run or self.api is None:
            return []
        new = []
        for cid, o in list(self.orders.items()):
            if not (o.active or o.unknown):
                continue
            try:
                result = self.api.get_order(cid)
            except WallexAPIError as exc:
                if exc.status == 404 and o.unknown and self.clock() - o.placed_at > 120:
                    o.unknown, o.active, o.status = False, False, "NOT_PLACED"
                    self._record({"type": "live_order_not_found", "client_id": cid})
                continue
            except OSError:
                continue
            new += self._apply(cid, result)
        self._save()
        return new

    def balances(self) -> dict[str, dict]:
        """{asset: {"total", "locked", "available"}} from /v1/account/balances (or balances-detail for a sub-account)."""
        if self.api is None:
            raise LiveRefused("no API client")
        out = {}
        if self.api.sub_account_id:
            for asset, b in (self.api.balances_detail() or {}).items():
                out[asset] = {"total": num(b.get("total")), "locked": num(b.get("freeze")), "available": num(b.get("available"))}
            return out
        for asset, b in ((self.api.balances() or {}).get("balances") or {}).items():
            total, locked = num(b.get("value")), num(b.get("locked"))
            out[asset] = {"total": total, "locked": locked, "available": total - locked}
        return out

    # ---- internals
    def _place(self, symbol: str, side: str, price: Decimal, qty: Decimal, m: dict, kind: str, reason: str) -> dict:
        cid = new_client_id(self.client_prefix)
        p, q = fmt(price, m["price_decimals"]), fmt(qty, m["qty_decimals"])
        order = LiveOrder(cid, symbol, side, p, q, kind, self.clock(), reason)
        if self.dry_run:
            req = self._builder.order_request(symbol, side, "LIMIT", p, q, cid)
            self._record({"type": "live_dry_run", "request": req, "kind": kind, "reason": reason})
            # resting dry-run orders stay "open" until cancelled, like real ones; dry-run market orders end at once
            order.active, order.status = kind == "limit", "DRY_RUN"
            self.orders[cid] = order
            self._save()
            return {"dry_run": True, "request": req,
                    "result": {"clientOrderId": cid, "symbol": symbol, "side": side, "type": "LIMIT", "price": p,
                               "origQty": q, "executedQty": "0", "status": "DRY_RUN", "active": False, "fills": []}}
        self.orders[cid] = order
        self._record({"type": "live_order_sent", "client_id": cid, "symbol": symbol, "side": side, "price": p,
                      "qty": q, "kind": kind, "reason": reason})
        try:
            result = self.api.create_order(symbol, side, "LIMIT", p, q, cid)
        except AmbiguousOrder as exc:
            order.unknown = True
            self._save()
            self._record({"type": "live_order_ambiguous", "client_id": cid, "cause": exc.cause})
            raise
        except WallexAPIError as exc:
            order.active, order.status = False, "REJECTED"
            self._save()
            self._record({"type": "live_order_rejected", "client_id": cid, "status": exc.status, "message": exc.message})
            raise
        fills = self._apply(cid, result)
        if kind == "market" and self.orders[cid].active:
            # emulate IOC: give the matcher a moment, then cancel any unfilled remainder
            self.sleep(self.ioc_wait)
            result = self.api.get_order(cid)
            fills += self._apply(cid, result)
            if self.orders[cid].active:
                fills += self.cancel(cid, reason="ioc_remainder")["fills"]
        return {"result": result, "fills": fills, "order": asdict(self.orders[cid])}

    def _apply(self, cid: str, result: dict) -> list[dict]:
        """Merge an order snapshot; return fills not seen before."""
        o = self.orders.get(cid)
        if o is None:
            o = self.orders[cid] = LiveOrder(cid, result.get("symbol", ""), result.get("side", ""), str(result.get("price", "")),
                                             str(result.get("origQty", "")), "external", self.clock())
        o.unknown = False
        o.status = str(result.get("status") or o.status).upper()
        o.active = bool(result.get("active", o.status not in TERMINAL)) and o.status not in TERMINAL
        o.last_sync = self.clock()
        exec_qty, exec_sum, fee = num(result.get("executedQty")), num(result.get("executedSum")), num(result.get("fee"))
        new = []
        rows = result.get("fills") or []
        if rows and len(rows) > o.fills_seen:
            for row in rows[o.fills_seen:]:
                new.append(self._fill(o, num(row.get("quantity")), num(row.get("price")), num(row.get("fee")),
                                      row.get("feeAsset"), row, num(row.get("timestamp")) or None))
            o.fills_seen = len(rows)
        elif exec_qty > o.executed_qty + 1e-12 and not rows:
            # no per-trade rows: synthesize one fill from the executed totals
            dq, ds = exec_qty - o.executed_qty, exec_sum - o.executed_sum
            new.append(self._fill(o, dq, ds / dq if dq else 0.0, max(fee - o.fee, 0.0), None, {}, None))
        o.executed_qty, o.executed_sum, o.fee = max(exec_qty, o.executed_qty), max(exec_sum, o.executed_sum), max(fee, o.fee)
        o.fills += new
        for f in new:
            self._record(f)
        self._save()
        return new

    def _fill(self, o: LiveOrder, qty: float, price: float, fee: float, fee_asset, row: dict, ts) -> dict:
        base = self.markets.get(o.symbol, {}).get("base") or o.symbol[:-3]
        if fee_asset in (None, "", "TMN"):
            fee_tmn = fee if fee_asset == "TMN" else None
        elif fee_asset == base:
            fee_tmn = fee * price
        else:
            fee_tmn = None
        coef, maker = num(row.get("feeCoefficient")), num(row.get("makerFeeCoefficient"))
        taker = num(row.get("takerFeeCoefficient"))
        liquidity = "maker" if row and coef == maker and coef != taker else ("taker" if row else None)
        return {"type": "live_fill", "ts": ts or self.clock(), "client_id": o.client_id, "symbol": o.symbol,
                "side": o.side, "qty": qty, "price": price, "notional_tmn": qty * price, "fee": fee,
                "fee_asset": fee_asset, "fee_tmn": fee_tmn, "liquidity": liquidity, "exchange_ts": row.get("timestamp"),
                "reason": o.reason}

    def _record(self, record: dict):
        if self.log:
            self.log.append({"ts": self.clock(), **record})

    def _save(self):
        if self.state_path:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: asdict(v) for k, v in self.orders.items()}))
            tmp.replace(self.state_path)


def load_markets(transport: Transport | None = None, timeout: float = 10.0) -> dict[str, dict]:
    """Public GET /v1/markets normalized like MarketFeed.meta (tickSize/stepSize are decimal places)."""
    status, raw = (transport or urllib_transport)("GET", BASE_URL + "/v1/markets", {"User-Agent": "Mozilla/5.0"}, None, timeout)
    if status != 200:
        raise WallexAPIError(status, "markets fetch failed")
    out = {}
    for sym, m in json.loads(raw)["result"]["symbols"].items():
        out[sym] = {"symbol": sym, "base": m["baseAsset"], "quote": m["quoteAsset"],
                    "price_decimals": int(num(m.get("tickSize"))), "qty_decimals": int(num(m.get("stepSize"))),
                    "min_qty": num(m.get("minQty")), "min_notional": num(m.get("minNotional"))}
    return out
