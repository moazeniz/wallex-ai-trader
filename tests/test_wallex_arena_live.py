"""Offline tests for experiments.wallex_arena.live: a fake transport stands in for Wallex; nothing hits the network."""
import json
import socket

import pytest

from experiments.wallex_arena import live
from experiments.wallex_arena.live import (AmbiguousOrder, LiveExecutor, LiveRefused, WallexAPIError,
                                           WallexPrivateAPI, round_dec)

KEY = "TEST_KEY_SHOULD_NEVER_BE_LOGGED"
MARKETS = {
    "BTCTMN": {"symbol": "BTCTMN", "base": "BTC", "quote": "TMN", "price_decimals": 0, "qty_decimals": 8,
               "min_qty": 1e-08, "min_notional": 50000},
    "SHIBTMN": {"symbol": "SHIBTMN", "base": "SHIB", "quote": "TMN", "price_decimals": 4, "qty_decimals": 0,
                "min_qty": 1, "min_notional": 50000},
    "WBTCTMN": {"symbol": "WBTCTMN", "base": "WBTC", "quote": "TMN", "price_decimals": 0, "qty_decimals": 8,
                "min_qty": 1e-08, "min_notional": 50000},
    "BTCUSDT": {"symbol": "BTCUSDT", "base": "BTC", "quote": "USDT", "price_decimals": 2, "qty_decimals": 6,
                "min_qty": 1e-06, "min_notional": 1},
}
ENABLED = {"WALLEX_LIVE_ENABLED": "1"}


def order(cid, symbol="BTCTMN", side="BUY", price="10000000000", qty="0.0001", status="NEW", active=True,
          executed="0", executed_sum="0", fee="0", fills=None):
    return {"clientOrderId": cid, "symbol": symbol, "side": side, "type": "LIMIT", "price": price, "origQty": qty,
            "status": status, "active": active, "executedQty": executed, "executedSum": executed_sum, "fee": fee,
            "fills": fills or []}


def fill_row(price, qty, fee, asset="BTC", coef="0.003", maker="0.0025", taker="0.003"):
    return {"price": price, "quantity": qty, "fee": fee, "feeAsset": asset, "feeCoefficient": coef,
            "makerFeeCoefficient": maker, "takerFeeCoefficient": taker, "isBuyer": True, "symbol": "BTCTMN",
            "sum": str(float(price) * float(qty)), "timestamp": "2026-10-22T10:00:00Z"}


class FakeTransport:
    """Records requests; answers from a queue of (status, payload) or exceptions, or from a handler."""

    def __init__(self, replies=None, handler=None):
        self.calls, self.replies, self.handler = [], list(replies or []), handler

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append({"method": method, "url": url, "headers": dict(headers),
                           "body": json.loads(body) if body else None, "timeout": timeout})
        reply = self.handler(self.calls[-1]) if self.handler else self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, payload = reply
        return status, json.dumps(payload).encode()


def ok(result, status=200):
    return status, {"success": True, "message": "ok", "result": result}


def api_with(transport, **kw):
    return WallexPrivateAPI(KEY, transport=transport, sleep=lambda s: None, **kw)


def executor(transport, tmp_path, env=ENABLED, **kw):
    return LiveExecutor(api_with(transport), MARKETS, env=env, kill_file=tmp_path / "LIVE_DISABLED",
                        log_path=tmp_path / "live.jsonl", sleep=lambda s: None, **kw)


# ---- request shapes (docs: spot-create-order, spot-cancel-order, spot-single-order-detail, spot-active-orders, ...)
def test_create_order_request_matches_docs():
    t = FakeTransport([ok(order("C1"), 201)])
    api_with(t).create_order("BTCTMN", "BUY", "LIMIT", "10000000000", "0.00010000", "C1")
    c = t.calls[0]
    assert c["method"] == "POST" and c["url"] == "https://api.wallex.ir/v1/account/orders"
    assert c["body"] == {"symbol": "BTCTMN", "side": "BUY", "type": "LIMIT", "price": "10000000000",
                         "quantity": "0.00010000", "client_id": "C1"}
    assert c["headers"]["x-api-key"] == KEY and c["headers"]["Content-Type"] == "application/json"
    assert "sub-account-client-id" not in c["headers"] and c["timeout"] == 10.0


def test_read_and_cancel_paths():
    t = FakeTransport(handler=lambda c: ok({"BTCTMN": [order("A")]} if "openOrders" in c["url"] else
                                           {"balances": {}} if c["url"].endswith("/balances") else order("A")))
    api = api_with(t)
    api.get_order("A")
    api.cancel_order("A")
    api.cancel_order("A", via="body")
    assert api.open_orders("BTCTMN") == [order("A")]
    api.balances()
    api.account_fee()
    got = [(c["method"], c["url"], c["body"]) for c in t.calls]
    assert got == [("GET", "https://api.wallex.ir/v1/account/orders/A", None),
                   ("DELETE", "https://api.wallex.ir/v1/account/orders/A", None),
                   ("DELETE", "https://api.wallex.ir/v1/account/orders", {"clientOrderId": "A"}),
                   ("GET", "https://api.wallex.ir/v1/account/openOrders?symbol=BTCTMN", None),
                   ("GET", "https://api.wallex.ir/v1/account/balances", None),
                   ("GET", "https://api.wallex.ir/v1/account/fee", None)]


def test_sub_account_header_on_balances_detail_and_orders_by_default():
    # spot-intro: credit-account spot calls send the sub-account key and the sub-account-client-id header.
    t = FakeTransport(handler=lambda c: ok(order("A") if "orders" in c["url"] else {}))
    api = api_with(t, sub_account_id="SUB1")
    api.balances_detail()
    api.get_order("A")
    assert t.calls[0]["url"].endswith("/v1/account/balances-detail")
    assert t.calls[0]["headers"]["sub-account-client-id"] == "SUB1"
    assert t.calls[1]["headers"]["sub-account-client-id"] == "SUB1"
    api_with(t, sub_account_id="SUB1", sub_account_on_orders=False).get_order("A")
    assert "sub-account-client-id" not in t.calls[2]["headers"]


def test_key_never_in_repr_or_built_request():
    api = api_with(FakeTransport())
    assert KEY not in repr(api)
    assert KEY not in json.dumps(api.order_request("BTCTMN", "BUY", "LIMIT", "1", "1", "C"))


def test_get_retries_but_post_does_not():
    t = FakeTransport([socket.timeout(), ok(order("A"))])
    assert api_with(t).get_order("A")["clientOrderId"] == "A"
    assert len(t.calls) == 2
    t = FakeTransport([(422, {"success": False, "message": "duplicate client_id", "code": 422})])
    with pytest.raises(WallexAPIError) as e:
        api_with(t).create_order("BTCTMN", "BUY", "LIMIT", "1", "1", "C")
    assert e.value.status == 422 and len(t.calls) == 1


# ---- ambiguous POST: resolved by GET /v1/account/orders/{client_id}, never re-POSTed
def test_ambiguous_post_resolved_by_lookup():
    t = FakeTransport([socket.timeout(), (404, {"success": False, "message": "not found"}), ok(order("C9"))])
    result = api_with(t).create_order("BTCTMN", "BUY", "LIMIT", "1", "1", "C9")
    assert result["clientOrderId"] == "C9"
    assert [c["method"] for c in t.calls] == ["POST", "GET", "GET"]


def test_ambiguous_post_unresolved_raises_and_sync_settles(tmp_path):
    def handler(c):
        if c["method"] == "POST":
            return (502, {"success": False, "message": "bad gateway"})
        return (404, {"success": False, "message": "not found"})
    t = FakeTransport(handler=handler)
    ex = executor(t, tmp_path)
    with pytest.raises(AmbiguousOrder) as e:
        ex.market_buy("BTCTMN", 1_000_000, 10_000_000_000)
    cid = e.value.client_id
    assert sum(c["method"] == "POST" for c in t.calls) == 1
    assert ex.orders[cid].unknown
    t.handler = lambda c: ok(order(cid, status="FILLED", active=False, executed="0.0000997",
                                   executed_sum="997000", fills=[fill_row("10000000000", "0.0000997", "0.0000003")]))
    fills = ex.sync()
    assert len(fills) == 1 and not ex.orders[cid].unknown and not ex.orders[cid].active


# ---- rounding and executor intents
def test_rounding_helpers():
    assert str(round_dec(0.1 + 0.2, 8, up=False)) == "0.30000000"
    assert str(round_dec(1.23456, 4, up=True)) == "1.2346"
    assert str(round_dec(1.23456, 4, up=False)) == "1.2345"
    assert str(round_dec(99.9, 0, up=False)) == "99"


def test_market_buy_rounds_price_down_qty_down_and_fits_budget(tmp_path):
    t = FakeTransport([ok(order("X", status="FILLED", active=False), 201)])
    ex = executor(t, tmp_path)
    ex.market_buy("SHIBTMN", budget_tmn=500_000, max_price=0.123456789)
    body = t.calls[0]["body"]
    assert body["type"] == "LIMIT" and body["side"] == "BUY"
    assert body["price"] == "0.1234"                     # never above max_price
    qty = int(body["quantity"])                          # stepSize 0 -> integer qty, rounded down
    assert qty == int(500_000 / 1.003 / 0.1234)
    assert qty * 0.1234 * 1.003 <= 500_000
    assert body["client_id"].startswith("ARENA_") and all(ch.isalnum() or ch == "_" for ch in body["client_id"])


def test_market_sell_rounds_price_up_and_qty_down(tmp_path):
    t = FakeTransport([ok(order("X", status="FILLED", active=False), 201)])
    executor(t, tmp_path).market_sell("SHIBTMN", qty=1_000_000.9, min_price=0.123401)
    body = t.calls[0]["body"]
    assert body["price"] == "0.1235" and body["quantity"] == "1000000" and body["side"] == "SELL"


def test_place_limit_rounding(tmp_path):
    t = FakeTransport([ok(order("X"), 201), ok(order("Y"), 201)])
    ex = executor(t, tmp_path)
    ex.place_limit("BTCTMN", "buy", 10_000_000_000.7, 0.000123456789)
    ex.place_limit("BTCTMN", "SELL", 10_000_000_000.2, 0.000123456789)
    assert (t.calls[0]["body"]["price"], t.calls[0]["body"]["quantity"]) == ("10000000000", "0.00012345")
    assert (t.calls[1]["body"]["price"], t.calls[1]["body"]["quantity"]) == ("10000000001", "0.00012345")


def test_min_order_and_max_notional_refused_without_network(tmp_path):
    t = FakeTransport()
    ex = executor(t, tmp_path, max_order_tmn=1_000_000)
    with pytest.raises(LiveRefused, match="below minimum"):
        ex.market_buy("BTCTMN", 99_000, 10_000_000_000)
    with pytest.raises(LiveRefused, match="below minimum"):
        ex.market_sell("BTCTMN", 0.000009, 10_000_000_000)  # 90,000 TMN
    with pytest.raises(LiveRefused, match="above per-order max"):
        ex.place_limit("BTCTMN", "BUY", 10_000_000_000, 0.0002)
    assert ex.min_order_tmn == 100_000 and LiveExecutor(None, MARKETS, min_order_tmn=10).min_order_tmn == 100_000
    assert t.calls == []


def test_whitelist_refused_without_network(tmp_path):
    t = FakeTransport()
    ex = executor(t, tmp_path)
    for sym in ("WBTCTMN", "BTCUSDT", "NOPETMN"):  # excluded list, non-TMN, unknown
        with pytest.raises(LiveRefused, match="not an allowed"):
            ex.place_limit(sym, "BUY", 10_000_000_000, 0.0001)
    ex2 = executor(t, tmp_path, allowed={"SHIBTMN"})
    with pytest.raises(LiveRefused, match="whitelist"):
        ex2.place_limit("BTCTMN", "BUY", 10_000_000_000, 0.0001)
    assert t.calls == []


# ---- kill switch
@pytest.mark.parametrize("env", [{}, {"WALLEX_LIVE_ENABLED": "0"}, {"WALLEX_LIVE_ENABLED": "true"}])
def test_disabled_by_default_blocks_every_order_method(tmp_path, env):
    t = FakeTransport()
    ex = executor(t, tmp_path, env=env)
    for call in (lambda: ex.market_buy("BTCTMN", 1_000_000, 10_000_000_000),
                 lambda: ex.market_sell("BTCTMN", 0.0001, 10_000_000_000),
                 lambda: ex.place_limit("BTCTMN", "BUY", 10_000_000_000, 0.0001),
                 lambda: ex.cancel("ANY")):
        with pytest.raises(LiveRefused, match="WALLEX_LIVE_ENABLED"):
            call()
    assert t.calls == []


def test_kill_file_overrides_env(tmp_path):
    (tmp_path / "LIVE_DISABLED").write_text("")
    t = FakeTransport()
    ex = executor(t, tmp_path)
    assert ex.enabled()[0] is False
    with pytest.raises(LiveRefused, match="kill switch"):
        ex.place_limit("BTCTMN", "BUY", 10_000_000_000, 0.0001)
    assert t.calls == []


def test_real_os_environment_default_is_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("WALLEX_LIVE_ENABLED", raising=False)
    ex = LiveExecutor(api_with(FakeTransport()), MARKETS, kill_file=tmp_path / "none")
    assert ex.enabled() == (False, "WALLEX_LIVE_ENABLED is not '1'")
    assert live.KILL_FILE.as_posix() == "/var/lib/wallex-arena/LIVE_DISABLED"


# ---- dry run
def test_dry_run_logs_exact_request_without_key_or_network(tmp_path):
    t = FakeTransport()
    ex = LiveExecutor(api_with(t), MARKETS, dry_run=True, env={}, kill_file=tmp_path / "x",
                      log_path=tmp_path / "live.jsonl")
    out = ex.market_buy("BTCTMN", 1_000_000, 10_000_000_000.9)
    req = out["request"]
    assert out["dry_run"] and out["result"]["status"] == "DRY_RUN"
    assert req["method"] == "POST" and req["url"] == "https://api.wallex.ir/v1/account/orders"
    assert req["body"]["price"] == "10000000000" and req["body"]["quantity"] == "0.00009970"
    assert req["headers"]["x-api-key"] == "***"
    ex.cancel(req["body"]["client_id"])
    assert t.calls == [] and ex.sync() == []
    text = (tmp_path / "live.jsonl").read_text()
    assert KEY not in text and "live_dry_run" in text
    assert live.DecisionLog.verify(tmp_path / "live.jsonl") == 2


def test_dry_run_with_no_api_object(tmp_path):
    ex = LiveExecutor(None, MARKETS, dry_run=True, env={}, kill_file=tmp_path / "x")
    assert ex.place_limit("SHIBTMN", "SELL", 0.5, 300_000)["request"]["body"]["quantity"] == "300000"


# ---- fills
def test_fill_parsing_incremental_with_fee_assets_and_liquidity(tmp_path):
    cid = "L1"
    t = FakeTransport()
    ex = executor(t, tmp_path)
    t.replies = [ok(order(cid, executed="0.00004", executed_sum="400000", fee="0.00000012",
                          fills=[fill_row("10000000000", "0.00004", "0.00000012")]), 201)]
    out = ex.place_limit("BTCTMN", "BUY", 10_000_000_000, 0.0001)
    cid = out["order"]["client_id"]
    f = out["fills"][0]
    assert (f["qty"], f["price"], f["fee"], f["fee_asset"]) == (0.00004, 10_000_000_000, 0.00000012, "BTC")
    assert f["fee_tmn"] == pytest.approx(1200) and f["liquidity"] == "taker"
    t.replies = [ok(order(cid, status="FILLED", active=False, executed="0.0001", executed_sum="1000000",
                          fills=[fill_row("10000000000", "0.00004", "0.00000012"),
                                 fill_row("10000000000", "0.00006", "150", asset="TMN", coef="0.0025")]))]
    new = ex.sync()
    assert len(new) == 1 and new[0]["qty"] == 0.00006 and new[0]["fee_tmn"] == 150 and new[0]["liquidity"] == "maker"
    assert ex.sync() == [] and len(t.calls) == 2  # finished orders are not polled again


def test_fill_synthesized_from_totals_when_no_rows(tmp_path):
    t = FakeTransport([ok(order("Z"), 201)])
    ex = executor(t, tmp_path)
    cid = ex.place_limit("BTCTMN", "SELL", 10_000_000_000, 0.0001)["order"]["client_id"]
    t.replies = [ok(order(cid, side="SELL", status="FILLED", active=False, executed="0.0001",
                          executed_sum="1000000", fee="3000"))]
    (f,) = ex.sync()
    assert (f["qty"], f["price"], f["fee"]) == (0.0001, 10_000_000_000, 3000)


def test_market_order_cancels_unfilled_remainder(tmp_path):
    seq = []

    def handler(c):
        seq.append(c["method"])
        cid = (c["body"] or {}).get("client_id") or c["url"].rsplit("/", 1)[1]
        part = dict(executed="0.00005", executed_sum="500000", fills=[fill_row("10000000000", "0.00005", "0.00000015")])
        if c["method"] == "DELETE":
            return ok(order(cid, status="CANCELED", active=False, **part))
        return ok(order(cid, **part), 201 if c["method"] == "POST" else 200)
    t = FakeTransport(handler=handler)
    out = executor(t, tmp_path).market_buy("BTCTMN", 1_000_000, 10_000_000_000)
    assert seq == ["POST", "GET", "DELETE"]
    assert len(out["fills"]) == 1 and out["order"]["status"] == "CANCELED" and not out["order"]["active"]


def test_state_survives_restart(tmp_path):
    t = FakeTransport([ok(order("S"), 201)])
    ex = executor(t, tmp_path, state_path=tmp_path / "orders.json")
    cid = ex.place_limit("BTCTMN", "BUY", 10_000_000_000, 0.0001)["order"]["client_id"]
    again = executor(t, tmp_path, state_path=tmp_path / "orders.json")
    assert again.orders[cid].active and again.orders[cid].symbol == "BTCTMN"


def test_balances_normalized(tmp_path):
    t = FakeTransport([ok({"balances": {"TMN": {"asset": "TMN", "value": "5000000", "locked": "1000000"}}})])
    assert executor(t, tmp_path).balances() == {"TMN": {"total": 5e6, "locked": 1e6, "available": 4e6}}
    t = FakeTransport([ok({"TMN": {"symbol": "TMN", "total": "5000000", "freeze": "0", "available": "5000000"}})])
    ex = LiveExecutor(api_with(t, sub_account_id="SUB1"), MARKETS, env={}, kill_file=tmp_path / "x")
    assert ex.balances()["TMN"]["available"] == 5e6
    assert t.calls[0]["headers"]["sub-account-client-id"] == "SUB1"
