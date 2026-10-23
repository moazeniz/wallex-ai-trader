import json
import os
import time
from types import SimpleNamespace

import pytest

from experiments.wallex_arena.account import PaperAccount
from experiments.wallex_arena.ai import Budget, ClaudeTrader, MakerDecision
from experiments.wallex_arena.engine import Engine, RestingOrder, challenge_times
from experiments.wallex_arena.wallex_mcp import ALLOWED_TOOLS, MCPToolError, WallexMCP, resample, run_ai_tool

NOW = int(time.time()) // 3600 * 3600


def rpc_ok(payload, sse=False):
    body = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}}
    if sse:
        return 200, ("event: message\ndata: " + json.dumps(body) + "\n\n").encode(), {"Content-Type": "text/event-stream"}
    return 200, json.dumps(body).encode(), {"Content-Type": "application/json"}


def candles_payload(step=60, n=180):
    t = [NOW - step * (n - i) for i in range(n)]
    return {"s": "ok", "t": t, "o": [str(100 + i) for i in range(n)], "h": [str(101 + i) for i in range(n)],
            "l": [str(99 + i) for i in range(n)], "c": [str(100.5 + i) for i in range(n)], "v": ["1"] * n}


class FakeTransport:
    def __init__(self, *responses):
        self.responses, self.sent = list(responses), []

    def __call__(self, url, body, headers, timeout):
        self.sent.append((url, json.loads(body), headers))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


def mcp(transport, rest=None, tmp_path=None):
    return WallexMCP(url="https://mcp.test/mcp", transport=transport, rest=rest, min_interval=0,
                     log_path=(tmp_path / "mcp_calls.jsonl") if tmp_path else None)


def test_candles_via_mcp_resampled_and_logged(tmp_path, capsys):
    tr = FakeTransport(rpc_ok(candles_payload(60, 180)))
    c = mcp(tr, tmp_path=tmp_path).candles("ethtmn", "15", 3)
    url, body, headers = tr.sent[0]
    assert body["method"] == "tools/call" and body["params"]["name"] == "wallex_basic_candle_history"
    assert body["params"]["arguments"]["resolution"] == "1" and body["params"]["arguments"]["symbol"] == "ETHTMN"
    assert "x-api-key" not in {k.lower() for k in headers}                     # public data: no key is ever sent
    assert c["source"] == "mcp" and 12 <= len(c["bars"]) <= 13 and all(b[0] % 900 == 0 for b in c["bars"])
    assert c["high"] >= c["low"] and c["change_pct"] > 0
    rec = json.loads((tmp_path / "mcp_calls.jsonl").read_text().splitlines()[0])
    assert rec["tool"] == "wallex_basic_candle_history" and rec["status"] == "ok" and rec["source"] == "mcp"
    logs = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    assert any(r["event"] == "mcp.call" and r["tool"] == "wallex_basic_candle_history" and "duration_ms" in r for r in logs)


def test_sse_responses_are_parsed():
    book = {"result": {"ask": [{"price": 101, "quantity": 2}], "bid": [{"price": 99, "quantity": 3}]}}
    ob = mcp(FakeTransport(rpc_ok(book, sse=True))).order_book("QQTMN", 5)
    assert ob["best_bid"] == 99 and ob["best_ask"] == 101 and ob["spread_pct"] == pytest.approx(2.0)


def test_state_changing_and_unlisted_tools_are_refused_without_network():
    tr = FakeTransport(rpc_ok({}))
    client = mcp(tr)
    for name in ("wallex_spot_post_account_orders", "wallex_spot_delete_orders_client", "wallex_basic_get_account_balances"):
        with pytest.raises(PermissionError):
            client.call_tool(name, {})
    assert tr.sent == [] and "wallex_spot_post_account_orders" not in ALLOWED_TOOLS


def test_mcp_outage_falls_back_to_rest_and_tool_errors_do_not():
    rest_calls = []

    def rest(path, params):
        rest_calls.append(path)
        return {"result": {"latestTrades": [{"timestamp": "2026-10-22T10:00:00Z", "price": "100", "quantity": "2", "isBuyOrder": True}]}}
    client = mcp(FakeTransport(ConnectionResetError("down")), rest=rest)
    out = client.recent_trades("QQTMN", 10)
    assert out["source"] == "rest" and out["buy_tmn"] == 200 and rest_calls == ["/v1/trades"]
    bad = {"jsonrpc": "2.0", "id": 1, "result": {"isError": True, "content": [{"type": "text", "text": '{"success":false,"message":"invalid symbol"}'}]}}
    client2 = mcp(FakeTransport((200, json.dumps(bad).encode(), {})), rest=rest)
    with pytest.raises(MCPToolError):
        client2.order_book("NOPETMN")
    assert rest_calls == ["/v1/trades"]                                         # no REST fallback for a definite tool error


def test_open_circuit_uses_rest(monkeypatch):
    from experiments.wallex_arena import resilience
    b = resilience.breaker("wallex.mcp")
    b.state, b.opened_at = "open", b.clock()
    tr = FakeTransport(rpc_ok({}))
    out = mcp(tr, rest=lambda p, q: {"result": {"ask": [{"price": 2, "quantity": 1}], "bid": [{"price": 1, "quantity": 1}]}}).order_book("QQTMN")
    assert out["source"] == "rest" and tr.sent == []


def test_market_info_from_mcp_markets_is_cached():
    markets = {"result": {"markets": [{"symbol": "SUITMN", "price": "266000", "change_24h": 1.2, "change_7D": -3.0,
                                       "highest_24h_price": "270000", "lowest_24h_price": "260000", "quote_volume_24h": 9e9,
                                       "fair_price": {"bid": "265900", "ask": "266100"}, "is_zero_fee": False, "price_precision": 0}]}}
    tr = FakeTransport(rpc_ok(markets))
    client = mcp(tr)
    a, b = client.market_info("SUITMN"), client.market_info("suitmn")
    assert a["price"] == 266000 and a["wallex_fair_bid"] == 265900 and b["source"] == "cache" and len(tr.sent) == 1


def test_resample_aggregates_ohlcv():
    rows = [(0, 1, 2, 0.5, 1.5, 1), (60, 1.5, 3, 1, 2, 2), (120, 2, 2.5, 1.8, 2.2, 3)]
    assert resample(rows, 120) == [[0, 1, 3, 0.5, 2, 3], [120, 2, 2.5, 1.8, 2.2, 3]]


def test_ai_tool_errors_are_returned_as_data():
    client = mcp(FakeTransport(rpc_ok({})))
    assert "error" in run_ai_tool(client, "get_candles", {"symbol": "X", "resolution": "7", "lookback_hours": 1})
    assert "error" in run_ai_tool(client, "place_order", {})


# ---- the AI research step: the model calls MCP tools, then makes the structured decision with that data
DECISION = {"market_view": "v", "plan": [], "flatten": [], "notes_for_next_cycle": "n"}


def test_anthropic_research_loop_calls_mcp_then_decides(tmp_path):
    sent = []

    def transport(url, body, headers, timeout):
        b = json.loads(body)
        sent.append(b)
        if "tools" in b and len(sent) == 1:
            return 200, json.dumps({"stop_reason": "tool_use", "usage": {"input_tokens": 10, "output_tokens": 5}, "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "tu1", "name": "get_candles", "input": {"symbol": "ETHTMN", "resolution": "60", "lookback_hours": 6}}]}).encode()
        if "tools" in b:
            return 200, json.dumps({"stop_reason": "end_turn", "usage": {}, "content": [{"type": "text", "text": "ETH trending up."}]}).encode()
        return 200, json.dumps({"stop_reason": "end_turn", "usage": {}, "content": [{"type": "text", "text": json.dumps(DECISION)}]}).encode()
    env = {"LLM_PROVIDER": "anthropic", "ANTHROPIC_BASE_URL": "http://gate.test", "ANTHROPIC_AUTH_TOKEN": "t", "WALLEX_MCP_TOOLS": "1"}
    client = mcp(FakeTransport(rpc_ok(candles_payload(3600, 6))))
    tr = ClaudeTrader("claude-sonnet-5-5", tmp_path / "k", Budget(tmp_path / "b.sqlite", 5), schema=MakerDecision, env=env, mcp=client)
    tr.clients["anthropic"].transport = transport
    tr._run('{"markets":["ETHTMN"]}')
    r = tr.take()
    assert r["ok"] and r["research"]["calls"][0]["tool"] == "get_candles" and r["research"]["calls"][0]["source"] == "mcp"
    assert r["research"]["note"] == "ETH trending up."
    tool_result = sent[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "tu1" and '"bars"' in tool_result["content"]
    final = sent[2]
    assert "output_config" in final and "tools" not in final and "official Wallex MCP" in final["messages"][0]["content"]


def test_boxapi_research_loop_uses_function_calls(tmp_path):
    sent = []

    def transport(url, body, headers, timeout):
        b = json.loads(body)
        sent.append(b)
        if "tools" in b and len(sent) == 1:
            return 200, json.dumps({"choices": [{"message": {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "get_order_book", "arguments": json.dumps({"symbol": "SUITMN", "levels": 5})}}]}}], "usage": {}}).encode()
        if "tools" in b:
            return 200, json.dumps({"choices": [{"message": {"content": "thin book"}}], "usage": {}}).encode()
        return 200, json.dumps({"choices": [{"message": {"content": json.dumps(DECISION)}, "finish_reason": "stop"}], "usage": {}}).encode()
    env = {"LLM_PROVIDER": "boxapi", "BOXAPI_KEYS": "k1", "WALLEX_MCP_TOOLS": "1"}
    book = {"result": {"ask": [{"price": 101, "quantity": 2}], "bid": [{"price": 99, "quantity": 3}]}}
    tr = ClaudeTrader("openai/gpt-6.1-sol", tmp_path / "k", Budget(tmp_path / "b.sqlite", 5), schema=MakerDecision, env=env, mcp=mcp(FakeTransport(rpc_ok(book))))
    tr.clients["boxapi"].transport = transport
    tr._run("{}")
    r = tr.take()
    assert r["ok"] and r["research"]["calls"][0]["tool"] == "get_order_book" and sent[1]["messages"][-1]["role"] == "tool"
    assert "max_completion_tokens" in sent[0] and "max_tokens" not in sent[0]


def test_research_failure_still_decides(tmp_path):
    def transport(url, body, headers, timeout):
        b = json.loads(body)
        if "tools" in b:
            return 400, b'{"error":{"message":"tools not supported"}}'
        return 200, json.dumps({"content": [{"type": "text", "text": json.dumps(DECISION)}], "usage": {}}).encode()
    env = {"LLM_PROVIDER": "anthropic", "ANTHROPIC_BASE_URL": "http://gate.test", "ANTHROPIC_AUTH_TOKEN": "t", "WALLEX_MCP_TOOLS": "1"}
    tr = ClaudeTrader("claude-sonnet-5-5", tmp_path / "k", Budget(tmp_path / "b.sqlite", 5), schema=MakerDecision, env=env, mcp=mcp(FakeTransport(rpc_ok({}))))
    tr.clients["anthropic"].transport = transport
    tr._run("{}")
    r = tr.take()
    assert r["ok"] and r["research"]["error"] == "LLMHTTPError"


def test_research_is_off_unless_enabled(tmp_path):
    tr = ClaudeTrader("claude-sonnet-5-5", tmp_path / "k", Budget(tmp_path / "b.sqlite", 5), schema=MakerDecision, env={})
    assert tr.mcp is None


# ---- end-of-challenge settlement in TMN (simulated clock)
def test_challenge_times_and_phases():
    end, flat, stop = challenge_times({"challenge_end_utc": "2026-10-23T20:00:00Z", "flatten_minutes": 60, "no_new_entries_minutes": 120})
    assert end - flat == 3600 and end - stop == 7200
    assert challenge_times({"liquidate_at_utc": "2026-10-23T19:00:00Z"})[1] - challenge_times({"liquidate_at_utc": "2026-10-23T19:00:00Z"})[2] == 3600
    eng = Engine.__new__(Engine)
    eng.challenge_end, eng.liquidate_at, eng.entries_stop_at = end, flat, stop
    eng.state = {"a": {"paused_until": 0, "halted": False}}
    assert eng.challenge_phase(stop - 1) == "trading" and eng.entries_allowed("a", stop - 1)
    assert eng.challenge_phase(stop + 1) == "no_new_entries" and not eng.entries_allowed("a", stop + 1)
    assert eng.challenge_phase(flat + 1) == "flattening" and eng.challenge_phase(end + 1) == "ended"


def flatten_engine(tmp_path):
    eng = Engine.__new__(Engine)
    acct = PaperAccount("a", tmp_path)
    book = {"asks": [(101.0, 1e9)], "bids": [(100.0, 1e9)], "bid": 100.0, "ask": 101.0, "ts": time.time(), "mid": 100.5}
    eng.feed = SimpleNamespace(books={"QQTMN": dict(book), "DUSTTMN": dict(book)}, usdt_tmn=lambda: 1.0, usdt_halted=lambda: False,
                               _depth=lambda s: (_ for _ in ()).throw(OSError("offline")))
    eng.accounts, eng.orders = {"a": acct}, {"a": {}}
    eng.state = {"a": {"paused_until": 0, "halted": False}}
    eng.challenge_end, eng.liquidate_at, eng.entries_stop_at = 10_000.0, 6_400.0, 2_800.0
    acct.book_fill("QQTMN", "BUY", 5000, 100.0, 1, "taker")              # 500k TMN position
    acct.book_fill("DUSTTMN", "BUY", 500, 100.0, 1, "taker")             # 50k TMN: below the 100k minimum order
    eng.orders["a"]["o1"] = RestingOrder("o1", "QQTMN", "SELL", 105.0, 5000.0, 6_000.0, "take profit", {})
    return eng, acct


def test_flatten_cancels_sells_reports_dust_and_does_not_resend_within_retry_window(tmp_path):
    eng, acct = flatten_engine(tmp_path)
    sells = []
    real = eng._market_sell
    eng._market_sell = lambda name, sym, qty, reason, signal: sells.append((sym, reason)) or real(name, sym, qty, reason, signal)
    assert eng._flatten("a", 6_401.0) is True
    assert eng.orders["a"] == {} and sells == [("QQTMN", "final_tmn_settlement")]
    assert "QQTMN" not in acct.positions() and "DUSTTMN" in acct.positions()
    rec = [json.loads(l)["record"] for l in (tmp_path / "a" / "decisions.jsonl").read_text().splitlines()]
    flat = [r for r in rec if r["type"] == "flatten"][-1]
    assert flat["cancelled"][0]["order_id"] == "o1" and flat["sells"][0]["result"] == "filled" and flat["flat"] is True
    assert flat["dust_below_min_order"][0]["symbol"] == "DUSTTMN"
    eng._flatten("a", 6_402.0)
    assert len(sells) == 1                                                  # nothing left to sell; no duplicate orders


def test_flatten_retries_unfilled_market_only_after_retry_window(tmp_path):
    eng, acct = flatten_engine(tmp_path)
    attempts = []
    eng._market_sell = lambda name, sym, qty, reason, signal: attempts.append(sym) or None   # exchange did not fill
    eng._flatten("a", 6_401.0)
    eng._flatten("a", 6_401.0 + 10)                                         # inside the retry window: no new order
    eng._flatten("a", 6_401.0 + Engine.FLATTEN_RETRY_SECONDS + 1)
    assert attempts == ["QQTMN", "QQTMN"] and eng.state["a"]["flat"] is False


@pytest.mark.skipif(os.environ.get("WALLEX_REAL_NETWORK") != "1", reason="real public Wallex MCP call: set WALLEX_REAL_NETWORK=1")
def test_real_public_mcp_candles_and_book():
    client = WallexMCP(min_interval=0.3)
    c = client.candles("BTCTMN", "60", 6)
    assert c["source"] == "mcp" and len(c["bars"]) >= 3 and c["last_close"] > 0
    ob = client.order_book("BTCTMN", 5)
    assert ob["source"] == "mcp" and ob["best_ask"] > ob["best_bid"] > 0
    names = {t["name"] for t in client.list_tools()}
    assert ALLOWED_TOOLS <= names
