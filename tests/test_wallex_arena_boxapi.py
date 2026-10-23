import json
import socket

import pytest

from experiments.wallex_arena import ai
from experiments.wallex_arena.ai import Budget, ClaudeTrader, MakerDecision
from experiments.wallex_arena.boxapi import BoxAPIClient, BoxAPIError, strict_schema

OK = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def fake(script):
    """script: list of (status | Exception, payload); records the key used for each call."""
    used = []

    def transport(url, body, headers, timeout):
        used.append(headers["Authorization"][7:])
        status, payload = script.pop(0)
        if isinstance(status, Exception):
            raise status
        return status, json.dumps(payload).encode()
    return transport, used


def test_round_robin_over_keys():
    t, used = fake([(200, OK)] * 3)
    c = BoxAPIClient(["k1", "k2", "k3"], transport=t)
    for _ in range(3):
        c.chat({})
    assert used == ["k1", "k2", "k3"]


def test_failover_on_429_5xx_and_timeout_then_cooldown():
    now = [1000.0]
    t, used = fake([(429, {}), (503, {}), (socket.timeout(), None), (200, OK), (200, OK)])
    c = BoxAPIClient(["k1", "k2", "k3", "k4"], transport=t, cooldown=60, clock=lambda: now[0])
    c.chat({})
    assert used == ["k1", "k2", "k3", "k4"] and c.stats["failovers"] == 3
    c.chat({})                       # k1-k3 cooling down: the healthy key goes first
    assert used[-1] == "k4"


def test_bad_request_is_not_retried_on_other_keys():
    t, used = fake([(400, {"error": "bad"})])
    with pytest.raises(BoxAPIError):
        BoxAPIClient(["k1", "k2"], transport=t).chat({})
    assert used == ["k1"]


def test_all_keys_failing_raises_without_leaking_keys():
    t, _ = fake([(401, {}), (402, {})])
    c = BoxAPIClient(["secret-key-1", "secret-key-2"], transport=t)
    with pytest.raises(BoxAPIError) as e:
        c.chat({})
    assert "secret" not in str(e.value) and "secret" not in repr(c)


def test_no_keys_is_an_error():
    with pytest.raises(BoxAPIError):
        BoxAPIClient.from_env({"BOXAPI_KEYS": " , "})


def test_strict_schema_requires_every_field():
    s = strict_schema(MakerDecision)
    assert s["additionalProperties"] is False and set(s["required"]) == set(MakerDecision.model_fields)
    item = s["$defs"]["MakerPlanItem"]
    assert item["additionalProperties"] is False and set(item["required"]) == {"symbol", "margin_pct", "size_fraction", "reason"}


def test_trader_boxapi_path_parses_decision_and_records_billed_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXAPI_KEYS", "k1,k2")
    decision = {"market_view": "calm", "plan": [{"symbol": "SUITMN", "margin_pct": 0.4, "size_fraction": 0.05, "reason": "wide spread"}],
                "flatten": [], "notes_for_next_cycle": "watch SUI"}
    sent = []

    def transport(url, body, headers, timeout):
        sent.append(json.loads(body))
        return 200, json.dumps({"id": "r1", "choices": [{"message": {"content": json.dumps(decision)}, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "cost": 0.0123}}).encode()

    budget = Budget(tmp_path / "b.sqlite", 5)
    tr = ClaudeTrader("openai/gpt-6.1-sol", tmp_path / "unused", budget, schema=MakerDecision, system_extra=ai.MAKER_SYSTEM)
    tr.client.transport = transport
    tr._run('{"x":1}')
    r = tr.take()
    assert r["ok"] and r["decision"]["plan"][0]["symbol"] == "SUITMN" and r["provider"] == "boxapi"
    assert abs(r["usd"] - 0.0123) < 1e-9 and abs(budget.spent_today() - 0.0123) < 1e-9 and tr.notes == "watch SUI"
    body = sent[0]
    assert body["model"] == "openai/gpt-6.1-sol" and "max_completion_tokens" in body and "max_tokens" not in body
    assert "temperature" not in body and body["response_format"]["json_schema"]["strict"] is True


def test_trader_boxapi_invalid_json_is_a_failed_decision(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXAPI_KEYS", "k1")
    tr = ClaudeTrader("openai/gpt-6.1-sol", tmp_path / "unused", Budget(tmp_path / "b.sqlite", 5), schema=MakerDecision)
    tr.client.transport = lambda *a: (200, json.dumps({"choices": [{"message": {"content": "not json"}}], "usage": {}}).encode())
    tr._run("{}")
    r = tr.take()
    assert r["ok"] is False and r["decision"] is None


# ---- LLM_PROVIDER switch (anthropic|boxapi) with fallback to the other provider
DECISION = {"market_view": "calm", "plan": [{"symbol": "SUITMN", "margin_pct": 0.4, "size_fraction": 0.05, "reason": "wide spread"}],
            "flatten": [], "notes_for_next_cycle": "n"}
RELAY = {"LLM_PROVIDER": "anthropic", "ANTHROPIC_BASE_URL": "http://relay.example/meow-llm", "ANTHROPIC_AUTH_TOKEN": "relay-secret",
         "BOXAPI_KEYS": "k1,k2"}


def anthropic_ok(seen):
    def transport(url, body, headers, timeout):
        seen.append((url, json.loads(body), headers))
        return 200, json.dumps({"id": "msg_1", "stop_reason": "end_turn", "usage": {"input_tokens": 1000, "output_tokens": 100},
                                "content": [{"type": "text", "text": json.dumps(DECISION)}]}).encode()
    return transport


def boxapi_ok(seen):
    def transport(url, body, headers, timeout):
        seen.append(json.loads(body))
        return 200, json.dumps({"id": "b1", "choices": [{"message": {"content": json.dumps(DECISION)}, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}}).encode()
    return transport


def failing(status, text="{}"):
    return lambda *a: (status, text.encode())


def trader(tmp_path, env, model="openai/gpt-6.1-sol"):
    return ClaudeTrader(model, tmp_path / "unused", Budget(tmp_path / "b.sqlite", 5), schema=MakerDecision, env=env)


def test_anthropic_primary_through_relay_with_bearer_and_structured_output(tmp_path):
    seen = []
    tr = trader(tmp_path, RELAY)
    assert [p for p, _ in tr.routes] == ["anthropic", "boxapi"] and tr.model == "claude-sonnet-5-5"
    tr.clients["anthropic"].transport = anthropic_ok(seen)
    tr._run("{}")
    r = tr.take()
    assert r["ok"] and r["provider"] == "anthropic" and r["model"] == "claude-sonnet-5-5" and "fallback_from" not in r
    url, body, headers = seen[0]
    assert url == "http://relay.example/meow-llm/v1/messages" and headers["Authorization"] == "Bearer relay-secret" and "x-api-key" not in headers
    assert body["max_tokens"] <= 5000 and "temperature" not in body and "stream" not in body
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False and "tools" not in body
    assert abs(r["usd"] - (1000 * 2.0 + 100 * 10.0) / 1e6) < 1e-12        # Sonnet price table
    assert "relay-secret" not in repr(tr.clients["anthropic"])


@pytest.mark.parametrize("status,text", [(429, "{}"), (529, "{}"), (500, "{}"),
                                         (400, '{"error":{"message":"Your credit balance is too low"}}')])
def test_anthropic_failure_falls_back_to_boxapi(tmp_path, status, text):
    seen = []
    tr = trader(tmp_path, RELAY)
    tr.clients["anthropic"].transport = failing(status, text)
    tr.clients["boxapi"].transport = boxapi_ok(seen)
    tr._run("{}")
    r = tr.take()
    assert r["ok"] and r["provider"] == "boxapi" and r["model"] == "openai/gpt-6.1-sol"
    assert r["fallback_from"][0]["provider"] == "anthropic" and seen[0]["model"] == "openai/gpt-6.1-sol"


def test_anthropic_bad_request_does_not_fall_back(tmp_path):
    seen = []
    tr = trader(tmp_path, RELAY)
    tr.clients["anthropic"].transport = failing(400, '{"error":{"message":"invalid schema"}}')
    tr.clients["boxapi"].transport = boxapi_ok(seen)
    tr._run("{}")
    r = tr.take()
    assert r["ok"] is False and not seen and "anthropic" in r["error"]


def test_boxapi_primary_falls_back_to_anthropic_when_keys_out_of_credit(tmp_path):
    seen = []
    tr = trader(tmp_path, dict(RELAY, LLM_PROVIDER="boxapi"), model="claude-sonnet-5-5")
    assert [p for p, _ in tr.routes] == ["boxapi", "anthropic"] and tr.model == "openai/gpt-6.1-sol"
    tr.clients["boxapi"].transport = failing(402)
    tr.clients["anthropic"].transport = anthropic_ok(seen)
    tr._run("{}")
    r = tr.take()
    assert r["ok"] and r["provider"] == "anthropic" and r["model"] == "claude-sonnet-5-5"
    assert r["fallback_from"][0]["provider"] == "boxapi"


def test_fallback_can_be_disabled_and_needs_credentials(tmp_path):
    assert len(trader(tmp_path, dict(RELAY, LLM_FALLBACK="0")).routes) == 1
    no_box = {k: v for k, v in RELAY.items() if k != "BOXAPI_KEYS"}
    assert [p for p, _ in trader(tmp_path, no_box).routes] == ["anthropic"]
    no_anthropic = {"LLM_PROVIDER": "boxapi", "BOXAPI_KEYS": "k1"}
    assert [p for p, _ in trader(tmp_path, no_anthropic).routes] == ["boxapi"]


def test_without_llm_provider_the_model_name_decides_and_key_file_uses_x_api_key(tmp_path):
    key = tmp_path / "k.env"
    key.write_text("ANTHROPIC_API_KEY=direct-key\n")
    seen = []
    tr = ClaudeTrader("claude-haiku-5-5", key, Budget(tmp_path / "b.sqlite", 5), schema=MakerDecision, env={})
    assert tr.routes == [("anthropic", "claude-haiku-5-5")]
    tr.clients["anthropic"].transport = anthropic_ok(seen)
    tr._run("{}")
    assert tr.take()["ok"] and seen[0][0] == "https://api.anthropic.com/v1/messages" and seen[0][2]["x-api-key"] == "direct-key"


@pytest.mark.parametrize("configured,anthropic_model,boxapi_model", [
    ("openai/gpt-6.1-sol", "claude-sonnet-5-5", "openai/gpt-6.1-sol"), ("openai/gpt-6-luna", "claude-haiku-5-5", "openai/gpt-6-luna"),
    ("claude-haiku-5-5", "claude-haiku-5-5", "openai/gpt-6-luna"), ("claude-sonnet-5-5", "claude-sonnet-5-5", "openai/gpt-6.1-sol")])
def test_model_tier_is_kept_across_providers(tmp_path, configured, anthropic_model, boxapi_model):
    tr = trader(tmp_path, RELAY, model=configured)
    assert tr.routes == [("anthropic", anthropic_model), ("boxapi", boxapi_model)]
