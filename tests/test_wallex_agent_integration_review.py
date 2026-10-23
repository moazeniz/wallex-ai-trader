"""Independent offline integration review: no network and no paid provider calls."""
import copy
import json
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import pytest

from experiments.wallex_agent import runner as runner_module
from experiments.wallex_agent.council import ResearchCouncil
from experiments.wallex_agent.llm import HaikuClient, MODEL
from experiments.wallex_agent.risk import RiskConfig, net_expected_return

NOW = 1_800_000_100.75
FAKE_KEY = "sk-ant-" + "fake_integration_secret_" * 3


def market_context(symbol="SOLTMN", bid=100_000.0, spread=0.001, depth=100.0, bar=None):
    ask = bid * (1 + spread / 2) / (1 - spread / 2)
    return {
        "symbol": symbol, "observed_at": NOW, "candles": [],
        "market": {"symbol": symbol, "amount_decimals": 8, "minimum_quantity": "0.00000001", "minimum_notional": "100000"},
        "book": {"bid": bid, "ask": ask, "spread_fraction": spread, "observed_at": NOW,
                 "bids": [{"price": bid, "quantity": depth}], "asks": [{"price": ask, "quantity": depth}]},
        "technical": {"last_closed_at": int(NOW) // 3600 * 3600 if bar is None else bar, "one_hour": {"atr_fraction": 0.01, "last_price": 100000.0}},
        "cross_market": {"source": "tradingview", "status": "unavailable", "quote_currency": "USDT", "conversion_to_tmn_applied": False},
    }


def forecast(gross=0.05, selected="ensemble"):
    return {"model_hash": "frozen-fake-model", "horizon_hours": 8, "training_cutoff": NOW - 10000,
            "predicted_gross_return": gross, "selected_candidate": selected}


class FakeQuant:
    def __init__(self, symbols, predictions=None):
        self.manifest = {"assets": list(symbols), "fee_rate_per_side": 0.0035, "slippage_rate_per_side": 0.001}
        self.predictions = predictions or {}
    def forecast(self, symbol, rows):
        result = self.predictions.get(symbol, forecast())
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


class FakeNews:
    def __init__(self):
        self.as_of = []
    def refresh(self):
        return {"status": "ok", "feeds": []}
    def snapshot(self, symbol, as_of, limit):
        self.as_of.append(as_of)
        return [{"id": "news-" + symbol, "title": symbol + " protocol update", "summary": "New observed facts",
                 "source": "offline-fixture", "url": "https://example.invalid/evidence",
                 "published_at": NOW - 30, "first_seen_at": NOW - 0.25, "available_at": NOW - 0.25}]


class FakeMarket:
    def __init__(self, contexts):
        self.contexts = contexts
        self.calls = []
        self.lock = Lock()
    def snapshot(self, symbol):
        with self.lock:
            self.calls.append(symbol)
            value = self.contexts[symbol]
            if callable(value):
                value = value(self.calls.count(symbol))
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


class FakeClient:
    def __init__(self, action="HOLD", failed=False):
        self.action, self.failed = action, failed
        self.calls, self.lock = [], Lock()
    def assess(self, context, news):
        assert all(x["available_at"] <= context["as_of"] for x in news)
        with self.lock:
            self.calls.append(copy.deepcopy(context))
        return {"status": "error" if self.failed else "ok", "proposed_action": "HOLD" if self.failed else self.action,
                "sentiment": 0.0, "confidence": 0.0 if self.failed else 0.8, "risk_flags": ["unverified_news"] if self.failed else [],
                "evidence_ids": [] if self.failed else [x["id"] for x in news], "thesis": "Offline research fixture",
                "invalidation": "New contrary evidence", "horizon_hours": 8,
                "error_code": "provider_or_secret_error" if self.failed else None,
                "usage": {"input_tokens": 42, "output_tokens": 17}, "cost_usd": 0.0001,
                "provenance": {"request_hash": "failure-hash" if self.failed else "success-hash", "reserved_usd": 0.002}}
    def budget_status(self):
        return {"attempts": len(self.calls), "charged_or_reserved_usd": 0, "max_calls": 1200, "max_usd": 2}


@pytest.fixture
def build_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module.time, "time", lambda: NOW)
    count = {"value": 0}
    def build(symbols=("SOLTMN", "XRPTMN"), contexts=None, predictions=None, action="HOLD", failed=False, research_limit=3, news=None):
        count["value"] += 1
        folder = tmp_path / str(count["value"])
        quant = FakeQuant(symbols, predictions)
        monkeypatch.setattr(runner_module.QuantModel, "load", lambda path: quant)
        market = FakeMarket(contexts or {s: market_context(s) for s in symbols})
        client = FakeClient(action=action, failed=failed)
        news = news or FakeNews()
        agent = runner_module.AgentRunner("unused-fake-model.json", folder / "state", folder / "artifacts",
                                          market=market, news=news, client=client, research_limit=research_limit)
        return agent
    return build


def seed_long(agent, symbol="SOLTMN"):
    return agent.portfolio.apply("seed-" + symbol, market_context(symbol), {"action": "BUY", "reason": ["offline-seed"], "horizon_hours": 8}, NOW)


def test_fractional_news_availability_reaches_all_specialists_without_truncation(build_runner):
    agent = build_runner(symbols=("SOLTMN",))
    result = agent.cycle()
    assert result["decisions"][0]["council"]["status"] == "ok"
    assert len(agent.client.calls) == 3
    assert all(c["as_of"] == NOW for c in agent.client.calls)
    assert agent.news.as_of == [NOW]


def test_council_semantic_cache_reuses_closed_bar_research_across_fresh_books(tmp_path):
    client = FakeClient()
    council = ResearchCouncil(client, tmp_path / "council.sqlite")
    context, pred = market_context(), forecast()
    news = FakeNews().snapshot("SOLTMN", NOW, 12)
    portfolio = {"positions": {}, "initial_cash_tmn": "5000000"}
    first = council.analyze(context, pred, news, portfolio, NOW)
    same = council.analyze(copy.deepcopy(context), copy.deepcopy(pred), copy.deepcopy(news), portfolio, NOW + 0.5)
    assert same["cached"] and same["key"] == first["key"] and len(client.calls) == 3
    changed_forecast = council.analyze(context, forecast(0.20), news, portfolio, NOW + 1)
    assert not changed_forecast["cached"] and changed_forecast["key"] != first["key"]
    fresh_quote = market_context(bid=110000)
    fresh_quote["cross_market"]["fetched_at"] = NOW + 2
    unchanged_research = council.analyze(fresh_quote, pred, news, portfolio, NOW + 2)
    assert unchanged_research["cached"] and unchanged_research["key"] == first["key"]
    assert len(client.calls) == 6
    changed_bar = copy.deepcopy(context)
    changed_bar["technical"]["one_hour"]["last_price"] = 105000.0
    changed_closed_data = council.analyze(changed_bar, pred, news, portfolio, NOW + 3)
    assert changed_closed_data["key"] != first["key"] and len(client.calls) == 9
    changed_news = copy.deepcopy(news)
    changed_news[0]["id"] = "material-new-evidence"
    updated = council.analyze(context, pred, changed_news, portfolio, NOW + 4)
    assert updated["key"] != first["key"] and len(client.calls) == 12
    from experiments.wallex_agent import llm
    with patch.object(llm, "SYSTEM", llm.SYSTEM + "\nA new prompt version"):
        changed_prompt = council.analyze(context, pred, news, portfolio, NOW + 5)
    assert changed_prompt["key"] != first["key"] and len(client.calls) == 15


def test_duplicate_decision_does_not_overwrite_original_artifact(build_runner):
    agent = build_runner(symbols=("SOLTMN",))
    first = agent.cycle()
    decision = first["decisions"][0]
    artifact = agent.artifacts / "decisions" / (decision["decision_id"] + ".json")
    original = artifact.read_bytes()
    second = agent.cycle()
    assert second["decisions"][0]["execution"]["duplicate"] is True
    assert artifact.read_bytes() == original
    assert len(agent.client.calls) == 3
    assert agent.portfolio.verify()["decision_count"] == 1


def test_generic_frozen_assets_supported_and_repeated_symbols_rejected(build_runner):
    symbols = ("SOLTMN", "XRPTMN", "1INCHTMN", "LINKTMN")
    agent = build_runner(symbols=symbols)
    result = agent.cycle()
    assert {x["symbol"] for x in result["decisions"]} == set(symbols)
    calls = len(agent.market.calls)
    with pytest.raises(ValueError, match="unique"):
        agent.cycle(symbols=("SOLTMN", "SOLTMN"))
    assert len(agent.market.calls) == calls
    with pytest.raises(ValueError, match="registered"):
        agent.cycle(symbols=("NOTLISTEDTMN",))


def test_provider_failure_preserves_safe_usage_provenance_and_blocks_entry(build_runner):
    agent = build_runner(symbols=("SOLTMN",), action="BUY", failed=True)
    result = agent.cycle()
    record = result["decisions"][0]
    assert record["proposal"]["action"] == "HOLD" and result["portfolio"]["fill_count"] == 0
    assert record["council"]["status"] == "incomplete"
    for analyst in record["council"]["assessments"]:
        assert analyst["error_code"] == "provider_or_secret_error"
        assert analyst["usage"] == {"input_tokens": 42, "output_tokens": 17}
        assert analyst["provenance"]["reserved_usd"] == 0.002


def test_shortlist_ranks_net_after_spread_instead_of_gross(build_runner):
    contexts = {"SOLTMN": market_context("SOLTMN", spread=0.004), "XRPTMN": market_context("XRPTMN", spread=0.0001)}
    predictions = {"SOLTMN": forecast(0.020), "XRPTMN": forecast(0.019)}
    agent = build_runner(contexts=contexts, predictions=predictions, research_limit=1)
    result = agent.cycle()
    assert [r["symbol"] for r in result["opportunity_ranking"]] == ["XRPTMN", "SOLTMN"]
    assert result["opportunity_ranking"][0]["expected_net_return_including_spread"] == pytest.approx(net_expected_return(0.019, 0.0001, agent.risk))
    assert result["specialist_research_targets"] == ["XRPTMN"]
    assert {c["symbol"] for c in agent.client.calls} == {"XRPTMN"}
    ranked_out = next(x for x in result["decisions"] if x["symbol"] == "SOLTMN")
    assert ranked_out["proposal"]["reason"] == ["not_among_top_current_net_opportunities"]


def test_stop_exit_works_even_when_model_council_is_unavailable(build_runner):
    agent = build_runner(symbols=("SOLTMN",), contexts={"SOLTMN": market_context(bid=95000)}, failed=True)
    seed_long(agent)
    result = agent.cycle()
    record = result["decisions"][0]
    assert record["proposal"]["forced_risk_exit"] is True
    assert record["proposal"]["action"] == "SELL"
    assert result["portfolio"]["positions"] == {}
    assert agent.client.calls == []
    assert result["audit"]["verified"]


def test_partial_stop_exit_can_continue_liquidation_on_next_fresh_cycle(build_runner):
    agent = build_runner(symbols=("SOLTMN",), contexts={"SOLTMN": market_context(bid=95000, depth=2.0)})
    seed_long(agent)
    first = agent.cycle()
    remaining = float(first["portfolio"]["positions"]["SOLTMN"]["quantity"])
    assert first["decisions"][0]["execution"]["event"]["fill"]["partial"]
    second = agent.cycle()
    assert float(second["portfolio"]["positions"]["SOLTMN"]["quantity"]) < remaining
    assert second["decisions"][0]["execution"].get("duplicate") is False


def test_forced_exit_is_not_cancelled_by_new_candle_bar(build_runner):
    initial = market_context(bid=95000)
    refreshed = market_context(bid=94900, bar=initial["technical"]["last_closed_at"] + 3600)
    agent = build_runner(symbols=("SOLTMN",), contexts={"SOLTMN": lambda call: initial if call == 1 else refreshed})
    seed_long(agent)
    result = agent.cycle()
    assert result["decisions"][0]["proposal"]["action"] == "SELL"
    assert result["portfolio"]["positions"] == {}


def test_open_positions_are_monitored_even_when_requested_scan_subset_omits_them(build_runner):
    agent = build_runner(contexts={"SOLTMN": market_context(bid=95000), "XRPTMN": market_context("XRPTMN")})
    seed_long(agent)
    result = agent.cycle(symbols=("XRPTMN",))
    assert "SOLTMN" in {r["symbol"] for r in result["decisions"]}
    assert "SOLTMN" not in result["portfolio"]["positions"]


def test_all_market_source_failures_record_errors_without_fake_orders(build_runner):
    secret_echo = "an arbitrary exception must not reach records"
    agent = build_runner(contexts={"SOLTMN": TimeoutError(secret_echo), "XRPTMN": OSError(secret_echo)})
    result = agent.cycle()
    assert result["source_errors"] == {"SOLTMN": "TimeoutError", "XRPTMN": "OSError"}
    assert result["decisions"] == [] and result["portfolio"]["fill_count"] == 0
    assert agent.client.calls == [] and secret_echo not in json.dumps(result)


def test_failed_execution_refresh_blocks_new_fill_and_sanitizes_exception(build_runner):
    agent = build_runner(symbols=("SOLTMN",), action="BUY", contexts={"SOLTMN": lambda n: market_context() if n == 1 else TimeoutError("sensitive-provider-echo")})
    result = agent.cycle()
    assert result["decisions"][0]["proposal"]["reason"] == ["execution_snapshot_unavailable"]
    assert result["portfolio"]["fill_count"] == 0
    assert "sensitive-provider-echo" not in json.dumps(result)


def test_bounded_provider_accepts_authorized_1200_calls_but_never_1201(tmp_path):
    secret = tmp_path / "fake.env"
    secret.write_text("ANTHROPIC_API_KEY=" + FAKE_KEY)
    secret.chmod(0o600)
    client = HaikuClient(secret, tmp_path / "budget.sqlite", max_calls=1200, transport=lambda *args: pytest.fail("No call expected"))
    assert client.budget_status()["max_calls"] == 1200
    assert client.budget_status()["max_usd"] == 2.0
    for value in [1201, 0, True, 1200.0]:
        with pytest.raises(ValueError, match="invalid_budget"):
            HaikuClient(secret, tmp_path / "budget.sqlite", max_calls=value)
    with pytest.raises(ValueError, match="invalid_budget"):
        HaikuClient(secret, tmp_path / "budget.sqlite", max_calls=1200, max_usd=2.01)


def test_council_provider_ledger_enforces_shared_attempt_cap_across_rounds(tmp_path):
    secret = tmp_path / "fake.env"
    secret.write_text("ANTHROPIC_API_KEY=" + FAKE_KEY)
    secret.chmod(0o600)
    sent = []
    def transport(body, headers, timeout):
        sent.append(body)
        evidence_id = json.loads(body["messages"][0]["content"])["news_evidence"][0]["id"]
        value = {"proposed_action": "HOLD", "sentiment": 0, "confidence": 0, "risk_flags": [], "evidence_ids": [evidence_id], "thesis": "Insufficient evidence", "invalidation": "New evidence", "horizon_hours": 8}
        return {"model": MODEL, "stop_reason": "tool_use", "content": [{"type": "tool_use", "name": "emit_assessment", "input": value}], "usage": {"input_tokens": 100, "output_tokens": 50}}
    client = HaikuClient(secret, tmp_path / "budget.sqlite", max_calls=2, transport=transport)
    council = ResearchCouncil(client, tmp_path / "council.sqlite")
    context, pred = market_context(), forecast()
    news = FakeNews().snapshot("SOLTMN", NOW, 12)
    portfolio = {"positions": {}, "initial_cash_tmn": "5000000"}
    first = council.analyze(context, pred, news, portfolio, NOW)
    second = council.analyze(context, pred, news, portfolio, NOW + 1)
    assert first["status"] == second["status"] == "incomplete"
    assert len(sent) == client.budget_status()["attempts"] == 2
    assert client.budget_status()["charged_or_reserved_usd"] == pytest.approx(2 * 0.000035)
    assert any(a.get("error_code") == "budget_exhausted" for a in first["assessments"])


def test_unchanged_paper_poll_does_not_retry_ambiguous_research_posts(tmp_path):
    secret = tmp_path / "fake.env"
    secret.write_text("ANTHROPIC_API_KEY=" + FAKE_KEY)
    secret.chmod(0o600)
    sent = []
    def ambiguous(*args):
        sent.append(1)
        raise TimeoutError("POST may have reached provider; do not automatically retry unchanged research")
    client = HaikuClient(secret, tmp_path / "budget.sqlite", max_calls=1200, transport=ambiguous)
    council = ResearchCouncil(client, tmp_path / "council.sqlite")
    context, pred = market_context(), forecast()
    news = FakeNews().snapshot("SOLTMN", NOW, 12)
    portfolio = {"positions": {}, "initial_cash_tmn": "5000000"}
    first = council.analyze(context, pred, news, portfolio, NOW)
    second = council.analyze(context, pred, news, portfolio, NOW + 300)
    assert first["status"] == second["status"] == "incomplete"
    assert len(sent) == client.budget_status()["attempts"] == 3
    assert second["cached"] is True
    assert client.budget_status()["charged_or_reserved_usd"] > 0


def test_incomplete_cache_expires_after_600_seconds_and_new_evidence_or_source_invalidates(tmp_path):
    secret = tmp_path / "fake.env"
    secret.write_text("ANTHROPIC_API_KEY=" + FAKE_KEY)
    secret.chmod(0o600)
    sent = []
    def ambiguous(*args):
        sent.append(1)
        raise TimeoutError("Ambiguous POST")
    client = HaikuClient(secret, tmp_path / "budget.sqlite", max_calls=1200, transport=ambiguous)
    council = ResearchCouncil(client, tmp_path / "council.sqlite")
    context, pred = market_context(), forecast()
    news = FakeNews().snapshot("SOLTMN", NOW, 12)
    portfolio = {"positions": {}, "initial_cash_tmn": "5000000"}
    council.analyze(context, pred, news, portfolio, NOW)
    assert len(sent) == 3
    retried = council.analyze(context, pred, news, portfolio, NOW + 601)
    assert retried["status"] == "incomplete" and retried["cached"] is False
    assert len(sent) == 6
    new_news = copy.deepcopy(news)
    new_news[0]["id"] = "materially-new-evidence"
    changed_news = council.analyze(context, pred, new_news, portfolio, NOW + 602)
    assert changed_news["cached"] is False and len(sent) == 9
    new_source = copy.deepcopy(context)
    new_source["cross_market"]["source"] = "different-source-provenance"
    changed_source = council.analyze(new_source, pred, news, portfolio, NOW + 603)
    assert changed_source["cached"] is False and len(sent) == 12
    assert client.budget_status()["attempts"] == 12
    assert client.budget_status()["charged_or_reserved_usd"] > 0
