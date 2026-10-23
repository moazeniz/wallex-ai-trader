import json
import socket

import pytest

from experiments.wallex_arena import obs, resilience
from experiments.wallex_arena.account import DecisionLog
from experiments.wallex_arena.jobqueue import JobQueue
from experiments.wallex_arena.live import AmbiguousOrder, WallexPrivateAPI
from experiments.wallex_arena.resilience import CircuitBreaker, CircuitOpen, HTTPStatusError

REQUIRED = {"ts", "level", "app", "env", "release", "event", "request_id", "trace_id", "span_id", "session", "msg"}


def lines(capsys):
    return [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]


def test_log_line_has_required_keys_redacts_and_truncates(capsys):
    obs.set_context("req-1", "00-" + "a" * 32 + "-" + "b" * 16 + "-01", session="user@x")
    obs.log("http.request", "info", "GET /", api_key="sk-secret", authorization="Bearer x", tokens_in=12, reason="r" * 1000)
    (rec,) = lines(capsys)
    assert REQUIRED <= set(rec) and rec["request_id"] == "req-1" and rec["trace_id"] == "a" * 32 and rec["span_id"] != "b" * 16
    assert rec["api_key"] == "[redacted]" and rec["authorization"] == "[redacted]" and rec["tokens_in"] == 12
    assert len(rec["reason"]) < 400 and "sha:" in rec["reason"] and rec["session"] != "user@x"
    assert obs.traceparent().startswith("00-" + "a" * 32)


def test_log_level_filter(capsys, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "warn")
    obs.log("external.call", "debug", "ok")
    obs.log("external.call", "info", "ok")
    obs.log("external.call", "warn", "slow")
    assert [r["level"] for r in lines(capsys)] == ["warn"]


def test_retry_transient_then_success_logs_every_attempt(capsys):
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise HTTPStatusError(503, "busy")
        return "ok"
    assert resilience.call("u1", fn, max_attempts=3, sleep=sleeps.append) == "ok"
    out = [r for r in lines(capsys) if r["event"] == "external.call"]
    assert len(calls) == 3 and len(sleeps) == 2 and [r["attempt"] for r in out] == [1, 2]   # success is debug (filtered)


def test_retry_honours_retry_after_and_skips_permanent_errors():
    sleeps = []
    with pytest.raises(HTTPStatusError):
        resilience.call("u2", lambda: (_ for _ in ()).throw(HTTPStatusError(429, "", retry_after="3")), max_attempts=2, sleep=sleeps.append)
    assert sleeps and sleeps[0] >= 3
    n = []
    with pytest.raises(HTTPStatusError):
        resilience.call("u3", lambda: n.append(1) or (_ for _ in ()).throw(HTTPStatusError(400, "bad")), max_attempts=4, sleep=lambda s: None)
    assert len(n) == 1


def test_backoff_is_full_jitter_and_capped():
    assert all(0 <= resilience.backoff(a, 0.5, 4.0) <= min(4.0, 0.5 * 2 ** a) for a in range(10) for _ in range(20))


def test_circuit_breaker_opens_half_opens_and_closes():
    now = [0.0]
    b = CircuitBreaker("x", threshold=2, reset_after=10, clock=lambda: now[0])
    b.before(); b.failure(); b.before(); b.failure()
    assert b.state == "open"
    with pytest.raises(CircuitOpen) as e:
        b.before()
    assert isinstance(e.value, OSError)
    now[0] = 11
    b.before()
    assert b.state == "half_open"
    with pytest.raises(CircuitOpen):
        b.before()                     # only one probe at a time
    b.success()
    assert b.state == "closed"


def test_order_placement_is_never_resent(capsys):
    sent = []

    def transport(method, url, headers, data, timeout):
        sent.append(method)
        if method == "POST":
            raise socket.timeout("timed out")
        return 404, b'{"success": false, "message": "not found"}'
    api = WallexPrivateAPI("k", transport=transport, sleep=lambda s: None)
    with pytest.raises(AmbiguousOrder):
        api.create_order("BTCTMN", "BUY", "LIMIT", "100", "1", "ARENA_1")
    assert sent.count("POST") == 1 and sent.count("GET") == 3
    assert "k" not in [r.get("api_key") for r in lines(capsys)]


def test_reads_retry_network_errors():
    n = []

    def transport(method, url, headers, data, timeout):
        n.append(1)
        if len(n) == 1:
            raise ConnectionResetError()
        return 200, b'{"success": true, "result": {"balances": {}}}'
    assert WallexPrivateAPI("k", transport=transport, sleep=lambda s: None).balances() is not None and len(n) == 2


def test_decision_log_records_are_mirrored(tmp_path, capsys):
    DecisionLog(tmp_path / "acct" / "live.jsonl").append({"ts": 1, "type": "live_refused", "op": "place_buy", "symbol": "BTCTMN"})
    DecisionLog(tmp_path / "engine.jsonl").append({"ts": 1, "type": "tick_error", "error": "KeyError", "trace": "Traceback..."})
    recs = {r["event"]: r for r in lines(capsys)}
    assert recs["order.live_refused"]["level"] == "warn" and recs["order.live_refused"]["account"] == "acct"
    assert recs["engine.tick_error"]["level"] == "error" and recs["engine.tick_error"]["stack"] == "Traceback..."


def test_job_queue_retry_dead_letter_replay_discard_and_idempotency(tmp_path, capsys):
    now = [1000.0]
    q = JobQueue(tmp_path / "jobs.sqlite", clock=lambda: now[0])
    runs = []

    def flaky(p):
        runs.append(p)
        if len(runs) < 2:
            raise RuntimeError("transient")
        return {"done": True}
    q.register("flaky", flaky)
    q.register("broken", lambda p: 1 / 0)
    a = q.enqueue("flaky", {"x": 1}, idem="same")
    assert q.enqueue("flaky", {"x": 1}, idem="same") == a
    b = q.enqueue("broken", {}, max_attempts=2)
    for _ in range(6):
        now[0] += 1000
        q.run_once()
    assert q.get(a)["state"] == "succeeded" and q.get(a)["result"] == {"done": True}
    assert q.get(b)["state"] == "dead" and "ZeroDivisionError" in q.get(b)["last_error"]
    events = [r["event"] for r in lines(capsys)]
    assert "job.retry" in events and "job.dead_letter" in events
    assert q.replay(b) and q.get(b)["state"] == "queued"
    now[0] += 1000; q.run_once(); now[0] += 1000; q.run_once()
    assert q.discard(b) and q.get(b)["state"] == "discarded"


def test_job_queue_requeues_interrupted_jobs(tmp_path):
    q = JobQueue(tmp_path / "j.sqlite")
    j = q.enqueue("k", {})
    with q._db() as db:
        db.execute("UPDATE jobs SET state='running' WHERE id=?", (j,))
    assert JobQueue(tmp_path / "j.sqlite").get(j)["state"] == "queued"
