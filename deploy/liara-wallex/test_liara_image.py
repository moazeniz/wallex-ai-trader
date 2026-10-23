"""Smoke checks that run inside the built Liara image: everything the engine, watchdog and dashboard import is present,
the engine's start-up objects build offline, and the live config template resolves to the LLM_PROVIDER model."""
import importlib
import json
import pkgutil
from pathlib import Path

import experiments.wallex_arena as arena

APP = Path(__file__).resolve().parent.parent


def test_every_engine_module_imports():
    names = [m.name for m in pkgutil.iter_modules(arena.__path__)]
    assert {"engine", "live", "live_account", "watchdog", "ai", "ai_strategy", "boxapi", "anthropic_http", "feed"} <= set(names)
    for n in names:
        importlib.import_module("experiments.wallex_arena." + n)
    importlib.import_module("experiments.wallex_arena.__main__")


def test_engine_newsfeed_builds(tmp_path):
    from experiments.wallex_arena.engine import NewsFeed      # imports experiments.wallex_agent.news at construction
    NewsFeed(tmp_path / "news.sqlite")


def test_live_template_account_builds_with_provider_switch(tmp_path, monkeypatch):
    from experiments.wallex_arena.ai_strategy import AIMaker
    from experiments.wallex_arena.engine import NewsFeed, Risk
    cfg = json.loads((APP / "liara/live-config.template.json").read_text())
    (spec,) = cfg["accounts"]
    assert spec["live"] == "real" and spec["class"] == "AIMaker"
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://gate.invalid/meow-llm")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "t")
    monkeypatch.setenv("BOXAPI_KEYS", "k")
    st = AIMaker(spec["name"], tmp_path, Path("/dev/null"), NewsFeed(tmp_path / "n.sqlite"), **spec["params"])
    assert [p for p, _ in st.trader.routes] == ["anthropic", "boxapi"] and st.trader.model == spec["params"]["model"]   # the template model (Haiku since 2026-10-23)
    Risk(**spec.get("risk", {}))


def test_dashboard_and_entrypoint_import(monkeypatch, tmp_path):
    monkeypatch.setenv("WALLEX_STATE_DIR", str(tmp_path))
    importlib.import_module("liara.web")
    src = (APP / "liara/start.py").read_text()
    compile(src, "start.py", "exec")


def _server(tmp_path, monkeypatch):
    import base64
    import threading
    monkeypatch.setenv("WALLEX_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("WALLEX_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("WALLEX_ADMIN_PASSWORD", "pw")
    monkeypatch.delenv("WALLEX_API_KEY", raising=False)
    web = importlib.reload(importlib.import_module("liara.web"))
    srv = web.make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    auth = {"Authorization": "Basic " + base64.b64encode(b"admin:pw").decode()}
    return web, srv, f"http://127.0.0.1:{srv.server_address[1]}", auth


def _req(url, headers=None, method="GET", body=None):
    import urllib.error
    import urllib.request
    r = urllib.request.Request(url, headers=headers or {}, method=method, data=body)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_health_ready_request_id_jobs_and_error_handler(tmp_path, monkeypatch, capsys):
    web, srv, base, auth = _server(tmp_path, monkeypatch)
    try:
        code, h, _ = _req(base + "/healthz", {"X-Request-ID": "abc-123"})
        assert code == 200 and h["X-Request-ID"] == "abc-123"
        code, _, raw = _req(base + "/readyz")
        ready = json.loads(raw)
        assert code == 200 and ready["ready"] and ready["queue"]["ok"] and "circuits" in ready and ready["engine"]["armed"] is False
        assert _req(base + "/api/jobs")[0] == 401
        post = dict(auth, **{"X-Requested-With": "dashboard"})
        code, _, raw = _req(base + "/api/balances?async=1", post, "POST", b"{}")
        job_id = json.loads(raw)["job_id"]
        assert code == 202
        for _ in range(3):
            with web.QUEUE._db() as db:
                db.execute("UPDATE jobs SET next_at=0 WHERE id=?", (job_id,))
            web.QUEUE.run_once()
        job = json.loads(_req(base + f"/api/jobs/{job_id}", auth)[2])
        assert job["state"] == "dead" and "WALLEX_API_KEY" in job["last_error"]          # no key: retried, then dead-lettered
        assert json.loads(_req(base + f"/api/jobs/{job_id}/replay", post, "POST", b"{}")[2])["ok"]
        assert json.loads(_req(base + "/api/disarm", post, "POST", b"{}")[2])["state"]["kill_file_present"]
        monkeypatch.setattr(web, "api_live", lambda: 1 / 0)
        code, h, raw = _req(base + "/api/live", auth)
        body = json.loads(raw)
        assert code == 500 and body["error"] == "internal_error" and body["request_id"] == h["X-Request-ID"] and "Traceback" not in raw.decode()
    finally:
        srv.shutdown()
        web.QUEUE.stop()
    logs = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    events = {r["event"] for r in logs}
    assert {"http.request", "auth.fail", "job.retry", "job.dead_letter", "control.disarm", "http.error"} <= events
    assert any(r["event"] == "http.error" and "stack" in r for r in logs)


def test_nav_pages_and_market_page_load_for_the_runtime_user(tmp_path, monkeypatch):
    """Every page in the dashboard nav returns 200 when logged in (the image also runs this as uid 10001)."""
    web, srv, base, auth = _server(tmp_path, monkeypatch)
    try:
        for path in web.NAV_PATHS:
            code, _, raw = _req(base + path, auth)
            assert code == 200, (path, code, raw[:200])
        assert b"Market (MCP)" in _req(base + "/market", auth)[2]
        live = json.loads(_req(base + "/api/live", auth)[2])
        assert "challenge" in live and "phase" in live["challenge"]
        assert _req(base + "/api/mcp/calls", auth)[0] == 200
        ready = json.loads(_req(base + "/readyz")[2])
        assert ready["snapshot"]["ok"]
    finally:
        srv.shutdown()
        web.QUEUE.stop()


def test_readyz_snapshot_rules(tmp_path, monkeypatch):
    web = importlib.import_module("liara.web")
    monkeypatch.setattr(web, "SNAP", tmp_path / "none")
    assert web.snapshot_status() == {"ok": True, "bundled": False}            # no bundled snapshot: still ready
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "MANIFEST.json").write_text(json.dumps({"files": {"a.json": {}, "b.json": {}}}))
    (snap / "a.json").write_text("{}")
    monkeypatch.setattr(web, "SNAP", snap)
    st = web.snapshot_status()
    assert st["ok"] is False and st["missing_or_unreadable"] == ["b.json"]   # bundled but incomplete: not ready


def _snap_tar(files: dict, ts: float, manifest_files: dict | None = None, extra_members: dict | None = None) -> bytes:
    import hashlib
    import io
    import tarfile
    man = {"kind": "push", "snapshot_ts": ts, "snapshot_utc": "test",
           "files": manifest_files if manifest_files is not None else {k: {"sha256": hashlib.sha256(v).hexdigest()} for k, v in files.items()}}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in {**files, "MANIFEST.json": json.dumps(man).encode(), **(extra_members or {})}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_snapshot_push_ingest_rules_and_serving(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setenv("SNAPSHOT_PUSH_TOKEN", "tok")
    monkeypatch.setenv("WALLEX_SNAPSHOT_DIR", str(tmp_path / "snaps"))
    web, srv, base, auth = _server(tmp_path, monkeypatch)
    url = base + "/api/snapshot/push"
    hdr = {"Authorization": "Bearer tok", "Content-Type": "application/gzip"}
    good = _snap_tar({"www/data/leaderboard.json": b'[{"account":"pushed-acct"}]', "report.md": b"# pushed report"}, 2_000_000_000)
    try:
        assert _req(url, {"Content-Type": "application/gzip"}, "POST", good)[0] == 401
        assert _req(url, {"Authorization": "Bearer nope"}, "POST", good)[0] == 401
        assert _req(url, dict(hdr, **{"X-Snapshot-Sha256": "0" * 64}), "POST", good)[0] == 400
        evil = _snap_tar({"a.json": b"{}"}, 2e9, extra_members={"../escape.txt": b"x"})
        assert _req(url, hdr, "POST", evil)[0] == 400 and not (tmp_path / "escape.txt").exists()
        assert _req(url, hdr, "POST", _snap_tar({"a.json": b"{}"}, 2e9, manifest_files={"a.json": {"sha256": "bad"}}))[0] == 400
        assert _req(url, hdr, "POST", _snap_tar({"a.json": b"{}"}, 2e9, extra_members={"unlisted.json": b"{}"}))[0] == 400
        code, _, raw = _req(url, dict(hdr, **{"X-Snapshot-Sha256": hashlib.sha256(good).hexdigest()}), "POST", good)
        assert code == 200 and json.loads(raw)["accepted"]
        assert _req(url, hdr, "POST", good)[0] == 409                                         # same version
        assert _req(url, hdr, "POST", _snap_tar({"x.json": b"{}"}, 1_000_000_000))[0] == 409   # older
        assert b"pushed-acct" in _req(base + "/data/leaderboard.json", auth)[2]               # served from the pushed snapshot
        assert b"pushed report" in _req(base + "/report", auth)[2]
        info = json.loads(_req(base + "/api/snapshot/info", auth)[2])
        assert info["source"] == "pushed" and info["snapshot_ts"] == 2_000_000_000
        home = _req(base + "/", auth)[2].decode()
        assert 'class="mm-foot"' in home and 'id="mm-upd"' in home and "/api/snapshot/info" in home   # small footer note
        assert "mm-banner" not in home and "PAPER / SHADOW TRADING RESULTS" not in home                   # no top banner
        newer = _snap_tar({"www/data/leaderboard.json": b'[{"account":"newer"}]'}, 2_000_000_100)
        assert _req(url, hdr, "POST", newer)[0] == 200
        assert b"newer" in _req(base + "/data/leaderboard.json", auth)[2]
        assert _req(base + "/docs", auth)[0] == 200                                          # missing in the push: bundled fallback
        monkeypatch.setattr(web, "PUSH_MAX_BYTES", 10)
        assert _req(url, hdr, "POST", good)[0] == 413
    finally:
        srv.shutdown()
        web.QUEUE.stop()


def test_live_strategy_switch_macro_allocator_with_provider_switch_and_caps(tmp_path, monkeypatch):
    """WALLEX_LIVE_STRATEGY=macro runs the macro allocator; it uses the LLM provider switch, the MCP research step and the
    35% caps like the default AI maker."""
    import importlib.util
    from experiments.wallex_arena.ai_strategy import AITrader
    from experiments.wallex_arena.engine import NewsFeed, Risk
    monkeypatch.setenv("WALLEX_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("WALLEX_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("WALLEX_DISK_DIR", str(tmp_path / "disk"))
    monkeypatch.setenv("WALLEX_LIVE_STRATEGY", "macro")
    monkeypatch.setenv("WALLEX_MAX_EXPOSURE", "0.35")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://gate.invalid/meow-llm")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "t")
    monkeypatch.setenv("BOXAPI_KEYS", "k")
    monkeypatch.setenv("WALLEX_MCP_TOOLS", "1")
    monkeypatch.delenv("WALLEX_ANTHROPIC_MODEL", raising=False)
    spec = importlib.util.spec_from_file_location("liara_start", APP / "liara/start.py")
    start = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(start)
    start.RUN.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(start.write_config(start.RUN / "wallex.env").read_text())
    (acct,) = cfg["accounts"]
    assert cfg["live_strategy"] == "macro" and acct["name"] == "live_macro" and acct["params"]["style"] == "macro" and acct["live"] == "real"
    assert acct["risk"]["max_exposure"] <= 0.35 and acct["risk"]["max_position_frac"] <= 0.35 and acct["params"]["universe"]
    st = AITrader(acct["name"], tmp_path, Path("/dev/null"), NewsFeed(tmp_path / "n.sqlite"), **acct["params"])
    assert st.trader.routes == [("anthropic", "claude-haiku-5-5"), ("boxapi", "openai/gpt-6-luna")] and st.trader.mcp is not None
    assert Risk(**acct["risk"]).max_exposure <= 0.35
    monkeypatch.setenv("WALLEX_LIVE_STRATEGY", "ai_maker")
    (maker,) = json.loads(start.write_config(start.RUN / "wallex.env").read_text())["accounts"]
    assert maker["class"] == "AIMaker" and maker["params"]["maker"]["max_exposure"] == 0.35 and maker["risk"]["max_exposure"] == 0.35
