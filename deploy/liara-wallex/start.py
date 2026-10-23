"""Container entrypoint on Liara: the dashboard always runs; the trading engine runs only when WALLEX_ENGINE=on.

Safety: unless WALLEX_LIVE_ENABLED=1, the kill file /var/lib/wallex-arena/LIVE_DISABLED is (re)created at every boot,
so a fresh deploy is always DISARMED. Real orders need WALLEX_LIVE_ENABLED=1 (Liara env, set by an operator), an API
key, AND the kill file removed (dashboard "Arm", admin password + typed confirmation). Disarming is one click.

Env: WALLEX_ADMIN_PASSWORD (dashboard); AI: LLM_PROVIDER (anthropic|boxapi, the other is the fallback), ANTHROPIC_BASE_URL +
ANTHROPIC_AUTH_TOKEN (the Arvan LLM gate), WALLEX_ANTHROPIC_MODEL, BOXAPI_KEYS / BOXAPI_BASE_URL, WALLEX_BOXAPI_MODEL; WALLEX_API_KEY / WALLEX_SUBACCOUNT_ID
(credit sub-account), WALLEX_ENGINE (off|on), WALLEX_LIVE_ENABLED (0|1), LLM_PROVIDER, WALLEX_GLOBAL_REST (optional).
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
STATE_LINK = Path(os.environ.get("WALLEX_STATE_DIR") or "/var/lib/wallex-arena")   # engine root; symlinked to the disk
KILL = STATE_LINK / "LIVE_DISABLED"
os.environ["WALLEX_KILL_FILE"] = str(KILL)                                        # live.py / watchdog read the same file
os.environ.setdefault("WALLEX_MCP_LOG", str(STATE_LINK / "mcp_calls.jsonl"))       # MCP calls from engine + web, for /market
DISK = Path(os.environ.get("WALLEX_DISK_DIR") or "/data")
RUN = Path(os.environ.get("WALLEX_RUN_DIR") or ("/run/wallex" if os.access("/run", os.W_OK) else "/tmp/wallex"))
SELFCHECK = RUN / "selfcheck.json"
PROC = RUN / "engine-proc.json"


from experiments.wallex_arena import obs  # noqa: E402

obs.configure("supervisor")
CHILDREN: dict[str, subprocess.Popen] = {}
STOPPING = threading.Event()


def log(*a, event="supervisor.info", level="info", **fields):
    obs.log(event, level, " ".join(str(x) for x in a), **fields)


def prepare_state() -> Path:
    if STATE_LINK.is_dir() and not STATE_LINK.is_symlink() and os.access(STATE_LINK, os.W_OK):
        state = STATE_LINK                     # state dir given directly (non-root image: WALLEX_STATE_DIR=/data/wallex-arena)
    else:
        state = DISK / "wallex-arena" if DISK.is_dir() and os.access(DISK, os.W_OK) else Path("/tmp/wallex-arena")
    state.mkdir(parents=True, exist_ok=True)
    if state == STATE_LINK:
        pass
    elif STATE_LINK.is_symlink() or STATE_LINK.exists():
        if STATE_LINK.resolve() != state.resolve():
            STATE_LINK.unlink()
    if not STATE_LINK.exists():
        STATE_LINK.parent.mkdir(parents=True, exist_ok=True)
        STATE_LINK.symlink_to(state)
    if os.environ.get("WALLEX_LIVE_ENABLED") != "1":
        KILL.touch()
    log("state", state, "persistent" if str(state).startswith(str(DISK)) else f"EPHEMERAL (no {DISK} disk)", "| kill file", KILL.exists())
    return state


def write_secrets():
    RUN.mkdir(parents=True, exist_ok=True)
    os.chmod(RUN, 0o700)
    env_file = RUN / "wallex.env"
    fd = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"WALLEX_API_KEY={os.environ.get('WALLEX_API_KEY', '')}\nWALLEX_SUBACCOUNT_ID={os.environ.get('WALLEX_SUBACCOUNT_ID', '')}\n")
    return env_file


CHALLENGE_END_DEFAULT = "2026-10-23T20:00:00Z"   # SET WALLEX_CHALLENGE_END_UTC TO THE OFFICIAL END TIME (placeholder: 23:30 Tehran on 1 Aban)


def challenge_settings() -> dict:
    """End-of-challenge settlement in TMN: no new positions `no_new_entries_minutes` before the end, cancel everything
    and sell all coins to TMN from `flatten_minutes` before the end (see engine._flatten)."""
    return {"challenge_end_utc": os.environ.get("WALLEX_CHALLENGE_END_UTC") or CHALLENGE_END_DEFAULT,
            "challenge_end_is_placeholder": not os.environ.get("WALLEX_CHALLENGE_END_UTC"),
            "flatten_minutes": float(os.environ.get("WALLEX_FLATTEN_MINUTES") or 60),
            "no_new_entries_minutes": float(os.environ.get("WALLEX_NO_NEW_ENTRIES_MINUTES") or 120)}


def write_config(env_file: Path) -> Path:
    cfg = json.loads((APP / "liara/live-config.template.json").read_text())
    provider = (os.environ.get("LLM_PROVIDER") or "anthropic").lower()
    model = (os.environ.get("WALLEX_BOXAPI_MODEL") or "openai/gpt-6-luna") if provider == "boxapi" else (os.environ.get("WALLEX_ANTHROPIC_MODEL") or "claude-haiku-5-5")
    # Position sizing: total maker inventory cap as a fraction of equity (paper-tested 0.45; live default 0.35, the
    # recommended 30-40% range). Per-market size is separately capped by the AI guardrail (size_max 0.08).
    exposure = min(max(float(os.environ.get("WALLEX_MAX_EXPOSURE") or 0.35), 0.05), 0.95)
    strategy = (os.environ.get("WALLEX_LIVE_STRATEGY") or "ai_maker").lower()
    alternatives = cfg.pop("alternatives", {}) or {}
    if strategy != "ai_maker":
        if strategy not in alternatives:
            raise SystemExit(f"WALLEX_LIVE_STRATEGY={strategy!r} is not one of {['ai_maker', *alternatives]}")
        cfg["accounts"] = [alternatives[strategy]]
    cfg["live_strategy"] = strategy
    for a in cfg["accounts"]:
        a["params"]["model"] = model
        if a["class"] == "AIMaker":
            a["params"].setdefault("maker", {})["max_exposure"] = exposure
        risk = a.setdefault("risk", {})               # engine-level caps apply to every strategy (maker and macro)
        risk["max_exposure"] = min(float(risk.get("max_exposure", 0.95)), exposure)
        risk["max_position_frac"] = min(float(risk.get("max_position_frac", 0.35)), exposure)
    cfg.update(challenge_settings())
    cfg.pop("liquidate_at_utc", None)
    cfg["wallex_env_file"] = str(env_file)
    cfg["kill_file"] = str(KILL)
    path = RUN / "live-config.json"
    path.write_text(json.dumps(cfg, indent=1))
    return path


def probe(url, headers=None, timeout=6):
    t = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **(headers or {})})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read(2048)
            return {"ok": True, "status": r.status, "ms": round(1000 * (time.time() - t))}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "ms": round(1000 * (time.time() - t))}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "ms": round(1000 * (time.time() - t))}


def selfcheck() -> dict:
    """Public, read-only connectivity checks from inside the runtime (no authenticated Wallex call)."""
    res = {"ts": time.time(),
           "wallex_markets": probe("https://api.wallex.ir/v1/markets"),
           "wallex_depth_btc": probe("https://api.wallex.ir/v1/depth?symbol=BTCTMN"),
           "binance": probe("https://api.binance.com/api/v3/ping"),
           "mexc": probe("https://api.mexc.com/api/v3/ping"),
           "kucoin": probe("https://api.kucoin.com/api/v1/timestamp"),
           "coingecko": probe("https://api.coingecko.com/api/v3/ping")}
    for url in filter(None, (os.environ.get("WALLEX_EXTRA_PROBES") or "").split(",")):   # optional, e.g. a proxy host
        res["extra " + url.strip()] = probe(url.strip())
    keys = [k for k in (os.environ.get("BOXAPI_KEYS") or "").split(",") if k.strip()]
    base = (os.environ.get("BOXAPI_BASE_URL") or "https://ai.boxapi.ir/ai/v1").rstrip("/")
    res["boxapi_models"] = probe(base + "/models", {"Authorization": "Bearer " + keys[0].strip()}) if keys else {"ok": False, "error": "BOXAPI_KEYS not set"}
    res["boxapi_keys"] = len(keys)
    if os.environ.get("WALLEX_GLOBAL_REST"):
        res["global_source"] = os.environ["WALLEX_GLOBAL_REST"]
    elif res["binance"]["ok"]:
        res["global_source"] = "https://api.binance.com"
    elif res["mexc"]["ok"]:
        res["global_source"] = "https://api.mexc.com"
    else:
        res["global_source"] = None
    SELFCHECK.write_text(json.dumps(res))
    summary = {k: (v.get("status") or v.get("error")) if isinstance(v, dict) else v for k, v in res.items() if k != "ts"}
    log("connectivity self-check", event="health.selfcheck", level="info" if res["wallex_markets"]["ok"] else "warn", checks=summary)
    return res


def selfcheck_job(payload):
    """Queued retry of the connectivity check: raises (-> backoff, then dead-letter) while Wallex is unreachable."""
    res = selfcheck()
    if not res["wallex_markets"]["ok"]:
        raise RuntimeError("Wallex public API unreachable: %s" % (res["wallex_markets"].get("status") or res["wallex_markets"].get("error")))
    return {k: v for k, v in res.items() if k != "ts"}


def supervise(name, argv, env, proc_state):
    starts = 0
    while not STOPPING.is_set():
        starts += 1
        p = subprocess.Popen(argv, cwd=APP, env=env)
        CHILDREN[name] = p
        proc_state[name] = {"running": True, "pid": p.pid, "starts": starts, "since": time.time()}
        PROC.write_text(json.dumps(proc_state))
        log(name, "started", event="process.start", process=name, pid=p.pid, attempt=starts)
        code = p.wait()
        proc_state[name] = {"running": False, "starts": starts, "last_exit": code, "exited_at": time.time()}
        PROC.write_text(json.dumps(proc_state))
        if STOPPING.is_set():
            break
        wait = min(300, 20 * 2 ** min(starts - 1, 4))     # crash loop: back off 20 s .. 5 min
        log(name, "exited", event="process.exit", level="error", process=name, exit_code=code, attempt=starts, backoff_ms=wait * 1000)
        STOPPING.wait(wait)


def start_engine(check, env_file, config):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore", WALLEX_ENV_FILE=str(env_file))
    if check.get("global_source"):
        env["WALLEX_GLOBAL_REST"] = check["global_source"]
    py = sys.executable
    state = {}
    threading.Thread(target=supervise, daemon=True, args=("engine", [py, "-m", "experiments.wallex_arena", "run", str(config), str(STATE_LINK), "/dev/null"], env, state)).start()
    threading.Thread(target=supervise, daemon=True, args=("watchdog", [py, "-m", "experiments.wallex_arena.watchdog", str(STATE_LINK / "status.json"), str(STATE_LINK / "watchdog.json")], env, state)).start()


def main():
    obs.install_crash_handlers()
    obs.set_context("boot-" + obs.new_id(4))
    log("starting", event="app.start", engine=os.environ.get("WALLEX_ENGINE", "off"), llm_provider=os.environ.get("LLM_PROVIDER"))
    prepare_state()
    env_file = write_secrets()
    config = write_config(env_file)
    PROC.write_text(json.dumps({}))
    engine_on = (os.environ.get("WALLEX_ENGINE") or "off").lower() in ("on", "1", "true", "yes")
    log("engine", "ON" if engine_on else "OFF (deployed, not running; set WALLEX_ENGINE=on to start)",
        "| WALLEX_LIVE_ENABLED", os.environ.get("WALLEX_LIVE_ENABLED", "unset"))

    from liara import web
    server = web.make_server(int(os.environ.get("PORT", "8080")), {"connectivity.selfcheck": selfcheck_job})

    def boot():
        check = selfcheck()
        if not check["wallex_markets"]["ok"]:
            web.QUEUE.enqueue("connectivity.selfcheck", {}, idem="boot-" + str(int(check["ts"])), max_attempts=8, delay=30)
        if engine_on:
            start_engine(check, env_file, config)
    threading.Thread(target=boot, daemon=True, name="boot").start()

    def shutdown(signum, frame):
        """Graceful stop: children get SIGTERM, the job worker re-queues in-flight jobs, the server stops."""
        log("shutdown", event="app.stop", level="warn", signal=signum)
        STOPPING.set()
        for name, p in CHILDREN.items():
            if p.poll() is None:
                p.terminate()
        for p in CHILDREN.values():
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        web.QUEUE.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    server.serve_forever()
    log("stopped", event="app.stop")


if __name__ == "__main__":
    main()
