"""Build and deploy meowmeow-wallex to Liara (run as root on the build host from a repository checkout).

usage: deploy_liara.py build            assemble $WALLEX_LIARA_BUILD (code + fresh results snapshot; snapshot needs the
                                        paper arena's state on this host, see make_snapshot.py)
       deploy_liara.py assemble OUT     code-only build context in OUT (used by the repository-root Dockerfile)
       deploy_liara.py deploy [MESSAGE] upload $WALLEX_LIARA_BUILD as a Docker source and release it (engine OFF, DISARMED)
       deploy_liara.py setenv K=V ...   merge env vars into the app (Liara replaces the whole list; this keeps the rest).
                                        The app restarts. Values are never printed.
       deploy_liara.py releases         list recent releases (id, tag, state, type, date)
       deploy_liara.py rollback ID      roll back to an earlier READY release (POST /v1/releases/{ID}/rollback)
Secrets are read from $BOXAPI_ENV_FILE (default /root/.boxapi.env) and $WALLEX_ADMIN_ENV_FILE
(default /root/.wallex-liara-admin.env) and are never printed. The Liara API helper (`lia.py`, or `liara_cli.py` from
the liara-sre repository) is imported from $LIA_DIR (default /root); it reads LIARA_API_TOKEN / LIARA_TEAM_ID.
"""
import io
import json
import os
import secrets
import shutil
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

sys.path.insert(0, os.environ.get("LIA_DIR") or "/root")   # lia.py (Liara API helper)
HERE = Path(__file__).resolve().parent
TREE = HERE.parent.parent                     # repository root (or the liara-live worktree)
OUT = Path(os.environ.get("WALLEX_LIARA_BUILD") or "/opt/wallex/liara-build")
APP = os.environ.get("WALLEX_LIARA_APP") or "meowmeow-wallex"
ADMIN_ENV = Path(os.environ.get("WALLEX_ADMIN_ENV_FILE") or os.environ.get("WALLEX_ADMIN_ENV") or "/root/.wallex-liara-admin.env")
BOXAPI_ENV = Path(os.environ.get("BOXAPI_ENV_FILE") or "/root/.boxapi.env")
GATE_ENV = Path(os.environ.get("LLM_GATE_ENV_FILE") or "/root/.liara.env")
DISK = {"name": "wallex-data", "mountTo": "/data"}


def read_env(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def lia_module():
    try:
        import lia
    except ImportError:
        import liara_cli as lia
    return lia


def assemble(out: Path):
    """Code-only image context: arena package (+ the stdlib news module its NewsFeed imports), checks, liara/ app."""
    if out.exists():
        shutil.rmtree(out)
    # results_report.py is the offline results-report generator (needs matplotlib); the app does not use it
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", "results_report.py")
    (out / "experiments").mkdir(parents=True)
    shutil.copy2(TREE / "experiments/__init__.py", out / "experiments/__init__.py")
    shutil.copytree(TREE / "experiments/wallex_arena", out / "experiments/wallex_arena", ignore=ign)
    # the engine's NewsFeed imports experiments.wallex_agent.news (stdlib only); the rest of wallex_agent is the separate
    # Paper agent (own Anthropic client) and is deliberately not shipped
    (out / "experiments/wallex_agent").mkdir()
    for f in ("__init__.py", "news.py"):
        shutil.copy2(TREE / "experiments/wallex_agent" / f, out / "experiments/wallex_agent" / f)
    (out / "checks").mkdir()
    for t in sorted((TREE / "tests").glob("test_wallex_arena*.py")):
        shutil.copy2(t, out / "checks" / t.name)
    shutil.copy2(HERE / "test_liara_image.py", out / "checks/test_liara_image.py")
    shutil.copy2(TREE / "tests/conftest.py", out / "checks/conftest.py")
    (out / "liara").mkdir()
    for f in ("start.py", "web.py"):
        shutil.copy2(HERE / f, out / "liara" / f)
    (out / "liara/__init__.py").write_text("")
    for f in ("Dockerfile", "requirements.txt", ".dockerignore"):
        shutil.copy2(HERE / f, out / f)
    arena = json.loads((TREE / "experiments/wallex_arena/arena-config.json").read_text())
    src = next(a for a in arena["accounts"] if a["name"] == "ai_maker_sonnet")
    acct = {"name": "live_ai_maker", "class": src["class"], "params": dict(src["params"], model="claude-haiku-5-5", maker=dict(src["params"]["maker"], max_exposure=0.35)),
            "live": "real", "initial_tmn": 5_000_000, "max_order_tmn": 2_000_000,
            "comment": "REAL Wallex credit account. Params of the paper account ai_maker_sonnet; model chosen by LLM_PROVIDER at start-up."}
    # Alternative live strategy (WALLEX_LIVE_STRATEGY=macro): the macro allocator (AITrader, style "macro", a basket of
    # TMN-priced assets that gain when the toman weakens), with the account's own risk block capped by start.py.
    msrc = next((a for a in arena["accounts"] if a["params"].get("style") == "macro"), None)
    alternatives = {}
    if msrc:
        alternatives["macro"] = {"name": "live_macro", "class": msrc["class"], "params": dict(msrc["params"]), "risk": dict(msrc.get("risk", {})),
                                 "live": "real", "initial_tmn": 5_000_000, "max_order_tmn": 2_000_000,
                                 "comment": f"REAL Wallex credit account. Params of the paper account {msrc['name']}; model chosen by LLM_PROVIDER."}
    cfg = {"comment": "Live config for meowmeow-wallex on Liara: one real account. Orders need WALLEX_LIVE_ENABLED=1, an API key and no kill file.",
           "alternatives": alternatives,
           **{k: arena[k] for k in ("tick_seconds", "focus_markets", "websocket", "rest_every_ticks")},
           "challenge_end_utc": "2026-10-23T20:00:00Z", "flatten_minutes": 60, "no_new_entries_minutes": 120,
           "accounts": [acct], "wallex_env_file": "/run/wallex/wallex.env", "kill_file": "/var/lib/wallex-arena/LIVE_DISABLED"}
    (out / "liara/live-config.template.json").write_text(json.dumps(cfg, indent=1))
    if (TREE / "snapshot/MANIFEST.json").is_file():  # optional results snapshot placed at the repository root
        shutil.copytree(TREE / "snapshot", out / "snapshot")


def build():
    assemble(OUT)
    subprocess.run([sys.executable, str(HERE / "make_snapshot.py"), str(OUT / "snapshot")], check=True)
    commit = subprocess.run(["git", "-C", str(TREE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(TREE), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", str(TREE), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip()
    (OUT / "BUILD_INFO.json").write_text(json.dumps({"branch": branch, "commit": commit, "dirty": bool(dirty)}, indent=1))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print("built", OUT, f"{size / 1e6:.1f} MB", "commit", commit[:10], "DIRTY" if dirty else "clean")


def tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
        for f in sorted(OUT.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                tf.add(f, arcname=str(f.relative_to(OUT)))
    return buf.getvalue()


def envs() -> dict:
    if not ADMIN_ENV.exists():
        fd = os.open(ADMIN_ENV, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"WALLEX_LIARA_URL=https://{APP}.liara.run\nWALLEX_ADMIN_USER=admin\nWALLEX_ADMIN_PASSWORD={secrets.token_urlsafe(18)}\n")
    box = read_env(BOXAPI_ENV)
    gate = read_env(GATE_ENV)   # only the Arvan LLM gate credentials; never an Anthropic key or relay token
    admin = read_env(ADMIN_ENV)
    if not admin.get("SNAPSHOT_PUSH_TOKEN"):     # bearer token for POST /api/snapshot/push (Hetzner push timer)
        with open(ADMIN_ENV, "a") as f:
            f.write(f"SNAPSHOT_PUSH_TOKEN={secrets.token_urlsafe(32)}\n")
        admin = read_env(ADMIN_ENV)
    mine = {"WALLEX_ADMIN_PASSWORD": admin["WALLEX_ADMIN_PASSWORD"], "SNAPSHOT_PUSH_TOKEN": admin["SNAPSHOT_PUSH_TOKEN"],
            "LLM_PROVIDER": "anthropic", "ANTHROPIC_BASE_URL": gate["LLM_GATE_URL"], "ANTHROPIC_AUTH_TOKEN": gate["LLM_GATE_TOKEN"],
            "WALLEX_ANTHROPIC_MODEL": "claude-haiku-5-5",
            "BOXAPI_KEYS": box["BOXAPI_KEYS"], "BOXAPI_BASE_URL": box.get("BOXAPI_BASE_URL") or "https://ai.boxapi.ir/ai/v1",
            "WALLEX_BOXAPI_MODEL": "openai/gpt-6-luna", "TZ": "UTC", "OBS_ENV": "liara",
            "WALLEX_MCP_TOOLS": "1", "WALLEX_MAX_EXPOSURE": "0.35"}
    # WALLEX_CHALLENGE_END_UTC is deliberately not forced: until an operator sets the official end time the dashboard
    # shows the built-in default (2026-10-23T20:00:00Z) marked as a placeholder.
    import lia
    code, r = lia.call("GET", f"/v1/projects/{APP}")
    current = {e["key"]: e["value"] for e in (r.get("project", {}).get("envs") or [])} if code == 200 else {}
    current.pop("WALLEX_MODEL", None)                  # superseded by LLM_PROVIDER + per-provider models
    # merge: keep whatever is set (e.g. an operator's WALLEX_ENGINE / WALLEX_LIVE_ENABLED / Wallex keys), safe defaults otherwise
    return {"WALLEX_ENGINE": "off", "WALLEX_LIVE_ENABLED": "0", **current, **mine}


def deploy(message):
    lia = lia_module()
    code, r = lia.call("GET", f"/v1/projects/{APP}/disks")
    names = [d["name"] for d in (r.get("disks") or [])] if code == 200 else []
    if DISK["name"] not in names:
        code, r = lia.call("POST", f"/v1/projects/{APP}/disks", {"name": DISK["name"], "size": 1})
        print("create disk", code, r if code >= 300 else "")
    data = tarball()
    b = "----x" + uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="source.tar.gz"\r\nContent-Type: application/gzip\r\n\r\n').encode() + data + f"\r\n--{b}--\r\n".encode()
    code, r = lia.call("POST", f"/v2/projects/{APP}/sources", raw=body, ctype="multipart/form-data; boundary=" + b)
    print("upload", code, f"{len(data) / 1e6:.1f} MB", r if code >= 300 else "")
    rel = {"sourceID": r["sourceID"], "port": 8080, "type": "docker", "envs": envs(), "disks": [DISK], "message": message,
           "build": {"location": "germany"}}
    code, r = lia.call("POST", f"/v2/projects/{APP}/releases", rel)
    print("release", code, r)
    print(wait_release(r["releaseID"]))


def wait_release(rid, timeout=600):
    """Poll one release; a build stuck past `timeout` (Liara's basic build limit is 5 min) is cancelled."""
    import time
    import lia
    t0 = time.time()
    while time.time() - t0 < timeout:
        code, r = lia.call("GET", f"/v1/releases/{rid}")
        state = (r.get("release") or {}).get("state") if code == 200 else code
        if state in ("READY", "FAILED", "CANCELED", "TIMEDOUT", "UNHEALTHY"):
            return f"{state} after {int(time.time() - t0)} s"
        time.sleep(10)
    code, r = lia.call("POST", f"/v2/releases/{rid}/cancel")
    return f"STUCK after {timeout} s: cancelled ({code}); the previous release keeps serving"


def setenv(pairs):
    lia = lia_module()
    code, r = lia.call("GET", f"/v1/projects/{APP}")
    cur = {e["key"]: e["value"] for e in r["project"].get("envs", [])}
    for p in pairs:
        k, v = p.split("=", 1)
        cur[k] = v
    code, r = lia.call("POST", "/v1/projects/update-envs", {"project": APP, "variables": [{"key": k, "value": v} for k, v in cur.items()]})
    print("update-envs", code, "keys:", sorted(cur), r if code >= 300 else "")


def releases():
    lia = lia_module()
    code, r = lia.call("GET", f"/v1/projects/{APP}/releases?page=1&count=15")
    if code >= 300:
        raise SystemExit(f"releases {code} {r}")
    cur = r.get("currentRelease")
    for x in r.get("releases", []):
        print(x["_id"], x.get("tag", ""), x.get("state"), x.get("type", ""), x.get("createdAt", ""), "<- current" if x["_id"] == cur else "")


def rollback(release_id):
    lia = lia_module()
    code, r = lia.call("POST", f"/v1/releases/{release_id}/rollback")
    print("rollback", code, r if code >= 300 else "accepted")
    if code < 300:
        print(lia.wait(APP))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "assemble":
        assemble(Path(sys.argv[2]))
    elif sys.argv[1] == "releases":
        releases()
    elif sys.argv[1] == "rollback":
        rollback(sys.argv[2])
    elif sys.argv[1] == "deploy":
        deploy(sys.argv[2] if len(sys.argv) > 2 else "meowmeow-wallex: dashboard + snapshot, engine off, disarmed")
    elif sys.argv[1] == "setenv":
        setenv(sys.argv[2:])
