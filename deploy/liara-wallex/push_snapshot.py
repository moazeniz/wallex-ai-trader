"""Push a fresh, light paper/shadow results snapshot from the arena host to the Liara dashboard.

usage (root on the arena host, run by wallex-liara-snapshot-push.timer every 5 min):  push_snapshot.py [--dry-run]

Read-only on the arena: it copies the dashboard export that wallex-dashboard-export.timer already writes every 30 s,
the arena status, the last 300 records of every account's decision log (AI input dashboards removed, to keep it
small), the shadow-test report, and the paper agent's status and newest decisions. It never writes to the arena,
its state or its services. The snapshot is versioned by the arena's own status timestamp. Liara rejects snapshots
that are not newer, verifies every file's sha256 and swaps the new snapshot in atomically.
Config: WALLEX_LIARA_URL and SNAPSHOT_PUSH_TOKEN in /root/.wallex-liara-admin.env (0600; never printed).
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ARENA = Path(os.environ.get("WALLEX_ARENA_STATE") or "/var/lib/wallex-arena")
PAPER = Path(os.environ.get("WALLEX_PAPER_STATE") or "/var/lib/wallex-paper-agent")
BASE = Path(os.environ.get("WALLEX_BASE_DIR") or "/opt/wallex")
WWW = Path(os.environ.get("WALLEX_DASHBOARD_WWW") or "/var/www/wallex-arena")
TREE = Path(__file__).resolve().parents[2]                 # the liara-live worktree (report code)
VENV_PY = Path(os.environ.get("WALLEX_PYTHON") or BASE / ".venv/bin/python")
SECRETS = Path(os.environ.get("WALLEX_ADMIN_ENV") or "/root/.wallex-liara-admin.env")
TAIL = 300


def log(**kw):
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "app": "wallex-liara-push", **kw}), flush=True)


def read_json(path: Path, tries=3):
    for i in range(tries):                                   # the exporter rewrites these files every 30 s
        try:
            return path.read_bytes(), json.loads(path.read_bytes())
        except (OSError, ValueError):
            if i + 1 == tries:
                raise
            time.sleep(1)


def build(out: Path) -> dict:
    status_raw, status = read_json(ARENA / "status.json")
    (out / "arena").mkdir(parents=True)
    (out / "arena/status.json").write_bytes(status_raw)
    (out / "www/data").mkdir(parents=True)
    for f in sorted((WWW / "data").glob("*.json")):
        raw, _ = read_json(f)
        (out / "www/data" / f.name).write_bytes(raw)
    shutil.copy2(WWW / "index.html", out / "www/index.html")
    for acct in sorted(p for p in (ARENA / "accounts").iterdir() if p.is_dir()):
        src = acct / "decisions.jsonl"
        if not src.is_file():
            continue
        with open(src, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 40_000_000))
            lines = fh.read().splitlines()[-TAIL:]
        rows = []
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            row.get("record", {}).pop("input", None)          # the AI's full input dashboard (~15 KB each)
            rows.append(json.dumps(row, ensure_ascii=False))
        d = out / "arena/accounts" / acct.name
        d.mkdir(parents=True)
        (d / "decisions.jsonl").write_text("\n".join(rows) + "\n")
    r = subprocess.run(["nice", "-n", "10", str(VENV_PY), "-m", "experiments.wallex_arena", "report", str(ARENA), str(out / "report.md")],
                       cwd=TREE, capture_output=True, text=True, timeout=180, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    if r.returncode != 0:
        log(level="warn", event="push.report_failed", error=r.stderr[-300:])
        (out / "report.md").unlink(missing_ok=True)
    if (PAPER / "status.json").is_file():
        (out / "paper-agent").mkdir()
        shutil.copy2(PAPER / "status.json", out / "paper-agent/status.json")
        dec = PAPER / "artifacts/decisions"
        if dec.is_dir():
            for f in sorted(dec.rglob("*.json"), key=lambda p: p.stat().st_mtime)[-40:]:
                dest = out / "paper-agent/decisions" / f.relative_to(dec)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
    for p in out.rglob("*"):
        os.chmod(p, 0o755 if p.is_dir() else 0o644)
    commit = subprocess.run(["git", "-C", str(BASE / "Trade"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    files = {str(f.relative_to(out)): {"bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
             for f in sorted(out.rglob("*")) if f.is_file()}
    ts = float(status["updated_at"])                          # version = the arena's own status time
    man = {"kind": "push", "snapshot_ts": ts, "snapshot_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts)),
           "source": "Hetzner arena, paper/shadow trading (simulated 5,000,000 TMN per account on live Wallex order books)",
           "arena_started_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(status["started_at"])),
           "arena_status_updated_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts)), "trade_repo_commit": commit,
           "accounts": len(status.get("leaderboard", [])), "decision_tail_records": TAIL, "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return man


def pack(out: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                tf.add(f, arcname=str(f.relative_to(out)))
    return buf.getvalue()


def push(body: bytes, url: str, token: str) -> tuple[int, str]:
    sha = hashlib.sha256(body).hexdigest()
    last = (0, "")
    for attempt in range(1, 4):
        req = urllib.request.Request(url.rstrip("/") + "/api/snapshot/push", data=body, method="POST",
                                     headers={"Authorization": "Bearer " + token, "Content-Type": "application/gzip",
                                              "X-Snapshot-Sha256": sha, "User-Agent": "wallex-liara-push/1"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, r.read()[:500].decode(errors="replace")
        except urllib.error.HTTPError as e:
            last = (e.code, e.read()[:300].decode(errors="replace"))
            if e.code < 500 and e.code != 429:
                return last                                   # 4xx (e.g. 409 not newer) is final
        except OSError as e:
            last = (0, type(e).__name__)
        time.sleep(5 * attempt)
    return last


def main():
    env = {}
    for line in SECRETS.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("'\"")
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="wallex-push-") as tmp:
        out = Path(tmp) / "snap"
        man = build(out)
        body = pack(out)
    if "--dry-run" in sys.argv:
        log(event="push.dry_run", snapshot_utc=man["snapshot_utc"], files=len(man["files"]), bytes=len(body))
        return
    status, text = push(body, env.get("WALLEX_LIARA_URL") or "https://meowmeow-wallex.liara.run", env["SNAPSHOT_PUSH_TOKEN"])
    ok = status == 200 or status == 409
    log(level="info" if ok else "error", event="push.done", status=status, snapshot_utc=man["snapshot_utc"], files=len(man["files"]),
        bytes=len(body), duration_ms=int((time.time() - t0) * 1000), response=text[:200])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
