"""Copy a consistent, read-only snapshot of the Hetzner paper/shadow results for the Liara dashboard.

usage (root on Hetzner): make_snapshot.py OUTDIR
Arena state (/var/lib/wallex-arena), the arena dashboard export, the shadow-test report, the paper agent's
portfolio and decisions (/var/lib/wallex-paper-agent, without the 126 MB of cycle/snapshot artifacts), the 7-day
backtest table and the Wallex docs. SQLite files are copied with the backup API so live writers cannot tear them.
No secrets are read: nothing under private/ is touched.
"""
import hashlib
import os
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# Source locations (defaults are the Hetzner layout; override with env when the results live elsewhere).
ARENA = Path(os.environ.get("WALLEX_ARENA_STATE") or "/var/lib/wallex-arena")
PAPER = Path(os.environ.get("WALLEX_PAPER_STATE") or "/var/lib/wallex-paper-agent")
BASE = Path(os.environ.get("WALLEX_BASE_DIR") or "/opt/wallex")
WWW = Path(os.environ.get("WALLEX_DASHBOARD_WWW") or "/var/www/wallex-arena")
VENV_PY = Path(os.environ.get("WALLEX_PYTHON") or BASE / ".venv/bin/python")
SOURCE = os.environ.get("WALLEX_SNAPSHOT_SOURCE") or "Hetzner dev server"


def backup_sqlite(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as s, sqlite3.connect(dst) as d:
        s.backup(d)


def copy(src: Path, dst: Path):
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main(out: Path):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    taken = time.time()
    # arena: engine state and every account's ledger, hash-chained decision log and live/dry-run request log
    a = out / "arena"
    for f in ("status.json", "engine.jsonl"):
        copy(ARENA / f, a / f)
    backup_sqlite(ARENA / "news.sqlite", a / "news.sqlite")
    for acct in sorted(p for p in (ARENA / "accounts").iterdir() if p.is_dir()):
        for f in acct.iterdir():
            if f.suffix == ".sqlite":
                backup_sqlite(f, a / "accounts" / acct.name / f.name)
            elif f.suffix in (".json", ".jsonl"):
                copy(f, a / "accounts" / acct.name / f.name)
    # dashboard page and a fresh export of its data
    www = out / "www"
    src_index = WWW / "index.html"
    if not src_index.is_file():  # repository checkout: the dashboard page ships in services/dashboard/
        src_index = Path(__file__).resolve().parents[2] / "services/dashboard/index.html"
    copy(src_index, www / "index.html")
    (www / "data").mkdir(parents=True)
    subprocess.run(["/usr/bin/python3", str(Path(__file__).resolve().parents[2] / "services/dashboard_export.py"), str(www / "data"), str(ARENA)], check=True,
                   stdout=subprocess.DEVNULL)
    # shadow-test report built from the arena's own logs (verifies every decision hash chain)
    subprocess.run([str(VENV_PY), "-m", "experiments.wallex_arena", "report", str(ARENA), str(out / "report.md")],
                   cwd=Path(__file__).resolve().parents[2], check=True)  # this branch (report handles AI-maker plans)
    # Paper agent: status, portfolio, LLM spend, decisions
    p = out / "paper-agent"
    for f in ("status.json", "daemon.json", "protection.json"):
        copy(PAPER / f, p / f)
    for f in ("paper-portfolio.sqlite", "llm-budget.sqlite"):
        if (PAPER / f).exists():
            backup_sqlite(PAPER / f, p / f)
    if (PAPER / "artifacts/decisions").is_dir():
        shutil.copytree(PAPER / "artifacts/decisions", p / "decisions")
    # research and docs
    copy(BASE / "research/v3/data/backtest_tmn.txt", out / "research/backtest_tmn.txt")
    for d in sorted((BASE / "Trade/docs").glob("wallex*.md")):
        copy(d, out / "docs" / d.name)
    for f in out.rglob("*"):          # the dashboard runs as an unprivileged user: everything readable, nothing secret here
        os.chmod(f, 0o755 if f.is_dir() else 0o644)
    commit = subprocess.run(["git", "-C", str(BASE / "Trade"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    files = {str(f.relative_to(out)): {"bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
             for f in sorted(out.rglob("*")) if f.is_file()}
    status = json.loads((ARENA / "status.json").read_text())
    manifest = {"snapshot_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(taken)), "snapshot_ts": taken,
                "source": f"{SOURCE}, paper/shadow trading (simulated 5,000,000 TMN per account on live Wallex order books)",
                "arena_started_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(status["started_at"])),
                "arena_status_updated_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(status["updated_at"])),
                "trade_repo_commit": commit, "accounts": len(status["leaderboard"]),
                "total_bytes": sum(v["bytes"] for v in files.values()), "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(manifest["snapshot_utc"], manifest["accounts"], "accounts", len(files), "files", manifest["total_bytes"], "bytes")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
