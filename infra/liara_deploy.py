#!/usr/bin/env python3
"""Release pipeline for Liara: package -> upload -> release -> watch -> smoke test -> automatic rollback.

  liara_deploy.py deploy   --manifest infra/liara.app.yaml [--app NAME] [--message TAG] [--dry-run]
                           [--no-rollback] [--allow-dirty] [--summary FILE]
  liara_deploy.py smoke    --manifest ... [--app NAME]              run the post-deploy checks only
  liara_deploy.py rollback --manifest ... [--app NAME] [--to RELEASE_ID]
  liara_deploy.py status   --manifest ... [--app NAME]              current release + recent history

Design notes
  * The uploaded source is exactly the tracked files of HEAD (git ls-files) minus .liaraignore, packed
    deterministically, so a release is reproducible from its commit (recorded in gitInfo).
  * Release bodies never carry `envs`: on Liara that replaces every variable. Env values live in Liara;
    the manifest lists the required NAMES and the pipeline refuses to deploy when one is missing.
  * Failure handling:
      build FAILED / TIMEDOUT / CANCELED -> traffic never moved; verify the old release still serves.
      UNHEALTHY (health check failed)    -> Liara keeps the old release when zero-downtime is on;
                                            otherwise we roll back explicitly.
      READY but smoke test fails         -> POST /v1/releases/{previous}/rollback, wait, smoke again.
  * Exit codes: 0 deployed, 1 failed before traffic moved, 2 rolled back (healthy), 3 rollback failed
    (page a human), 4 preflight refused (bad config, quota, missing env).
"""
from __future__ import annotations

import argparse
import fnmatch
import gzip
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from liara_api import Liara, LiaraError, bundle_caps, load_manifest, redact  # noqa: E402

TERMINAL = {"READY", "FAILED", "CANCELED", "TIMEDOUT", "UNHEALTHY"}
EXIT_OK, EXIT_FAILED, EXIT_ROLLED_BACK, EXIT_ROLLBACK_FAILED, EXIT_REFUSED = 0, 1, 2, 3, 4
POLL = float(os.getenv("LIARA_POLL_SECONDS", "5"))


def log(msg: str) -> None:
    print(redact(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"), flush=True)


def gha(kind: str, msg: str) -> None:
    """GitHub Actions annotation (::error:: / ::warning:: / ::notice::); plain text elsewhere."""
    print(f"::{kind}::{redact(msg)}" if os.getenv("GITHUB_ACTIONS") else f"{kind.upper()}: {redact(msg)}", flush=True)


# -- source packaging -------------------------------------------------------------------
def read_ignore(root: Path) -> list[str]:
    for name in (".liaraignore", ".dockerignore", ".gitignore"):  # same precedence as the Liara CLI
        p = root / name
        if p.exists():
            out = []
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    if line.startswith("!"):
                        gha("warning", f"{name}: negated pattern '{line}' is not supported and was ignored")
                        continue
                    out.append(line)
            return out
    return []


def is_ignored(path: str, patterns: list[str]) -> bool:
    """gitignore-style matching good enough for upload filters: anchored paths, bare names, **/ prefixes."""
    parts = path.split("/")
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    for raw in patterns:
        pat = raw.rstrip("/")
        anchored = pat.startswith("/") or ("/" in pat.lstrip("/") and not pat.startswith("**/"))
        pat = pat.lstrip("/")
        if pat.startswith("**/"):
            sub = pat[3:]
            # match the remainder against every suffix of every prefix
            for pre in prefixes:
                segs = pre.split("/")
                if any(fnmatch.fnmatchcase("/".join(segs[i:]), sub) for i in range(len(segs))):
                    return True
        elif anchored:
            if any(fnmatch.fnmatchcase(pre, pat) for pre in prefixes):
                return True
        elif any(fnmatch.fnmatchcase(seg, pat) for seg in parts):
            return True
    return False


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def package(root: Path, allow_dirty: bool = False, subdir: str = "") -> tuple[bytes, list[str]]:
    """Deterministic .tar.gz of the tracked files of HEAD minus the ignore list.

    `subdir` packages one directory of the repo as the build context (e.g. a second app living in demo-shop/)."""
    if not allow_dirty and git(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("working tree has uncommitted changes; commit them or pass --allow-dirty")
    listed = git(root, "ls-files", "-z")
    files = sorted(f for f in listed.split("\0") if f) if listed else sorted(
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts)
    prefix = subdir.strip("/") + "/" if subdir.strip("/") else ""
    if prefix:
        files = [f[len(prefix):] for f in files if f.startswith(prefix)]
        root = root / prefix
        if not files:
            raise SystemExit(f"source directory {subdir!r} has no tracked files")
    patterns = read_ignore(root)
    keep = [f for f in files if not is_ignored(f, patterns) and (root / f).is_file()]
    mtime = int(git(root, "log", "-1", "--format=%ct") or 0)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for f in keep:
            info = tf.gettarinfo(str(root / f), arcname=f)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = mtime
            with open(root / f, "rb") as fh:
                tf.addfile(info, fh)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=mtime) as gz:
        gz.write(raw.getvalue())
    return out.getvalue(), keep


def git_info(root: Path, message: str) -> dict:
    if os.getenv("GITHUB_REPOSITORY"):
        remote = f"{os.getenv('GITHUB_SERVER_URL', 'https://github.com')}/{os.getenv('GITHUB_REPOSITORY')}"
    else:
        remote = git(root, "ls-remote", "--get-url")
    tags = git(root, "tag", "--points-at", "HEAD")
    return {
        "branch": os.getenv("GITHUB_REF_NAME") or git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": os.getenv("GITHUB_SHA") or git(root, "rev-parse", "HEAD"),
        "message": git(root, "log", "-1", "--format=%B")[:500] or message,
        "committedAt": git(root, "log", "-1", "--format=%ct"),
        "remote": redact(remote),  # never leak a token embedded in a remote URL
        "author": {"name": git(root, "log", "-1", "--format=%aN"), "email": git(root, "log", "-1", "--format=%aE")},
        "tags": [t for t in tags.splitlines() if t],
    }


# -- manifest ---------------------------------------------------------------------------
def pick_app(manifest: dict, name: str | None) -> dict:
    apps = manifest.get("apps") or []
    if name:
        for a in apps:
            if a["name"] == name:
                return a
        raise SystemExit(f"app {name!r} is not declared in the manifest")
    if len(apps) != 1:
        raise SystemExit("manifest declares several apps; pass --app")
    return apps[0]


def release_body(app: dict, source_id: str, message: str, gitinfo: dict) -> dict:
    body = {"sourceID": source_id, "type": app.get("platform", "docker"), "port": int(app["port"]),
            "message": message[:200], "gitInfo": gitinfo,
            "disks": [{"name": d["name"], "mountTo": d["mountTo"]} for d in app.get("disks", []) if d.get("mountTo")]}
    b = app.get("build") or {}
    if b:
        body["build"] = {k: v for k, v in {"location": b.get("location"), "dockerfile": b.get("dockerfile"),
                                            "cache": b.get("cache", True)}.items() if v is not None}
    hc = app.get("healthCheck")
    if hc:
        body["healthCheck"] = {"command": hc["command"] if isinstance(hc["command"], list) else hc["command"].split(" "),
                               **{k: int(hc[k]) for k in ("interval", "timeout", "retries", "startPeriod") if k in hc}}
    return body


# -- smoke tests ------------------------------------------------------------------------
def http_get(url: str, timeout: float) -> tuple[int, bytes, float]:
    t0 = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent": "meowmeow-smoke/1.0", "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(200_000), time.monotonic() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read(200_000), time.monotonic() - t0
    except Exception as e:  # noqa: BLE001 - every network failure is a failed check
        return 0, str(e).encode(), time.monotonic() - t0


def json_subset(want, got) -> bool:
    if isinstance(want, dict):
        return isinstance(got, dict) and all(k in got and json_subset(v, got[k]) for k, v in want.items())
    return want == got


def run_check(base: str, chk: dict) -> tuple[bool, str, float]:
    status, body, dt = http_get(base.rstrip("/") + chk.get("path", "/"), float(chk.get("timeout", 15)))
    want = int(chk.get("expectStatus", 200))
    if status != want:
        return False, f"HTTP {status} (want {want})", dt
    if chk.get("contains") and chk["contains"].encode() not in body:
        return False, f"body lacks {chk['contains']!r}", dt
    if "expectJson" in chk:
        try:
            got = json.loads(body)
        except ValueError:
            return False, "body is not JSON", dt
        if not json_subset(chk["expectJson"], got):
            return False, f"JSON mismatch, want subset {chk['expectJson']}", dt
    if chk.get("maxLatencyMs") and dt * 1000 > float(chk["maxLatencyMs"]):
        return False, f"slow: {dt*1000:.0f} ms > {chk['maxLatencyMs']} ms", dt
    return True, f"HTTP {status}", dt


def smoke(app: dict, attempts: int | None = None, delay: float | None = None) -> tuple[bool, list[dict]]:
    """Every check must pass; each gets several attempts because a fresh container may still be warming up."""
    base = app.get("url") or f"https://{app['name']}.liara.run"
    s = app.get("smoke") or {"checks": [{"path": "/", "expectStatus": 200}]}
    checks = s["checks"] if isinstance(s, dict) else s
    attempts = attempts or int((s.get("attempts") if isinstance(s, dict) else None) or 6)
    delay = POLL * 2 if delay is None else delay
    results = []
    for chk in checks:
        ok, why, dt = False, "", 0.0
        for i in range(attempts):
            ok, why, dt = run_check(base, chk)
            if ok:
                break
            if i + 1 < attempts:
                time.sleep(delay)
        results.append({"path": chk.get("path", "/"), "ok": ok, "detail": why, "ms": round(dt * 1000), "tries": i + 1})
        log(f"smoke {chk.get('path', '/')}: {'PASS' if ok else 'FAIL'} {why} {dt*1000:.0f} ms (try {i + 1})")
    return all(r["ok"] for r in results), results


# -- release watching ---------------------------------------------------------------------
def wait_release(li: Liara, rid: str, timeout: float) -> dict:
    t0, last = time.monotonic(), None
    while time.monotonic() - t0 < timeout:
        rel = li.release(rid)
        state = rel.get("state") or rel.get("status")
        if state != last:
            log(f"release {rid}: {state}" + (f" (queue {rel['queue']})" if rel.get("queue") else ""))
            last = state
        if state in TERMINAL:
            return rel
        time.sleep(POLL)
    return {"state": "WATCH_TIMEOUT", "_id": rid}


def tail_build_logs(li: Liara, rid: str, n: int = 40) -> list[str]:
    try:
        lines = [x.get("line", "") for x in li.build_logs(rid).get("buildOutput", [])]
    except LiaraError:
        return []
    return [redact(line.rstrip()) for line in lines[-n:]]


def previous_ready(li: Liara, app_name: str, exclude: str) -> dict | None:
    data = li.releases(app_name, 20)
    for r in data.get("releases", []):
        if r["_id"] != exclude and r.get("state") == "READY" and r.get("type") in ("DEPLOY", "ENV_UPDATE"):
            return r
    return None


def do_rollback(li: Liara, app: dict, target: str, bad: str | None, timeout: float) -> bool:
    log(f"rolling back {app['name']} to release {target}")
    try:
        li.rollback(target)
    except LiaraError as e:
        gha("error", f"rollback API refused ({e.status} {e.code}): {e}")
        return False
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        data = li.releases(app["name"], 5)
        cur = data.get("currentRelease")
        newest = (data.get("releases") or [{}])[0]
        if cur and cur != bad and newest.get("state") in ("READY", None):
            log(f"current release is now {cur}")
            ok, _ = smoke(app)
            return ok
        if newest.get("state") in ("FAILED", "UNHEALTHY", "CANCELED", "TIMEDOUT") and newest.get("_id") not in (bad, target):
            gha("error", f"rollback release {newest.get('_id')} ended {newest.get('state')}")
            return False
        time.sleep(POLL)
    return False


# -- summary --------------------------------------------------------------------------------
class Summary:
    def __init__(self, path: str | None, app: dict):
        self.path = path or os.getenv("GITHUB_STEP_SUMMARY")
        self.app, self.rows, self.facts = app, [], {}

    def step(self, name: str, ok: bool | None, detail: str = "") -> None:
        icon = {True: "pass", False: "FAIL", None: "skip"}[ok]
        self.rows.append(f"| {name} | {icon} | {redact(detail)} |")

    def write(self, outcome: str) -> None:
        url = self.app.get("url") or f"https://{self.app['name']}.liara.run"
        md = [f"## Liara deploy: `{self.app['name']}` - {outcome}", "",
              f"App: {url}  ", f"Console: https://console.liara.ir/apps/{self.app['name']}  "]
        if os.getenv("GITHUB_REPOSITORY") and os.getenv("GITHUB_SHA"):
            md.append(f"Commit: {os.getenv('GITHUB_SERVER_URL', 'https://github.com')}/{os.getenv('GITHUB_REPOSITORY')}/commit/{os.getenv('GITHUB_SHA')}  ")
        md += [f"{k}: `{v}`  " for k, v in self.facts.items()]
        md += ["", "| Step | Result | Detail |", "|---|---|---|", *self.rows, ""]
        text = "\n".join(md)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        print(text)
        out = os.getenv("GITHUB_OUTPUT")
        if out:
            with open(out, "a") as fh:
                fh.write(f"outcome={outcome}\n" + "".join(f"{k}={v}\n" for k, v in self.facts.items()))


# -- commands ---------------------------------------------------------------------------------
def cmd_deploy(a) -> int:
    root = Path(a.root).resolve()
    app = pick_app(load_manifest(a.manifest), a.app)
    message = a.message or os.getenv("GITHUB_REF_NAME") or git(root, "describe", "--tags", "--always") or "deploy"
    S = Summary(a.summary, app)
    archive, files = package(root, a.allow_dirty, app.get("source", ""))
    S.step("package", True, f"{len(files)} files, {len(archive)/1e6:.2f} MB from ./{app.get('source', '')} "
                            "(tracked files minus .liaraignore)")
    gi = git_info(root, message)
    S.facts.update({"commit": gi["commit"][:12], "message": message})

    if a.dry_run:
        body = release_body(app, "<sourceID>", message, gi)
        S.step("release body", None, "dry run: " + json.dumps({k: v for k, v in body.items() if k != "gitInfo"}))
        try:
            li = Liara()
            p = li.project(app["name"])
            rels = li.releases(app["name"], 1)
            S.step("preflight (read-only)", True, f"scale={p.get('scale')} current={rels.get('currentRelease')}")
            missing = sorted(set(app.get("envs", {}).get("required", [])) - {e["key"] for e in p.get("envs", [])})
            S.step("required env names", not missing, "missing: " + ", ".join(missing) if missing else "all present")
        except SystemExit as e:
            S.step("preflight (read-only)", None, str(e))
        S.write("dry-run")
        return EXIT_OK

    li = Liara()
    try:
        p = li.project(app["name"])
    except LiaraError as e:
        S.step("preflight", False, str(e))
        S.write("refused")
        return EXIT_REFUSED
    me = li.me()
    caps = bundle_caps(me, p.get("planID", ""), p.get("bundlePlanID", ""))
    problems = []
    if p.get("scale", 1) == 0:
        problems.append("app is stopped (scale=0); start it before deploying")
    missing = sorted(set(app.get("envs", {}).get("required", [])) - {e["key"] for e in p.get("envs", [])})
    if missing:
        problems.append("missing env vars on Liara: " + ", ".join(missing))
    hist = li.releases(app["name"], 50)
    today = datetime.now(timezone.utc).date().isoformat()
    used = sum(1 for r in hist.get("releases", []) if str(r.get("createdAt", "")).startswith(today))
    quota = caps.get("maxDeploymentCountInDay")
    if isinstance(quota, int) and used >= quota:
        problems.append(f"daily deploy quota used ({used}/{quota} on bundle {p.get('bundlePlanID')})")
    if len(archive) > int(caps.get("maxSourceSize") or 128) * 1024 * 1024:
        problems.append(f"source is {len(archive)/1e6:.1f} MB, over the bundle limit")
    if problems:
        for msg in problems:
            gha("error", msg)
        S.step("preflight", False, "; ".join(problems))
        S.write("refused")
        return EXIT_REFUSED
    prev = hist.get("currentRelease")
    zd = bool(p.get("zeroDowntime")) and bool(caps.get("zeroDowntime"))
    S.facts.update({"previous_release": prev or "none", "zero_downtime": str(zd).lower()})
    S.step("preflight", True, f"plan {p.get('planID')}/{p.get('bundlePlanID')}, deploys today {used}/{quota}, previous {prev}")

    sid = li.upload_source(app["name"], archive)
    S.step("upload", True, f"sourceID {sid}")
    rid = li.create_release(app["name"], release_body(app, sid, message, gi))
    S.facts["release_id"] = rid
    log(f"release {rid} created; watching (build timeout {caps.get('buildTimeout', 5)} min)")
    rel = wait_release(li, rid, timeout=60 * (int(caps.get("buildTimeout") or 5) + 10))
    state = rel.get("state")
    S.facts["state"] = state
    S.step("build + start", state == "READY", f"{state} {rel.get('failReason') or ''}".strip())

    if state != "READY":
        for line in tail_build_logs(li, rid):
            print("  | " + line)
        cur = li.releases(app["name"], 1).get("currentRelease")
        if cur == rid and prev and not a.no_rollback:  # unhealthy release took traffic (no zero-downtime)
            ok = do_rollback(li, app, prev, rid, a.timeout)
            S.step("rollback", ok, f"to {prev}")
            S.write("rolled back" if ok else "ROLLBACK FAILED")
            return EXIT_ROLLED_BACK if ok else EXIT_ROLLBACK_FAILED
        ok, res = smoke(app, attempts=2)
        S.step("old release still serving", ok, f"current {cur}")
        S.write("failed (no traffic change)")
        return EXIT_FAILED

    ok, res = smoke(app)
    S.step("smoke tests", ok, "; ".join(f"{r['path']} {r['detail']} {r['ms']}ms" for r in res))
    if ok:
        S.write("deployed")
        return EXIT_OK
    gha("error", f"smoke tests failed on release {rid}")
    if a.no_rollback or not prev:
        S.step("rollback", None, "disabled" if a.no_rollback else "no previous release")
        S.write("FAILED (left in place)")
        return EXIT_ROLLBACK_FAILED
    ok = do_rollback(li, app, prev, rid, a.timeout)
    S.step("rollback", ok, f"to {prev}")
    S.write("rolled back" if ok else "ROLLBACK FAILED")
    return EXIT_ROLLED_BACK if ok else EXIT_ROLLBACK_FAILED


def cmd_smoke(a) -> int:
    ok, _ = smoke(pick_app(load_manifest(a.manifest), a.app))
    return EXIT_OK if ok else EXIT_FAILED


def cmd_rollback(a) -> int:
    app = pick_app(load_manifest(a.manifest), a.app)
    li = Liara()
    cur = li.releases(app["name"], 1).get("currentRelease")
    target = a.to or (previous_ready(li, app["name"], cur) or {}).get("_id")
    if not target:
        gha("error", "no earlier READY release to roll back to")
        return EXIT_ROLLBACK_FAILED
    return EXIT_OK if do_rollback(li, app, target, cur, a.timeout) else EXIT_ROLLBACK_FAILED


def cmd_status(a) -> int:
    app = pick_app(load_manifest(a.manifest), a.app)
    li = Liara()
    p, data = li.project(app["name"]), li.releases(app["name"], 8)
    print(f"{app['name']}: status={p.get('status')} scale={p.get('scale')} plan={p.get('planID')}/{p.get('bundlePlanID')} "
          f"zeroDowntime={p.get('zeroDowntime')}")
    for r in data.get("releases", []):
        mark = "*" if r["_id"] == data.get("currentRelease") else " "
        commit = ((r.get("gitInfo") or {}).get("commit") or "")[:10]
        print(f" {mark} {r.get('tag', ''):>4} {r['_id']} {r.get('type', ''):10} {r.get('state', ''):9} {r.get('createdAt', '')} {commit}")
    return EXIT_OK


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("deploy", "smoke", "rollback", "status"):
        s = sub.add_parser(name)
        s.add_argument("--manifest", default="infra/liara.app.yaml")
        s.add_argument("--app")
        s.add_argument("--timeout", type=float, default=600, help="seconds to wait for a rollback to settle")
        if name == "deploy":
            s.add_argument("--message")
            s.add_argument("--root", default=".")
            s.add_argument("--dry-run", action="store_true", help="package + render the release body + read-only preflight")
            s.add_argument("--no-rollback", action="store_true")
            s.add_argument("--allow-dirty", action="store_true")
            s.add_argument("--summary", help="append a Markdown summary here (default $GITHUB_STEP_SUMMARY)")
        if name == "rollback":
            s.add_argument("--to", help="release id (default: newest READY release before the current one)")
    a = ap.parse_args(argv)
    return {"deploy": cmd_deploy, "smoke": cmd_smoke, "rollback": cmd_rollback, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
