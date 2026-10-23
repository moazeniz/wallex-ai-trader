"""Minimal, dependency-free Liara API client shared by the platform tools.

Credentials are read from the environment and never printed:
  LIARA_API_TOKEN  API token (team-wide; Liara console -> API)
  LIARA_TEAM_ID    team id, appended to every call as ?teamID=
  LIARA_API_BASE   default https://api.iran.liara.ir (the Iran edge the console uses)

Every response body that is echoed anywhere passes through `redact()` first.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

DEFAULT_BASE = "https://api.iran.liara.ir"
RETRYABLE = {408, 429, 500, 502, 503, 504}
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
_BEARER = re.compile(r"(?i)(bearer\s+|token[\"'=:\s]+)[A-Za-z0-9._~+/-]{12,}")


def redact(text) -> str:
    """Mask the configured token, any JWT and anything that looks like a bearer/token value."""
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    for name in ("LIARA_API_TOKEN", "GITHUB_TOKEN", "STATUS_INGEST_TOKEN"):
        v = os.getenv(name)
        if v and len(v) >= 8:
            s = s.replace(v, "***")
    s = _JWT.sub("***jwt***", s)
    return _BEARER.sub(lambda m: m.group(1) + "***", s)


class LiaraError(RuntimeError):
    def __init__(self, what: str, status: int, payload):
        self.status, self.payload = status, payload
        code = (payload or {}).get("data", {}) if isinstance(payload, dict) else {}
        self.code = (code or {}).get("code") if isinstance(code, dict) else None
        super().__init__(f"{what}: HTTP {status} {redact(payload)[:400]}")


class Liara:
    """Thin wrapper over the Liara REST API (team scoped)."""

    def __init__(self, token: str | None = None, team: str | None = None, base: str | None = None,
                 retries: int = 3, backoff: float = 2.0, timeout: float = 60.0):
        self.token = token or os.getenv("LIARA_API_TOKEN", "")
        self.team = team if team is not None else os.getenv("LIARA_TEAM_ID", "")
        self.base = (base or os.getenv("LIARA_API_BASE") or DEFAULT_BASE).rstrip("/")
        self.retries, self.backoff, self.timeout = retries, backoff, timeout
        if not self.token:
            raise SystemExit("LIARA_API_TOKEN is not set (export it or load it from your secret store)")

    # -- transport -----------------------------------------------------------------
    def url(self, path: str, base: str | None = None) -> str:
        sep = "&" if "?" in path else "?"
        team = f"{sep}teamID={urllib.parse.quote(self.team)}" if self.team else ""
        return (base or self.base) + path + team

    def request(self, method: str, path: str, body=None, raw: bytes | None = None, ctype: str | None = None,
                ok=(200, 201, 202, 204), what: str | None = None, base: str | None = None, timeout: float | None = None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        headers = {"Authorization": "Bearer " + self.token, "Accept": "application/json",
                   "User-Agent": "meowmeow-platform/1.0"}
        if data is not None:
            headers["Content-Type"] = ctype or "application/json"
        last = None
        # Only idempotent reads are retried automatically; writes are retried on connection errors only
        # when the server never answered (no status), so a deploy can never be created twice.
        attempts = self.retries if method == "GET" else 1
        for i in range(max(1, attempts)):
            req = urllib.request.Request(self.url(path, base), data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                    status, text = r.status, r.read()
            except urllib.error.HTTPError as e:
                status, text = e.code, e.read()
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = LiaraError(what or f"{method} {path}", 0, {"error": str(e)})
                time.sleep(self.backoff * (i + 1))
                continue
            try:
                payload = json.loads(text or b"null")
            except ValueError:
                payload = text.decode(errors="replace")[:500]
            if status in ok:
                return payload
            last = LiaraError(what or f"{method} {path}", status, payload)
            if method == "GET" and status in RETRYABLE:
                time.sleep(self.backoff * (i + 1))
                continue
            raise last
        raise last

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, body=None, **kw):
        return self.request("POST", path, body=body, **kw)

    # -- apps ----------------------------------------------------------------------
    def projects(self):
        return self.get("/v1/projects").get("projects", [])

    def project(self, app):
        return self.get(f"/v1/projects/{app}", what=f"read app {app}")["project"]

    def releases(self, app, count=20):
        return self.get(f"/v1/projects/{app}/releases?page=1&count={count}", what=f"list releases of {app}")

    def release(self, rid):
        return self.get(f"/v1/releases/{rid}", what=f"read release {rid}")["release"]

    def build_logs(self, rid, since=""):
        q = f"?since={urllib.parse.quote(since)}" if since else ""
        return self.get(f"/v2/releases/{rid}/build-logs{q}", what="build logs")

    def disks(self, project_id):
        return self.get(f"/v1/projects/{project_id}/disks", what="list disks")

    def me(self):
        return self.get("/v1/me", what="read account")

    def upload_source(self, app, archive: bytes) -> str:
        b = "----meowmeow" + uuid.uuid4().hex
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"source.tar.gz\"\r\n"
                f"Content-Type: application/gzip\r\n\r\n").encode() + archive + f"\r\n--{b}--\r\n".encode()
        r = self.request("POST", f"/v2/projects/{app}/sources", raw=body, ctype="multipart/form-data; boundary=" + b,
                         what="upload source", timeout=600)
        return r["sourceID"]

    def create_release(self, app, body) -> str:
        return self.post(f"/v2/projects/{app}/releases", body, what="create release")["releaseID"]

    def rollback(self, rid):
        return self.post(f"/v1/releases/{rid}/rollback", what=f"roll back to {rid}")


def load_manifest(path: str) -> dict:
    """YAML when PyYAML is installed, JSON otherwise (the manifests are JSON-compatible YAML)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def bundle_caps(me: dict, plan: str, bundle: str) -> dict:
    """Feature flags of a bundle for a hardware plan, from /v1/me (zeroDowntime, backups, deploy quota...)."""
    try:
        return me["plans"]["projectBundlePlans"][plan][bundle]
    except (KeyError, TypeError):
        return {}


def hourly_price(me: dict, plan: str, bundle: str) -> float:
    """Toman per hour = hardware plan price + bundle price (both hourly on /v1/me)."""
    try:
        return float(me["plans"]["projects"][plan]["price"]) + float(bundle_caps(me, plan, bundle).get("price") or 0)
    except (KeyError, TypeError, ValueError):
        return 0.0
