"""Structured JSON-lines logging to stdout (team observability standard).

Every line: ts (ISO-8601 UTC, ms), level, app, env, release, component, event, request_id, trace_id, span_id, session,
msg, plus typed optional fields. Context (request_id / trace) lives in contextvars and is propagated through
X-Request-ID / W3C traceparent. Secrets are never logged: keys that look like credentials are redacted and long
strings are truncated (with a short hash). LOG_LEVEL (default info) filters; successful external calls are debug.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import secrets
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40, "critical": 50}
SECRET_WORDS = ("key", "token", "secret", "password", "authorization", "auth", "cookie")
MAX_STR = 300

_request_id = contextvars.ContextVar("request_id", default=None)
_trace = contextvars.ContextVar("trace", default=None)     # (trace_id, span_id)
_session = contextvars.ContextVar("session", default=None)
_lock = threading.Lock()
_static: dict = {}


def _release() -> str:
    if os.environ.get("RELEASE"):
        return os.environ["RELEASE"]
    for p in (Path(__file__).resolve().parents[2] / "BUILD_INFO.json",):
        try:
            return json.loads(p.read_text())["commit"][:10]
        except (OSError, ValueError, KeyError):
            pass
    return "dev"


def configure(component: str | None = None, **static):
    _static.update({"app": os.environ.get("OBS_APP") or "meowmeow-wallex", "env": os.environ.get("OBS_ENV") or "dev",
                    "release": _release()})
    if component:
        _static["component"] = component
    _static.update(static)


def short_hash(s: str, n: int = 12) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()[:n]


def _clean(k, v, depth=0):
    if any(w in str(k).lower() for w in SECRET_WORDS) and k not in ("tokens_in", "tokens_out", "max_tokens"):
        return "[redacted]"
    if isinstance(v, str) and len(v) > MAX_STR:
        return v[:MAX_STR] + f"…[{len(v)} chars sha:{short_hash(v, 8)}]"
    if isinstance(v, dict) and depth < 3:
        return {kk: _clean(kk, vv, depth + 1) for kk, vv in list(v.items())[:40]}
    if isinstance(v, (list, tuple)) and depth < 3:
        return [_clean(k, x, depth + 1) for x in list(v)[:20]]
    return v


def new_id(nbytes=8) -> str:
    return secrets.token_hex(nbytes)


def set_context(request_id=None, traceparent: str | None = None, session=None):
    """Bind request/trace context (incoming X-Request-ID / traceparent are reused when valid)."""
    rid = request_id if request_id and len(request_id) <= 128 and request_id.isprintable() else new_id()
    trace_id, span_id = None, new_id(8)
    if traceparent:
        parts = traceparent.strip().split("-")
        if len(parts) == 4 and len(parts[1]) == 32 and all(c in "0123456789abcdef" for c in parts[1]):
            trace_id = parts[1]
    _request_id.set(rid)
    _trace.set((trace_id or new_id(16), span_id))
    _session.set(short_hash(session) if session else None)
    return rid


def context() -> dict:
    t = _trace.get() or (None, None)
    return {"request_id": _request_id.get(), "trace_id": t[0], "span_id": t[1], "session": _session.get()}


def traceparent() -> str | None:
    t = _trace.get()
    return f"00-{t[0]}-{t[1]}-01" if t else None


def enabled(level: str) -> bool:
    return LEVELS.get(level, 20) >= LEVELS.get((os.environ.get("LOG_LEVEL") or "info").lower(), 20)


def log(event: str, level: str = "info", msg: str = "", **fields):
    if not enabled(level):
        return
    if not _static:
        configure()
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"), "level": level,
           **_static, "event": event, **context(), "msg": msg[:MAX_STR]}
    for k, v in fields.items():
        if v is not None:
            rec[k] = _clean(k, v)
    line = json.dumps(rec, ensure_ascii=False, default=str)
    with _lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def log_error(event: str, exc: BaseException, level: str = "error", msg: str = "", **fields):
    log(event, level, msg or str(exc)[:200], error_type=type(exc).__name__, error=str(exc)[:200],
        stack="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:], **fields)


def install_crash_handlers():
    """Unhandled exceptions in the main thread or any thread become critical `app.crash` lines."""
    def hook(t, v, tb):
        log("app.crash", "critical", str(v)[:200], error_type=t.__name__, stack="".join(traceback.format_exception(t, v, tb))[-4000:])

    sys.excepthook = hook
    threading.excepthook = lambda a: hook(a.exc_type, a.exc_value, a.exc_traceback) if a.exc_type is not SystemExit else None


class timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        return False

    @property
    def ms(self) -> int:
        return int((time.perf_counter() - self.t) * 1000)
