"""BoxAPI (OpenAI-compatible chat completions) with key rotation, for the AI accounts.

Keys come from BOXAPI_KEYS (comma-separated); base URL from BOXAPI_BASE_URL. Calls rotate round-robin over the keys
and fail over to the next key on 401/402/403/429/5xx, timeouts and connection errors; a failed key cools down before
it is tried again. Keys never appear in logs, exceptions or repr.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Callable

DEFAULT_BASE = "https://ai.boxapi.ir/ai/v1"
FAILOVER_STATUS = {401, 402, 403, 408, 409, 429}


class BoxAPIError(Exception):
    pass


def strict_schema(model) -> dict:
    """Pydantic model -> JSON schema accepted by strict structured outputs (all properties required, no extras)."""
    def fix(s):
        if isinstance(s, dict):
            if s.get("type") == "object" and "properties" in s:
                s["additionalProperties"] = False
                s["required"] = list(s["properties"])
            for v in s.values():
                fix(v)
        elif isinstance(s, list):
            for v in s:
                fix(v)
        return s
    return fix(model.model_json_schema())


def _urllib_transport(url: str, body: bytes, headers: dict, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class BoxAPIClient:
    def __init__(self, keys: list[str], base_url: str = DEFAULT_BASE, cooldown: float = 60.0,
                 transport: Callable[[str, bytes, dict, float], tuple[int, bytes]] | None = None, clock=time.time):
        self._keys = [k.strip() for k in keys if k and k.strip()]
        if not self._keys:
            raise BoxAPIError("no BoxAPI keys configured (BOXAPI_KEYS)")
        self.base_url = base_url.rstrip("/")
        self.cooldown = cooldown
        self.transport = transport or _urllib_transport
        self.clock = clock
        self._next = 0
        self._cool_until = [0.0] * len(self._keys)
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "failovers": 0, "errors": []}

    @classmethod
    def from_env(cls, env: dict | None = None, **kw) -> "BoxAPIClient":
        env = os.environ if env is None else env
        return cls((env.get("BOXAPI_KEYS") or "").split(","), env.get("BOXAPI_BASE_URL") or DEFAULT_BASE, **kw)

    def __repr__(self) -> str:
        return f"BoxAPIClient(base_url={self.base_url!r}, keys={len(self._keys)})"

    def _order(self) -> list[int]:
        with self._lock:
            n, start = len(self._keys), self._next
            self._next = (self._next + 1) % n
            now = self.clock()
            order = [(start + i) % n for i in range(n)]
            ready = [i for i in order if self._cool_until[i] <= now]
            return ready + [i for i in order if i not in ready]  # all cooling: still try, soonest-free last

    def chat(self, body: dict, timeout: float | None = None) -> dict:
        timeout = timeout or float(os.environ.get("BOXAPI_TIMEOUT_S") or 60)
        """POST /chat/completions; returns the parsed JSON response. Raises BoxAPIError when every key failed."""
        from . import obs
        from .resilience import breaker
        br = breaker("llm.boxapi")
        br.before()                                     # open circuit: fail fast so the caller can fall back
        data = json.dumps(body).encode()
        last = "no attempt"
        for n, i in enumerate(self._order(), 1):
            t0 = time.perf_counter()
            headers = {"Authorization": "Bearer " + self._keys[i], "Content-Type": "application/json"}
            try:
                status, raw = self.transport(self.base_url + "/chat/completions", data, headers, timeout)
            except (socket.timeout, TimeoutError, urllib.error.URLError, ConnectionError, OSError) as exc:
                status, raw, last = None, b"", f"key{i}:{type(exc).__name__}"
            obs.log("external.call", "debug" if status and 200 <= status < 300 else "warn", "boxapi chat", upstream="llm.boxapi",
                    endpoint="/chat/completions", attempt=n, max_attempts=len(self._keys), status=status, key_slot=i,
                    duration_ms=int((time.perf_counter() - t0) * 1000), error=None if status else last)
            if status is not None and 200 <= status < 300:
                self.stats["calls"] += 1
                br.success()
                return json.loads(raw)
            if status is not None:
                last = f"key{i}:http_{status}"
                if status not in FAILOVER_STATUS and status < 500:
                    br.success()                         # the service answered; the request itself is bad
                    raise BoxAPIError(f"{last}: {raw[:200].decode(errors='replace')}")  # a bad request fails on every key
            with self._lock:
                self._cool_until[i] = self.clock() + self.cooldown
            self.stats["failovers"] += 1
            self.stats["errors"] = (self.stats["errors"] + [last])[-10:]
        br.failure()
        raise BoxAPIError(f"all BoxAPI keys failed; last {last}")
