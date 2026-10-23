"""Anthropic Messages API over plain HTTP (no SDK), for direct keys or the team relay.

ANTHROPIC_BASE_URL (default https://api.anthropic.com; on Liara the team LLM gate, http://<gate-host>/meow-llm),
ANTHROPIC_AUTH_TOKEN (sent as Authorization: Bearer, used by the relay) or ANTHROPIC_API_KEY / a key file (x-api-key).
Non-streaming POST /v1/messages. Credentials never appear in logs, exceptions or repr.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .resilience import HTTPStatusError, call, is_transient

DEFAULT_BASE = "https://api.anthropic.com"
VERSION = "2023-06-01"


class LLMHTTPError(Exception):
    def __init__(self, provider: str, status: int | None, detail: str):
        super().__init__(f"{provider}: {'http_%s' % status if status else 'network'} {detail}"[:300])
        self.provider, self.status, self.detail = provider, status, detail


def _urllib_transport(url: str, body: bytes, headers: dict, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class AnthropicHTTP:
    def __init__(self, base_url: str = DEFAULT_BASE, auth_token: str | None = None, api_key: str | None = None,
                 transport: Callable[[str, bytes, dict, float], tuple[int, bytes]] | None = None, max_attempts: int = 2,
                 sleep: Callable[[float], None] = time.sleep, timeout: float | None = None):
        self.max_attempts, self.sleep = max_attempts, sleep
        # short per-request timeout so a hanging gateway hands over to the fallback provider quickly
        self.timeout = timeout or float(os.environ.get("LLM_TIMEOUT_S") or 30)
        self.base_url = base_url.rstrip("/")
        self._token, self._key = auth_token or None, api_key or None
        self.transport = transport or _urllib_transport
        self.stats = {"calls": 0, "errors": []}

    @classmethod
    def from_env(cls, key_path: Path | None = None, env: dict | None = None, **kw) -> "AnthropicHTTP":
        env = os.environ if env is None else env
        key = env.get("ANTHROPIC_API_KEY")
        if not key and not env.get("ANTHROPIC_AUTH_TOKEN") and key_path and Path(key_path).is_file():
            from .ai import read_key
            try:
                key = read_key(Path(key_path))
            except ValueError:
                key = None
        return cls(env.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE, env.get("ANTHROPIC_AUTH_TOKEN"), key, **kw)

    @property
    def configured(self) -> bool:
        return bool(self._token or self._key)

    def __repr__(self) -> str:
        auth = "bearer" if self._token else "x-api-key" if self._key else "missing"
        return f"AnthropicHTTP(base_url={self.base_url!r}, auth={auth})"

    def messages(self, body: dict, timeout: float | None = None) -> dict:
        timeout = timeout or self.timeout
        if not self.configured:
            raise LLMHTTPError("anthropic", None, "no credentials (ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY)")
        headers = {"Content-Type": "application/json", "anthropic-version": VERSION}
        if self._token:
            headers["Authorization"] = "Bearer " + self._token
        else:
            headers["x-api-key"] = self._key
        data = json.dumps(body).encode()

        def once():
            status, raw = self.transport(self.base_url + "/v1/messages", data, headers, timeout)
            if 200 <= status < 300:
                return json.loads(raw)
            raise HTTPStatusError(status, raw[:300].decode(errors="replace"))
        try:   # one retry for 429/5xx/network (backoff + jitter), breaker "llm.anthropic"; then the caller falls back
            out = call("llm.anthropic", once, max_attempts=self.max_attempts, base=1.0, cap=8.0, endpoint="/v1/messages",
                       retry_if=lambda e: isinstance(e, HTTPStatusError) and e.status in (429, 500, 502, 504, 529), sleep=self.sleep)
            # timeouts, connection errors and 503 (the gateway kill switch) are not retried: the caller falls back at once
        except HTTPStatusError as exc:
            self.stats["errors"] = (self.stats["errors"] + [f"http_{exc.status}"])[-10:]
            raise LLMHTTPError("anthropic", exc.status, exc.detail) from None
        except OSError as exc:  # timeouts, connection errors, open circuit
            self.stats["errors"] = (self.stats["errors"] + [type(exc).__name__])[-10:]
            raise LLMHTTPError("anthropic", None, type(exc).__name__) from None
        self.stats["calls"] += 1
        return out
