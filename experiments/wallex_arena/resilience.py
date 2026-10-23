"""One retry helper for the codebase: exponential backoff with full jitter, per-upstream circuit breakers, every attempt
logged as `external.call` (successes at debug level, failures and retries at warn).

Use it only for idempotent calls: public market data, account/order *reads*, LLM calls, connectivity probes.
Order placement is never retried here: a create-order POST is sent once, and an ambiguous outcome is settled by looking
the order up by its client order id (live.py create_order). Cancels are not auto-retried either; the engine's next
sync re-reads the order. See LIVE.md.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Callable

from . import obs

TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524, 529}


class CircuitOpen(ConnectionError):   # an OSError, so existing network-failure handlers treat it the same way
    def __init__(self, upstream: str, retry_in: float):
        super().__init__(f"circuit open for {upstream}; next probe in {retry_in:.0f}s")
        self.upstream, self.retry_in = upstream, retry_in


class HTTPStatusError(Exception):
    """Raised by call sites that turn a non-2xx response into an exception the retry helper can classify."""
    def __init__(self, status: int, detail: str = "", retry_after: float | None = None):
        super().__init__(f"http_{status} {detail}"[:200])
        self.status, self.detail, self.retry_after = status, detail, retry_after


class CircuitBreaker:
    """closed -> open after `threshold` consecutive failures -> half_open after `reset_after` s (one probe) -> closed."""

    def __init__(self, upstream: str, threshold: int = 5, reset_after: float = 30.0, clock=time.monotonic):
        self.upstream, self.threshold, self.reset_after, self.clock = upstream, threshold, reset_after, clock
        self.failures, self.opened_at, self.state = 0, 0.0, "closed"
        self._lock = threading.Lock()
        self._probing = False

    def before(self):
        with self._lock:
            if self.state == "open":
                wait = self.reset_after - (self.clock() - self.opened_at)
                if wait > 0:
                    raise CircuitOpen(self.upstream, wait)
                self.state = "half_open"
                obs.log("circuit.half_open", "info", f"{self.upstream} probing", upstream=self.upstream, circuit="half_open")
            if self.state == "half_open":
                if self._probing:
                    raise CircuitOpen(self.upstream, 1.0)
                self._probing = True

    def success(self):
        with self._lock:
            if self.state != "closed":
                obs.log("circuit.closed", "info", f"{self.upstream} recovered", upstream=self.upstream, circuit="closed")
            self.failures, self.state, self._probing = 0, "closed", False

    def failure(self):
        with self._lock:
            self.failures += 1
            self._probing = False
            if self.state == "half_open" or (self.state == "closed" and self.failures >= self.threshold):
                self.state, self.opened_at = "open", self.clock()
                obs.log("circuit.open", "warn", f"{self.upstream} failing", upstream=self.upstream, circuit="open", attempt=self.failures)


_breakers: dict[str, CircuitBreaker] = {}
_reg_lock = threading.Lock()


def breaker(upstream: str, threshold: int = 5, reset_after: float = 30.0) -> CircuitBreaker:
    with _reg_lock:
        if upstream not in _breakers:
            _breakers[upstream] = CircuitBreaker(upstream, threshold, reset_after)
        return _breakers[upstream]


def breaker_states() -> dict:
    return {k: {"state": b.state, "failures": b.failures} for k, b in _breakers.items()}


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, HTTPStatusError):
        return exc.status in TRANSIENT_STATUS or exc.status >= 500
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def backoff(attempt: int, base: float = 0.5, cap: float = 20.0) -> float:
    """Full jitter: uniform(0, min(cap, base * 2**attempt))."""
    return random.uniform(0, min(cap, base * 2 ** attempt))


def call(upstream: str, fn: Callable[[], object], *, max_attempts: int = 3, base: float = 0.5, cap: float = 20.0,
         retry_if: Callable[[BaseException], bool] = is_transient, endpoint: str = "", use_breaker: bool = True,
         sleep: Callable[[float], None] = time.sleep, **log_fields):
    """Run fn() with retries for transient errors. Honors Retry-After on HTTPStatusError. Re-raises the last error."""
    br = breaker(upstream) if use_breaker else None
    for attempt in range(1, max_attempts + 1):
        if br:
            br.before()                                   # raises CircuitOpen without calling upstream
        t = time.perf_counter()
        try:
            result = fn()
        except BaseException as exc:
            ms = int((time.perf_counter() - t) * 1000)
            retry = attempt < max_attempts and retry_if(exc)
            if br:   # only transient failures count against the circuit; a definite answer (4xx, tool error) closes a probe
                br.failure() if is_transient(exc) else br.success()
            wait = None
            if retry:
                wait = backoff(attempt - 1, base, cap)
                try:
                    ra = float(getattr(exc, "retry_after", None) or 0)
                except (TypeError, ValueError):
                    ra = 0.0
                if ra > 0:
                    wait = max(wait, min(ra, cap))
            obs.log("external.call", "warn", f"{upstream} {endpoint} failed", upstream=upstream, endpoint=endpoint, attempt=attempt,
                    max_attempts=max_attempts, duration_ms=ms, status=getattr(exc, "status", None), error_type=type(exc).__name__,
                    error=str(exc)[:160], backoff_ms=int(wait * 1000) if wait is not None else None,
                    circuit=br.state if br else None, **log_fields)
            if not retry:
                raise
            sleep(wait)
            continue
        if br:
            br.success()
        obs.log("external.call", "debug", f"{upstream} {endpoint} ok", upstream=upstream, endpoint=endpoint, attempt=attempt,
                max_attempts=max_attempts, duration_ms=int((time.perf_counter() - t) * 1000), circuit=br.state if br else None, **log_fields)
        return result
