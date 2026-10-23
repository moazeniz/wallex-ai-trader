"""Durable SQLite job queue (outbox + dead-letter) for background work that may fail. NEVER used for orders.

States: queued -> running -> succeeded | retrying (backoff) -> ... -> dead (after max_attempts). Jobs carry an
idempotency key: enqueueing the same (kind, idem) again returns the existing job. A worker thread runs registered
handlers, retries with exponential backoff + jitter, and logs `job.start/job.succeeded/job.retry/job.dead_letter`;
`queue.depth` / `queue.dead` metrics are logged every minute. On start-up (and on shutdown) jobs left `running` are
put back to `queued`, so in-flight work survives a restart.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable

from . import obs
from .resilience import backoff

SCHEMA = """CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, kind TEXT, idem TEXT, payload TEXT, state TEXT, attempts INTEGER,
max_attempts INTEGER, next_at REAL, created_at REAL, updated_at REAL, last_error TEXT, result TEXT, UNIQUE(kind, idem))"""


class JobQueue:
    def __init__(self, path: Path, clock=time.time):
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handlers: dict[str, Callable[[dict], object]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        with self._db() as db:
            db.execute(SCHEMA)
            n = db.execute("UPDATE jobs SET state='queued', updated_at=? WHERE state='running'", (self.clock(),)).rowcount
        if n:
            obs.log("queue.recovered", "warn", f"{n} interrupted jobs re-queued", queue="jobs", value=n)

    def _db(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        return _Closing(db)

    def register(self, kind: str, handler: Callable[[dict], object]):
        self.handlers[kind] = handler

    def enqueue(self, kind: str, payload: dict, idem: str | None = None, max_attempts: int = 5, delay: float = 0.0) -> str:
        now = self.clock()
        idem = idem or uuid.uuid4().hex
        with self._db() as db:
            row = db.execute("SELECT id FROM jobs WHERE kind=? AND idem=?", (kind, idem)).fetchone()
            if row:
                return row["id"]
            job_id = uuid.uuid4().hex[:16]
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                       (job_id, kind, idem, json.dumps(payload), "queued", 0, max_attempts, now + delay, now, now, None, None))
        obs.log("job.enqueued", "info", kind, queue="jobs", job_id=job_id, kind=kind)
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, state: str | None = None, limit: int = 50) -> list[dict]:
        with self._db() as db:
            q = "SELECT * FROM jobs" + (" WHERE state=?" if state else "") + " ORDER BY updated_at DESC LIMIT ?"
            rows = db.execute(q, ((state, limit) if state else (limit,))).fetchall()
        return [self._row(r) for r in rows]

    def counts(self) -> dict:
        with self._db() as db:
            return {r[0]: r[1] for r in db.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state")}

    def replay(self, job_id: str) -> bool:
        with self._db() as db:
            n = db.execute("UPDATE jobs SET state='queued', attempts=0, next_at=?, updated_at=? WHERE id=? AND state IN ('dead','succeeded')",
                           (self.clock(), self.clock(), job_id)).rowcount
        obs.log("job.replay", "info", job_id, queue="jobs", job_id=job_id, value=n)
        return bool(n)

    def discard(self, job_id: str) -> bool:
        with self._db() as db:
            n = db.execute("UPDATE jobs SET state='discarded', updated_at=? WHERE id=? AND state='dead'", (self.clock(), job_id)).rowcount
        obs.log("job.discard", "info", job_id, queue="jobs", job_id=job_id, value=n)
        return bool(n)

    def run_once(self) -> bool:
        """Claim and run one due job. Returns False when nothing was due."""
        now = self.clock()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE state IN ('queued','retrying') AND next_at<=? ORDER BY next_at LIMIT 1", (now,)).fetchone()
            if not row:
                db.execute("COMMIT")
                return False
            db.execute("UPDATE jobs SET state='running', attempts=attempts+1, updated_at=? WHERE id=?", (now, row["id"]))
            db.execute("COMMIT")
        job = self._row(row)
        job["attempts"] += 1
        handler = self.handlers.get(job["kind"])
        obs.set_context(request_id="job-" + job["id"])
        obs.log("job.start", "info", job["kind"], queue="jobs", job_id=job["id"], attempt=job["attempts"], max_attempts=job["max_attempts"])
        try:
            if handler is None:
                raise LookupError(f"no handler for {job['kind']}")
            result = handler(job["payload"])
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"[:500]
            dead = job["attempts"] >= job["max_attempts"]
            wait = backoff(job["attempts"], base=5.0, cap=600.0)
            with self._db() as db:
                db.execute("UPDATE jobs SET state=?, next_at=?, updated_at=?, last_error=? WHERE id=?",
                           ("dead" if dead else "retrying", self.clock() + wait, self.clock(), err, job["id"]))
            if dead:
                obs.log("job.dead_letter", "error", job["kind"], queue="jobs", job_id=job["id"], attempt=job["attempts"],
                        error_type=type(exc).__name__, error=str(exc)[:200], stack=traceback.format_exc()[-3000:])
            else:
                obs.log("job.retry", "warn", job["kind"], queue="jobs", job_id=job["id"], attempt=job["attempts"],
                        max_attempts=job["max_attempts"], backoff_ms=int(wait * 1000), error_type=type(exc).__name__, error=str(exc)[:200])
            return True
        with self._db() as db:
            db.execute("UPDATE jobs SET state='succeeded', updated_at=?, result=?, last_error=NULL WHERE id=?",
                       (self.clock(), json.dumps(result, default=str)[:20000], job["id"]))
        obs.log("job.succeeded", "info", job["kind"], queue="jobs", job_id=job["id"], attempt=job["attempts"])
        return True

    def start(self, poll: float = 1.0, metrics_every: float = 60.0):
        def loop():
            last_metrics = 0.0
            while not self._stop.is_set():
                try:
                    busy = self.run_once()
                    if time.time() - last_metrics >= metrics_every:
                        c = self.counts()
                        obs.log("queue.metrics", "info", "", queue="jobs", metric="queue.depth",
                                value=c.get("queued", 0) + c.get("retrying", 0) + c.get("running", 0), dead=c.get("dead", 0))
                        last_metrics = time.time()
                except Exception as exc:  # the worker never dies
                    obs.log_error("queue.worker_error", exc)
                    busy = False
                if not busy:
                    self._stop.wait(poll)
        self._thread = threading.Thread(target=loop, name="job-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        with self._db() as db:   # anything still marked running goes back to queued for the next start
            db.execute("UPDATE jobs SET state='queued', updated_at=? WHERE state='running'", (self.clock(),))

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        d["payload"] = json.loads(d["payload"] or "{}")
        d["result"] = json.loads(d["result"]) if d.get("result") else None
        return d


class _Closing:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self.db

    def __exit__(self, *a):
        self.db.close()
        return False
