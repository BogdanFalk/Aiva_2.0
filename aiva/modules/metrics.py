"""Lightweight operations profiler — 'where does her time go?'.

A single `metrics` table on the shared connection, written at a few chokepoints (LLM
side-calls, tool handlers, voice-turn latency). The design keeps it cheap and OFF the
real-time audio path: instrument sites call the module-level `record()` / `span()`, which
no-op unless a recorder is installed at boot, and every write is a small insert on the
event-loop thread (never per audio frame). The dashboard rolls it up into the Statistics
section (count / mean / max / total / share-of-time, plus cost).

Usage:
    from modules import metrics
    metrics.record("tool", "look_at_screen", ms, cost=0.0)
    async with metrics.span("llm", "becoming"):
        ...

stdlib-only, so importing it stays cheap everywhere (the dashboard imports it too).
"""

import sqlite3
import time
from datetime import datetime, timedelta

_REC = None   # installed at boot by main.py; until then every call is a no-op


def set_recorder(rec):
    global _REC
    _REC = rec


def record(system, op, ms, *, tokens=0, cost=0.0, ok=True, meta=None):
    if _REC is not None:
        _REC.record(system, op, ms, tokens=tokens, cost=cost, ok=ok, meta=meta)


class span:
    """`async with metrics.span('llm', 'becoming'): ...` — times the block and records it."""

    def __init__(self, system, op, *, tokens=0, cost=0.0):
        self.system, self.op = system, op
        self.tokens, self.cost = tokens, cost
        self._t0 = None

    async def __aenter__(self):
        self._t0 = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, *_):
        record(self.system, self.op, (time.perf_counter() - self._t0) * 1000.0,
               tokens=self.tokens, cost=self.cost, ok=exc_type is None)
        return False


class Metrics:
    """Owns the `metrics` table. record() is a single insert (loop-thread only, like the
    experience bus); it prunes old rows periodically so the table stays bounded."""

    def __init__(self, conn: sqlite3.Connection, retain_days=30):
        self.conn = conn
        self.retain_days = retain_days
        self._n = 0
        self._create()

    def _create(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                system TEXT NOT NULL,
                op TEXT NOT NULL,
                ms REAL NOT NULL,
                tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0,
                ok INTEGER NOT NULL DEFAULT 1,
                meta TEXT
            )""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(ts)")
        self.conn.commit()

    def record(self, system, op, ms, *, tokens=0, cost=0.0, ok=True, meta=None):
        try:
            self.conn.execute(
                "INSERT INTO metrics (ts, system, op, ms, tokens, cost_usd, ok, meta) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(), system, op, float(ms), int(tokens or 0),
                 float(cost or 0.0), 1 if ok else 0, meta))
            self.conn.commit()
            self._n += 1
            if self._n % 200 == 0:
                self._prune()
        except sqlite3.Error:
            pass  # profiling must never take her down

    def _prune(self):
        cutoff = (datetime.now() - timedelta(days=self.retain_days)).isoformat()
        try:
            self.conn.execute("DELETE FROM metrics WHERE ts < ?", (cutoff,))
            self.conn.commit()
        except sqlite3.Error:
            pass
