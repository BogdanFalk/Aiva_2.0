"""`aiva/house.db`: the one database several processes share.

The house process writes the conversation log and events (`house_memory.HouseMemory`); every
sister's process opens the same file for the spend caps (`limits.ServiceLimits` on this
connection), so the once-per-period alert markers in `limits_kv` and the monthly counters are
shared across characters. Multi-process safety comes from WAL + a busy timeout — not from ATTACH
(limits.py SQL is unqualified, `VACUUM INTO` backups skip attached dbs, the dashboard opens by path).

`migrate_limits_from` copies a character's old `service_usage`/`service_pause`/`limits_kv` rows
out of her `memory.db` exactly once (stamped in the house's `limits_kv`), so the counters she has
already spent this month survive the move.
"""

import os
import sqlite3
from datetime import datetime

from modules.limits import ServiceLimits

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "house.db")
MIGRATION_KEY = "migrated_from_memory"
LIMITS_TABLES = ("service_usage", "service_pause", "limits_kv")


def house_db_path():
    return os.getenv("AIVA_HOUSE_DB") or DEFAULT_PATH


def open_house_db(path=None, *, timeout=5.0) -> sqlite3.Connection:
    """WAL, busy_timeout, synchronous=NORMAL — safe for the house and both sisters at once.
    `check_same_thread` keeps its default: use each connection from one thread only."""
    path = path or house_db_path()
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _has_table(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def migrate_limits_from(house_conn, memory_conn, *, now=None) -> int:
    """Copy the spend-cap tables from a character's own db into the shared house db, once.
    Returns the number of rows copied (0 when already migrated or nothing to copy). Usage adds up
    (both sides may have counted this month), a pause from the old db replaces, alert markers are
    kept if the house already has them."""
    ServiceLimits(house_conn, caps={})          # creates the three tables if missing
    if house_conn.execute("SELECT 1 FROM limits_kv WHERE key=?", (MIGRATION_KEY,)).fetchone():
        return 0
    copied = 0
    if _has_table(memory_conn, "service_usage"):
        for service, period, used in memory_conn.execute(
                "SELECT service, period, used FROM service_usage").fetchall():
            house_conn.execute(
                "INSERT INTO service_usage (service, period, used) VALUES (?,?,?) "
                "ON CONFLICT(service, period) DO UPDATE SET used = used + excluded.used",
                (service, period, used or 0))
            copied += 1
    if _has_table(memory_conn, "service_pause"):
        for row in memory_conn.execute(
                "SELECT service, period, reason, since_ts, resets_ts FROM service_pause").fetchall():
            house_conn.execute(
                "INSERT OR REPLACE INTO service_pause (service, period, reason, since_ts, resets_ts) "
                "VALUES (?,?,?,?,?)", row)
            copied += 1
    if _has_table(memory_conn, "limits_kv"):
        for key, value in memory_conn.execute("SELECT key, value FROM limits_kv").fetchall():
            if key == MIGRATION_KEY:
                continue
            house_conn.execute("INSERT OR IGNORE INTO limits_kv (key, value) VALUES (?,?)", (key, value))
            copied += 1
    stamp = now if isinstance(now, str) else datetime.now().isoformat()
    house_conn.execute("INSERT OR REPLACE INTO limits_kv (key, value) VALUES (?, ?)", (MIGRATION_KEY, stamp))
    house_conn.commit()
    return copied
