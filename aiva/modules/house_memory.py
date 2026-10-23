"""The house's own memory: the conversation log every sister's line lands in, the cross-sister
event stream (kin), and a small kv for the floor's persisted state and daily counters.

`HouseMemory(conn)` takes an open sqlite connection (see `housedb.open_house_db`; tests pass
`sqlite3.connect(":memory:")`). WAL + synchronous=NORMAL like `memory.py`, so the read-only
dashboard never blocks the house's writes. Written by the house process only.

Tables:
- house_transcript(id, ts, turn_id, speaker user|<slug>, addressee him|<slug>|both, text,
  audible INT, kind, interrupted INT) — one row per user line (kind = how it was routed) and one
  per `said`. On a user line `both` is both sisters; on a sister's line it is him and her sister.
- house_events(id, ts, slug, kind, summary, capability, relayed_to) — what a sister did/made/said,
  relayed to the others; each sister keeps her own cursor into it.
- house_kv(key PRIMARY KEY, value, updated_ts) — sister_talk mode, turn sequence, day counters.

Timestamps are ISO strings. `now` may be injected as a callable returning either an ISO string or
epoch seconds (the Floor uses a float clock); `day_count`/`bump_day_count` mirror inner.py's
"YYYY-MM-DD:n" encoding so the daily caps reset at midnight without a scheduler.
"""

from datetime import datetime, timedelta


def _iso(v):
    if v is None:
        return datetime.now().isoformat()
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v).isoformat()
    return str(v)


class HouseMemory:
    def __init__(self, conn, *, now=None):
        self.conn = conn
        self._now = now or (lambda: datetime.now().isoformat())
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        self._create_tables()

    def _ts(self, now=None):
        return _iso(now if now is not None else self._now())

    def _create_tables(self):
        c = self.conn
        c.execute("""CREATE TABLE IF NOT EXISTS house_transcript (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            turn_id INTEGER,
            speaker TEXT NOT NULL,
            addressee TEXT NOT NULL,
            text TEXT NOT NULL,
            audible INTEGER NOT NULL DEFAULT 1,
            kind TEXT NOT NULL DEFAULT '',
            interrupted INTEGER NOT NULL DEFAULT 0)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_house_transcript_ts ON house_transcript(ts)")
        c.execute("""CREATE TABLE IF NOT EXISTS house_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            slug TEXT NOT NULL,
            kind TEXT NOT NULL,
            summary TEXT NOT NULL,
            capability TEXT,
            relayed_to TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS house_kv (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_ts TEXT)""")
        c.commit()

    # --- transcript -------------------------------------------------------------------

    def log_user_line(self, turn_id, text, addressee, kind="", *, now=None) -> int:
        cur = self.conn.execute(
            "INSERT INTO house_transcript (ts, turn_id, speaker, addressee, text, audible, kind, interrupted) "
            "VALUES (?,?,?,?,?,1,?,0)",
            (self._ts(now), turn_id, "user", addressee or "", text or "", kind or ""))
        self.conn.commit()
        return cur.lastrowid

    def log_said(self, turn_id, slug, text, addressee, audible=True, interrupted=False, *, kind="", now=None) -> int:
        cur = self.conn.execute(
            "INSERT INTO house_transcript (ts, turn_id, speaker, addressee, text, audible, kind, interrupted) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (self._ts(now), turn_id, slug, addressee or "him", text or "", 1 if audible else 0,
             kind or "", 1 if interrupted else 0))
        self.conn.commit()
        return cur.lastrowid

    def recent_transcript(self, limit=50, before_id=None, day=None) -> list:
        """Newest first. `before_id` pages backwards; `day` = 'YYYY-MM-DD' filters one day."""
        where, args = [], []
        if before_id is not None:
            where.append("id < ?")
            args.append(before_id)
        if day:
            where.append("ts LIKE ?")
            args.append(f"{day}%")
        sql = ("SELECT id, ts, turn_id, speaker, addressee, text, audible, kind, interrupted "
               "FROM house_transcript")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        rows = self.conn.execute(sql, args).fetchall()
        return [{"id": r[0], "ts": r[1], "turn_id": r[2], "speaker": r[3], "addressee": r[4],
                 "text": r[5], "audible": bool(r[6]), "kind": r[7], "interrupted": bool(r[8])}
                for r in rows]

    # --- events ------------------------------------------------------------------------

    def add_event(self, slug, kind, summary, capability=None, relayed_to=None, *, now=None) -> int:
        if isinstance(relayed_to, (list, tuple, set)):
            relayed_to = ",".join(sorted(relayed_to))
        cur = self.conn.execute(
            "INSERT INTO house_events (ts, slug, kind, summary, capability, relayed_to) VALUES (?,?,?,?,?,?)",
            (self._ts(now), slug, kind, summary or "", capability, relayed_to))
        self.conn.commit()
        return cur.lastrowid

    def events_since(self, after_id=0, limit=200) -> list:
        rows = self.conn.execute(
            "SELECT id, ts, slug, kind, summary, capability, relayed_to FROM house_events "
            "WHERE id > ? ORDER BY id ASC LIMIT ?", (int(after_id or 0), int(limit))).fetchall()
        return [{"id": r[0], "ts": r[1], "slug": r[2], "kind": r[3], "summary": r[4],
                 "capability": r[5], "relayed_to": r[6]} for r in rows]

    def prune(self, days=30, *, now=None) -> int:
        cutoff = (datetime.fromisoformat(self._ts(now)) - timedelta(days=days)).isoformat()
        n = self.conn.execute("DELETE FROM house_transcript WHERE ts < ?", (cutoff,)).rowcount
        n += self.conn.execute("DELETE FROM house_events WHERE ts < ?", (cutoff,)).rowcount
        self.conn.commit()
        return n

    # --- kv + daily counters ------------------------------------------------------------

    def kv_get(self, key, default=None):
        r = self.conn.execute("SELECT value FROM house_kv WHERE key=?", (key,)).fetchone()
        return r[0] if r else default

    def kv_set(self, key, value, *, now=None):
        self.conn.execute(
            "INSERT INTO house_kv (key, value, updated_ts) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ts=excluded.updated_ts",
            (key, None if value is None else str(value), self._ts(now)))
        self.conn.commit()

    def day_count(self, key, now=None) -> int:
        raw = self.kv_get(key, "") or ""
        today = self._ts(now)[:10]
        if raw.startswith(today + ":"):
            try:
                return int(raw.split(":", 1)[1])
            except ValueError:
                return 0
        return 0

    def bump_day_count(self, key, now=None) -> int:
        n = self.day_count(key, now) + 1
        self.kv_set(key, f"{self._ts(now)[:10]}:{n}", now=now)
        return n
