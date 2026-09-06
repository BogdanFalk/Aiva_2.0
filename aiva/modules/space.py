"""Her space — a curated corner where Aiva keeps things she gathers from the web.

Like her book library, but for anything she finds and brings back: an image she likes, a clip,
a page worth remembering, a link. Each item carries WHY she kept it (her taste, in her voice),
where it came from, and where the file lives — so the space feels like hers, not a download dump.

`SpaceMemory` owns the `finds` table on the shared `Memory.conn` (sqlite calls on the event-loop
thread only, same rule as every other store). Deduped by file sha256 or subject embedding, recalled
by embedding with a lexical fallback, and permanently vetoable (`drop_find`) like reading traits.
The actual files live under `aiva/space/` (see main.py); this is only the index over them.

Audit:  cd aiva && ..\\.venv\\Scripts\\python -m modules.space --show
"""

import json
import os
import re
import sqlite3
from datetime import datetime

from modules.task_memory import cosine, jaccard, normalize_goal, pack_vec, unpack_vec

KINDS = {"image", "video", "text", "link", "file"}
DEDUPE_COSINE = 0.86
DEDUPE_JACCARD = 0.60


def _now():
    return datetime.now().isoformat()


def _ago(ts):
    try:
        days = (datetime.now() - datetime.fromisoformat(ts)).days
    except (TypeError, ValueError):
        return "unknown"
    return "today" if days <= 0 else ("yesterday" if days == 1 else f"{days} days ago")


def _domain(url):
    m = re.match(r"https?://([^/]+)", url or "")
    return m.group(1).lower().lstrip("www.") if m else ""


class SpaceMemory:
    def __init__(self, conn: sqlite3.Connection, *, embed=None, log=print, now=None):
        self.conn = conn
        self._embed = embed              # async (text) -> vec | None
        self.log = log
        self._now = now or _now
        self._create_tables()

    def _create_tables(self):
        c = self.conn
        c.execute("""CREATE TABLE IF NOT EXISTS finds (
            id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL DEFAULT 'image',
            subject TEXT NOT NULL, why TEXT NOT NULL DEFAULT '', source_url TEXT,
            path TEXT, file_name TEXT, sha256 TEXT, tags TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'kept', added_ts TEXT NOT NULL, updated_ts TEXT NOT NULL,
            meta_json TEXT NOT NULL DEFAULT '{}', embedding BLOB,
            UNIQUE(sha256))""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_finds_status ON finds(status)")
        c.commit()

    # -- row helpers --
    _COLS = ("id", "kind", "subject", "why", "source_url", "path", "file_name", "sha256",
             "tags", "status", "added_ts", "updated_ts", "meta_json")

    def _row(self, row):
        d = dict(zip(self._COLS, row))
        d["meta"] = json.loads(d.pop("meta_json") or "{}")
        d["tags"] = [t for t in (d["tags"] or "").split(",") if t]
        return d

    def _select(self, where="", params=(), limit=200):
        cols = ", ".join(self._COLS)
        rows = self.conn.execute(
            f"SELECT {cols} FROM finds {where} ORDER BY updated_ts DESC, id DESC LIMIT ?",
            (*params, limit)).fetchall()
        return [self._row(r) for r in rows]

    async def _vec(self, text):
        if not self._embed:
            return None
        try:
            return await self._embed(text)
        except Exception:
            return None

    @staticmethod
    def _text_of(subject, why, tags):
        return ". ".join(p for p in (subject, why, " ".join(tags or [])) if p)

    def _match(self, subject, why, tags, vec):
        """An existing find this one duplicates (any status), or None — by embedding, else lexical."""
        q_lex = normalize_goal(self._text_of(subject, why, tags))
        for r in self.conn.execute(
                "SELECT id, subject, why, tags, status, embedding FROM finds").fetchall():
            other = unpack_vec(r[5])
            if vec and other:
                score, thresh = cosine(vec, other), DEDUPE_COSINE
            else:
                score, thresh = jaccard(q_lex, normalize_goal(
                    self._text_of(r[1], r[2], (r[3] or "").split(",")))), DEDUPE_JACCARD
            if score >= thresh:
                return {"id": r[0], "status": r[4]}
        return None

    async def save_find(self, subject, why="", *, kind="image", source_url=None, path=None,
                        file_name=None, tags=None, sha256=None, status="kept", meta=None):
        """Add a find, or reinforce/update a near-duplicate. A vetoed (dropped) match stays gone.
        Returns {"status": saved|updated|blocked, "id", "subject"}."""
        subject = (subject or "").strip()
        if not subject:
            return {"status": "blocked", "reason": "no subject"}
        kind = kind if kind in KINDS else "file"
        why = (why or "").strip()
        tags = [t.strip() for t in (tags or []) if t and t.strip()]
        now = self._now()
        # sha256 is the strongest dedupe key (identical file); else fall back to meaning
        dup = None
        if sha256:
            r = self.conn.execute("SELECT id, status FROM finds WHERE sha256 = ?", (sha256,)).fetchone()
            if r:
                dup = {"id": r[0], "status": r[1]}
        vec = await self._vec(subject)               # identity of a find is its subject
        if dup is None:
            dup = self._match(subject, why, tags, vec)
        if dup and dup["status"] == "dropped":
            return {"status": "blocked", "id": dup["id"], "subject": subject}
        if dup:
            self.conn.execute(
                "UPDATE finds SET subject=?, why=?, kind=?, source_url=COALESCE(?, source_url), "
                "path=COALESCE(?, path), file_name=COALESCE(?, file_name), sha256=COALESCE(?, sha256), "
                "tags=?, status='kept', updated_ts=? WHERE id=?",
                (subject, why, kind, source_url, path, file_name, sha256, ",".join(tags), now, dup["id"]))
            self.conn.commit()
            return {"status": "updated", "id": dup["id"], "subject": subject}
        cur = self.conn.execute(
            "INSERT INTO finds (kind, subject, why, source_url, path, file_name, sha256, tags, "
            "status, added_ts, updated_ts, meta_json, embedding) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (kind, subject, why, source_url, path, file_name, sha256, ",".join(tags), status, now,
             now, json.dumps(meta or {}, ensure_ascii=False), pack_vec(vec)))
        self.conn.commit()
        return {"status": "saved", "id": cur.lastrowid, "subject": subject}

    def list_finds(self, status="kept", kind=None, limit=50):
        where = "WHERE status = ?"
        params = [status]
        if kind:
            where += " AND kind = ?"
            params.append(kind)
        return self._select(where, tuple(params), limit)

    async def recall(self, query, k=5):
        """Her strongest-matching kept finds for a query, ranked by embedding (lexical fallback)."""
        rows = self.conn.execute(
            f"SELECT {', '.join(self._COLS)}, embedding FROM finds WHERE status = 'kept'").fetchall()
        if not rows:
            return {"items": [], "note": "Your space is empty for that — say you haven't kept anything like it."}
        qvec = await self._vec(query or "")
        qlex = normalize_goal(query or "")
        scored = []
        for r in rows:
            d = self._row(r[:-1])
            other = unpack_vec(r[-1])
            if qvec and other:
                score = cosine(qvec, other)
            else:
                score = jaccard(qlex, normalize_goal(self._text_of(d["subject"], d["why"], d["tags"])))
            scored.append((score, d))
        scored.sort(key=lambda s: s[0], reverse=True)
        items = [d for _, d in scored[:k]]
        return {"items": items,
                "note": "These are things you kept in your space; talk about them in your own words, "
                        "and offer to open one or give the link if he wants it."}

    async def drop_find(self, what, reason=None):
        """Veto a find (id -> substring -> embedding/lexical). Permanent. Returns the subject."""
        row = None
        if isinstance(what, int) or (isinstance(what, str) and what.isdigit()):
            row = self.conn.execute("SELECT id, subject FROM finds WHERE id = ?",
                                    (int(what),)).fetchone()
        if not row:
            q = str(what or "").strip().lower()
            for r in self.conn.execute(
                    "SELECT id, subject, why, tags FROM finds WHERE status != 'dropped'").fetchall():
                hay = " ".join((r[1] or "", r[2] or "", r[3] or "")).lower()
                if q and q in hay:
                    row = (r[0], r[1])
                    break
        if not row:
            vec = await self._vec(str(what))
            best, best_s = None, 0.0
            for r in self.conn.execute(
                    "SELECT id, subject, why, tags, embedding FROM finds "
                    "WHERE status != 'dropped'").fetchall():
                other = unpack_vec(r[4])
                s = cosine(vec, other) if (vec and other) else jaccard(
                    normalize_goal(str(what)), normalize_goal(self._text_of(r[1], r[2], (r[3] or "").split(","))))
                if s > best_s:
                    best, best_s = (r[0], r[1]), s
            if best and best_s >= 0.6:
                row = best
        if not row:
            return None
        self.conn.execute("UPDATE finds SET status='dropped', updated_ts=? WHERE id=?",
                          (self._now(), row[0]))
        self.conn.commit()
        return row[1]

    def status(self):
        counts = {}
        for kind, n in self.conn.execute(
                "SELECT kind, COUNT(*) FROM finds WHERE status='kept' GROUP BY kind").fetchall():
            counts[kind] = n
        recent = self.list_finds(limit=8)
        return {"total": sum(counts.values()), "by_kind": counts, "recent": recent,
                "note": "This is your own space of things you've gathered; mention it lightly, "
                        "only when it's relevant or he asks."}

    def status_lines(self, limit=8):
        lines = []
        for d in self.list_finds(limit=limit):
            src = _domain(d["source_url"]) if d["source_url"] else ""
            why = f" — {d['why']}" if d["why"] else ""
            lines.append(f"{d['kind']}: {d['subject']}{why}" + (f" (from {src})" if src else ""))
        return lines


def _show():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db = os.path.join(here, "memory.db")
    if not os.path.exists(db):
        print("(no memory.db yet)")
        return
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT kind, subject, why, source_url, status, added_ts FROM finds "
                           "ORDER BY updated_ts DESC").fetchall()
    except sqlite3.Error:
        print("(no finds table yet)")
        return
    if not rows:
        print("(her space is empty)")
        return
    for kind, subject, why, url, status, ts in rows:
        tag = f"[{kind}/{status}]"
        print(f"  {tag:<16} {subject}" + (f" — {why}" if why else "")
              + (f"  <{_domain(url)}>" if url else "") + f"   ({_ago(ts)})")


if __name__ == "__main__":
    import sys
    _show() if "--show" in sys.argv else print(__doc__)
