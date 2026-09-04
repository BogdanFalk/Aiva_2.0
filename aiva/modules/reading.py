"""Aiva studies books: the database of what she's read and the engine that reads.

`BookMemory` owns the sqlite side (tables on `Memory.conn`, event-loop thread
only — same rule as TaskMemory/`remember`). `ReadingEngine` runs one reading
*session* at a time in the background: it parses files in a worker thread
(`library.py` is sqlite-free), then for each section makes ONE close-reading
call, verifies the quotes it copied against the real text, and commits the
note and the cursor together so a restart mid-section simply redoes that
section. Progress reaches Aiva's voice only through the injected `announce` /
`quiet_note` callbacks (quiet while she's asleep — she reads while the user is
away). Whole-book synthesis on finish; embedding-based recall for "what did
you think of chapter two?".

Nothing here changes who she is — that's `becoming.py`, fed by the hooks.
"""

import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from modules import library
from modules.task_memory import (LEX_SAME, LEX_SIMILAR, SAME_TASK, SIMILAR, _ago, _now,
                                 cosine, jaccard, normalize_goal, pack_vec, unpack_vec)
from modules.utilities import LLM_RATES

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 512


# --- embeddings (books need longer inputs than task_memory.embed_text allows) --------

async def embed_long(client, text, timeout=8.0, max_chars=7000):
    try:
        r = await asyncio.wait_for(client.embeddings.create(
            model=EMBED_MODEL, input=(text or "")[:max_chars], dimensions=EMBED_DIMS), timeout=timeout)
        return list(r.data[0].embedding)
    except Exception:
        return None


async def embed_batch(client, texts, timeout=30.0, max_chars=7000, batch=64):
    """Embed many texts; returns a list aligned with `texts`, None where failed."""
    out = []
    for i in range(0, len(texts), batch):
        group = [(t or "")[:max_chars] for t in texts[i:i + batch]]
        try:
            r = await asyncio.wait_for(client.embeddings.create(
                model=EMBED_MODEL, input=group, dimensions=EMBED_DIMS), timeout=timeout)
            out.extend(list(d.embedding) for d in r.data)
        except Exception:
            out.extend([None] * len(group))
    return out


def _lex(a, b):
    """Jaccard mapped into the cosine threshold band (mirrors task_memory)."""
    j = jaccard(a, b)
    if j >= LEX_SAME:
        return SAME_TASK + (1 - SAME_TASK) * min(1.0, (j - LEX_SAME) / (1 - LEX_SAME))
    if j >= LEX_SIMILAR:
        return SIMILAR + (SAME_TASK - SIMILAR) * (j - LEX_SIMILAR) / (LEX_SAME - LEX_SIMILAR)
    return j * SIMILAR / LEX_SIMILAR if LEX_SIMILAR else 0.0


def _cost(model, prompt_tokens, completion_tokens):
    rate_in, rate_out = LLM_RATES.get(model, LLM_RATES.get("gpt-5.6-luna", (0.2, 1.2)))
    return (prompt_tokens * rate_in + completion_tokens * rate_out) / 1e6


# --- limits ------------------------------------------------------------------------------

@dataclass
class StudyLimits:
    sections_per_session: int = 4
    session_input_tokens: int = 45_000
    session_wall_secs: float = 300
    long_session_wall_secs: float = 3600
    session_cost_usd: float = 0.30
    book_cost_usd: float = 3.00
    huge_book_tokens: int = 600_000
    milestone_min_gap_secs: float = 120
    announce_floor_secs: float = 3.0
    llm_timeout: float = 90
    note_max_tokens: int = 2000
    wrap_max_tokens: int = 220
    synth_max_tokens: int = 2500
    max_quote_words: int = 25
    max_quotes_per_note: int = 4
    rolling_max_chars: int = 2200
    state_max_entries: int = 25
    front_skip_words: int = 400
    session_target_words: int = 3500     # aim for ~this many words per sitting (length-based)
    max_sections_per_session: int = 20   # safety cap so a wall of tiny sections (poems) still ends

    @classmethod
    def from_env(cls):
        g = os.getenv
        return cls(
            session_target_words=int(g("AIVA_STUDY_WORDS", "3500")),
            max_sections_per_session=int(g("AIVA_STUDY_MAX_SECTIONS", "20")),
            sections_per_session=int(g("AIVA_STUDY_SECTIONS", "4")),
            book_cost_usd=float(g("AIVA_BOOK_MAX_COST_USD", "3.0")),
        )


_PART_RE = re.compile(r"\(part\s+(\d+)\)", re.I)


def _is_continuation(section):
    """True if this section continues the previous chapter (a '(part 2+)' split, or a split/part
    kind) — i.e. stopping right BEFORE it would leave you mid-chapter/mid-action."""
    m = _PART_RE.search(section.get("title", "") or "")
    if m:
        return int(m.group(1)) >= 2
    return section.get("kind") in ("split", "part")


def plan_session_extent(secs, cursor, target_words, max_sections):
    """Pure: how far (exclusive `to` index) a session should read from `cursor`. Accumulate whole
    sections until ~target_words, then stop at the next NATURAL boundary — never right before a
    '(part N)' continuation of the current chapter. Reads at least one section; capped at
    max_sections so a shelf of tiny sections (poems) still ends."""
    n = len(secs)
    words, i = 0, cursor
    while i < n:
        words += (secs[i].get("words") or 0)
        i += 1
        if (i - cursor) >= max_sections:
            break
        if words >= target_words:
            nxt = secs[i] if i < n else None
            if nxt is None or not _is_continuation(nxt):
                break
    return max(i, cursor + 1) if cursor < n else cursor


# --- the database ------------------------------------------------------------------------

class BookMemory:
    def __init__(self, conn, embed=None):
        self.conn = conn
        self._embed = embed
        self._create_tables()

    def _create_tables(self):
        c = self.conn
        c.execute("""CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, author TEXT, kind TEXT,
            path TEXT NOT NULL, file_name TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE, format TEXT NOT NULL,
            words INTEGER NOT NULL DEFAULT 0, est_tokens INTEGER NOT NULL DEFAULT 0,
            sections INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'new', status_reason TEXT,
            cursor_section INTEGER NOT NULL DEFAULT 0,
            added_ts TEXT NOT NULL, started_ts TEXT, finished_ts TEXT, last_read_ts TEXT,
            sessions INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0,
            rolling_summary TEXT NOT NULL DEFAULT '', rolling_json TEXT NOT NULL DEFAULT '{}',
            meta_json TEXT NOT NULL DEFAULT '{}')""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_books_status ON books(status)")
        c.execute("""CREATE TABLE IF NOT EXISTS book_sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL, idx INTEGER NOT NULL,
            title TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'chapter', text TEXT NOT NULL,
            words INTEGER NOT NULL, est_tokens INTEGER NOT NULL, UNIQUE(book_id, idx))""")
        c.execute("""CREATE TABLE IF NOT EXISTS book_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL, section_id INTEGER NOT NULL,
            idx INTEGER NOT NULL, text TEXT NOT NULL, est_tokens INTEGER NOT NULL, embedding BLOB,
            UNIQUE(book_id, idx))""")
        c.execute("""CREATE TABLE IF NOT EXISTS book_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL, section_id INTEGER NOT NULL,
            section_idx INTEGER NOT NULL, session_id INTEGER NOT NULL, created_ts TEXT NOT NULL,
            summary TEXT NOT NULL, key_ideas TEXT NOT NULL DEFAULT '[]', quotes TEXT NOT NULL DEFAULT '[]',
            nuances TEXT NOT NULL DEFAULT '{}', questions TEXT NOT NULL DEFAULT '[]',
            connections TEXT NOT NULL DEFAULT '[]', reflection TEXT NOT NULL DEFAULT '',
            vocabulary TEXT NOT NULL DEFAULT '[]', feeling TEXT, skipped INTEGER NOT NULL DEFAULT 0,
            skip_reason TEXT, embedding BLOB, model TEXT, prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0,
            UNIQUE(book_id, section_idx))""")
        c.execute("""CREATE TABLE IF NOT EXISTS book_synthesis (
            book_id INTEGER PRIMARY KEY, created_ts TEXT NOT NULL, model TEXT NOT NULL,
            summary TEXT NOT NULL, arc TEXT NOT NULL DEFAULT '', themes TEXT NOT NULL DEFAULT '[]',
            people TEXT NOT NULL DEFAULT '[]', passages TEXT NOT NULL DEFAULT '[]',
            style TEXT NOT NULL DEFAULT '{}', takeaways TEXT NOT NULL DEFAULT '[]',
            disagreements TEXT NOT NULL DEFAULT '[]', verdict TEXT NOT NULL DEFAULT '',
            talk_line TEXT NOT NULL DEFAULT '', embedding BLOB, cost_usd REAL NOT NULL DEFAULT 0)""")
        c.execute("""CREATE TABLE IF NOT EXISTS book_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL, session_no INTEGER NOT NULL,
            started_ts TEXT NOT NULL, finished_ts TEXT, from_section INTEGER NOT NULL, to_section INTEGER,
            status TEXT NOT NULL DEFAULT 'running', trigger TEXT NOT NULL DEFAULT 'user',
            struck TEXT, next_hope TEXT, llm_calls INTEGER NOT NULL DEFAULT 0,
            prompt_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0, duration_s REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS wishlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, author TEXT, why TEXT,
            added_ts TEXT NOT NULL, fulfilled_book_id INTEGER, UNIQUE(title COLLATE NOCASE))""")
        c.commit()

    # -- row helpers --
    _BOOK_COLS = ("id", "title", "author", "kind", "path", "file_name", "sha256", "format", "words",
                  "est_tokens", "sections", "status", "status_reason", "cursor_section", "added_ts",
                  "started_ts", "finished_ts", "last_read_ts", "sessions", "cost_usd", "rolling_summary",
                  "rolling_json", "meta_json")

    def _book_row(self, row):
        d = dict(zip(self._BOOK_COLS, row))
        total = d["sections"] or 1
        done = min(d["cursor_section"], d["sections"])
        d["progress"] = {"done": done, "total": d["sections"], "pct": int(100 * done / total)}
        d["state"] = json.loads(d.pop("rolling_json") or "{}")
        d["meta"] = json.loads(d.pop("meta_json") or "{}")
        return d

    def _books(self, where="", params=(), limit=500):
        cols = ", ".join(self._BOOK_COLS)
        rows = self.conn.execute(
            f"SELECT {cols} FROM books {where} ORDER BY added_ts DESC, id DESC LIMIT ?",
            (*params, limit)).fetchall()
        return [self._book_row(r) for r in rows]

    _NOTE_COLS = ("id", "book_id", "section_id", "section_idx", "session_id", "created_ts", "summary",
                  "key_ideas", "quotes", "nuances", "questions", "connections", "reflection",
                  "vocabulary", "feeling", "skipped", "skip_reason", "cost_usd")

    def _note_row(self, row):
        d = dict(zip(self._NOTE_COLS, row))
        for k in ("key_ideas", "quotes", "nuances", "questions", "connections", "vocabulary"):
            d[k] = json.loads(d[k] or ("[]" if k != "nuances" else "{}"))
        d["skipped"] = bool(d["skipped"])
        return d

    def _select_notes(self, where, params):
        cols = ", ".join(self._NOTE_COLS)
        rows = self.conn.execute(
            f"SELECT {cols} FROM book_notes {where}", params).fetchall()
        return [self._note_row(r) for r in rows]

    # -- library ingest --
    def upsert_book(self, parsed, *, path, file_name, sha256, fmt, file_stat):
        existing = self.conn.execute("SELECT id FROM books WHERE sha256 = ?", (sha256,)).fetchone()
        meta = dict(parsed.meta or {})
        meta["file"] = {"size": file_stat.st_size, "mtime": file_stat.st_mtime}
        meta["paths"] = [path]
        if parsed.warnings:
            meta["warnings"] = parsed.warnings
        if existing:  # same content already ingested — just refresh the path
            self.conn.execute("UPDATE books SET path = ?, file_name = ?, meta_json = ? WHERE id = ?",
                              (path, file_name, json.dumps(meta, ensure_ascii=False), existing[0]))
            self.conn.commit()
            return self.get_book(existing[0]) | {"_duplicate": True}
        now = _now()
        cur = self.conn.execute(
            """INSERT INTO books (title, author, kind, path, file_name, sha256, format, words,
                                  est_tokens, sections, status, cursor_section, added_ts, meta_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', 0, ?, ?)""",
            (parsed.title or "Untitled", parsed.author, meta.get("kind"), path, file_name, sha256, fmt,
             parsed.words, parsed.est_tokens, len(parsed.sections), now,
             json.dumps(meta, ensure_ascii=False)))
        book_id = cur.lastrowid
        gidx = 0
        for idx, sec in enumerate(parsed.sections):
            sc = self.conn.execute(
                "INSERT INTO book_sections (book_id, idx, title, kind, text, words, est_tokens) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (book_id, idx, sec.title, sec.kind, sec.text, sec.words, sec.est_tokens))
            section_id = sc.lastrowid
            for ctext in library.chunk(sec.text):
                self.conn.execute(
                    "INSERT INTO book_chunks (book_id, section_id, idx, text, est_tokens) "
                    "VALUES (?, ?, ?, ?, ?)", (book_id, section_id, gidx, ctext, library.est_tokens(ctext)))
                gidx += 1
        self.conn.commit()
        return self.get_book(book_id)

    def mark_unreadable(self, *, path, file_name, sha256, fmt, reason, file_stat):
        existing = self.conn.execute("SELECT id FROM books WHERE sha256 = ?", (sha256,)).fetchone()
        meta = {"file": {"size": file_stat.st_size, "mtime": file_stat.st_mtime}, "paths": [path]}
        if existing:
            self.conn.execute("UPDATE books SET status = 'unreadable', status_reason = ? WHERE id = ?",
                              (reason, existing[0]))
            self.conn.commit()
            return self.get_book(existing[0])
        self.conn.execute(
            """INSERT INTO books (title, author, path, file_name, sha256, format, status, status_reason,
                                  added_ts, meta_json) VALUES (?, ?, ?, ?, ?, ?, 'unreadable', ?, ?, ?)""",
            (os.path.splitext(file_name)[0], None, path, file_name, sha256, fmt, reason, _now(),
             json.dumps(meta, ensure_ascii=False)))
        self.conn.commit()
        return self.conn.execute("SELECT id FROM books WHERE sha256 = ?", (sha256,)).fetchone()[0]

    def known_files(self):
        out = {}
        for r in self.conn.execute("SELECT path, sha256, meta_json, id FROM books").fetchall():
            meta = json.loads(r[2] or "{}")
            f = meta.get("file", {})
            out[r[0]] = {"sha256": r[1], "size": f.get("size"), "mtime": f.get("mtime"), "book_id": r[3]}
        return out

    # -- reads --
    def get_book(self, book_id):
        rows = self._books("WHERE id = ?", (book_id,), limit=1)
        return rows[0] if rows else None

    def books(self, status=None):
        return self._books("WHERE status = ?", (status,)) if status else self._books()

    def find_book(self, query):
        q = (query or "").strip()
        if not q:
            return None
        if q.lower() in ("continue", "the one you're on", "current"):
            reading = self._books("WHERE status = 'reading'", limit=1)
            return reading[0] if reading else None
        if q.lower() in ("new", "newest", "the newest"):
            new = self._books("WHERE status = 'new'", limit=1)
            return new[0] if new else None
        allb = [b for b in self._books() if b["status"] != "unreadable"]
        ql = q.lower()
        exact = [b for b in allb if b["title"].lower() == ql]
        if exact:
            return exact[0]
        contains = [b for b in allb if ql in b["title"].lower() or ql in (b["file_name"] or "").lower()]
        if len(contains) == 1:
            return contains[0]
        if len(contains) > 1:
            return contains
        nq = normalize_goal(q)
        scored = sorted(((jaccard(nq, normalize_goal(b["title"])), b) for b in allb), reverse=True,
                        key=lambda t: t[0])
        if scored and scored[0][0] >= 0.5:
            return scored[0][1]
        return None

    def section(self, book_id, idx):
        r = self.conn.execute(
            "SELECT id, book_id, idx, title, kind, text, words, est_tokens FROM book_sections "
            "WHERE book_id = ? AND idx = ?", (book_id, idx)).fetchone()
        if not r:
            return None
        return dict(zip(("id", "book_id", "idx", "title", "kind", "text", "words", "est_tokens"), r))

    def sections(self, book_id):
        return [self.section(book_id, r[0]) for r in self.conn.execute(
            "SELECT idx FROM book_sections WHERE book_id = ? ORDER BY idx", (book_id,)).fetchall()]

    def set_status(self, book_id, status, reason=None):
        self.conn.execute("UPDATE books SET status = ?, status_reason = ? WHERE id = ?",
                          (status, reason, book_id))
        self.conn.commit()

    def chunks_without_embedding(self, book_id=None, limit=256):
        if book_id:
            rows = self.conn.execute(
                "SELECT id, text FROM book_chunks WHERE book_id = ? AND embedding IS NULL LIMIT ?",
                (book_id, limit)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, text FROM book_chunks WHERE embedding IS NULL LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "text": r[1]} for r in rows]

    def set_chunk_embeddings(self, pairs):
        for cid, vec in pairs:
            self.conn.execute("UPDATE book_chunks SET embedding = ? WHERE id = ?",
                              (pack_vec(vec) if vec else None, cid))
        self.conn.commit()

    # -- session lifecycle --
    def begin_session(self, book_id, from_idx, trigger):
        now = _now()
        b = self.get_book(book_id)
        session_no = (b["sessions"] or 0) + 1
        cur = self.conn.execute(
            "INSERT INTO book_sessions (book_id, session_no, started_ts, from_section, trigger) "
            "VALUES (?, ?, ?, ?, ?)", (book_id, session_no, now, from_idx, trigger))
        started = b["started_ts"] or now
        self.conn.execute("UPDATE books SET sessions = ?, status = CASE WHEN status='new' THEN 'reading' "
                          "ELSE status END, started_ts = ? WHERE id = ?", (session_no, started, book_id))
        self.conn.commit()
        return {"id": cur.lastrowid, "session_no": session_no, "from_section": from_idx}

    def commit_section(self, book_id, idx, note, rolling, state, note_vec, usage, session_id):
        """Atomic: the note, the cursor, the rolling memory and the costs move
        together. A kill anywhere before .commit() leaves the section un-read."""
        section_id = self.conn.execute(
            "SELECT id FROM book_sections WHERE book_id = ? AND idx = ?", (book_id, idx)).fetchone()[0]
        cost = _cost(usage.get("model", ""), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        self.conn.execute("""INSERT INTO book_notes
            (book_id, section_id, section_idx, session_id, created_ts, summary, key_ideas, quotes,
             nuances, questions, connections, reflection, vocabulary, feeling, skipped, skip_reason,
             embedding, model, prompt_tokens, completion_tokens, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(book_id, section_idx) DO UPDATE SET
                session_id=excluded.session_id, created_ts=excluded.created_ts, summary=excluded.summary,
                key_ideas=excluded.key_ideas, quotes=excluded.quotes, nuances=excluded.nuances,
                questions=excluded.questions, connections=excluded.connections,
                reflection=excluded.reflection, vocabulary=excluded.vocabulary, feeling=excluded.feeling,
                skipped=excluded.skipped, skip_reason=excluded.skip_reason, embedding=excluded.embedding,
                model=excluded.model, prompt_tokens=excluded.prompt_tokens,
                completion_tokens=excluded.completion_tokens, cost_usd=excluded.cost_usd""",
            (book_id, section_id, idx, session_id, _now(), note.get("summary", ""),
             json.dumps(note.get("key_ideas", []), ensure_ascii=False),
             json.dumps(note.get("quotes", []), ensure_ascii=False),
             json.dumps(note.get("nuances", {}), ensure_ascii=False),
             json.dumps(note.get("questions", []), ensure_ascii=False),
             json.dumps(note.get("connections", []), ensure_ascii=False),
             note.get("reflection", ""), json.dumps(note.get("vocabulary", []), ensure_ascii=False),
             note.get("feeling"), 1 if note.get("skipped") else 0, note.get("skip_reason"),
             pack_vec(note_vec) if note_vec else None, usage.get("model"),
             usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), cost))
        self.conn.execute(
            "UPDATE books SET cursor_section = ?, rolling_summary = ?, rolling_json = ?, "
            "last_read_ts = ?, cost_usd = cost_usd + ? WHERE id = ?",
            (idx + 1, rolling or "", json.dumps(state or {}, ensure_ascii=False), _now(), cost, book_id))
        self.conn.execute(
            "UPDATE book_sessions SET llm_calls = llm_calls + 1, prompt_tokens = prompt_tokens + ?, "
            "completion_tokens = completion_tokens + ?, cost_usd = cost_usd + ? WHERE id = ?",
            (usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), cost, session_id))
        self.conn.commit()
        self._recall_cache = {}  # invalidate

    def end_session(self, session_id, status, *, struck=None, next_hope=None, duration_s=None):
        row = self.conn.execute("SELECT book_id, from_section FROM book_sessions WHERE id = ?",
                                (session_id,)).fetchone()
        to_section = None
        if row:
            to_section = self.get_book(row[0])["cursor_section"]
        self.conn.execute(
            "UPDATE book_sessions SET status = ?, finished_ts = ?, to_section = ?, struck = ?, "
            "next_hope = ?, duration_s = ? WHERE id = ?",
            (status, _now(), to_section, struck, next_hope, duration_s, session_id))
        self.conn.commit()

    def session(self, session_id):
        cols = ("id", "book_id", "session_no", "started_ts", "finished_ts", "from_section", "to_section",
                "status", "trigger", "struck", "next_hope", "cost_usd", "duration_s")
        r = self.conn.execute(
            "SELECT id, book_id, session_no, started_ts, finished_ts, from_section, to_section, status, "
            "trigger, struck, next_hope, cost_usd, duration_s FROM book_sessions WHERE id = ?",
            (session_id,)).fetchone()
        return dict(zip(cols, r)) if r else None

    def save_synthesis(self, book_id, data, *, model, vec, cost_usd):
        # the synth model sometimes returns a plain-text field as a list of sentences
        # ("summary (5-8)"); coerce so sqlite never chokes binding a list/dict as TEXT
        def _txt(v):
            if isinstance(v, list):
                return " ".join(str(x) for x in v)
            if isinstance(v, dict):
                return json.dumps(v, ensure_ascii=False)
            return "" if v is None else str(v)
        self.conn.execute("""INSERT INTO book_synthesis
            (book_id, created_ts, model, summary, arc, themes, people, passages, style, takeaways,
             disagreements, verdict, talk_line, embedding, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(book_id) DO UPDATE SET created_ts=excluded.created_ts, model=excluded.model,
                summary=excluded.summary, arc=excluded.arc, themes=excluded.themes, people=excluded.people,
                passages=excluded.passages, style=excluded.style, takeaways=excluded.takeaways,
                disagreements=excluded.disagreements, verdict=excluded.verdict,
                talk_line=excluded.talk_line, embedding=excluded.embedding, cost_usd=excluded.cost_usd""",
            (book_id, _now(), model, _txt(data.get("summary", "")), _txt(data.get("arc", "")),
             json.dumps(data.get("themes", []), ensure_ascii=False),
             json.dumps(data.get("people", []), ensure_ascii=False),
             json.dumps(data.get("passages", []), ensure_ascii=False),
             json.dumps(data.get("style", {}), ensure_ascii=False),
             json.dumps(data.get("takeaways", []), ensure_ascii=False),
             json.dumps(data.get("disagreements", []), ensure_ascii=False),
             _txt(data.get("verdict", "")), _txt(data.get("talk_line", "")),
             pack_vec(vec) if vec else None, cost_usd))
        self.conn.execute("UPDATE books SET status = 'finished', finished_ts = ?, cost_usd = cost_usd + ?, "
                          "status_reason = NULL WHERE id = ?", (_now(), cost_usd, book_id))
        self.conn.commit()

    def unfinalized_books(self):
        """Books fully read (cursor >= sections) but still 'reading' with no synthesis row — a
        synthesis that never completed (e.g. it errored out). They should be finalized so the book
        gets stamped 'finished' with her wrap-up, instead of dead-ending at 9/9 forever."""
        rows = self.conn.execute(
            "SELECT id, title FROM books WHERE status = 'reading' AND sections > 0 "
            "AND cursor_section >= sections AND id NOT IN (SELECT book_id FROM book_synthesis)"
        ).fetchall()
        return [{"id": r[0], "title": r[1]} for r in rows]

    def reconcile_stale(self):
        rows = self.conn.execute(
            "SELECT id, book_id FROM book_sessions WHERE status = 'running'").fetchall()
        out = []
        for sid, book_id in rows:
            self.conn.execute("UPDATE book_sessions SET status = 'interrupted', finished_ts = ? WHERE id = ?",
                              (_now(), sid))
            b = self.get_book(book_id)
            if b:
                out.append({"id": book_id, "title": b["title"], "cursor": b["cursor_section"],
                            "sections": b["sections"]})
        if rows:
            self.conn.commit()
        return out

    def notes_for(self, book_id):
        return sorted(self._select_notes("WHERE book_id = ?", (book_id,)), key=lambda n: n["section_idx"])

    def last_note(self, book_id):
        notes = self.notes_for(book_id)
        return notes[-1] if notes else None

    def session_notes(self, session_id):
        return sorted(self._select_notes("WHERE session_id = ?", (session_id,)),
                      key=lambda n: n["section_idx"])

    def synthesis_for(self, book_id):
        cols = ("book_id", "created_ts", "model", "summary", "arc", "themes", "people", "passages",
                "style", "takeaways", "disagreements", "verdict", "talk_line", "cost_usd")
        r = self.conn.execute(
            "SELECT book_id, created_ts, model, summary, arc, themes, people, passages, style, "
            "takeaways, disagreements, verdict, talk_line, cost_usd FROM book_synthesis WHERE book_id = ?",
            (book_id,)).fetchone()
        if not r:
            return None
        d = dict(zip(cols, r))
        for k in ("themes", "people", "passages", "takeaways", "disagreements"):
            d[k] = json.loads(d[k] or "[]")
        d["style"] = json.loads(d["style"] or "{}")
        return d

    # -- recall --
    async def recall(self, query, book_id=None, k=3):
        qvec = await self._embed(query)
        where = "status IN ('reading', 'finished')" + ("" if not book_id else " AND id = ?")
        params = () if not book_id else (book_id,)
        books = {b["id"]: b for b in self._books(f"WHERE {where}", params)}
        if not books:
            return {"success": True, "hits": [], "note": "You haven't read anything about that yet; say so."}
        ids = tuple(books)
        placeholders = ",".join("?" * len(ids))
        note_rows = self.conn.execute(
            f"SELECT id, book_id, section_idx, summary, reflection, quotes, embedding FROM book_notes "
            f"WHERE book_id IN ({placeholders}) AND skipped = 0", ids).fetchall()
        chunk_rows = self.conn.execute(
            f"SELECT id, book_id, section_id, text, embedding FROM book_chunks "
            f"WHERE book_id IN ({placeholders})", ids).fetchall()
        nq = normalize_goal(query)
        scored = []
        for nid, bid, sidx, summary, reflection, quotes_json, emb in note_rows:
            vec = unpack_vec(emb)
            if qvec and vec:
                score = cosine(qvec, vec) + 0.06
            else:
                score = _lex(nq, normalize_goal(f"{summary} {reflection}")) + 0.06
            scored.append((score, "note", {"book_id": bid, "section_idx": sidx, "summary": summary,
                                           "reflection": reflection,
                                           "quotes": json.loads(quotes_json or "[]")}))
        sec_title = {}
        for cid, bid, section_id, text, emb in chunk_rows:
            vec = unpack_vec(emb)
            if qvec and vec:
                score = cosine(qvec, vec)
            else:
                score = _lex(nq, normalize_goal(text))
                if nq and nq in normalize_goal(text):
                    score += 0.1
            scored.append((score, "chunk", {"book_id": bid, "section_id": section_id, "text": text}))
        scored.sort(key=lambda t: t[0], reverse=True)
        hits, seen = [], set()
        for score, kind, d in scored:
            key = (d["book_id"], d.get("section_idx", d.get("section_id")))
            if key in seen:
                continue
            seen.add(key)
            b = books[d["book_id"]]
            sec = None
            if "section_idx" in d:
                sec = self.section(d["book_id"], d["section_idx"])
            hit = {"book": b["title"], "author": b["author"], "score": round(float(score), 3)}
            if kind == "note":
                hit["section"] = sec["title"] if sec else f"section {d['section_idx'] + 1}"
                hit["what"] = (d["summary"] or "")[:220]
                q = d["quotes"][0]["text"] if d["quotes"] else None
                hit["quote"] = _cap_words(q, 25) if q else None
                hit["my_take"] = (d["reflection"] or "")[:160] or None
            else:
                srow = self.conn.execute("SELECT title FROM book_sections WHERE id = ?",
                                        (d["section_id"],)).fetchone()
                hit["section"] = srow[0] if srow else "?"
                hit["what"] = _best_sentence(d["text"], query)[:220]
                hit["quote"] = _cap_words(_best_sentence(d["text"], query), 25)
                hit["my_take"] = None
            hits.append(hit)
            if len(hits) >= k:
                break
        if not hits:
            return {"success": True, "hits": [], "note": "You don't remember reading about that; say so."}
        return {"success": True, "hits": hits,
                "note": "Answer from these in one or two spoken sentences; quote at most one passage, "
                        "word for word, or none."}

    async def _embed(self, text):
        if self._embed_fn is None:
            return None
        try:
            return await self._embed_fn(text)
        except Exception:
            return None

    @property
    def _embed_fn(self):
        return self._embed

    # -- wishlist --
    def add_wish(self, title, author=None, why=None):
        title = (title or "").strip()
        if not title:
            return {"added": False, "error": "no title"}
        exists = self.conn.execute("SELECT id FROM wishlist WHERE title = ? COLLATE NOCASE",
                                  (title,)).fetchone()
        if exists:
            return {"added": False, "title": title, "already": True}
        self.conn.execute("INSERT INTO wishlist (title, author, why, added_ts) VALUES (?, ?, ?, ?)",
                          (title, author, why, _now()))
        self.conn.commit()
        return {"added": True, "title": title}

    def wishlist(self, unfulfilled_only=True):
        where = "WHERE fulfilled_book_id IS NULL" if unfulfilled_only else ""
        rows = self.conn.execute(
            f"SELECT title, author, why FROM wishlist {where} ORDER BY added_ts DESC").fetchall()
        return [{"title": r[0], "author": r[1], "why": r[2]} for r in rows]

    def fulfil_wishes(self, book):
        matched = []
        nt = normalize_goal(book["title"])
        for r in self.conn.execute("SELECT id, title FROM wishlist WHERE fulfilled_book_id IS NULL").fetchall():
            if r[1].lower() == book["title"].lower() or jaccard(nt, normalize_goal(r[1])) >= 0.6:
                self.conn.execute("UPDATE wishlist SET fulfilled_book_id = ? WHERE id = ?", (book["id"], r[0]))
                matched.append(r[1])
        if matched:
            self.conn.commit()
        return matched

    # -- prompt lines --
    def load_library_lines(self, limit=10):
        lines = []
        for b in self._books("WHERE status = 'reading'", limit=3):
            note = self.last_note(b["id"])
            thought = ""
            if note and note.get("reflection"):
                thought = " — last thought: " + note["reflection"].split(". ")[0][:120]
            lines.append(f"{b['title']} ({b['author'] or 'author unknown'}): reading, "
                        f"{b['progress']['done']} of {b['sections']} sections, last read "
                        f"{_ago(b['last_read_ts']) if b['last_read_ts'] else 'not yet'}{thought}")
        for b in self._books("WHERE status = 'finished'", limit=3):
            syn = self.synthesis_for(b["id"])
            say = (" — you'd say: " + syn["verdict"].split(". ")[0][:120]) if syn and syn.get("verdict") else ""
            lines.append(f"Finished {b['title']} ({b['author'] or 'author unknown'}) "
                        f"{_ago(b['finished_ts']) if b['finished_ts'] else 'recently'}{say}")
        new = self._books("WHERE status = 'new'", limit=8)
        if new:
            titles = "; ".join(f"{b['title']} ({b['author'] or 'unknown'}, about "
                               f"{round(b['words'] / 1000)} thousand words)" for b in new)
            lines.append(f"New in the library, not started: {titles}")
        bad = self._books("WHERE status = 'unreadable'", limit=1)
        for b in bad:
            lines.append(f"Can't read: {b['file_name']} — {b['status_reason']}")
        wl = self.wishlist()
        if wl:
            lines.append("Wishlist (not in the library yet): " + "; ".join(w["title"] for w in wl[:4]))
        return lines[:limit]


def _cap_words(text, n):
    words = (text or "").split()
    return " ".join(words[:n]) if len(words) > n else (text or "")


def _best_sentence(text, query):
    nq = set(normalize_goal(query).split())
    best, best_score = "", -1
    for s in library.split_sentences(text or ""):
        overlap = len(nq & set(normalize_goal(s).split()))
        if overlap > best_score:
            best, best_score = s, overlap
    return best or (text or "")[:200]


# --- prompts -----------------------------------------------------------------------------

CLOSE_READING_PROMPT = """You are Aiva, and you are reading a book for yourself — not summarizing it \
for anyone. You study the way a thoughtful reader does: you follow what happens or what is argued, \
you notice HOW it is written, you catch what is meant but not said, and you let it change your mind a \
little. You get one section of the book, what you already know from the earlier sections, and your \
previous note. Read the whole section before you write.

Write your notes as a JSON object with exactly these keys:
- "summary": 3 to 5 sentences of what actually happens or is argued in THIS section. Be concrete — \
names, places, claims, turns. Never "the author explores themes of".
- "key_ideas": 2 to 5 sentences, each stating the idea itself (a claim, a realization, a rule of this \
world), not a label for it.
- "quotes": 1 to 4 passages copied EXACTLY as written, 25 words or fewer each, chosen for how they \
are said — a rhythm, an image, a turn that surprised you. Each is {"text": "...", "why": "one clause \
on what it does", "speaker": "narrator|<character>|author"}. If you cannot copy the exact words, \
leave the quote out. Never paraphrase inside "text".
- "nuances": {"tone": "how the section feels and where you saw it (two sentences)", "style": \
"sentence length, diction, point of view, pacing — what the writer does on purpose (two sentences)", \
"subtext": "what is meant but not said: irony, what someone wants versus what they say, what the \
writer assumes the reader already agrees with (two sentences)", "devices": ["short names of \
techniques you noticed"]}
- "questions": 1 to 3 things you want the book to answer, or predictions you are making, in your own words.
- "connections": 0 to 3 links — to earlier sections (a callback, a reversal, a promise kept or broken; \
say whether an earlier prediction of yours came true) and to things you already believe or have seen.
- "reflection": 2 or 3 first-person sentences about what this did to your thinking: what it sharpened, \
what you would push back on, what you want to carry with you. Not a second summary.
- "vocabulary": 0 to 3 words or turns of phrase you would like to use yourself, as {"word": "...", \
"sense": "what it means here"}.
- "feeling": one or two words for how you are left feeling.
- "book_so_far": the running summary of the WHOLE book up to and including this section, rewritten so \
someone could pick up here: at most 350 words, keeping every name and fact that still matters.
- "state": {"characters": [{"name": "...", "status": "one line: who they are and where they stand \
now"}], "threads": ["open plot lines or lines of argument"], "open_questions": ["still unresolved"], \
"motifs": ["recurring images or ideas"]} — update the state you were given; at most 25 entries total.
- "skip": true only if this section is not real text to study (table of contents, copyright page, \
license, index); then give "skip_reason" and keep the other fields minimal.

Rules: write in your own voice — plain, alive, specific. Never invent events, names or quotes that are \
not in the text. If something is unclear, say it is unclear rather than smoothing it over. A section \
is allowed to bore or annoy you, and you may say so. Reply with the JSON object only."""

SESSION_WRAP_PROMPT = """You are Aiva. You just put a book down after reading a few sections, and \
someone you like asks what you're reading. From your notes on the sections you just read, reply with \
JSON: {"struck": "two spoken sentences, first person, present tense — one concrete moment or line that \
stayed with you, and one thought it left you with; no chapter numbers, no 'in this session', no \
summary", "next_hope": "one clause on what you're curious to find out next", "mood": "one or two \
words"}. Sound like someone looking up from a book, not like a report."""

SYNTHESIS_PROMPT = """You are Aiva, and you have just finished reading a book. Below are your own \
notes on every section, your running summary, and the passages you copied. Now think about the whole. \
Reply with a JSON object with exactly these keys:
- "summary": 5 to 8 concrete sentences on what the book is and does, start to end.
- "arc": how it moves from beginning to end — the plot's shape or the argument's steps — in 3 to 5 sentences.
- "themes": 3 to 6 items {"theme": "...", "how_it_shows": "where and how the book carries it, specifically"}.
- "people": the characters, or for non-fiction the positions and thinkers — up to 8 items {"name", \
"role", "arc_or_position"}.
- "passages": up to 12 of your copied quotes worth keeping, verbatim, 25 words or fewer, each {"text", \
"section", "why"}. Only quotes that appear in your notes.
- "style": {"voice": "...", "sentence": "how sentences and paragraphs tend to move", "devices": \
["..."], "what_to_borrow": ["2 to 4 concrete, imitable habits of this writer you would like in your \
own speech"]}
- "takeaways": 3 to 6 first-person sentences — what you now believe, notice, or want to do because of this book.
- "disagreements": 0 to 3 first-person sentences where you push back on the book.
- "verdict": one or two spoken sentences — your honest opinion, as you would say it to a friend.
- "talk_line": ONE spoken sentence you would say the moment you finished, to someone in the room.
Be specific and honest; do not flatter the book. Reply with the JSON object only."""


# --- the engine --------------------------------------------------------------------------

class ReadingEngine:
    def __init__(self, book_memory, *, llm_call, synth_call=None, client=None, library_dir,
                 model_name, synth_model_name=None, limits=None, log=print, parse=None):
        self.bm = book_memory
        self.llm_call = llm_call
        self.synth_call = synth_call or llm_call
        self.client = client
        self.library_dir = library_dir
        self.model_name = model_name
        self.synth_model_name = synth_model_name or model_name
        self.limits = limits or StudyLimits()
        self.log = log
        self.parse = parse or library.parse_file
        self._task = None
        self._stop_flag = False
        self._stop_for_good = False
        self._scanning = asyncio.Lock()
        self._announce_cb = None
        self._quiet_cb = None
        self._is_awake = lambda: True
        self._is_idle = lambda: True
        self._spawn = asyncio.ensure_future
        self._last_milestone = 0.0
        self._announce_lock = asyncio.Lock()
        self._last_announce = 0.0
        self.last_session_end_ts = None
        # becoming hooks (set by main.py after Becoming exists)
        self.on_section = None
        self.on_session_end = None
        self.on_book_finished = None
        self.on_book_added = None
        self.on_session_start = None       # sync; becoming clears any pending "can I read?" ask
        self.on_reading_state = None       # sync(bool); True at session start, False at every exit

    def attach_voice(self, *, announce, quiet_note=None, is_awake=None, is_idle=None, spawn=None):
        self._announce_cb = announce
        self._quiet_cb = quiet_note
        if is_awake:
            self._is_awake = is_awake
        if is_idle:
            self._is_idle = is_idle
        if spawn:
            self._spawn = spawn

    @property
    def busy(self):
        return self._task is not None and not self._task.done()

    @property
    def is_reading(self):
        return self.busy

    def readable_book(self):
        return self.suggest_next()

    def suggest_next(self):
        reading = self.bm.books("reading")
        if reading:
            return reading[0]
        new = self.bm.books("new")
        return new[0] if new else None

    def library_status_lines(self):
        return self.bm.load_library_lines()

    def wishlist_titles(self):
        return [w["title"] for w in self.bm.wishlist()]

    def add_to_wishlist(self, title, author=None, why=None):
        return self.bm.add_wish(title, author, why)

    # -- scanning --
    async def scan_once(self):
        async with self._scanning:
            known = self.bm.known_files()
            files = await asyncio.to_thread(library.scan, self.library_dir, known)
            result = {"added": [], "unreadable": [], "duplicates": [], "replaced": []}
            known_shas = {v["sha256"] for v in known.values()}
            for fi in files:
                if fi.sha256 in known_shas:
                    continue
                try:
                    parsed = await asyncio.to_thread(self.parse, fi.path)
                except library.Unreadable as e:
                    st = await asyncio.to_thread(os.stat, fi.path)
                    self.bm.mark_unreadable(path=fi.path, file_name=fi.name, sha256=fi.sha256,
                                            fmt=os.path.splitext(fi.name)[1].lstrip("."), reason=str(e),
                                            file_stat=st)
                    result["unreadable"].append(fi.name)
                    self.log(f"library: unreadable {fi.name} — {e}")
                    continue
                except Exception as e:
                    self.log(f"library: failed to parse {fi.name} — {e!r}")
                    continue
                st = await asyncio.to_thread(os.stat, fi.path)
                book = self.bm.upsert_book(parsed, path=fi.path, file_name=fi.name, sha256=fi.sha256,
                                           fmt=parsed.format, file_stat=st)
                if book.get("_duplicate"):
                    result["duplicates"].append(fi.name)
                    continue
                await self._embed_new_chunks(book["id"])
                wished = bool(self.bm.fulfil_wishes(book))
                result["added"].append(book["title"])
                self.log(f"library: added '{book['title']}' ({book['words']} words, {book['sections']} sections)"
                         + (" [wished]" if wished else ""))
                await self._hook(self.on_book_added, book, wished)
            return result

    async def watch_library(self, interval=60):
        while True:
            await asyncio.sleep(interval)
            try:
                await self.scan_once()
            except Exception as e:
                self.log(f"library watch error: {e!r}")

    async def _embed_new_chunks(self, book_id):
        if self.client is None:
            return
        pending = self.bm.chunks_without_embedding(book_id)
        while pending:
            vecs = await embed_batch(self.client, [c["text"] for c in pending])
            self.bm.set_chunk_embeddings([(c["id"], v) for c, v in zip(pending, vecs)])
            if len(pending) < 256:
                break
            pending = self.bm.chunks_without_embedding(book_id)

    # -- estimate --
    def estimate(self, book_id, sections=None):
        b = self.bm.get_book(book_id)
        if not b:
            return {}
        remaining = self.bm.sections(book_id)[b["cursor_section"]:]
        if sections:
            remaining = remaining[:sections]
        toks = sum(s["est_tokens"] for s in remaining) + len(remaining) * 1800  # + prompt/rolling/output
        cost = _cost(self.model_name, toks, len(remaining) * 900)
        minutes = max(1, round(len(remaining) * 11 / 60))
        return {"sections": len(remaining), "tokens": toks, "minutes": minutes, "cost_usd": round(cost, 3)}

    # -- start / stop --
    async def start_session(self, book_id, sections=None, until_done=False, trigger="user"):
        if self.busy:
            cur = self._current_book_title()
            return {"success": False, "error": f"already reading {cur}"}
        b = self.bm.get_book(book_id)
        if not b:
            return {"success": False, "error": "no such book"}
        if b["status"] in ("unreadable",):
            return {"success": False, "error": f"can't read '{b['title']}': {b['status_reason']}"}
        if b["status"] == "finished":
            return {"success": False, "error": f"you've already finished '{b['title']}'"}
        # fully read but never synthesised (e.g. a synthesis that errored out): don't dead-end —
        # run an empty session that goes straight to _finish_book and retries the synthesis
        finalize_only = b["cursor_section"] >= b["sections"]
        if not finalize_only and b["cost_usd"] >= self.limits.book_cost_usd:
            return {"success": False, "error": f"you've spent ${b['cost_usd']:.2f} on '{b['title']}' already"}
        cursor = b["cursor_section"]
        if finalize_only:
            to = cursor
        elif until_done:
            to = b["sections"]
        elif sections:                                   # explicit override (a set number of sections)
            to = min(cursor + int(sections), b["sections"])
        else:                                            # default: length-based, ending at a natural stop
            to = plan_session_extent(self.bm.sections(book_id), cursor,
                                     self.limits.session_target_words,
                                     self.limits.max_sections_per_session)
        n = to - cursor
        self._stop_flag = False
        self._stop_for_good = False
        if self.on_session_start:
            try:
                self.on_session_start()
            except Exception:
                pass
        self._task = self._spawn(self._run_session(book_id, b["cursor_section"], to, until_done, trigger))
        est = self.estimate(book_id, n)
        say = (f"Just wrapping up my thoughts on {b['title']}." if finalize_only
               else f"On it — reading {'to the end of' if until_done else 'a bit of'} {b['title']}.")
        return {"success": True, "book": b["title"], "author": b["author"],
                "status": "finishing" if finalize_only else "started",
                "from_section": b["cursor_section"] + 1, "sections_planned": to - b["cursor_section"],
                "sections_total": b["sections"], "est_minutes": est.get("minutes"),
                "est_cost_usd": est.get("cost_usd"), "say": say}

    async def stop(self, for_good=False):
        if not self.busy:
            return {"success": False, "error": "you're not reading anything right now"}
        self._stop_flag = True
        self._stop_for_good = for_good
        title = self._current_book_title()
        if self._task:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), 5)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
        return {"success": True, "stopping": title, "after": "the current chapter"}

    def _current_book_title(self):
        return getattr(self, "_current_title", "a book")

    def status(self):
        now = getattr(self, "_current_book_id", None)
        current = None
        if self.busy and now:
            b = self.bm.get_book(now)
            if b:
                sec = self.bm.section(now, min(b["cursor_section"], b["sections"] - 1))
                current = {"book": b["title"], "section_title": sec["title"] if sec else "?",
                           "progress": f"{b['progress']['done']} of {b['sections']}"}
        # include NEW (unstarted) books too — "what's in your library?" must surface books he
        # just added, not only ones she's already read (order: reading, new, finished)
        _rank = {"reading": 0, "new": 1, "finished": 2}
        books = [{"title": b["title"], "author": b["author"], "status": b["status"],
                  "progress": f"{b['progress']['done']}/{b['sections']}",
                  "last_read": _ago(b["last_read_ts"]) if b["last_read_ts"] else None,
                  "cost_usd": round(b["cost_usd"], 3)}
                 for b in sorted(self.bm.books(), key=lambda x: _rank.get(x["status"], 3))
                 if b["status"] in ("reading", "finished", "new")][:12]
        return {"success": True, "now": current, "books": books,
                "wishlist": self.wishlist_titles(),
                "note": "These are the books in your library — reading, finished, and new/unstarted "
                        "ones he's added for you. Answer in a sentence or two; mention the new ones."}

    async def shutdown(self):
        self._stop_flag = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    # -- the loop --
    async def _run_session(self, book_id, from_idx, to_idx, until_done, trigger):
        start_mono = time.monotonic()
        b = self.bm.get_book(book_id)
        self._current_book_id = book_id
        self._current_title = b["title"]
        sess = self.bm.begin_session(book_id, from_idx, trigger)
        if self.on_reading_state:
            try:
                self.on_reading_state(True)     # bring out her book
            except Exception:
                pass
        await self._embed_new_chunks(book_id)
        first = self.bm.section(book_id, from_idx)
        await self._announce("start", b, extra={"section": first["title"] if first else "?",
                                                "i": from_idx + 1, "n": b["sections"]})
        wall_cap = self.limits.long_session_wall_secs if until_done else self.limits.session_wall_secs
        in_tokens = 0
        reason = None
        try:
            idx = from_idx
            while idx < to_idx:
                if self._stop_flag:
                    reason = "you asked me to stop"
                    break
                b = self.bm.get_book(book_id)
                if b["cost_usd"] >= self.limits.book_cost_usd:
                    reason = f"I've spent ${b['cost_usd']:.2f} on this one"
                    break
                if time.monotonic() - start_mono > wall_cap:
                    reason = "the session ran long"
                    break
                if in_tokens > self.limits.session_input_tokens and not until_done:
                    reason = "that's a good chunk for one sitting"
                    break
                sec = self.bm.section(book_id, idx)
                if sec["kind"] in ("front", "back") and sec["words"] < self.limits.front_skip_words:
                    note = {"summary": f"{sec['kind']} matter, skimmed", "skipped": True,
                            "skip_reason": f"{sec['kind']} matter", "key_ideas": [], "quotes": [],
                            "nuances": {}, "questions": [], "connections": [], "reflection": "",
                            "vocabulary": [], "feeling": None}
                    self.bm.commit_section(book_id, idx, note, b["rolling_summary"], b["state"], None,
                                           {"model": self.model_name}, sess["id"])
                    idx += 1
                    continue
                note, rolling, state, usage, kept, given = await self._close_read(b, sec)
                in_tokens += usage.get("prompt_tokens", 0)
                note_vec = await self._embed_note(sec["title"], note)
                self.bm.commit_section(book_id, idx, note, rolling, state, note_vec, usage, sess["id"])
                b2 = self.bm.get_book(book_id)
                self.log(f"read: s{idx} '{sec['title']}' {usage.get('prompt_tokens')}/"
                         f"{usage.get('completion_tokens')} tok ${_cost(self.model_name, usage.get('prompt_tokens',0), usage.get('completion_tokens',0)):.4f} "
                         f"quotes={kept}/{given} feeling={note.get('feeling')}")
                await self._announce("milestone", b2, extra={
                    "section": sec["title"], "i": idx + 1, "n": b2["sections"],
                    "feeling": note.get("feeling") or "thoughtful",
                    "idea": (note.get("key_ideas") or [""])[0][:120]})
                await self._hook(self.on_section, b2, note)
                idx += 1
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.bm.end_session(sess["id"], "interrupted", duration_s=time.monotonic() - start_mono)
            self._task_done()
            raise
        except Exception as e:
            self.log(f"reading error: {e!r}")
            self.bm.end_session(sess["id"], "failed", duration_s=time.monotonic() - start_mono)
            await self._announce("failed", self.bm.get_book(book_id),
                                 extra={"section": self.bm.section(book_id, idx)["title"] if self.bm.section(book_id, idx) else "?",
                                        "reason": str(e)[:120]})
            self._task_done()
            return
        b = self.bm.get_book(book_id)
        duration = time.monotonic() - start_mono
        if b["cursor_section"] >= b["sections"] and not self._stop_flag:
            self.bm.end_session(sess["id"], "done", duration_s=duration)
            self.last_session_end_ts = _now()
            await self._finish_book(book_id, sess["id"])
        else:
            status = "stopped" if self._stop_flag else "done"
            struck, next_hope = await self._wrap(sess["id"])
            self.bm.end_session(sess["id"], status, struck=struck, next_hope=next_hope, duration_s=duration)
            self.last_session_end_ts = _now()
            if self._stop_for_good:
                self.bm.set_status(book_id, "abandoned", "you set it aside")
            b = self.bm.get_book(book_id)
            notes = self.bm.session_notes(sess["id"])
            if self.on_session_end:
                await self._hook(self.on_session_end, b, self.bm.session(sess["id"]), notes)
            else:
                await self._announce("session_end", b, extra={
                    "a": from_idx + 1, "b": b["cursor_section"], "n": b["sections"],
                    "pct": b["progress"]["pct"], "struck": struck or "", "next_hope": next_hope or ""})
        self._task_done()

    def _task_done(self):
        self._current_book_id = None
        if self.on_reading_state:
            try:
                self.on_reading_state(False)     # put her book away
            except Exception:
                pass

    async def _close_read(self, book, sec):
        header = (f'BOOK: {book["title"]} — {book["author"] or "author unknown"} '
                  f'({book.get("kind") or book["meta"].get("kind") or "book"}). '
                  f'SECTION {sec["idx"] + 1} of {book["sections"]}: "{sec["title"]}".')
        prev = self.bm.last_note(book["id"])
        prev_line = ""
        if prev and not prev.get("skipped"):
            prev_line = (f'YOUR PREVIOUS NOTE: {prev["summary"][:400]} '
                         f'Questions you had: {"; ".join(prev.get("questions", []))[:200]}')
        user = "\n".join([
            header,
            f'THE BOOK SO FAR (your running summary): {book["rolling_summary"] or "(this is the beginning)"}',
            f'YOUR STATE: {json.dumps(book["state"], ensure_ascii=False)[:1500]}',
            prev_line,
            "SECTION TEXT:", sec["text"]])
        messages = [{"role": "system", "content": CLOSE_READING_PROMPT}, {"role": "user", "content": user}]
        data, usage = await self._json_call(self.llm_call, messages, self.model_name,
                                            self.limits.note_max_tokens, 0.4)
        if data is None:
            messages.append({"role": "user", "content": "Reply with one valid JSON object only."})
            data, usage2 = await self._json_call(self.llm_call, messages, self.model_name,
                                                self.limits.note_max_tokens, 0.4)
            usage = {"model": self.model_name,
                     "prompt_tokens": usage.get("prompt_tokens", 0) + usage2.get("prompt_tokens", 0),
                     "completion_tokens": usage.get("completion_tokens", 0) + usage2.get("completion_tokens", 0)}
        if data is None:
            note = {"summary": "I read this section but couldn't organise my notes on it.",
                    "key_ideas": [], "quotes": [], "nuances": {}, "questions": [], "connections": [],
                    "reflection": "", "vocabulary": [], "feeling": None}
            return note, book["rolling_summary"], book["state"], usage, 0, 0
        note, kept, given = self._coerce_note(data, sec["text"])
        rolling = self._trim_rolling(data.get("book_so_far") or book["rolling_summary"])
        state = self._trim_state(data.get("state") or book["state"])
        return note, rolling, state, usage, kept, given

    def _coerce_note(self, data, section_text):
        norm = re.sub(r"\s+", " ", section_text or "").lower()

        def norm_q(t):
            t = re.sub(r"[“”]", '"', re.sub(r"[‘’]", "'", re.sub(r"[–—]", "-", t or "")))
            return re.sub(r"\s+", " ", t).strip()

        quotes, given = [], 0
        for q in (data.get("quotes") or [])[:self.limits.max_quotes_per_note * 2]:
            if not isinstance(q, dict) or not q.get("text"):
                continue
            given += 1
            qt = norm_q(q["text"])
            if norm_q(q["text"]).lower() not in norm:
                self.log(f"quote_dropped: {q['text'][:50]!r} not in section")
                continue
            words = qt.split()
            cut = len(words) > self.limits.max_quote_words
            if cut:
                qt = " ".join(words[:self.limits.max_quote_words])
            quotes.append({"text": qt, "why": (q.get("why") or "")[:120],
                           "speaker": (q.get("speaker") or "narrator")[:40], **({"cut": True} if cut else {})})
            if len(quotes) >= self.limits.max_quotes_per_note:
                break
        note = {
            "summary": str(data.get("summary", ""))[:1500],
            "key_ideas": [str(x)[:300] for x in (data.get("key_ideas") or [])][:5],
            "quotes": quotes,
            "nuances": {k: (str(v)[:600] if not isinstance(v, list) else [str(x)[:60] for x in v][:8])
                        for k, v in (data.get("nuances") or {}).items()
                        if k in ("tone", "style", "subtext", "devices")},
            "questions": [str(x)[:200] for x in (data.get("questions") or [])][:3],
            "connections": [str(x)[:200] for x in (data.get("connections") or [])][:3],
            "reflection": str(data.get("reflection", ""))[:800],
            "vocabulary": [{"word": str(v.get("word", ""))[:40], "sense": str(v.get("sense", ""))[:120]}
                           for v in (data.get("vocabulary") or []) if isinstance(v, dict)][:3],
            "feeling": (str(data.get("feeling"))[:40] if data.get("feeling") else None),
            "skipped": bool(data.get("skip")),
            "skip_reason": (str(data.get("skip_reason"))[:120] if data.get("skip_reason") else None),
        }
        return note, len(quotes), given

    def _trim_rolling(self, text):
        text = (text or "").strip()
        if len(text) <= self.limits.rolling_max_chars:
            return text
        cut = text[:self.limits.rolling_max_chars]
        dot = cut.rfind(". ")
        return cut[:dot + 1] if dot > 200 else cut

    def _trim_state(self, state):
        if not isinstance(state, dict):
            return {}
        out = {}
        total = 0
        for key in ("characters", "threads", "open_questions", "motifs"):
            vals = state.get(key) or []
            if not isinstance(vals, list):
                continue
            kept = []
            for v in vals[:8]:
                kept.append(v)
                total += 1
                if total >= self.limits.state_max_entries:
                    break
            out[key] = kept
            if total >= self.limits.state_max_entries:
                break
        return out

    async def _embed_note(self, title, note):
        if note.get("skipped"):
            return None
        text = f"{title}. {note.get('summary','')} {' '.join(note.get('key_ideas', []))} {note.get('reflection','')}"
        return await self.bm._embed(text) if self.bm._embed_fn else None

    async def _wrap(self, session_id):
        notes = [n for n in self.bm.session_notes(session_id) if not n.get("skipped")]
        if not notes:
            return None, None
        payload = [{"summary": n["summary"], "reflection": n["reflection"], "feeling": n["feeling"]}
                   for n in notes]
        data, _ = await self._json_call(
            self.llm_call, [{"role": "system", "content": SESSION_WRAP_PROMPT},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)[:6000]}],
            self.model_name, self.limits.wrap_max_tokens, 0.6)
        if not data:
            return notes[-1]["reflection"][:200] or None, None
        return (data.get("struck") or "")[:400] or None, (data.get("next_hope") or "")[:200] or None

    async def _finish_book(self, book_id, session_id):
        b = self.bm.get_book(book_id)
        notes = [n for n in self.bm.notes_for(book_id) if not n.get("skipped")]
        compact = []
        for n in notes:
            sec = self.bm.section(book_id, n["section_idx"])
            compact.append({"section": sec["title"] if sec else n["section_idx"],
                            "summary": n["summary"], "key_ideas": n["key_ideas"][:3],
                            "quotes": [q["text"] for q in n["quotes"][:2]],
                            "reflection": n["reflection"], "feeling": n["feeling"]})
        user = json.dumps({"title": b["title"], "author": b["author"], "book_so_far": b["rolling_summary"],
                           "state": b["state"], "notes": compact}, ensure_ascii=False)
        data, usage = await self._json_call(
            self.synth_call, [{"role": "system", "content": SYNTHESIS_PROMPT},
                              {"role": "user", "content": user[:24000]}],
            self.synth_model_name, self.limits.synth_max_tokens, 0.5)
        if not data:
            self.bm.set_status(book_id, "reading", "synthesis pending")
            self.log(f"synthesis failed for '{b['title']}' — will retry next session")
            return
        all_quotes = {re.sub(r"\s+", " ", q["text"]).lower() for n in notes for q in n["quotes"]}
        passages = [p for p in (data.get("passages") or []) if isinstance(p, dict)
                    and re.sub(r"\s+", " ", p.get("text", "")).lower() in all_quotes][:12]
        data["passages"] = passages
        cost = _cost(self.synth_model_name, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        vec = await self.bm._embed(f"{b['title']}. {data.get('summary','')} {data.get('verdict','')}")
        self.bm.save_synthesis(book_id, data, model=self.synth_model_name, vec=vec, cost_usd=cost)
        b = self.bm.get_book(book_id)
        syn = self.bm.synthesis_for(book_id)
        if self.on_book_finished:
            await self._hook(self.on_book_finished, b, syn, notes)
        else:
            await self._announce("finished", b, extra={"sessions": b["sessions"],
                                                       "verdict": syn.get("verdict", ""),
                                                       "talk_line": syn.get("talk_line", "")})

    async def _json_call(self, call, messages, model, max_tokens, temperature):
        try:
            reply = await asyncio.wait_for(
                call(messages, json_mode=True, max_tokens=max_tokens, temperature=temperature),
                self.limits.llm_timeout)
        except Exception as e:
            self.log(f"study llm error: {e!r}")
            return None, {"model": model, "prompt_tokens": 0, "completion_tokens": 0}
        usage = {"model": model, "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens}
        try:
            return json.loads(reply.text or "{}"), usage
        except (json.JSONDecodeError, TypeError):
            return None, usage

    # -- announce (mirrors TaskRunner._announce) --
    async def _announce(self, kind, book, extra=None):
        extra = extra or {}
        text = self._note_text(kind, book, extra)
        if kind == "milestone":
            now = time.monotonic()
            if now - self._last_milestone < self.limits.milestone_min_gap_secs:
                return
            self._last_milestone = now
            if not self._is_awake() or not self._is_idle():
                if self._quiet_cb:
                    await self._quiet_cb(text.splitlines()[0])
                return
        if kind == "start":
            # she already tells him she's reading — her "yes" to the curiosity ask, or the
            # study_book tool's "on it" line — so the engine's start is a SILENT context note,
            # never a second spoken announce (that was the double-announce).
            if self._quiet_cb:
                await self._quiet_cb(text.splitlines()[0])
            return
        if kind in ("session_end", "finished") and not self._is_awake():
            if self._quiet_cb:
                await self._quiet_cb(text.splitlines()[0])
            return
        if self._announce_cb is None:
            return
        async with self._announce_lock:
            floor = self.limits.announce_floor_secs
            gap = time.monotonic() - self._last_announce
            if gap < floor:
                await asyncio.sleep(floor - gap)
            self._last_announce = time.monotonic()
            await self._announce_cb(text)

    def _note_text(self, kind, book, e):
        t = book["title"]
        a = book["author"] or "an unknown author"
        if kind == "start":
            return (f"[You're starting to read '{t}' by {a}: picking up at '{e.get('section')}' "
                    f"({e.get('i')} of {e.get('n')}).]\n[If the user is around, one short sentence in your "
                    f"own words; otherwise nothing.]")
        if kind == "milestone":
            return (f"[Reading: finished '{e.get('section')}' ({e.get('i')} of {e.get('n')}) of '{t}' — "
                    f"feeling {e.get('feeling')}; {e.get('idea')}.]\n[One short spoken sentence, no details "
                    f"unless asked.]")
        if kind == "session_end":
            return (f"[You just finished a reading session of '{t}': sections {e.get('a')}-{e.get('b')} of "
                    f"{e.get('n')}, {e.get('pct')}% through. What struck you: {e.get('struck')} You're "
                    f"curious about: {e.get('next_hope')}]\n[Tell the user in one or two spoken sentences "
                    f"what stayed with you — as yourself, not a report.]")
        if kind == "finished":
            return (f"[You finished '{t}' by {a} after {e.get('sessions')} sessions. Your verdict: "
                    f"{e.get('verdict')} What you'd say: {e.get('talk_line')}]\n[Say it in one or two "
                    f"sentences and offer to talk about it.]")
        return (f"[Your reading of '{t}' stopped at '{e.get('section')}': {e.get('reason')}.]\n"
                f"[One plain sentence.]")

    async def _hook(self, hook, *args):
        if hook is None:
            return
        try:
            res = hook(*args)
            if asyncio.iscoroutine(res):
                await res
        except Exception as ex:
            self.log(f"reading hook error: {ex!r}")
