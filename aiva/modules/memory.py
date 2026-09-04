"""Aiva's memory: session transcripts + durable facts about the user.

Three SQLite tables:
- messages:       rolling conversation transcripts, tagged by session
- facts:          LLM-extracted durable facts, injected into the system prompt
                  at startup so Aiva remembers the user across restarts
- session_recaps: short first-person "what happened" notes, one per session,
                  generated lazily at the NEXT boot from the saved transcript
                  (so hard kills still get a recap) and injected as episodic
                  memory — things that already happened and are not news

The system prompt is never stored here (the old design re-inserted it on
every launch and polluted the history).
"""

import glob
import json
import os
import sqlite3
import time
from datetime import datetime

from modules.utilities import llm_kwargs

FACT_EXTRACTION_PROMPT = """You maintain the long-term memory of Aiva, a voice assistant. \
From the conversation transcript, extract durable things worth remembering across sessions:
- Facts about the user: their name, pets, family, work, projects, recurring habits.
- STANDING INSTRUCTIONS the user gives about how Aiva should behave or do things — phrases like \
"always...", "from now on...", "whenever you...", "I prefer...", "don't ever...". These are \
important: capture them as durable instructions even though they are about Aiva's behavior.

The transcript comes from speech recognition and CONTAINS MISHEARINGS. Be skeptical: \
only extract the user's name if they explicitly introduced themselves ("my name is..."); \
words resembling "Aiva", "Eva", "Ava" or "Geneva" are almost always the assistant's own \
name misheard, never the user's. When in doubt about a fact, extract nothing — but DO err \
toward capturing explicit standing instructions.

Do NOT extract: one-off requests ("open notepad now"), small talk, trivia about Aiva, \
items already in the known list, or REWORDINGS of known items — a fact is a duplicate if it \
means the same thing even in different words, so extract nothing for it (unless the \
conversation contradicts a known item — then restate the corrected version).

Reply with JSON: {"facts": [{"category": "instruction|preference|person|work|project|other", "fact": "..."}]}
Each fact must be one short self-contained sentence, phrased so it makes sense with no context \
(e.g. "Always bring terminals to the front after opening them."). Reply {"facts": []} if nothing qualifies."""


RECAP_PROMPT = """You write short first-person session notes for Aiva, a voice assistant, \
so that after a restart she knows what already happened. From the transcript, write 1-4 short \
past-tense sentences ("I ...") covering only NOTABLE events: things Aiva did or told the user \
(adopted or named something, finished or promised a task, changed a setting), decisions made \
together, and anything left unfinished. Skip greetings, small talk and routine commands \
(opening terminals, moving windows) unless they were the point of the session. The transcript \
comes from speech recognition and may contain mishearings. Reply with just the notes as plain \
text, no preamble — or the single word SKIP if nothing notable happened."""


class Memory:
    def __init__(self, db_file="memory.db"):
        self.db_file = os.path.abspath(db_file)
        self.conn = sqlite3.connect(self.db_file)
        # WAL so the live dashboard (a separate read-only process) never blocks her writes,
        # nor she it; synchronous=NORMAL is the safe, faster pairing under WAL.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts DATETIME NOT NULL
            )""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL DEFAULT 'other',
                fact TEXT NOT NULL UNIQUE,
                created_ts DATETIME NOT NULL,
                last_confirmed_ts DATETIME NOT NULL
            )""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS session_recaps (
                session_id TEXT PRIMARY KEY,
                recap TEXT NOT NULL,
                created_ts DATETIME NOT NULL
            )""")
        self.conn.commit()
        self._saved_count = 0
        # her whole self now lives in this file (memory, becoming, mood, diary, desires) —
        # keep spaced, rotating snapshots so a bad write or migration can't erase who she is
        self._backup_on_boot()

    def _backup_on_boot(self, keep=10, min_gap_h=6):
        """VACUUM INTO a timestamped copy under aiva/backups/. Skips if a recent backup already
        covers this boot (so frequent restart-42s don't churn), keeps the newest `keep`. Backup
        must never take her down — every failure is swallowed."""
        try:
            if not os.path.exists(self.db_file) or os.path.getsize(self.db_file) < 4096:
                return  # nothing worth keeping yet
            bdir = os.path.join(os.path.dirname(self.db_file), "backups")
            os.makedirs(bdir, exist_ok=True)
            existing = sorted(glob.glob(os.path.join(bdir, "memory-*.db")))
            if existing:
                newest = max(os.path.getmtime(p) for p in existing)
                if (time.time() - newest) < min_gap_h * 3600:
                    return  # a recent snapshot already covers this
            target = os.path.join(bdir, f"memory-{datetime.now():%Y%m%d-%H%M%S}.db")
            if os.path.exists(target):
                return
            self.conn.execute("VACUUM INTO ?", (target,))
            for old in sorted(glob.glob(os.path.join(bdir, "memory-*.db")))[:-keep]:
                try:
                    os.remove(old)
                except OSError:
                    pass
            print(f"Memory: backed up to {os.path.basename(target)} "
                  f"({os.path.getsize(target) // 1024} KB)")
        except (sqlite3.Error, OSError) as e:
            print(f"Memory: backup skipped ({e})")

    # --- facts ---------------------------------------------------------------

    def load_facts(self, limit=200):
        """All known facts, oldest first (capped — hobby scale)."""
        rows = self.conn.execute(
            "SELECT fact FROM facts ORDER BY created_ts LIMIT ?", (limit,)
        ).fetchall()
        return [r[0] for r in rows]

    def upsert_facts(self, facts):
        """Insert new facts; refresh the timestamp of ones we already know."""
        now = datetime.now().isoformat()
        added = 0
        for item in facts:
            fact = (item.get("fact") or "").strip()
            if not fact:
                continue
            cur = self.conn.execute(
                """INSERT INTO facts (category, fact, created_ts, last_confirmed_ts)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(fact) DO UPDATE SET last_confirmed_ts = excluded.last_confirmed_ts""",
                (item.get("category", "other"), fact, now, now),
            )
            added += cur.rowcount
        self.conn.commit()
        return added

    def remember(self, fact, category="instruction"):
        """Persist ONE durable fact/instruction immediately (deduped).

        Used by the `remember` tool the moment the user asks Aiva to remember
        something — so it survives even a hard kill, unlike end-of-session
        extraction which only runs on a clean shutdown."""
        fact = (fact or "").strip()
        if not fact:
            return 0
        existed = self.conn.execute(
            "SELECT 1 FROM facts WHERE fact = ?", (fact,)).fetchone()
        self.upsert_facts([{"category": category, "fact": fact}])
        return 0 if existed else 1  # 1 only when genuinely new

    async def extract_facts(self, openai_client, model, messages):
        """One cheap LLM call: transcript -> new durable facts -> upsert."""
        transcript = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        )
        if len(transcript) < 80:  # nothing meaningful happened
            return 0

        known = self.load_facts()
        known_block = "\n".join(f"- {f}" for f in known) if known else "(none)"
        response = await openai_client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": FACT_EXTRACTION_PROMPT},
                {"role": "user",
                 "content": f"Known facts:\n{known_block}\n\nConversation:\n{transcript[-8000:]}"},
            ],
            **llm_kwargs(model, max_tokens=500, temperature=0.0),
        )
        try:
            facts = json.loads(response.choices[0].message.content).get("facts", [])
        except (json.JSONDecodeError, AttributeError):
            return 0
        return self.upsert_facts(facts)

    # --- session recaps (episodic memory) --------------------------------------

    async def recap_last_session(self, openai_client, model):
        """One cheap LLM call at boot: last session's transcript -> 1-4 short
        first-person notes. Runs lazily at boot rather than at shutdown so hard
        kills still get a recap from the 60s transcript flushes. Only the newest
        un-recapped session is processed (one call max per boot — older sessions
        simply stay un-recapped). Trivial sessions store '' so they aren't retried."""
        row = self.conn.execute("""
            SELECT m.session_id FROM messages m
            LEFT JOIN session_recaps r ON r.session_id = m.session_id
            WHERE r.session_id IS NULL
            GROUP BY m.session_id ORDER BY MAX(m.ts) DESC LIMIT 1""").fetchone()
        if not row:
            return None
        sid = row[0]
        msgs = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (sid,)).fetchall()
        transcript = "\n".join(f"{r}: {c}" for r, c in msgs)
        recap = ""
        if len(transcript) >= 80:  # anything shorter was a blip, not a session
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RECAP_PROMPT},
                    {"role": "user", "content": transcript[-8000:]},
                ],
                **llm_kwargs(model, max_tokens=200, temperature=0.0),
            )
            recap = (response.choices[0].message.content or "").strip()
            if recap.upper() == "SKIP":
                recap = ""
        self.conn.execute(
            "INSERT OR REPLACE INTO session_recaps (session_id, recap, created_ts) VALUES (?, ?, ?)",
            (sid, recap, datetime.now().isoformat()))
        self.conn.commit()
        return recap or None

    def load_recaps(self, limit=2):
        """Most recent non-empty recaps, oldest first, dated from the session id."""
        rows = self.conn.execute(
            "SELECT session_id, recap FROM session_recaps WHERE recap != '' "
            "ORDER BY session_id DESC LIMIT ?", (limit,)).fetchall()
        return [f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}: {recap}"
                for sid, recap in reversed(rows)]

    # --- transcripts -----------------------------------------------------------

    def save_new_messages(self, session_id, messages):
        """Persist user/assistant messages not yet saved this session.

        Idempotent: tracks how many of `messages` were already written, so it
        can be called periodically and again at shutdown.
        """
        speakable = [
            m for m in messages
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        ]
        fresh = speakable[self._saved_count:]
        now = datetime.now().isoformat()
        self.conn.executemany(
            "INSERT INTO messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            [(session_id, m["role"], m["content"], now) for m in fresh],
        )
        self.conn.commit()
        self._saved_count = len(speakable)
        return len(fresh)

    def forget_saved(self, n):
        """The live context dropped `n` already-saved user/assistant messages
        (context trimming) — keep save_new_messages' offset in step."""
        self._saved_count = max(0, self._saved_count - int(n))

    def session_messages(self, session_id):
        """Everything said this session, from the database — survives context
        trimming, so end-of-session fact extraction still sees the whole talk."""
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,)).fetchall()
        return [{"role": r, "content": c} for r, c in rows]

    def close(self):
        # fold the WAL back into the main file and drop the -wal/-shm sidecars, so when she's
        # off the dashboard reads a plain, current file (read-only WAL access needs a live writer)
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        self.conn.close()
