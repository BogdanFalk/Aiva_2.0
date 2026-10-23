"""Aiva's memory: session transcripts + durable facts about the user.

Three SQLite tables:
- messages:       rolling conversation transcripts, tagged by session
- facts:          LLM-extracted durable facts, injected into the system prompt
                  at startup so Aiva remembers the user across restarts
- session_recaps: short first-person "what happened" notes, one per session,
                  generated lazily at the NEXT boot from the saved transcript
                  (so hard kills still get a recap) and injected as episodic
                  memory — things that already happened and are not news
- store_meta:     which character this db belongs to and which embedding model
                  its vectors came from (guards against booting on the wrong
                  db or silently poisoning recall after a provider swap)

One db per character: Aiva's memory.db is untouched; a sister gets her own file
with the same schema. The prompts are templates rendered with her name once in
__init__; the unsuffixed constants are Aiva's rendering for anything importing them.

The system prompt is never stored here (the old design re-inserted it on
every launch and polluted the history).
"""

import glob
import json
import os
import sqlite3
import time
from datetime import datetime

from modules.persona import named
from modules.utilities import llm_kwargs
from modules.voice_tags import strip_tags

# {name}: the character; {name_variants}: how STT mishears HER name (a quoted list);
# {sister_note}: " (her sister is Saga)" or empty.
FACT_EXTRACTION_PROMPT_T = """You maintain the long-term memory of {name}, a voice assistant. \
From the conversation transcript, extract durable things worth remembering across sessions:
- Facts about the user: their name, pets, family, work, projects, recurring habits.
- STANDING INSTRUCTIONS the user gives about how {name} should behave or do things — phrases like \
"always...", "from now on...", "whenever you...", "I prefer...", "don't ever...". These are \
important: capture them as durable instructions even though they are about {name}'s behavior.

The transcript comes from speech recognition and CONTAINS MISHEARINGS. Be skeptical: \
only extract the user's name if they explicitly introduced themselves ("my name is..."); \
words resembling {name_variants} are almost always the assistant's own \
name misheard, never the user's. When in doubt about a fact, extract nothing — but DO err \
toward capturing explicit standing instructions.

Bracketed lines of the form [<Sister> said to him: "..."] or [<Sister> said to you: "..."] were \
spoken by {name}'s sister{sister_note}, another AI living on this PC — never by the user. Never \
attribute her words to the user. Durable things about HER (what she can do, has done or made, what \
she likes) go in category "sister"; skip her small talk.

Do NOT extract: one-off requests ("open notepad now"), small talk, trivia about {name}, \
items already in the known list, or REWORDINGS of known items — a fact is a duplicate if it \
means the same thing even in different words, so extract nothing for it.

When the conversation shows a known item is NO LONGER TRUE (he changed jobs, moved, renamed a \
pet, dropped a standing instruction, a plan changed), extract the corrected version and set \
"replaces" to that known item copied exactly, so the old one stops being remembered as true. \
Only do this when he said so or it is plainly implied; never on a guess.

Reply with JSON: {"facts": [{"category": "instruction|preference|person|work|project|sister|other", "fact": "...", "replaces": "<exact known item, only when it is no longer true>"}]}
Each fact must be one short self-contained sentence, phrased so it makes sense with no context \
(e.g. "Always bring terminals to the front after opening them."). Reply {"facts": []} if nothing qualifies."""


RECAP_PROMPT_T = """You write short first-person session notes for {name}, a voice assistant, \
so that after a restart she knows what already happened. From the transcript, write 1-4 short \
past-tense sentences ("I ...") covering only NOTABLE events: things {name} did or told the user \
(adopted or named something, finished or promised a task, changed a setting), decisions made \
together, and anything left unfinished. Skip greetings, small talk and routine commands \
(opening terminals, moving windows) unless they were the point of the session. Bracketed lines \
like [<Sister> said to him: "..."] are {name}'s sister speaking, not the user. The transcript \
comes from speech recognition and may contain mishearings. Reply with just the notes as plain \
text, no preamble — or the single word SKIP if nothing notable happened."""


# today's hand-picked list for Aiva when no variants are given
AIVA_MISHEARD = '"Aiva", "Eva", "Ava" or "Geneva"'

FACT_EXTRACTION_PROMPT = named(FACT_EXTRACTION_PROMPT_T, "Aiva", name_variants=AIVA_MISHEARD, sister_note="")
RECAP_PROMPT = named(RECAP_PROMPT_T, "Aiva")


def _fact_key(text):
    """How two spellings of one fact compare: case, spacing and end punctuation don't count."""
    return " ".join((text or "").lower().split()).rstrip(".!?;: ")


def misheard_list(name, variants=()):
    """'"Saga", "Sagah", "Sarga" or "Sega"' — how STT mishears her name, for the fact prompt.
    Single-word variants only (multi-word ones like "hey va" aren't names), title-cased, capped."""
    seen, out = {name.lower()}, [f'"{name}"']
    for v in variants:
        v = (v or "").strip().lower()
        if not v or " " in v or v in seen:
            continue
        seen.add(v)
        out.append(f'"{v.title()}"')
        if len(out) >= 8:
            break
    if len(out) == 1:
        return out[0]
    return ", ".join(out[:-1]) + " or " + out[-1]


class Memory:
    def __init__(self, db_file="memory.db", *, backup_stem=None, name="Aiva", name_variants=(),
                 sisters=()):
        self.name = name
        self.sisters = tuple(sisters)
        self.name_variants = tuple(name_variants)
        variants_text = (misheard_list(name, self.name_variants) if self.name_variants
                         else (AIVA_MISHEARD if name == "Aiva" else f'"{name}"'))
        sister_note = f" (her sister is {' and '.join(self.sisters)})" if self.sisters else ""
        self.fact_prompt = named(FACT_EXTRACTION_PROMPT_T, name, name_variants=variants_text,
                                 sister_note=sister_note)
        self.recap_prompt = named(RECAP_PROMPT_T, name)

        self.in_memory = db_file == ":memory:"
        self.db_file = db_file if self.in_memory else os.path.abspath(db_file)
        if backup_stem:
            self.backup_stem = backup_stem
        elif self.in_memory:
            self.backup_stem = "memory"
        else:
            self.backup_stem = os.path.splitext(os.path.basename(self.db_file))[0] or "memory"
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
        # facts that stopped being true: moved here (with what replaced them) rather than deleted,
        # so `facts` stays "true now" for every reader and nothing he said is lost
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts_superseded (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL DEFAULT 'other',
                fact TEXT NOT NULL,
                replaced_by TEXT NOT NULL,
                created_ts DATETIME NOT NULL,
                superseded_ts DATETIME NOT NULL
            )""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS session_recaps (
                session_id TEXT PRIMARY KEY,
                recap TEXT NOT NULL,
                created_ts DATETIME NOT NULL
            )""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS store_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_ts TEXT NOT NULL
            )""")
        self.conn.commit()
        self._saved_count = 0
        self.superseded = []          # (old, new) facts replaced this session, for the log
        # her whole self now lives in this file (memory, becoming, mood, diary, desires) —
        # keep spaced, rotating snapshots so a bad write or migration can't erase who she is
        if not self.in_memory:
            self._backup_on_boot()

    def _backup_on_boot(self, keep=10, min_gap_h=6):
        """VACUUM INTO a timestamped copy under aiva/backups/. Skips if a recent backup already
        covers this boot (so frequent restart-42s don't churn), keeps the newest `keep`. Backup
        must never take her down — every failure is swallowed. Snapshots are named by
        `backup_stem` so two characters' backups never prune each other."""
        try:
            if not os.path.exists(self.db_file) or os.path.getsize(self.db_file) < 4096:
                return  # nothing worth keeping yet
            bdir = os.path.join(os.path.dirname(self.db_file), "backups")
            os.makedirs(bdir, exist_ok=True)
            pattern = os.path.join(bdir, f"{self.backup_stem}-*.db")
            existing = sorted(glob.glob(pattern))
            if existing:
                newest = max(os.path.getmtime(p) for p in existing)
                if (time.time() - newest) < min_gap_h * 3600:
                    return  # a recent snapshot already covers this
            target = os.path.join(bdir, f"{self.backup_stem}-{datetime.now():%Y%m%d-%H%M%S}.db")
            if os.path.exists(target):
                return
            self.conn.execute("VACUUM INTO ?", (target,))
            for old in sorted(glob.glob(pattern))[:-keep]:
                try:
                    os.remove(old)
                except OSError:
                    pass
            print(f"Memory: backed up to {os.path.basename(target)} "
                  f"({os.path.getsize(target) // 1024} KB)")
        except (sqlite3.Error, OSError) as e:
            print(f"Memory: backup skipped ({e})")

    # --- store meta: whose db is this, which embeddings are in it -------------------

    def meta_get(self, key, default=None):
        row = self.conn.execute("SELECT value FROM store_meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def meta_set(self, key, value):
        self.conn.execute(
            "INSERT INTO store_meta (key, value, updated_ts) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_ts = excluded.updated_ts",
            (key, str(value), datetime.now().isoformat()))
        self.conn.commit()

    def check_identity(self, slug):
        """Stamp a fresh db with its character; refuse to run a character on another one's db."""
        current = self.meta_get("character")
        if current is None:
            self.meta_set("character", slug)
            return slug
        if current != slug:
            raise RuntimeError(
                f"Memory: {self.db_file} belongs to '{current}', not '{slug}' — refusing to boot "
                f"{slug} on another character's db (point her at her own db file)")
        return current

    def _embedding_tables(self):
        """Every table with a nullable `embedding` column — discovered live from sqlite_master, so
        stores created later on this same connection are covered too. NOT NULL embedding columns
        (speaker voiceprints, which carry their own model_id) are left alone."""
        names = [r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()]
        out = []
        for t in names:
            for _cid, col, _type, notnull, _dflt, _pk in self.conn.execute(
                    f'PRAGMA table_info("{t}")').fetchall():
                if col == "embedding" and not notnull:
                    out.append(t)
                    break
        return out

    def check_embeddings(self, model, dim):
        """Stamp the embedding model/dim this db's vectors came from. On a mismatch (a provider or
        model swap) NULL every stored embedding — every store falls back to lexical matching and
        re-embeds as it goes — and re-stamp. Returns the tables cleared ([] when nothing changed)."""
        dim_s = "" if dim is None else str(int(dim))
        cur_model, cur_dim = self.meta_get("embed_model"), self.meta_get("embed_dim")
        if cur_model is None and cur_dim is None:
            self.meta_set("embed_model", model)
            self.meta_set("embed_dim", dim_s)
            return []
        if cur_model == model and (cur_dim or "") == dim_s:
            return []
        tables = self._embedding_tables()
        print(f"Memory: EMBEDDING MODEL CHANGED ({cur_model}/{cur_dim or '?'} -> {model}/{dim_s or '?'}) "
              f"— clearing stored vectors in {len(tables)} tables ({', '.join(tables)}); recall falls "
              f"back to lexical matching until they are re-embedded")
        for t in tables:
            self.conn.execute(f'UPDATE "{t}" SET embedding = NULL')
        self.meta_set("embed_model", model)
        self.meta_set("embed_dim", dim_s)
        return tables

    def is_fresh(self):
        """True until she has ever talked or learned anything (no messages, no facts) — her first
        meeting with whoever boots her."""
        has_msg = self.conn.execute("SELECT 1 FROM messages LIMIT 1").fetchone()
        has_fact = self.conn.execute("SELECT 1 FROM facts LIMIT 1").fetchone()
        return has_msg is None and has_fact is None

    # --- facts ---------------------------------------------------------------

    def load_facts(self, limit=200):
        """All known facts about the user, oldest first (capped — hobby scale). What she knows
        about her SISTER (category 'sister') is rendered by the kin block, not here."""
        rows = self.conn.execute(
            "SELECT fact FROM facts WHERE COALESCE(category, '') != 'sister' "
            "ORDER BY created_ts LIMIT ?", (limit,)
        ).fetchall()
        return [r[0] for r in rows]

    def upsert_facts(self, facts):
        """Insert new facts; refresh the timestamp of ones we already know. An item with
        `replaces` (a known fact that is no longer true) moves that fact to `facts_superseded`."""
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
            old = (item.get("replaces") or "").strip() if isinstance(item.get("replaces"), str) else ""
            if old:
                self._supersede(old, fact, now)
        self.conn.commit()
        return added

    def _supersede(self, old, new, now):
        """Move the known fact matching `old` (exactly, or ignoring case/spacing/end punctuation)
        to facts_superseded. A `replaces` that names nothing known, or the new fact itself, is ignored."""
        key = _fact_key(old)
        if not key or key == _fact_key(new):
            return False
        for fid, category, fact, created in self.conn.execute(
                "SELECT id, category, fact, created_ts FROM facts").fetchall():
            if _fact_key(fact) == key:
                self.conn.execute(
                    "INSERT INTO facts_superseded (category, fact, replaced_by, created_ts, superseded_ts) "
                    "VALUES (?, ?, ?, ?, ?)", (category or "other", fact, new, created, now))
                self.conn.execute("DELETE FROM facts WHERE id = ?", (fid,))
                self.superseded.append((fact, new))
                print(f"Memory: no longer true: {fact!r} -> {new!r}")
                return True
        return False

    def load_superseded(self, limit=50):
        """What she used to know and what replaced it, newest first."""
        return self.conn.execute(
            "SELECT fact, replaced_by, superseded_ts FROM facts_superseded ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()

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

    # what she may retire on her own evidence: things she noted about projects and the PC — never
    # his instructions, preferences or what she knows about people (those only change by his word)
    RETIRABLE = ("project", "work", "other", "learned")

    def retire_fact(self, fact):
        """Delete ONE fact, matched exactly, that she has proved wrong (task second thoughts:
        "the witness lives in the Miso renderer" vs a report showing there is no such thing).
        Returns its category when retired, else None. The boot backups keep the old row."""
        row = self.conn.execute("SELECT category FROM facts WHERE fact = ?", ((fact or "").strip(),)).fetchone()
        if row is None or (row[0] or "other") not in self.RETIRABLE:
            return None
        self.conn.execute("DELETE FROM facts WHERE fact = ?", (fact.strip(),))
        self.conn.commit()
        return row[0] or "other"

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
                {"role": "system", "content": self.fact_prompt},
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
                    {"role": "system", "content": self.recap_prompt},
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
        # her voice cues ([laughs]) are delivery, not words: the stored transcript is what she said
        self.conn.executemany(
            "INSERT INTO messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            [(session_id, m["role"], strip_tags(m["content"]) if m["role"] == "assistant" else m["content"],
              now) for m in fresh],
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
