"""Aiva's memory: session transcripts + durable facts about the user.

Two SQLite tables:
- messages: rolling conversation transcripts, tagged by session
- facts:    LLM-extracted durable facts, injected into the system prompt
            at startup so Aiva remembers the user across restarts

The system prompt is never stored here (the old design re-inserted it on
every launch and polluted the history).
"""

import json
import sqlite3
from datetime import datetime

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

Do NOT extract: one-off requests ("open notepad now"), small talk, trivia about Aiva, or \
items already in the known list (unless the conversation contradicts them — then restate the \
corrected version).

Reply with JSON: {"facts": [{"category": "instruction|preference|person|work|project|other", "fact": "..."}]}
Each fact must be one short self-contained sentence, phrased so it makes sense with no context \
(e.g. "Always bring terminals to the front after opening them."). Reply {"facts": []} if nothing qualifies."""


class Memory:
    def __init__(self, db_file="memory.db"):
        self.conn = sqlite3.connect(db_file)
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
        self.conn.commit()
        self._saved_count = 0

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
            temperature=0.0,
            max_tokens=500,
        )
        try:
            facts = json.loads(response.choices[0].message.content).get("facts", [])
        except (json.JSONDecodeError, AttributeError):
            return 0
        return self.upsert_facts(facts)

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

    def close(self):
        self.conn.close()
