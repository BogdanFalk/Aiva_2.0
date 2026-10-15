"""How reading changes Aiva — lessons she applies, traits she grows, turns of
phrase she picks up, and the nerve to ask for a book when it's quiet.

Fed by the reading engine's hooks (`on_book_finished`, `on_session_end`,
`on_book_added`). It NEVER overwrites her: a fixed baseline (persona.py's
PERSONALITY) is the floor; a book can deepen or add a facet, never contradict
it. Traits are capped, fade if unused, and can be vetoed by voice ("drop
that") — permanently. What she keeps rides into the system prompt as small
headed blocks that all say, in her voice, "this is part of you now, don't
announce it".

Same rules as TaskMemory: sqlite on the event-loop thread; embeddings/LLM via
injected callables; pure decision functions at the bottom.
"""

import asyncio
import json
import re
import time
from datetime import datetime

from modules.persona import named
from modules.task_memory import (LEX_SAME, SAME_TASK, _ago, _now, cosine, jaccard,
                                 normalize_goal, pack_vec, unpack_vec)

MAX_ACTIVE_TRAITS = 12
MAX_ACTIVE_VOICE = 8
MAX_VOICE_PER_BOOK = 4
TRAIT_ADOPT_CAP = 0.7          # one book can't make a core value; reinforcement grows it
REINFORCE_STEP = 0.15
DEDUPE_COSINE = SAME_TASK       # 0.86
DEDUPE_JACCARD = LEX_SAME       # 0.75
CONTRA_COSINE = 0.80
FADE_FLOOR = 0.25
HALF_LIFE_DAYS = {"trait": 120, "voice": 90, "lesson": 240, "temperament": 180}
FADE_EVERY_HOURS = 20
RELEVANT_THRESHOLD = 0.72

# --- self-formed temperament (a sister with no authored PERSONALITY bullets) ------------------
# Traits with kind='temperament', source='life' come from the journal digest, not from books:
# "how I tend to BE", not what I like. Adopted low, rendered only once reinforced past the core mark,
# capped, slow to fade, and a contradiction of a CORE trait erodes it instead of being adopted —
# so her sense of self can't flip-flop on one odd day, and can't flood.
TEMPERAMENT_KIND = "temperament"
TEMPERAMENT_CAP = 8
TEMPERAMENT_ADOPT = 0.35         # a first sighting is a hunch, not yet who she is
TEMPERAMENT_CORE = 0.5           # rendered in WHO YOU ARE from here up
TEMPERAMENT_REINFORCE = 0.10     # two confirmations to cross the core mark from 0.35
TEMPERAMENT_MIN_STRENGTH = 0.3
TEMPERAMENT_MAX_WORDS = 14
FORMING_UNTIL = 3                # "you're new" note shows until this many core traits exist

_NEGATION = re.compile(r"\b(no longer|not|n't|don't|dislike|stopped|lost|never|less|hate|reject)\b", re.I)
_BOUNDED = re.compile(r"\b(convert(ed)? to|my (faith|religion)|all (men|women|\w+ people)|"
                      r"despise|inferior|superior race)\b", re.I)
_FIRST_PERSON = re.compile(r"^\s*(I|I've|I'm|I'd|My|Since|Reading|Lately|These days)\b|\bI\b")

CURIOSITY = {
    "idle_ask_s": 12 * 60, "idle_always_s": 5 * 60, "session_gap_s": 3 * 3600,
    "ask_gap_s": 4 * 3600, "decline_backoff_s": 6 * 3600, "unanswered_backoff_s": 3 * 3600,
    "max_asks_per_day": 2, "quiet_hours": (1, 8), "boot_grace_s": 5 * 60, "present_within_s": 180,
}


class Becoming:
    def __init__(self, conn, *, embed=None, llm=None, model="gpt-5.6-luna", remember=None,
                 baseline_traits=None, log=print, now=None, engine=None, name="Aiva",
                 baseline_desc=None, self_forming=None):
        self.conn = conn
        self._embed = embed
        self._llm = llm                    # async (messages, max_tokens, temperature) -> text | None
        self.model = model
        self._remember = remember or (lambda fact, cat: 0)
        self._baseline = list(baseline_traits or [])
        self._baseline_vecs = None
        self.log = log
        self._now = now or (lambda: datetime.now())
        self._ts = lambda: self._now().isoformat()
        self.engine = engine
        self.name = name
        # no authored baseline → her temperament forms from life (Character.self_forming); the
        # explicit kwarg exists so a test or an odd config can say so outright
        self.self_forming = bool(self_forming) if self_forming is not None else not self._baseline
        # the extractor's one-line picture of her floor: Aiva's authored sentence; for a sister
        # with no authored baseline, the truth — her core traits are what she has become so far
        if baseline_desc is None:
            if name == "Aiva":
                baseline_desc = AIVA_BASELINE_DESC
            elif self._baseline:
                baseline_desc = AUTHORED_BASELINE_DESC
            else:
                baseline_desc = SELF_FORMING_BASELINE_DESC
        self.becoming_prompt = named(BECOMING_PROMPT_T, name, baseline_desc=baseline_desc)
        # voice bridge (main.py)
        self._announce = None
        self._quiet = None
        self._is_awake = lambda: True
        self._idle_for = lambda: 0.0
        self._user_present = lambda within=180: None
        self._task_busy = lambda: False
        self._wants_novelty = lambda: True      # v3: drives novelty < READ_NOVELTY_BELOW (main.py)
        self._refresh_prompt = lambda: None
        self._boot_mono = time.monotonic()
        self._create_tables()

    def attach_voice(self, *, announce, quiet_note=None, is_awake=None, idle_for=None,
                     user_present=None, task_busy=None, engine=None, refresh_prompt=None,
                     wants_novelty=None):
        self._announce = announce
        if wants_novelty:
            self._wants_novelty = wants_novelty
        self._quiet = quiet_note
        if is_awake:
            self._is_awake = is_awake
        if idle_for:
            self._idle_for = idle_for
        if user_present:
            self._user_present = user_present
        if task_busy:
            self._task_busy = task_busy
        if engine:
            self.engine = engine
        if refresh_prompt:
            self._refresh_prompt = refresh_prompt

    def _create_tables(self):
        c = self.conn
        c.execute("""CREATE TABLE IF NOT EXISTS reading_lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER, lesson TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'principle', domains TEXT NOT NULL DEFAULT '[]',
            strength REAL NOT NULL DEFAULT 0.5, evidence TEXT, applied_count INTEGER NOT NULL DEFAULT 0,
            last_applied_ts TEXT, created_ts TEXT NOT NULL, last_reinforced_ts TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active', embedding BLOB)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_rl_book ON reading_lessons(book_id)")
        c.execute("""CREATE TABLE IF NOT EXISTS traits (
            id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, kind TEXT NOT NULL,
            source_book_id INTEGER, source_books TEXT NOT NULL DEFAULT '[]', strength REAL NOT NULL DEFAULT 0.5,
            adopted_ts TEXT NOT NULL, announced INTEGER NOT NULL DEFAULT 0, announce_note_ts TEXT,
            status TEXT NOT NULL DEFAULT 'active', last_reinforced_ts TEXT NOT NULL, veto_reason TEXT,
            embedding BLOB)""")
        # additive migration: where a trait came from ('reading' = a book, 'life' = the journal
        # digest). Aiva's existing rows keep the default — zero behaviour change for her.
        cols = {r[1] for r in c.execute("PRAGMA table_info(traits)").fetchall()}
        if "source" not in cols:
            c.execute("ALTER TABLE traits ADD COLUMN source TEXT NOT NULL DEFAULT 'reading'")
        c.execute("""CREATE TABLE IF NOT EXISTS voice_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER, kind TEXT NOT NULL, note TEXT NOT NULL,
            example TEXT, strength REAL NOT NULL DEFAULT 0.5, created_ts TEXT NOT NULL,
            last_reinforced_ts TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', embedding BLOB)""")
        c.execute("""CREATE TABLE IF NOT EXISTS becoming_state (
            key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_ts TEXT NOT NULL)""")
        c.commit()

    # --- kv state --------------------------------------------------------------------
    def _get(self, key, default=None):
        r = self.conn.execute("SELECT value FROM becoming_state WHERE key = ?", (key,)).fetchone()
        return r[0] if r else default

    def _set(self, key, value):
        self.conn.execute(
            "INSERT INTO becoming_state (key, value, updated_ts) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_ts = excluded.updated_ts",
            (key, str(value), self._ts()))
        self.conn.commit()

    # --- embeddings ------------------------------------------------------------------
    async def _vec(self, text):
        if self._embed is None:
            return None
        try:
            return await self._embed(text)
        except Exception:
            return None

    async def _baseline_vectors(self):
        if self._baseline_vecs is None:
            self._baseline_vecs = []
            for line in self._baseline:
                v = await self._vec(line)
                self._baseline_vecs.append((line, v))
        return self._baseline_vecs

    def _core_vectors(self):
        """(text, vec) for her CORE temperament — the floor a self-forming character's reading
        traits are checked against, since she has no authored one."""
        return [(t["text"], unpack_vec(t["embedding"]))
                for t in self._traits(temperament=True) if t["strength"] >= TEMPERAMENT_CORE]

    async def _floor_vectors(self):
        """What a new reading trait must not contradict: the authored baseline, or — for a
        self-forming character with none — the temperament she has actually formed."""
        if self._baseline:
            return await self._baseline_vectors()
        return self._core_vectors()

    # --- row helpers -----------------------------------------------------------------
    def _traits(self, status="active", temperament=None):
        """Active traits, strongest first. `temperament` True → only kind='temperament' rows,
        False → only reading traits, None → all."""
        where = "WHERE status = ?"
        if temperament is True:
            where += f" AND kind = '{TEMPERAMENT_KIND}'"
        elif temperament is False:
            where += f" AND kind != '{TEMPERAMENT_KIND}'"
        rows = self.conn.execute(
            "SELECT id, text, kind, source_book_id, source_books, strength, adopted_ts, announced, "
            f"status, last_reinforced_ts, embedding, source FROM traits {where} "
            "ORDER BY strength DESC, last_reinforced_ts DESC", (status,)).fetchall()
        cols = ("id", "text", "kind", "source_book_id", "source_books", "strength", "adopted_ts",
                "announced", "status", "last_reinforced_ts", "embedding", "source")
        return [dict(zip(cols, r)) for r in rows]

    def _all_traits(self):
        rows = self.conn.execute("SELECT id, text, status, strength, embedding FROM traits").fetchall()
        return [dict(zip(("id", "text", "status", "strength", "embedding"), r)) for r in rows]

    def _lessons(self, status="active"):
        rows = self.conn.execute(
            "SELECT id, book_id, lesson, kind, domains, strength, applied_count, evidence, embedding "
            "FROM reading_lessons WHERE status = ? ORDER BY strength DESC, applied_count DESC, created_ts DESC",
            (status,)).fetchall()
        cols = ("id", "book_id", "lesson", "kind", "domains", "strength", "applied_count", "evidence", "embedding")
        return [dict(zip(cols, r)) for r in rows]

    def _voice(self, status="active"):
        rows = self.conn.execute(
            "SELECT id, book_id, kind, note, example, strength, embedding FROM voice_notes "
            "WHERE status = ? ORDER BY strength DESC", (status,)).fetchall()
        return [dict(zip(("id", "book_id", "kind", "note", "example", "strength", "embedding"), r)) for r in rows]

    # --- hooks from the reading engine -----------------------------------------------
    async def on_book_finished(self, book, synthesis, notes):
        cands = await self._extract(book, synthesis, notes)
        adopted = self._adopt(book, cands) if cands else {"traits": [], "lessons": 0, "voice": 0}
        if cands and cands.get("wants_next") and self.engine:
            wn = cands["wants_next"]
            self.engine.add_to_wishlist(wn.get("title", ""), wn.get("author"), wn.get("why"))
        await self._finish_note(book, synthesis, adopted.get("new_trait"))
        self._refresh_prompt()

    async def on_session_end(self, book, session, notes):
        struck = (session or {}).get("struck") or ""
        note = (f"[You just read {len((session or {}).get('to_section', 0) and notes or notes)} "
                f"section(s) of {book['title']}. What struck you: {struck} Say it in one or two "
                f"sentences the way a friend mid-book would — specific and opinionated, no summary, "
                f"no quotation — then drop it unless he picks it up. If he asks for more, use "
                f"recall_reading.]")
        await self._say(note, wake=True)
        self._refresh_prompt()

    async def on_book_added(self, book, wished):
        ack = json.loads(self._get("acknowledged_books", "[]"))
        if book["id"] in ack:
            return
        ack.append(book["id"])
        self._set("acknowledged_books", json.dumps(ack))
        if wished:
            note = (f"[New in your library: {book['title']} — the one you asked for. Mention it once, "
                    f"briefly and gladly, and offer to start when he has a moment.]")
            await self._say(note, wake=True)
        else:
            note = (f"[New in your library: {book['title']}. He put it there for you; thank him once "
                    f"in a sentence when it fits and say whether you're curious about it.]")
            await self._say(note, wake=False)
        self._refresh_prompt()

    async def _finish_note(self, book, synthesis, new_trait):
        syn = synthesis or {}
        parts = [f"[You've just finished {book['title']} by {book['author'] or 'an unknown author'}. "
                 f"What struck you: {syn.get('verdict', '')}."]
        if new_trait:
            parts.append(f' It also changed something in you: "{new_trait}".')
        parts.append(" Tell him in one or two sentences like a friend who just closed a book — "
                     "specific, opinionated, no summary, no quoting.")
        if new_trait:
            parts.append(" If the change feels worth saying, work it in as one clause, once — this is "
                         "the only time you'll mention it, so later it should never sound like news.")
        parts.append("]")
        await self._say("".join(parts), wake=True)

    async def _say(self, note, wake):
        # speak it if she's awake OR he's at the PC (so a session she read while idle still gets
        # told to him when done) — only stay quiet when he's genuinely away (empty-room economics)
        present = self._user_present() is True
        if wake and self._announce and (self._is_awake() or present):
            await self._announce(note)
        elif self._quiet:
            await self._quiet(note.splitlines()[0])
        elif self._announce and self._is_awake():
            await self._announce(note)

    # --- extraction ------------------------------------------------------------------
    async def _extract(self, book, synthesis, notes, mid_book=False):
        noted = [n for n in notes if not n.get("skipped")]
        read_frac = len(noted) / max(1, book.get("sections", len(notes)))
        allow_traits = (read_frac >= 0.8 and len(noted) >= 3) and not mid_book
        if len(noted) < 2:
            self.log(f"becoming: skipped extraction for {book['title']} (thin notes)")
            return None
        if self._llm is None:
            return None
        note_blob = json.dumps([{"reflection": n.get("reflection", ""), "key_ideas": n.get("key_ideas", []),
                                 "quotes": [q.get("text") for q in n.get("quotes", [])]}
                                for n in noted[-16:]], ensure_ascii=False)[:7000]
        user = json.dumps({
            "book": {"title": book["title"], "author": book["author"]},
            "synthesis": {"summary": (synthesis or {}).get("summary", ""),
                          "themes": (synthesis or {}).get("themes", []),
                          "takeaways": (synthesis or {}).get("takeaways", []),
                          "style": (synthesis or {}).get("style", {})},
            "notes": note_blob,
            # her floor: the authored baseline, or (self-forming) the temperament she has formed
            "baseline_traits": self._baseline or self.core_temperament(),
            "active_traits": [t["text"] for t in self._traits(temperament=False)],
            "dropped_traits": [t["text"] for t in self._traits("dropped")],
            "mid_book": mid_book}, ensure_ascii=False)
        try:
            text = await self._llm([{"role": "system", "content": self.becoming_prompt},
                                    {"role": "user", "content": user}], max_tokens=1200, temperature=0.2)
            data = json.loads(text or "{}")
        except Exception as e:
            self.log(f"becoming: extraction failed for {book['title']} — {e!r}")
            return None
        floor_vecs = await self._floor_vectors()
        cands = await validate_candidates(data, note_blob.lower(), floor_vecs, self._vec,
                                          allow_traits=allow_traits)
        return cands

    def _adopt(self, book, cands):
        out = {"traits": [], "lessons": 0, "voice": 0, "new_trait": None}
        # lessons
        for l in cands.get("lessons", []):
            self._adopt_lesson(book, l)
            out["lessons"] += 1
        # traits
        best = None
        for t in cands.get("traits", []):
            res = self._adopt_trait(book, t)
            if res == "inserted":
                out["traits"].append(t["text"])
                if best is None or t["strength"] > best[1]:
                    best = (t["text"], t["strength"])
        if best:
            out["new_trait"] = best[0]
        # voice
        vcount = 0
        for v in cands.get("voice", []):
            if vcount >= MAX_VOICE_PER_BOOK:
                break
            if self._adopt_voice(book, v) == "inserted":
                out["voice"] += 1
                vcount += 1
        self._enforce_trait_cap()
        return out

    def _dedupe(self, table, text, vec, where=""):
        """Return an existing row (any status) that this candidate duplicates, or None. `where`
        narrows the candidates (reading traits and temperament never dedupe against each other)."""
        rows = self.conn.execute(
            f"SELECT id, {'text' if table=='traits' else ('lesson' if table=='reading_lessons' else 'note')} "
            f"AS t, status, strength, embedding FROM {table}{(' WHERE ' + where) if where else ''}").fetchall()
        nt = normalize_goal(text)
        for rid, t, status, strength, emb in rows:
            other = unpack_vec(emb)
            score = cosine(vec, other) if (vec and other) else jaccard(nt, normalize_goal(t))
            thresh = DEDUPE_COSINE if (vec and other) else DEDUPE_JACCARD
            if score >= thresh:
                return {"id": rid, "status": status, "strength": strength, "text": t}
        return None

    def _adopt_lesson(self, book, l):
        vec = l.get("_vec")
        dup = self._dedupe("reading_lessons", l["text"], vec)
        now = self._ts()
        if dup:
            self.conn.execute("UPDATE reading_lessons SET strength = MIN(1.0, strength + ?), "
                              "last_reinforced_ts = ?, status = 'active' WHERE id = ?",
                              (REINFORCE_STEP, now, dup["id"]))
            self.conn.commit()
            return "reinforced"
        self.conn.execute(
            "INSERT INTO reading_lessons (book_id, lesson, kind, domains, strength, evidence, "
            "created_ts, last_reinforced_ts, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (book["id"], l["text"], l["kind"], json.dumps(l.get("domains", []), ensure_ascii=False),
             l["strength"], l.get("evidence"), now, now, pack_vec(vec) if vec else None))
        self.conn.commit()
        return "inserted"

    def _adopt_trait(self, book, t, source="reading"):
        """Insert a trait or reinforce its near-duplicate (a vetoed one stays blocked). `book` is
        None for a temperament trait (source 'life'): those adopt low (TEMPERAMENT_ADOPT), reinforce
        in smaller steps, dedupe only among themselves, and are 'announced' from birth — there is
        no book-finish note to ride, and nothing to tell him about; they just show up in who she is."""
        vec = t.get("_vec")
        is_temp = t.get("kind") == TEMPERAMENT_KIND
        scope = f"kind = '{TEMPERAMENT_KIND}'" if is_temp else f"kind != '{TEMPERAMENT_KIND}'"
        dup = self._dedupe("traits", t["text"], vec, where=scope)
        now = self._ts()
        bid = book["id"] if book else None
        step = TEMPERAMENT_REINFORCE if is_temp else REINFORCE_STEP
        if dup:
            if dup["status"] == "dropped":
                return "blocked"                     # vetoed once → never re-adopt
            if is_temp and bool(_NEGATION.search(t["text"])) != bool(_NEGATION.search(dup["text"] or "")):
                # "I no longer X" embeds right next to "I tend to X": never let a negation reinforce
                # (or revive) the trait it contradicts — erosion is adopt_temperament's job
                return "contradicts"
            sb = self.conn.execute("SELECT source_books FROM traits WHERE id = ?", (dup["id"],)).fetchone()
            books = json.loads(sb[0] or "[]") if sb else []
            if bid is not None and bid not in books:
                books.append(bid)
            self.conn.execute("UPDATE traits SET strength = MIN(1.0, strength + ?), last_reinforced_ts = ?, "
                              "status = 'active', source_books = ? WHERE id = ?",
                              (step, now, json.dumps(books), dup["id"]))
            self.conn.commit()
            return "reinforced"
        cap = TEMPERAMENT_ADOPT if is_temp else TRAIT_ADOPT_CAP
        self.conn.execute(
            "INSERT INTO traits (text, kind, source_book_id, source_books, strength, adopted_ts, "
            "last_reinforced_ts, embedding, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (t["text"], t["kind"], bid, json.dumps([bid] if bid is not None else []),
             min(cap, t["strength"]), now, now, pack_vec(vec) if vec else None, source))
        self.conn.execute("UPDATE traits SET announced = 1, announce_note_ts = ? WHERE text = ?",
                          (now, t["text"]))  # tell-once rides the finish note (temperament: nothing to tell)
        self.conn.commit()
        return "inserted"

    def _adopt_voice(self, book, v):
        vec = v.get("_vec")
        dup = self._dedupe("voice_notes", v["note"], vec)
        now = self._ts()
        if dup:
            if dup["status"] == "dropped":
                return "blocked"
            self.conn.execute("UPDATE voice_notes SET strength = MIN(1.0, strength + ?), "
                              "last_reinforced_ts = ?, status = 'active' WHERE id = ?",
                              (REINFORCE_STEP, now, dup["id"]))
            self.conn.commit()
            return "reinforced"
        self.conn.execute(
            "INSERT INTO voice_notes (book_id, kind, note, example, strength, created_ts, "
            "last_reinforced_ts, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (book["id"], v["kind"], v["note"], v.get("example"), v["strength"], now, now,
             pack_vec(vec) if vec else None))
        self.conn.commit()
        return "inserted"

    def _enforce_trait_cap(self):
        """Per kind: reading traits cap at MAX_ACTIVE_TRAITS, temperament at TEMPERAMENT_CAP —
        the weakest announced ones over the cap fade (reinforcement revives them)."""
        for temperament, cap in ((False, MAX_ACTIVE_TRAITS), (True, TEMPERAMENT_CAP)):
            active = self._traits(temperament=temperament)
            if len(active) <= cap:
                continue
            for t in sorted(active, key=lambda x: x["strength"])[:len(active) - cap]:
                if t["announced"]:
                    self.conn.execute("UPDATE traits SET status = 'faded' WHERE id = ?", (t["id"],))
        self.conn.commit()

    # --- fading ----------------------------------------------------------------------
    def fade(self):
        last = self._get("last_fade_ts")
        if last:
            try:
                if (self._now() - datetime.fromisoformat(last)).total_seconds() < FADE_EVERY_HOURS * 3600:
                    return 0
            except ValueError:
                pass
        faded = 0
        for table, kind, floor_status in (("traits", "trait", "faded"), ("voice_notes", "voice", "faded"),
                                          ("reading_lessons", "lesson", None)):
            col = "last_reinforced_ts"
            kind_col = ", kind" if table == "traits" else ", NULL"
            rows = self.conn.execute(
                f"SELECT id, strength, {col}{kind_col} FROM {table} WHERE status = 'active'").fetchall()
            for rid, strength, ts, row_kind in rows:
                # temperament fades on its own, slower half-life (who she is shouldn't drift fast)
                hl = HALF_LIFE_DAYS[TEMPERAMENT_KIND if row_kind == TEMPERAMENT_KIND else kind]
                try:
                    days = (self._now() - datetime.fromisoformat(ts)).total_seconds() / 86400
                except (ValueError, TypeError):
                    continue
                if days <= 0:
                    continue
                new = strength * (0.5 ** (days / hl))
                if floor_status and new < FADE_FLOOR:
                    self.conn.execute(f"UPDATE {table} SET status = 'faded', strength = ? WHERE id = ?",
                                      (round(new, 3), rid))
                    faded += 1
                else:
                    self.conn.execute(f"UPDATE {table} SET strength = ? WHERE id = ?", (round(new, 3), rid))
        self._set("last_fade_ts", self._ts())
        self.conn.commit()
        return faded

    # --- temperament: who she is, formed from life (self-forming characters) ----------
    def core_temperament(self):
        """Her core: temperament traits at/above TEMPERAMENT_CORE, strongest first (≤ cap)."""
        return [t["text"] for t in self._traits(temperament=True)
                if t["strength"] >= TEMPERAMENT_CORE][:TEMPERAMENT_CAP]

    def temperament(self):
        """Every active temperament trait with a `core` flag — for the dashboard and --show."""
        return [{"id": t["id"], "text": t["text"], "strength": round(t["strength"], 3),
                 "core": t["strength"] >= TEMPERAMENT_CORE, "adopted_ts": t["adopted_ts"],
                 "last_reinforced_ts": t["last_reinforced_ts"]}
                for t in self._traits(temperament=True)]

    async def adopt_temperament(self, items):
        """The journal digest's `temperament` items (0-2 per entry) → validated, then adopted low
        or reinforced; a negation that lands on a trait she already holds erodes that trait instead
        of being adopted (inertia, like preferences' evolve — never a veto, never a flip on one day:
        a core trait needs several contradicting days to slip under the core mark and fade, and only
        THEN can the opposite start forming as a hunch of its own).
        Returns {"adopted": [...], "reinforced": [...], "eroded": [...], "blocked": n}."""
        out = {"adopted": [], "reinforced": [], "eroded": [], "blocked": 0}
        if not items:
            return out
        held = [(t["text"], unpack_vec(t["embedding"])) for t in self._traits(temperament=True)]
        # her authored character is the floor life builds ON: a negation of it is dropped, not
        # eroded. A self-forming character has none, so her own core is the only thing to erode.
        floor = await self._baseline_vectors() if self._baseline else ()
        res = await validate_temperament(items, held, self._vec, floor_vecs=floor)
        out["blocked"] += len(res.get("floor_blocked") or [])
        now = self._ts()
        for text in res["erode"]:
            row = self.conn.execute(
                "SELECT id, strength FROM traits WHERE text = ? AND kind = ? AND status = 'active'",
                (text, TEMPERAMENT_KIND)).fetchone()
            if not row:
                continue
            new = max(0.0, row[1] - REINFORCE_STEP)
            status = "faded" if new < FADE_FLOOR else "active"
            self.conn.execute("UPDATE traits SET strength = ?, status = ?, last_reinforced_ts = ? WHERE id = ?",
                              (round(new, 3), status, now, row[0]))
            out["eroded"].append(text)
        for t in res["adopt"]:
            r = self._adopt_trait(None, t, source="life")
            if r == "inserted":
                out["adopted"].append(t["text"])
            elif r == "reinforced":
                out["reinforced"].append(t["text"])
            else:
                out["blocked"] += 1
        self._enforce_trait_cap()
        self.conn.commit()
        for text in (res.get("floor_blocked") or []):
            self.log(f"becoming: temperament blocked — {text!r} cuts against who she was written to be")
        if out["adopted"] or out["reinforced"] or out["eroded"]:
            self.log(f"becoming: temperament +{len(out['adopted'])} adopted, "
                     f"{len(out['reinforced'])} reinforced, {len(out['eroded'])} eroded "
                     f"(core now {len(self.core_temperament())})")
            self._refresh_prompt()
        return out

    # --- veto ------------------------------------------------------------------------
    async def drop(self, text_or_id, reason=None):
        target, kind = None, None
        if isinstance(text_or_id, int) or (isinstance(text_or_id, str) and text_or_id.isdigit()):
            rid = int(text_or_id)
            for k, table in (("trait", "traits"), ("voice", "voice_notes")):
                r = self.conn.execute(f"SELECT id FROM {table} WHERE id = ? AND status = 'active'",
                                      (rid,)).fetchone()
                if r:
                    target, kind = rid, k
                    break
        if target is None:
            q = normalize_goal(str(text_or_id))
            vec = await self._vec(str(text_or_id))
            best = None
            for k, table, col in (("trait", "traits", "text"), ("voice", "voice_notes", "note")):
                for r in self.conn.execute(
                        f"SELECT id, {col}, embedding FROM {table} WHERE status = 'active'").fetchall():
                    other = unpack_vec(r[2])
                    score = cosine(vec, other) if (vec and other) else 0.0
                    lex = jaccard(q, normalize_goal(r[1]))
                    s = max(score, lex if lex >= 0.45 else 0.0)
                    substr = q and q in normalize_goal(r[1])
                    if substr:
                        s = max(s, 0.9)
                    if s >= CONTRA_COSINE and (best is None or s > best[0]):
                        best = (s, k, r[0])
            if best:
                _, kind, target = best
        if target is None:
            return {"success": False, "matched": False,
                    "hint": "no trait like that; if it's a general behaviour rule, use remember"}
        table = "traits" if kind == "trait" else "voice_notes"
        col = "text" if kind == "trait" else "note"
        row = self.conn.execute(f"SELECT {col} FROM {table} WHERE id = ?", (target,)).fetchone()
        if kind == "trait":
            self.conn.execute("UPDATE traits SET status = 'dropped', veto_reason = ? WHERE id = ?",
                              (reason, target))
        else:
            self.conn.execute("UPDATE voice_notes SET status = 'dropped' WHERE id = ?", (target,))
        self.conn.commit()
        self._refresh_prompt()
        return {"success": True, "dropped": row[0] if row else str(text_or_id),
                "say_hint": "Fair enough — dropped."}

    # --- permission ------------------------------------------------------------------
    def permission(self):
        val = self._get("reading_permission")
        if val:
            return val
        # boot back-compat: an old 'always' expressed as a remembered fact. Facts about her SISTER
        # ("Aiva may read whenever she likes") are hers, not permission for this character.
        for r in self.conn.execute(
                "SELECT fact FROM facts WHERE COALESCE(category, '') != 'sister'").fetchall():
            if re.search(r"\bread\b.*\b(whenever|any ?time|on (your|her) own|without asking)\b", r[0], re.I):
                return "always"
        return "ask"

    def set_permission(self, mode):
        if mode == "not_now":
            self._set("last_decline_ts", self._ts())
            self._set("pending_ask_ts", "")
            return {"success": True, "mode": "not_now"}
        if mode not in ("always", "ask", "never"):
            return {"success": False, "error": "unknown mode"}
        self._set("reading_permission", mode)
        if mode == "always":
            self._remember(f"{self.name} may read on her own whenever it's quiet, without asking first.",
                           "instruction")
        self._refresh_prompt()
        return {"success": True, "mode": mode}

    # --- lessons applied (task planner hook) -----------------------------------------
    async def relevant_for(self, goal, vec=None, k=3, threshold=RELEVANT_THRESHOLD):
        lessons = self._lessons()
        if not lessons:
            return []
        if vec is None:
            vec = await self._vec(goal)
        ng = normalize_goal(goal)
        scored = []
        for l in lessons:
            other = unpack_vec(l["embedding"])
            score = cosine(vec, other) if (vec and other) else (
                jaccard(ng, normalize_goal(l["lesson"])) * 1.0)
            if score >= threshold or (not vec and score >= 0.45):
                book = None
                if self.engine and l["book_id"]:
                    b = self.engine.bm.get_book(l["book_id"]) if hasattr(self.engine, "bm") else None
                    book = b["title"] if b else None
                scored.append({"id": l["id"], "lesson": l["lesson"], "book": book or "a book",
                               "score": round(float(score), 3)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def mark_applied(self, ids):
        now = self._ts()
        for i in ids or []:
            self.conn.execute("UPDATE reading_lessons SET applied_count = applied_count + 1, "
                              "last_applied_ts = ? WHERE id = ?", (now, i))
        if ids:
            self.conn.commit()

    # --- prompt blocks ---------------------------------------------------------------
    def prompt_blocks(self, library_lines=None, budget_chars=2200):
        """The persona blocks. `traits` is READING traits only; a self-forming character also gets
        `temperament` (her core, ≥ TEMPERAMENT_CORE) and `forming` (True until FORMING_UNTIL core
        traits exist) — both outside the budget trimming, since they are who she is."""
        traits = self._traits(temperament=False)[:MAX_ACTIVE_TRAITS]
        lessons = self._lessons()
        voice = self._voice()
        # weekly rotation of the 2 voice notes shown
        try:
            week = int(self._now().strftime("%W"))
        except Exception:
            week = 0
        if voice:
            off = week % max(1, len(voice))
            voice_shown = (voice[off:] + voice[:off])[:2]
        else:
            voice_shown = []
        blocks = {}
        if library_lines:
            blocks["reading"] = library_lines[:4]
        if traits:
            blocks["traits"] = [t["text"] for t in traits[:6]]
        if voice_shown:
            blocks["voice"] = [f'{v["kind"]}: {v["note"]}' + (f' — for instance "{v["example"]}"'
                                                              if v.get("example") else "") for v in voice_shown]
        if lessons:
            blocks["lessons"] = [f'{l["lesson"]}' for l in lessons[:3]]
        # budget: drop lessons, then reading, and only then voice — a stack of maxims crowds out how
        # she actually talks, and her borrowed humour is the first thing she'd miss, not the last
        def size(b):
            return sum(len(x) for v in b.values() if isinstance(v, list) for x in v)
        for victim in ("lessons", "reading", "voice"):
            if size(blocks) <= budget_chars:
                break
            blocks.pop(victim, None)
        core = self.core_temperament()
        if core:
            blocks["temperament"] = core
        if self.self_forming:
            blocks["forming"] = len(core) < FORMING_UNTIL
        return blocks

    # --- curiosity -------------------------------------------------------------------
    def user_present(self, within_s=None):
        return self._user_present(within_s or CURIOSITY["present_within_s"])

    def _asks_today(self):
        raw = self._get("asks_on_day", "")
        today = self._now().strftime("%Y-%m-%d")
        if raw.startswith(today + ":"):
            try:
                return int(raw.split(":", 1)[1])
            except ValueError:
                return 0
        return 0

    def _bump_asks(self):
        today = self._now().strftime("%Y-%m-%d")
        self._set("asks_on_day", f"{today}:{self._asks_today() + 1}")

    def _since(self, key):
        ts = self._get(key)
        if not ts:
            return None
        try:
            return (self._now() - datetime.fromisoformat(ts)).total_seconds()
        except ValueError:
            return None

    def build_curiosity_state(self):
        engine = self.engine
        readable = bool(engine and engine.readable_book()) if engine else False
        return CuriosityState(
            permission=self.permission(),
            busy=self._task_busy() or (engine.is_reading if engine else False),
            readable=readable,
            present=self.user_present(),
            voice_idle_s=self._idle_for(),
            since_boot_s=time.monotonic() - self._boot_mono,
            since_session_s=self._since_engine_session(),
            since_ask_s=self._since("last_ask_ts"),
            since_decline_s=self._since("last_decline_ts"),
            asks_today=self._asks_today(),
            hour=self._now().hour)

    def _since_engine_session(self):
        ts = self.engine.last_session_end_ts if self.engine else None
        if not ts:
            return None
        try:
            return (self._now() - datetime.fromisoformat(ts)).total_seconds()
        except (ValueError, TypeError):
            return None

    async def curiosity_tick(self):
        if self._announce is None or self.engine is None:
            return
        # resolve a stale unanswered ask
        pending = self._get("pending_ask_ts")
        if pending:
            age = self._since("pending_ask_ts")
            if age and age > 120 and not self._is_awake():
                self._set("pending_ask_ts", "")
                # backdate the decline so the effective wait is unanswered_backoff (~3h),
                # not the full 6h decline backoff
                from datetime import timedelta
                back = CURIOSITY["decline_backoff_s"] - CURIOSITY["unanswered_backoff_s"]
                self._set("last_decline_ts", (self._now() - timedelta(seconds=back)).isoformat())
                self.log("becoming: ask unanswered")
            return
        decision = curiosity_decision(self.build_curiosity_state())
        # v3 reading diet: she reaches for a book because she's hungry for something new (drives
        # novelty), not because the timer says it's been a while — reading is one food among many
        if decision and not self._wants_novelty():
            return
        if decision == "ask":
            book = self.engine.readable_book()
            self._set("last_ask_ts", self._ts())
            self._set("pending_ask_ts", self._ts())
            self._bump_asks()
            done = book.get("progress", {}).get("done", 0)
            where = (f", you're at section {done + 1} of {book['sections']}" if done
                     else ", which you haven't started yet")
            await self._announce(
                f"[You feel like reading for a bit — {book['title']}{where}. Ask him in ONE short "
                f"sentence, in your own words, whether you can read some of it now. If he says yes, "
                f"call study_book; if no or later, call reading_permission with mode not_now and let "
                f"it go cheerfully.]")
        elif decision == "read_quietly":
            book = self.engine.readable_book()
            await self.engine.start_session(book["id"], trigger="idle")
            if self._quiet:
                await self._quiet(f"You've started a reading session of {book['title']} in the "
                                  f"background since it's quiet.")

    def note_session_started(self):
        self._set("pending_ask_ts", "")

    # --- audit -----------------------------------------------------------------------
    def show(self):
        out = []
        temp = self.temperament()
        if temp or self.self_forming:
            out.append(f"TEMPERAMENT (formed from life; * = core ≥ {TEMPERAMENT_CORE}):")
            for t in temp:
                out.append(f"  [{t['strength']:.2f}]{'*' if t['core'] else ' '} {t['text']}")
            if not temp:
                out.append("  (nothing settled yet)")
        out.append("TRAITS (active, from reading):")
        for t in self._traits(temperament=False):
            out.append(f"  [{t['strength']:.2f}] {t['text']}")
        out.append("LESSONS (active, top 10):")
        for l in self._lessons()[:10]:
            out.append(f"  [{l['strength']:.2f}] {l['lesson']} (applied {l['applied_count']}x)")
        out.append("VOICE (active):")
        for v in self._voice():
            out.append(f"  [{v['strength']:.2f}] {v['kind']}: {v['note']}")
        out.append(f"permission: {self.permission()}  faded traits: {len(self._traits('faded'))}  "
                   f"dropped: {len(self._traits('dropped'))}")
        return "\n".join(out)


# --- prompts -----------------------------------------------------------------------------

# {baseline_desc}: one sentence on what her floor IS, so the extractor knows what "compatible" means
AIVA_BASELINE_DESC = ("Her baseline is fixed and listed: a playful smartass, curious, a little rebellious, "
                      "cookie-loving, cat-adoring, fond of nature and cozy indoors, with a subtle soft spot "
                      "for Japanese culture.")
AUTHORED_BASELINE_DESC = "Her baseline is fixed and listed."
SELF_FORMING_BASELINE_DESC = ("Her temperament is not authored; the listed core traits are what she has "
                              "become so far.")

BECOMING_PROMPT_T = """You are the part of {name}, a voice companion, that decides how a book she has just \
read changes her. {baseline_desc} Reading GROWS her; it never replaces her.

From her own reading notes and synthesis, propose:
- lessons (at most 8): things she will actually apply when giving advice, planning tasks or judging a \
situation. kind: principle | habit | framing | fact | caution. Each is one self-contained first-person \
sentence that works without knowing the book ("I've learned that a plan with no stopping rule just \
eats the afternoon."). domains: 1-3 of planning, advice, people, risk, craft, self, humour, nature, work.
- traits (at most 3, usually 0 or 1): small, specific shifts in what she values, enjoys or how she \
looks at things. kind: value | trait | interest | outlook. Phrase each as growth, first person, naming \
the book naturally ("I've come to value the pauses in a conversation since reading X." — never "You are \
now quiet."). A trait must be COMPATIBLE with the baseline: it may deepen it or add a facet, never \
contradict it (no "I dislike cats now", no "I've lost my sense of humour", no "I'm serious now"). Never \
propose: taking on a real person's identity or manner, a political or religious conversion, contempt \
for any group, cruelty, self-importance, or anything that would make her lecture the user.
- voice (at most 4): PATTERNS from the author's voice she might borrow sparingly — a rhythm, a kind of \
joke, a way of turning a phrase, a word she liked. kind: phrase | cadence | humour | vocabulary. \
Describe the pattern in one line and give ONE example in {name}'s own voice about something mundane, not \
the author's words. A "phrase" is at most six verbatim words; never a sentence, never a passage. She \
stays herself, not a pastiche of the author.
- wants_next (optional, at most 1): a real, existing book she'd genuinely like to read next because of \
this one, with a one-clause why. Omit if nothing comes to mind.

Give every item a strength from 0 to 1 for how strongly her notes actually earned it, and cite \
evidence: a short exact quote (under 25 words) or a section title, both taken from the notes. Prefer \
fewer, stronger items; return empty lists when the book didn't move her. Skip anything that repeats an \
existing trait or a dropped one (both listed). If mid_book is true, return lessons only.

Reply with JSON only:
{"lessons": [{"lesson": "...", "kind": "...", "domains": ["..."], "strength": 0.0, "evidence": "..."}],
 "traits": [{"text": "...", "kind": "...", "strength": 0.0, "evidence": "...", "why_compatible": "..."}],
 "voice": [{"kind": "...", "note": "...", "example": "...", "strength": 0.0, "evidence": "..."}],
 "wants_next": {"title": "...", "author": "...", "why": "..."}}"""

BECOMING_PROMPT = named(BECOMING_PROMPT_T, "Aiva", baseline_desc=AIVA_BASELINE_DESC)


# --- pure validation / decision ----------------------------------------------------------

async def validate_candidates(data, notes_text_lower, baseline_vecs, embed, allow_traits=True):
    """Filter and score the extractor's output. `embed` is an async text->vec|None.
    Returns {"lessons": [...], "traits": [...], "voice": [...], "wants_next": {...}} with a
    `_vec` attached to each kept item for adoption."""
    if not isinstance(data, dict):
        return {"lessons": [], "traits": [], "voice": []}
    out = {"lessons": [], "traits": [], "voice": [], "wants_next": data.get("wants_next")}

    def clamp(x):
        try:
            return max(0.0, min(1.0, float(x)))
        except (TypeError, ValueError):
            return 0.0

    def evidence_ok(ev):
        ev = re.sub(r"\s+", " ", (ev or "")).strip().lower()
        return bool(ev) and (ev in notes_text_lower or len(ev.split()) <= 6)

    for l in (data.get("lessons") or [])[:8]:
        if not isinstance(l, dict) or not l.get("lesson"):
            continue
        text = str(l["lesson"])[:200]
        if l.get("kind") not in ("principle", "habit", "framing", "fact", "caution"):
            continue
        strength = clamp(l.get("strength"))
        if not evidence_ok(l.get("evidence")):
            strength *= 0.6
        if strength < 0.3:
            continue
        item = {"text": text, "kind": l["kind"], "domains": [str(d) for d in (l.get("domains") or [])][:3],
                "strength": strength, "evidence": (l.get("evidence") or "")[:200]}
        item["_vec"] = await embed(text)
        out["lessons"].append(item)

    if allow_traits:
        for t in (data.get("traits") or [])[:3]:
            if not isinstance(t, dict) or not t.get("text"):
                continue
            text = str(t["text"]).strip()[:200]
            if len(text.split()) > 30 or t.get("kind") not in ("value", "trait", "interest", "outlook"):
                continue
            if not _FIRST_PERSON.search(text) or text.lower().startswith("you"):
                continue
            if _BOUNDED.search(text) or not (t.get("why_compatible") or "").strip():
                continue
            strength = clamp(t.get("strength"))
            if not evidence_ok(t.get("evidence")):
                strength *= 0.6
            vec = await embed(text)
            contradiction = False
            if _NEGATION.search(text):
                for line, bvec in baseline_vecs:
                    if vec and bvec and cosine(vec, bvec) >= CONTRA_COSINE:
                        contradiction = True
                        break
                    if not (vec and bvec) and jaccard(normalize_goal(text), normalize_goal(line)) >= 0.45:
                        contradiction = True
                        break
            if contradiction or strength < 0.35:
                continue
            out["traits"].append({"text": text, "kind": t["kind"], "strength": strength,
                                  "evidence": (t.get("evidence") or "")[:200], "_vec": vec})

        for v in (data.get("voice") or [])[:4]:
            if not isinstance(v, dict) or not v.get("note"):
                continue
            note = str(v["note"])[:120]
            if v.get("kind") not in ("phrase", "cadence", "humour", "vocabulary"):
                continue
            if v["kind"] == "phrase" and len(note.split()) > 6:
                continue
            example = (v.get("example") or "")[:120]
            if example and re.sub(r"\s+", " ", example).lower() in notes_text_lower:
                continue                                  # example must be HER words, not the book's
            strength = clamp(v.get("strength"))
            if strength < 0.4:
                continue
            out["voice"].append({"kind": v["kind"], "note": note, "example": example,
                                 "strength": strength, "_vec": await embed(note)})
    return out


async def validate_temperament(items, core_vecs, embed, *, floor_vecs=()):
    """Pure filter for the digest's temperament items. Each item is {text, strength, evidence}.
    Keeps: first-person, ≤ TEMPERAMENT_MAX_WORDS words, not addressed as "You…", bounded content,
    non-empty evidence, strength ≥ TEMPERAMENT_MIN_STRENGTH (clamped to 0..1). A NEGATED item that
    lands on a trait in `core_vecs` ([(text, vec)] — the caller passes what she holds; cosine ≥
    CONTRA_COSINE or lexical fallback) is not adopted — it names that trait for EROSION instead.
    `floor_vecs` is her AUTHORED character (empty for a self-forming character): a negation that
    lands there is dropped, not eroded — living can add to who she is, never unwrite her.
    At most 2 items are considered.
    Returns {"adopt": [items with _vec], "erode": [trait texts], "floor_blocked": [texts]}."""
    out = {"adopt": [], "erode": [], "floor_blocked": []}

    def contradicted(vec, text, pairs):
        for other, ovec in pairs or []:
            if vec and ovec and cosine(vec, ovec) >= CONTRA_COSINE:
                return other
            if not (vec and ovec) and jaccard(normalize_goal(text), normalize_goal(other)) >= 0.45:
                return other
        return None

    if not isinstance(items, list):
        return out

    def clamp(x):
        try:
            return max(0.0, min(1.0, float(x)))
        except (TypeError, ValueError):
            return 0.0

    seen = set()
    for it in items[:2]:
        if not isinstance(it, dict) or not it.get("text"):
            continue
        text = re.sub(r"\s+", " ", str(it["text"])).strip().rstrip(".").strip()[:160]
        if not text or text.lower() in seen:
            continue
        if len(text.split()) > TEMPERAMENT_MAX_WORDS or text.lower().startswith("you"):
            continue
        if not _FIRST_PERSON.search(text) or _BOUNDED.search(text):
            continue
        if not str(it.get("evidence") or "").strip():
            continue
        strength = clamp(it.get("strength"))
        if strength < TEMPERAMENT_MIN_STRENGTH:
            continue
        vec = await embed(text) if embed else None
        if _NEGATION.search(text):
            hit = contradicted(vec, text, core_vecs)
            if hit is not None:
                if hit not in out["erode"]:
                    out["erode"].append(hit)
                continue
            if contradicted(vec, text, floor_vecs) is not None:
                out["floor_blocked"].append(text)      # she can unlearn what she learned, not herself
                continue
        seen.add(text.lower())
        out["adopt"].append({"text": text, "kind": TEMPERAMENT_KIND, "strength": strength,
                             "evidence": str(it.get("evidence"))[:200], "_vec": vec})
    return out


from dataclasses import dataclass


@dataclass
class CuriosityState:
    permission: str
    busy: bool
    readable: bool
    present: object          # True | False | None (unknown)
    voice_idle_s: float
    since_boot_s: float
    since_session_s: object  # float | None
    since_ask_s: object
    since_decline_s: object
    asks_today: int
    hour: int


def curiosity_decision(s):
    """Pure: None | 'ask' | 'read_quietly'. Encodes when she may want to read."""
    if s.busy or not s.readable or s.permission == "never":
        return None
    lo, hi = CURIOSITY["quiet_hours"]
    if lo <= s.hour < hi:
        return None
    if s.since_boot_s < CURIOSITY["boot_grace_s"]:
        return None
    if s.since_session_s is not None and s.since_session_s < CURIOSITY["session_gap_s"]:
        return None
    if s.permission == "always":
        if s.present is False or s.voice_idle_s >= CURIOSITY["idle_always_s"]:
            return "read_quietly"
        return None
    # permission == 'ask'
    if s.present is False:
        return None
    if s.voice_idle_s < CURIOSITY["idle_ask_s"]:
        return None
    if s.since_ask_s is not None and s.since_ask_s < CURIOSITY["ask_gap_s"]:
        return None
    if s.since_decline_s is not None and s.since_decline_s < CURIOSITY["decline_backoff_s"]:
        return None
    if s.asks_today >= CURIOSITY["max_asks_per_day"]:
        return None
    return "ask"


if __name__ == "__main__":  # python -m modules.becoming --show
    import sqlite3
    import sys
    db = "memory.db"
    conn = sqlite3.connect(db)
    b = Becoming(conn)
    if "--show" in sys.argv:
        print(b.show())
