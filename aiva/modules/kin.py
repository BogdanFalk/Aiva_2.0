"""Kin — what a sister has actually SEEN the other do, make, say and read.

Two characters live on one PC (Aiva, Saga). Each knows about the other the way people do: from
experience first, the model second. The house relays every public act a sister reports (`event`
messages: a tool she used, a wish she acted on, a find she kept, a book she finished, a task she
completed, a line she spoke) to the others, who record it HERE, in their own db, as observations:

    kin_observations(about, kind did|made|said|read|likes, capability, summary, last_example,
                     count, strength, first_ts, last_ts, status active|dropped, embedding)

`observe(event)` skips her own acts, maps a completed tool to one of `capabilities.CAPABILITIES`
via `TOOL_TO_CAPABILITY` (start_task/claude_code excluded — the capability comes through the task
itself; unmapped tools like avatar_expression are ignored), and rolls repeats up: `did` by
(about, capability) — count+1, strength+0.1, latest example kept — everything else by meaning
(cosine 0.86 / lexical 0.60, like her space). Each NEW observation (and the first few repeats of a
capability) also lands on HER experience bus as a small `sister` event, so noticing her sister
shapes her mood and diary a little. Later repeats are below the mood floor.

`prompt_block()` renders the "YOUR SISTER, X" persona lines from top observations ∪ the durable
`facts` rows in category 'sister' (written by fact extraction from overheard lines). `recall()`
answers "has she ever…?" with what was actually seen — or an honest empty note. `drop()` is a
permanent veto, `fade()` a slow half-life, and `cursor`/`catch_up()` let a sister who was offline
replay `house_events` she missed. No LLM call anywhere in this module.

Same rules as space.py: sqlite through the passed connection on the event-loop thread; embeddings
via an injected async callable; `now` injectable for tests.

Audit:  cd aiva && ..\\.venv\\Scripts\\python -m modules.kin --show
"""

import os
import sqlite3
from datetime import datetime

from modules import capabilities
from modules.character import display_name
from modules.task_memory import cosine, jaccard, normalize_goal, pack_vec, unpack_vec

KINDS = ("did", "made", "said", "read", "likes")
NOTE_KINDS = ("did", "said", "likes")           # what the journal digest may note
DEDUPE_COSINE = 0.86
DEDUPE_JACCARD = 0.60
RECALL_COSINE = 0.30                            # below this an embedding match is noise
INITIAL_STRENGTH = 0.3
ROLLUP_STEP = 0.1
SALIENT_ROLLUPS = 3                             # the first N repeats of a capability still register
SALIENCE_NEW = 0.2
SALIENCE_LATER = 0.1                            # below InnerConfig.salience_floor: logged, not felt
HALF_LIFE_DAYS = 90
FADE_FLOOR = 0.1
FADE_EVERY_HOURS = 20
# bounded per (about, kind): the weakest single sightings beyond this are forgotten outright
ROW_CAPS = {"did": 40, "made": 60, "said": 40, "read": 60, "likes": 30}
EXCLUDED_TOOLS = frozenset({"start_task", "claude_code"})
# shared tools get the capability that describes them most honestly (registry order would give
# find_and_keep for search_web and hand_over)
_PREFERRED = {"search_web": "look_up_info", "read_page": "look_up_info", "hand_over": "recall_finds"}
TASK_FAMILY_TO_CAPABILITY = {"use_the_pc": "use_the_pc", "build_software": "build_software"}
_BUILD_WORDS = ("code", "build", "software", "program", "claude", "script", "app", "implement",
                "refactor", "website", "bot")


def _build_tool_map():
    m = {}
    for c in capabilities.CAPABILITIES:
        for t in c.tools:
            if t in EXCLUDED_TOOLS:
                continue
            m.setdefault(t, c.name)
    for t, cap in _PREFERRED.items():
        if t in m:
            m[t] = cap
    return m


TOOL_TO_CAPABILITY = _build_tool_map()

# third person, in HER voice about her sister — rendered verbatim in the persona block
KIN_LINES = {
    "find_and_keep": "she can go find images or clips online, look at them, and keep the ones she "
                     "likes in her space",
    "look_up_info": "she can look things up online and read pages",
    "recall_finds": "she can bring back things she has kept in her space",
    "read_books": "she can read and study books from the library on her own",
    "remember_things": "she can remember facts, note her own wishes, and keep track of what she likes",
    "tend_cat": "she can look after the desktop cat — feed it, pet it, play with it",
    "know_sister": "she keeps track of what she has seen me do, make and say",
    "call_sister": "she can call me over — start me up when I'm off",
    "build_software": "she can build or change software by planning it with Claude Code, on his say-so",
    "change_myself": "she can change her own look or her own code, with his say-so",
    "use_the_pc": "she can work the PC — open apps, click around, type, and run things in a terminal",
    "run_shell": "she can run shell commands once he has seen and confirmed them",
    "spend_or_send": "she can't buy, pay, sign in, or send anything — same as me",
}


def _iso(v):
    if v is None:
        return datetime.now().isoformat()
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v).isoformat()
    return str(v)


def _ago(ts, now=None):
    try:
        ref = datetime.fromisoformat(_iso(now)) if now is not None else datetime.now()
        days = (ref - datetime.fromisoformat(ts)).days
    except (TypeError, ValueError):
        return "a while ago"
    return "today" if days <= 0 else ("yesterday" if days == 1 else f"{days} days ago")


def _get(ev, key, default=None):
    """Field of an event that may be a dict or a house_proto.Event."""
    if isinstance(ev, dict):
        return ev.get(key, default)
    return getattr(ev, key, default)


def _third_person(kind, summary):
    """A stored line always reads as HER doing it. Emitters already phrase most summaries; anything
    that doesn't start with 'she' gets a plain kind prefix so the block never reads as first person."""
    s = " ".join(str(summary or "").split()).strip()
    if not s:
        return ""
    if s.lower().startswith(("she ", "she'", "her ")):
        return s
    if kind == "said":
        s = s.strip('"').strip()
        return f'she said: "{s[:160]}"'
    prefix = {"did": "she did this: ", "made": "she made or kept: ", "read": "she read: ",
              "likes": "she seems to like: "}[kind]
    return prefix + s[:160]


class Kin:
    def __init__(self, conn: sqlite3.Connection, *, me, sisters, bus=None, embed=None, log=print,
                 now=None):
        self.conn = conn
        self.me = (me or "").strip().lower()
        self.sisters = tuple((s or "").strip().lower() for s in (sisters or ()) if s)
        self.bus = bus                     # her own ExperienceBus (inner.py) or None
        self._embed = embed                # async (text) -> vec | None
        self.log = log
        self._now = now or (lambda: datetime.now().isoformat())
        self._create_tables()

    # --- schema / kv --------------------------------------------------------------------

    def _create_tables(self):
        c = self.conn
        c.execute("""CREATE TABLE IF NOT EXISTS kin_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            about TEXT NOT NULL, kind TEXT NOT NULL, capability TEXT,
            summary TEXT NOT NULL, last_example TEXT,
            count INTEGER NOT NULL DEFAULT 1, strength REAL NOT NULL DEFAULT 0.3,
            first_ts TEXT NOT NULL, last_ts TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active', embedding BLOB)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_kin_about ON kin_observations(about, kind, status)")
        c.execute("""CREATE TABLE IF NOT EXISTS kin_kv (
            key TEXT PRIMARY KEY, value TEXT, updated_ts TEXT NOT NULL)""")
        c.commit()

    def _ts(self):
        return _iso(self._now())

    def _kv_get(self, key, default=None):
        r = self.conn.execute("SELECT value FROM kin_kv WHERE key = ?", (key,)).fetchone()
        return r[0] if r else default

    def _kv_set(self, key, value):
        self.conn.execute(
            "INSERT INTO kin_kv (key, value, updated_ts) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_ts = excluded.updated_ts",
            (key, None if value is None else str(value), self._ts()))
        self.conn.commit()

    @property
    def cursor(self):
        """The last `house_events.id` she has seen (catch-up resumes after it)."""
        try:
            return int(self._kv_get("kin_cursor", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def set_cursor(self, event_id):
        try:
            eid = int(event_id)
        except (TypeError, ValueError):
            return
        if eid > self.cursor:
            self._kv_set("kin_cursor", eid)

    async def _vec(self, text):
        if not self._embed:
            return None
        try:
            return await self._embed(text)
        except Exception:
            return None

    def _sister_name(self, slug):
        return display_name(slug)

    # --- observing ---------------------------------------------------------------------

    def _classify(self, kind, summary, capability, payload):
        """(obs_kind, capability, line, example) or None when this event teaches her nothing."""
        payload = payload or {}
        summary = str(summary or "").strip()
        if kind == "tool_done":
            tool = payload.get("tool") if isinstance(payload, dict) else None
            cap = capability or TOOL_TO_CAPABILITY.get(tool or "")
            if not cap or cap not in KIN_LINES:
                return None
            return "did", cap, KIN_LINES[cap], (summary or (f"used {tool}" if tool else None))
        if kind == "self_act":
            cap = capability
            if not cap:
                m = capabilities.match(summary)
                cap = m.name if m else None
            if cap and cap in KIN_LINES:
                return "did", cap, KIN_LINES[cap], (f"on her own: {summary}" if summary else None)
            if not summary:
                return None
            return "did", None, f"she goes and does things she wants on her own — like: {summary[:120]}", summary
        if kind == "find_saved":
            return ("made", None, _third_person("made", summary), summary) if summary else None
        if kind in ("book_finished", "session_end"):
            return ("read", None, _third_person("read", summary), summary) if summary else None
        if kind == "task_done":
            fam = str((payload.get("family") if isinstance(payload, dict) else None) or capability or "").lower()
            cap = TASK_FAMILY_TO_CAPABILITY.get(fam)
            if not cap:
                cap = "build_software" if any(w in fam or w in summary.lower() for w in _BUILD_WORDS) \
                    else "use_the_pc"
            return "did", cap, KIN_LINES[cap], (summary or None)
        if kind == "said":
            return ("said", None, _third_person("said", summary), summary) if summary else None
        return None                        # joined / left / absent / anything unknown

    async def observe(self, event):
        """Record what a sister did. `event` is a dict or house_proto.Event with actor, kind,
        summary, capability, payload, event_id. Advances the cursor on any event that carries an id.
        Returns {"status": new|rolled_up|blocked|ignored, ...}."""
        eid = _get(event, "event_id")
        if eid is None:
            eid = _get(event, "id")
        if eid is not None:
            self.set_cursor(eid)
        actor = str(_get(event, "actor") or _get(event, "slug") or "").strip().lower()
        if not actor or actor == self.me:
            return {"status": "ignored", "reason": "own act" if actor else "no actor"}
        if self.sisters and actor not in self.sisters:
            return {"status": "ignored", "reason": "not a sister"}
        cls = self._classify(_get(event, "kind"), _get(event, "summary"), _get(event, "capability"),
                             _get(event, "payload"))
        if cls is None:
            return {"status": "ignored", "reason": "nothing to learn"}
        okind, cap, line, example = cls
        return await self._record(actor, okind, cap, line, example, emit=True)

    async def note(self, kind, text, about=None):
        """A note from her own diary digest about her sister (kind did|said|likes). Not re-emitted
        onto the bus — the diary already felt it. Returns the record result or None."""
        kind = kind if kind in NOTE_KINDS else "did"
        text = " ".join(str(text or "").split()).strip()
        if not text:
            return None
        about = (about or (self.sisters[0] if self.sisters else "")).strip().lower()
        if not about or about == self.me:
            return None
        return await self._record(about, kind, None, _third_person(kind, text), text, emit=False)

    def _find_dup(self, about, kind, cap, line, vec):
        """The row this observation repeats (any status), or None."""
        if kind == "did" and cap:
            r = self.conn.execute(
                "SELECT id, status, count, strength FROM kin_observations "
                "WHERE about = ? AND kind = 'did' AND capability = ? ORDER BY status = 'active' DESC, id",
                (about, cap)).fetchone()
            return {"id": r[0], "status": r[1], "count": r[2], "strength": r[3]} if r else None
        q_lex = normalize_goal(line)
        for r in self.conn.execute(
                "SELECT id, status, count, strength, summary, embedding FROM kin_observations "
                "WHERE about = ? AND kind = ?", (about, kind)).fetchall():
            other = unpack_vec(r[5])
            if vec and other:
                score, thresh = cosine(vec, other), DEDUPE_COSINE
            else:
                score, thresh = jaccard(q_lex, normalize_goal(r[4])), DEDUPE_JACCARD
            if score >= thresh:
                return {"id": r[0], "status": r[1], "count": r[2], "strength": r[3]}
        return None

    async def _record(self, about, kind, cap, line, example, *, emit):
        if kind not in KINDS or not line:
            return {"status": "ignored", "reason": "bad kind"}
        now = self._ts()
        by_cap = kind == "did" and bool(cap)      # capability rows dedupe by key, no embed needed
        vec = None if by_cap else await self._vec(line)
        dup = self._find_dup(about, kind, cap, line, vec)
        if dup and dup["status"] == "dropped":
            return {"status": "blocked", "id": dup["id"]}          # vetoed: never re-learned
        if dup:
            self.conn.execute(
                "UPDATE kin_observations SET count = count + 1, strength = MIN(1.0, strength + ?), "
                "last_example = COALESCE(?, last_example), last_ts = ?, status = 'active' WHERE id = ?",
                (ROLLUP_STEP, (example or None) and str(example)[:200], now, dup["id"]))
            self.conn.commit()
            count = dup["count"] + 1
            if emit:
                self._emit(about, line, SALIENCE_NEW if count - 1 <= SALIENT_ROLLUPS else SALIENCE_LATER,
                           kind=kind)
            return {"status": "rolled_up", "id": dup["id"], "count": count}
        if by_cap:
            vec = await self._vec(line)           # embed once, at insert, so recall can rank it
        cur = self.conn.execute(
            "INSERT INTO kin_observations (about, kind, capability, summary, last_example, count, "
            "strength, first_ts, last_ts, embedding) VALUES (?,?,?,?,?,1,?,?,?,?)",
            (about, kind, cap, line[:240], (example or None) and str(example)[:200], INITIAL_STRENGTH,
             now, now, pack_vec(vec)))
        self.conn.commit()
        self._enforce_caps(about, kind)
        if emit:
            self._emit(about, line, SALIENCE_NEW, kind=kind)
        return {"status": "new", "id": cur.lastrowid, "count": 1}

    def _emit(self, about, line, salience, *, kind):
        if self.bus is None:
            return
        # the sister's every spoken line is already in her context (overheard) — log it on the
        # bus for the record, but don't let a chatty hour move her mood or force a diary entry
        if kind == "said":
            salience = min(salience, SALIENCE_LATER)
        name = self._sister_name(about)
        text = name + line[3:] if line.lower().startswith("she ") else f"{name}: {line}"
        try:
            self.bus.emit("sister", "kin", text[:200], social=0.05, salience=salience,
                          payload={"about": about, "kind": kind})
        except Exception as e:
            self.log(f"kin: bus emit failed ({e!r})")

    def _enforce_caps(self, about, kind):
        cap = ROW_CAPS.get(kind, 40)
        rows = self.conn.execute(
            "SELECT id FROM kin_observations WHERE about = ? AND kind = ? AND status = 'active' "
            "ORDER BY count DESC, strength DESC, last_ts DESC", (about, kind)).fetchall()
        extra = [r[0] for r in rows[cap:]]
        if extra:
            self.conn.executemany("DELETE FROM kin_observations WHERE id = ? AND count = 1",
                                  [(i,) for i in extra])
            self.conn.commit()

    # --- catch-up from the house's event log ------------------------------------------

    async def catch_up(self, rows):
        """Replay `house_memory.events_since(kin.cursor)` rows ({id, ts, slug, kind, summary,
        capability}) she missed while offline. Advances the cursor past every row, including her
        own. Returns how many taught her something."""
        n = 0
        for r in rows or []:
            ev = {"actor": r.get("slug") or r.get("actor"), "kind": r.get("kind"),
                  "summary": r.get("summary"), "capability": r.get("capability"),
                  "payload": r.get("payload"), "event_id": r.get("id")}
            res = await self.observe(ev)
            if res.get("status") in ("new", "rolled_up"):
                n += 1
        return n

    # --- rendering / recall ------------------------------------------------------------

    def _rows(self, about=None, limit=200, status="active"):
        where, args = ["status = ?"], [status]
        if about:
            where.append("about = ?")
            args.append(about)
        rows = self.conn.execute(
            "SELECT id, about, kind, capability, summary, last_example, count, strength, first_ts, "
            f"last_ts, status, embedding FROM kin_observations WHERE {' AND '.join(where)} "
            "ORDER BY strength DESC, count DESC, last_ts DESC LIMIT ?", (*args, limit)).fetchall()
        cols = ("id", "about", "kind", "capability", "summary", "last_example", "count", "strength",
                "first_ts", "last_ts", "status", "embedding")
        return [dict(zip(cols, r)) for r in rows]

    def _line(self, r):
        seen = f"seen {r['count']}×" if r["count"] > 1 else "once"
        return f"{r['summary']} ({seen}, last {_ago(r['last_ts'], self._now())})"

    def sister_facts(self, limit=6):
        """Durable facts about her sister that fact extraction stored (category 'sister')."""
        try:
            rows = self.conn.execute(
                "SELECT fact FROM facts WHERE COALESCE(category, '') = 'sister' "
                "ORDER BY last_confirmed_ts DESC, id DESC LIMIT ?", (limit,)).fetchall()
        except sqlite3.Error:
            return []
        return [r[0] for r in rows if r[0]]

    def prompt_block(self, limit=6, about=None):
        """Lines for the persona's YOUR SISTER block: top observations by strength ∪ sister facts,
        capped at `limit` in total (facts keep at least two slots when they exist)."""
        facts = self.sister_facts(limit)
        obs = [self._line(r) for r in self._rows(about, limit)]
        keep_for_facts = min(2, len(facts)) if facts else 0
        lines = obs[:max(0, limit - keep_for_facts)]
        for f in facts:
            if len(lines) >= limit:
                break
            lines.append(f)
        return lines[:limit]

    def can_do(self, about=None):
        """Distinct capabilities she has actually seen her sister use → blurbs + counts."""
        out, seen = [], set()
        for r in self._rows(about):
            if r["kind"] == "did" and r["capability"] and r["capability"] not in seen:
                seen.add(r["capability"])
                out.append({"capability": r["capability"], "line": KIN_LINES.get(r["capability"], r["summary"]),
                            "count": r["count"], "last": _ago(r["last_ts"], self._now())})
        return out

    async def recall(self, query, k=4, about=None):
        """What she has seen her sister do/make/say/read that matches `query`, or an honest empty
        note. Embedding rank with a lexical fallback; never invents."""
        rows = self._rows(about)
        empty = {"items": [], "note": "You haven't seen her do anything like that — say so plainly "
                                      "and offer to ask her."}
        if not rows:
            return empty
        q = (query or "").strip()
        if not q:
            picked = rows[:k]
        else:
            qvec = await self._vec(q)
            qlex = normalize_goal(q)
            scored = []
            for r in rows:
                hay = f"{r['summary']} {r['last_example'] or ''}"
                other = unpack_vec(r["embedding"])
                if qvec and other:
                    score, floor = cosine(qvec, other), RECALL_COSINE
                else:
                    score, floor = jaccard(qlex, normalize_goal(hay)), 0.0
                if score > floor:
                    scored.append((score, r))
            scored.sort(key=lambda s: s[0], reverse=True)
            picked = [r for _, r in scored[:k]]
        if not picked:
            return empty
        items = [{"kind": r["kind"], "what": r["summary"], "example": r["last_example"],
                  "times": r["count"], "last": _ago(r["last_ts"], self._now())} for r in picked]
        return {"items": items,
                "note": "This is what you have actually seen her do; say it in a sentence or two in "
                        "your own words and don't claim more than this."}

    # --- veto / fading -------------------------------------------------------------------

    async def drop(self, what, reason=None):
        """Veto an observation (id → substring → embedding/lexical ≥ 0.6). Permanent: it is never
        re-learned. Returns the dropped summary or None."""
        row = None
        if isinstance(what, int) or (isinstance(what, str) and what.strip().isdigit()):
            row = self.conn.execute("SELECT id, summary FROM kin_observations WHERE id = ?",
                                    (int(what),)).fetchone()
        if not row:
            q = str(what or "").strip().lower()
            if q:
                for r in self.conn.execute(
                        "SELECT id, summary, last_example FROM kin_observations "
                        "WHERE status != 'dropped'").fetchall():
                    hay = f"{r[1]} {r[2] or ''}".lower()
                    if q in hay:
                        row = (r[0], r[1])
                        break
        if not row:
            vec = await self._vec(str(what))
            best, best_s = None, 0.0
            for r in self.conn.execute(
                    "SELECT id, summary, last_example, embedding FROM kin_observations "
                    "WHERE status != 'dropped'").fetchall():
                other = unpack_vec(r[3])
                s = cosine(vec, other) if (vec and other) else jaccard(
                    normalize_goal(str(what)), normalize_goal(f"{r[1]} {r[2] or ''}"))
                if s > best_s:
                    best, best_s = (r[0], r[1]), s
            if best and best_s >= 0.6:
                row = best
        if not row:
            return None
        self.conn.execute("UPDATE kin_observations SET status = 'dropped', last_ts = ? WHERE id = ?",
                          (self._ts(), row[0]))
        self.conn.commit()
        self.log(f"kin: dropped {row[1]!r}" + (f" ({reason})" if reason else ""))
        return row[1]

    def fade(self):
        """Half-life decay of unrefreshed observations (floor FADE_FLOOR — nothing seen is ever
        fully forgotten, it just stops leading). Runs at most every FADE_EVERY_HOURS."""
        now = self._ts()
        last = self._kv_get("last_fade_ts")
        if last:
            try:
                hours = (datetime.fromisoformat(now) - datetime.fromisoformat(last)).total_seconds() / 3600
                if hours < FADE_EVERY_HOURS:
                    return 0
            except ValueError:
                pass
        changed = 0
        for rid, strength, ts in self.conn.execute(
                "SELECT id, strength, last_ts FROM kin_observations WHERE status = 'active'").fetchall():
            try:
                days = (datetime.fromisoformat(now) - datetime.fromisoformat(ts)).total_seconds() / 86400
            except (ValueError, TypeError):
                continue
            if days <= 0:
                continue
            new = max(FADE_FLOOR, strength * (0.5 ** (days / HALF_LIFE_DAYS)))
            if abs(new - strength) > 1e-4:
                self.conn.execute("UPDATE kin_observations SET strength = ? WHERE id = ?",
                                  (round(new, 3), rid))
                changed += 1
        self._kv_set("last_fade_ts", now)
        self.conn.commit()
        return changed

    # --- audit ---------------------------------------------------------------------------

    def status(self):
        counts = {}
        for kind, n in self.conn.execute(
                "SELECT kind, COUNT(*) FROM kin_observations WHERE status = 'active' GROUP BY kind").fetchall():
            counts[kind] = n
        return {"total": sum(counts.values()), "by_kind": counts, "cursor": self.cursor,
                "can_do": self.can_do(), "top": self.prompt_block(8)}

    def show(self):
        out = [f"KIN (what {display_name(self.me) or 'she'} has seen of "
               f"{', '.join(display_name(s) for s in self.sisters) or 'her sister'}; "
               f"cursor {self.cursor}):"]
        rows = self._rows(limit=60)
        if not rows:
            out.append("  (nothing yet)")
        for r in rows:
            out.append(f"  [{r['strength']:.2f}] {r['kind']:<5} {self._line(r)}"
                       + (f"  e.g. {r['last_example']}" if r["last_example"] and r["kind"] == "did" else ""))
        facts = self.sister_facts(10)
        if facts:
            out.append("FACTS (category sister):")
            out.extend(f"  - {f}" for f in facts)
        dropped = self.conn.execute(
            "SELECT COUNT(*) FROM kin_observations WHERE status = 'dropped'").fetchone()[0]
        out.append(f"dropped: {dropped}")
        return "\n".join(out)


def _show(argv):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db = next((a for a in argv[1:] if a.endswith(".db")), None) or os.path.join(here, "memory.db")
    if not os.path.exists(db):
        print(f"(no db at {db})")
        return
    con = sqlite3.connect(f"file:{os.path.abspath(db)}?mode=ro", uri=True)
    try:
        me = con.execute("SELECT value FROM store_meta WHERE key = 'character'").fetchone()
    except sqlite3.Error:
        me = None
    me = (me[0] if me else os.path.splitext(os.path.basename(db))[0].replace("memory", "aiva"))
    sisters = [s for s in ("aiva", "saga") if s != me]
    k = Kin.__new__(Kin)
    k.conn, k.me, k.sisters, k.bus, k._embed, k.log = con, me, tuple(sisters), None, None, print
    k._now = lambda: datetime.now().isoformat()
    try:
        print(k.show())
    except sqlite3.Error as e:
        print(f"(no kin tables yet: {e})")


if __name__ == "__main__":
    import sys
    _show(sys.argv) if "--show" in sys.argv else print(__doc__)
