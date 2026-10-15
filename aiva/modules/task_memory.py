"""Aiva's task memory: what she has done, how it went, and what the user said.

Three SQLite tables on the SAME connection as `Memory` (aiva/memory.db):
- tasks:                one row per task the runner ran — live state while it
                        runs (status/plan/log/pending) and the learning record
                        afterwards (outcome, grade, feedback, correction flag,
                        embedding). One table = one episode: no duplicate rows.
- flows:                named, replayable step lists that grew out of tasks
                        that went well (or that the user asked to keep)
- approach_preferences: which of several offered approaches the user picked
                        for a task family, and why — so she can see patterns

Everything here runs on the event-loop thread: the sqlite connection is
single-thread (same rule as the `remember` tool). Never call into this from
`asyncio.to_thread` workers.

The pure functions at the bottom (`decide_autonomy`, `non_trivial`,
`always_confirm_reason`) encode the user's autonomy rule: just do what she
has done before, confirm what is new to her or what she was corrected on,
and always confirm irreversible/shell steps.
"""

import json
import math
import re
import sqlite3
import struct
import uuid
from datetime import datetime

from modules.persona import named
from modules.utilities import llm_kwargs

# --- thresholds (logged per task so they can be tuned from aiva/logs) --------
SAME_TASK = 0.86        # cosine: same task, possibly different wording
SIMILAR = 0.72          # same family / shape, likely a different target
LEX_SAME = 0.75         # Jaccard on normalized tokens (fallback when no embedding)
LEX_SIMILAR = 0.45
ROUTINE_SUCCESSES = 2
PATTERN_STREAK = 3
EMBED_DIMS = 512
SCAN_LIMIT = 2000       # newest rows considered for similarity

FAMILIES = ("streaming-setup", "app-launch", "window-layout", "repo-edit", "repo-ops",
            "file-cleanup", "media-control", "system-settings", "web-lookup", "messaging",
            "avatar", "cat")
RITUAL_FAMILIES = {"streaming-setup", "window-layout", "app-launch"}
APPROACHES = ("ui", "terminal", "shell", "claude_code", "files", "browser", "mixed")

LIVE_STATUSES = ("planning", "awaiting_user", "running", "verifying")

_FILLER = {
    "please", "hey", "hi", "um", "uh", "hmm", "ok", "okay", "so", "just", "now", "quickly",
    "kindly", "for", "me", "can", "you", "could", "would", "will", "go", "ahead", "i", "want",
    "need", "like", "the", "a", "an", "to", "my", "and", "then", "also", "some", "this", "that",
}
_NAME_SHAPE = re.compile(r"^[aeiy]{1,3}[vwf]ah?$")


# --- text helpers ----------------------------------------------------------------

def normalize_goal(text, name_variants=()):
    """Lowercase, drop her name / filler / stutters; keep path-ish characters."""
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9/\\.\-_: ]+", " ", text)
    names = {v.lower() for v in name_variants}
    out = []
    for tok in text.split():
        if tok in names or _NAME_SHAPE.match(tok) or tok in _FILLER:
            continue
        if out and out[-1] == tok:
            continue  # stutter: "open open notepad"
        out.append(tok)
    return " ".join(out)


def jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def pack_vec(vec):
    return struct.pack(f"<{len(vec)}f", *vec) if vec else None


def unpack_vec(blob):
    if not blob:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


async def embed_text(client, text, timeout=4.0):
    """text-embedding-3-small (512 dims) through the shared AsyncOpenAI client.
    Returns None on any failure so callers fall back to lexical matching."""
    import asyncio
    try:
        resp = await asyncio.wait_for(
            client.embeddings.create(model="text-embedding-3-small",
                                     input=(text or "")[:1000], dimensions=EMBED_DIMS),
            timeout=timeout)
        return list(resp.data[0].embedding)
    except Exception:
        return None


def _now():
    # microseconds: episodes created in the same second must still order
    return datetime.now().isoformat()


def new_task_id():
    return f"t-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def _ago(ts):
    """'2 days ago' / 'today' from an ISO timestamp — for prompt lines."""
    try:
        days = (datetime.now() - datetime.fromisoformat(ts)).days
    except (TypeError, ValueError):
        return "unknown"
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


# --- the memory ------------------------------------------------------------------

class TaskMemory:
    def __init__(self, conn: sqlite3.Connection, embed=None, name_variants=(), name="Aiva"):
        self.conn = conn
        self._embed = embed              # async callable(text) -> vector | None
        self._names = tuple(name_variants)
        self.name = name
        self.grade_prompt = named(GRADE_PROMPT_T, name)
        self.feedback_prompt = named(FEEDBACK_EXTRACTION_PROMPT_T, name)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                created_ts TEXT NOT NULL,
                updated_ts TEXT NOT NULL,
                finished_ts TEXT,
                goal_text TEXT NOT NULL,
                goal_normalized TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '',
                family TEXT,
                approach TEXT,
                status TEXT NOT NULL DEFAULT 'planning',
                outcome TEXT,
                plan_json TEXT NOT NULL DEFAULT '[]',
                log_json TEXT NOT NULL DEFAULT '[]',
                pending_json TEXT,
                result_summary TEXT,
                duration_s REAL,
                delegated_to TEXT NOT NULL DEFAULT 'none',
                agent_grade TEXT,
                agent_grade_reason TEXT,
                user_feedback TEXT,
                corrected INTEGER NOT NULL DEFAULT 0,
                flow_id INTEGER,
                embedding BLOB,
                llm_calls INTEGER NOT NULL DEFAULT 0,
                tool_calls INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0,
                self_lesson TEXT
            )""")
        # self_lesson (2026-10-13): what she worked out herself when a step went wrong — added to
        # existing dbs in place; NULL on every older row
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(tasks)")}
        if "self_lesson" not in have:
            self.conn.execute("ALTER TABLE tasks ADD COLUMN self_lesson TEXT")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_family ON tasks(family)")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                trigger_phrases TEXT NOT NULL DEFAULT '[]',
                steps TEXT NOT NULL,
                family TEXT,
                replay TEXT NOT NULL DEFAULT 'exact',
                source_task_id TEXT,
                run_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                last_run_ts TEXT,
                confirmed_by_user INTEGER NOT NULL DEFAULT 0,
                created_ts TEXT NOT NULL,
                embedding BLOB
            )""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS approach_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                family TEXT NOT NULL,
                chosen_approach TEXT NOT NULL,
                rejected_approaches TEXT NOT NULL DEFAULT '[]',
                user_reason TEXT,
                task_id TEXT,
                ts TEXT NOT NULL
            )""")
        self.conn.commit()

    # --- rows -> dicts ---------------------------------------------------------

    _TASK_COLS = ("rowid", "id", "created_ts", "updated_ts", "finished_ts", "goal_text", "goal_normalized",
                  "details", "family", "approach", "status", "outcome", "plan_json", "log_json",
                  "pending_json", "result_summary", "duration_s", "delegated_to", "agent_grade",
                  "agent_grade_reason", "user_feedback", "corrected", "flow_id", "embedding",
                  "llm_calls", "tool_calls", "cost_usd", "self_lesson")

    def _task_row(self, row):
        d = dict(zip(self._TASK_COLS, row))
        d["plan"] = json.loads(d.pop("plan_json") or "[]")
        d["log"] = json.loads(d.pop("log_json") or "[]")
        d["pending"] = json.loads(d["pending_json"]) if d.get("pending_json") else None
        d.pop("pending_json", None)
        d["corrected"] = bool(d["corrected"])
        return d

    def _select_tasks(self, where="", params=(), limit=SCAN_LIMIT):
        cols = ", ".join(self._TASK_COLS)
        rows = self.conn.execute(
            f"SELECT {cols} FROM tasks {where} ORDER BY created_ts DESC, rowid DESC LIMIT ?",
            (*params, limit)).fetchall()
        return [self._task_row(r) for r in rows]

    _FLOW_COLS = ("id", "name", "trigger_phrases", "steps", "family", "replay", "source_task_id",
                  "run_count", "success_count", "last_run_ts", "confirmed_by_user", "created_ts",
                  "embedding")

    def _flow_row(self, row):
        d = dict(zip(self._FLOW_COLS, row))
        d["trigger_phrases"] = json.loads(d["trigger_phrases"] or "[]")
        d["steps"] = json.loads(d["steps"] or "[]")
        d["confirmed_by_user"] = bool(d["confirmed_by_user"])
        return d

    def flows(self):
        cols = ", ".join(self._FLOW_COLS)
        rows = self.conn.execute(f"SELECT {cols} FROM flows ORDER BY created_ts, id").fetchall()
        return [self._flow_row(r) for r in rows]

    # --- embeddings -----------------------------------------------------------

    async def embed(self, text):
        if self._embed is None:
            return None
        try:
            return await self._embed(text)
        except Exception:
            return None

    # --- matching ---------------------------------------------------------------

    async def find_similar(self, goal, k=3, vec=None):
        """[(episode, score)] newest-first ties; returns finished episodes only.
        Sets self.last_method to 'embedding' or 'lexical' for logging."""
        norm = normalize_goal(goal, self._names)
        if vec is None:
            vec = await self.embed(goal)
        candidates = self._select_tasks("WHERE outcome IS NOT NULL")
        scored = []
        if vec:
            self.last_method = "embedding"
            for ep in candidates:
                other = unpack_vec(ep.get("embedding"))
                if other:
                    scored.append((ep, cosine(vec, other)))
                else:  # older rows without an embedding: lexical, scaled to cosine-ish range
                    scored.append((ep, self._lex_as_cosine(norm, ep["goal_normalized"])))
        else:
            self.last_method = "lexical"
            for ep in candidates:
                scored.append((ep, jaccard(norm, ep["goal_normalized"])))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    @staticmethod
    def _lex_as_cosine(a, b):
        j = jaccard(a, b)
        if j >= LEX_SAME:
            return SAME_TASK + (1 - SAME_TASK) * min(1.0, (j - LEX_SAME) / (1 - LEX_SAME))
        if j >= LEX_SIMILAR:
            return SIMILAR + (SAME_TASK - SIMILAR) * (j - LEX_SIMILAR) / (LEX_SAME - LEX_SIMILAR)
        return j * SIMILAR / LEX_SIMILAR if LEX_SIMILAR else 0.0

    def _thresholds(self):
        if getattr(self, "last_method", "lexical") == "embedding":
            return SAME_TASK, SIMILAR
        return LEX_SAME, LEX_SIMILAR

    async def familiarity(self, goal, vec=None):
        """How well she knows this task. Levels: routine, seen, attempted,
        similar, novel. A flow trigger hit is routine unless corrected."""
        flow = self.find_flow(goal)
        matches = await self.find_similar(goal, k=5, vec=vec)
        same_thr, sim_thr = self._thresholds()
        same = [(ep, s) for ep, s in matches if s >= same_thr]
        similar = [(ep, s) for ep, s in matches if sim_thr <= s < same_thr]
        result = {
            "level": "novel", "method": getattr(self, "last_method", "lexical"),
            "score": round(matches[0][1], 3) if matches else 0.0,
            "best": matches[0][0] if matches else None,
            "matches": matches, "successes": 0, "corrected": False,
            "last_correction": None, "last_failure": None, "flow": flow,
        }
        pool = same or similar
        if pool:
            eps = [ep for ep, _ in pool]
            successes = [ep for ep in eps if ep["outcome"] == "success"]
            corrections = [ep for ep in eps if ep["corrected"]]
            failures = [ep for ep in eps if ep["outcome"] in ("failed", "partial", "cancelled")]
            result["successes"] = len(successes)
            if corrections:
                order = lambda ep: (ep["created_ts"], ep.get("rowid", 0))
                newest_corr = max(order(ep) for ep in corrections)
                clean_after = [ep for ep in successes
                               if not ep["corrected"] and order(ep) > newest_corr]
                if not clean_after:
                    result["corrected"] = True
                    result["last_correction"] = (corrections[0].get("user_feedback") or "").strip() or None
            if failures:
                result["last_failure"] = failures[0].get("result_summary") or "an earlier attempt"
        if flow and not (flow.get("_corrected")):
            result["level"] = "routine"
        elif same:
            if result["successes"] >= ROUTINE_SUCCESSES and not result["corrected"]:
                result["level"] = "routine"
            elif result["successes"] >= 1:
                result["level"] = "seen"
            else:
                result["level"] = "attempted"
        elif similar:
            result["level"] = "similar"
        if flow and flow.get("_corrected"):
            result["level"] = "seen"
            result["corrected"] = True
        return result

    def find_flow(self, phrase):
        """Exact/contains match on flow name or trigger phrases (normalized)."""
        norm = normalize_goal(phrase, self._names)
        if not norm:
            return None
        for flow in self.flows():
            keys = [normalize_goal(flow["name"], self._names)] + \
                   [normalize_goal(t, self._names) for t in flow["trigger_phrases"]]
            for key in keys:
                if key and (key == norm or f" {key} " in f" {norm} " or f" {norm} " in f" {key} "):
                    flow["_corrected"] = self._flow_corrected(flow["id"])
                    return flow
        return None

    async def find_flow_fuzzy(self, phrase):
        vec = await self.embed(phrase)
        if not vec:
            return None
        best = None
        for flow in self.flows():
            other = unpack_vec(flow.get("embedding"))
            if other:
                s = cosine(vec, other)
                if best is None or s > best[1]:
                    best = (flow, s)
        if best and best[1] >= SIMILAR:
            best[0]["_corrected"] = self._flow_corrected(best[0]["id"])
            return best
        return None

    def _flow_corrected(self, flow_id):
        row = self.conn.execute(
            "SELECT corrected FROM tasks WHERE flow_id = ? AND outcome IS NOT NULL "
            "ORDER BY created_ts DESC, rowid DESC LIMIT 1", (flow_id,)).fetchone()
        return bool(row and row[0])

    # --- episode lifecycle -----------------------------------------------------

    def record_episode(self, task_id, goal_text, details="", family=None, approach=None,
                       plan_steps=None, *, status="planning", delegated_to="none",
                       flow_id=None, embedding=None):
        now = _now()
        self.conn.execute(
            """INSERT INTO tasks (id, created_ts, updated_ts, goal_text, goal_normalized, details,
                                  family, approach, status, plan_json, delegated_to, flow_id, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (task_id, now, now, goal_text, normalize_goal(goal_text, self._names), details or "",
             family, approach, status, json.dumps(plan_steps or [], ensure_ascii=False),
             delegated_to, flow_id, pack_vec(embedding) if embedding else None))
        self.conn.commit()
        return task_id

    def update_task(self, task_id, **fields):
        """Live-state update: status, plan, log, pending, family, approach,
        delegated_to, llm_calls, tool_calls, cost_usd, embedding."""
        cols, vals = ["updated_ts = ?"], [_now()]
        for key, val in fields.items():
            if key == "plan":
                cols.append("plan_json = ?"); vals.append(json.dumps(val, ensure_ascii=False))
            elif key == "log":
                cols.append("log_json = ?"); vals.append(json.dumps(val, ensure_ascii=False))
            elif key == "pending":
                cols.append("pending_json = ?")
                vals.append(json.dumps(val, ensure_ascii=False) if val is not None else None)
            elif key == "embedding":
                cols.append("embedding = ?"); vals.append(pack_vec(val) if val else None)
            elif key in ("status", "family", "approach", "delegated_to", "llm_calls",
                         "tool_calls", "cost_usd", "flow_id", "details", "self_lesson"):
                cols.append(f"{key} = ?"); vals.append(val)
        vals.append(task_id)
        self.conn.execute(f"UPDATE tasks SET {', '.join(cols)} WHERE id = ?", vals)
        self.conn.commit()

    def finish_episode(self, task_id, outcome, result_summary, duration_s, *,
                       agent_grade=None, agent_grade_reason=None):
        status = {"success": "done", "partial": "done", "failed": "failed",
                  "cancelled": "cancelled"}.get(outcome, outcome)
        now = _now()
        self.conn.execute(
            """UPDATE tasks SET status = ?, outcome = ?, result_summary = ?, duration_s = ?,
                                agent_grade = ?, agent_grade_reason = ?, finished_ts = ?,
                                updated_ts = ?, pending_json = NULL
               WHERE id = ?""",
            (status, outcome, result_summary, duration_s, agent_grade, agent_grade_reason,
             now, now, task_id))
        self.conn.commit()

    def get_task(self, task_id):
        rows = self._select_tasks("WHERE id = ?", (task_id,), limit=1)
        return rows[0] if rows else None

    def recent_tasks(self, limit=5):
        return self._select_tasks(limit=limit)

    def add_feedback(self, task_id, text, is_correction):
        row = self.conn.execute("SELECT user_feedback, corrected FROM tasks WHERE id = ?",
                                (task_id,)).fetchone()
        if not row:
            return False
        existing = (row[0] or "").strip()
        merged = f"{existing}\n{text.strip()}".strip() if existing else text.strip()
        self.conn.execute(
            "UPDATE tasks SET user_feedback = ?, corrected = ?, updated_ts = ? WHERE id = ?",
            (merged, 1 if (is_correction or row[1]) else 0, _now(), task_id))
        self.conn.commit()
        return True

    def last_episode(self, within_s=7200, finished_only=True):
        rows = self._select_tasks(
            "WHERE outcome IS NOT NULL" if finished_only else "", limit=1)
        if not rows:
            return None
        ep = rows[0]
        ref = ep.get("finished_ts") or ep.get("updated_ts")
        try:
            age = (datetime.now() - datetime.fromisoformat(ref)).total_seconds()
        except (TypeError, ValueError):
            age = 0
        return ep if age <= within_s else None

    def reconcile_stale(self):
        """At boot: tasks left mid-flight by a restart -> interrupted."""
        marks = ",".join("?" * len(LIVE_STATUSES))
        stale = self._select_tasks(f"WHERE status IN ({marks})", LIVE_STATUSES)
        now = _now()
        for ep in stale:
            self.conn.execute(
                "UPDATE tasks SET status = 'interrupted', outcome = 'cancelled', "
                "result_summary = COALESCE(result_summary, 'lost at shutdown'), "
                "finished_ts = ?, updated_ts = ?, pending_json = NULL WHERE id = ?",
                (now, now, ep["id"]))
        if stale:
            self.conn.commit()
        return stale

    # --- flows ------------------------------------------------------------------

    def flow_candidate(self, task_id):
        """Should she offer to keep this task as a flow? Returns a suggestion or None."""
        ep = self.get_task(task_id)
        if not ep or ep["outcome"] != "success" or ep.get("flow_id"):
            return None
        steps = [s for s in ep["plan"] if s.get("status") != "skipped"]
        if len(steps) < 2:
            return None
        if ep.get("delegated_to", "none") != "none" and len(steps) <= 2:
            return None
        norm = ep["goal_normalized"]
        for flow in self.flows():
            if jaccard(norm, normalize_goal(flow["name"], self._names)) >= LEX_SAME:
                return None
        same_successes = [
            other for other in self._select_tasks("WHERE outcome = 'success' AND id != ?", (task_id,))
            if jaccard(norm, other["goal_normalized"]) >= LEX_SAME
        ]
        ritual = (ep.get("family") or "") in RITUAL_FAMILIES
        if not same_successes and not ritual:
            return None
        name = " ".join(norm.split()[:4]) or "that task"
        return {"suggest": True, "name": name, "triggers": [norm], "task_id": task_id}

    def save_flow(self, name, steps, trigger_phrases=None, family=None, source_task_id=None,
                  *, replay="exact", confirmed_by_user=True, embedding=None):
        name = " ".join((name or "").strip().split())
        if not name:
            raise ValueError("a flow needs a name")
        clean_steps = []
        for s in steps or []:
            s = dict(s)
            for k in ("status", "attempts", "evidence", "started", "finished"):
                s.pop(k, None)
            clean_steps.append(s)
        triggers = list(dict.fromkeys((trigger_phrases or []) + [name]))
        self.conn.execute(
            """INSERT INTO flows (name, trigger_phrases, steps, family, replay, source_task_id,
                                  confirmed_by_user, created_ts, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET trigger_phrases = excluded.trigger_phrases,
                   steps = excluded.steps, family = excluded.family, replay = excluded.replay,
                   source_task_id = excluded.source_task_id,
                   confirmed_by_user = excluded.confirmed_by_user,
                   embedding = COALESCE(excluded.embedding, flows.embedding)""",
            (name, json.dumps(triggers, ensure_ascii=False), json.dumps(clean_steps, ensure_ascii=False),
             family, replay, source_task_id, 1 if confirmed_by_user else 0, _now(),
             pack_vec(embedding) if embedding else None))
        flow_id = self.conn.execute("SELECT id FROM flows WHERE name = ?", (name,)).fetchone()[0]
        if source_task_id:
            self.conn.execute("UPDATE tasks SET flow_id = ? WHERE id = ?", (flow_id, source_task_id))
        self.conn.commit()
        return flow_id

    def get_flow(self, name_or_id):
        cols = ", ".join(self._FLOW_COLS)
        if isinstance(name_or_id, int):
            row = self.conn.execute(f"SELECT {cols} FROM flows WHERE id = ?", (name_or_id,)).fetchone()
        else:
            row = self.conn.execute(f"SELECT {cols} FROM flows WHERE lower(name) = lower(?)",
                                    (str(name_or_id).strip(),)).fetchone()
        return self._flow_row(row) if row else None

    def delete_flow(self, name):
        cur = self.conn.execute("DELETE FROM flows WHERE lower(name) = lower(?)", (name.strip(),))
        self.conn.commit()
        return cur.rowcount > 0

    def bump_flow_stats(self, flow_id, success):
        self.conn.execute(
            "UPDATE flows SET run_count = run_count + 1, success_count = success_count + ?, "
            "last_run_ts = ? WHERE id = ?", (1 if success else 0, _now(), flow_id))
        self.conn.commit()

    def load_flows(self, limit=30):
        """Prompt-ready lines: name, triggers, step titles, stats."""
        lines = []
        for flow in self.flows()[:limit]:
            triggers = [t for t in flow["trigger_phrases"] if t.lower() != flow["name"].lower()][:2]
            say = ", ".join((s.get("title") or s.get("say") or "?") for s in flow["steps"])
            if len(say) > 90:
                say = say[:87] + "..."
            trig = f' (say: {", ".join(repr(t) for t in triggers)})' if triggers else ""
            runs = f"{flow['run_count']} run{'s' if flow['run_count'] != 1 else ''}"
            last = f", last {_ago(flow['last_run_ts'])}" if flow.get("last_run_ts") else ""
            lines.append(f"{flow['name']}{trig}: {say} — {runs}{last}")
        return lines

    # --- preferences / delegation -------------------------------------------

    def record_preference(self, family, chosen, rejected, reason, task_id=None, remember=None):
        """Store the user's pick. At PATTERN_STREAK consistent picks, persist a
        durable preference fact through `remember` (returns 1 only when new,
        so the spoken 'I noticed a pattern' line happens exactly once)."""
        self.conn.execute(
            "INSERT INTO approach_preferences (family, chosen_approach, rejected_approaches, "
            "user_reason, task_id, ts) VALUES (?, ?, ?, ?, ?, ?)",
            (family or "other", chosen, json.dumps(list(rejected or []), ensure_ascii=False),
             reason, task_id, _now()))
        self.conn.commit()
        pref = self.preference_for(family) or {"streak": 1, "reasons": []}
        out = {"streak": pref["streak"], "pattern_new": False, "pattern_line": None}
        if pref["streak"] >= PATTERN_STREAK and remember is not None:
            gist = next((r for r in pref["reasons"] if r), None)
            fact = (f"For {family or 'this kind of'} tasks, default to the {chosen} approach; "
                    f"the user chose it {pref['streak']} times"
                    + (f" ({gist})" if gist else "") + ".")
            added = remember(fact, "preference")
            if added:
                out["pattern_new"] = True
                out["pattern_line"] = (
                    f"You've gone with the {chosen} route {pref['streak']} times for "
                    f"{family or 'this kind of'} work, so I'll default to it from now on "
                    f"unless you say otherwise.")
        return out

    def preference_for(self, family):
        rows = self.conn.execute(
            "SELECT chosen_approach, user_reason FROM approach_preferences WHERE family = ? "
            "ORDER BY ts DESC, id DESC LIMIT 10", (family or "other",)).fetchall()
        if not rows:
            return None
        streak = 0
        for chosen, _ in rows:
            if chosen != rows[0][0]:
                break
            streak += 1
        return {"approach": rows[0][0], "streak": streak, "total": len(rows),
                "reasons": [r for _, r in rows if r]}

    def delegation_record(self, family):
        rows = self.conn.execute(
            "SELECT agent_grade, agent_grade_reason FROM tasks WHERE family = ? AND "
            "delegated_to != 'none' AND agent_grade IS NOT NULL ORDER BY created_ts DESC, rowid DESC LIMIT 20",
            (family or "other",)).fetchall()
        grades = [g for g, _ in rows]
        return {
            "good": grades.count("good"), "partial": grades.count("partial"),
            "bad": grades.count("bad"),
            "last_bad_reason": next((r for g, r in rows if g == "bad"), None),
            "last_two_bad": len(grades) >= 2 and grades[0] == "bad" and grades[1] == "bad",
        }

    # --- prompt + shutdown ------------------------------------------------------

    def load_lessons(self, limit=5, self_limit=3):
        """Prompt lines: the user's corrections first, then what she worked out herself when a
        step went wrong (marked as such — his word outranks her own reading of a report)."""
        rows = self._select_tasks("WHERE corrected = 1 AND user_feedback IS NOT NULL", limit=limit)
        lines = []
        for ep in rows:
            fb = (ep["user_feedback"] or "").strip().splitlines()
            gist = fb[-1][:120] if fb else ""
            lines.append(f"{ep['created_ts'][:10]}, \"{ep['goal_text'][:60]}\": {gist}")
        for ep in self._select_tasks("WHERE self_lesson IS NOT NULL AND self_lesson != ''", limit=self_limit):
            lines.append(f"{ep['created_ts'][:10]}, \"{ep['goal_text'][:60]}\" (you found this yourself): "
                         f"{ep['self_lesson'][:160]}")
        return lines

    async def extract_feedback(self, openai_client, model, session_task_ids, messages, remember=None):
        """Shutdown pass: attach reactions the LLM never recorded to this
        session's episodes. Returns the number of episodes updated."""
        if not session_task_ids:
            return 0
        episodes = [self.get_task(tid) for tid in session_task_ids]
        episodes = [e for e in episodes if e and not (e.get("user_feedback") or "").strip()]
        if not episodes:
            return 0
        transcript = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str))
        if len(transcript) < 80:
            return 0
        ep_block = "\n".join(f"- id {e['id']} at {e['created_ts'][11:16]}: {e['goal_text']} "
                             f"(outcome {e['outcome']})" for e in episodes)
        resp = await openai_client.chat.completions.create(
            model=model, **llm_kwargs(model, max_tokens=400, temperature=0.0),
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": self.feedback_prompt},
                      {"role": "user", "content": f"Tasks:\n{ep_block}\n\nConversation:\n{transcript[-8000:]}"}])
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return 0
        updated = 0
        for item in data.get("feedback", []):
            tid = item.get("task_id") or item.get("episode_id")
            text = (item.get("feedback") or "").strip()
            if tid and text and self.add_feedback(str(tid), text, bool(item.get("is_correction"))):
                updated += 1
                rule = (item.get("general_rule") or "").strip()
                if rule and remember is not None:
                    remember(rule, "instruction")
        return updated

    async def grade_agent_work(self, openai_client, model, goal, evidence, step=None):
        """Grade a delegated job from evidence. Deterministic rules first.

        `step` ({title, expect, prompt}) is what THIS job was asked for: an inspect step is graded
        on its report, not on the whole task's goal. With `evidence["agent_changed"]` (the files
        the job itself touched — see TaskRunner's worktree snapshot) the user's own uncommitted
        work is never mistaken for the agent's. Returns {grade, reason, blocked, question}:
        blocked is "question" (the agent stopped to ask) or "premise" (it reports that something
        the request took for granted isn't so) — neither is a botched job, they need a rethink."""
        status = (evidence.get("job_status") or "").lower()
        mode = evidence.get("mode", "execute")
        changed = evidence.get("agent_changed")
        diff = (evidence.get("diff_stat") or "").strip() if changed is None else "\n".join(changed)
        tests = evidence.get("tests") or {}
        result_text = evidence.get("result_text") or ""
        if status.startswith("failed") or status == "killed":
            return _graded("bad", f"the job {status} before finishing")
        # asked to change things, changed nothing, and ends on a question: it's waiting on an
        # answer. (An inspect report closing on "want me to go ahead?" is just good manners —
        # that one is left to the model grader.)
        question = agent_question(result_text) if mode == "execute" else None
        if question and not diff:
            return _graded("partial", "it stopped to ask a question before changing anything",
                           blocked="question", question=question)
        if mode == "execute" and evidence.get("diff_checked") and not diff:
            return _graded("bad", "it says it's done but nothing in the project changed")
        cap = None
        # tests SHE ran (TaskRunner): failing now caps the grade, unless they already failed before
        # the job — then it's his breakage, and the model grader weighs the two runs
        was_failing = (evidence.get("tests_before") or {}).get("returncode") not in (None, 0)
        if tests and tests.get("returncode") not in (None, 0) and not was_failing:
            cap = "partial"
        if openai_client is None:
            grade = cap or "good"
            reason = "tests are failing" if cap else ("the report is in" if mode == "inspect"
                                                      else "the change is in place")
            return _graded(grade, reason)
        summary = json.dumps({k: (str(v)[:4000] if k == "result_text" else str(v)[:1500])
                              if v is not None else None for k, v in evidence.items()}, ensure_ascii=False)
        asked = f"\nThis step: {json.dumps(step, ensure_ascii=False)}" if step else ""
        try:
            resp = await openai_client.chat.completions.create(
                model=model, **llm_kwargs(model, max_tokens=200, temperature=0.0),
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": self.grade_prompt},
                          {"role": "user", "content": f"Whole task: {goal}{asked}\nEvidence: {summary}"}])
            data = json.loads(resp.choices[0].message.content or "{}")
            grade = data.get("grade") if data.get("grade") in ("good", "partial", "bad") else "partial"
            reason = (data.get("reason") or "the evidence was unclear").strip()
            blocked = data.get("blocked") if data.get("blocked") in ("question", "premise") else None
            q = (data.get("question") or "").strip() or None
        except Exception as e:
            grade, reason, blocked, q = "partial", f"I couldn't finish checking it ({e})", None, None
        if cap == "partial" and grade == "good":
            grade, reason = "partial", reason + ", but tests are failing"
        # blocked is NOT a bad job (it was right to stop) — the runner treats it as not done, and the
        # grade stays honest so delegation_record never blames Claude Code for her wrong premise
        return _graded(grade, reason, blocked=blocked, question=q)


# --- prompts ---------------------------------------------------------------------

# {name}: the character whose tasks these are; TaskMemory renders once per character
GRADE_PROMPT_T = """You audit an autonomous coding agent for {name}, a voice assistant. You get the \
whole task, the ONE step this job was asked to do, the agent's own report, the files the agent itself \
changed (agent_changed; the project may also hold the user's own older uncommitted work, which is NOT \
the agent's doing), and the project's tests as {name} ran them herself after the job ("tests"; \
"tests_before" is the same run from before the job, so failures already there are not the agent's). \
Trust those runs over what the report says about tests. Grade THIS step against its own expectation, not the whole \
task: an inspect or review step is good when its report answers what was asked, with no edits. The \
step's later_steps are what the plan does after this job (e.g. writing the findings into a note): \
never mark this step down for work that belongs to one of them. \
good = the evidence matches the claim and nothing unrelated changed; partial = did some of it, \
missed part, or left tests failing; bad = claims done but the evidence contradicts it or it changed \
the wrong things. Separately, set "blocked": "question" if the agent stopped to ask something it \
needs answered before going on, "premise" if it reports that something the request took for granted \
does not exist or is not so, else null; when blocked, "question" holds what it needs to know. \
Reply with JSON only: {"grade": "good|partial|bad", "reason": "<one spoken sentence citing the \
evidence>", "blocked": "question|premise|null", "question": "<or null>"}."""

FEEDBACK_EXTRACTION_PROMPT_T = """You maintain {name}'s task memory. From the conversation transcript, \
find the user's reactions to the listed tasks {name} did — praise, corrections ("no, that's wrong", \
"you shouldn't have..."), or advice for next time. The transcript comes from speech recognition \
and contains mishearings; be skeptical and extract nothing when unsure. Only cover tasks in the \
list. Reply with JSON only: {"feedback": [{"task_id": "...", "feedback": "<one clean sentence>", \
"is_correction": true|false, "general_rule": "<optional self-contained standing instruction>"}]} \
or {"feedback": []}."""

GRADE_PROMPT = named(GRADE_PROMPT_T, "Aiva")
FEEDBACK_EXTRACTION_PROMPT = named(FEEDBACK_EXTRACTION_PROMPT_T, "Aiva")


# --- reading an agent's report (pure) -------------------------------------------------

def _graded(grade, reason, blocked=None, question=None):
    return {"grade": grade, "reason": reason, "blocked": blocked, "question": question}


_MD = re.compile(r"[*_`#>]+")


def agent_question(report):
    """The question an agent's report ENDS on ("Which one did you mean?"), or None. Only the
    closing line counts: a question mid-report is usually rhetorical."""
    lines = [_MD.sub("", ln).strip() for ln in (report or "").strip().splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines or not lines[-1].endswith("?"):
        return None
    return lines[-1][:240]


def clip_middle(text, limit=3000):
    """Keep the START and the end of a long report: an agent states what it found first and asks
    last — keeping only the tail once dropped "there is no Miso renderer" and kept the question."""
    text = text or ""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head].rstrip() + "\n[…]\n" + text[-(limit - head):].lstrip()


def _squash(text):
    return re.sub(r"[\s`*_\"'“”‘’]+", " ", (text or "").lower()).strip()


def found_in(evidence, report, min_len=8):
    """True when `evidence` (a short quote a belief rests on) really appears in `report`, ignoring
    case, whitespace, quotes and markdown — the same verbatim rule her reading quotes obey. What she
    learns from a report must be something the report SAYS, never her paraphrase of it."""
    ev = _squash(evidence)
    return len(ev) >= min_len and ev in _squash(report)


def parse_porcelain(text):
    """`git status --porcelain` lines -> the set of paths listed (rename targets for renames)."""
    paths = set()
    for line in (text or "").splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path.strip('"'))
    return paths


# --- pure decision helpers -----------------------------------------------------

def non_trivial(plan):
    steps = plan.get("steps", [])
    return (len(steps) >= 3
            or any(s.get("kind") == "delegate" for s in steps)
            or any(s.get("risk") == "irreversible" for s in steps)
            or (plan.get("est_minutes") or 0) > 2)


def always_confirm_reason(plan):
    """First reason a plan must be confirmed regardless of familiarity."""
    for s in plan.get("steps", []):
        if s.get("risk") == "irreversible":
            return f"can't be undone ({s.get('title', 'one step')})"
        if s.get("kind") == "shell":
            return "runs a shell command"
        tool = ((s.get("action") or {}).get("tool") or "").lower()
        if s.get("kind") == "message" or "message" in tool or "send" in tool or "pay" in tool:
            return "sends something on your behalf"
    return None


_DELEGATE_WORDS = re.compile(
    r"\b(code|coding|bug|fix|refactor|feature|function|class|module|test|tests|repo|repository|"
    r"commit|implement|script|compile|build error|unit test|pull request|merge)\b", re.I)
_PATH_SHAPE = re.compile(r"([A-Za-z]:\\|/|~/|\.py\b|\.ts\b|\.js\b|\.cs\b)")


def suggest_delegation(goal, plan=None):
    """Cheap sanity check on the planner's delegate flag (never authoritative)."""
    text = f"{goal} {plan.get('summary', '') if plan else ''}"
    coding = bool(_DELEGATE_WORDS.search(text))
    pathish = bool(_PATH_SHAPE.search(text)) or "repo" in text.lower()
    if coding and pathish:
        return {"delegate": True, "why": "coding work inside a project"}
    if coding:
        return {"delegate": True, "why": "coding work"}
    return {"delegate": False, "why": "desktop or terminal work she can do herself"}


def _gist(text, n=90):
    text = (text or "").strip().splitlines()
    return (text[-1][:n] if text else "").rstrip(".")


def _mid(text):
    """Lower-case the first letter so a summary reads naturally mid-sentence."""
    text = (text or "").strip()
    return text[:1].lower() + text[1:] if text and not text[:2].isupper() else text


def free_range_blocker(plan):
    """What still needs a yes under free range: acting on his behalf outside the PC."""
    for s in plan.get("steps", []):
        tool = ((s.get("action") or {}).get("tool") or "").lower()
        if s.get("kind") == "message" or "message" in tool or "send" in tool or "pay" in tool:
            return "sends something on your behalf"
    return None


def decide_autonomy(goal, plan, fam, pref=None, delegation=None, confirm_first=False, free_range=False):
    """The user's rule, made explicit. Returns
    {mode: just_do|confirm_plan|offer_options, why, say, lesson, approach_hint, must_confirm_steps}.
    The voice LLM may be MORE careful than `mode`, never less.

    `free_range` (user decision 2026-09-22, AIVA_TASK_FREE_RANGE=1): he wants her to just code —
    novelty, past failures, corrections, shell, delegate and irreversible steps no longer ask.
    Only `free_range_blocker` (sends/pays on his behalf) and an explicit confirm_first still do."""
    if free_range and not confirm_first:
        summary = _mid((plan.get("summary") or goal or "that").rstrip("."))
        lesson = _gist(fam.get("last_correction")) or None
        reason = free_range_blocker(plan)
        if reason:
            return {"mode": "confirm_plan", "why": f"this {reason}",
                    "say": f"This {reason}, so I want a clear yes: {summary}?", "lesson": lesson,
                    "approach_hint": None, "must_confirm_steps": []}
        say = f"On it — {summary}."
        if lesson:
            say = say.rstrip(".") + f", and I'll keep in mind: {lesson}."
        return {"mode": "just_do", "why": "free range", "say": say, "lesson": lesson,
                "approach_hint": None, "must_confirm_steps": []}
    steps = plan.get("steps", [])
    summary = _mid((plan.get("summary") or goal or "that").rstrip("."))
    level = fam.get("level", "novel")
    corrected = bool(fam.get("corrected"))
    best = fam.get("best") or {}
    must = [s.get("n") for s in steps
            if s.get("risk") == "irreversible" or s.get("kind") in ("shell", "message")]
    lesson = _gist(fam.get("last_correction")) or None
    alternatives = plan.get("alternatives") or []
    approach_hint = None

    if corrected:
        mode, why = "confirm_plan", "you were corrected on this kind of task before"
        say = (f"Last time you told me {lesson}, so this time I'll {summary}. Go ahead?"
               if lesson else f"You corrected me on this before, so let me check: I'd {summary}. Go ahead?")
    elif level == "routine":
        mode, why, say = "just_do", "routine for you", f"On it — {summary}."
    elif level == "seen":
        mode, why = "just_do", "done before"
        say = f"Doing it the way that worked last time: {summary}."
    elif level == "similar":
        mode, why = "just_do", "same kind of task as before"
        prev = _gist(best.get("goal_text"), 50)
        say = f"Like {prev} but for {summary} — starting now." if prev else f"On it — {summary}."
    elif level == "attempted":
        mode, why = "confirm_plan", "an earlier attempt didn't succeed"
        fail = _mid(_gist(fam.get("last_failure"), 60)) or "it didn't work out"
        say = f"I tried this before and it didn't work out — {fail}. This time I'd {summary}. Want me to?"
    else:  # novel
        if pref and pref.get("streak", 0) >= PATTERN_STREAK:
            mode, why = "confirm_plan", "new task, but you have a standing preference"
            approach_hint = pref["approach"]
            say = f"You usually want the {approach_hint} route for this, so I'd {summary}. Okay?"
        elif plan.get("options_distinct") and len(alternatives) >= 2 and non_trivial(plan):
            mode, why = "offer_options", "new and there are genuinely different ways"
            a, b = alternatives[0], alternatives[1]
            say = (f"Haven't done this before, and I see two ways: {a.get('summary', a.get('label'))}, "
                   f"which is {a.get('tradeoff', 'one option')}, or {b.get('summary', b.get('label'))}, "
                   f"which is {b.get('tradeoff', 'the other')}. Which do you like, and why?")
        else:
            mode, why, say = "confirm_plan", "new task for her", f"New one for me. I'd {summary}, okay?"

    # escalations only — nothing below may loosen the mode
    if delegation and delegation.get("last_two_bad") and any(s.get("kind") == "delegate" for s in steps):
        mode, why = "confirm_plan", "Claude Code did badly on this kind of task twice"
        say = (f"Claude Code botched this kind of thing twice, so I'd rather {summary} myself. "
               f"Or should I still hand it over?")
    reason = always_confirm_reason(plan)
    if reason and mode == "just_do":
        mode, why = "confirm_plan", f"this {reason}"
        say = f"This {reason}, so I want a clear yes: {summary}?"
    if confirm_first and mode == "just_do":
        mode, why, say = "confirm_plan", "you asked to be checked with first", f"I'd {summary}, okay?"

    if mode == "just_do" and lesson:
        say = say.rstrip(".") + f", and I'll keep in mind: {lesson}."
    return {"mode": mode, "why": why, "say": say, "lesson": lesson,
            "approach_hint": approach_hint, "must_confirm_steps": [n for n in must if n is not None]}
