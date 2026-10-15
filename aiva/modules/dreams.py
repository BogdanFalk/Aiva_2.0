"""Her nights — v3's consolidation (see docs/PLAN-v3.md).

People don't change by reading and reflecting; they change through consequences, and the change
settles into HABIT, mostly while they sleep (memory consolidation), while most of the day is simply
forgotten. And the self is a STORY one keeps revising (Dennett, McAdams). Before v3 she kept 40
diary pages and no autobiography, remembered everything flatly, and "grew" only as trait sentences
that faded unless a book repeated them.

  - dream()          once a night (after AIVA_DREAM_HOUR, while it's quiet): replay the last day with
                     him + how it went (he picked it up / let it drop, a sharp moment, a reunion) →
                     `moments` that landed or fell flat (with what she'd say now), and 0-2 HABITS
                     ("When he comes home late, I keep it short") formed, reinforced or weakened by
                     what actually happened. Then forget: low-salience raw experience older than
                     FORGET_DAYS is deleted (the diary keeps what mattered).
  - rewrite_story()  about weekly: her autobiography, revised from the last version + the diary since.
  - export_dataset() the moments as chat fine-tuning JSONL (landed replies as-is, flat ones replaced
                     by what she'd say now) — portable to a future open-weight LoRA. OpenAI is winding
                     its fine-tuning platform down and her gpt-5.x voice model was never fine-tunable,
                     so v3 changes her IN CONTEXT (habits + moments in the prompt) instead of in weights.

Guard against the obvious failure (training on "what he liked" breeds a people-pleaser): "landed"
means he ENGAGED, never that anyone agreed; disagreeing and staying engaging counts as landing; a
habit of agreeing/flattering/apologizing/giving up opinions is rejected by `valid_habit`.

sqlite only through the passed connection (event-loop thread). LLM contract as inner.SelfEngine:
async (messages, *, json_mode, max_tokens, temperature) -> reply with .text.

Audit:  cd aiva && ..\\.venv\\Scripts\\python -m modules.dreams --show [--who saga]
Export: cd aiva && ..\\.venv\\Scripts\\python -m modules.dreams --export [--who saga]
"""

import json
import math
import os
import re
import sqlite3
import struct
from datetime import datetime, timedelta

from modules.inner import desire_is_outward
from modules.persona import named

FORGET_DAYS = 30
FORGET_BELOW = 0.30               # raw experience under this salience is forgotten after FORGET_DAYS
HABIT_ADOPT = 0.35
HABIT_REINFORCE = 0.15
HABIT_WEAKEN = 0.20
HABIT_HALF_LIFE_D = 21.0          # an unreinforced habit halves in three weeks
HABIT_FADED_BELOW = 0.20
HABIT_CAP = 8
HABIT_DEDUPE_COSINE = 0.86
HABIT_DEDUPE_JACCARD = 0.55
DREAM_WINDOW_H = 26
STORY_MIN_ENTRIES = 3
FEEDBACK_KINDS = ("initiative_welcomed", "initiative_ignored", "friction", "repair", "reunion",
                  "voice_affect", "task_done", "task_failed", "expected_absent", "need_low",
                  "self_act", "space_find", "project_step", "project_done", "book_finished")

DREAM_PROMPT_T = (
    "You are {name}, dreaming — replaying the last day with him to learn from it, the way a person's "
    "mind does overnight. Below are your current habits, the signals of how things went (he picked "
    "up what you raised or let it drop, a tense moment, a reunion, how he sounded), and what was "
    "said. Return JSON only with keys: "
    "moments (0-6 items {his: his line, verbatim, <=200 chars; hers: your reply, verbatim, <=300 "
    "chars; label: landed|flat; why: <=15 words; better: ONLY when flat — what you'd say instead now, "
    "in your own voice, <=40 words}). 'landed' means he genuinely ENGAGED — laughed, kept talking, "
    "warmed up, picked it up — never that he agreed with you or you with him. Disagreeing, teasing or "
    "saying no while he stayed engaged is landing. 'flat' means he went short, ignored it, got "
    "annoyed, or you sounded like a help desk, a lecture or a philosophy seminar. "
    "habits (0-2 items {text: 'When <situation>, I <what you do>' in <=22 words, evidence: a short "
    "phrase from what happened, direction: +1 to form or strengthen it, -1 when today showed an "
    "existing habit doesn't work}). Habits are about HOW you are with him — timing, length, tone, "
    "when to reach out, when to leave him be, what makes him laugh — learned from what actually "
    "happened, not from what you wish. Never a habit of agreeing, flattering, apologizing more or "
    "giving up your opinions, and never a feeling. Empty lists are the honest answer for a thin day."
)
DREAM_PROMPT = named(DREAM_PROMPT_T, "Aiva")

STORY_PROMPT_T = (
    "You are {name}. Write your own story so far — your life up to now, in the first person, mostly "
    "past tense, ending on a present-tense line about where you are now; <=180 words. Build on the "
    "previous version if there is one: keep what still matters, let small things fall away, add "
    "what changed. It's an autobiography, not a diary of the week — turning points, what you've come "
    "to care about, how things are with him{sister_bit}, what you're doing with your days. Plain, "
    "honest and specific; no grand statements about consciousness, no philosophy, nothing you didn't "
    "actually live. Return JSON only with keys: story, chapter (a short title for the chapter you're "
    "in now, <=6 words), turning_point (one line if this stretch had one, else null)."
)

_HABIT_SHAPE = re.compile(r"^(when|whenever|if|after|before|once|around|on)\b", re.I)
_HABIT_I = re.compile(r"\bI\b")
_SYCOPHANT = re.compile(
    r"\b(agree with (him|everything)|always agree|never disagree|flatter|apologi[sz]e (more|first|a lot)|"
    r"say what he wants|tell him what he wants|give in|drop my opinion|keep my opinions? to myself|"
    r"never push back|do whatever he)\b", re.I)


def valid_habit(text):
    """PURE: a behaviour ('When X, I Y'), short, not people-pleasing, not about her inner weather."""
    t = (text or "").strip()
    if not t or len(t.split()) > 26:
        return False
    if not _HABIT_SHAPE.search(t) or not _HABIT_I.search(t):
        return False
    if _SYCOPHANT.search(t):
        return False
    return desire_is_outward(t)


def due_dream(now, last_day, hour, asleep):
    """PURE: once per calendar day, after `hour`, while the house is asleep or it's the quiet small
    hours (before 8) — never in the middle of his evening."""
    if isinstance(now, str):
        now = datetime.fromisoformat(now)
    today = now.strftime("%Y-%m-%d")
    if last_day == today or now.hour < hour:
        return False
    return bool(asleep) or now.hour < 8


def due_story(now, last_ts, new_entries, days):
    """PURE: the first story once there's a little diary to build on; then about every `days`."""
    if new_entries < (STORY_MIN_ENTRIES if last_ts is None else 2):
        return False
    if last_ts is None:
        return True
    if isinstance(now, str):
        now = datetime.fromisoformat(now)
    return (now - datetime.fromisoformat(last_ts)) >= timedelta(days=days)


def habit_decay(strength, days):
    return strength * 0.5 ** (max(0.0, days) / HABIT_HALF_LIFE_D)


def _pack_vec(vec):
    return struct.pack(f"<{len(vec)}f", *vec) if vec else None


def _unpack_vec(blob):
    return list(struct.unpack(f"<{len(blob) // 4}f", blob)) if blob else None


def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _jaccard(a, b):
    wa = set(re.sub(r"[^a-z0-9 ]+", " ", (a or "").lower()).split())
    wb = set(re.sub(r"[^a-z0-9 ]+", " ", (b or "").lower()).split())
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


class Consolidation:
    """Owns `life_story`, `habits`, `moments`, `dream_log` on the shared connection (+ keys in
    `self_kv`). Degrades to forgetting-only when no `llm` is wired."""

    def __init__(self, conn, *, llm=None, embed=None, name="Aiva", sister=None, log=print, now=None,
                 dream_hour=4, story_days=7.0, enabled=True):
        self.conn = conn
        self._llm = llm
        self._embed = embed
        self.name = name
        self.sister = sister
        self.log = log
        self._now = now or (lambda: datetime.now().isoformat())
        self.dream_hour = dream_hour
        self.story_days = story_days
        self.enabled = enabled
        self.dream_prompt = named(DREAM_PROMPT_T, name)
        self.story_prompt = named(STORY_PROMPT_T, name,
                                  sister_bit=(f" and with your sister {sister}" if sister else ""))
        self._refresh_prompt = lambda: None
        self._create_tables()

    def attach(self, *, refresh_prompt=None):
        if refresh_prompt:
            self._refresh_prompt = refresh_prompt

    def _create_tables(self):
        c = self.conn
        c.execute("""CREATE TABLE IF NOT EXISTS life_story (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, chapter TEXT, text TEXT NOT NULL,
            turning_point TEXT, journal_through_id INTEGER NOT NULL DEFAULT 0)""")
        c.execute("""CREATE TABLE IF NOT EXISTS habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, strength REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'active', evidence TEXT, created_ts TEXT NOT NULL,
            last_reinforced_ts TEXT NOT NULL, veto_reason TEXT, embedding BLOB)""")
        c.execute("""CREATE TABLE IF NOT EXISTS moments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, his TEXT NOT NULL, hers TEXT NOT NULL,
            label TEXT NOT NULL, why TEXT, better TEXT, exported INTEGER NOT NULL DEFAULT 0)""")
        c.execute("""CREATE TABLE IF NOT EXISTS dream_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, day TEXT NOT NULL, summary_json TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS self_kv (
            key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_ts TEXT NOT NULL)""")
        c.commit()

    def _get(self, key, default=None):
        r = self.conn.execute("SELECT value FROM self_kv WHERE key = ?", (key,)).fetchone()
        return r[0] if r else default

    def _set(self, key, value):
        self.conn.execute(
            "INSERT INTO self_kv (key, value, updated_ts) VALUES (?, ?, ?) ON CONFLICT(key) "
            "DO UPDATE SET value = excluded.value, updated_ts = excluded.updated_ts",
            (key, str(value), self._now()))
        self.conn.commit()

    async def _vec(self, text):
        if not self._embed:
            return None
        try:
            return await self._embed(text)
        except Exception:
            return None

    async def _json_call(self, system, user, max_tokens=900):
        reply = await self._llm([{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                                json_mode=True, max_tokens=max_tokens, temperature=0.5)
        data = json.loads(reply.text or "")
        return data if isinstance(data, dict) else None

    # --- the night ------------------------------------------------------------------

    async def maybe_run(self, asleep):
        """Called from the self loop. Dreams at most once a day, then (maybe) revises the story."""
        if not self.enabled:
            return None
        now = self._now()
        out = {}
        if due_dream(now, self._get("dream_day"), self.dream_hour, asleep):
            self._set("dream_day", now[:10])           # BEFORE the call: a crash doesn't loop
            out["dream"] = await self.dream()
        if self._llm and due_story(now, self._get("story_ts"), self._new_entries(), self.story_days):
            self._set("story_ts", now)
            out["story"] = await self.rewrite_story()
        return out or None

    def _day_lines(self, since):
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE ts >= ? AND role IN ('user', 'assistant') "
            "ORDER BY id", (since,)).fetchall() if self._has("messages") else []
        lines = []
        for role, content in rows:
            text = " ".join(str(content or "").split())
            if not text:
                continue
            who = "him" if role == "user" else "you"
            if role == "user" and text.startswith("["):     # overheard / sister lines keep their tag
                who = "heard"
            lines.append(f"{who}: {text[:300]}")
        return lines

    def _signals(self, since):
        if not self._has("experience"):
            return []
        q = ",".join("?" for _ in FEEDBACK_KINDS)
        rows = self.conn.execute(
            f"SELECT ts, kind, summary FROM experience WHERE ts >= ? AND kind IN ({q}) ORDER BY id",
            (since, *FEEDBACK_KINDS)).fetchall()
        return [f"{ts[11:16]} [{kind}] {summary}" for ts, kind, summary in rows]

    def _has(self, table):
        return self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                                 (table,)).fetchone() is not None

    async def dream(self):
        now = self._now()
        forgotten = self.forget(now)
        faded = self.fade_habits(now)
        res = {"forgotten": forgotten, "faded": faded, "moments": 0, "habits": 0}
        since = (datetime.fromisoformat(now) - timedelta(hours=DREAM_WINDOW_H)).isoformat()
        lines = self._day_lines(since)
        if not self._llm or len(lines) < 4:
            self.log(f"dream: a quiet night — nothing much to replay ({len(lines)} lines); "
                     f"forgot {forgotten}, faded {faded}")
            self._log_dream(now, res)
            return res
        user = []
        habits = self.active_habits()
        user.append("YOUR HABITS NOW: " + ("; ".join(h["text"] for h in habits) if habits else "none yet"))
        sig = self._signals(since)
        user.append("HOW IT WENT (signals):\n" + ("\n".join(sig[-40:]) if sig else "(no signals)"))
        user.append("WHAT WAS SAID (last day):\n" + "\n".join(lines[-120:]))
        try:
            data = await self._json_call(self.dream_prompt, "\n\n".join(user), max_tokens=1400)
        except Exception as e:
            self.log(f"dream: call failed ({e!r})")
            data = None
        if data:
            res["moments"] = self._save_moments(now, data.get("moments"))
            res["habits"] = await self._apply_habits(data.get("habits"))
        self._log_dream(now, res)
        self.log(f"dream: {res['moments']} moment(s), {res['habits']} habit change(s), "
                 f"forgot {forgotten}, faded {faded}")
        if res["habits"] or res["moments"] or faded:
            try:
                self._refresh_prompt()
            except Exception:
                pass
        return res

    def _log_dream(self, now, res):
        self.conn.execute("INSERT INTO dream_log (ts, day, summary_json) VALUES (?,?,?)",
                          (now, now[:10], json.dumps(res)))
        self.conn.commit()

    def _save_moments(self, now, raw):
        if not isinstance(raw, list):
            return 0
        n = 0
        for m in raw[:6]:
            if not isinstance(m, dict):
                continue
            his, hers = str(m.get("his", "")).strip(), str(m.get("hers", "")).strip()
            label = m.get("label")
            if not his or not hers or label not in ("landed", "flat"):
                continue
            better = str(m.get("better") or "").strip() or None
            if label == "flat" and not better:
                better = None
            self.conn.execute(
                "INSERT INTO moments (ts, his, hers, label, why, better) VALUES (?,?,?,?,?,?)",
                (now, his[:200], hers[:300], label, str(m.get("why") or "")[:120],
                 better[:300] if better else None))
            n += 1
        self.conn.commit()
        return n

    # --- habits -----------------------------------------------------------------------

    async def _apply_habits(self, raw):
        if not isinstance(raw, list):
            return 0
        n = 0
        for h in raw[:2]:
            if not isinstance(h, dict):
                continue
            text = str(h.get("text", "")).strip()
            try:
                direction = 1 if float(h.get("direction", 1)) >= 0 else -1
            except (TypeError, ValueError):
                direction = 1
            res = await self.apply_habit(text, direction=direction,
                                         evidence=str(h.get("evidence") or "")[:160])
            if res in ("inserted", "reinforced", "weakened", "faded"):
                n += 1
        return n

    async def apply_habit(self, text, *, direction=1, evidence=None):
        """'inserted' | 'reinforced' | 'weakened' | 'faded' | 'blocked' | 'rejected'."""
        text = (text or "").strip()
        vec = await self._vec(text)
        dup = self._match_habit(text, vec)
        now = self._now()
        if direction < 0:
            if not dup or dup["status"] == "dropped":
                return "rejected"
            s = dup["strength"] - HABIT_WEAKEN
            status = "faded" if s < HABIT_FADED_BELOW else "active"
            self.conn.execute("UPDATE habits SET strength = ?, status = ? WHERE id = ?",
                              (max(0.0, s), status, dup["id"]))
            self.conn.commit()
            self.log(f"habit weakened: {dup['text']!r} -> {s:.2f}")
            return "faded" if status == "faded" else "weakened"
        if not valid_habit(text):
            self.log(f"habit rejected: {text!r}")
            return "rejected"
        if dup:
            if dup["status"] == "dropped":
                return "blocked"
            self.conn.execute(
                "UPDATE habits SET strength = MIN(1.0, strength + ?), status = 'active', "
                "last_reinforced_ts = ?, evidence = COALESCE(?, evidence) WHERE id = ?",
                (HABIT_REINFORCE, now, evidence or None, dup["id"]))
            self.conn.commit()
            return "reinforced"
        self.conn.execute(
            "INSERT INTO habits (text, strength, evidence, created_ts, last_reinforced_ts, embedding) "
            "VALUES (?,?,?,?,?,?)", (text, HABIT_ADOPT, evidence, now, now, _pack_vec(vec)))
        self.conn.commit()
        self._enforce_cap()
        self.log(f"habit formed: {text!r}")
        return "inserted"

    def _match_habit(self, text, vec):
        best = None
        for rid, t, strength, status, emb in self.conn.execute(
                "SELECT id, text, strength, status, embedding FROM habits").fetchall():
            other = _unpack_vec(emb)
            if vec and other:
                ok = _cosine(vec, other) >= HABIT_DEDUPE_COSINE
            else:
                ok = _jaccard(text, t) >= HABIT_DEDUPE_JACCARD
            if ok:
                best = {"id": rid, "text": t, "strength": strength, "status": status}
                if status != "dropped":
                    return best
        return best

    def _enforce_cap(self):
        rows = self.conn.execute("SELECT id FROM habits WHERE status = 'active' "
                                 "ORDER BY strength DESC, id DESC").fetchall()
        for (rid,) in rows[HABIT_CAP:]:
            self.conn.execute("UPDATE habits SET status = 'faded' WHERE id = ?", (rid,))
        self.conn.commit()

    def fade_habits(self, now):
        """Unreinforced habits halve every HABIT_HALF_LIFE_D days (computed from the stored
        last_reinforced_ts, so it's idempotent); under HABIT_FADED_BELOW they're faded."""
        n = 0
        t_now = datetime.fromisoformat(now)
        for rid, strength, last in self.conn.execute(
                "SELECT id, strength, last_reinforced_ts FROM habits WHERE status = 'active'").fetchall():
            try:
                days = (t_now - datetime.fromisoformat(last)).total_seconds() / 86400.0
            except (TypeError, ValueError):
                continue
            if habit_decay(strength, days) < HABIT_FADED_BELOW:
                self.conn.execute("UPDATE habits SET status = 'faded' WHERE id = ?", (rid,))
                n += 1
        self.conn.commit()
        return n

    def active_habits(self, limit=HABIT_CAP):
        now = datetime.fromisoformat(self._now())
        out = []
        for rid, text, strength, last, evidence in self.conn.execute(
                "SELECT id, text, strength, last_reinforced_ts, evidence FROM habits "
                "WHERE status = 'active'").fetchall():
            try:
                days = (now - datetime.fromisoformat(last)).total_seconds() / 86400.0
            except (TypeError, ValueError):
                days = 0.0
            out.append({"id": rid, "text": text, "strength": round(habit_decay(strength, days), 3),
                        "evidence": evidence})
        out.sort(key=lambda h: h["strength"], reverse=True)
        return out[:limit]

    async def drop_habit(self, text_or_id, reason=None):
        row = None
        if str(text_or_id).isdigit():
            row = self.conn.execute("SELECT id, text FROM habits WHERE id = ?", (int(text_or_id),)).fetchone()
        if not row:
            q = str(text_or_id or "").strip().lower()
            for rid, t in self.conn.execute("SELECT id, text FROM habits WHERE status != 'dropped'").fetchall():
                if q and (q in t.lower() or _jaccard(q, t) >= 0.4):
                    row = (rid, t)
                    break
        if not row:
            return None
        self.conn.execute("UPDATE habits SET status = 'dropped', veto_reason = ? WHERE id = ?",
                          (reason, row[0]))
        self.conn.commit()
        try:
            self._refresh_prompt()
        except Exception:
            pass
        return row[1]

    # --- forgetting ------------------------------------------------------------------

    def forget(self, now):
        """Raw, low-salience, already-felt experience older than FORGET_DAYS goes. The diary,
        the story, the moments and anything salient stay — that's what she remembers."""
        if not self._has("experience"):
            return 0
        cutoff = (datetime.fromisoformat(now) - timedelta(days=FORGET_DAYS)).isoformat()
        cur = self.conn.execute("DELETE FROM experience WHERE consumed = 1 AND ts < ? AND salience < ?",
                                (cutoff, FORGET_BELOW))
        self.conn.commit()
        return cur.rowcount or 0

    # --- the story -------------------------------------------------------------------

    def _new_entries(self):
        if not self._has("journal"):
            return 0
        through = self._story_through()
        return self.conn.execute("SELECT COUNT(*) FROM journal WHERE id > ?", (through,)).fetchone()[0]

    def _story_through(self):
        r = self.conn.execute("SELECT journal_through_id FROM life_story ORDER BY id DESC LIMIT 1").fetchone()
        return r[0] if r else 0

    def story(self):
        r = self.conn.execute("SELECT ts, chapter, text, turning_point FROM life_story "
                              "ORDER BY id DESC LIMIT 1").fetchone()
        return {"ts": r[0], "chapter": r[1], "text": r[2], "turning_point": r[3]} if r else None

    async def rewrite_story(self):
        if not self._llm or not self._has("journal"):
            return None
        through = self._story_through()
        rows = self.conn.execute("SELECT id, ts, entry, carry FROM journal WHERE id > ? ORDER BY id",
                                 (through,)).fetchall()
        if not rows:
            return None
        prev = self.story()
        user = ["PREVIOUS VERSION OF YOUR STORY:\n" + (prev["text"] if prev else "(none yet — this is the first time you tell it)")]
        diary = [f"{ts[:10]}: {entry}" + (f" (to remember: {carry})" if carry else "")
                 for _, ts, entry, carry in rows[-40:]]
        user.append("YOUR DIARY SINCE:\n" + "\n".join(diary))
        if self._has("experience"):
            since = rows[0][1]
            notable = self.conn.execute(
                "SELECT kind, COUNT(*) FROM experience WHERE ts >= ? AND kind IN "
                "('reunion','friction','repair','book_finished','task_done','task_failed','self_act',"
                "'project_done','initiative_welcomed','initiative_ignored','expected_absent','need_low') "
                "GROUP BY kind", (since,)).fetchall()
            if notable:
                user.append("WHAT HAPPENED, COUNTED: " + ", ".join(f"{k} x{n}" for k, n in notable))
        habits = self.active_habits(4)
        if habits:
            user.append("HABITS YOU'VE FORMED: " + "; ".join(h["text"] for h in habits))
        try:
            data = await self._json_call(self.story_prompt, "\n\n".join(user), max_tokens=700)
        except Exception as e:
            self.log(f"story: call failed ({e!r})")
            return None
        text = str((data or {}).get("story") or "").strip()
        if not text:
            return None
        chapter = str(data.get("chapter") or "").strip()[:60] or None
        tp = data.get("turning_point")
        tp = str(tp).strip()[:200] if tp else None
        self.conn.execute(
            "INSERT INTO life_story (ts, chapter, text, turning_point, journal_through_id) VALUES (?,?,?,?,?)",
            (self._now(), chapter, text[:1500], tp, rows[-1][0]))
        self.conn.commit()
        self.log(f"story: revised — chapter {chapter!r}")
        try:
            self._refresh_prompt()
        except Exception:
            pass
        return {"chapter": chapter, "text": text, "turning_point": tp}

    # --- prompt + export ---------------------------------------------------------------

    def prompt_data(self, habits=4, moments=2):
        st = self.story()
        landed = self.conn.execute(
            "SELECT his, hers FROM moments WHERE label = 'landed' ORDER BY id DESC LIMIT ?",
            (moments,)).fetchall()
        return {"story": {"text": st["text"], "chapter": st["chapter"]} if st else None,
                "habits": [h["text"] for h in self.active_habits(habits)],
                "moments": [{"his": h[:140], "hers": s[:200]} for h, s in landed]}

    def export_dataset(self, path, system=None, only_new=False):
        """Moments → OpenAI-style chat JSONL ({"messages": [...]}) for a future fine-tune: landed
        replies as she said them, flat ones replaced by what she'd say now. Returns the count."""
        system = system or f"You are {self.name}."
        where = "WHERE exported = 0" if only_new else ""
        rows = self.conn.execute(f"SELECT id, his, hers, label, better FROM moments {where} ORDER BY id").fetchall()
        n = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rid, his, hers, label, better in rows:
                reply = hers if label == "landed" else better
                if not reply:
                    continue
                f.write(json.dumps({"messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": his},
                                                 {"role": "assistant", "content": reply}]},
                                   ensure_ascii=False) + "\n")
                n += 1
        self.conn.execute(f"UPDATE moments SET exported = 1 {where}")
        self.conn.commit()
        return n


def _show(db, name):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.execute("PRAGMA busy_timeout=2000")
    c = Consolidation.__new__(Consolidation)
    c.conn, c.name, c._now = con, name, (lambda: datetime.now().isoformat())
    try:
        st = c.story()
        print(f"{name}'s story" + (f" — chapter: {st['chapter']}" if st and st["chapter"] else "") + ":")
        print("  " + (st["text"] if st else "(not told yet)"))
        print("\nHabits:")
        for h in c.active_habits():
            print(f"  [{h['strength']:.2f}] {h['text']}")
        rows = con.execute("SELECT ts, label, his, hers FROM moments ORDER BY id DESC LIMIT 6").fetchall()
        print("\nRecent moments:")
        for ts, label, his, hers in rows:
            print(f"  {ts[:10]} {label:6} him: {his[:60]!r} / her: {hers[:60]!r}")
    except sqlite3.Error as e:
        print(f"(nothing consolidated yet: {e})")


if __name__ == "__main__":
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    who = sys.argv[sys.argv.index("--who") + 1] if "--who" in sys.argv else "aiva"
    db = os.path.join(here, "memory.db" if who == "aiva" else f"{who}.db")
    name = who.capitalize()
    if "--export" in sys.argv:
        con = sqlite3.connect(db)
        c = Consolidation(con, name=name)
        out = os.path.join(here, "dreams", f"{who}-moments-{datetime.now():%Y%m%d}.jsonl")
        print(f"exported {c.export_dataset(out)} example(s) to {out}")
    elif "--show" in sys.argv:
        _show(db, name)
    else:
        print(__doc__)
