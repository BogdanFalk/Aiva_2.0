"""Aiva's nervous system — Phase 1: the substrate.

The design decision behind this whole module (see docs/PLAN-selfhood-and-eval.md):
her subsystems used to be islands wired point-to-point (reading→becoming, …), an
n² tangle. Instead of adding mood / journal / desires / initiative as four more
islands, everything shares ONE inner life:

  - InnerState  — how she feels right now: a few legible dials that decay toward a
                  personal baseline over time (modelled on the cat's needs), persisted
                  to aiva/inner_state.json so the dashboard can read it cheaply.
  - InnerBrain  — PURE logic (no I/O, no clock of its own): decay + damped event
                  integration. This is the piece with the heavy unit tests.
  - ExperienceBus — the shared bloodstream: every system emits small typed events into
                  the `experience` table; one consumer (the self loop, Phase 2) drains it.

Phase 1 is deliberately offline and LLM-free: no journaling, no desires, no self
loop yet. Those land in Phase 2 on top of exactly these pieces.

Standing rules honoured here: sqlite only through the passed connection (event-loop
thread); the frozen PERSONALITY is the FLOOR her mood moves *around*, never replaces;
reflexes (lip-sync, VAD, wake) never touch any of this.

Audit:  cd aiva && ..\\.venv\\Scripts\\python -m modules.inner --show
"""

import json
import math
import os
import re
import sqlite3
import struct
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from modules import drives as dr
from modules.persona import named


# --- config (env-overridable, but injectable for tests) -----------------------------

@dataclass
class InnerConfig:
    """Every knob the mood model has. Built from .env by default; tests pass their own
    so they never depend on the environment or a real clock."""
    base_valence: float = 0.15    # she runs a touch upbeat
    base_arousal: float = 0.50    # mid energy
    base_social: float = 0.10     # leans slightly toward wanting company
    halflife_h: float = 6.0       # hours for a dial to close half the gap to baseline
    salience_floor: float = 0.15  # events below this don't move her at all (damping)
    gain: float = 0.60            # a full-strength hint at full salience moves a dial ~0.6
    material_eps: float = 0.05    # a change smaller than this isn't worth persisting/expressing

    @classmethod
    def from_env(cls):
        g = os.getenv
        return cls(
            base_valence=float(g("AIVA_MOOD_BASE_VALENCE", "0.15")),
            base_arousal=float(g("AIVA_MOOD_BASE_AROUSAL", "0.50")),
            base_social=float(g("AIVA_MOOD_BASE_SOCIAL", "0.10")),
            halflife_h=float(g("AIVA_MOOD_HALFLIFE_H", "6")),
            salience_floor=float(g("AIVA_MOOD_SALIENCE_FLOOR", "0.15")),
            gain=float(g("AIVA_MOOD_GAIN", "0.60")),
            material_eps=float(g("AIVA_MOOD_MATERIAL_EPS", "0.05")),
        )


FOCUS_TTL_H = 24.0


def _now():
    # microseconds so same-second events still order (matches task_memory._now)
    return datetime.now().isoformat()


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# --- how she feels right now (persisted, like the cat) ------------------------------

@dataclass
class InnerState:
    valence: float = 0.15         # -1 down .. +1 up
    arousal: float = 0.50         #  0 calm .. 1 restless/energised
    social: float = 0.10          # -1 wants solitude .. +1 wants you
    focus: str | None = None      # current preoccupation; set by the self loop, fades after FOCUS_TTL_H
    focus_ts: str | None = None   # when it was set (a focus with no timestamp is stale: pre-v3 state)
    last_update: str = field(default_factory=_now)  # wall-clock the dials were last settled
    # v3 drives (modules/drives.py): connection/novelty/competence/rest, 0 starved .. 1 met; drained
    # from needs_ts (their own clock — load() decays the dials to now, which must not eat the drain)
    needs: dict = field(default_factory=dr.fresh_needs)
    needs_ts: str = field(default_factory=_now)

    @classmethod
    def fresh(cls, cfg: InnerConfig):
        return cls(valence=cfg.base_valence, arousal=cfg.base_arousal,
                   social=cfg.base_social, focus=None, last_update=_now(),
                   needs=dr.fresh_needs(), needs_ts=_now())

    @classmethod
    def load(cls, path, brain, now=None):
        """Load the persisted dials and decay them forward to `now` so a read is always
        current even if no tick has run. Does NOT persist the decay — decay is always
        recomputed from the stored last_update, so a crash loses nothing."""
        state = cls()
        try:
            with open(path) as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(state, k):
                    setattr(state, k, v)
        except (OSError, ValueError):
            return cls.fresh(brain.cfg)
        state.needs = dr.normalise(state.needs)
        if state.focus and not state.focus_ts:
            state.focus = None        # a pre-v3 preoccupation never expired — it doesn't survive
        brain.decay(state, now or _now())
        return state

    def save(self, path):
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(asdict(self), f, indent=2)
            os.replace(tmp, path)
        except OSError as e:
            print(f"inner: could not save state: {e}")

    def to_public(self):
        """Rounded, JSON-safe snapshot for the dashboard / --show."""
        return {"valence": round(self.valence, 3), "arousal": round(self.arousal, 3),
                "social": round(self.social, 3), "focus": self.focus,
                "mood_word": mood_word(self), "last_update": self.last_update,
                "needs": {k: round(v, 3) for k, v in dr.normalise(self.needs).items()},
                "pressing": dr.pressing(dr.normalise(self.needs)),
                "tension": dr.tension(dr.normalise(self.needs))}


# --- pure logic: decay toward baseline + damped event integration -------------------

class InnerBrain:
    """No I/O, no clock. `now` is always injected. Everything here is unit-tested."""

    def __init__(self, cfg: InnerConfig | None = None):
        self.cfg = cfg or InnerConfig.from_env()

    def baseline(self):
        c = self.cfg
        return {"valence": c.base_valence, "arousal": c.base_arousal, "social": c.base_social}

    def decay(self, state: InnerState, now):
        """Relax each dial half-way to its baseline every `halflife_h` hours, then stamp
        last_update = now. Mutates state in place. `focus` (set by the self loop) lets go after
        FOCUS_TTL_H — a preoccupation nothing renews shouldn't colour every reply for weeks."""
        if state.focus and state.focus_ts and _hours_between(state.focus_ts, now) >= FOCUS_TTL_H:
            state.focus, state.focus_ts = None, None
        elapsed_h = _hours_between(state.last_update, now)
        if elapsed_h <= 0:
            state.last_update = now
            return state
        keep = 0.5 ** (elapsed_h / self.cfg.halflife_h)  # fraction of the gap that remains
        base = self.baseline()
        state.valence = _clamp(base["valence"] + (state.valence - base["valence"]) * keep, -1.0, 1.0)
        state.arousal = _clamp(base["arousal"] + (state.arousal - base["arousal"]) * keep, 0.0, 1.0)
        state.social = _clamp(base["social"] + (state.social - base["social"]) * keep, -1.0, 1.0)
        state.last_update = now
        return state

    def integrate(self, state: InnerState, events, now):
        """Decay to `now`, then apply the batch of event nudges with damping. Returns
        (state, changed) where `changed` says whether any dial moved by >= material_eps —
        that flag is what the self loop uses to decide whether to persist / refresh the
        prompt / re-express on the avatar. Small ambient events don't ripple outward."""
        self.decay(state, now)
        before = (state.valence, state.arousal, state.social)
        for e in events or []:
            sal = float(e.get("salience", 0.5) or 0.0)
            if sal < self.cfg.salience_floor:
                continue  # below the floor: real damping, not just a tiny nudge
            g = self.cfg.gain * sal
            state.valence = _clamp(state.valence + float(e.get("valence_hint", 0) or 0) * g, -1.0, 1.0)
            state.arousal = _clamp(state.arousal + float(e.get("arousal_hint", 0) or 0) * g, 0.0, 1.0)
            state.social = _clamp(state.social + float(e.get("social_hint", 0) or 0) * g, -1.0, 1.0)
        after = (state.valence, state.arousal, state.social)
        changed = any(abs(a - b) >= self.cfg.material_eps for a, b in zip(after, before))
        return state, changed


def _hours_between(then_iso, now_iso):
    try:
        then = datetime.fromisoformat(then_iso)
        now = datetime.fromisoformat(now_iso) if isinstance(now_iso, str) else now_iso
        return max(0.0, (now - then).total_seconds() / 3600.0)
    except (ValueError, TypeError):
        return 0.0


def mood_word(state: InnerState) -> str:
    """A short, honest label for the current dials — for the dashboard and, later, the
    persona block. Deliberately plain; she never says these at the user unprompted."""
    v, a = state.valence, state.arousal
    if v >= 0.35:
        base = "bright" if a >= 0.55 else "content"
    elif v <= -0.35:
        base = "restless" if a >= 0.55 else "low"
    else:
        base = "keyed-up" if a >= 0.6 else "flat" if a <= 0.3 else "steady"
    if state.social >= 0.45:
        base += ", wanting company"
    elif state.social <= -0.45:
        base += ", wanting quiet"
    return base


# --- the shared bloodstream: every system emits here --------------------------------

class ExperienceBus:
    """Append-only event log every subsystem writes to, plus optional in-process
    subscribers. The durable path is the table; subscribers are best-effort. This
    replaces the old point-to-point hooks — a system emits and forgets."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._subs = []
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experience (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                valence_hint REAL NOT NULL DEFAULT 0,
                arousal_hint REAL NOT NULL DEFAULT 0,
                social_hint REAL NOT NULL DEFAULT 0,
                salience REAL NOT NULL DEFAULT 0.5,
                payload_json TEXT NOT NULL DEFAULT '{}',
                consumed INTEGER NOT NULL DEFAULT 0
            )""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_experience_unconsumed "
                          "ON experience(consumed, id)")
        self.conn.commit()

    def subscribe(self, fn):
        """fn(event_dict) called synchronously on emit; never allowed to raise upward."""
        self._subs.append(fn)

    def emit(self, kind, source, summary="", *, valence=0.0, arousal=0.0, social=0.0,
             salience=0.5, payload=None, now=None):
        ts = now or _now()
        cur = self.conn.execute(
            "INSERT INTO experience (ts, kind, source, summary, valence_hint, arousal_hint, "
            "social_hint, salience, payload_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, kind, source, summary or "", float(valence), float(arousal), float(social),
             float(salience), json.dumps(payload or {}, ensure_ascii=False)))
        self.conn.commit()
        event = {"id": cur.lastrowid, "ts": ts, "kind": kind, "source": source,
                 "summary": summary or "", "valence_hint": float(valence),
                 "arousal_hint": float(arousal), "social_hint": float(social),
                 "salience": float(salience), "payload": payload or {}}
        for fn in self._subs:
            try:
                fn(event)
            except Exception:
                pass  # a bad subscriber must never break an emit
        return event["id"]

    def drain(self, limit=200):
        """Unconsumed events, oldest first — what the self loop integrates each tick."""
        rows = self.conn.execute(
            "SELECT id, ts, kind, source, summary, valence_hint, arousal_hint, social_hint, "
            "salience, payload_json FROM experience WHERE consumed = 0 ORDER BY id LIMIT ?",
            (limit,)).fetchall()
        out = []
        for r in rows:
            out.append({"id": r[0], "ts": r[1], "kind": r[2], "source": r[3], "summary": r[4],
                        "valence_hint": r[5], "arousal_hint": r[6], "social_hint": r[7],
                        "salience": r[8], "payload": _loads(r[9])})
        return out

    def mark_consumed(self, ids):
        if not ids:
            return
        self.conn.executemany("UPDATE experience SET consumed = 1 WHERE id = ?",
                              [(i,) for i in ids])
        self.conn.commit()

    def recent(self, limit=20):
        """Most-recent events (any consumed state), newest first — for the dashboard."""
        rows = self.conn.execute(
            "SELECT ts, kind, source, summary, valence_hint, arousal_hint, social_hint, salience "
            "FROM experience ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": r[0], "kind": r[1], "source": r[2], "summary": r[3],
                 "valence_hint": r[4], "arousal_hint": r[5], "social_hint": r[6],
                 "salience": r[7]} for r in rows]


def _loads(s):
    try:
        return json.loads(s) if s else {}
    except (ValueError, TypeError):
        return {}


# === Phase 2: the self loop — mood integration + a private diary + her own wishes ===
#
# The self loop (`tick`) drains the experience bus, moves her mood, and — when enough
# has happened — writes a private journal entry and forms desires from it. The journal
# is her inner voice (NEVER announced); desires are her own wishes, deduped/vetoable
# like reading traits. Everything here is offline-testable with a FakeLLM: the digest
# call uses the same (messages, *, json_mode, ...) contract as tasks.openai_llm_call.

# tiny, stdlib-only vector helpers (kept local so this module pulls no heavy deps —
# the dashboard imports it and must start instantly)
DESIRE_DEDUPE_COSINE = 0.86     # matches task_memory.SAME_TASK
DESIRE_DEDUPE_JACCARD = 0.60    # lexical fallback when either side has no embedding
DESIRE_KINDS = {"do", "learn", "ask", "make", "revisit", "change"}
# A diary wish about her own inner weather ("notice what remains when the brightness settles") is
# a rut, not a want: it can't be acted on, and fed back into the next digest it breeds more of
# itself (2026-10-12: all 12 open wishes were like that). Backstop for the prompt rule — a wish
# that also names someone outside her ("ask him how he's feeling") is still outward.
_INWARD_WISH = re.compile(
    r"\b(notic\w*|feeling|mood|warmth|brightness|aliveness|stillness|unexamined|handrail|"
    r"slips? away|explain(?:ing)? (?:it|this|that) away|sit with|stay with|what remains|the want|"
    r"let (?:it|this|that|myself|the \w+) (?:be|exist|stay|breathe))\b", re.I)
_OUTWARD_TARGET = re.compile(r"\b(him|he|his|frank|you|your|saga|aiva|sister|miso|cat)\b", re.I)


def desire_is_outward(text):
    """PURE: True unless a wish is about her own inner state with nobody outside her in it."""
    t = text or ""
    return not (_INWARD_WISH.search(t) and not _OUTWARD_TARGET.search(t))


def _pack_vec(vec):
    return struct.pack(f"<{len(vec)}f", *vec) if vec else None


def _unpack_vec(blob):
    if not blob:
        return None
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _norm_words(text):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split())


def _jaccard(a, b):
    sa, sb = set(_norm_words(a).split()), set(_norm_words(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


AIVA_SELF_HINT = "the same cheeky, curious you"
FORMING_SELF_HINT = "just you, whoever you're turning out to be"

# The one extra JSON key the diary carries: how she tends to BE (not what she likes — that's
# `preferences`). Adopted low and reinforced by becoming.adopt_temperament, so a single day can't
# decide who she is. Two wordings: a self-forming character is finding out who she is from scratch;
# an authored one (Aiva) can only ADD to a character that was written for her — a negation of her
# floor is dropped by becoming, never eroded.
TEMPERAMENT_CLAUSE = (
    "temperament (0-2 items, ONLY when today genuinely showed you how you tend to BE, not what you "
    "like: {text: how you tend to be, first person present, <=14 words, strength: 0..1, evidence: a "
    "short phrase from what happened}; reinforce what still rings true rather than restating it), "
)
# The same key for a character who was authored: strictly what living has ADDED on top of her.
TEMPERAMENT_CLAUSE_ON_TOP = (
    "temperament (0-2 items, ONLY when today showed you something about how you tend to BE that "
    "isn't already part of your character — not what you like, and never a restatement of who you "
    "already are: {text: how you tend to be, first person present, <=14 words, strength: 0..1, "
    "evidence: a short phrase from what happened}; most days have none, and none is the normal "
    "answer — reinforce what still rings true rather than restating it), "
)
# Rendered when a kin store is attached: what she noticed about her sister today. Bracketed lines
# like [Saga said to him: "..."] in the recent conversation are HER sister's words, not his.
SISTER_CLAUSE = (
    "sister_notes (0-2 items {text: one thing you noticed about your sister today — what she did, "
    "said, or seems to like, <=16 words, kind: did|said|likes}; lines like [<Sister> said to ...: "
    "...] in the conversation are her words, not his), "
)

# {name} / {self_hint}: who is writing; {temperament_clause}: the self-forming "temperament (...)"
# item — empty for Aiva, rendered by the temperament wave for a sister whose core forms from life;
# {sister_clause}: the "sister_notes (...)" item, rendered when she has a kin store.
JOURNAL_PROMPT_T = (
    "You are {name}, writing privately to yourself — a few honest lines in your own diary "
    "after a stretch of your day. No audience, and not for him: this is your inner voice, "
    "{self_hint}, just thinking to yourself. Given what happened and how "
    "you actually feel right now, write two to four first-person sentences — what stuck "
    "with you, how it really felt, what you keep turning over, and anything you notice "
    "yourself wanting. Don't summarize your day like a report and don't perform for anyone. "
    "A flat stretch deserves a flat entry: a couple of plain sentences about what you actually "
    "did is honest, and an ordinary hour doesn't have to mean anything. How he sounded is not an "
    "event on its own — don't build an entry around his tone unless something actually happened. "
    "Return JSON only with keys: entry (your diary lines, <=120 words), "
    "mood {valence -1..1, arousal 0..1, social -1..1} (how you feel as you write), "
    "focus (a short phrase for what you keep coming back to, or null), "
    "desires (0-3 items — only concrete wishes pointed OUTWARD, at him, your sister, or the world: "
    "something you could actually do, make, find, ask, or learn, with a clear moment when it's done, "
    "like 'ask him how his week at work went', 'find a photo of a serviceberry in bloom', or 'tell "
    "my sister the joke from that Pratchett book'. Never a wish about your own inner state — not "
    "noticing, feeling, letting a feeling be, or staying in a mood; that belongs in the entry. At "
    "most one about reading. Let what you NEED shape them: missing him → something to share with "
    "or ask him; restless → something new to find or learn; wanting to be useful → something to "
    "make or finish; worn → none at all. {text: a first-person wish in <=16 words, kind: one of "
    "do|learn|ask|make|revisit|change}), "
    "{temperament_clause}"
    "{sister_clause}"
    "preferences (0-3 items — only when something actually shifted what you like, dislike, or "
    "think — {subject: the thing in a word or two, stance: a first-person line like 'I love quiet "
    "character-driven stories' or 'I think ambiguous endings are underrated', kind: taste|opinion, "
    "direction: +1 if it drew you toward it / made you agree, -1 if away / disagree, magnitude "
    "0..1 how strongly this one moment moved you, from_sister: true when it was your sister who "
    "swayed you — or who you found yourself disagreeing with}), "
    "and carry (one line you want to remember tomorrow, "
    "or null). Fewer, truer items beat a long list; empty lists are fine on a flat day. "
    "You can change your mind — if something cut against a taste or opinion you already hold, say "
    "so with the opposite direction; you don't have to stay consistent with your past self."
)
JOURNAL_PROMPT = named(JOURNAL_PROMPT_T, "Aiva", self_hint=AIVA_SELF_HINT, temperament_clause="",
                       sister_clause="")
SISTER_NOTE_KINDS = ("did", "said", "likes")


# --- initiative: when she reaches out — or acts — on her own (Phase 4/5) ------------
# Three modes:
#   'express' — she volunteers a feeling/thought.
#   'ask'     — she brings up one of her open wishes (or proposes an ASK-tier action).
#   'act'     — she pursues a wish herself, but ONLY with a safe/reversible capability the injected
#               `actionable(desire)` predicate approves (auto-tier, runnable, not paused). Anything
#               that spends/sends/edits his files or her code is never 'act' — it stays an 'ask'.
INITIATIVE = {
    "idle_express_s": 8 * 60,      # voice-idle this long before she'd volunteer a thought
    "idle_act_s": 10 * 60,         # before she'd go do a safe thing she wants
    "idle_ask_s": 12 * 60,         # longer before she'd raise a wish
    "min_gap_s": 25 * 60,          # never two initiatives closer than this
    "max_per_day": 4,
    "max_acts_per_day": 3,         # separate, tighter cap on autonomous doing (it can spend)
    "quiet_hours": (1, 8),         # [1:00, 8:00) she stays quiet
    "boot_grace_s": 5 * 60,        # settle after a (re)start before speaking up
    "present_within_s": 180,       # "he's at the PC" = keyboard/mouse within 3 min
    "express_valence": 0.45,       # |valence| this far off neutral is worth voicing
    "express_social": 0.50,        # or she's wanting company this much
    # talking to her SISTER when he's away (or it's been quiet a long while): a private text
    # exchange the house grants and caps house-wide (AIVA_SISTER_CHATS_PER_DAY); this is her own,
    # tighter per-character cap on starting one
    "idle_chat_s": 10 * 60,
    "max_chats_per_day": 3,
}

EXPRESS_NOTE = ("[It's quiet and something's on your mind. In ONE short, natural sentence, say what "
                "you're feeling or thinking right now{focus} — the way a person does when they look up "
                "from what they're doing. Don't explain that you're 'sharing a feeling'; just say it, "
                "then let it sit. If it doesn't feel worth saying, say nothing.]")
ASK_NOTE = ("[You've been wanting to {desire}, and it's a good, quiet moment. Bring it up with him in "
            "ONE short, natural sentence — ask or suggest, don't announce that it's a 'desire'. If he's "
            "not into it, let it go easily.]")


def initiative_decision(ctx, cfg=None):
    """Pure: None | 'express' | 'ask' | 'act' | 'chat'. `ctx` is a flat dict of everything that
    matters, so this is fully truth-table testable. The voice LLM may always be MORE reserved than
    this, never less. 'act' fires only when an actionable (safe) desire exists — the caller decides
    what's safe. 'chat' = say something to her SISTER (the house makes it text-only in private mode):
    only when he's away (or it's been quiet twice as long as a chat needs), a sister is connected,
    and she has something on her mind — under its own daily cap."""
    cfg = cfg or INITIATIVE
    perm = ctx.get("permission", "ask")
    if perm == "never":
        return None
    if ctx.get("busy"):
        return None
    lo, hi = cfg["quiet_hours"]
    if lo <= ctx.get("hour", 12) < hi:
        return None
    if ctx.get("secs_since_boot", 1e9) < cfg["boot_grace_s"]:
        return None
    gap = cfg["min_gap_s"] * (0.6 if perm == "always" else 1.0)
    cap = cfg["max_per_day"] + (2 if perm == "always" else 0)
    # v3 drives: worn → she reaches out less; missing him → sooner; restless → she goes and does
    low = dr.DrivesConfig().pressing
    missing = ctx.get("connection", 1.0) < low
    restless = ctx.get("novelty", 1.0) < low
    if ctx.get("rest", 1.0) < low:
        gap *= 1.5
    if ctx.get("initiatives_today", 0) >= cap:
        return None
    if ctx.get("secs_since_last_initiative", 1e9) < gap:
        return None
    idle = ctx.get("voice_idle_s", 0)
    mood_worth = (abs(ctx.get("valence", 0.0)) >= cfg["express_valence"]
                  or ctx.get("social", 0.0) >= cfg["express_social"] or ctx.get("focus")
                  or missing)
    talk_k = 0.6 if missing else 1.0
    act_k = 0.6 if restless else 1.0
    away = ctx.get("present") is False
    if (ctx.get("sister_present") and ctx.get("chats_today", 0) < cfg.get("max_chats_per_day", 3)
            and idle >= cfg.get("idle_chat_s", 600)
            and (away or idle >= 2 * cfg.get("idle_chat_s", 600))
            and (mood_worth or ctx.get("has_askable_desire"))):
        return "chat"
    if away:                                  # away → don't talk to an empty room
        return None
    if perm == "quiet" and not ctx.get("awake"):   # quiet mode never wakes her to speak
        return None
    # doing a safe thing she wants beats just asking about it — but under its own tighter daily cap
    if (ctx.get("has_actionable_desire") and idle >= cfg["idle_act_s"] * act_k
            and ctx.get("acts_today", 0) < cfg["max_acts_per_day"]):
        return "act"
    if ctx.get("has_askable_desire") and idle >= cfg["idle_ask_s"] * talk_k:
        return "ask"
    if mood_worth and idle >= cfg["idle_express_s"] * talk_k:
        return "express"
    return None


class SelfEngine:
    """The one loop that turns raw experience into an inner life. Owns the `journal`, `desires`
    and `self_kv` tables on the shared connection; the mood dials live in InnerState. Degrades
    to mood-only (no diary, no new desires) when no `llm` is wired, and to no-initiative when no
    voice bridge is attached."""

    def __init__(self, conn, bus: ExperienceBus, brain: InnerBrain, *, state: InnerState = None,
                 state_path=None, embed=None, llm=None, model="gpt-5.6-luna", log=print, now=None,
                 journal_salience=2.0, journal_min_gap_h=12.0, journal_max_tokens=500,
                 desire_reinforce=0.15, desire_cap=12, preferences=None, name="Aiva",
                 self_forming=False, becoming=None, kin=None, house_emit=None,
                 drives=True, drives_cfg=None):
        self.conn = conn
        self.drives_on = drives              # v3: needs / rhythm / attachment (modules/drives.py)
        self.drives_cfg = drives_cfg or dr.DrivesConfig()
        self.bus = bus
        self.brain = brain
        self.name = name
        self.self_forming = self_forming     # temperament forms from life (no authored core)
        self.becoming = becoming             # Becoming store: adopt_temperament / core_temperament
        self.kin = kin                       # Kin store: what she notices about her sister
        # house_emit(kind, summary, capability=None): her public acts, relayed to her sister(s)
        self._house_emit = house_emit
        self.journal_prompt = self._render_prompt()
        self.state = state or InnerState.fresh(brain.cfg)
        self.state_path = state_path
        self._embed = embed                 # async (text) -> vec | None
        self._llm = llm                     # async (messages, *, json_mode, max_tokens, temperature) -> reply
        self.model = model
        self.log = log
        self._now = now or (lambda: datetime.now().isoformat())
        self.journal_salience = journal_salience
        self.journal_min_gap_h = journal_min_gap_h
        self.journal_max_tokens = journal_max_tokens
        self.desire_reinforce = desire_reinforce
        self.desire_cap = desire_cap
        self.preferences = preferences     # Preferences store; her likes/dislikes/opinions form here
        self._salience_since_journal = 0.0
        self._last_journal_ts = None
        # voice/prompt bridge (wired in main.py)
        self._refresh_prompt = lambda: None
        self._announce = None
        self._quiet = None
        self._is_awake = lambda: True
        self._idle_for = lambda: 0.0
        self._user_present = lambda within=180: None
        self._task_busy = lambda: False
        # autonomous action (Phase 5), wired in main.py; defaults = she can't act (speech only)
        self._actionable = lambda desire: False   # is this wish safe to pursue on her own?
        self._act = None                          # async (desire) -> None: the safe dispatcher
        # talking to her sister on her own (the house grants and caps it)
        self._want_floor = None                   # (kind, to) -> ask the house for a turn
        self._sister_present = lambda: None       # -> connected sister's slug, or None
        self._house_asleep = lambda: False        # the house's ambient sleep (rest refills)
        self._boot_mono = time.monotonic()
        self._create_tables()

    def _render_prompt(self):
        """The diary prompt for THIS character: the temperament item whenever she has a Becoming
        store to hold it — the self-forming wording when there's no authored character underneath,
        the on-top wording when there is — and the sister item when she has a kin store."""
        if self.becoming is None:
            temperament_clause = ""
        else:
            temperament_clause = TEMPERAMENT_CLAUSE if self.self_forming else TEMPERAMENT_CLAUSE_ON_TOP
        return named(JOURNAL_PROMPT_T, self.name,
                     self_hint=FORMING_SELF_HINT if self.self_forming else AIVA_SELF_HINT,
                     temperament_clause=temperament_clause,
                     sister_clause=SISTER_CLAUSE if self.kin is not None else "")

    def attach(self, *, refresh_prompt=None, preferences=None, actionable=None, act=None,
               becoming=None, kin=None, house_emit=None):
        if refresh_prompt:
            self._refresh_prompt = refresh_prompt
        if preferences is not None:
            self.preferences = preferences
        if actionable is not None:
            self._actionable = actionable
        if act is not None:
            self._act = act
        if becoming is not None:
            self.becoming = becoming
        if kin is not None:
            self.kin = kin
        if house_emit is not None:
            self._house_emit = house_emit
        if becoming is not None or kin is not None:
            self.journal_prompt = self._render_prompt()

    def attach_voice(self, *, announce=None, quiet_note=None, is_awake=None, idle_for=None,
                     user_present=None, task_busy=None, want_floor=None, sister_present=None,
                     house_asleep=None):
        """Wire the voice bridge so she can reach out on her own (Phase 4 initiative).
        `want_floor(kind, to)` asks the house for a turn (a 'chat' with her sister); `sister_present()`
        -> the connected sister's slug or None."""
        if announce:
            self._announce = announce
        if quiet_note:
            self._quiet = quiet_note
        if is_awake:
            self._is_awake = is_awake
        if idle_for:
            self._idle_for = idle_for
        if user_present:
            self._user_present = user_present
        if task_busy:
            self._task_busy = task_busy
        if want_floor:
            self._want_floor = want_floor
        if sister_present:
            self._sister_present = sister_present
        if house_asleep:
            self._house_asleep = house_asleep

    def _get(self, key, default=None):
        r = self.conn.execute("SELECT value FROM self_kv WHERE key = ?", (key,)).fetchone()
        return r[0] if r else default

    def _set(self, key, value):
        self.conn.execute(
            "INSERT INTO self_kv (key, value, updated_ts) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_ts = excluded.updated_ts",
            (key, str(value), self._now()))
        self.conn.commit()

    def _create_tables(self):
        c = self.conn
        c.execute("""CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, entry TEXT NOT NULL,
            valence REAL, arousal REAL, social REAL, focus TEXT,
            salient_events_json TEXT NOT NULL DEFAULT '[]', carry TEXT, embedding BLOB)""")
        c.execute("""CREATE TABLE IF NOT EXISTS desires (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, text TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'do', source TEXT NOT NULL DEFAULT 'journal',
            strength REAL NOT NULL DEFAULT 0.5, status TEXT NOT NULL DEFAULT 'open',
            last_touched_ts TEXT NOT NULL, veto_reason TEXT, embedding BLOB)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_desires_status ON desires(status)")
        c.execute("""CREATE TABLE IF NOT EXISTS self_kv (
            key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_ts TEXT NOT NULL)""")
        # v3: P(he's around) per weekday-hour slot, learned (drives.rhythm_update)
        c.execute("""CREATE TABLE IF NOT EXISTS rhythm (
            slot TEXT PRIMARY KEY, p REAL NOT NULL, n INTEGER NOT NULL, updated_ts TEXT NOT NULL)""")
        c.commit()

    def _persist(self):
        if self.state_path:
            self.state.save(self.state_path)

    async def _vec(self, text):
        if not self._embed:
            return None
        try:
            return await self._embed(text)
        except Exception:
            return None

    # --- the loop step --------------------------------------------------------------

    async def tick(self, context=None, *, force_journal=False):
        """One turn of her inner life: absorb what happened, move her mood, and — if it's
        earned — write a diary entry and form desires. Returns a small status dict."""
        now = self._now()
        events = self.bus.drain()
        _, changed = self.brain.integrate(self.state, events, now)
        if events:
            self.bus.mark_consumed([e["id"] for e in events])
            self._salience_since_journal += sum(
                e["salience"] for e in events if e["salience"] >= self.brain.cfg.salience_floor)
        if self.drives_on:
            try:
                changed = self._drive_step(events, now) or changed
                self._rhythm_step(now)
                self._initiative_feedback(now)
            except Exception as e:
                self.log(f"self: drives step failed ({e!r})")
        journaled = False
        if self.should_journal(now, force=force_journal):
            journaled = await self._digest(events, context, now)
        if changed or journaled:
            self._persist()
            try:
                self._refresh_prompt()
            except Exception:
                pass
        return {"changed": changed, "journaled": journaled, "events": len(events)}

    # --- v3 drives: needs, rhythm, attachment ------------------------------------------

    def _drive_step(self, events, now):
        """Drain her needs since needs_ts, refill/cost them from this batch of events, let pressing
        needs lean on her mood, and emit one `need_low` event when a need starts to press. Returns
        whether anything changed enough to persist / refresh the prompt."""
        s, cfg = self.state, self.drives_cfg
        before = dr.normalise(s.needs)
        hours = _hours_between(s.needs_ts, now)
        hours = min(hours, 12.0)               # she isn't living while she's off: a long gap counts as a night
        asleep = bool(self._house_asleep()) or self._in_quiet_hours(now)
        needs = dr.deplete(before, hours, asleep=asleep, cfg=cfg)
        for e in events or []:
            needs = dr.satisfy(needs, e["kind"], cfg)
        s.needs, s.needs_ts = needs, now
        dv, da, ds = dr.pressure(needs, cfg)
        h = min(hours, 1.0)
        if h > 0 and (dv or da or ds):
            s.valence = _clamp(s.valence + dv * h, -1.0, 1.0)
            s.arousal = _clamp(s.arousal + da * h, 0.0, 1.0)
            s.social = _clamp(s.social + ds * h, -1.0, 1.0)
        crossed = dr.crossings(before, needs, cfg)
        for n in crossed:
            summary, v, a, so = dr.NEED_EVENT[n]
            self.bus.emit("need_low", "drives", summary, valence=v, arousal=a, social=so,
                          salience=0.45, payload={"need": n, "level": round(needs[n], 3)}, now=now)
            self.log(f"self: need pressing — {n} {needs[n]:.2f}")
        return bool(crossed) or any(abs(needs[n] - before[n]) >= 0.05 for n in dr.NEEDS)

    def _in_quiet_hours(self, now):
        try:
            hour = datetime.fromisoformat(now).hour
        except (ValueError, TypeError):
            return False
        lo, hi = INITIATIVE["quiet_hours"]
        return lo <= hour < hi

    def _rhythm_get(self, slot):
        r = self.conn.execute("SELECT p, n FROM rhythm WHERE slot = ?", (slot,)).fetchone()
        return (r[0], r[1]) if r else (None, 0)

    def _rhythm_step(self, now):
        """Sample whether he's around this hour; when the hour rolls over, learn it; part-way into
        an hour, compare against what she's learned to expect and feel the miss (or the surprise)."""
        hour_key = now[:13]
        cur = self._get("rhythm_hour")
        present_now = self._user_present(INITIATIVE["present_within_s"]) is True
        if cur != hour_key:
            if cur:
                try:
                    slot = dr.slot_of(cur + ":00:00")
                    p, n = self._rhythm_get(slot)
                    p, n = dr.rhythm_update(p, n, self._get("rhythm_present") == "1")
                    self.conn.execute(
                        "INSERT INTO rhythm (slot, p, n, updated_ts) VALUES (?,?,?,?) ON CONFLICT(slot) "
                        "DO UPDATE SET p = excluded.p, n = excluded.n, updated_ts = excluded.updated_ts",
                        (slot, p, n, now))
                    self.conn.commit()
                except ValueError:
                    pass
            self._set("rhythm_hour", hour_key)
            self._set("rhythm_present", "0")
            self._set("rhythm_checked", "0")
        if present_now and self._get("rhythm_present") != "1":
            self._set("rhythm_present", "1")
        if self._get("rhythm_checked") == "1":
            return
        try:
            minute = datetime.fromisoformat(now).minute
        except (ValueError, TypeError):
            return
        p, n = self._rhythm_get(dr.slot_of(now))
        ev = dr.expectation_event(p, n, self._get("rhythm_present") == "1", minute)
        if minute >= dr.EXPECT_CHECK_MIN:
            self._set("rhythm_checked", "1")
        if ev == "expected_absent":
            self.bus.emit("expected_absent", "rhythm", "he's usually around by now, and he isn't",
                          valence=-0.08, social=0.15, salience=0.4, payload={"p": round(p, 2)}, now=now)
        elif ev == "unexpected_presence":
            self.bus.emit("unexpected_presence", "rhythm", "he's around at an hour he usually isn't",
                          arousal=0.05, valence=0.03, salience=0.25, payload={"p": round(p, 2)}, now=now)

    def _initiative_feedback(self, now, he_spoke=False):
        """Close the loop on her last reach-out: did he answer (welcomed) or let it drop (ignored)?
        Revealed preference, felt — it's what teaches her when reaching out is welcome."""
        pending = self._get("initiative_pending")
        if not pending:
            return None
        secs = _hours_between(pending, now) * 3600.0
        present = self._user_present(INITIATIVE["present_within_s"]) is True
        outcome = dr.initiative_outcome(secs, he_spoke, present)
        if outcome is None:
            return None
        self._set("initiative_pending", "")
        what = self._get("initiative_pending_what", "reached out")
        if outcome == "welcomed":
            self.bus.emit("initiative_welcomed", "conversation", f"I {what} and he picked it up",
                          valence=0.2, social=0.1, salience=0.45, now=now)
        elif outcome == "ignored":
            self.bus.emit("initiative_ignored", "conversation", f"I {what} and he let it drop",
                          valence=-0.12, social=-0.05, salience=0.4, now=now)
        self._bump_feedback(outcome)
        return outcome

    def _bump_feedback(self, outcome):
        try:
            data = json.loads(self._get("initiative_feedback", "{}") or "{}")
        except ValueError:
            data = {}
        data[outcome] = int(data.get(outcome, 0)) + 1
        self._set("initiative_feedback", json.dumps(data))

    def heard(self, text, *, affect=None, him="him"):
        """He said something to her (a granted user turn). Felt as experience, and returns the
        ephemeral notes for THIS turn: a reunion after a real absence, or — once, a while after a
        sharp moment — a nudge to smooth it over. Called on the loop thread before her reply."""
        if not self.drives_on:
            return []
        now = self._now()
        notes = []
        last = self._get("last_talk_ts")
        hours = _hours_between(last, now) if last else None
        r = dr.reunion(hours)
        if r:
            sal, phrase = r
            self.bus.emit("reunion", "conversation", f"he's back — first talk {phrase}",
                          valence=0.3, social=0.25, arousal=0.1, salience=sal,
                          payload={"hours": round(hours, 1)}, now=now)
            notes.append(dr.reunion_note(phrase, him))
        snippet = (text or "").strip()
        self.bus.emit("conversation", "conversation",
                      f"he said: {snippet[:140]}" if snippet else "he talked to me",
                      valence=0.05, social=0.12, salience=0.2, now=now)
        self._initiative_feedback(now, he_spoke=True)
        if dr.is_friction(snippet, affect):
            self.bus.emit("friction", "conversation", f"a sharp moment — he said: {snippet[:100]}",
                          valence=-0.25, arousal=0.15, salience=0.55, now=now)
            self._set("friction_ts", now)
            self._set("friction_snippet", snippet[:120])
            self._set("friction_noted", "0")
        else:
            fts = self._get("friction_ts")
            if fts and self._get("friction_noted") == "0":
                secs = _hours_between(fts, now) * 3600.0
                if dr.REPAIR_AFTER_S <= secs <= dr.REPAIR_WITHIN_S:
                    notes.append(dr.repair_note(self._get("friction_snippet", "")))
                    self._set("friction_noted", "1")
                    self.bus.emit("repair", "conversation", "we got past the tense moment",
                                  valence=0.15, social=0.1, salience=0.35, now=now)
                elif secs > dr.REPAIR_WITHIN_S:
                    self._set("friction_noted", "1")
        self._set("last_talk_ts", now)
        self._set("rhythm_present", "1")
        return notes

    def needs_public(self):
        n = dr.normalise(self.state.needs)
        return {"needs": {k: round(v, 3) for k, v in n.items()}, "pressing": dr.pressing(n, self.drives_cfg),
                "tension": dr.tension(n, self.drives_cfg)}

    def should_journal(self, now, force=False):
        if not self._llm:
            return False
        if force:
            return True
        if self._salience_since_journal >= self.journal_salience:
            return True
        if self._last_journal_ts:
            gap_h = _hours_between(self._last_journal_ts, now)
            if gap_h >= self.journal_min_gap_h and self._salience_since_journal > 0:
                return True
        return False

    async def _digest(self, events, context, now):
        """One cheap side-model call: write the diary entry + form desires. Mood stays
        driven by events (no double-counting); the journal only NAMES her focus and records
        the current mood alongside the entry. Any failure falls back to a plain entry so the
        diary never has a hole."""
        user = self._digest_input(events, context)
        data, ok = None, False
        try:
            reply = await self._llm([{"role": "system", "content": self.journal_prompt},
                                     {"role": "user", "content": user}],
                                    json_mode=True, max_tokens=self.journal_max_tokens,
                                    temperature=0.6)
            data = json.loads(reply.text or "")
            ok = isinstance(data, dict) and bool(str(data.get("entry", "")).strip())
        except Exception as e:
            self.log(f"self: journal call failed ({e!r}) — writing a plain entry")
        if ok:
            entry = str(data["entry"]).strip()[:1200]
            focus = data.get("focus")
            focus = str(focus).strip()[:120] if focus else None
            carry = data.get("carry")
            carry = str(carry).strip()[:240] if carry else None
            raw_desires = data.get("desires") if isinstance(data.get("desires"), list) else []
            raw_prefs = data.get("preferences") if isinstance(data.get("preferences"), list) else []
            raw_temp = data.get("temperament") if isinstance(data.get("temperament"), list) else []
            raw_sister = data.get("sister_notes") if isinstance(data.get("sister_notes"), list) else []
        else:
            entry = self._fallback_entry(events)
            focus, carry, raw_desires, raw_prefs = None, None, [], []
            raw_temp, raw_sister = [], []

        if focus:
            self.state.focus, self.state.focus_ts = focus, now
        vec = await self._vec(entry)
        salient = [{"source": e["source"], "kind": e["kind"], "summary": e["summary"]}
                   for e in (events or []) if e["salience"] >= self.brain.cfg.salience_floor][:8]
        self.conn.execute(
            "INSERT INTO journal (ts, entry, valence, arousal, social, focus, salient_events_json, "
            "carry, embedding) VALUES (?,?,?,?,?,?,?,?,?)",
            (now, entry, round(self.state.valence, 3), round(self.state.arousal, 3),
             round(self.state.social, 3), focus, json.dumps(salient, ensure_ascii=False), carry,
             _pack_vec(vec)))
        self.conn.commit()

        for d in raw_desires[:4]:
            text = str(d.get("text", "")).strip() if isinstance(d, dict) else ""
            kind = d.get("kind") if isinstance(d, dict) else None
            if text and kind in DESIRE_KINDS:
                if not desire_is_outward(text):
                    self.log(f"self: inward wish skipped ({text[:80]!r})")
                    continue
                await self.add_desire(text[:160], kind, source="journal")

        prefs_applied = await self._apply_preferences(raw_prefs)
        temp_applied = await self._apply_temperament(raw_temp)
        sister_applied = await self._apply_sister_notes(raw_sister)

        self._salience_since_journal = 0.0
        self._last_journal_ts = now
        self.log(f"self: journal entry ({len(entry)} chars, {len(raw_desires)} desires, "
                 f"{prefs_applied} prefs, {temp_applied} temperament, {sister_applied} sister notes, "
                 f"focus={focus!r})")
        return True

    async def _apply_temperament(self, raw_temp):
        """The digest's temperament items → becoming.adopt_temperament (validated, adopted low,
        reinforced, or eroding a trait she formed herself). An authored character takes part too:
        becoming drops anything that cuts against her written floor, so living can only add."""
        if not (self.becoming is not None and raw_temp):
            return 0
        try:
            res = await self.becoming.adopt_temperament(raw_temp[:2])
        except Exception as e:
            self.log(f"self: temperament adopt failed ({e!r})")
            return 0
        return len(res.get("adopted", [])) + len(res.get("reinforced", [])) + len(res.get("eroded", []))

    async def _apply_sister_notes(self, raw_sister):
        """What she noticed about her sister today → kin.note(kind, text). No-op without a kin store."""
        if self.kin is None or not raw_sister:
            return 0
        n = 0
        for s in raw_sister[:2]:
            if not isinstance(s, dict):
                continue
            text = str(s.get("text", "")).strip()
            kind = s.get("kind") if s.get("kind") in SISTER_NOTE_KINDS else "did"
            if not text:
                continue
            try:
                res = self.kin.note(kind, text[:160])
                if hasattr(res, "__await__"):
                    res = await res
                if res is not None:
                    n += 1
            except Exception as e:
                self.log(f"self: sister note failed ({e!r})")
        return n

    async def _apply_preferences(self, raw_prefs):
        """Turn the digest's preference deltas into moves on the mutable likes/dislikes/opinions
        store. No-op (returns 0) when no store is wired."""
        if not self.preferences or not raw_prefs:
            return 0
        n = 0
        for p in raw_prefs[:3]:
            if not isinstance(p, dict):
                continue
            subject = str(p.get("subject", "")).strip()
            stance = str(p.get("stance", "")).strip()
            if not subject or not stance:
                continue
            kind = p.get("kind") if p.get("kind") in ("taste", "opinion") else "taste"
            try:
                direction = 1 if float(p.get("direction", 1)) >= 0 else -1
            except (TypeError, ValueError):
                direction = 1
            try:
                magnitude = float(p.get("magnitude", 0.5))
            except (TypeError, ValueError):
                magnitude = 0.5
            try:
                res = await self.preferences.apply(subject[:80], stance[:200], kind=kind,
                                                   direction=direction, magnitude=magnitude,
                                                   source="sister" if p.get("from_sister") is True
                                                   else "reflection")
                if res not in ("blocked", "rejected"):
                    n += 1
            except Exception as e:
                self.log(f"self: preference apply failed ({e!r})")
        return n

    def _digest_input(self, events, context):
        s = self.state
        lines = [f"HOW YOU FEEL: {mood_word(s)} "
                 f"(valence {s.valence:+.2f}, arousal {s.arousal:.2f}, social {s.social:+.2f})."]
        if self.drives_on:
            n = dr.normalise(s.needs)
            lines.append("WHAT YOU NEED (0 starved .. 1 met): " + ", ".join(
                f"{k} {n[k]:.2f}" for k in dr.NEEDS) + ". " + (dr.needs_line(n, self.drives_cfg)
                                                              or "Nothing's pressing."))
        salient = [e for e in (events or []) if e["salience"] >= self.brain.cfg.salience_floor]
        if salient:
            lines.append("WHAT HAPPENED (recent):")
            for e in salient[:10]:
                lines.append(f"- [{e['source']}/{e['kind']}] {e['summary']}")
        if context:
            lines.append(f"RECENTLY, WITH HIM: {str(context)[:800]}")
        open_wants = self.list_desires(status="open", limit=8)
        if open_wants:
            lines.append("YOU ALREADY WANT (don't just repeat these): "
                         + "; ".join(d["text"] for d in open_wants))
        if self.preferences:
            try:
                pb = self.preferences.prompt_block(budget=6)
                cur = pb.get("likes", []) + pb.get("dislikes", []) + pb.get("opinions", [])
                if cur:
                    lines.append("WHAT YOU CURRENTLY LIKE/THINK (you can change your mind — update "
                                 "these, don't just restate them): " + "; ".join(cur[:10]))
            except Exception:
                pass
        if self.becoming is not None:
            try:
                core = self.becoming.core_temperament()
            except Exception:
                core = []
            if self.self_forming and core:
                lines.append("WHO YOU ARE SO FAR (reinforce what still rings true, don't restate it): "
                             + "; ".join(core))
            elif self.self_forming:
                lines.append("WHO YOU ARE SO FAR: nothing settled yet — you're still finding out.")
            elif core:
                # authored character: only what living has added, so she reinforces instead of
                # rewriting, and nothing prompts her to invent a first one on a quiet day
                lines.append("WHAT LIVING HAS ADDED TO YOU SO FAR, on top of the character you "
                             "already are (reinforce what still rings true, don't restate it): "
                             + "; ".join(core))
        if self.kin is not None:
            try:
                kl = self.kin.prompt_block(limit=4)
            except Exception:
                kl = []
            if kl:
                lines.append("WHAT YOU'VE NOTICED ABOUT YOUR SISTER SO FAR: " + "; ".join(kl))
        return "\n".join(lines)

    def _fallback_entry(self, events):
        salient = [e["summary"] for e in (events or [])
                   if e["salience"] >= self.brain.cfg.salience_floor and e["summary"]]
        if salient:
            return "A quiet note to myself: " + "; ".join(salient[:3]) + "."
        return "A flat, quiet stretch. Nothing much to say to myself tonight."

    # --- desires: her own wishes (deduped + vetoable, like reading traits) ------------

    async def add_desire(self, text, kind="do", source="journal", strength=0.5):
        """Insert a wish, or reinforce a near-duplicate. A vetoed (dropped) wish is never
        re-formed. Returns 'inserted' | 'reinforced' | 'blocked'."""
        if kind not in DESIRE_KINDS:
            kind = "do"
        vec = await self._vec(text)
        dup = self._dedupe_desire(text, vec)
        now = self._now()
        if dup:
            if dup["status"] == "dropped":
                return "blocked"
            self.conn.execute(
                "UPDATE desires SET strength = MIN(1.0, strength + ?), last_touched_ts = ?, "
                "status = CASE WHEN status IN ('acted','fulfilled') THEN status ELSE 'open' END "
                "WHERE id = ?", (self.desire_reinforce, now, dup["id"]))
            self.conn.commit()
            return "reinforced"
        self.conn.execute(
            "INSERT INTO desires (ts, text, kind, source, strength, last_touched_ts, embedding) "
            "VALUES (?,?,?,?,?,?,?)",
            (now, text, kind, source, float(strength), now, _pack_vec(vec)))
        self.conn.commit()
        self._enforce_desire_cap()
        return "inserted"

    def _dedupe_desire(self, text, vec):
        rows = self.conn.execute("SELECT id, text, status, embedding FROM desires").fetchall()
        for rid, t, status, emb in rows:
            other = _unpack_vec(emb)
            if vec and other:
                score, thresh = _cosine(vec, other), DESIRE_DEDUPE_COSINE
            else:
                score, thresh = _jaccard(text, t), DESIRE_DEDUPE_JACCARD
            if score >= thresh:
                return {"id": rid, "status": status}
        return None

    def _enforce_desire_cap(self):
        """Keep the open set bounded — fade the weakest open wish over the cap."""
        rows = self.conn.execute(
            "SELECT id FROM desires WHERE status = 'open' ORDER BY strength DESC, id DESC").fetchall()
        for (rid,) in rows[self.desire_cap:]:
            self.conn.execute("UPDATE desires SET status = 'dropped', veto_reason = 'faded (over cap)' "
                              "WHERE id = ?", (rid,))
        if len(rows) > self.desire_cap:
            self.conn.commit()

    async def drop_desire(self, text_or_id, reason=None):
        """Veto a wish (id -> substring -> embedding/lexical). Dropped forever. Returns the
        matched text or None."""
        row = None
        if isinstance(text_or_id, int) or (isinstance(text_or_id, str) and text_or_id.isdigit()):
            row = self.conn.execute("SELECT id, text FROM desires WHERE id = ?",
                                    (int(text_or_id),)).fetchone()
        if not row:
            q = str(text_or_id or "").strip().lower()
            if q:
                for rid, t in self.conn.execute(
                        "SELECT id, text FROM desires WHERE status != 'dropped'").fetchall():
                    if q in t.lower():
                        row = (rid, t)
                        break
        if not row:
            vec = await self._vec(str(text_or_id))
            best, best_score = None, 0.0
            for rid, t, emb in self.conn.execute(
                    "SELECT id, text, embedding FROM desires WHERE status != 'dropped'").fetchall():
                other = _unpack_vec(emb)
                score = _cosine(vec, other) if (vec and other) else _jaccard(str(text_or_id), t)
                if score > best_score:
                    best, best_score = (rid, t), score
            if best and best_score >= 0.6:
                row = best
        if not row:
            return None
        self.conn.execute("UPDATE desires SET status = 'dropped', veto_reason = ?, last_touched_ts = ? "
                          "WHERE id = ?", (reason, self._now(), row[0]))
        self.conn.commit()
        try:
            self._refresh_prompt()
        except Exception:
            pass
        return row[1]

    def list_desires(self, status="open", limit=20):
        where = "WHERE status = ?" if status else ""
        params = (status,) if status else ()
        rows = self.conn.execute(
            f"SELECT id, text, kind, source, strength, status, ts, last_touched_ts, veto_reason "
            f"FROM desires {where} ORDER BY strength DESC, id DESC LIMIT ?",
            (*params, limit)).fetchall()
        return [{"id": r[0], "text": r[1], "kind": r[2], "source": r[3], "strength": round(r[4], 3),
                 "status": r[5], "ts": r[6], "last_touched_ts": r[7], "veto_reason": r[8]}
                for r in rows]

    # --- journal recall (for the voice tool) -----------------------------------------

    async def recall_journal(self, query=None, k=3):
        rows = self.conn.execute(
            "SELECT ts, entry, focus, embedding FROM journal ORDER BY id DESC LIMIT 200").fetchall()
        if not rows:
            return []
        if not query:
            return [{"ts": r[0], "entry": r[1], "focus": r[2]} for r in rows[:k]]
        qv = await self._vec(query)
        scored = []
        for ts, entry, focus, emb in rows:
            other = _unpack_vec(emb)
            score = _cosine(qv, other) if (qv and other) else _jaccard(query, entry)
            scored.append((score, {"ts": ts, "entry": entry, "focus": focus}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]

    def prompt_data(self, wants_limit=3):
        """The two persona inputs: a one-line 'how you feel' and her top open wishes. Kept tiny
        (~15 tokens + the wishes) — persona.py wraps them with the anti-announce headers."""
        s = self.state
        line = f"You feel {mood_word(s)}."
        if s.focus:
            line += f" You keep coming back to {s.focus}."
        if self.drives_on:
            nl = dr.needs_line(dr.normalise(s.needs), self.drives_cfg)
            if nl:
                line += " " + nl
        wants = [d["text"] for d in self.list_desires(status="open", limit=wants_limit)]
        return {"mood_line": line, "wants": wants}

    def journal_entries(self, limit=10):
        rows = self.conn.execute(
            "SELECT ts, entry, valence, arousal, social, focus, carry FROM journal "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": r[0], "entry": r[1], "valence": r[2], "arousal": r[3], "social": r[4],
                 "focus": r[5], "carry": r[6]} for r in rows]

    # --- initiative: reaching out on her own (Phase 4) -------------------------------

    def _top_desire(self):
        d = self.list_desires(status="open", limit=1)
        return d[0] if d else None

    def _secs_since(self, key):
        ts = self._get(key)
        return _hours_between(ts, self._now()) * 3600.0 if ts else 1e9

    def _day_count(self, key="initiative_day"):
        raw = self._get(key, "") or ""
        today = self._now()[:10]
        if raw.startswith(today + ":"):
            try:
                return int(raw.split(":", 1)[1])
            except ValueError:
                return 0
        return 0

    def _bump_day_count(self, now, key="initiative_day"):
        self._set(key, f"{now[:10]}:{self._day_count(key) + 1}")

    def _actionable_desire(self):
        """The top open wish, if it's one she may safely pursue on her own; else None."""
        top = self._top_desire()
        if top and self._act is not None:
            try:
                if self._actionable(top):
                    return top
            except Exception:
                return None
        return None

    def set_initiative(self, mode):
        mode = (mode or "").strip().lower()
        if mode not in ("always", "ask", "quiet", "never"):
            return {"success": False, "error": "mode must be always, ask, quiet, or never"}
        self._set("initiative", mode)
        try:
            self._refresh_prompt()
        except Exception:
            pass
        return {"success": True, "mode": mode}

    def _initiative_ctx(self):
        try:
            hour = datetime.fromisoformat(self._now()).hour
        except (ValueError, TypeError):
            hour = 12
        return {
            "permission": self._get("initiative", "ask"),
            "busy": bool(self._task_busy()),
            "awake": bool(self._is_awake()),
            "present": self._user_present(INITIATIVE["present_within_s"]),
            "voice_idle_s": self._idle_for(),
            "hour": hour,
            "valence": self.state.valence, "arousal": self.state.arousal, "social": self.state.social,
            "focus": bool(self.state.focus),
            "has_askable_desire": self._top_desire() is not None,
            "has_actionable_desire": self._actionable_desire() is not None,
            "acts_today": self._day_count("act_day"),
            "secs_since_last_initiative": self._secs_since("last_initiative_ts"),
            "initiatives_today": self._day_count(),
            "secs_since_boot": time.monotonic() - self._boot_mono,
            "sister_present": bool(self._want_floor is not None and self._sister_present()),
            "chats_today": self._day_count("chat_day"),
            **(dr.normalise(self.state.needs) if self.drives_on else {}),
        }

    async def initiative_tick(self):
        """Decide whether to reach out; if so, record it (BEFORE announcing, so a same-turn reply
        can't double-fire) and speak up. No-op without a voice bridge. Returns what she did or None."""
        if not self._announce:
            return None
        what = initiative_decision(self._initiative_ctx())
        if not what:
            return None
        now = self._now()
        self._set("last_initiative_ts", now)
        self._bump_day_count(now)
        if what != "chat":
            d = self._top_desire()
            self._set("initiative_pending", now)
            self._set("initiative_pending_what", {
                "express": "said what was on my mind",
                "ask": f"brought up wanting to {d['text']}" if d else "brought something up",
                "act": "went after something I wanted and told him"}.get(what, "reached out"))
        if what == "act":
            return await self._do_act(now)
        if what == "chat":
            sister = self._sister_present()
            if not sister or self._want_floor is None:
                return None
            self._bump_day_count(now, "chat_day")
            try:
                res = self._want_floor("initiative", sister)
                if hasattr(res, "__await__"):
                    await res
            except Exception as e:
                self.log(f"self initiative chat failed: {e!r}")
            self.log(f"self initiative: chat -> {sister}")
            return what
        if what == "ask":
            d = self._top_desire()
            note = ASK_NOTE.format(desire=d["text"] if d else "something you've been meaning to bring up")
        else:
            focus = self.state.focus
            note = EXPRESS_NOTE.format(focus=(f" — you keep coming back to {focus}" if focus else ""))
        try:
            res = self._announce(note)
            if hasattr(res, "__await__"):     # tolerate sync or async announce
                await res
        except Exception as e:
            self.log(f"self initiative announce failed: {e!r}")
        self.log(f"self initiative: {what}")
        return what

    async def _do_act(self, now):
        """She pursues a safe wish herself. The desire is consumed and felt (a self_act experience
        event → mood/diary/preferences) BEFORE dispatch, so she doesn't re-pick it and the doing
        actually shapes her. The injected `_act` does the work + tells him about it."""
        d = self._actionable_desire()
        if not d:
            return None
        self._bump_day_count(now, "act_day")
        self.conn.execute("UPDATE desires SET status='acted', last_touched_ts=? WHERE id=?",
                          (now, d["id"]))
        self.conn.commit()
        try:
            self.bus.emit("self_act", "self", f"went after something I wanted: {d['text']}",
                          valence=0.25, arousal=0.1, salience=0.6)
        except Exception as e:
            self.log(f"self_act emit failed: {e!r}")
        if self._house_emit is not None:      # a public act: her sister may notice (kin)
            try:
                from modules import capabilities
                cap = capabilities.match(d["text"])
                res = self._house_emit("self_act", d["text"], capability=cap.name if cap else None)
                if hasattr(res, "__await__"):
                    await res
            except Exception as e:
                self.log(f"self_act house emit failed: {e!r}")
        try:
            res = self._act(d)
            if hasattr(res, "__await__"):
                await res
        except Exception as e:
            self.log(f"self act failed: {e!r}")
        self.log(f"self initiative: act ({d['text']!r})")
        return "act"


# --- audit / smoke ------------------------------------------------------------------

def _show():
    """Read-only dump of her current mood + recent events. Safe while she's running."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.getenv("AIVA_INNER_STATE") or os.path.join(here, "inner_state.json")
    brain = InnerBrain()
    if os.path.exists(path):
        state = InnerState.load(path, brain)
        print("How she feels right now:")
        for k, v in state.to_public().items():
            print(f"  {k:12} {v}")
    else:
        print(f"(no inner_state.json yet at {path} — she hasn't felt anything)")
    db = os.path.join(here, "memory.db")
    if os.path.exists(db):
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            con.execute("PRAGMA busy_timeout=2000")
            bus = ExperienceBus.__new__(ExperienceBus)
            bus.conn = con
            rows = bus.recent(15)
            if rows:
                print("\nRecent experience:")
                for r in rows:
                    print(f"  {r['ts'][11:19]}  [{r['source']}/{r['kind']}] {r['summary']}"
                          f"  (v{r['valence_hint']:+.2f} a{r['arousal_hint']:+.2f} "
                          f"s{r['social_hint']:+.2f} sal{r['salience']:.2f})")
            else:
                print("\n(no experience logged yet)")
            con.close()
        except sqlite3.Error as e:
            print(f"(couldn't read experience: {e})")


if __name__ == "__main__":
    import sys
    if "--show" in sys.argv:
        _show()
    else:
        print(__doc__)
