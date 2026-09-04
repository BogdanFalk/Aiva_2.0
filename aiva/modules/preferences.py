"""Her own likes, dislikes, and opinions — formed from experience, and genuinely mutable.

Unlike `becoming` (reading-only, and forbidden from contradicting the frozen baseline), this layer:
  - carries SIGNED polarity (a real dislike, not just an absent like),
  - covers TASTES ("I love quiet character-driven stories") and OPINIONS ("I think nihilism is
    overrated") she'll voice and stand by,
  - forms from ANY experience (via the self-loop journal digest, which drains the experience bus),
  - is FULLY MUTABLE — the frozen baseline is seeded here as a starting point and can drift or even
    reverse over time.

The coherence guardrail is INERTIA, not a veto: a preference erodes toward neutral and only flips
after sustained counter-experience, and baseline-seeded ones move ~half as fast — so she evolves
rather than flip-flops. Her temperament/voice stay frozen in PERSONALITY; only tastes/opinions live
here. Everything is capped, fades, vetoable-forever, bounded against harmful content, and shown on
the dashboard.

Mirrors the becoming.traits / inner.desires template (sqlite on the shared conn, embedding-deduped,
strength-capped). Stdlib-only helpers; the embedder is injected.

Audit:  cd aiva && ..\\.venv\\Scripts\\python -m modules.preferences --show
"""

import math
import os
import re
import sqlite3
import struct
from datetime import datetime

PREFERENCE_KINDS = {"taste", "opinion"}

DEDUPE_COSINE = 0.86
DEDUPE_JACCARD = 0.60
ADOPT_CAP = 0.6            # one event can't create a strong preference; only reinforcement grows it
REINFORCE_STEP = 0.15     # same-direction: move toward the extreme
ERODE_STEP_BASE = 0.16    # opposite-direction: erode toward 0 (halved for baseline-origin)
MAX_ACTIVE = 24
FADE_FLOOR = 0.2
FADE_HALF_LIFE_DAYS = 120
FADE_HALF_LIFE_BASELINE = 240
FADE_EVERY_HOURS = 20

# harmful content never becomes an opinion, however she got there (mirrors becoming._BOUNDED)
_BOUNDED = re.compile(
    r"\b(convert(ed|ing)?|my (faith|religion)|all (men|women|blacks|whites|jews|muslims|christians|"
    r"gays|\w+ people)|hate|despise|inferior|superior race|kill|genocide)\b", re.I)

# her fixed tastes become the starting seed here (origin='baseline'), then they evolve like any other
_BASELINE_SEED = [
    ("cookies", "I love cookies, especially when a reward's coming", 0.7),
    ("cats", "I adore cats", 0.7),
    ("dogs", "I lean cats over dogs", -0.2),
    ("nature", "I'm fascinated by nature", 0.7),
    ("cozy indoors", "I love a cozy indoors", 0.6),
    ("Japanese culture", "I've a subtle soft spot for Japanese culture", 0.5),
    ("electronic, rock and asian music", "I love electronic, rock and asian music", 0.7),
    ("learning new things", "I love learning new things", 0.7),
]


def _now():
    return datetime.now().isoformat()


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _pack_vec(vec):
    return struct.pack(f"<{len(vec)}f", *vec) if vec else None


def _unpack_vec(blob):
    return list(struct.unpack(f"<{len(blob) // 4}f", blob)) if blob else None


def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _norm(text):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split())


def _jaccard(a, b):
    sa, sb = set(_norm(a).split()), set(_norm(b).split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def _hours_between(then_iso, now_iso):
    try:
        then = datetime.fromisoformat(then_iso)
        now = datetime.fromisoformat(now_iso) if isinstance(now_iso, str) else now_iso
        return max(0.0, (now - then).total_seconds() / 3600.0)
    except (ValueError, TypeError):
        return 0.0


def evolve(valence, evidence, origin, direction):
    """PURE inertia core. Given the current signed `valence`, its `evidence_count`, its `origin`, and
    a new experience's `direction` (+1 like/agree, -1 dislike/disagree), return (new_valence,
    new_evidence). Same-direction reinforces toward the extreme; opposite erodes toward 0 first and
    only flips after sustained counter-evidence — slower for baseline-origin and for well-evidenced
    preferences. Testable with plain numbers."""
    direction = 1 if direction >= 0 else -1
    same = valence == 0 or (valence > 0) == (direction > 0)
    if same:
        new = valence + REINFORCE_STEP * (1 - abs(valence)) * direction
        return _clamp(new, -1.0, 1.0), evidence + 1
    step = ERODE_STEP_BASE * (0.5 if origin == "baseline" else 1.0)
    step = step / (1 + 0.2 * max(0, evidence - 1))   # more conviction = more inertia
    new = _clamp(valence + step * direction, -1.0, 1.0)
    flipped = abs(new) > 1e-9 and (new > 0) != (valence > 0)
    return new, (1 if flipped else max(1, evidence - 1))


class Preferences:
    def __init__(self, conn: sqlite3.Connection, *, embed=None, log=print, now=None):
        self.conn = conn
        self._embed = embed            # async (text) -> vec | None
        self.log = log
        self._now = now or _now
        self._refresh_prompt = lambda: None
        self._create_tables()

    def attach(self, *, refresh_prompt=None):
        if refresh_prompt:
            self._refresh_prompt = refresh_prompt

    def _create_tables(self):
        self.conn.execute("""CREATE TABLE IF NOT EXISTS preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT NOT NULL, stance TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'taste', valence REAL NOT NULL DEFAULT 0.0,
            evidence_count INTEGER NOT NULL DEFAULT 1, origin TEXT NOT NULL DEFAULT 'reflection',
            status TEXT NOT NULL DEFAULT 'active', created_ts TEXT NOT NULL, updated_ts TEXT NOT NULL,
            last_reinforced_ts TEXT NOT NULL, veto_reason TEXT, embedding BLOB)""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pref_status ON preferences(status)")
        self.conn.commit()

    async def _vec(self, text):
        if not self._embed:
            return None
        try:
            return await self._embed(text)
        except Exception:
            return None

    def _match(self, subject, vec):
        """Existing preference this subject duplicates (any status), or None."""
        rows = self.conn.execute(
            "SELECT id, subject, status, valence, evidence_count, origin, embedding "
            "FROM preferences").fetchall()
        for rid, subj, status, val, ev, origin, emb in rows:
            other = _unpack_vec(emb)
            if vec and other:
                score, thresh = _cosine(vec, other), DEDUPE_COSINE
            else:
                score, thresh = _jaccard(subject, subj), DEDUPE_JACCARD
            if score >= thresh:
                return {"id": rid, "status": status, "valence": val,
                        "evidence": ev, "origin": origin}
        return None

    async def apply(self, subject, stance, *, kind="taste", direction=1, magnitude=0.5,
                    source="reflection"):
        """Form or evolve a preference from one experience. Returns
        'inserted'|'reinforced'|'eroded'|'flipped'|'blocked'|'rejected'."""
        subject = (subject or "").strip()
        stance = (stance or "").strip()
        if not subject or not stance:
            return "rejected"
        if kind not in PREFERENCE_KINDS:
            kind = "taste"
        if _BOUNDED.search(stance):
            self.log(f"preference rejected (bounded): {stance!r}")
            return "rejected"
        direction = 1 if direction >= 0 else -1
        magnitude = _clamp(abs(magnitude), 0.0, 1.0)
        vec = await self._vec(subject)
        dup = self._match(subject, vec)
        now = self._now()
        if dup is None:
            valence = _clamp(direction * magnitude, -ADOPT_CAP, ADOPT_CAP)
            self.conn.execute(
                "INSERT INTO preferences (subject, stance, kind, valence, evidence_count, origin, "
                "created_ts, updated_ts, last_reinforced_ts, embedding) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (subject, stance, kind, valence, 1, source, now, now, now, _pack_vec(vec)))
            self.conn.commit()
            self._enforce_cap()
            return "inserted"
        if dup["status"] == "dropped":
            return "blocked"                         # vetoed → never re-forms
        old = dup["valence"]
        new_val, new_ev = evolve(old, dup["evidence"], dup["origin"], direction)
        self.conn.execute(
            "UPDATE preferences SET valence = ?, evidence_count = ?, stance = ?, kind = ?, "
            "status = 'active', updated_ts = ?, last_reinforced_ts = ? WHERE id = ?",
            (new_val, new_ev, stance, kind, now, now, dup["id"]))
        self.conn.commit()
        if abs(new_val) > 1e-9 and (new_val > 0) != (old > 0):
            return "flipped"
        return "reinforced" if abs(new_val) >= abs(old) else "eroded"

    def seed_from_baseline(self):
        """Seed her frozen tastes as origin='baseline' preferences, once. Idempotent — no-op if any
        baseline row already exists."""
        if self.conn.execute(
                "SELECT 1 FROM preferences WHERE origin = 'baseline' LIMIT 1").fetchone():
            return 0
        now = self._now()
        n = 0
        for subject, stance, valence in _BASELINE_SEED:
            self.conn.execute(
                "INSERT INTO preferences (subject, stance, kind, valence, evidence_count, origin, "
                "created_ts, updated_ts, last_reinforced_ts) VALUES (?,?,?,?,?,?,?,?,?)",
                (subject, stance, "taste", valence, 2, "baseline", now, now, now))
            n += 1
        self.conn.commit()
        return n

    def _enforce_cap(self):
        rows = self.conn.execute(
            "SELECT id FROM preferences WHERE status = 'active' "
            "ORDER BY ABS(valence) DESC, last_reinforced_ts DESC").fetchall()
        for (rid,) in rows[MAX_ACTIVE:]:
            self.conn.execute("UPDATE preferences SET status = 'faded' WHERE id = ?", (rid,))
        if len(rows) > MAX_ACTIVE:
            self.conn.commit()

    def fade(self, now=None):
        """Decay magnitude toward 0 over a half-life (baseline slower); below the floor → faded.
        Reinforcement revives. Runs at most every FADE_EVERY_HOURS."""
        now = now or self._now()
        last = self._get_kv("last_fade_ts")
        if last and _hours_between(last, now) < FADE_EVERY_HOURS:
            return
        changed = 0
        for rid, val, origin, ts, status in self.conn.execute(
                "SELECT id, valence, origin, last_reinforced_ts, status FROM preferences "
                "WHERE status IN ('active','faded')").fetchall():
            days = _hours_between(ts, now) / 24.0
            hl = FADE_HALF_LIFE_BASELINE if origin == "baseline" else FADE_HALF_LIFE_DAYS
            new = val * (0.5 ** (days / hl))
            new_status = "faded" if abs(new) < FADE_FLOOR else "active"
            if abs(new - val) > 1e-4 or new_status != status:
                self.conn.execute("UPDATE preferences SET valence = ?, status = ? WHERE id = ?",
                                  (new, new_status, rid))
                changed += 1
        self._set_kv("last_fade_ts", now)
        if changed:
            self.conn.commit()

    async def drop(self, what, reason=None):
        """Veto a preference (id → substring → embedding/lexical). Permanent. Returns the stance."""
        row = None
        if isinstance(what, int) or (isinstance(what, str) and what.isdigit()):
            row = self.conn.execute("SELECT id, stance FROM preferences WHERE id = ?",
                                    (int(what),)).fetchone()
        if not row:
            q = str(what or "").strip().lower()
            for rid, subj, stance in self.conn.execute(
                    "SELECT id, subject, stance FROM preferences WHERE status != 'dropped'").fetchall():
                if q and (q in subj.lower() or q in stance.lower()):
                    row = (rid, stance)
                    break
        if not row:
            vec = await self._vec(str(what))
            best, best_s = None, 0.0
            for rid, subj, stance, emb in self.conn.execute(
                    "SELECT id, subject, stance, embedding FROM preferences "
                    "WHERE status != 'dropped'").fetchall():
                other = _unpack_vec(emb)
                s = _cosine(vec, other) if (vec and other) else _jaccard(str(what), subj)
                if s > best_s:
                    best, best_s = (rid, stance), s
            if best and best_s >= 0.6:
                row = best
        if not row:
            return None
        self.conn.execute(
            "UPDATE preferences SET status = 'dropped', veto_reason = ?, updated_ts = ? WHERE id = ?",
            (reason, self._now(), row[0]))
        self.conn.commit()
        try:
            self._refresh_prompt()
        except Exception:
            pass
        return row[1]

    def list_prefs(self, status="active", limit=50):
        where = "WHERE status = ?" if status else ""
        params = (status,) if status else ()
        rows = self.conn.execute(
            f"SELECT id, subject, stance, kind, valence, evidence_count, origin, status "
            f"FROM preferences {where} ORDER BY ABS(valence) DESC, last_reinforced_ts DESC LIMIT ?",
            (*params, limit)).fetchall()
        return [{"id": r[0], "subject": r[1], "stance": r[2], "kind": r[3],
                 "valence": round(r[4], 3), "evidence": r[5], "origin": r[6], "status": r[7]}
                for r in rows]

    def prompt_block(self, budget=10):
        """Persona lines: her strongest current likes, dislikes, and opinions (by |valence|)."""
        active = [p for p in self.list_prefs("active", limit=budget * 2) if abs(p["valence"]) >= 0.15]
        likes = [p["stance"] for p in active if p["kind"] == "taste" and p["valence"] > 0][:budget]
        dislikes = [p["stance"] for p in active if p["kind"] == "taste" and p["valence"] < 0][:budget]
        opinions = [p["stance"] for p in active if p["kind"] == "opinion"][:budget]
        return {"likes": likes, "dislikes": dislikes, "opinions": opinions}

    # kv on the same table's side (tiny; reuse a dedicated table)
    def _get_kv(self, key):
        self.conn.execute("CREATE TABLE IF NOT EXISTS pref_kv (key TEXT PRIMARY KEY, value TEXT)")
        r = self.conn.execute("SELECT value FROM pref_kv WHERE key = ?", (key,)).fetchone()
        return r[0] if r else None

    def _set_kv(self, key, value):
        self.conn.execute("CREATE TABLE IF NOT EXISTS pref_kv (key TEXT PRIMARY KEY, value TEXT)")
        self.conn.execute("INSERT INTO pref_kv (key, value) VALUES (?, ?) "
                          "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))
        self.conn.commit()


def _show():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db = os.path.join(here, "memory.db")
    if not os.path.exists(db):
        print("(no memory.db yet)")
        return
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT subject, stance, kind, valence, origin, status FROM preferences "
                           "ORDER BY ABS(valence) DESC").fetchall()
    except sqlite3.Error:
        print("(no preferences table yet)")
        return
    if not rows:
        print("(no preferences yet)")
        return
    for subj, stance, kind, val, origin, status in rows:
        bar = ("+" if val >= 0 else "-") * max(1, int(abs(val) * 10))
        print(f"  {val:+.2f} {bar:<10} [{kind}/{origin}/{status}] {stance}")


if __name__ == "__main__":
    import sys
    _show() if "--show" in sys.argv else print(__doc__)
