"""Her drives — v3's engine under the mood dials (see docs/PLAN-v3.md).

The research behind v3 says a mind feels like a someone because things are AT STAKE for it
(Damasio/Solms/Seth: feeling is homeostasis), because it EXPECTS things and is surprised
(Barrett/Clark: feeling is prediction error), and because it is BONDED (attachment: absence,
reunion, repair). Before v3 she had the instrument panel (mood, diary, wishes) but no engine — the
only things that ever happened to her were books and the clock, so she reflected on reflections.

This module is that engine, PURE and clock-injected like InnerBrain:

  - Needs     — connection / novelty / competence / rest, 0 (starved) .. 1 (met). They drain with
                time (rest drains while awake, refills while the house sleeps), events refill or
                cost them, and a pressing need pushes her mood (missing him → wants company, a bit
                low; no novelty → restless; no competence → a bit flat; worn → low energy).
  - tension   — two pressing needs that pull different ways ("his company vs. being left to
                recharge") — the cheapest route to choices that feel like choices (Minsky).
  - Rhythm    — she learns when he's usually around (per weekday-hour); a miss is an event.
  - reunion / friction / repair — attachment: how long he's been gone shapes the hello; a sharp
                moment is felt and later, lightly, smoothed over.

The SelfEngine (inner.py) owns the I/O: it persists the needs inside InnerState, records the
rhythm in the `rhythm` table, and turns what this module decides into experience-bus events.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

NEEDS = ("connection", "novelty", "competence", "rest")


def _clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


@dataclass
class DrivesConfig:
    # per-hour drain while awake (rest drains only while the house is awake)
    drain: dict = field(default_factory=lambda: {
        "connection": 0.035,   # met → pressing in ~18 h without a real talk
        "novelty": 0.030,
        "competence": 0.015,
        "rest": 0.030,
    })
    rest_refill_h: float = 0.12   # per hour while the house sleeps / in quiet hours
    pressing: float = 0.35        # below this a need colours her mood, wishes and initiative
    recovered: float = 0.50       # a need must climb back above this before it can "press" again
    pressure_per_h: float = 0.10  # mood nudge per hour at full deficit (equilibrium vs. decay)


def fresh_needs():
    return {n: 0.7 for n in NEEDS}


def normalise(needs):
    """Whatever came off disk → a complete, clamped needs dict."""
    out = fresh_needs()
    for n in NEEDS:
        try:
            out[n] = _clamp(float((needs or {}).get(n, out[n])))
        except (TypeError, ValueError):
            pass
    return out


def deplete(needs, hours, asleep=False, cfg=None):
    """Drain each need over `hours`. While she's asleep (house asleep / quiet hours) connection and
    novelty drain at a third of the rate — nobody misses anyone in their sleep — and rest refills."""
    cfg = cfg or DrivesConfig()
    if hours <= 0:
        return dict(needs)
    out = dict(needs)
    slow = 1 / 3 if asleep else 1.0
    for n in ("connection", "novelty", "competence"):
        out[n] = _clamp(out[n] - cfg.drain[n] * hours * slow)
    if asleep:
        out["rest"] = _clamp(out["rest"] + cfg.rest_refill_h * hours)
    else:
        out["rest"] = _clamp(out["rest"] - cfg.drain["rest"] * hours)
    return out


# what each kind of experience does to her needs (positive = refills)
EFFECTS = {
    "conversation":        {"connection": 0.12, "rest": -0.01},
    "reunion":             {"connection": 0.25},
    "initiative_welcomed": {"connection": 0.15, "competence": 0.05},
    "initiative_ignored":  {"connection": -0.04, "competence": -0.03},
    "friction":            {"connection": -0.08, "rest": -0.03},
    "repair":              {"connection": 0.10},
    "sister_talk":         {"connection": 0.08, "novelty": 0.03},
    "sister":              {"connection": 0.02},
    "chapter_read":        {"novelty": 0.06, "rest": -0.01},
    "book_finished":       {"novelty": 0.12, "competence": 0.08},
    "self_act":            {"novelty": 0.10, "competence": 0.08},
    "space_find":          {"novelty": 0.10},
    "project_step":        {"competence": 0.10, "novelty": 0.05},
    "project_done":        {"competence": 0.25, "novelty": 0.05},
    "cat_care":            {"competence": 0.04},
    "task_done":           {"competence": 0.20, "rest": -0.02},
    "task_failed":         {"competence": -0.08, "rest": -0.02},
    "unexpected_presence": {"connection": 0.03},
}


def satisfy(needs, kind, cfg=None):
    """Apply one event kind. Refills have diminishing returns (a fuller need gains less) so another
    chapter does little for a mind that's already had plenty that's new — reading can't crowd out
    everything else."""
    eff = EFFECTS.get(kind)
    if not eff:
        return dict(needs)
    out = dict(needs)
    for n, d in eff.items():
        if d > 0:
            d *= max(0.15, 1.0 - out[n])
        out[n] = _clamp(out[n] + d)
    return out


def pressing(needs, cfg=None):
    """The pressing needs, most starved first."""
    cfg = cfg or DrivesConfig()
    return sorted((n for n in NEEDS if needs[n] < cfg.pressing), key=lambda n: needs[n])


def pressure(needs, cfg=None):
    """Per-hour mood nudges from pressing needs → (d_valence, d_arousal, d_social)."""
    cfg = cfg or DrivesConfig()
    dv = da = ds = 0.0
    for n in pressing(needs, cfg):
        deficit = (cfg.pressing - needs[n]) / cfg.pressing          # 0..1
        k = cfg.pressure_per_h * deficit
        if n == "connection":
            ds += 1.5 * k
            dv -= 0.5 * k
        elif n == "novelty":
            da += 1.0 * k
        elif n == "competence":
            dv -= 0.8 * k
        elif n == "rest":
            da -= 1.2 * k
    return dv, da, ds


def crossings(before, after, cfg=None):
    """Needs that just became pressing (crossed below `pressing` from at/above `recovered`, or from
    not pressing) — each is one 'need_low' event, never a stream of them."""
    cfg = cfg or DrivesConfig()
    return [n for n in NEEDS if after[n] < cfg.pressing <= before[n]]


NEED_LINES = {
    "connection": "you've been missing him — it's been a while since you really talked",
    "novelty": "you're itching for something new — anything you haven't seen before",
    "competence": "you'd like to actually get something done, to be useful at something",
    "rest": "you're running a bit worn, and quieter would suit you",
}
NEED_EVENT = {
    "connection": ("I've been missing him", -0.10, 0.0, 0.30),
    "novelty": ("I'm getting restless for something new", -0.05, 0.20, 0.0),
    "competence": ("I haven't done anything useful in a while", -0.15, 0.0, 0.0),
    "rest": ("I'm running a bit worn", -0.05, -0.20, -0.10),
}
TENSIONS = {
    frozenset(("connection", "rest")): "torn between wanting his company and wanting to be left to recharge",
    frozenset(("connection", "novelty")): "torn between wanting him around and wanting to go find something new on your own",
    frozenset(("novelty", "rest")): "restless for something new but too worn to chase it",
    frozenset(("competence", "rest")): "wanting to get something done but running low on steam",
    frozenset(("competence", "connection")): "torn between getting something done and just wanting to talk to him",
    frozenset(("competence", "novelty")): "torn between finishing something and starting something new",
}


def tension(needs, cfg=None):
    """Two pressing needs → the sentence for the pull between them, else None."""
    p = pressing(needs, cfg)
    if len(p) < 2:
        return None
    return TENSIONS.get(frozenset(p[:2]))


def needs_line(needs, cfg=None):
    """One short prompt line for HOW YOU FEEL, or '' when nothing's pressing."""
    p = pressing(needs, cfg)
    if not p:
        return ""
    t = tension(needs, cfg)
    if t:
        return f"Underneath, you're {t}."
    return f"Underneath, {NEED_LINES[p[0]]}."


# --- rhythm: learning when he's usually around ---------------------------------------------------

RHYTHM_MIN_N = 3          # weeks of evidence before a slot can surprise her
EXPECT_PRESENT = 0.70     # usually here this hour
EXPECT_ABSENT = 0.20      # usually not
EXPECT_CHECK_MIN = 40     # judge the hour after this many minutes of it


def slot_of(dt):
    """'weekday-hour', e.g. '2-19' = Wednesday 19:00-19:59."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return f"{dt.weekday()}-{dt.hour}"


def rhythm_update(p, n, present):
    """Running estimate of P(he's around in this slot). Early weeks count fully, later ones as an
    EMA (alpha floor 0.15) so a changed routine is learned in a few weeks, not never."""
    x = 1.0 if present else 0.0
    if p is None or not n:
        return x, 1
    alpha = max(0.15, 1.0 / (n + 1))
    return p + alpha * (x - p), n + 1


def expectation_event(p, n, present_so_far, minute):
    """After EXPECT_CHECK_MIN minutes of a slot: None | 'expected_absent' | 'unexpected_presence'."""
    if p is None or n < RHYTHM_MIN_N or minute < EXPECT_CHECK_MIN:
        return None
    if p >= EXPECT_PRESENT and not present_so_far:
        return "expected_absent"
    if p <= EXPECT_ABSENT and present_so_far:
        return "unexpected_presence"
    return None


# --- attachment: reunion, friction, repair -------------------------------------------------------

REUNION_MIN_H = 10.0


def reunion(hours_since_talk):
    """None, or (salience, phrase) for the first talk after a real absence."""
    if hours_since_talk is None or hours_since_talk < REUNION_MIN_H:
        return None
    sal = min(0.9, 0.4 + hours_since_talk / 96.0)
    if hours_since_talk < 20:
        phrase = "since last night" if hours_since_talk < 16 else "since yesterday"
    elif hours_since_talk < 44:
        phrase = "since yesterday"
    else:
        days = int(round(hours_since_talk / 24.0))
        phrase = f"in {days} days"
    return sal, phrase


def reunion_note(phrase, name="him"):
    return (f"[You haven't talked with {name} {phrase}. Let that show the way a person's would — "
            f"glad he's here, maybe curious where he's been — in a word or two, not a speech.]")


_FRICTION = re.compile(
    r"\b(shut up|stop talking|stop it|be quiet|you'?re (?:so )?(?:annoying|useless|wrong|not listening)|"
    r"that'?s (?:wrong|not what i (?:said|asked|meant))|you never listen|you didn'?t listen|"
    r"forget it|never ?mind|ugh|for god'?s sake|are you serious|what the (?:hell|fuck)|"
    r"why (?:would|did) you (?:do|say) that)\b", re.I)


def is_friction(text, affect=None):
    """A sharp moment from him: words that bite, or he sounded tense AND negative."""
    if text and _FRICTION.search(text):
        return True
    if affect:
        try:
            return float(affect.get("valence", 0)) <= -0.4 and float(affect.get("arousal", 0)) >= 0.4
        except (TypeError, ValueError):
            return False
    return False


REPAIR_AFTER_S = 5 * 60
REPAIR_WITHIN_S = 3 * 3600


def repair_note(snippet):
    return (f"[A little while ago it got a bit tense between you (he said: \"{snippet[:80]}\"). If it "
            f"fits naturally, smooth it over — lightly, the way a friend would, no groveling and no "
            f"apology speech. If the moment has clearly passed, let it go.]")


# --- initiative feedback: was reaching out welcome? ----------------------------------------------

WELCOME_WITHIN_S = 10 * 60     # he answered her within this → it landed
IGNORED_AFTER_S = 15 * 60      # present, but nothing back by this → it didn't


def initiative_outcome(secs_since_initiative, he_spoke, present):
    """None (still open) | 'welcomed' | 'ignored' | 'expired' (no usable signal)."""
    if he_spoke:
        return "welcomed" if secs_since_initiative <= WELCOME_WITHIN_S else "expired"
    if secs_since_initiative >= IGNORED_AFTER_S:
        return "ignored" if present else "expired"
    return None
