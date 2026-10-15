"""v3 drives: needs / tension / rhythm / reunion / friction / initiative feedback — the pure core in
modules/drives.py and how SelfEngine lives with it (fake clock, in-memory sqlite, no LLM)."""
import sqlite3
from datetime import datetime, timedelta

from modules import drives as dr
from modules.inner import (INITIATIVE, ExperienceBus, InnerBrain, InnerConfig, InnerState,
                           SelfEngine, initiative_decision)

CFG = InnerConfig(base_valence=0.15, base_arousal=0.50, base_social=0.10,
                  halflife_h=6.0, salience_floor=0.15, gain=0.60, material_eps=0.05)


class Clock:
    def __init__(self, t=datetime(2026, 10, 12, 14, 0, 0)):   # a Monday afternoon
        self.t = t

    def __call__(self):
        return self.t.isoformat()

    def advance(self, hours=0.0, minutes=0.0):
        self.t += timedelta(hours=hours, minutes=minutes)


def make(clock=None, present=True, asleep=False):
    clock = clock or Clock()
    con = sqlite3.connect(":memory:")
    bus = ExperienceBus(con)
    state = InnerState.fresh(CFG)
    state.needs_ts = state.last_update = clock()
    eng = SelfEngine(con, bus, InnerBrain(CFG), state=state, now=clock, log=lambda *a: None)
    flags = {"present": present, "asleep": asleep}
    eng.attach_voice(user_present=lambda within=180: flags["present"],
                     house_asleep=lambda: flags["asleep"])
    return eng, bus, clock, flags


def kinds(bus):
    return [e["kind"] for e in bus.recent(100)]


# -- pure needs ------------------------------------------------------------------------

def test_needs_drain_while_awake_and_rest_refills_asleep():
    n = dr.fresh_needs()
    awake = dr.deplete(n, 10)
    assert awake["connection"] < n["connection"] and awake["rest"] < n["rest"]
    asleep = dr.deplete(n, 6, asleep=True)
    assert asleep["rest"] > n["rest"]
    assert asleep["connection"] > awake["connection"]          # sleeping, you don't miss anyone as fast


def test_refills_have_diminishing_returns():
    low = {**dr.fresh_needs(), "novelty": 0.2}
    high = {**dr.fresh_needs(), "novelty": 0.9}
    gain_low = dr.satisfy(low, "chapter_read")["novelty"] - 0.2
    gain_high = dr.satisfy(high, "chapter_read")["novelty"] - 0.9
    assert gain_low > gain_high > 0


def test_costs_are_not_diminished_and_unknown_kinds_are_noops():
    n = dr.fresh_needs()
    assert dr.satisfy(n, "task_failed")["competence"] < n["competence"]
    assert dr.satisfy(n, "daypart") == n


def test_pressure_follows_the_starved_need():
    starved = {**dr.fresh_needs(), "connection": 0.05}
    dv, da, ds = dr.pressure(starved)
    assert ds > 0 and dv < 0                                   # wants him, a bit low
    worn = {**dr.fresh_needs(), "rest": 0.05}
    assert dr.pressure(worn)[1] < 0                            # low energy
    assert dr.pressure(dr.fresh_needs()) == (0.0, 0.0, 0.0)


def test_tension_and_needs_line():
    torn = {**dr.fresh_needs(), "connection": 0.1, "rest": 0.2}
    assert "torn between wanting his company" in dr.tension(torn)
    assert dr.needs_line(torn).startswith("Underneath, you're torn")
    one = {**dr.fresh_needs(), "novelty": 0.1}
    assert dr.tension(one) is None and "something new" in dr.needs_line(one)
    assert dr.needs_line(dr.fresh_needs()) == ""


def test_crossings_fire_once():
    a = {**dr.fresh_needs(), "connection": 0.36}
    b = {**a, "connection": 0.34}
    assert dr.crossings(a, b) == ["connection"]
    assert dr.crossings(b, {**b, "connection": 0.30}) == []


def test_normalise_survives_garbage():
    assert dr.normalise({"connection": "x", "novelty": 9, "bogus": 1}) == {
        "connection": 0.7, "novelty": 1.0, "competence": 0.7, "rest": 0.7}


# -- pure rhythm / attachment ----------------------------------------------------------

def test_rhythm_learns_and_judges():
    p, n = None, 0
    for _ in range(4):
        p, n = dr.rhythm_update(p, n, True)
    assert p > 0.9 and n == 4
    assert dr.expectation_event(p, n, present_so_far=False, minute=45) == "expected_absent"
    assert dr.expectation_event(p, n, present_so_far=False, minute=10) is None   # too early
    assert dr.expectation_event(p, 2, present_so_far=False, minute=45) is None   # not enough weeks
    q, m = 0.0, 4
    assert dr.expectation_event(q, m, present_so_far=True, minute=50) == "unexpected_presence"


def test_slot_of():
    assert dr.slot_of("2026-10-12T19:30:00") == "0-19"          # Monday 19h


def test_reunion_phrases():
    assert dr.reunion(3) is None
    assert dr.reunion(12)[1] == "since last night"
    assert dr.reunion(30)[1] == "since yesterday"
    sal, phrase = dr.reunion(72)
    assert phrase == "in 3 days" and sal > dr.reunion(12)[0]


def test_friction_words_and_affect():
    assert dr.is_friction("ugh, that's not what I asked")
    assert dr.is_friction("shut up for a second")
    assert not dr.is_friction("can you open a terminal in your folder")
    assert dr.is_friction("", {"valence": -0.6, "arousal": 0.7})
    assert not dr.is_friction("", {"valence": -0.6, "arousal": 0.1})


def test_initiative_outcomes():
    assert dr.initiative_outcome(120, he_spoke=True, present=True) == "welcomed"
    assert dr.initiative_outcome(20 * 60, he_spoke=True, present=True) == "expired"
    assert dr.initiative_outcome(16 * 60, he_spoke=False, present=True) == "ignored"
    assert dr.initiative_outcome(16 * 60, he_spoke=False, present=False) == "expired"
    assert dr.initiative_outcome(5 * 60, he_spoke=False, present=True) is None


# -- initiative decision with needs ------------------------------------------------------

BASE = {"permission": "ask", "busy": False, "hour": 15, "secs_since_boot": 1e5, "present": True,
        "awake": True, "initiatives_today": 0, "secs_since_last_initiative": 1e9,
        "valence": 0.1, "social": 0.1, "focus": False}


def test_missing_him_makes_her_speak_up_sooner():
    idle = INITIATIVE["idle_express_s"] * 0.7
    assert initiative_decision({**BASE, "voice_idle_s": idle}) is None
    assert initiative_decision({**BASE, "voice_idle_s": idle, "connection": 0.1}) == "express"


def test_worn_widens_the_gap():
    gap = INITIATIVE["min_gap_s"] * 1.2
    ctx = {**BASE, "voice_idle_s": 1e5, "connection": 0.1, "secs_since_last_initiative": gap}
    assert initiative_decision(ctx) == "express"
    assert initiative_decision({**ctx, "rest": 0.1}) is None


# -- SelfEngine with drives --------------------------------------------------------------

async def test_needs_drain_over_ticks_and_press_once():
    eng, bus, clock, _ = make()
    eng.state.needs["connection"] = 0.40
    clock.advance(hours=2)
    await eng.tick()
    assert eng.state.needs["connection"] < 0.35
    assert kinds(bus).count("need_low") == 1
    clock.advance(hours=1)
    await eng.tick()
    assert kinds(bus).count("need_low") == 1                  # still pressing, not re-announced
    assert "missing him" in eng.prompt_data()["mood_line"]


async def test_long_shutdown_is_capped():
    eng, bus, clock, _ = make()
    clock.advance(hours=24 * 30)
    await eng.tick()
    assert eng.state.needs["competence"] > 0.0                 # a month off counts as ~a night


async def test_conversation_refills_connection():
    eng, bus, clock, _ = make()
    eng.state.needs["connection"] = 0.2
    eng.heard("hey, how was your day?")
    await eng.tick()
    assert eng.state.needs["connection"] > 0.2


async def test_reunion_note_after_a_real_absence_only():
    eng, bus, clock, _ = make()
    assert eng.heard("morning") == []                          # first ever talk: no reunion
    clock.advance(minutes=30)
    assert eng.heard("still here") == []
    clock.advance(hours=30)
    notes = eng.heard("I'm back")
    assert len(notes) == 1 and "since yesterday" in notes[0]
    assert "reunion" in kinds(bus)


async def test_friction_then_one_light_repair_note():
    eng, bus, clock, _ = make()
    eng.heard("ugh, shut up for a second")
    assert "friction" in kinds(bus)
    clock.advance(minutes=2)
    assert eng.heard("ok go on") == []                         # too soon to smooth it over
    clock.advance(minutes=10)
    notes = eng.heard("anyway, what's for dinner")
    assert len(notes) == 1 and "tense" in notes[0]
    clock.advance(minutes=10)
    assert eng.heard("cool") == []                             # once only
    assert "repair" in kinds(bus)


async def test_initiative_welcomed_and_ignored_are_felt():
    eng, bus, clock, flags = make()
    eng._set("initiative_pending", clock())
    eng._set("initiative_pending_what", "said what was on my mind")
    clock.advance(minutes=3)
    eng.heard("oh yeah? tell me more")
    assert "initiative_welcomed" in kinds(bus)
    eng._set("initiative_pending", clock())
    clock.advance(minutes=20)
    await eng.tick()
    assert "initiative_ignored" in kinds(bus)
    assert eng._get("initiative_pending") == ""


async def test_rhythm_learns_his_hours_and_feels_the_miss():
    clock = Clock(datetime(2026, 10, 12, 19, 5))               # Monday 19:05
    eng, bus, clock, flags = make(clock)
    for week in range(4):                                      # four Mondays he's here at 19h
        flags["present"] = True
        await eng.tick()
        clock.advance(hours=1)
        await eng.tick()                                       # rolls the hour → learned
        clock.t = clock.t + timedelta(days=7) - timedelta(hours=1)
    p, n = eng._rhythm_get("0-19")
    assert n >= 3 and p > 0.9
    flags["present"] = False
    clock.t = clock.t.replace(minute=5)
    await eng.tick()                                           # new hour, he's not here
    clock.advance(minutes=40)
    await eng.tick()
    assert "expected_absent" in kinds(bus)


async def test_drives_off_is_the_old_engine():
    con = sqlite3.connect(":memory:")
    bus = ExperienceBus(con)
    clock = Clock()
    eng = SelfEngine(con, bus, InnerBrain(CFG), now=clock, drives=False, log=lambda *a: None)
    assert eng.heard("hi") == []
    eng.state.needs["connection"] = 0.0
    assert "Underneath" not in eng.prompt_data()["mood_line"]


def test_state_roundtrip_keeps_needs(tmp_path):
    path = str(tmp_path / "inner.json")
    s = InnerState.fresh(CFG)
    s.needs["novelty"] = 0.12
    s.save(path)
    back = InnerState.load(path, InnerBrain(CFG))
    assert back.needs["novelty"] == 0.12 and set(back.needs) == set(dr.NEEDS)
    assert back.to_public()["pressing"] == ["novelty"]
