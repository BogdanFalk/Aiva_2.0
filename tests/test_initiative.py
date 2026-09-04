"""Phase 4 tests: the pure initiative decision (truth table) + initiative_tick behaviour
(records state before announcing, respects the gap, honours the permission switch)."""
import sqlite3
from datetime import datetime, timedelta

import pytest

from modules.inner import (INITIATIVE, ExperienceBus, InnerBrain, InnerConfig, InnerState,
                           SelfEngine, initiative_decision)

CFG = InnerConfig()
T0 = datetime(2026, 8, 27, 15, 0, 0)   # 15:00, outside quiet hours


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t.isoformat()

    def advance(self, hours):
        self.t += timedelta(hours=hours)


def base_ctx(**over):
    """A context that, by default, WOULD fire an 'express'."""
    ctx = {"permission": "ask", "busy": False, "awake": False, "present": True,
           "voice_idle_s": 20 * 60, "hour": 15, "valence": 0.6, "arousal": 0.5, "social": 0.2,
           "focus": False, "has_askable_desire": False,
           "secs_since_last_initiative": 1e9, "initiatives_today": 0, "secs_since_boot": 1e9}
    ctx.update(over)
    return ctx


# -- pure decision truth table -----------------------------------------------------

def test_default_context_expresses():
    assert initiative_decision(base_ctx()) == "express"


def test_askable_desire_prefers_ask():
    assert initiative_decision(base_ctx(has_askable_desire=True)) == "ask"


def test_permission_never_silences():
    assert initiative_decision(base_ctx(permission="never")) is None


def test_busy_silences():
    assert initiative_decision(base_ctx(busy=True)) is None


def test_quiet_hours_silence():
    assert initiative_decision(base_ctx(hour=3)) is None


def test_boot_grace_silences():
    assert initiative_decision(base_ctx(secs_since_boot=60)) is None


def test_daily_cap_silences():
    assert initiative_decision(base_ctx(initiatives_today=INITIATIVE["max_per_day"])) is None


def test_gap_silences():
    assert initiative_decision(base_ctx(secs_since_last_initiative=60)) is None


def test_away_silences():
    assert initiative_decision(base_ctx(present=False)) is None


def test_unknown_presence_is_allowed():
    assert initiative_decision(base_ctx(present=None)) == "express"


def test_quiet_mode_needs_awake():
    assert initiative_decision(base_ctx(permission="quiet", awake=False)) is None
    assert initiative_decision(base_ctx(permission="quiet", awake=True)) == "express"


def test_flat_mood_without_focus_stays_quiet():
    assert initiative_decision(base_ctx(valence=0.1, social=0.0, focus=False)) is None


def test_focus_alone_is_worth_expressing():
    assert initiative_decision(base_ctx(valence=0.1, social=0.0, focus=True)) == "express"


def test_wanting_company_is_worth_expressing():
    assert initiative_decision(base_ctx(valence=0.0, social=0.6, focus=False)) == "express"


def test_express_needs_enough_idle():
    assert initiative_decision(base_ctx(voice_idle_s=60)) is None


def test_ask_needs_more_idle_than_express():
    # idle between the two thresholds: an askable desire falls back to express, not ask
    idle = (INITIATIVE["idle_express_s"] + INITIATIVE["idle_ask_s"]) // 2
    assert initiative_decision(base_ctx(has_askable_desire=True, voice_idle_s=idle)) == "express"


def test_always_relaxes_cap():
    # at the normal cap, 'always' still fires (cap +2)
    assert initiative_decision(base_ctx(permission="always",
                                        initiatives_today=INITIATIVE["max_per_day"])) == "express"


# -- initiative_tick ---------------------------------------------------------------

def make_engine(clock):
    con = sqlite3.connect(":memory:")
    bus = ExperienceBus(con)
    eng = SelfEngine(con, bus, InnerBrain(CFG), now=clock)
    eng._boot_mono -= 10_000    # past the boot-grace window (real monotonic clock)
    return eng


async def test_tick_noop_without_voice_bridge():
    eng = make_engine(Clock())
    eng.state.valence = 0.7
    assert await eng.initiative_tick() is None      # no announce attached


async def test_tick_expresses_and_records_then_respects_gap():
    clock = Clock()
    eng = make_engine(clock)
    said = []
    eng.attach_voice(announce=lambda note: said.append(note),
                     idle_for=lambda: 20 * 60, user_present=lambda within=180: True,
                     is_awake=lambda: True)
    eng.state.valence = 0.7                          # worth voicing
    what = await eng.initiative_tick()
    assert what == "express" and len(said) == 1 and "one short" in said[0].lower()
    # immediately again → the gap silences her
    assert await eng.initiative_tick() is None and len(said) == 1


async def test_tick_asks_about_a_desire():
    clock = Clock()
    eng = make_engine(clock)
    said = []
    eng.attach_voice(announce=lambda note: said.append(note),
                     idle_for=lambda: 20 * 60, user_present=lambda within=180: True)
    await eng.add_desire("ask him how the Fianu work is going", "ask")
    what = await eng.initiative_tick()
    assert what == "ask" and "Fianu" in said[0]


async def test_set_initiative_never_silences_tick():
    eng = make_engine(Clock())
    said = []
    eng.attach_voice(announce=lambda note: said.append(note),
                     idle_for=lambda: 20 * 60, user_present=lambda within=180: True)
    eng.state.social = 0.7
    eng.set_initiative("never")
    assert await eng.initiative_tick() is None and said == []


async def test_announce_may_be_async():
    eng = make_engine(Clock())
    said = []

    async def announce(note):
        said.append(note)
    eng.attach_voice(announce=announce, idle_for=lambda: 20 * 60,
                     user_present=lambda within=180: True)
    eng.state.valence = 0.7
    assert await eng.initiative_tick() == "express" and len(said) == 1
