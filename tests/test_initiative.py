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


# -- pure decision: the 'act' branch -----------------------------------------------

def test_actionable_desire_prefers_act_over_ask():
    ctx = base_ctx(has_actionable_desire=True, has_askable_desire=True)
    assert initiative_decision(ctx) == "act"


def test_act_needs_more_idle_than_express():
    # idle between express and act thresholds → express, not act
    idle = (INITIATIVE["idle_express_s"] + INITIATIVE["idle_act_s"]) // 2
    ctx = base_ctx(has_actionable_desire=True, voice_idle_s=idle, valence=0.6)
    assert initiative_decision(ctx) == "express"


def test_act_daily_cap_falls_back_to_ask():
    ctx = base_ctx(has_actionable_desire=True, has_askable_desire=True,
                   acts_today=INITIATIVE["max_acts_per_day"])
    assert initiative_decision(ctx) == "ask"           # out of acts → she just asks instead


def test_act_respects_quiet_hours_and_never():
    assert initiative_decision(base_ctx(has_actionable_desire=True, hour=3)) is None
    assert initiative_decision(base_ctx(has_actionable_desire=True, permission="never")) is None


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


async def test_a_raised_wish_rests_a_day_then_three_before_she_raises_it_again():
    # 2026-10-17: the same wish at 11:49, 12:46, 13:01 and 13:17
    clock = Clock()
    eng = make_engine(clock)
    said = []
    eng.attach_voice(announce=lambda note: said.append(note),
                     idle_for=lambda: 20 * 60, user_present=lambda within=180: True)
    await eng.add_desire("run one focused check on Saga's constellation", "do", strength=0.9)
    assert await eng.initiative_tick() == "ask" and "Saga" in said[-1]
    clock.advance(1)                                  # well past the gap, but it's resting
    assert await eng.initiative_tick() != "ask"
    await eng.add_desire("ask him how the Fianu work is going", "ask", strength=0.5)
    clock.advance(1)                                  # a weaker wish isn't resting, so it gets its turn
    assert await eng.initiative_tick() == "ask" and "Fianu" in said[-1]
    clock.advance(23)                                 # a day on: the first one may come up once more
    assert await eng.initiative_tick() == "ask" and "Saga" in said[-1]
    for _ in range(2):                                # then it rests three days
        clock.advance(24)
        what = await eng.initiative_tick()
        assert not (what == "ask" and "Saga" in said[-1])
    clock.advance(25)
    assert await eng.initiative_tick() == "ask" and "Saga" in said[-1]


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


async def test_tick_acts_on_a_safe_wish_then_consumes_and_feels_it():
    clock = Clock()
    eng = make_engine(clock)
    said, acted = [], []
    eng.attach_voice(announce=lambda note: said.append(note),
                     idle_for=lambda: 20 * 60, user_present=lambda within=180: True)
    # everything is actionable; the dispatcher just records the wish it was handed
    async def fake_act(desire):
        acted.append(desire["text"])
    eng.attach(actionable=lambda d: True, act=fake_act)
    await eng.add_desire("find a cozy desktop wallpaper", "make")

    what = await eng.initiative_tick()
    assert what == "act" and acted == ["find a cozy desktop wallpaper"]
    # the wish is consumed (status 'acted', no longer open) so she won't re-pick it
    assert eng.list_desires(status="open") == []
    assert any(d["text"] == "find a cozy desktop wallpaper" for d in eng.list_desires(status="acted"))
    # and she FELT it — a self_act experience event was emitted
    ev = eng.bus.drain()
    assert any(e["kind"] == "self_act" for e in ev)


async def test_tick_without_act_hook_does_not_act():
    eng = make_engine(Clock())
    eng.attach_voice(announce=lambda note: None,
                     idle_for=lambda: 20 * 60, user_present=lambda within=180: True)
    eng.attach(actionable=lambda d: True)   # actionable says yes, but no _act dispatcher wired
    await eng.add_desire("find a wallpaper", "make")
    # with no _act, _actionable_desire() returns None → falls through to 'ask', never 'act'
    assert await eng.initiative_tick() == "ask"


# -- 'chat': talking to her sister when he's away ------------------------------------

def test_chat_when_away_with_sister_present():
    ctx = base_ctx(present=False, sister_present=True, voice_idle_s=11 * 60)
    assert initiative_decision(ctx) == "chat"


def test_no_chat_without_a_sister_and_away_stays_silent():
    assert initiative_decision(base_ctx(present=False)) is None
    assert initiative_decision(base_ctx(present=False, sister_present=False)) is None


def test_chat_daily_cap():
    ctx = base_ctx(present=False, sister_present=True, voice_idle_s=11 * 60,
                   chats_today=INITIATIVE["max_chats_per_day"])
    assert initiative_decision(ctx) is None


def test_chat_needs_something_on_her_mind():
    ctx = base_ctx(present=False, sister_present=True, voice_idle_s=11 * 60, valence=0.0,
                   social=0.0, focus=False, has_askable_desire=False)
    assert initiative_decision(ctx) is None
    ctx["has_askable_desire"] = True
    assert initiative_decision(ctx) == "chat"


def test_chat_when_present_needs_double_idle():
    ctx = base_ctx(present=True, sister_present=True, voice_idle_s=11 * 60)
    assert initiative_decision(ctx) == "express"          # he's here: talk to him first
    ctx["voice_idle_s"] = 2 * INITIATIVE["idle_chat_s"] + 1
    assert initiative_decision(ctx) == "chat"


@pytest.mark.asyncio
async def test_initiative_tick_chat_asks_the_house_for_the_floor():
    con = sqlite3.connect(":memory:")
    clock = Clock()
    eng = SelfEngine(con, ExperienceBus(con), InnerBrain(CFG), now=clock, log=lambda m: None)
    eng._boot_mono -= 10 * 60
    eng.state.valence = 0.7
    asked = []
    eng.attach_voice(announce=lambda note: None, idle_for=lambda: 15 * 60,
                     user_present=lambda within=180: False,
                     want_floor=lambda kind, to: asked.append((kind, to)),
                     sister_present=lambda: "saga")
    assert await eng.initiative_tick() == "chat"
    assert asked == [("initiative", "saga")]
    assert eng._day_count("chat_day") == 1
