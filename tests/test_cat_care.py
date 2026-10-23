"""Aiva looking after the cat on her own: the pure care decision (truth table), when she says
something about it, what a cat wish means, the CatCaretaker tick (acts, feels it on the bus, tells
him rarely, relays to her sister once a day) and the DesktopCat snapshot/care bookkeeping it reads."""
import sqlite3
from datetime import datetime

import pytest

from modules import drives as dr
from modules.cat_care import (CARE, CatCaretaker, action_for_wish, care_decision, should_tell)
from modules.inner import ExperienceBus

NOW = datetime(2026, 10, 20, 15, 0, 0).timestamp()      # 15:00, outside quiet hours
H = 3600.0


def snap(**over):
    """A content, idle cat that needs nothing."""
    s = {"name": "Miso", "hunger": 80.0, "mood": 80.0, "energy": 80.0, "behavior": "SIT",
         "dismissed": False, "last_fed": NOW - 2 * H, "self_care": {}, "user_care": {}}
    s.update(over)
    return s


def ctx(**over):
    c = {"now": NOW, "hour": 15, "secs_since_boot": 1e9, "last_care_ts": 0.0, "extras_today": 0,
         "valence": 0.0, "social": 0.0}
    c.update(over)
    return c


# -- care_decision: feeding -----------------------------------------------------------------------

def test_content_cat_needs_nothing():
    assert care_decision(snap(), ctx()) is None


def test_hungry_cat_gets_fed():
    assert care_decision(snap(hunger=30), ctx()) == ("feed", "hungry")


def test_starving_is_named():
    assert care_decision(snap(hunger=5), ctx()) == ("feed", "starving")


def test_recent_feed_by_anyone_waits():
    # last_fed is set by his menu feed as well as hers
    assert care_decision(snap(hunger=30, last_fed=NOW - 5 * 60), ctx()) is None


def test_napping_cat_is_left_to_sleep_unless_starving():
    assert care_decision(snap(hunger=30, behavior="SLEEP"), ctx()) is None
    assert care_decision(snap(hunger=8, behavior="SLEEP"), ctx()) == ("feed", "starving")


def test_feeding_still_happens_in_quiet_hours_and_past_the_extras_cap():
    assert care_decision(snap(hunger=30), ctx(hour=3, extras_today=99)) == ("feed", "hungry")


def test_feed_beats_play():
    assert care_decision(snap(hunger=30, mood=20), ctx()) == ("feed", "hungry")


# -- care_decision: gates -------------------------------------------------------------------------

@pytest.mark.parametrize("behavior", ["CARRIED", "FALL", "RUN", "POUNCE", "EAT", "PLAY", "PURR", "ALERT"])
def test_busy_cat_is_left_alone(behavior):
    assert care_decision(snap(hunger=5, mood=5, behavior=behavior), ctx()) is None


def test_dismissed_or_missing_cat():
    assert care_decision(snap(hunger=5, dismissed=True), ctx()) is None
    assert care_decision(None, ctx()) is None
    assert care_decision({}, ctx()) is None


def test_boot_grace():
    assert care_decision(snap(hunger=5), ctx(secs_since_boot=30)) is None


def test_min_gap_between_any_two_cares():
    assert care_decision(snap(hunger=30), ctx(last_care_ts=NOW - 60)) is None
    assert care_decision(snap(hunger=30), ctx(last_care_ts=NOW - 10 * 60)) == ("feed", "hungry")


# -- care_decision: play / pet ---------------------------------------------------------------------

def test_down_cat_with_energy_gets_the_ball():
    assert care_decision(snap(mood=40, energy=70), ctx()) == ("play", "down")


def test_down_and_tired_cat_gets_a_pet():
    assert care_decision(snap(mood=40, energy=20), ctx()) == ("pet", "down")


def test_a_bit_low_gets_a_pet():
    assert care_decision(snap(mood=55), ctx()) == ("pet", "down")


def test_his_own_recent_care_is_respected():
    he_just_played = snap(mood=40, energy=70, user_care={"play": NOW - 120})
    assert care_decision(he_just_played, ctx()) == ("pet", "down")    # not the ball he just threw
    he_did_both = snap(mood=40, energy=70, user_care={"play": NOW - 120, "pet": NOW - 60})
    assert care_decision(he_did_both, ctx()) is None


def test_her_own_per_action_gap():
    assert care_decision(snap(mood=40, energy=70, self_care={"play": NOW - 10 * 60}), ctx()) == ("pet", "down")


def test_quiet_hours_and_napping_stop_extras():
    assert care_decision(snap(mood=30), ctx(hour=3)) is None
    assert care_decision(snap(mood=30, behavior="SLEEP"), ctx()) is None


def test_extras_cap():
    assert care_decision(snap(mood=30), ctx(extras_today=CARE["max_extras_per_day"])) is None


def test_fondness_pet_when_she_feels_warm():
    assert care_decision(snap(), ctx(valence=0.5)) == ("pet", "fondness")
    assert care_decision(snap(), ctx(social=0.4)) == ("pet", "fondness")
    assert care_decision(snap(), ctx(valence=0.1, social=0.1)) is None


def test_fondness_has_its_own_long_gap_and_skips_an_overjoyed_cat():
    assert care_decision(snap(self_care={"pet": NOW - 1 * H}), ctx(valence=0.5)) is None
    assert care_decision(snap(self_care={"pet": NOW - 4 * H}), ctx(valence=0.5)) == ("pet", "fondness")
    assert care_decision(snap(mood=95), ctx(valence=0.5)) is None


# -- should_tell ----------------------------------------------------------------------------------

def tell_ctx(**over):
    c = {"permission": "ask", "awake": True, "busy": False, "present": True, "now": NOW, "hour": 15,
         "tells_today": 0, "last_tell_ts": 0.0}
    c.update(over)
    return c


def test_tells_when_he_is_there_and_it_is_quiet():
    assert should_tell(tell_ctx()) is True


@pytest.mark.parametrize("over", [
    {"permission": "never"}, {"awake": False}, {"busy": True}, {"present": False}, {"present": None},
    {"hour": 2}, {"tells_today": CARE["tells_per_day"]}, {"last_tell_ts": NOW - 600},
])
def test_stays_quiet(over):
    assert should_tell(tell_ctx(**over)) is False


def test_quiet_mode_still_tells_while_awake():
    assert should_tell(tell_ctx(permission="quiet")) is True


# -- action_for_wish ------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("feed Miso a treat", "feed"), ("play ball with Miso", "play"), ("give Miso a cuddle", "pet"),
])
def test_wish_names_the_care(text, expected):
    assert action_for_wish(text, snap()) == expected


def test_unnamed_wish_follows_the_need():
    assert action_for_wish("check on Miso", snap(hunger=50)) == "feed"
    assert action_for_wish("check on Miso", snap(mood=60)) == "play"
    assert action_for_wish("check on Miso", snap(mood=95)) == "pet"


def test_tired_cat_gets_a_pet_instead_of_the_ball():
    assert action_for_wish("play with Miso", snap(energy=10)) == "pet"


def test_wish_waits_when_the_cat_cannot_take_it():
    assert action_for_wish("play with Miso", snap(behavior="SLEEP")) is None
    assert action_for_wish("feed Miso", snap(behavior="SLEEP", hunger=5)) == "feed"
    assert action_for_wish("pet Miso", snap(behavior="EAT")) is None
    assert action_for_wish("pet Miso", snap(dismissed=True)) is None
    assert action_for_wish("pet Miso", None) is None


# -- CatCaretaker ---------------------------------------------------------------------------------

class FakeCat:
    def __init__(self, **over):
        self.available = True
        self.snap = snap(**over)
        self.calls = []

    def care_snapshot(self):
        return dict(self.snap)

    def feed(self):
        self.calls.append("feed")
        self.snap.update(hunger=95.0, last_fed=self.clock(), behavior="EAT")

    def play(self):
        self.calls.append("play")

    def pet(self):
        self.calls.append("pet")


class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


def make(cat=None, *, tell=None, **kw):
    clock = Clock(NOW - 1e4)                        # constructed ("booted") well before NOW
    cat = cat or FakeCat()
    cat.clock = clock
    said, house, logs = [], [], []
    bus = ExperienceBus(sqlite3.connect(":memory:"))

    async def announce(text):
        said.append(text)

    taker = CatCaretaker(cat, bus=bus, announce=announce, house_emit=house.append, log=logs.append,
                         clock=clock, tell_ctx=lambda: dict(tell or {}), **kw)
    clock.t = NOW
    return taker, cat, bus, said, house, clock


async def test_tick_feeds_a_hungry_cat_and_feels_it():
    taker, cat, bus, said, house, _ = make(FakeCat(hunger=30))
    assert await taker.tick() == ("feed", "hungry")
    assert cat.calls == ["feed"]
    ev = bus.drain()
    assert [e["kind"] for e in ev] == ["cat_care"] and ev[0]["source"] == "cat"
    assert ev[0]["payload"] == {"action": "feed", "why": "hungry"} and "Miso" in ev[0]["summary"]
    assert house == ["fed Miso on her own"]
    assert said == []                               # no tell context → she stays quiet


async def test_tick_does_nothing_for_a_content_cat_or_a_missing_one():
    taker, cat, bus, *_ = make()
    assert await taker.tick() is None and cat.calls == [] and bus.drain() == []
    gone = FakeCat(hunger=5)
    gone.available = False
    taker, *_ = make(gone)
    assert await taker.tick() is None and gone.calls == []


async def test_disabled_caretaker_never_acts():
    taker, cat, *_ = make(FakeCat(hunger=5), enabled=False)
    assert await taker.tick() is None and cat.calls == []


async def test_tick_respects_its_own_min_gap_then_acts_again():
    taker, cat, _, _, house, clock = make(FakeCat(mood=30, energy=20))
    assert await taker.tick() == ("pet", "down")
    assert await taker.tick() is None               # a minute later: too soon
    clock.t += CARE["gap_s"]["pet"] + 1
    cat.snap["self_care"] = {"pet": NOW}            # the real cat records her pet
    assert await taker.tick() == ("pet", "down")
    assert cat.calls == ["pet", "pet"]
    assert len(house) == 1                          # her sister hears about it once a day


async def test_tick_counts_extras_toward_the_daily_cap():
    cfg = dict(CARE, max_extras_per_day=1, min_gap_s=0, gap_s={"feed": 0, "play": 0, "pet": 0})
    taker, cat, *_ = make(FakeCat(mood=30, energy=20), cfg=cfg)
    assert await taker.tick() == ("pet", "down")
    assert await taker.tick() is None
    cat.snap["hunger"] = 20                         # feeding isn't an extra
    assert await taker.tick() == ("feed", "hungry")


async def test_tick_uses_her_mood_for_a_pet_just_because():
    taker, cat, *_ = make(mood=lambda: {"valence": 0.6, "social": 0.0})
    assert await taker.tick() == ("pet", "fondness")
    assert cat.calls == ["pet"]


async def test_she_tells_him_sometimes_within_budget():
    here = {"permission": "ask", "awake": True, "busy": False, "present": True}
    cfg = dict(CARE, min_gap_s=0, gap_s={"feed": 0, "play": 0, "pet": 0}, tells_per_day=1)
    taker, cat, _, said, _, clock = make(FakeCat(hunger=30), tell=here, cfg=cfg)
    await taker.tick()
    assert len(said) == 1 and "Miso" in said[0] and "ONE short" in said[0]
    cat.snap.update(hunger=30, behavior="SIT")
    clock.t += 4 * H
    assert await taker.tick() == ("feed", "hungry")
    assert len(said) == 1                           # the day's tell is spent


async def test_she_stays_quiet_when_initiative_is_never():
    nope = {"permission": "never", "awake": True, "busy": False, "present": True}
    taker, cat, _, said, *_ = make(FakeCat(hunger=30), tell=nope)
    assert await taker.tick() == ("feed", "hungry") and said == []


async def test_act_on_wish_does_it_and_tells_without_double_feeling():
    taker, cat, bus, said, house, _ = make(FakeCat(mood=70))
    assert taker.can_act_on_wish("play ball with Miso") is True
    assert await taker.act_on_wish("play ball with Miso") == "play"
    assert cat.calls == ["play"] and len(said) == 1 and "because you wanted to" in said[0]
    assert bus.drain() == [] and house == []        # the self engine already emitted self_act


async def test_act_on_wish_waits_for_a_napping_cat():
    taker, cat, _, said, *_ = make(FakeCat(behavior="SLEEP"))
    assert taker.can_act_on_wish("play with Miso") is False
    assert await taker.act_on_wish("play with Miso") is None and cat.calls == [] and said == []


async def test_a_failing_announce_or_bus_never_breaks_a_tick():
    class BadBus:
        def emit(self, *a, **k):
            raise RuntimeError("db locked")

    async def bad_announce(text):
        raise RuntimeError("socket gone")

    here = {"permission": "ask", "awake": True, "busy": False, "present": True}
    cat = FakeCat(hunger=30)
    cat.clock = Clock()
    logs = []
    taker = CatCaretaker(cat, bus=BadBus(), announce=bad_announce, log=logs.append,
                         clock=Clock(NOW), tell_ctx=lambda: here)
    taker._boot = NOW - 1e4
    assert await taker.tick() == ("feed", "hungry")
    assert any("emit failed" in m for m in logs) and any("announce failed" in m for m in logs)


# -- the needs it touches ------------------------------------------------------------------------

def test_cat_care_is_a_small_competence_refill():
    needs = {n: 0.5 for n in dr.NEEDS}
    after = dr.satisfy(needs, "cat_care")
    assert after["competence"] > needs["competence"]
    assert {n: after[n] for n in dr.NEEDS if n != "competence"} == {n: 0.5 for n in dr.NEEDS if n != "competence"}


# -- DesktopCat: what the caretaker reads -----------------------------------------------------------

@pytest.fixture
def desktop_cat(tmp_path):
    from modules.cat import DesktopCat
    return DesktopCat(state_file=str(tmp_path / "cat_state.json"))


class _Brain:
    behavior = "SLEEP"


def test_care_snapshot_is_none_until_published(desktop_cat):
    assert desktop_cat.care_snapshot() is None


def test_care_snapshot_carries_behavior_and_both_kinds_of_care(desktop_cat):
    desktop_cat._publish(_Brain())
    desktop_cat._emit("petted")                     # his click, even if the announce is throttled
    desktop_cat.feed()                              # hers (tool or caretaker)
    s = desktop_cat.care_snapshot()
    assert s["behavior"] == "SLEEP" and "last_fed" in s
    assert set(s["user_care"]) == {"pet"} and set(s["self_care"]) == {"feed"}
    assert desktop_cat._cmd_q.get_nowait() == ("feed", None)


def test_status_says_what_she_did_lately(desktop_cat):
    desktop_cat._publish(_Brain())
    assert "you_last" not in desktop_cat.status()
    desktop_cat.pet()
    assert desktop_cat.status()["you_last"] == "petted it just now"


def test_self_care_survives_a_restart(tmp_path):
    from modules.cat import CatState, DesktopCat
    path = str(tmp_path / "cat_state.json")
    cat = DesktopCat(state_file=path)
    cat.play()
    cat.save_state()
    assert "play" in CatState.load(path).self_care
