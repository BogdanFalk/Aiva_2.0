"""Phase 1 substrate tests: the pure mood logic (decay + damped integration) with an
injected clock, plus the ExperienceBus on in-memory sqlite. No LLM, no real time."""
import sqlite3
from datetime import datetime, timedelta

import pytest

from modules.inner import (ExperienceBus, InnerBrain, InnerConfig, InnerState,
                           mood_word)

# a fixed, generous config so tests never depend on .env or wall-clock
CFG = InnerConfig(base_valence=0.15, base_arousal=0.50, base_social=0.10,
                  halflife_h=6.0, salience_floor=0.15, gain=0.60, material_eps=0.05)
T0 = datetime(2026, 8, 27, 12, 0, 0)


def iso(hours=0.0):
    return (T0 + timedelta(hours=hours)).isoformat()


@pytest.fixture
def brain():
    return InnerBrain(CFG)


# -- decay -------------------------------------------------------------------------

def test_decay_one_halflife_closes_half_the_gap(brain):
    s = InnerState(valence=1.0, arousal=1.0, social=1.0, last_update=iso(0))
    brain.decay(s, iso(6))  # exactly one half-life
    assert s.valence == pytest.approx(0.15 + (1.0 - 0.15) * 0.5, abs=1e-6)
    assert s.arousal == pytest.approx(0.50 + (1.0 - 0.50) * 0.5, abs=1e-6)
    assert s.social == pytest.approx(0.10 + (1.0 - 0.10) * 0.5, abs=1e-6)
    assert s.last_update == iso(6)


def test_decay_relaxes_toward_baseline_over_time(brain):
    s = InnerState(valence=-0.9, arousal=0.0, social=-0.8, last_update=iso(0))
    brain.decay(s, iso(60))  # ten half-lives → essentially baseline
    assert s.valence == pytest.approx(0.15, abs=0.01)
    assert s.arousal == pytest.approx(0.50, abs=0.01)
    assert s.social == pytest.approx(0.10, abs=0.01)


def test_decay_zero_elapsed_only_stamps(brain):
    s = InnerState(valence=0.7, last_update=iso(0))
    brain.decay(s, iso(0))
    assert s.valence == 0.7 and s.last_update == iso(0)


def test_decay_never_touches_focus(brain):
    s = InnerState(focus="the ending of All Systems Red", last_update=iso(0))
    brain.decay(s, iso(24))
    assert s.focus == "the ending of All Systems Red"


# -- integrate: damping, clamping, materiality -------------------------------------

def test_integrate_applies_nudges(brain):
    s = InnerState.fresh(CFG)
    s.last_update = iso(0)
    _, changed = brain.integrate(s, [{"valence_hint": 0.5, "salience": 1.0}], iso(0))
    assert changed is True
    assert s.valence == pytest.approx(0.15 + 0.5 * CFG.gain * 1.0, abs=1e-6)


def test_integrate_clamps_to_range(brain):
    s = InnerState(valence=0.9, arousal=0.05, social=0.0, last_update=iso(0))
    brain.integrate(s, [{"valence_hint": 1.0, "arousal_hint": -1.0, "salience": 1.0}], iso(0))
    assert s.valence == 1.0          # clamped at +1
    assert s.arousal == 0.0          # clamped at 0
    assert -1.0 <= s.social <= 1.0


def test_salience_below_floor_is_ignored(brain):
    s = InnerState.fresh(CFG)
    s.last_update = iso(0)
    v0 = s.valence
    _, changed = brain.integrate(s, [{"valence_hint": 1.0, "salience": 0.10}], iso(0))
    assert changed is False
    assert s.valence == v0           # damped out entirely


def test_small_above_floor_nudge_applies_but_is_immaterial(brain):
    s = InnerState.fresh(CFG)
    s.last_update = iso(0)
    v0 = s.valence
    # salience 0.2 (> floor), tiny hint → delta 0.1*0.6*0.2 = 0.012 < material_eps 0.05
    _, changed = brain.integrate(s, [{"valence_hint": 0.1, "salience": 0.2}], iso(0))
    assert changed is False
    assert s.valence == pytest.approx(v0 + 0.012, abs=1e-6)  # it DID move, just not materially


def test_integrate_decays_first(brain):
    s = InnerState(valence=1.0, last_update=iso(0))
    # one half-life passes AND an event arrives in the same tick
    brain.integrate(s, [{"valence_hint": 0.0, "salience": 1.0}], iso(6))
    assert s.valence == pytest.approx(0.575, abs=1e-6)  # decay happened before nudges


def test_multiple_events_accumulate(brain):
    s = InnerState.fresh(CFG)
    s.last_update = iso(0)
    brain.integrate(s, [{"social_hint": 0.4, "salience": 1.0},
                        {"social_hint": 0.4, "salience": 1.0}], iso(0))
    assert s.social == pytest.approx(0.10 + 2 * 0.4 * CFG.gain, abs=1e-6)


def test_baseline_matches_config(brain):
    assert brain.baseline() == {"valence": 0.15, "arousal": 0.50, "social": 0.10}


# -- persistence -------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path, brain):
    p = str(tmp_path / "inner_state.json")
    s = InnerState(valence=-0.3, arousal=0.7, social=0.4, focus="x", last_update=iso(0))
    s.save(p)
    back = InnerState.load(p, brain, now=iso(0))  # zero elapsed → no decay
    assert (back.valence, back.arousal, back.social, back.focus) == (-0.3, 0.7, 0.4, "x")


def test_load_missing_file_is_fresh_baseline(tmp_path, brain):
    s = InnerState.load(str(tmp_path / "nope.json"), brain, now=iso(0))
    assert (s.valence, s.arousal, s.social) == (0.15, 0.50, 0.10)


def test_decay_is_applied_on_load(tmp_path, brain):
    p = str(tmp_path / "inner_state.json")
    InnerState(valence=1.0, last_update=iso(0)).save(p)
    back = InnerState.load(p, brain, now=iso(6))  # one half-life later
    assert back.valence == pytest.approx(0.575, abs=1e-6)


# -- ExperienceBus -----------------------------------------------------------------

@pytest.fixture
def bus():
    return ExperienceBus(sqlite3.connect(":memory:"))


def test_emit_drain_consume(bus):
    a = bus.emit("chapter_read", "reading", "a sad chapter", valence=-0.4, salience=0.8)
    b = bus.emit("task_failed", "tasks", "build broke", valence=-0.3, salience=0.7)
    c = bus.emit("cat_pet", "cat", "petted Miso", valence=0.2, social=0.3, salience=0.5)
    drained = bus.drain()
    assert [e["id"] for e in drained] == [a, b, c]              # oldest-first
    assert drained[0]["kind"] == "chapter_read" and drained[0]["payload"] == {}
    bus.mark_consumed([a, b])
    left = bus.drain()
    assert [e["id"] for e in left] == [c]                       # only the unconsumed one


def test_payload_roundtrips(bus):
    bus.emit("book_finished", "reading", "done", payload={"title": "All Systems Red", "n": 9})
    e = bus.drain()[0]
    assert e["payload"] == {"title": "All Systems Red", "n": 9}


def test_recent_is_newest_first(bus):
    for i in range(3):
        bus.emit("turn_ended", "conversation", f"turn {i}")
    recent = bus.recent(2)
    assert [r["summary"] for r in recent] == ["turn 2", "turn 1"]


def test_bad_subscriber_never_breaks_emit(bus):
    seen = []
    bus.subscribe(lambda e: seen.append(e["kind"]))
    bus.subscribe(lambda e: (_ for _ in ()).throw(RuntimeError("boom")))  # raises
    bus.emit("morning", "time", "a new day")
    assert seen == ["morning"]                                 # good subscriber still ran


# -- mood_word sanity --------------------------------------------------------------

def test_mood_word_reads_the_dials():
    assert "low" in mood_word(InnerState(valence=-0.6, arousal=0.2))
    assert "restless" in mood_word(InnerState(valence=-0.6, arousal=0.8))
    assert "bright" in mood_word(InnerState(valence=0.6, arousal=0.7))
    assert "wanting company" in mood_word(InnerState(valence=0.0, arousal=0.5, social=0.6))
    assert "wanting quiet" in mood_word(InnerState(valence=0.0, arousal=0.5, social=-0.6))
