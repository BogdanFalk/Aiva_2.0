"""Phase-2 offline tests for speaker recognition: the pure open-set `decide`, the SpeakerMemory
store (in-memory sqlite, synthetic voiceprints), and the SpeakerRecognizer flow with an injected
fake embedder. No ONNX model, no audio."""
import sqlite3
from datetime import datetime, timedelta

import pytest

from modules.speakers import (UNKNOWN_NOTE, Decision, SpeakerConfig, SpeakerMemory,
                              SpeakerRecognizer, _cosine, _normalize, decide)

CFG = SpeakerConfig()   # defaults: t_high .62, t_low .45, margin .06, sticky_keep .50, sticky 45s
T0 = datetime(2026, 8, 28, 12, 0, 0)


def iso(hours=0.0):
    return (T0 + timedelta(hours=hours)).isoformat()


def voice(person, variant=0, dim=16):
    """A deterministic normalized voiceprint: same `person` clusters tightly (cosine ~1),
    different persons are near-orthogonal (cosine ~0)."""
    v = [0.0] * dim
    v[person % dim] = 1.0
    v[(person * 7 + 1 + variant) % dim] += 0.05   # small per-variant perturbation
    return _normalize(v)


# -- sanity on the synthetic voices ------------------------------------------------

def test_voice_helper_clusters():
    assert _cosine(voice(1, 0), voice(1, 3)) > 0.9      # same person
    assert _cosine(voice(1, 0), voice(2, 0)) < 0.4      # different people


# -- pure decision ----------------------------------------------------------------

def test_decide_confident_known():
    d = decide([(1, "Frank", 0.80), (2, "Sarah", 0.50)], None, CFG, now=iso())
    assert d.kind == "known" and d.name == "Frank"


def test_decide_empty_is_unknown():
    assert decide([], None, CFG, now=iso()).kind == "unknown"


def test_decide_below_t_low_is_unknown():
    assert decide([(1, "Frank", 0.30)], None, CFG, now=iso()).kind == "unknown"


def test_decide_within_margin_tie_is_unknown():
    # two knowns both above t_high but within margin of each other -> not confident
    assert decide([(1, "Frank", 0.80), (2, "Sarah", 0.77)], None, CFG, now=iso()).kind == "unknown"


def test_decide_holds_recent_sticky_when_unsure():
    scored = [(1, "Frank", 0.55)]                     # between t_low and t_high
    d = decide(scored, (1, "Frank", iso()), CFG, now=iso(0.001))
    assert d.kind == "hold" and d.name == "Frank"


def test_decide_stale_sticky_is_unknown():
    scored = [(1, "Frank", 0.55)]
    d = decide(scored, (1, "Frank", iso(-2)), CFG, now=iso())   # sticky heard 2h ago > sticky_secs
    assert d.kind == "unknown"


def test_decide_confident_competitor_overrides_sticky():
    scored = [(2, "Sarah", 0.90), (1, "Frank", 0.40)]
    d = decide(scored, (1, "Frank", iso()), CFG, now=iso(0.001))
    assert d.kind == "known" and d.name == "Sarah"


# -- SpeakerMemory ----------------------------------------------------------------

@pytest.fixture
def mem():
    return SpeakerMemory(sqlite3.connect(":memory:"), model_id="test-16")


def test_enroll_and_identify_nearest(mem):
    mem.enroll("Frank", voice(1))
    mem.enroll("Sarah", voice(2))
    top = mem.identify(voice(1, variant=4))
    assert top[0][1] == "Frank" and top[0][2] > 0.9
    assert mem.identify(voice(2, variant=4))[0][1] == "Sarah"


def test_recentroid_running_mean(mem):
    sid = mem.enroll("Frank", voice(1, 0))
    mem.add_sample_and_recentroid(sid, voice(1, 1))
    mem.add_sample_and_recentroid(sid, voice(1, 2))
    row = mem.conn.execute("SELECT sample_count FROM speakers WHERE id=?", (sid,)).fetchone()
    assert row[0] == 3
    assert mem.identify(voice(1, 5))[0][1] == "Frank"      # still matches Frank


def test_enroll_duplicate_name_folds(mem):
    a = mem.enroll("Frank", voice(1, 0))
    b = mem.enroll("Frank", voice(1, 1))                    # same name -> same speaker + sample
    assert a == b and mem.count() == 1


def test_rename_and_forget(mem):
    mem.enroll("Frank", voice(1))
    mem.enroll("Sarah", voice(2))
    assert mem.rename("Frank", "Franklin")
    assert "Franklin" in mem.names() and "Frank" not in mem.names()
    assert mem.forget("Sarah")
    assert "Sarah" not in mem.names() and mem.count() == 1


def test_model_and_dim_guard(mem):
    mem.enroll("Frank", voice(1, dim=16))
    # a differently-sized voiceprint can't match (dim guard)
    assert mem.identify(voice(1, dim=8)) == []
    # a different model_id sees none of the old rows
    other = SpeakerMemory(mem.conn, model_id="other-model")
    assert other.identify(voice(1, dim=16)) == []


# -- SpeakerRecognizer flow -------------------------------------------------------

def make_rec(**kw):
    mem = SpeakerMemory(sqlite3.connect(":memory:"), model_id="test-16")
    return SpeakerRecognizer(mem, SpeakerConfig(primary_name="Frank"), enabled=True,
                             log=lambda m: None, **kw)


async def test_disabled_is_noop():
    rec = make_rec()
    rec.enabled = False
    assert await rec.on_turn(voice(1)) is None


async def test_first_voice_seeds_primary_silently():
    rec = make_rec()
    note = await rec.on_turn(voice(1))
    assert note is None                       # silent — no interrogation of Frank
    assert rec.current == "Frank" and rec.mem.count() == 1


async def test_unknown_asks_then_enrolls_then_recognizes():
    rec = make_rec()
    await rec.on_turn(voice(1))               # seed Frank
    # a new person speaks -> unknown -> ask
    note = await rec.on_turn(voice(2))
    assert note == UNKNOWN_NOTE and rec.pending is not None and rec.current is None
    # she enrolls them
    res = rec.enroll_current("Sarah")
    assert res["success"] and rec.current == "Sarah" and rec.pending is None
    assert "Sarah" in rec.mem.names()
    # Frank speaks again -> known, and the speaker CHANGED -> a note
    note2 = await rec.on_turn(voice(1, variant=3))
    assert note2 == "[Speaking now: Frank.]"
    # Frank keeps talking -> unchanged -> no note
    assert await rec.on_turn(voice(1, variant=5)) is None


async def test_repeated_unknown_folds_without_reasking():
    rec = make_rec()
    await rec.on_turn(voice(1))               # seed Frank
    assert await rec.on_turn(voice(2, 0)) == UNKNOWN_NOTE
    # same unknown voice again (e.g. saying their name) -> folded, not re-asked
    assert await rec.on_turn(voice(2, 1)) is None
    assert rec.pending is not None


async def test_enroll_without_pending_fails():
    rec = make_rec()
    assert rec.enroll_current("Sarah")["success"] is False


async def test_rename_and_forget_current():
    rec = make_rec()
    await rec.on_turn(voice(1))               # Frank current
    assert rec.rename_current("Franklin")["success"] and rec.current == "Franklin"
    assert rec.forget("Franklin")["success"] and rec.current is None


async def test_embed_pcm_uses_injected_embedder():
    async def fake_embed(pcm):
        return voice(1)
    rec = make_rec(embed=fake_embed)
    assert _cosine(await rec.embed_pcm(b"\x00\x01"), voice(1)) > 0.99
    rec.enabled = False
    assert await rec.embed_pcm(b"\x00\x01") is None
