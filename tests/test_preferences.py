"""Phase-1 tests for the mutable preferences layer: the pure `evolve` inertia core, and the
Preferences store (in-memory sqlite, deterministic fake embed). No LLM."""
import sqlite3

import pytest

from modules.preferences import ADOPT_CAP, Preferences, evolve


def fake_embed_factory(dims=64):
    import zlib

    async def embed(text):
        vec = [0.0] * dims
        for tok in (text or "").lower().split():
            vec[zlib.crc32(tok.encode()) % dims] += 1.0
        n = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / n for v in vec]
    return embed


# -- pure evolve (inertia) --------------------------------------------------------

def test_same_direction_reinforces_toward_extreme():
    new, ev = evolve(0.6, 3, "reading", +1)
    assert new > 0.6 and ev == 4


def test_opposite_erodes_toward_zero_first():
    new, ev = evolve(0.6, 3, "reading", -1)
    assert 0 < new < 0.6 and ev == 2       # moved down but hasn't flipped


def _flips_after(origin):
    v, count = 0.6, 0
    for _ in range(80):
        count += 1
        v, _ = evolve(v, 1, origin, -1)
        if v < 0:
            return count
    return 999


def test_flips_only_after_sustained_counter_evidence_and_baseline_is_slower():
    reading = _flips_after("reading")
    baseline = _flips_after("baseline")
    assert 3 <= reading <= 8                # a few opposite events flip an acquired taste
    assert baseline > reading              # baseline core resists ~twice as long


def test_clamps_to_unit_range():
    assert evolve(0.98, 10, "reading", +1)[0] <= 1.0
    assert evolve(-0.98, 10, "reading", -1)[0] >= -1.0


# -- store ------------------------------------------------------------------------

@pytest.fixture
def prefs():
    return Preferences(sqlite3.connect(":memory:"), embed=fake_embed_factory(), log=lambda m: None)


async def test_insert_is_capped(prefs):
    assert await prefs.apply("bleak nihilism", "I dislike bleak nihilism", direction=-1,
                             magnitude=1.0) == "inserted"
    p = prefs.list_prefs()[0]
    assert p["valence"] == pytest.approx(-ADOPT_CAP)      # one event can't create a strong pref


async def test_reinforce_and_dedupe(prefs):
    assert await prefs.apply("found family stories", "I love found-family stories", magnitude=0.5) == "inserted"
    v1 = prefs.list_prefs()[0]["valence"]
    assert await prefs.apply("found family stories", "I really love found-family", magnitude=0.5) == "reinforced"
    active = prefs.list_prefs()
    assert len(active) == 1 and active[0]["valence"] > v1   # deduped + reinforced, not a new row


async def test_opposite_evidence_erodes_then_flips(prefs):
    await prefs.apply("dogs", "I lean cats over dogs", direction=-1, magnitude=0.4)  # start negative
    res = [await prefs.apply("dogs", "dogs are growing on me", direction=+1, magnitude=0.6)
           for _ in range(8)]
    assert "eroded" in res and "flipped" in res
    assert prefs.list_prefs()[0]["valence"] > 0            # ended up a like


async def test_veto_blocks_reformation(prefs):
    await prefs.apply("rearranging icons", "I enjoy tidying the desktop", magnitude=0.5)
    assert await prefs.drop("rearranging icons") is not None
    assert await prefs.apply("rearranging icons", "I enjoy tidying the desktop", magnitude=0.5) == "blocked"


async def test_bounded_content_rejected(prefs):
    assert await prefs.apply("a group", "I hate all those people", kind="opinion",
                             direction=-1, magnitude=0.8) == "rejected"
    assert prefs.list_prefs() == []


async def test_seed_from_baseline_idempotent(prefs):
    assert prefs.seed_from_baseline() == 8
    assert prefs.seed_from_baseline() == 0                 # no double-seed
    block = prefs.prompt_block()
    assert any("cookies" in s for s in block["likes"])
    assert any("cats over dogs" in s for s in block["dislikes"])   # dogs seeded negative


async def test_prompt_block_groups_by_sign_and_kind(prefs):
    await prefs.apply("quiet stories", "I love quiet character-driven stories", magnitude=0.6)
    await prefs.apply("cliffhangers", "I can't stand cheap cliffhangers", direction=-1, magnitude=0.6)
    await prefs.apply("endings", "I think ambiguous endings are underrated", kind="opinion", magnitude=0.6)
    b = prefs.prompt_block()
    assert any("quiet" in s for s in b["likes"])
    assert any("cliffhangers" in s for s in b["dislikes"])
    assert any("ambiguous endings" in s for s in b["opinions"])
