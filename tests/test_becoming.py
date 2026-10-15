import asyncio
import json
import math
import sqlite3
from datetime import datetime, timedelta

import pytest

from modules import becoming as bc
from modules.becoming import Becoming, CuriosityState, curiosity_decision, validate_candidates
from modules.persona import baseline_traits, build_system_prompt

BASELINE = baseline_traits()


def vec_of(text):
    v = [0.0] * 512
    for tok in (text or "").lower().split():
        v[hash(tok) % 512] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


async def fake_embed(text):
    return vec_of(text)


def make(embed=True, llm=None, remember=None, now=None, engine=None):
    conn = sqlite3.connect(":memory:")
    # facts table (for permission back-compat) lives in Memory normally; make a stub
    conn.execute("CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, category TEXT, fact TEXT, "
                 "created_ts TEXT, last_confirmed_ts TEXT)")
    b = Becoming(conn, embed=fake_embed if embed else None, llm=llm, remember=remember,
                 baseline_traits=BASELINE, log=lambda m: None, now=now, engine=engine)
    return b, conn


def book(bid=1, title="Meditations", author="Marcus", sections=10):
    return {"id": bid, "title": title, "author": author, "sections": sections}


# --- validation ------------------------------------------------------------------------

async def test_validation_filters():
    # a controlled embed so the contradiction gate can be exercised deterministically:
    # anything mentioning "curious" collides with the curiosity TEMPERAMENT baseline line (cosine
    # 1.0). (Tastes like music are no longer baseline — they moved to the mutable preferences layer —
    # so becoming's gate now protects her fixed CHARACTER, which is what this exercises.)
    CURIOUS = [1.0] + [0.0] * 511
    async def embed(text):
        return CURIOUS if "curious" in (text or "").lower() else vec_of(text)
    baseline_vecs = [(l, (CURIOUS if "curious" in l.lower() else vec_of(l))) for l in BASELINE]
    data = {
        "lessons": [{"lesson": "I've learned to leave a stopping rule on any plan.", "kind": "habit",
                     "domains": ["planning"], "strength": 0.8, "evidence": "the stopping rule"}],
        "traits": [
            {"text": "You are now serious.", "kind": "trait", "strength": 0.9, "evidence": "x",
             "why_compatible": "no"},                                   # rejected: "You ..."
            {"text": "I no longer feel curious about anything.", "kind": "value",
             "strength": 0.9, "evidence": "curious", "why_compatible": "deepens"},  # rejected: contradicts a baseline line
            {"text": "I've come to value quiet mornings since reading Meditations.", "kind": "value",
             "strength": 0.7, "evidence": "the morning", "why_compatible": "adds a facet"}],
        "voice": [
            {"kind": "phrase", "note": "one two three four five six seven", "example": "x",
             "strength": 0.9},                                          # rejected: >6 word phrase
            {"kind": "humour", "note": "deadpan understatement of disasters",
             "example": "well, that went great.", "strength": 0.8}],
    }
    notes_lower = "the stopping rule mattered. the morning was cold. i felt curious. cats slept."
    out = await validate_candidates(data, notes_lower, baseline_vecs, embed)
    assert [l["text"] for l in out["lessons"]][0].startswith("I've learned")
    assert len(out["traits"]) == 1 and out["traits"][0]["text"].startswith("I've come to value quiet")
    assert len(out["voice"]) == 1 and out["voice"][0]["kind"] == "humour"


async def test_validation_missing_evidence_halves_strength():
    bv = [(l, vec_of(l)) for l in BASELINE]
    data = {"lessons": [{"lesson": "I've learned that pace matters.", "kind": "principle",
                         "domains": ["craft"], "strength": 0.5, "evidence": "this exact phrase does not appear anywhere in the notes"}]}
    out = await validate_candidates(data, "totally different words here", bv, fake_embed)
    # 0.5 * 0.6 = 0.30 which is the floor; keep it (>= 0.3)
    assert out["lessons"] and abs(out["lessons"][0]["strength"] - 0.30) < 1e-6


async def test_thin_notes_skips_extraction():
    calls = []

    async def llm(messages, **kw):
        calls.append(1)
        return "{}"

    b, conn = make(llm=llm)
    res = await b._extract(book(sections=10), {"summary": "s"}, [{"reflection": "r"}])  # 1 noted < 2
    assert res is None and calls == []


async def test_traits_gated_by_read_fraction():
    async def llm(messages, **kw):
        return json.dumps({"traits": [{"text": "I've grown to love X since reading it.", "kind": "value",
                                       "strength": 0.7, "evidence": "x", "why_compatible": "adds"}],
                           "lessons": [], "voice": []})
    b, conn = make(llm=llm)
    notes = [{"reflection": "r", "key_ideas": [], "quotes": []} for _ in range(3)]
    res = await b._extract(book(sections=10), {"summary": "s"}, notes)   # 3/10 = 30% < 80%
    assert res["traits"] == []                                          # traits not allowed
    res2 = await b._extract(book(sections=3), {"summary": "s"}, notes)   # 3/3 = 100%
    assert len(res2["traits"]) == 1


# --- adoption / dedupe / cap / fade ----------------------------------------------------

def test_adopt_and_dedupe_reinforces():
    b, conn = make()
    cands = {"traits": [{"text": "I value slow mornings since reading Meditations.", "kind": "value",
                         "strength": 0.9, "_vec": vec_of("I value slow mornings Meditations")}],
             "lessons": [], "voice": []}
    out = b._adopt(book(), cands)
    assert out["new_trait"] and b._traits()[0]["strength"] == bc.TRAIT_ADOPT_CAP   # capped at 0.7
    # same trait again → reinforced, not a second row
    b._adopt(book(bid=2), {"traits": [dict(cands["traits"][0])], "lessons": [], "voice": []})
    assert len(b._traits()) == 1 and b._traits()[0]["strength"] > bc.TRAIT_ADOPT_CAP


async def test_dropped_trait_blocks_readoption():
    b, conn = make()
    t = {"text": "I love rainy walks now.", "kind": "interest", "strength": 0.7,
         "_vec": vec_of("I love rainy walks now")}
    b._adopt(book(), {"traits": [t], "lessons": [], "voice": []})
    res = await b.drop("rainy walks")
    assert res["success"] and b._traits() == []                        # dropped, none active
    b._adopt(book(bid=2), {"traits": [dict(t)], "lessons": [], "voice": []})
    assert b._traits() == []                                           # re-adoption blocked


def test_trait_cap_fades_weakest_announced():
    b, conn = make()
    for i in range(14):
        b._adopt(book(bid=i), {"traits": [{"text": f"trait number {i} from a book", "kind": "trait",
                                           "strength": 0.5 + i * 0.01,
                                           "_vec": vec_of(f"trait {i} book")}],
                               "lessons": [], "voice": []})
    assert len(b._traits()) <= bc.MAX_ACTIVE_TRAITS


def test_fade_with_fake_clock():
    clock = {"t": datetime(2026, 1, 1)}
    b, conn = make(now=lambda: clock["t"])
    b._adopt(book(), {"traits": [{"text": "a faint interest in maps", "kind": "interest", "strength": 0.4,
                                  "_vec": vec_of("faint interest maps")}], "lessons": [], "voice": []})
    clock["t"] = datetime(2026, 6, 1)                                  # ~150 days, half-life 120
    faded = b.fade()
    assert faded >= 1 and b._traits() == [] and len(b._traits("faded")) == 1


# --- drop by id / substring ------------------------------------------------------------

async def test_drop_by_id_and_no_match():
    b, conn = make()
    b._adopt(book(), {"traits": [{"text": "I hum sea shanties now.", "kind": "trait", "strength": 0.6,
                                  "_vec": vec_of("hum sea shanties")}], "lessons": [], "voice": []})
    tid = b._traits()[0]["id"]
    assert (await b.drop(tid))["success"]
    miss = await b.drop("something I never picked up")
    assert miss["success"] is False and "remember" in miss["hint"]


# --- permission ------------------------------------------------------------------------

def test_permission_modes_and_backcompat():
    remembered = []
    b, conn = make(remember=lambda f, c: remembered.append(f) or 1)
    assert b.permission() == "ask"
    b.set_permission("always")
    assert b.permission() == "always" and remembered                  # persisted as a fact too
    b.set_permission("never")
    assert b.permission() == "never"
    # back-compat: a fact expressing standing permission, no state row
    b2, conn2 = make()
    conn2.execute("INSERT INTO facts (category, fact, created_ts, last_confirmed_ts) VALUES "
                  "('instruction', 'You can read whenever you like on your own.', '', '')")
    conn2.commit()
    assert b2.permission() == "always"


# --- curiosity decision truth table ----------------------------------------------------

def cs(**kw):
    base = dict(permission="ask", busy=False, readable=True, present=True, voice_idle_s=13 * 60,
                since_boot_s=10 * 60, since_session_s=None, since_ask_s=None, since_decline_s=None,
                asks_today=0, hour=14)
    base.update(kw)
    return CuriosityState(**base)


@pytest.mark.parametrize("state,expected", [
    (cs(), "ask"),
    (cs(busy=True), None),
    (cs(readable=False), None),
    (cs(permission="never"), None),
    (cs(hour=3), None),                                   # quiet hours
    (cs(since_boot_s=60), None),                          # boot grace
    (cs(present=False), None),                            # away + ask → wait
    (cs(voice_idle_s=60), None),                          # not idle enough
    (cs(asks_today=2), None),                             # daily cap
    (cs(since_ask_s=600), None),                          # asked recently
    (cs(since_decline_s=600), None),                      # declined recently
    (cs(since_session_s=600), None),                      # just had a session
    (cs(permission="always", present=False), "read_quietly"),
    (cs(permission="always", present=True, voice_idle_s=6 * 60), "read_quietly"),
    (cs(permission="always", present=True, voice_idle_s=60), None),
])
def test_curiosity_decision(state, expected):
    assert curiosity_decision(state) == expected


# --- prompt blocks ---------------------------------------------------------------------

def test_prompt_blocks_caps_and_drop_order():
    b, conn = make()
    for i in range(8):
        b._adopt(book(bid=i), {"traits": [{"text": f"a distinct trait about topic {i} here", "kind": "trait",
                                           "strength": 0.9 - i * 0.05, "_vec": vec_of(f"trait topic {i}")}],
                               "lessons": [{"text": f"a lesson about topic {i} to apply", "kind": "principle",
                                            "domains": ["self"], "strength": 0.8, "_vec": vec_of(f"lesson {i}")}],
                               "voice": [{"kind": "phrase", "note": f"phrase {i}", "example": "eg",
                                          "strength": 0.7, "_vec": vec_of(f"phrase {i}")}]})
    blocks = b.prompt_blocks(library_lines=["Reading X: 1 of 5"])
    assert len(blocks["traits"]) <= 6 and len(blocks.get("voice", [])) <= 2 and len(blocks["lessons"]) <= 3
    # a squeezed budget drops the maxims first and keeps the voice she picked up
    mid = b.prompt_blocks(library_lines=["Reading X"], budget_chars=350)
    assert "lessons" not in mid and "voice" in mid and "traits" in mid
    # a tiny budget keeps only who she's become
    tight = b.prompt_blocks(library_lines=["Reading X"], budget_chars=200)
    assert "traits" in tight and "voice" not in tight and "lessons" not in tight


# --- lessons applied hook --------------------------------------------------------------

async def test_relevant_for_and_mark_applied():
    b, conn = make()
    b._adopt(book(), {"lessons": [{"text": "Batch similar errands to save the afternoon.", "kind": "habit",
                                   "domains": ["planning"], "strength": 0.8,
                                   "_vec": vec_of("batch similar errands afternoon")}],
                      "traits": [], "voice": []})
    hits = await b.relevant_for("batch similar errands to save the afternoon",
                                vec=vec_of("batch similar errands to save the afternoon"))
    assert hits and hits[0]["lesson"].startswith("Batch")
    b.mark_applied([h["id"] for h in hits])
    assert b._lessons()[0]["applied_count"] == 1


async def test_session_end_share_speaks_when_user_present_even_if_asleep():
    """She read via curiosity while idle/asleep, but if he's at the PC she should SPEAK the
    'what struck me' share when done, not file it silently (regression)."""
    b, conn = make()
    said, quieted = [], []

    async def announce(n): said.append(n)
    async def quiet(n): quieted.append(n)
    b.attach_voice(announce=announce, quiet_note=quiet, is_awake=lambda: False,
                   user_present=lambda within=180: True, refresh_prompt=lambda: None)
    await b.on_session_end(book(title="All Systems Red"),
                           {"struck": "The quiet parts stayed with me.", "to_section": 3},
                           notes=[{"section_idx": 0}, {"section_idx": 1}])
    assert said and "All Systems Red" in said[0]
    assert not quieted


async def test_session_end_share_stays_quiet_when_user_away():
    b, conn = make()
    said, quieted = [], []

    async def announce(n): said.append(n)
    async def quiet(n): quieted.append(n)
    b.attach_voice(announce=announce, quiet_note=quiet, is_awake=lambda: False,
                   user_present=lambda within=180: False, refresh_prompt=lambda: None)
    await b.on_session_end(book(), {"struck": "x", "to_section": 2}, notes=[{"section_idx": 0}])
    assert not said and quieted


def test_permission_ignores_facts_about_her_sister():
    """A remembered fact that AIVA may read whenever she likes is about the sister, not standing
    permission for this character (category 'sister' is filtered)."""
    b, conn = make()
    conn.execute("INSERT INTO facts (category, fact, created_ts, last_confirmed_ts) VALUES "
                 "('sister', 'Aiva may read on her own whenever it is quiet, without asking first.', '', '')")
    conn.commit()
    assert b.permission() == "ask"
    conn.execute("INSERT INTO facts (category, fact, created_ts, last_confirmed_ts) VALUES "
                 "('instruction', 'You can read whenever you like on your own.', '', '')")
    conn.commit()
    assert b.permission() == "always"                 # her own permission still works
