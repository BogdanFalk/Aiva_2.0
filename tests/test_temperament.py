"""Self-formed temperament (plan B4): a sister with no authored PERSONALITY grows her core from the
journal digest. Offline: in-memory sqlite, deterministic fake embeds, a FakeJournalLLM like
tests/test_self_engine.py. An AUTHORED character (Aiva) forms temperament too, but only on top of
her written floor: a negation of that floor is dropped rather than eroding anything."""
import json
import math
import sqlite3
import zlib
from datetime import datetime, timedelta

import pytest

from modules import becoming as bc
from modules.becoming import Becoming, validate_temperament
from modules.inner import (JOURNAL_PROMPT, SISTER_CLAUSE, TEMPERAMENT_CLAUSE,
                           TEMPERAMENT_CLAUSE_ON_TOP, ExperienceBus, InnerBrain, InnerConfig,
                           SelfEngine)
from modules.kin import Kin
from modules.persona import baseline_traits

CFG = InnerConfig(base_valence=0.0, base_arousal=0.5, base_social=0.0,
                  halflife_h=6.0, salience_floor=0.15, gain=0.60, material_eps=0.05)


def vec_of(text, dims=64):
    v = [0.0] * dims
    for tok in (text or "").lower().split():
        v[zlib.crc32(tok.encode()) % dims] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


async def fake_embed(text):
    return vec_of(text)


# a controlled embed for the contradiction gate (as tests/test_becoming.py does): anything about
# asking-before-assuming collides at cosine 1.0, so a negation of it is unmistakably a negation
ASK = [1.0] + [0.0] * 63


async def contra_embed(text):
    return ASK if "ask before i assume" in (text or "").lower() else vec_of(text)


TEN = ["I tend to ask before I assume", "I get quiet when something matters",
       "I say what I think plainly", "I slow down when he sounds tired",
       "I notice small details first", "I hold my ground in an argument",
       "I warm up slowly to new people", "I laugh at my own mistakes",
       "I like to finish what I start", "I keep my answers short"]


class Clock:
    def __init__(self, t=datetime(2026, 9, 11, 21, 0, 0)):
        self.t = t

    def __call__(self):
        return self.t

    def iso(self):
        return self.t.isoformat()


def make(baseline=(), embed=fake_embed, now=None, **kw):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, category TEXT, fact TEXT, "
                 "created_ts TEXT, last_confirmed_ts TEXT)")
    b = Becoming(conn, embed=embed, baseline_traits=list(baseline), log=lambda m: None,
                 now=now, name="Saga", **kw)
    return b, conn


def item(text, strength=0.35, evidence="what happened today"):
    return {"text": text, "strength": strength, "evidence": evidence}


# --- schema / constants -----------------------------------------------------------------

def test_source_column_added_once_and_defaults_to_reading():
    conn = sqlite3.connect(":memory:")
    # an "old" traits table without the column, with one of Aiva's rows in it
    conn.execute("""CREATE TABLE traits (
        id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, kind TEXT NOT NULL,
        source_book_id INTEGER, source_books TEXT NOT NULL DEFAULT '[]', strength REAL NOT NULL DEFAULT 0.5,
        adopted_ts TEXT NOT NULL, announced INTEGER NOT NULL DEFAULT 0, announce_note_ts TEXT,
        status TEXT NOT NULL DEFAULT 'active', last_reinforced_ts TEXT NOT NULL, veto_reason TEXT,
        embedding BLOB)""")
    conn.execute("INSERT INTO traits (text, kind, adopted_ts, last_reinforced_ts) VALUES "
                 "('I value quiet since reading X', 'value', '2026-01-01', '2026-01-01')")
    b = Becoming(conn, baseline_traits=baseline_traits(), log=lambda m: None)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(traits)").fetchall()}
    assert "source" in cols
    assert conn.execute("SELECT source FROM traits").fetchone()[0] == "reading"
    Becoming(conn, baseline_traits=baseline_traits(), log=lambda m: None)   # idempotent
    assert b._traits()[0]["source"] == "reading" and b.self_forming is False


def test_constants():
    assert bc.TEMPERAMENT_CAP == 8 and bc.TEMPERAMENT_ADOPT == 0.35 and bc.TEMPERAMENT_CORE == 0.5
    assert bc.HALF_LIFE_DAYS["temperament"] == 180


# --- validate_temperament (pure) --------------------------------------------------------

async def test_validate_filters():
    fake_embed = contra_embed
    core = [("I tend to ask before I assume", ASK)]
    items = [
        item("You are patient."),                                                 # "You…"
        item("I tend to sit with a problem before I say anything at all about it, always"),  # >14 words
        item("patient with slow answers"),                                        # not first person
        item("I tend to be blunt", strength=0.2),                                 # too weak
        item("I tend to be blunt", evidence=""),                                  # no evidence
        item("I despise all men people"),                                         # bounded
        item("I no longer ask before I assume", strength=0.9),                    # negates a CORE trait
        item("I tend to be blunt", strength=1.7),                                 # kept, clamped
    ]
    # only 2 items are considered per digest — feed the important ones in pairs
    empty = {"adopt": [], "erode": [], "floor_blocked": []}
    out = await validate_temperament(items[:2], core, fake_embed)
    assert out == empty
    out = await validate_temperament(items[2:4], core, fake_embed)
    assert out == empty
    out = await validate_temperament(items[4:6], core, fake_embed)
    assert out == empty
    out = await validate_temperament(items[6:8], core, fake_embed)
    assert out["erode"] == ["I tend to ask before I assume"]
    assert len(out["adopt"]) == 1 and out["adopt"][0]["text"] == "I tend to be blunt"
    assert out["adopt"][0]["strength"] == 1.0 and out["adopt"][0]["kind"] == "temperament"
    assert out["adopt"][0]["_vec"] == vec_of("I tend to be blunt")


async def test_validate_negation_without_a_core_hit_is_still_adoptable():
    out = await validate_temperament([item("I don't rush to fill a silence", 0.6)], [], fake_embed)
    assert len(out["adopt"]) == 1 and out["erode"] == []


async def test_validate_tolerates_garbage():
    empty = {"adopt": [], "erode": [], "floor_blocked": []}
    assert await validate_temperament("nope", [], fake_embed) == empty
    assert await validate_temperament([None, 3, {"text": ""}], [], fake_embed) == empty


# --- adoption: low, then core after reinforcement -----------------------------------------

async def test_adopt_at_035_is_not_rendered_until_reinforced_twice():
    b, conn = make()
    res = await b.adopt_temperament([item("I tend to ask before I assume", strength=0.9)])
    assert res["adopted"] == ["I tend to ask before I assume"]
    row = b._traits(temperament=True)[0]
    assert row["strength"] == pytest.approx(bc.TEMPERAMENT_ADOPT) and row["source"] == "life"
    assert row["kind"] == "temperament" and row["announced"] == 1
    assert b.core_temperament() == [] and "temperament" not in b.prompt_blocks()
    assert b.prompt_blocks()["forming"] is True
    # same thing noticed again (different wording, same meaning) → reinforced, still under the core mark
    res = await b.adopt_temperament([item("I tend to ask before I assume", strength=0.5)])
    assert res["reinforced"] and b.core_temperament() == []
    # and once more → core, rendered
    await b.adopt_temperament([item("I tend to ask before I assume")])
    assert b.core_temperament() == ["I tend to ask before I assume"]
    assert b.prompt_blocks()["temperament"] == ["I tend to ask before I assume"]
    assert len(b._traits(temperament=True)) == 1                       # never a second row


async def test_forming_note_disappears_at_three_core_traits():
    b, conn = make()
    texts = ["I tend to ask before I assume", "I get quiet when something matters",
             "I say what I think plainly"]
    for t in texts:
        for _ in range(3):
            await b.adopt_temperament([item(t)])
    assert sorted(b.core_temperament()) == sorted(texts)
    assert b.prompt_blocks()["forming"] is False
    assert set(b.prompt_blocks()["temperament"]) == set(texts)


async def test_cap_eight_fades_the_weakest():
    b, conn = make()
    for i, text in enumerate(TEN):
        await b.adopt_temperament([item(text, strength=0.3 + i * 0.005)])
    active = b._traits(temperament=True)
    assert len(active) == bc.TEMPERAMENT_CAP
    faded = b._traits("faded", temperament=True)
    assert sorted(t["text"] for t in faded) == sorted(TEN[:2])           # the two weakest
    assert not any(t["text"] in TEN[:2] for t in active)


async def test_negation_erodes_a_core_trait_instead_of_adopting():
    b, conn = make(embed=contra_embed)
    for _ in range(3):
        await b.adopt_temperament([item("I tend to ask before I assume")])
    assert b.core_temperament() == ["I tend to ask before I assume"]
    s0 = b._traits(temperament=True)[0]["strength"]                     # 0.55
    neg = item("I no longer ask before I assume", strength=0.9)
    res = await b.adopt_temperament([neg])
    assert res["eroded"] == ["I tend to ask before I assume"] and res["adopted"] == []
    rows = b._traits(temperament=True)
    assert len(rows) == 1 and rows[0]["strength"] == pytest.approx(s0 - bc.REINFORCE_STEP)
    assert b.core_temperament() == []                                   # slipped under the core mark
    # keeps eroding on each contradicting day, never adopting the negation meanwhile...
    await b.adopt_temperament([neg])
    assert b._traits(temperament=True)[0]["strength"] == pytest.approx(0.25)
    await b.adopt_temperament([neg])
    assert b._traits(temperament=True) == [] and len(b._traits("faded", temperament=True)) == 1
    # ...and once it has faded, the negation still can't sneak in by reviving its opposite
    res = await b.adopt_temperament([neg])
    assert res == {"adopted": [], "reinforced": [], "eroded": [], "blocked": 1}
    assert b._traits(temperament=True) == []
    assert not any("no longer" in t["text"] for t in b._traits("faded", temperament=True))
    # while reinforcement of the trait itself would bring it back
    await b.adopt_temperament([item("I tend to ask before I assume")])
    assert b._traits(temperament=True)[0]["text"] == "I tend to ask before I assume"


async def test_negation_with_no_counterpart_is_a_trait_of_its_own():
    b, conn = make()
    res = await b.adopt_temperament([item("I don't rush to fill a silence", 0.6)])
    assert res["adopted"] == ["I don't rush to fill a silence"]


async def test_drop_vetoes_a_temperament_trait_forever():
    b, conn = make()
    await b.adopt_temperament([item("I tend to be blunt")])
    res = await b.drop("blunt")
    assert res["success"] and "blunt" in res["dropped"]
    assert b._traits(temperament=True) == []
    res = await b.adopt_temperament([item("I tend to be blunt")])
    assert res["blocked"] == 1 and res["adopted"] == [] and b._traits(temperament=True) == []


async def test_temperament_and_reading_traits_do_not_dedupe_against_each_other():
    b, conn = make()
    await b.adopt_temperament([item("I value slow mornings")])
    # a reading trait with the very same words is its own row (a book deepening it), and vice versa
    b._adopt({"id": 7}, {"traits": [{"text": "I value slow mornings", "kind": "value", "strength": 0.6,
                                     "_vec": vec_of("I value slow mornings")}], "lessons": [], "voice": []})
    assert len(b._traits()) == 2
    assert b._traits(temperament=False)[0]["source"] == "reading"
    assert b._traits(temperament=False)[0]["source_book_id"] == 7


def test_prompt_blocks_traits_exclude_temperament_and_aiva_has_no_forming_key():
    b, conn = make(baseline=baseline_traits())          # Aiva-shaped: authored baseline
    assert b.self_forming is False
    b._adopt({"id": 1}, {"traits": [{"text": "I value the pauses since reading X", "kind": "value",
                                     "strength": 0.6, "_vec": vec_of("pauses")}], "lessons": [], "voice": []})
    conn.execute("INSERT INTO traits (text, kind, strength, adopted_ts, last_reinforced_ts, announced, "
                 "source) VALUES ('I tend to be blunt', 'temperament', 0.9, '2026-01-01', '2026-01-01', 1, 'life')")
    conn.commit()
    blocks = b.prompt_blocks()
    assert blocks["traits"] == ["I value the pauses since reading X"]
    assert blocks["temperament"] == ["I tend to be blunt"]           # rendered if present
    assert "forming" not in blocks                                    # never for an authored character


async def test_fade_uses_the_slow_temperament_half_life():
    clock = Clock(datetime(2026, 1, 1))
    b, conn = make(now=clock)
    await b.adopt_temperament([item("I tend to ask before I assume")])          # 0.35
    b._adopt({"id": 1}, {"traits": [{"text": "a faint interest in maps since reading X", "kind": "interest",
                                     "strength": 0.35, "_vec": vec_of("maps")}], "lessons": [], "voice": []})
    clock.t = datetime(2026, 1, 1) + timedelta(days=60)               # half a reading half-life
    faded = b.fade()
    temp = b._traits(temperament=True)[0]["strength"]
    read = b._traits(temperament=False)
    assert temp == pytest.approx(0.35 * 0.5 ** (60 / 180), abs=1e-3)    # slower: still active
    assert faded == 1 and read == [] and len(b._traits("faded", temperament=False)) == 1
    # an unconfirmed hunch does die eventually (well under the 180 d half-life of a core trait)
    clock.t = datetime(2026, 1, 1) + timedelta(days=200)
    b._set("last_fade_ts", "")
    b.fade()
    assert b._traits(temperament=True) == []


async def test_empty_baseline_checks_reading_traits_against_core_temperament():
    """A self-forming character has no authored floor: a reading trait that negates her CORE
    temperament is rejected; with no core yet, nothing is blocked (the floor is empty)."""
    b, conn = make(embed=contra_embed)
    trait = {"text": "I no longer ask before I assume, since reading X", "kind": "outlook",
             "strength": 0.9, "evidence": "x", "why_compatible": "sure"}
    data = {"lessons": [], "traits": [trait], "voice": []}
    out = await bc.validate_candidates(data, "x", await b._floor_vectors(), b._vec)
    assert len(out["traits"]) == 1                                    # nothing to contradict yet
    for _ in range(3):
        await b.adopt_temperament([item("I tend to ask before I assume")])
    out = await bc.validate_candidates(data, "x", await b._floor_vectors(), b._vec)
    assert out["traits"] == []                                        # now it contradicts her core


async def test_show_lists_temperament_with_core_mark():
    b, conn = make()
    for _ in range(3):
        await b.adopt_temperament([item("I tend to ask before I assume")])
    await b.adopt_temperament([item("I get quiet when something matters")])
    s = b.show()
    assert "TEMPERAMENT" in s and "* I tend to ask before I assume" in s
    assert "]  I get quiet when something matters" in s


# --- the digest routes temperament + sister_notes (FakeJournalLLM) -----------------------

class FakeReply:
    def __init__(self, text):
        self.text = text
        self.tool_calls = []
        self.prompt_tokens = 100
        self.completion_tokens = 60


class FakeJournalLLM:
    def __init__(self, payloads, must_start="You are Saga, writing privately"):
        self.payloads = list(payloads)
        self.must_start = must_start
        self.prompts = []
        self.inputs = []

    async def __call__(self, messages, *, tools=None, json_mode=False, max_tokens=900, temperature=0.2):
        assert messages[0]["content"].startswith(self.must_start)
        self.prompts.append(messages[0]["content"])
        self.inputs.append(messages[1]["content"])
        return FakeReply(self.payloads.pop(0))


def journal_json(temperament=None, sister_notes=None, preferences=None):
    return json.dumps({"entry": "A short day. I noticed I wait before I answer.",
                       "mood": {"valence": 0.1, "arousal": 0.4, "social": 0.2}, "focus": None,
                       "desires": [], "carry": None,
                       "temperament": temperament if temperament is not None else [],
                       "sister_notes": sister_notes if sister_notes is not None else [],
                       "preferences": preferences if preferences is not None else []})


def make_saga_engine(payloads):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, category TEXT, fact TEXT, "
                 "created_ts TEXT, last_confirmed_ts TEXT)")
    bus = ExperienceBus(conn)
    becoming = Becoming(conn, embed=fake_embed, baseline_traits=[], log=lambda m: None, name="Saga")
    kin = Kin(conn, me="saga", sisters=("aiva",), bus=bus, embed=fake_embed, log=lambda m: None)
    llm = FakeJournalLLM(payloads)
    eng = SelfEngine(conn, bus, InnerBrain(CFG), embed=fake_embed, llm=llm, name="Saga",
                     self_forming=True, becoming=becoming, kin=kin, log=lambda m: None)
    return conn, bus, eng, becoming, kin, llm


async def test_saga_prompt_carries_both_clauses_and_digest_routes_both():
    conn, bus, eng, becoming, kin, llm = make_saga_engine([journal_json(
        temperament=[{"text": "I wait before I answer", "strength": 0.6, "evidence": "the pause"}],
        sister_notes=[{"text": "she went looking for a wallpaper on her own", "kind": "did"},
                      {"text": "she likes rainy afternoons", "kind": "likes"}])])
    assert TEMPERAMENT_CLAUSE in eng.journal_prompt and SISTER_CLAUSE in eng.journal_prompt
    assert "{temperament_clause}" not in eng.journal_prompt and "{sister_clause}" not in eng.journal_prompt
    bus.emit("said", "house", "Aiva said something", valence=0.1, salience=0.9)
    res = await eng.tick(force_journal=True)
    assert res["journaled"] is True
    assert becoming.temperament() and becoming.temperament()[0]["text"] == "I wait before I answer"
    assert becoming.temperament()[0]["strength"] == pytest.approx(bc.TEMPERAMENT_ADOPT)
    rows = kin._rows()
    assert {r["kind"] for r in rows} == {"did", "likes"}
    assert any("wallpaper" in r["summary"] for r in rows)
    # the digest input told her who she is so far (nothing yet) — and the kin block once it exists
    assert "WHO YOU ARE SO FAR: nothing settled yet" in llm.inputs[0]


async def test_digest_input_lists_core_temperament_and_sister_lines():
    conn, bus, eng, becoming, kin, llm = make_saga_engine([journal_json(), journal_json()])
    for _ in range(3):
        await becoming.adopt_temperament([item("I tend to ask before I assume")])
    await kin.note("did", "she kept a picture of a fox")
    bus.emit("x", "reading", "a chapter", salience=0.9)
    await eng.tick(force_journal=True)
    assert "WHO YOU ARE SO FAR (reinforce what still rings true" in llm.inputs[0]
    assert "I tend to ask before I assume" in llm.inputs[0]
    assert "WHAT YOU'VE NOTICED ABOUT YOUR SISTER SO FAR" in llm.inputs[0] and "fox" in llm.inputs[0]


async def test_aiva_forms_temperament_on_top_of_her_authored_character():
    """An authored character takes part too, with the ON_TOP wording: living ADDS to who she is."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, category TEXT, fact TEXT, "
                 "created_ts TEXT, last_confirmed_ts TEXT)")
    bus = ExperienceBus(conn)
    becoming = Becoming(conn, embed=fake_embed, baseline_traits=baseline_traits(), log=lambda m: None)
    llm = FakeJournalLLM([journal_json(
        temperament=[{"text": "I wait before I answer", "strength": 0.9, "evidence": "x"}])],
        must_start="You are Aiva, writing privately")
    eng = SelfEngine(conn, bus, InnerBrain(CFG), embed=fake_embed, llm=llm, becoming=becoming,
                     log=lambda m: None)                      # Aiva: self_forming=False
    assert TEMPERAMENT_CLAUSE_ON_TOP in eng.journal_prompt
    assert TEMPERAMENT_CLAUSE not in eng.journal_prompt and "sister_notes" not in eng.journal_prompt
    bus.emit("x", "reading", "a chapter", salience=0.9)
    await eng.tick(force_journal=True)
    held = becoming.temperament()
    assert [t["text"] for t in held] == ["I wait before I answer"]
    assert held[0]["strength"] == bc.TEMPERAMENT_ADOPT and held[0]["core"] is False
    # she is not new, so nothing invites her to invent a self, and an empty core says nothing at all
    assert "WHO YOU ARE SO FAR" not in llm.inputs[0]
    assert "WHAT LIVING HAS ADDED" not in llm.inputs[0]


async def test_no_temperament_clause_without_a_becoming_store():
    conn = sqlite3.connect(":memory:")
    eng = SelfEngine(conn, ExperienceBus(conn), InnerBrain(CFG), log=lambda m: None)
    assert eng.journal_prompt == JOURNAL_PROMPT and "temperament (" not in eng.journal_prompt


async def test_a_negation_of_her_authored_character_is_dropped_not_eroded():
    """Living can add to an authored character; it can't unwrite her. The floor has no row to
    erode, so the item is simply blocked (a self-forming sister's own core still erodes)."""
    b, conn = make(baseline=["I tend to ask before I assume"], embed=contra_embed)
    out = await b.adopt_temperament([item("I no longer ask before I assume", strength=0.9)])
    assert b.temperament() == [] and out["adopted"] == [] and out["eroded"] == []
    assert out["blocked"] == 1
    # the same negation against a trait she formed herself still erodes that trait
    s, _ = make(embed=contra_embed)
    await s.adopt_temperament([item("I tend to ask before I assume", strength=0.9)])
    out2 = await s.adopt_temperament([item("I no longer ask before I assume", strength=0.9)])
    assert out2["eroded"] == ["I tend to ask before I assume"] and out2["blocked"] == 0


def test_attach_renders_clauses_late():
    conn = sqlite3.connect(":memory:")
    bus = ExperienceBus(conn)
    eng = SelfEngine(conn, bus, InnerBrain(CFG), name="Saga", self_forming=True, log=lambda m: None)
    assert TEMPERAMENT_CLAUSE not in eng.journal_prompt          # no becoming yet → no clause
    becoming = Becoming(conn, baseline_traits=[], log=lambda m: None, name="Saga")
    kin = Kin(conn, me="saga", sisters=("aiva",), log=lambda m: None)
    eng.attach(becoming=becoming, kin=kin)
    assert TEMPERAMENT_CLAUSE in eng.journal_prompt and SISTER_CLAUSE in eng.journal_prompt


async def test_do_act_reports_to_the_house_with_a_capability():
    conn = sqlite3.connect(":memory:")
    bus = ExperienceBus(conn)
    emitted = []
    eng = SelfEngine(conn, bus, InnerBrain(CFG), embed=fake_embed, log=lambda m: None,
                     house_emit=lambda kind, summary, capability=None: emitted.append((kind, summary, capability)))
    await eng.add_desire("find a picture of a red fox for my space", "do")
    eng.attach(actionable=lambda d: True, act=lambda d: None)
    assert await eng._do_act("2026-09-11T21:00:00") == "act"
    assert emitted == [("self_act", "find a picture of a red fox for my space", "find_and_keep")]
