"""Phase 2 tests: the self loop (mood integration + diary + desires), fully offline with
a FakeLLM and a fake clock. The journal call uses the same (messages, *, json_mode, ...)
contract as tasks.openai_llm_call, so the fake mirrors the reading-test style."""
import json
import math
import sqlite3
from datetime import datetime, timedelta

import pytest

from modules.inner import (JOURNAL_PROMPT, DESIRE_KINDS, ExperienceBus, InnerBrain,
                           InnerConfig, SelfEngine)
from modules.preferences import Preferences

CFG = InnerConfig(base_valence=0.15, base_arousal=0.50, base_social=0.10,
                  halflife_h=6.0, salience_floor=0.15, gain=0.60, material_eps=0.05)
T0 = datetime(2026, 8, 27, 20, 0, 0)


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t.isoformat()

    def advance(self, hours):
        self.t += timedelta(hours=hours)


def fake_embed_factory(dims=64):
    import zlib

    async def embed(text):
        # deterministic across processes (Python's hash() is salted per-run)
        vec = [0.0] * dims
        for tok in (text or "").lower().split():
            vec[zlib.crc32(tok.encode()) % dims] += 1.0
        n = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / n for v in vec]
    return embed


class FakeReply:
    def __init__(self, text, pt=200, ct=120):
        self.text = text
        self.tool_calls = []
        self.prompt_tokens = pt
        self.completion_tokens = ct


def journal_json(entry="Today the quiet felt companionable.", valence=0.2, arousal=0.4,
                 social=0.3, focus="the quiet of the room", desires=None, carry="be gentler tomorrow",
                 preferences=None):
    return json.dumps({"entry": entry, "mood": {"valence": valence, "arousal": arousal,
                       "social": social}, "focus": focus,
                       "desires": desires if desires is not None else [],
                       "preferences": preferences if preferences is not None else [],
                       "carry": carry})


class FakeJournalLLM:
    """Returns queued journal payloads; dispatches on the JOURNAL_PROMPT prefix like the
    reading fake. `bad_once` returns non-JSON on the first call."""
    def __init__(self, payloads=None, bad_once=False):
        self.payloads = list(payloads or [])
        self.bad_once = bad_once
        self._bad = False
        self.calls = 0

    async def __call__(self, messages, *, tools=None, json_mode=False, max_tokens=900, temperature=0.2):
        assert messages[0]["content"].startswith(JOURNAL_PROMPT[:40])
        self.calls += 1
        if self.bad_once and not self._bad:
            self._bad = True
            return FakeReply("this is not json at all")
        return FakeReply(self.payloads.pop(0) if self.payloads else journal_json())


def make_engine(llm=None, embed=None, clock=None, **kw):
    con = sqlite3.connect(":memory:")
    bus = ExperienceBus(con)
    eng = SelfEngine(con, bus, InnerBrain(CFG), embed=embed, llm=llm,
                     now=clock or Clock(), **kw)
    return con, bus, eng


# -- tick: mood integration --------------------------------------------------------

async def test_tick_integrates_events_and_moves_mood():
    con, bus, eng = make_engine()
    bus.emit("chapter_read", "reading", "a bleak chapter", valence=-0.4, arousal=-0.2, salience=0.8)
    v0 = eng.state.valence
    res = await eng.tick()
    assert res["changed"] is True and res["events"] == 1
    assert eng.state.valence < v0                    # mood moved down
    assert bus.drain() == []                          # event consumed


async def test_tick_without_llm_never_journals():
    con, bus, eng = make_engine(llm=None)
    bus.emit("task_failed", "tasks", "build broke", valence=-0.5, salience=0.9)
    res = await eng.tick(force_journal=True)          # even forced: no llm -> no diary
    assert res["journaled"] is False
    assert eng.journal_entries() == []


# -- journaling --------------------------------------------------------------------

async def test_forced_journal_writes_entry_focus_and_desires():
    llm = FakeJournalLLM([journal_json(
        entry="Reading about isolation hit close; the cat helped.",
        focus="isolation in All Systems Red",
        desires=[{"text": "ask him how the Fianu work is going", "kind": "ask"},
                 {"text": "read more of All Systems Red", "kind": "do"}])])
    con, bus, eng = make_engine(llm=llm, embed=fake_embed_factory())
    bus.emit("chapter_read", "reading", "isolation chapter", valence=-0.3, salience=0.7)
    res = await eng.tick(force_journal=True)
    assert res["journaled"] is True
    rows = eng.journal_entries()
    assert len(rows) == 1 and "isolation" in rows[0]["entry"]
    assert eng.state.focus == "isolation in All Systems Red"
    wants = {d["text"] for d in eng.list_desires()}
    assert "ask him how the Fianu work is going" in wants
    assert eng.list_desires()[0]["kind"] in DESIRE_KINDS


async def test_journal_records_current_integrated_mood_not_llm_mood():
    # the llm claims a cheerful mood, but events pushed her down — the row must reflect events
    llm = FakeJournalLLM([journal_json(valence=0.9, arousal=0.9, social=0.9)])
    con, bus, eng = make_engine(llm=llm, embed=fake_embed_factory())
    bus.emit("task_failed", "tasks", "it broke", valence=-0.6, salience=1.0)
    await eng.tick(force_journal=True)
    row = eng.journal_entries()[0]
    assert row["valence"] == pytest.approx(round(eng.state.valence, 3))
    assert row["valence"] < 0                          # not the llm's +0.9


async def test_journal_triggers_on_salience_threshold():
    llm = FakeJournalLLM()
    con, bus, eng = make_engine(llm=llm, embed=fake_embed_factory(), journal_salience=1.5)
    bus.emit("a", "reading", "one", valence=-0.2, salience=0.8)
    bus.emit("b", "cat", "two", valence=0.2, salience=0.8)   # 0.8+0.8 = 1.6 >= 1.5
    res = await eng.tick()
    assert res["journaled"] is True
    assert eng._salience_since_journal == 0.0          # reset after journaling


async def test_below_threshold_does_not_journal():
    llm = FakeJournalLLM()
    con, bus, eng = make_engine(llm=llm, embed=fake_embed_factory(), journal_salience=5.0)
    bus.emit("a", "reading", "one", valence=-0.2, salience=0.8)
    res = await eng.tick()
    assert res["journaled"] is False and eng.journal_entries() == []


async def test_bad_json_falls_back_to_plain_entry():
    llm = FakeJournalLLM(bad_once=True)
    con, bus, eng = make_engine(llm=llm, embed=fake_embed_factory())
    bus.emit("chapter_read", "reading", "a memorable scene", valence=-0.2, salience=0.8)
    res = await eng.tick(force_journal=True)
    assert res["journaled"] is True
    row = eng.journal_entries()[0]
    assert "memorable scene" in row["entry"]           # built from the event
    assert eng.list_desires() == []                    # no desires from a failed parse


# -- desires: dedupe, veto, cap ----------------------------------------------------

async def test_desire_dedupe_reinforces():
    con, bus, eng = make_engine(embed=fake_embed_factory())
    assert await eng.add_desire("ask him how the Fianu project is going", "ask") == "inserted"
    s1 = eng.list_desires()[0]["strength"]
    assert await eng.add_desire("ask him how the Fianu work is going", "ask") == "reinforced"
    open_now = eng.list_desires()
    assert len(open_now) == 1 and open_now[0]["strength"] > s1


async def test_distinct_desires_coexist():
    con, bus, eng = make_engine(embed=fake_embed_factory())
    await eng.add_desire("read more science fiction", "do")
    await eng.add_desire("learn to make better coffee", "learn")
    assert len(eng.list_desires()) == 2


async def test_drop_desire_blocks_readoption():
    con, bus, eng = make_engine(embed=fake_embed_factory())
    await eng.add_desire("rearrange the desktop icons", "do")
    dropped = await eng.drop_desire("rearrange the desktop")   # substring match
    assert dropped and "rearrange" in dropped
    assert eng.list_desires(status="open") == []
    assert await eng.add_desire("rearrange the desktop icons", "do") == "blocked"


async def test_drop_desire_by_id():
    con, bus, eng = make_engine(embed=fake_embed_factory())
    await eng.add_desire("water the plants", "do")
    did = eng.list_desires()[0]["id"]
    assert await eng.drop_desire(did) is not None
    assert eng.list_desires(status="dropped")[0]["id"] == did


async def test_desire_cap_fades_weakest():
    con, bus, eng = make_engine(embed=fake_embed_factory(), desire_cap=2)
    await eng.add_desire("first wish entirely unique alpha", "do", strength=0.9)
    await eng.add_desire("second wish entirely unique beta", "do", strength=0.8)
    await eng.add_desire("third wish entirely unique gamma", "do", strength=0.3)  # weakest, over cap
    assert len(eng.list_desires(status="open")) == 2
    assert any("gamma" in d["text"] for d in eng.list_desires(status="dropped"))


# -- preferences: formed by the digest ---------------------------------------------

def make_engine_with_prefs(payloads, clock=None):
    con = sqlite3.connect(":memory:")
    bus = ExperienceBus(con)
    embed = fake_embed_factory()
    prefs = Preferences(con, embed=embed, log=lambda m: None)
    eng = SelfEngine(con, bus, InnerBrain(CFG), embed=embed,
                     llm=FakeJournalLLM(payloads), now=clock or Clock(), preferences=prefs)
    return con, bus, eng, prefs


async def test_digest_forms_a_like_from_a_positive_experience():
    con, bus, eng, prefs = make_engine_with_prefs([journal_json(
        entry="That gentle sci-fi really landed.",
        preferences=[{"subject": "character-driven sci-fi", "kind": "taste",
                      "stance": "I love quiet, character-driven sci-fi", "direction": 1,
                      "magnitude": 0.6}])])
    bus.emit("book_finished", "reading", "finished a warm sci-fi novel", valence=0.4, salience=0.9)
    await eng.tick(force_journal=True)
    active = prefs.list_prefs()
    assert len(active) == 1 and active[0]["valence"] > 0
    assert "character-driven" in active[0]["stance"]


async def test_digest_erodes_before_flipping_on_opposite_evidence():
    # a like forms, then sustained opposite evidence erodes it toward 0 and eventually flips
    likes = [journal_json(preferences=[{"subject": "loud action", "kind": "taste",
             "stance": "I love loud action", "direction": 1, "magnitude": 0.6}])]
    against = [journal_json(preferences=[{"subject": "loud action", "kind": "taste",
              "stance": "loud action is wearing on me", "direction": -1, "magnitude": 0.6}])
              for _ in range(8)]
    con, bus, eng, prefs = make_engine_with_prefs(likes + against)
    bus.emit("x", "reading", "a rousing chapter", valence=0.3, salience=0.9)
    await eng.tick(force_journal=True)
    assert prefs.list_prefs()[0]["valence"] > 0            # liked at first
    seen_positive_after = False
    for _ in range(8):
        bus.emit("y", "reading", "another loud set-piece", valence=-0.2, salience=0.9)
        await eng.tick(force_journal=True)
        v = prefs.list_prefs()[0]["valence"]
        if v > 0:
            seen_positive_after = True
    assert seen_positive_after                              # eroded gradually (didn't flip instantly)
    assert prefs.list_prefs()[0]["valence"] < 0            # but eventually flipped to a dislike


async def test_digest_without_pref_store_is_harmless():
    con, bus, eng = make_engine(llm=FakeJournalLLM([journal_json(
        preferences=[{"subject": "x", "stance": "I like x", "direction": 1, "magnitude": 0.5}])]),
        embed=fake_embed_factory())
    bus.emit("z", "reading", "something", valence=0.3, salience=0.9)
    res = await eng.tick(force_journal=True)               # no preferences ref → just ignored
    assert res["journaled"] is True


# -- recall ------------------------------------------------------------------------

async def test_recall_journal_ranks_by_relevance():
    llm = FakeJournalLLM([journal_json(entry="The storm scene about sailing stayed with me.", focus=None),
                          journal_json(entry="I kept thinking about coffee and mornings.", focus=None)])
    con, bus, eng = make_engine(llm=llm, embed=fake_embed_factory())
    await eng.tick(force_journal=True)
    await eng.tick(force_journal=True)
    hits = await eng.recall_journal("coffee mornings", k=1)
    assert len(hits) == 1 and "coffee" in hits[0]["entry"]
