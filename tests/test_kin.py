"""Kin (plan B5): what a sister has actually seen the other do. Offline: in-memory sqlite, a
deterministic fake embed, her own ExperienceBus to catch the `sister` events, house_proto.Event
objects and dict events both accepted. Also covers the tools.py tool hook and the kin handlers."""
import math
import sqlite3
import types
import zlib
from datetime import datetime, timedelta

import pytest

from modules import capabilities as cap
from modules import kin as kinmod
from modules.house_memory import HouseMemory
from modules.house_proto import Event
from modules.inner import ExperienceBus
from modules.kin import KIN_LINES, TOOL_TO_CAPABILITY, Kin
from modules.tasks import RUNNER_TOOL_ALLOW
from modules.tools import (TOOL_SCHEMAS, _timed, bind_tool_hook, make_kin_handlers,
                           register_kin_tools)


def vec_of(text, dims=64):
    v = [0.0] * dims
    for tok in (text or "").lower().split():
        v[zlib.crc32(tok.encode()) % dims] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


async def fake_embed(text):
    return vec_of(text)


class Clock:
    def __init__(self, t=datetime(2026, 9, 11, 20, 0, 0)):
        self.t = t

    def __call__(self):
        return self.t.isoformat()


def make(embed=fake_embed, with_bus=True, clock=None, sisters=("saga",)):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, category TEXT, fact TEXT, "
                 "created_ts TEXT, last_confirmed_ts TEXT)")
    bus = ExperienceBus(conn) if with_bus else None
    k = Kin(conn, me="aiva", sisters=sisters, bus=bus, embed=embed, log=lambda m: None, now=clock or Clock())
    return k, conn, bus


def ev(kind, summary="", *, actor="saga", capability=None, payload=None, event_id=None):
    return Event(kind=kind, summary=summary, capability=capability, payload=payload, actor=actor,
                 event_id=event_id)


def sister_events(bus):
    return [e for e in bus.recent(100) if e["kind"] == "sister"]


# --- the registry-derived map + authored lines --------------------------------------------

def test_tool_map_excludes_start_task_and_claude_code_and_covers_every_capability():
    assert "start_task" not in TOOL_TO_CAPABILITY and "claude_code" not in TOOL_TO_CAPABILITY
    names = {c.name for c in cap.CAPABILITIES}
    assert set(TOOL_TO_CAPABILITY.values()) <= names
    assert set(KIN_LINES) == names                                # a line for every capability
    assert all(line.startswith(("she ", "she'")) for line in KIN_LINES.values())
    assert TOOL_TO_CAPABILITY["search_web"] == "look_up_info"
    assert TOOL_TO_CAPABILITY["save_to_space"] == "find_and_keep"
    assert TOOL_TO_CAPABILITY["recall_about_sister"] == "know_sister"
    assert TOOL_TO_CAPABILITY["launch_app"] == "use_the_pc"
    assert "avatar_expression" not in TOOL_TO_CAPABILITY and "get_weather" not in TOOL_TO_CAPABILITY


# --- observe ---------------------------------------------------------------------------

async def test_observe_ignores_her_own_acts_and_strangers():
    k, conn, bus = make()
    assert (await k.observe(ev("tool_done", payload={"tool": "search_web"}, actor="aiva")))["status"] == "ignored"
    assert (await k.observe(ev("tool_done", payload={"tool": "search_web"}, actor="")))["status"] == "ignored"
    assert (await k.observe(ev("tool_done", payload={"tool": "search_web"}, actor="bob")))["status"] == "ignored"
    assert k._rows() == [] and sister_events(bus) == []


async def test_tool_done_rolls_up_by_capability():
    k, conn, bus = make()
    r1 = await k.observe(ev("tool_done", "used search_web", payload={"tool": "search_web"}))
    assert r1["status"] == "new"
    r2 = await k.observe(ev("tool_done", "used read_page", payload={"tool": "read_page"}))
    assert r2["status"] == "rolled_up" and r2["id"] == r1["id"] and r2["count"] == 2
    rows = k._rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "did" and row["capability"] == "look_up_info"
    assert row["summary"] == KIN_LINES["look_up_info"]
    assert row["count"] == 2 and row["strength"] == pytest.approx(kinmod.INITIAL_STRENGTH + kinmod.ROLLUP_STEP)
    assert row["last_example"] == "used read_page"
    # the capability may also arrive already resolved (house_events rows carry no payload)
    r3 = await k.observe({"actor": "saga", "kind": "tool_done", "summary": "", "capability": "look_up_info"})
    assert r3["status"] == "rolled_up" and r3["count"] == 3


async def test_unmapped_and_excluded_tools_are_ignored():
    k, conn, bus = make()
    for tool in ("avatar_expression", "get_weather", "start_task", "claude_code", None):
        res = await k.observe(ev("tool_done", payload={"tool": tool}))
        assert res["status"] == "ignored"
    assert k._rows() == []


async def test_self_act_maps_via_capabilities_match():
    k, conn, bus = make()
    res = await k.observe(ev("self_act", "find a picture of a red fox for my space"))
    assert res["status"] == "new"
    row = k._rows()[0]
    assert row["capability"] == "find_and_keep" and row["last_example"].startswith("on her own:")
    # an act nothing matches is still hers, kept as a plain 'did' line
    res = await k.observe(ev("self_act", "hummed to herself for a while"))
    assert res["status"] == "new" and k._rows(limit=10)[-1]["capability"] is None


async def test_find_saved_and_said_dedupe_by_meaning():
    k, conn, bus = make()
    a = await k.observe(ev("find_saved", "image: a blue enamel hairclip with a star charm"))
    b = await k.observe(ev("find_saved", "image: a blue enamel hairclip with a star charm"))
    assert a["status"] == "new" and b["status"] == "rolled_up" and b["id"] == a["id"]
    c = await k.observe(ev("find_saved", "text: notes on sourdough starters"))
    assert c["status"] == "new"
    s1 = await k.observe(ev("said", "I think ambiguous endings are underrated"))
    s2 = await k.observe(ev("said", "I think ambiguous endings are underrated"))
    s3 = await k.observe(ev("said", "the cat has opinions about my keyboard"))
    assert s1["status"] == "new" and s2["status"] == "rolled_up" and s3["status"] == "new"
    kinds = {r["kind"] for r in k._rows()}
    assert kinds == {"made", "said"}
    made = [r for r in k._rows() if r["kind"] == "made"]
    assert all(r["summary"].startswith("she made or kept: ") for r in made)
    said = [r for r in k._rows() if r["kind"] == "said"]
    assert all(r["summary"].startswith('she said: "') for r in said)


async def test_read_and_task_done_and_presence_events():
    k, conn, bus = make()
    assert (await k.observe(ev("book_finished", "finished All Systems Red by Martha Wells")))["status"] == "new"
    assert (await k.observe(ev("session_end", "read 3 sections of All Systems Red")))["status"] == "new"
    assert (await k.observe(ev("task_done", "opened Notepad and saved the list", payload={"family": "use_the_pc"})))["status"] == "new"
    assert (await k.observe(ev("task_done", "built the parser with Claude Code", payload={"family": "code-changes"})))["status"] == "new"
    assert (await k.observe(ev("task_done", "renamed the screenshots", payload={"family": "file-management"})))["status"] == "rolled_up"
    for kind in ("joined", "left", "absent", "weird"):
        assert (await k.observe(ev(kind, "x")))["status"] == "ignored"
    caps = {r["capability"] for r in k._rows() if r["kind"] == "did"}
    assert caps == {"use_the_pc", "build_software"}
    assert [r["kind"] for r in k._rows() if r["kind"] == "read"] == ["read", "read"]


async def test_lexical_fallback_without_embeddings():
    k, conn, bus = make(embed=None)
    a = await k.observe(ev("said", "I really love quiet rainy afternoons at home"))
    b = await k.observe(ev("said", "I really love quiet rainy afternoons at home"))
    assert a["status"] == "new" and b["status"] == "rolled_up"


# --- her own bus: felt at 0.2, then quietly logged --------------------------------------

async def test_new_observation_and_first_three_rollups_are_felt_then_quiet():
    k, conn, bus = make()
    for i in range(6):
        await k.observe(ev("tool_done", f"used search_web {i}", payload={"tool": "search_web"}))
    evs = list(reversed(sister_events(bus)))                  # oldest first
    assert len(evs) == 6
    assert [round(e["salience"], 2) for e in evs] == [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
    assert all(e["source"] == "kin" and e["social_hint"] == pytest.approx(0.05) for e in evs)
    assert evs[0]["summary"] == "Saga can look things up online and read pages"


async def test_said_events_never_move_her_mood():
    k, conn, bus = make()
    await k.observe(ev("said", "hello there"))
    assert sister_events(bus)[0]["salience"] == pytest.approx(kinmod.SALIENCE_LATER)


async def test_no_bus_is_fine():
    k, conn, bus = make(with_bus=False)
    assert (await k.observe(ev("tool_done", payload={"tool": "search_web"})))["status"] == "new"


# --- note (digest-derived) ----------------------------------------------------------------

async def test_note_records_without_emitting():
    k, conn, bus = make()
    assert (await k.note("likes", "she likes rainy afternoons"))["status"] == "new"
    assert (await k.note("did", "went looking for a wallpaper on her own"))["status"] == "new"
    assert (await k.note("bogus", "she hums"))["status"] == "new"        # unknown kind → did
    assert await k.note("did", "") is None
    assert {r["kind"] for r in k._rows()} == {"likes", "did"}
    assert sister_events(bus) == []


# --- prompt_block / can_do / recall -------------------------------------------------------

async def test_prompt_block_caps_at_limit_and_includes_sister_facts():
    k, conn, bus = make()
    for i in range(8):
        await k.observe(ev("find_saved", f"image: find{i} topic{i} thing{i}"))
    for _ in range(7):
        await k.observe(ev("tool_done", "used search_web", payload={"tool": "search_web"}))
    conn.execute("INSERT INTO facts (category, fact, created_ts, last_confirmed_ts) VALUES "
                 "('sister', 'Saga may read whenever she likes', '2026-09-01', '2026-09-01')")
    conn.execute("INSERT INTO facts (category, fact, created_ts, last_confirmed_ts) VALUES "
                 "('person', 'Frank works at Fianu', '2026-09-01', '2026-09-01')")
    conn.commit()
    lines = k.prompt_block(limit=6)
    assert len(lines) == 6
    assert lines[0] == "she can look things up online and read pages (seen 7×, last today)"
    assert "Saga may read whenever she likes" in lines and "Frank works at Fianu" not in lines
    assert sum(1 for l in lines if l.startswith("she made or kept")) == 4
    assert k.prompt_block(limit=2) == [lines[0], "Saga may read whenever she likes"]


def test_prompt_block_without_a_facts_table_or_rows():
    conn = sqlite3.connect(":memory:")
    k = Kin(conn, me="saga", sisters=("aiva",), log=lambda m: None)
    assert k.prompt_block() == []


async def test_can_do_lists_distinct_capabilities():
    k, conn, bus = make()
    await k.observe(ev("tool_done", payload={"tool": "search_web"}))
    await k.observe(ev("tool_done", payload={"tool": "read_page"}))
    await k.observe(ev("tool_done", payload={"tool": "feed_cat"}))
    await k.observe(ev("find_saved", "image: a fox"))
    cd = k.can_do()
    assert [c["capability"] for c in cd] == ["look_up_info", "tend_cat"]
    assert cd[0]["count"] == 2 and cd[0]["line"] == KIN_LINES["look_up_info"]


async def test_recall_ranks_and_is_honest_when_empty():
    k, conn, bus = make()
    empty = await k.recall("has she ever looked something up")
    assert empty["items"] == [] and "haven't seen her do anything like that" in empty["note"]
    await k.observe(ev("tool_done", "used search_web", payload={"tool": "search_web"}))
    await k.observe(ev("find_saved", "image: a red fox in snow"))
    hit = await k.recall("did she look things up online")
    assert hit["items"] and hit["items"][0]["what"] == KIN_LINES["look_up_info"]
    assert hit["items"][0]["times"] == 1 and hit["items"][0]["last"] == "today"
    assert "actually seen her do" in hit["note"]
    miss = await k.recall("cook a lasagna dinner tonight")
    assert miss["items"] == [] and "offer to ask her" in miss["note"]
    latest = await k.recall("")
    assert len(latest["items"]) == 2


# --- veto / fade / caps -------------------------------------------------------------------

async def test_drop_blocks_relearning():
    k, conn, bus = make()
    await k.observe(ev("tool_done", payload={"tool": "search_web"}))
    dropped = await k.drop("look things up")
    assert dropped == KIN_LINES["look_up_info"]
    assert k._rows() == [] and k.prompt_block() == []
    res = await k.observe(ev("tool_done", payload={"tool": "read_page"}))
    assert res["status"] == "blocked" and k._rows() == []
    assert await k.drop("something never seen") is None
    await k.observe(ev("find_saved", "image: a fox"))
    fid = k._rows()[0]["id"]
    assert await k.drop(fid) is not None and k._rows() == []


async def test_fade_half_life_and_floor():
    clock = Clock(datetime(2026, 1, 1))
    k, conn, bus = make(clock=clock)
    await k.observe(ev("tool_done", payload={"tool": "search_web"}))              # 0.3
    for _ in range(8):
        await k.observe(ev("find_saved", "image: a fox"))                         # 0.3 + 7*0.1 = 1.0
    clock.t = datetime(2026, 1, 1) + timedelta(days=90)
    assert k.fade() == 2
    by_kind = {r["kind"]: r["strength"] for r in k._rows()}
    assert by_kind["made"] == pytest.approx(0.5, abs=1e-3)
    assert by_kind["did"] == pytest.approx(0.15, abs=1e-3)
    assert k.fade() == 0                                                          # stamped: not again yet
    clock.t += timedelta(days=3650)
    assert k.fade() == 2
    assert all(r["strength"] == pytest.approx(kinmod.FADE_FLOOR) for r in k._rows())


async def test_said_rows_are_bounded():
    k, conn, bus = make(embed=None)                                  # lexical: deterministic
    for i in range(kinmod.ROW_CAPS["said"] + 5):
        await k.observe(ev("said", f"alpha{i} beta{i} gamma{i} delta{i}"))   # no shared words
    assert len([r for r in k._rows(limit=500) if r["kind"] == "said"]) == kinmod.ROW_CAPS["said"]


# --- cursor + catch-up from house_events --------------------------------------------------

async def test_catch_up_advances_cursor_and_skips_own_rows():
    k, conn, bus = make()
    house = HouseMemory(sqlite3.connect(":memory:"))
    house.add_event("saga", "tool_done", "used search_web", capability="look_up_info")
    house.add_event("aiva", "find_saved", "image: mine", capability=None)
    house.add_event("saga", "find_saved", "image: a red fox in snow")
    house.add_event("saga", "joined", "Saga joined")
    assert k.cursor == 0
    n = await k.catch_up(house.events_since(k.cursor))
    assert n == 2 and k.cursor == 4
    assert {r["kind"] for r in k._rows()} == {"did", "made"}
    assert await k.catch_up(house.events_since(k.cursor)) == 0     # nothing new
    house.add_event("saga", "said", "hello you")
    assert await k.catch_up(house.events_since(k.cursor)) == 1 and k.cursor == 5


async def test_observe_advances_cursor_from_relayed_event_id_even_for_own_acts():
    k, conn, bus = make()
    await k.observe(ev("tool_done", payload={"tool": "search_web"}, actor="aiva", event_id=12))
    assert k.cursor == 12
    await k.observe(ev("tool_done", payload={"tool": "search_web"}, event_id=9))   # never goes back
    assert k.cursor == 12
    k.set_cursor("x")
    assert k.cursor == 12


def test_status_and_show():
    k, conn, bus = make()
    st = k.status()
    assert st["total"] == 0 and st["cursor"] == 0 and st["can_do"] == []
    assert "nothing yet" in k.show()


# --- tools.py: the per-service hook fires from _timed; kin voice tools ---------------------

class _FakeLLM:
    """Stands in for a pipecat LLM service: register_function is all the kin tools need."""
    def __init__(self):
        self.registered = {}

    def register_function(self, name, handler):
        self.registered[name] = handler


class _Params:
    def __init__(self, llm, arguments=None):
        self.llm = llm
        self.arguments = arguments or {}
        self.results = []

    async def result_callback(self, result):
        self.results.append(result)


async def test_bind_tool_hook_fires_from_timed_per_service():
    seen_a, seen_b = [], []
    llm_a, llm_b = _FakeLLM(), _FakeLLM()
    bind_tool_hook(llm_a, lambda name, ok: seen_a.append((name, ok)))
    bind_tool_hook(llm_b, lambda name, ok: seen_b.append((name, ok)))

    async def fine(params):
        await params.result_callback({"success": True})

    async def broken(params):
        raise RuntimeError("boom")

    await _timed("search_web", fine)(_Params(llm_a))
    with pytest.raises(RuntimeError):
        await _timed("read_page", broken)(_Params(llm_a))
    await _timed("feed_cat", fine)(_Params(llm_b))
    await _timed("feed_cat", fine)(_Params(None))             # no service, no hook, no error
    assert seen_a == [("search_web", True), ("read_page", False)]
    assert seen_b == [("feed_cat", True)]
    bind_tool_hook(llm_a, None)
    await _timed("search_web", fine)(_Params(llm_a))
    assert len(seen_a) == 2                                      # unbound


async def test_hook_errors_never_break_the_tool():
    llm = _FakeLLM()
    bind_tool_hook(llm, lambda name, ok: 1 / 0)

    async def fine(params):
        await params.result_callback({"success": True})
    p = _Params(llm)
    await _timed("search_web", fine)(p)
    assert p.results == [{"success": True}]


async def test_kin_tools_registered_and_speak_honestly():
    k, conn, bus = make()
    llm = _FakeLLM()
    handlers = register_kin_tools(llm, k)
    assert set(handlers) == {"recall_about_sister", "forget_about_sister"} == set(llm.registered)
    p = _Params(llm, {"query": "can she look things up"})
    await llm.registered["recall_about_sister"](p)
    assert p.results[0]["success"] and p.results[0]["items"] == []
    assert "haven't seen her" in p.results[0]["note"]
    await k.observe(ev("tool_done", "used search_web", payload={"tool": "search_web"}))
    p = _Params(llm, {"query": "can she look things up online"})
    await llm.registered["recall_about_sister"](p)
    assert p.results[0]["items"][0]["what"] == KIN_LINES["look_up_info"]
    assert p.results[0]["items"][0]["for_instance"] == "used search_web"
    p = _Params(llm, {"what": "look things up"})
    await llm.registered["forget_about_sister"](p)
    assert p.results[0]["success"] and p.results[0]["dropped"] == KIN_LINES["look_up_info"]
    p = _Params(llm, {"what": ""})
    await llm.registered["forget_about_sister"](p)
    assert p.results[0]["success"] is False


def test_kin_schemas_and_allow_lists():
    names = {fs.name for fs in TOOL_SCHEMAS.standard_tools}
    assert {"recall_about_sister", "forget_about_sister"} <= names
    assert "recall_about_sister" in RUNNER_TOOL_ALLOW and "forget_about_sister" not in RUNNER_TOOL_ALLOW
    assert "recall_about_sister" in cap.SAFE_TOOLS
    assert set(make_kin_handlers(types.SimpleNamespace())) == {"recall_about_sister", "forget_about_sister"}
