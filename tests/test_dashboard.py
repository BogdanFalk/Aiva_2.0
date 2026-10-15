"""Dashboard (offline): two tiny character dbs + a house.db in tmp_path, built through the real
modules (Memory, Becoming, HouseMemory, ServiceLimits); the dashboard's path resolution is
monkeypatched. Asserts the per-tab payloads — collect("aiva"), collect("saga") with a missing db
(→ "not started yet"), collect("conversations") (ordering, badges, day filter, paging) — the
Household strip on every tab, the API routing, and that the module imports without pipecat/Qt.

The Controls strip: `send_control` against a real `HouseServer` on port 0 (the house's
`on_control` gets the press, a control client is never a sister, nothing is ever `said`), against
a fake server that won't `welcome`, and with no house at all; `control_response` validation with
an injected sender; the `POST /api/control` route through a real `ThreadingHTTPServer` on port 0
with `send_control` faked (nothing here ever talks to a live house)."""
import asyncio
import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer

import pytest

import dashboard
from modules.becoming import Becoming
from modules.house_memory import HouseMemory
from modules.housedb import open_house_db
from modules.limits import ServiceLimits
from modules.memory import Memory

AIVA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aiva")

NOW = datetime.now().replace(microsecond=0)
PERIOD = NOW.strftime("%Y-%m")
TODAY = NOW.strftime("%Y-%m-%d")
YESTERDAY = (NOW - timedelta(days=1)).strftime("%Y-%m-%d")

# kin.py lands from another implementer; until then the dashboard only knows the agreed columns
KIN_DDL = """CREATE TABLE IF NOT EXISTS kin_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, about TEXT NOT NULL, kind TEXT NOT NULL, capability TEXT,
    summary TEXT NOT NULL, last_example TEXT, count INTEGER NOT NULL DEFAULT 1,
    strength REAL NOT NULL DEFAULT 0.3, first_ts TEXT NOT NULL, last_ts TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active', embedding BLOB)"""


# the traits table as it was BEFORE the temperament work (no `source` column) — an old install
LEGACY_TRAITS_DDL = """CREATE TABLE IF NOT EXISTS traits (
    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, kind TEXT NOT NULL,
    source_book_id INTEGER, source_books TEXT NOT NULL DEFAULT '[]', strength REAL NOT NULL DEFAULT 0.5,
    adopted_ts TEXT NOT NULL, announced INTEGER NOT NULL DEFAULT 0, announce_note_ts TEXT,
    status TEXT NOT NULL DEFAULT 'active', last_reinforced_ts TEXT NOT NULL, veto_reason TEXT,
    embedding BLOB)"""


def _has_column(conn, table, column):
    return any(r[1] == column for r in conn.execute(f"PRAGMA table_info({table})").fetchall())


def make_character_db(path, *, temperament=(), kin=(), legacy_traits=False):
    """A character core: Memory tables + becoming tables (the real Becoming, whose traits table
    carries `source` once the temperament work is in — or the LEGACY traits DDL without it when
    `legacy_traits`), optional kin_observations (`kin=None` leaves the table out), one reading
    trait, a message and a fact. Returns whether traits has a `source` column."""
    mem = Memory(str(path))
    c = mem.conn
    if legacy_traits:
        c.execute(LEGACY_TRAITS_DDL)
    else:
        Becoming(c)
    has_source = _has_column(c, "traits", "source")
    ts = NOW.isoformat()
    for text, strength in temperament:
        cols = "text, kind, strength, adopted_ts, last_reinforced_ts" + (", source" if has_source else "")
        vals = [text, "temperament", strength, ts, ts] + (["life"] if has_source else [])
        c.execute(f"INSERT INTO traits ({cols}) VALUES ({','.join('?' * len(vals))})", vals)
    c.execute("INSERT INTO traits (text, kind, strength, adopted_ts, last_reinforced_ts) "
              "VALUES (?,?,?,?,?)", ("I read slowly on purpose", "value", 0.6, ts, ts))
    if kin is not None:
        c.execute(KIN_DDL)
        for about, kind, cap, summary, count, strength, status in kin:
            c.execute("INSERT INTO kin_observations (about, kind, capability, summary, last_example, "
                      "count, strength, first_ts, last_ts, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (about, kind, cap, summary, "an example", count, strength, ts, ts, status))
    c.execute("INSERT INTO messages (session_id, role, content, ts) VALUES ('s1','user','hi',?)", (ts,))
    c.execute("INSERT INTO facts (category, fact, created_ts, last_confirmed_ts) "
              "VALUES ('other','likes tea',?,?)", (ts, ts))
    c.commit()
    mem.close()  # checkpoints the WAL → a plain file the read-only reader can open
    return has_source


def make_house(path):
    """house.db with a two-day transcript (audible + private + interrupted lines), one event, the
    sister-talk mode, the connected list and this month's spend counters."""
    conn = open_house_db(str(path))
    hm = HouseMemory(conn)
    ServiceLimits(conn, caps={})
    y = datetime.fromisoformat(f"{YESTERDAY}T20:00:00")
    t = datetime.fromisoformat(f"{TODAY}T09:00:00")
    hm.log_user_line(1, "Saga, what's the weather?", "saga", kind="named", now=y)
    hm.log_said(1, "saga", "Sunny, he says.", "him", audible=True, now=y + timedelta(seconds=3))
    hm.log_said(2, "saga", "Aiva, did you see that?", "aiva", audible=False, now=t)
    hm.log_said(2, "aiva", "I did.", "saga", audible=False, interrupted=True, now=t + timedelta(seconds=2))
    hm.log_user_line(3, "you two, quiet", "both", kind="group", now=t + timedelta(seconds=5))
    hm.add_event("aiva", "did", "Aiva looked something up online", capability="web_search",
                 relayed_to="saga", now=t)
    hm.kv_set("sister_talk", "private")
    hm.kv_set("present", json.dumps(["aiva"]))
    for svc, used in (("tts", 1500), ("tts.aiva", 1234), ("tts.saga", 266), ("tavily", 7)):
        conn.execute("INSERT INTO service_usage (service, period, used) VALUES (?,?,?)", (svc, PERIOD, used))
    conn.commit()
    conn.close()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Aiva started (db + body up), Saga not started (no db, no body), house up on :8761."""
    aiva_db, saga_db = tmp_path / "memory.db", tmp_path / "saga.db"
    make_character_db(aiva_db, legacy_traits=True, kin=[
        ("saga", "did", "web_search", "She can look things up online", 7, 0.8, "active"),
        ("saga", "said", None, "She said she likes rain", 1, 0.3, "active"),
        ("saga", "did", "tend_cat", "She fed the cat (vetoed)", 1, 0.9, "dropped"),
    ])
    make_house(tmp_path / "house.db")
    (tmp_path / "avatar_state.json").write_text(json.dumps(
        {"pid": os.getpid(), "port": 5000, "boot_id": "b1", "monitor": 0, "layer": "top"}), encoding="utf-8")
    (tmp_path / "house_state.json").write_text(json.dumps(
        {"pid": os.getpid(), "port": 8761, "boot_id": "h1", "protocol": 1, "ts": NOW.isoformat()}), encoding="utf-8")
    (tmp_path / "inner_state.json").write_text(json.dumps(
        {"valence": 0.2, "arousal": 0.5, "social": 0.1, "focus": None, "last_update": NOW.isoformat()}), encoding="utf-8")
    chars = {
        "aiva": {"slug": "aiva", "name": "Aiva", "self_forming": False, "db": str(aiva_db),
                 "inner": str(tmp_path / "inner_state.json"), "cat": str(tmp_path / "cat_state.json"),
                 "avatar": str(tmp_path / "avatar_state.json")},
        "saga": {"slug": "saga", "name": "Saga", "self_forming": True, "db": str(saga_db),
                 "inner": str(tmp_path / "saga_inner_state.json"), "cat": None,
                 "avatar": str(tmp_path / "saga_avatar_state.json")},
    }
    monkeypatch.setattr(dashboard, "CHARACTERS", chars)
    monkeypatch.setattr(dashboard, "HOUSE_DB", str(tmp_path / "house.db"))
    monkeypatch.setattr(dashboard, "HOUSE_STATE", str(tmp_path / "house_state.json"))
    monkeypatch.setattr(dashboard, "_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(dashboard, "ENV_PATH", str(tmp_path / ".env"))   # never the real .env
    monkeypatch.setenv("AIVA_TTS_MONTHLY_CHARS", "100000")
    monkeypatch.delenv("AIVA_TAVILY_MONTHLY_CAP", raising=False)
    # the house looks up (our pid, :8761) but nobody listens there — skip the live-state peek so a
    # firewall that drops loopback SYNs can't add a timeout per collect (the peek itself is
    # exercised against a real HouseServer in test_send_control_reaches_the_house...)
    monkeypatch.setattr(dashboard, "_peek_house", lambda port, timeout=None: None)
    return tmp_path


# --- import & wiring ------------------------------------------------------------------

def test_module_imports_without_pipecat_or_qt():
    code = (f"import sys; sys.path.insert(0, {AIVA_DIR!r}); import dashboard; "
            "print(','.join(m for m in ('pipecat','PyQt5','PySide6','numpy','openai','live2d') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_characters_map_comes_from_character_module():
    chars = dashboard._build_characters()
    assert list(chars) == ["aiva", "saga"]
    assert chars["aiva"]["name"] == "Aiva" and chars["aiva"]["db"].endswith("memory.db")
    assert chars["aiva"]["cat"] and chars["aiva"]["cat"].endswith("cat_state.json")
    assert chars["saga"]["name"] == "Saga" and chars["saga"]["db"].endswith("saga.db")
    assert chars["saga"]["cat"] is None and chars["saga"]["self_forming"] is True
    assert chars["saga"]["avatar"].endswith("saga_avatar_state.json")
    assert dashboard.HOUSE_DB.endswith("house.db") and dashboard.HOUSE_STATE.endswith("house_state.json")
    assert [t["who"] for t in dashboard.tabs()][-3:] == ["conversations", "tasks", "settings"]


def test_tool_groups_tolerate_sister_tools_not_in_tools_py_yet():
    sisters = dict(dashboard.TOOL_GROUPS)["Sisters & the house"]
    assert set(sisters) == {"set_sister_talk", "recall_about_sister", "forget_about_sister"}
    caps = dashboard._collect_capabilities()
    labels = [g["label"] for g in caps["tool_groups"]]
    assert "Presence & voice" in labels
    for g in caps["tool_groups"]:
        if g["label"] == "Sisters & the house":
            assert set(g["tools"]) <= set(sisters)   # only what tools.py actually defines
    assert caps["tool_count"] > 30


# --- a character tab ------------------------------------------------------------------

def test_collect_aiva_started_with_kin_and_house_limits(home):
    d = dashboard.collect("aiva")
    assert (d["who"], d["name"], d["started"], d["db_present"]) == ("aiva", "Aiva", True, True)
    assert "error" not in d and "note" not in d
    assert [t["who"] for t in d["tabs"]] == ["aiva", "saga", "conversations", "tasks", "settings"]
    assert d["has_cat"] is True and d["cat"] is None           # a cat path, no state file yet
    assert d["inner"]["mood"] is not None and "valence" in d["inner"]["mood"]

    kin = d["kin"]
    assert kin["available"] and kin["total"] == 2 and kin["dropped"] == 1
    assert [k["summary"] for k in kin["items"]] == ["She can look things up online", "She said she likes rain"]
    top = kin["items"][0]
    assert (top["about"], top["about_name"], top["kind"], top["capability"]) == ("saga", "Saga", "did", "web_search")
    assert top["count"] == 7 and top["strength"] == 0.8 and top["last"] is not None
    assert kin["by_kind"] == {"did": 1, "said": 1}

    tp = d["temperament"]                          # legacy traits table: no rows, no `source` column
    assert tp["available"] and tp["traits"] == [] and tp["core_count"] == 0 and tp["forming"] is True
    assert tp["core_mark"] == 0.5
    # the reading panel keeps reading traits and never repeats temperament
    assert [t["text"] for t in d["becoming"]["traits"]] == ["I read slowly on purpose"]
    h = d["headline"]
    assert h["traits"] == 1 and h["temperament_core"] == 0 and h["facts"] == 1 and h["messages"] == 1

    lim = d["limits"]
    assert lim["source"] == "house"
    by = {s["service"]: s for s in lim["services"]}
    assert by["tts"]["used"] == 1500 and by["tts"]["cap"] == 100000 and by["tts"]["pct"] == 1
    assert by["tavily"]["used"] == 7 and by["tavily"]["cap"] == 1000 and by["tavily"]["paused"] is False
    assert lim["paused"] == [] and lim["book_cost"] == 0 and lim["task_cost"] == 0
    # presence reads HER avatar state file
    assert d["presence"]["avatar_running"] is True and d["presence"]["avatar_pid"] == os.getpid()
    json.dumps(d, default=str)


def test_collect_saga_not_started_yet(home):
    d = dashboard.collect("saga")
    assert (d["who"], d["name"], d["self_forming"]) == ("saga", "Saga", True)
    assert d["started"] is False and d["db_present"] is False and "error" not in d
    assert "hasn't started yet" in d["note"] and "saga.db" in d["note"]
    assert d["headline"] == {} and "kin" not in d and "temperament" not in d
    assert d["has_cat"] is False and d["cat"] is None
    assert d["presence"]["state"] == "offline" and d["presence"]["avatar_running"] is False
    assert d["inner"]["mood"] is None
    # the shared bits are still there on a not-started tab
    assert d["household"]["house"]["up"] is True
    assert d["limits"]["source"] == "house" and d["limits"]["book_cost"] == 0


def test_collect_saga_temperament_with_core_mark_and_forming_note(home):
    saga_db = home / "saga.db"
    has_source = make_character_db(saga_db, kin=None, temperament=[
        ("I say what I think", 0.7), ("I go quiet when unsure", 0.4), ("I get curious fast", 0.55)])
    d = dashboard.collect("saga")
    assert d["started"] is True
    tp = d["temperament"]
    assert [t["text"] for t in tp["traits"]] == ["I say what I think", "I get curious fast", "I go quiet when unsure"]
    assert [t["core"] for t in tp["traits"]] == [True, True, False]
    assert tp["core_count"] == 2 and tp["forming"] is True
    # `source` rides along once becoming's schema has the column, and is simply None before
    assert tp["traits"][0]["source"] == ("life" if has_source else None)
    assert d["headline"]["temperament_core"] == 2 and d["headline"]["traits"] == 1
    # no kin table yet → tolerated, marked unavailable
    assert d["kin"] == {"available": False, "items": [], "total": 0, "dropped": 0, "by_kind": {}}
    # the third trait crosses the core mark → she has formed
    w = sqlite3.connect(str(saga_db))
    w.execute("UPDATE traits SET strength=0.5 WHERE text='I go quiet when unsure'")
    w.commit()
    w.close()
    tp = dashboard.collect("saga")["temperament"]
    assert tp["core_count"] == 3 and tp["forming"] is False


def test_collect_unknown_character_raises(home):
    with pytest.raises(KeyError):
        dashboard.collect("nobody")


# --- the conversations tab -----------------------------------------------------------------

def test_conversations_newest_first_with_badges(home):
    d = dashboard.collect("conversations")
    assert (d["who"], d["name"]) == ("conversations", "Conversations")
    conv = d["conversations"]
    assert conv["available"] is True and conv["total"] == 5 and conv["has_more"] is False
    assert conv["next_before"] is None and conv["day"] is None and conv["limit"] == 200
    lines = conv["lines"]
    ids = [ln["id"] for ln in lines]
    assert len(lines) == 5 and ids == sorted(ids, reverse=True)
    newest = lines[0]
    assert (newest["speaker"], newest["speaker_name"]) == ("user", "You")
    assert (newest["addressee"], newest["addressee_name"]) == ("both", "both")
    assert newest["kind"] == "group" and newest["badge"] == "audible" and newest["audible"] is True
    assert newest["day"] == TODAY and newest["time"] == "09:00:05"
    aiva = next(ln for ln in lines if ln["speaker"] == "aiva")
    assert aiva["speaker_name"] == "Aiva" and aiva["addressee_name"] == "Saga"
    assert aiva["badge"] == "private" and aiva["audible"] is False and aiva["interrupted"] is True
    spoken = next(ln for ln in lines if ln["speaker"] == "saga" and ln["addressee"] == "him")
    assert spoken["addressee_name"] == "you" and spoken["badge"] == "audible" and spoken["interrupted"] is False
    assert conv["days"] == [TODAY, YESTERDAY]
    ev = d["house_events"]
    assert len(ev) == 1 and ev[0]["name"] == "Aiva" and ev[0]["kind"] == "did"
    assert ev[0]["capability"] == "web_search" and ev[0]["relayed_to"] == "saga" and ev[0]["time"] == "09:00:00"


def test_conversations_day_filter_and_paging(home):
    dy = dashboard.collect("conversations", day=YESTERDAY)["conversations"]
    assert dy["day"] == YESTERDAY and dy["total"] == 2 and len(dy["lines"]) == 2
    assert all(ln["day"] == YESTERDAY for ln in dy["lines"])
    assert dy["lines"][0]["speaker"] == "saga" and dy["lines"][1]["speaker"] == "user"
    assert dashboard.collect("conversations", day="nope")["conversations"]["day"] is None
    assert dashboard.collect("conversations", day=f"{TODAY}T09")["conversations"]["day"] is None

    everything = [ln["id"] for ln in dashboard.collect("conversations")["conversations"]["lines"]]
    p1 = dashboard.collect("conversations", limit=2)["conversations"]
    assert [ln["id"] for ln in p1["lines"]] == everything[:2]
    assert p1["has_more"] is True and p1["next_before"] == everything[1]
    p2 = dashboard.collect("conversations", limit=2, before=p1["next_before"])["conversations"]
    assert [ln["id"] for ln in p2["lines"]] == everything[2:4] and p2["has_more"] is True
    p3 = dashboard.collect("conversations", limit=2, before=p2["next_before"])["conversations"]
    assert [ln["id"] for ln in p3["lines"]] == everything[4:] and p3["has_more"] is False
    assert p3["next_before"] is None and p3["before"] == p2["next_before"]
    # garbage paging/limit values are ignored, never raise
    loose = dashboard.collect("conversations", before="x", limit="lots")["conversations"]
    assert loose["before"] is None and loose["limit"] == 200 and len(loose["lines"]) == 5


# --- the household strip ----------------------------------------------------------------

def test_household_strip_on_every_tab(home):
    hh = dashboard.collect("aiva")["household"]
    bodies = {b["slug"]: b for b in hh["bodies"]}
    assert list(bodies) == ["aiva", "saga"]
    assert bodies["aiva"]["up"] is True and bodies["aiva"]["pid"] == os.getpid()
    assert bodies["aiva"]["name"] == "Aiva" and bodies["aiva"]["monitor"] == 0
    assert bodies["saga"]["up"] is False and bodies["saga"]["pid"] is None
    house = hh["house"]
    assert house["up"] is True and house["port"] == 8761 and house["pid"] == os.getpid()
    assert house["boot_id"] == "h1" and house["state_present"] is True and house["since"] is not None
    assert hh["present"] == ["aiva"] and hh["present_names"] == ["Aiva"]
    assert hh["sister_talk"] == "private"
    assert hh["tts"] == {"period": PERIOD, "total": 1500, "per": {"aiva": 1234, "saga": 266}, "cap": 100000}
    assert hh["house_db_present"] is True
    # identical strip on the other tabs
    assert dashboard.collect("saga")["household"]["tts"] == hh["tts"]
    assert dashboard.collect("conversations")["household"]["sister_talk"] == "private"


def test_household_when_house_is_down_or_missing(home, monkeypatch):
    (home / "house_state.json").write_text(json.dumps(
        {"pid": 999999999, "port": 8761, "boot_id": "h2", "protocol": 1}), encoding="utf-8")
    hh = dashboard.collect("aiva")["household"]
    assert hh["house"]["up"] is False and hh["house"]["state_present"] is True
    assert hh["present"] == [] and hh["present_names"] == []      # a dead house has nobody connected
    assert hh["sister_talk"] == "private" and hh["tts"]["total"] == 1500   # the db is still there
    # no state file at all
    (home / "house_state.json").unlink()
    hh = dashboard.collect("saga")["household"]
    assert hh["house"] == {"up": False, "pid": None, "port": None, "boot_id": None, "protocol": None,
                           "since": None, "state_present": False, "state": None}
    # no house.db at all: limits fall back to her own db, conversations report unavailable
    monkeypatch.setattr(dashboard, "HOUSE_DB", str(home / "nope.db"))
    d = dashboard.collect("aiva")
    assert d["limits"]["source"] == "character"
    assert next(s for s in d["limits"]["services"] if s["service"] == "tts")["used"] == 0
    hh = d["household"]
    assert hh["house_db_present"] is False and hh["sister_talk"] is None
    assert hh["tts"] == {"period": PERIOD, "total": 0, "per": {"aiva": 0, "saga": 0}, "cap": 100000}
    c = dashboard.collect("conversations")
    assert c["house_db_present"] is False
    assert c["conversations"]["available"] is False and c["conversations"]["lines"] == []
    assert c["house_events"] == []
    assert dashboard.collect("saga")["limits"] is None          # no db of her own either


# --- routes, page, CLI ------------------------------------------------------------------------

def test_api_response_routes(home):
    code, payload = dashboard.api_response("")
    assert code == 200 and payload["who"] == "aiva"
    code, payload = dashboard.api_response("who=saga")
    assert code == 200 and payload["who"] == "saga" and payload["started"] is False
    code, payload = dashboard.api_response("who=Conversations")
    assert code == 200 and payload["who"] == "conversations"
    code, payload = dashboard.api_response(f"who=conversations&day={TODAY}&limit=2")
    assert payload["conversations"]["day"] == TODAY and len(payload["conversations"]["lines"]) == 2
    assert payload["conversations"]["has_more"] is True
    code, payload = dashboard.api_response("who=bogus")
    assert code == 404 and "unknown who" in payload["error"]
    assert payload["known"] == ["aiva", "saga", "conversations", "tasks", "settings"]
    json.dumps(payload, default=str)


def test_page_carries_tabs_and_household_strip(home):
    html = dashboard.page()
    assert 'id="tabs"' in html and 'id="household"' in html and "location.hash" in html
    assert '"who": "conversations"' in html and '"label": "Saga"' in html and '"label": "Aiva"' in html
    assert "/api/data?who=" in html
    assert "__POLL__" not in html and "__TABS__" not in html and "__TEMP_CORE__" not in html


def test_main_once_prints_one_snapshot(home, capsys):
    assert dashboard.main(["--once", "--who", "conversations"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["who"] == "conversations" and out["conversations"]["total"] == 5
    assert out["household"]["sister_talk"] == "private"
    assert dashboard.main(["--once", "--who", "saga"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["who"] == "saga" and out["started"] is False
    assert dashboard.main(["--once"]) == 0
    assert json.loads(capsys.readouterr().out)["who"] == "aiva"
    assert dashboard.main(["--once", "--who", "nobody"]) == 2
    assert "unknown --who" in capsys.readouterr().err


# --- the controls strip -----------------------------------------------------------------------

async def _until(pred, timeout=4.0, step=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        await asyncio.sleep(step)
    return pred()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def test_send_control_reaches_the_house_as_a_control_client(tmp_path):
    """A real HouseServer on port 0: the press lands in `on_control`, the dashboard is a control
    client (no presence, no floor, nothing said), and the state peek uses the same handshake."""
    from modules import house_proto as proto
    from modules.house_floor import Floor
    from modules.house_server import HouseServer
    from modules.sisters import ExchangeConfig
    state_path = str(tmp_path / "house_state.json")
    memory = HouseMemory(sqlite3.connect(":memory:"))
    floor = Floor(memory, private=True, log=lambda m: None, cfg=ExchangeConfig(quiet_hours=(0, 0)))
    server = HouseServer(floor, memory, port=0, state_path=state_path, log=lambda m: None,
                         mic_state=lambda: {"awake": True, "mic_muted": True}, tick_s=0.02, ping_s=60)
    got, seen, said = [], [], []
    server.on_control = lambda action, target: got.append((action, target))
    real_dispatch = server._dispatch

    async def spy(session, msg):
        seen.append((session.slug, type(msg)))
        await real_dispatch(session, msg)
    server._dispatch = spy
    floor.on_said = lambda *a, **k: said.append(a)
    await server.start()
    try:
        # send_control blocks on a socket; run it off the loop so the server can answer
        r = await asyncio.to_thread(dashboard.send_control, "mute", state_path=state_path)
        assert r == {"ok": True, "action": "mute", "target": None}
        r = await asyncio.to_thread(dashboard.send_control, "sister_talk", "audible", state_path=state_path)
        assert r == {"ok": True, "action": "sister_talk", "target": "audible"}
        assert await _until(lambda: got == [("mute", None), ("sister_talk", "audible")])
        # a control client is a button, never a sister: no presence, no floor, nothing said
        assert floor.present == [] and server.sessions == {} and said == []
        assert memory.recent_transcript(limit=5) == []
        assert await _until(lambda: not server.controls)          # the bye closed each session
        assert {t for _, t in seen} == {proto.Control, proto.Bye}   # never a Said (or anything else)
        assert all(slug == "control:dashboard" for slug, _ in seen)
        # the Household strip's labels come from the same hello → welcome → bye, pressing nothing
        peek = await asyncio.to_thread(dashboard._peek_house, server.port)
        assert peek == {"awake": True, "mic_muted": True, "mode": "private", "present": []}
        assert await _until(lambda: not server.controls)
        assert got == [("mute", None), ("sister_talk", "audible")]
        assert {t for _, t in seen} == {proto.Control, proto.Bye}
    finally:
        await server.stop()
    # the house is gone (it cleared its state file) → nobody to send to
    assert dashboard.send_control("mute", state_path=state_path) == {"ok": False, "error": "the house isn't running"}


async def test_send_say_reaches_the_house_as_a_typed_line(tmp_path):
    """The Conversations chat box against a real HouseServer: the line lands in `on_say` (which the
    house hands to the Floor), the sender stays a control client, and an empty line never leaves."""
    from modules import house_proto as proto
    from modules.house_floor import Floor
    from modules.house_server import HouseServer
    from modules.sisters import ExchangeConfig
    state_path = str(tmp_path / "house_state.json")
    memory = HouseMemory(sqlite3.connect(":memory:"))
    floor = Floor(memory, private=True, log=lambda m: None, cfg=ExchangeConfig(quiet_hours=(0, 0)))
    server = HouseServer(floor, memory, port=0, state_path=state_path, log=lambda m: None,
                         mic_state=lambda: {"awake": False, "mic_muted": True}, tick_s=0.02, ping_s=60)
    typed, seen = [], []
    server.on_say = lambda text: typed.append(text)
    real_dispatch = server._dispatch

    async def spy(session, msg):
        seen.append((session.slug, type(msg)))
        await real_dispatch(session, msg)
    server._dispatch = spy
    await server.start()
    try:
        r = await asyncio.to_thread(dashboard.send_say, "  Saga, come here  ", state_path=state_path)
        assert r == {"ok": True, "text": "Saga, come here"}
        assert await _until(lambda: typed == ["Saga, come here"])
        assert floor.present == [] and server.sessions == {}      # typing is not a sister connecting
        assert {t for _, t in seen} == {proto.Say, proto.Bye}
        assert all(slug == "control:dashboard" for slug, _ in seen)
        # an empty line never reaches the socket at all
        assert dashboard.send_say("   ", state_path=state_path) == {"ok": False, "error": "nothing to say"}
        assert typed == ["Saga, come here"]
    finally:
        await server.stop()
    assert dashboard.send_say("hello", state_path=state_path) == {"ok": False, "error": "the house isn't running"}


async def test_send_control_insists_on_a_welcome(tmp_path, monkeypatch):
    """A server that answers the hello with anything but `welcome` → ok False; the hello itself is
    the control-role one; no house (closed port, no/stale state file, no house_proto) → ok False."""
    from modules import house_proto as proto
    reply, hellos = {"line": proto.encode(proto.Reject(reason="protocol 9 != 1"))}, []

    async def handle(reader, writer):
        hellos.append(proto.decode(await reader.readline()))
        writer.write(reply["line"])
        await writer.drain()
        try:
            await asyncio.wait_for(reader.read(), 2.0)        # until the dashboard hangs up
        except asyncio.TimeoutError:
            pass
        writer.close()
    srv = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    try:
        r = await asyncio.to_thread(dashboard.send_control, "mute", port=port)
        assert r["ok"] is False and "protocol 9" in r["error"]
        h = hellos[0]
        assert isinstance(h, proto.Hello)
        assert (h.slug, h.name, h.role, h.has_voice) == ("dashboard", "Dashboard", "control", False)
        assert h.pid == os.getpid() and h.version == str(dashboard.AIVA_VERSION)
        reply["line"] = proto.encode(proto.Ping())
        r = await asyncio.to_thread(dashboard.send_control, "mute", port=port)
        assert r["ok"] is False and "expected welcome" in r["error"]
        reply["line"] = b"not a house message\n"
        r = await asyncio.to_thread(dashboard.send_control, "mute", port=port)
        assert r["ok"] is False and r["error"]
        assert len(hellos) == 3 and all(x.role == "control" for x in hellos)
        assert await asyncio.to_thread(dashboard._peek_house, port) is None   # the peek swallows it
    finally:
        srv.close()
        await srv.wait_closed()
    down = {"ok": False, "error": "the house isn't running"}
    # a just-closed asyncio port on Windows may still be accepted by the kernel (a pending AcceptEx)
    # and then never answered — that is "no welcome", not "isn't running"; either way ok False
    r = dashboard.send_control("wake", port=port, timeout=0.5)
    assert r["ok"] is False and ("isn't running" in r["error"] or "no welcome" in r["error"])
    # nobody accepting on a port (refused, or a dropped SYN behind a firewall) = not running
    assert dashboard.send_control("wake", port=_free_port(), timeout=0.5) == down
    assert dashboard.send_control("wake", state_path=str(tmp_path / "nope.json")) == down
    # a stale state file (its pid is gone) is decided from the pid — no socket, no timeout
    (tmp_path / "stale.json").write_text(json.dumps({"pid": 999999999, "port": _free_port()}), encoding="utf-8")
    t0 = time.time()
    assert dashboard.send_control("wake", state_path=str(tmp_path / "stale.json")) == down
    assert time.time() - t0 < 0.4
    assert dashboard._house_port(str(tmp_path / "stale.json")) is None
    (tmp_path / "live.json").write_text(json.dumps({"pid": os.getpid(), "port": 4321}), encoding="utf-8")
    assert dashboard._house_port(str(tmp_path / "live.json")) == 4321
    monkeypatch.setattr(dashboard, "HOUSE_STATE", str(tmp_path / "nope.json"))
    assert dashboard.send_control("wake") == down                             # the module default
    # an older checkout without house_proto degrades to an error, never an exception
    monkeypatch.setattr(dashboard, "_proto", lambda: None)
    assert "house_proto" in dashboard.send_control("wake", port=port)["error"]
    assert dashboard._peek_house(port) is None


def test_control_response_validates_before_it_sends(home):
    sent = []

    def fake_send(action, target=None):
        sent.append((action, target))
        return {"ok": True, "action": action, "target": target}
    bad = [b"", b"nope", b"[1,2]", b'"mute"', {"action": "dance"}, {"action": "mute", "target": "aiva"},
           {"action": "quit", "target": "saga"}, {"action": "restart_sister"},
           {"action": "start_sister", "target": "audible"}, {"action": "quit_sister", "target": "nobody"},
           {"action": "sister_talk", "target": "saga"}, {"target": "aiva"}, {"action": "restart_house", "target": "aiva"}]
    for body in bad:
        code, payload = dashboard.control_response(body, send=fake_send)
        assert code == 400 and payload["ok"] is False and payload["error"], body
    assert sent == []                                       # nothing invalid ever reaches the house
    assert dashboard.control_response({"action": "dance"}, send=fake_send)[1]["actions"] == list(dashboard.CONTROL_ACTIONS)
    assert dashboard.control_response({"action": "mute", "target": "x"}, send=fake_send)[1]["targets"] == [
        "aiva", "saga", "audible", "private"]
    assert set(dashboard.CONTROL_ACTIONS) == {"mute", "interrupt", "sister_talk", "sleep", "wake", "restart_sisters",
                                              "restart_sister", "quit_sister", "start_sister",
                                              "mute_sister", "unmute_sister", "start_house",
                                              "restart_house", "quit"}
    # per-sister voice mute needs a real sister slug and goes to the house (not launched locally)
    assert dashboard.control_response({"action": "mute_sister"}, send=fake_send)[0] == 400
    assert dashboard.control_response({"action": "mute_sister", "target": "nobody"}, send=fake_send)[0] == 400

    good = [(b'{"action": "mute"}', ("mute", None)),
            ({"action": "mute", "target": None}, ("mute", None)),
            ({"action": "interrupt", "target": ""}, ("interrupt", None)),
            ({"action": "Sister_Talk", "target": "Audible"}, ("sister_talk", "audible")),
            ({"action": "sister_talk", "target": "private"}, ("sister_talk", "private")),
            ({"action": "sister_talk"}, ("sister_talk", None)),
            ({"action": "sleep"}, ("sleep", None)), ({"action": "wake"}, ("wake", None)),
            ('{"action": "start_sister", "target": "saga"}', ("start_sister", "saga")),
            ({"action": "restart_sister", "target": "aiva"}, ("restart_sister", "aiva")),
            ({"action": "quit_sister", "target": "Saga"}, ("quit_sister", "saga")),
            ({"action": "mute_sister", "target": "aiva"}, ("mute_sister", "aiva")),
            ({"action": "unmute_sister", "target": "Saga"}, ("unmute_sister", "saga")),
            ({"action": "restart_sisters"}, ("restart_sisters", None)),
            ({"action": "restart_house"}, ("restart_house", None)),
            ({"action": "quit"}, ("quit", None))]
    fake_launch = lambda name: {"ok": True}                 # start_* launch locally, never via the house
    for body, expect in good:
        code, payload = dashboard.control_response(body, send=fake_send, launch=fake_launch)
        assert (code, payload) == (200, {"ok": True, "action": expect[0], "target": expect[1]}), body
    assert sent == [e for _, e in good if e[0] != "start_sister"]
    # the house not taking it → 503 with its words; a broken sender → 503, never a traceback
    code, payload = dashboard.control_response(
        {"action": "wake"}, send=lambda a, t: {"ok": False, "error": "the house isn't running"})
    assert (code, payload) == (503, {"ok": False, "error": "the house isn't running", "action": "wake", "target": None})

    def boom(a, t):
        raise RuntimeError("socket exploded")
    code, payload = dashboard.control_response({"action": "wake"}, send=boom)
    assert code == 503 and "socket exploded" in payload["error"]
    code, payload = dashboard.control_response({"action": "wake"}, send=lambda a, t: "yes")
    assert code == 503 and payload["ok"] is False


def test_say_response_validates_then_sends(home):
    """The Conversations chat box: one trimmed line, size-capped, never raises."""
    sent = []

    def fake_send(text):
        sent.append(text)
        return {"ok": True, "text": text}
    for body in [b"", b"nope", b"[1,2]", {"text": ""}, {"text": "   "}, {},
                 {"text": "x" * (dashboard.MAX_SAY_CHARS + 1)}]:
        code, payload = dashboard.say_response(body, send=fake_send)
        assert code == 400 and payload["ok"] is False and payload["error"], body
    assert not sent
    code, payload = dashboard.say_response(b'{"text": "  Aiva,   move   down \\n"}', send=fake_send)
    assert code == 200 and payload["ok"] is True
    assert sent == ["Aiva, move down"]                       # whitespace collapsed, not re-spoken raw

    def boom(text):
        raise RuntimeError("socket exploded")
    code, payload = dashboard.say_response({"text": "hi"}, send=boom)
    assert code == 503 and "socket exploded" in payload["error"]
    code, payload = dashboard.say_response({"text": "hi"}, send=lambda t: "yes")
    assert code == 503 and payload["ok"] is False


def test_http_control_route_and_get_untouched(home, monkeypatch):
    """POST /api/control through the real handler (server on port 0, send_control faked); any other
    POST path is 404; GET routes are unchanged; off-machine presses are refused."""
    calls = []

    def fake_send(action, target=None, **kw):
        calls.append((action, target))
        return {"ok": True, "action": action, "target": target}
    monkeypatch.setattr(dashboard, "send_control", fake_send)
    typed = []
    monkeypatch.setattr(dashboard, "send_say", lambda text, **kw: typed.append(text) or {"ok": True, "text": text})
    launched = []
    monkeypatch.setattr(dashboard, "launch_bat",
                        lambda name: launched.append(name) or {"ok": True, "launched": f"start_{name}.bat"})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, data=None, method=None):
        req = urllib.request.Request(base + path, data=data, method=method or ("POST" if data is not None else "GET"),
                                     headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
    try:
        code, body = call("/api/control", b'{"action":"start_sister","target":"saga"}')
        assert (code, json.loads(body)) == (200, {"ok": True, "launched": "start_saga.bat",
                                                  "action": "start_sister", "target": "saga"})
        assert launched == ["saga"]                                       # local launch, not the house
        code, body = call("/api/control", b'{"action":"quit","target":"saga"}')
        assert code == 400 and json.loads(body)["ok"] is False
        code, body = call("/api/control", b'{"action":"mute"}')
        assert code == 200 and json.loads(body)["action"] == "mute"
        assert call("/api/control", b"")[0] == 400                      # empty body
        code, body = call("/api/say", b'{"text":"Saga, come here"}')
        assert code == 200 and json.loads(body)["ok"] is True and typed == ["Saga, come here"]
        assert call("/api/say", b'{"text":"  "}')[0] == 400             # nothing to say
        assert call("/api/data", b"{}")[0] == 404                       # POST knows only control and say
        assert call("/api/control")[0] == 404                           # and GET doesn't know it
        assert call("/api/say")[0] == 404
        assert calls == [("mute", None)]                                  # start_* never touch the house
        code, body = call("/api/data?who=saga")                          # GET is as it was
        assert code == 200 and json.loads(body)["who"] == "saga"
        assert call("/healthz") == (200, b"ok")
        code, body = call("/")
        assert code == 200 and b'id="controls"' in body
        monkeypatch.setattr(dashboard.Handler, "_local_client", lambda self: False)
        code, body = call("/api/control", b'{"action":"mute"}')
        assert code == 403 and json.loads(body)["ok"] is False
        assert call("/api/say", b'{"text":"hey"}')[0] == 403 and typed == ["Saga, come here"]
        assert calls == [("mute", None)]                                  # start_* never touch the house      # refused before any send
    finally:
        srv.shutdown()
        srv.server_close()


def test_page_carries_controls_strip(home):
    html = dashboard.page()
    assert 'id="controls"' in html and 'id="toast"' in html and "/api/control" in html
    for label in ("Interrupt", "Sleep", "Wake", "Restart house", "Quit everything", "make audible", "make private",
                  "Unmute mic", "Mute/unmute mic", "Start house", "Start ", "Restart ", "Quit "):
        assert label in html, label
    assert "confirm(" in html                                   # quit / quit sister / restart house ask first
    external = html.replace('<link rel="icon" href="/favicon.svg" type="image/svg+xml">', "")
    assert "<link" not in external and 'src="' not in external  # still one inline page; the icon is served by us
    # the household payload carries the live state slot the labels read (None while nobody answers)
    assert "state" in dashboard.collect("aiva")["household"]["house"]


def test_page_carries_the_chat_box(home):
    """One chat box, on the Conversations tab, posting to /api/say. It lives in its own persistent
    host (#sayhost), built once and never replaced by a poll, so focus/caret/draft survive typing."""
    html = dashboard.page()
    assert 'id="sayBox"' in html and 'id="sayBtn"' in html and "/api/say" in html
    assert f'maxlength="{dashboard.MAX_SAY_CHARS}"' in html and "__MAXSAY__" not in html
    assert 'id="sayhost"' in html                             # persistent host outside polled #content
    assert "function renderSayBox" in html and "function sendSay()" in html
    assert "wireSayBox" not in html                           # the old racy re-wire path is gone
    assert "e.key === 'Enter'" in html                        # Enter sends


def test_start_actions_launch_locally_and_never_need_the_house():
    import dashboard as d
    launched, sent = [], []
    fake_launch = lambda name: launched.append(name) or {"ok": True, "launched": f"start_{name}.bat"}
    fake_send = lambda action, target: sent.append((action, target)) or {"ok": True}
    st, res = d.control_response({"action": "start_sister", "target": "saga"}, send=fake_send, launch=fake_launch)
    assert st == 200 and res["ok"] and launched == ["saga"] and sent == []
    st, res = d.control_response({"action": "start_house"}, send=fake_send, launch=fake_launch)
    assert st == 200 and launched == ["saga", "house"] and sent == []
    st, res = d.control_response({"action": "start_house", "target": "saga"}, send=fake_send, launch=fake_launch)
    assert st == 400                                     # start_house takes no target
    st, res = d.control_response({"action": "mute"}, send=fake_send, launch=fake_launch)
    assert st == 200 and sent == [("mute", None)]        # the rest still goes to the house
    assert "start_house" in d.CONTROL_ACTIONS
    # launch_bat refuses unknown names, and refuses to start anything at all under pytest
    assert d.launch_bat("nope")["ok"] is False
    assert d.launch_bat("saga") == {"ok": False, "error": "launching is disabled here"}


def test_favicon_is_served_inline(home):
    import dashboard as d
    html = d.page()
    assert 'rel="icon" href="/favicon.svg"' in html
    assert d.FAVICON_SVG.startswith("<svg") and "circle" in d.FAVICON_SVG and len(d.FAVICON_SVG) < 2000



# --- the Settings tab ----------------------------------------------------------------------------

ENV_TEXT = ("# keys\nOPENAI_API_KEY=sk-secret-abcd1234\n\n# the voice model\nAIVA_LLM_MODEL=gpt-5.6-luna\n"
            "AIVA_WAKE_THRESHOLD=0.4\nUNRELATED=keep   # mine\n")


def test_settings_tab_comes_after_conversations(home):
    assert [t["who"] for t in dashboard.tabs()] == ["aiva", "saga", "conversations", "tasks", "settings"]
    assert dashboard.tabs()[-1]["label"] == "Settings"


def test_settings_payload_groups_values_and_masks_secrets(home):
    (home / ".env").write_text(ENV_TEXT, encoding="utf-8")
    code, d = dashboard.api_response("who=settings")
    assert code == 200 and d["who"] == "settings" and d["env_present"] is True
    assert d["env_path"] == str(home / ".env")
    assert "household" in d and [t["who"] for t in d["tabs"]][-1] == "settings"
    rows = {s["key"]: s for g in d["settings"] for s in g["settings"]}
    groups = [g["group"] for g in d["settings"]]
    assert groups[0] == "API keys & services" and "Tasks" in groups and "Saga" in groups
    assert rows["OPENAI_API_KEY"]["value"] == "••••1234" and rows["OPENAI_API_KEY"]["editable"] is False
    assert "sk-secret" not in json.dumps(d, ensure_ascii=False)
    w = rows["AIVA_WAKE_THRESHOLD"]
    assert (w["value"], w["default"], w["source"], w["modified"], w["type"]) == (0.4, 0.5, ".env", True, "float")
    assert w["restart"] == ["house"] and w["description"]
    assert rows["AIVA_TASK_MAX_WALL_MIN"]["value"] == 20.0 and rows["AIVA_TASK_MAX_WALL_MIN"]["source"] == "default"
    assert rows["AIVA_TASK_MODEL"]["value"] == "gpt-5.6-luna"            # falls back to the voice model
    assert rows["becoming.MAX_ACTIVE_TRAITS"]["readonly"] is True
    sm = d["settings_summary"]
    assert sm["total"] == len(rows) and sm["secrets_set"] == 1 and sm["in_env"] == 3
    json.dumps(d, default=str)
    assert dashboard.main(["--once", "--who", "settings"]) == 0


def test_settings_payload_without_an_env_file(home):
    d = dashboard.collect("settings")
    assert d["env_present"] is False and d["settings_summary"]["in_env"] == 0


def test_settings_response_validates_then_writes_the_tmp_env(home):
    env = home / ".env"
    env.write_text(ENV_TEXT, encoding="utf-8")
    before = env.read_bytes()
    for body in (b"", b"nope", b"[]", {"updates": []}, {"x": 1}):
        code, payload = dashboard.settings_response(body)
        assert code == 400 and payload["ok"] is False, body
    code, payload = dashboard.settings_response({"updates": {"OPENAI_API_KEY": "sk-evil", "AIVA_WAKE_THRESHOLD": 9}})
    assert code == 400 and set(payload["errors"]) == {"OPENAI_API_KEY", "AIVA_WAKE_THRESHOLD"}
    assert env.read_bytes() == before                                  # nothing written on any error
    code, payload = dashboard.settings_response(
        json.dumps({"updates": {"AIVA_WAKE_THRESHOLD": "0.45", "AIVA_SISTER_TALK": "audible",
                                "AIVA_TASK_MAX_COST_USD": 2, "AIVA_LLM_MODEL": None}}).encode())
    assert code == 200 and payload["ok"] is True
    assert payload["restart"] == ["house", "aiva", "saga"] and "restart" in payload["message"]
    text = env.read_text(encoding="utf-8")
    assert text.startswith("# keys\nOPENAI_API_KEY=sk-secret-abcd1234\n\n# the voice model\nAIVA_WAKE_THRESHOLD=0.45\n"
                           "UNRELATED=keep   # mine\n")
    assert "AIVA_LLM_MODEL" not in text and "AIVA_TASK_MAX_COST_USD=2" in text and "AIVA_SISTER_TALK=audible" in text
    assert (home / ".env.bak").read_bytes() == before
    rows = {s["key"]: s for g in dashboard.collect("settings")["settings"] for s in g["settings"]}
    assert rows["AIVA_SISTER_TALK"]["value"] == "audible" and rows["AIVA_LLM_MODEL"]["source"] == "default"


def test_http_settings_route_is_local_and_same_origin(home, monkeypatch):
    env = home / ".env"
    env.write_text(ENV_TEXT, encoding="utf-8")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(data, origin=None):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        req = urllib.request.Request(base + "/api/settings", data=data, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        code, body = post(b'{"updates":{"AIVA_SELF_TICK_S":"30"}}', origin=base)      # the page itself
        assert code == 200 and body["restart"] == ["aiva", "saga"]
        assert "AIVA_SELF_TICK_S=30" in env.read_text(encoding="utf-8")
        code, body = post(b'{"updates":{"AIVA_SELF_TICK_S":"45"}}')                  # a local script: no Origin
        assert code == 200
        code, body = post(b'{"updates":{"AIVA_SELF_TICK_S":"5"}}', origin="http://evil.example")
        assert code == 403 and "AIVA_SELF_TICK_S=45" in env.read_text(encoding="utf-8")
        code, body = post(b'{"updates":{"AIVA_SELF_TICK_S":"abc"}}', origin=base)
        assert code == 400 and "AIVA_SELF_TICK_S" in body["errors"]
        with urllib.request.urlopen(base + "/api/data?who=settings", timeout=5) as r:
            assert json.loads(r.read())["who"] == "settings"
        monkeypatch.setattr(dashboard.Handler, "_local_client", lambda self: False)
        code, body = post(b'{"updates":{"AIVA_SELF_TICK_S":"50"}}', origin=base)
        assert code == 403 and "AIVA_SELF_TICK_S=45" in env.read_text(encoding="utf-8")
    finally:
        srv.shutdown()
        srv.server_close()


def test_page_carries_the_settings_tab(home):
    html = dashboard.page()
    assert '"who": "settings"' in html and '"label": "Settings"' in html
    assert "/api/settings" in html and "function renderSettings" in html and "function saveSettings" in html
    assert "d.who==='settings'" in html and "restart_sister" in html and "beforeunload" in html


# --- the Tasks tab ---------------------------------------------------------------------------

T_CODE = "t-20260920-101500-abcd"      # delegated to Claude Code, newer (`claude` event)
T_OLD = "t-20260919-090000-0f0f"       # an old delegate: no `claude` event, legacy temp log only
T_SHELL = "t-20260918-080000-1234"     # a plain tool + shell task that failed
SESSION = "11111111-2222-4333-8444-555555555555"
OLD_SESSION = "99999999-8888-4777-8666-555555555555"
PROJECT = "C:/Work/Proj_1"


def _task_row(tid, created, goal, status, outcome, plan, log, **kw):
    row = {"id": tid, "created_ts": created, "updated_ts": created, "finished_ts": kw.pop("finished", None),
           "goal_text": goal, "goal_normalized": goal.lower(), "details": kw.pop("details", ""),
           "family": kw.pop("family", "repo-edit"), "approach": kw.pop("approach", None),
           "status": status, "outcome": outcome, "plan_json": json.dumps(plan), "log_json": json.dumps(log),
           "pending_json": kw.pop("pending", None), "result_summary": kw.pop("summary", None),
           "duration_s": kw.pop("duration", None), "delegated_to": kw.pop("delegated", "none"),
           "agent_grade": kw.pop("grade", None), "cost_usd": kw.pop("cost", 0.0),
           "llm_calls": 3, "tool_calls": 4}
    assert not kw
    return row


def _transcript_lines(prompt, cwd, *, edit="a.py", ts="2026-09-20T07:15:10.000Z"):
    return [
        {"type": "queue-operation", "operation": "enqueue", "timestamp": ts, "content": prompt},
        {"type": "user", "timestamp": ts, "cwd": cwd, "message": {"role": "user", "content": prompt}},
        {"type": "assistant", "timestamp": ts, "cwd": cwd, "message": {"id": "m1", "model": "claude-x", "content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": cwd + "\\src\\a.py"}},
            {"type": "tool_use", "name": "Read", "input": {"file_path": cwd + "\\src\\a.py"}},
            {"type": "tool_use", "name": "Grep", "input": {"pattern": "def main", "path": "src"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "is_error": True, "content": "x"}]}},
        {"type": "assistant", "message": {"id": "m2", "content": [
            {"type": "tool_use", "name": "Edit", "input": {"file_path": cwd + "\\src\\" + edit}},
            {"type": "tool_use", "name": "MultiEdit", "input": {"file_path": cwd + "\\src\\" + edit}},
            {"type": "tool_use", "name": "Write", "input": {"file_path": cwd + "\\tests\\test_a.py"}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}}]}},
        {"type": "assistant", "message": {"id": "m2", "content": [{"type": "text", "text": "Done: fixed it."}]}},
        {"type": "cost-state", "totalCostUSD": 1.25, "totalLinesAdded": 12, "totalLinesRemoved": 3,
         "totalDuration": 61000},
    ]


@pytest.fixture
def tasks_home(home, monkeypatch):
    """Aiva's db gets a tasks table (the real TaskMemory schema) with three tasks; a coding log +
    .json for the new job, a legacy temp log for the old one, and a fake ~/.claude/projects with
    both sessions' transcripts."""
    from modules.task_memory import TaskMemory
    mem = Memory(str(home / "memory.db"))
    TaskMemory(mem.conn)
    code_plan = [{"n": 1, "title": "Fix the bug", "kind": "delegate", "status": "done", "attempts": 1,
                  "action": {"project_dir": PROJECT, "prompt": "Fix the parser bug in src/a.py"},
                  "expect": "tests pass", "evidence": ["Claude Code done: fixed", "grade good: ok"],
                  "started": "2026-09-20T10:15:05", "finished": "2026-09-20T10:16:10"}]
    code_log = [
        {"ts": "2026-09-20T10:15:00", "event": "start", "detail": "Fix it"},
        {"ts": "2026-09-20T10:15:01", "event": "familiarity", "detail": "novel (lexical, 0.1)"},
        {"ts": "2026-09-20T10:15:02", "event": "decision", "detail": "confirm_plan — never done this"},
        {"ts": "2026-09-20T10:15:03", "event": "announce(question)", "detail": "[Task wants ...]"},
        {"ts": "2026-09-20T10:15:05", "event": "delegate", "detail": f"{T_CODE}-s1 in {PROJECT}"},
        {"ts": "2026-09-20T10:16:10", "event": "claude", "detail": json.dumps(
            {"job": f"{T_CODE}-s1", "session": SESSION, "cost": 1.25, "status": "done",
             "log": f"logs/coding/{T_CODE}-s1.log"})},
        {"ts": "2026-09-20T10:16:11", "event": "grade", "detail": "good: tests pass"},
        {"ts": "2026-09-20T10:16:12", "event": "finish", "detail": "success: fixed"},
    ]
    old_plan = [{"n": 1, "title": "Inspect", "kind": "delegate", "status": "done",
                 "action": {"project_dir": PROJECT, "prompt": "Inspect the  repo\nand report"}}]
    old_log = [{"ts": "2026-09-19T09:00:05", "event": "delegate", "detail": f"{T_OLD}-s1 in {PROJECT}"},
               {"ts": "2026-09-19T09:02:00", "event": "finish", "detail": "success: inspected"}]
    shell_plan = [
        {"n": 1, "title": "Write the file", "kind": "tool", "status": "done",
         "action": {"tool": "create_file", "args": {"path": "~/x.py", "content": "x" * 1000}}},
        {"n": 2, "title": "Run it", "kind": "shell", "status": "failed", "attempts": 2,
         "action": {"command": "python x.py", "purpose": "check"}, "failure": "rc=1"},
        {"n": 3, "title": "Report", "kind": "tool", "status": "pending", "action": {"tool": "remember", "args": {}}}]
    shell_log = [{"ts": "2026-09-18T08:00:01", "event": "tool", "detail": "create_file(...) -> ok"},
                 {"ts": "2026-09-18T08:00:02", "event": "error", "detail": "RuntimeError('boom')"}]
    rows = [
        _task_row(T_CODE, "2026-09-20T10:15:00.100000", "Fix the parser bug", "done", "success", code_plan,
                  code_log, delegated="claude_code", grade="good", duration=72, cost=0.004,
                  finished="2026-09-20T10:16:12", summary="Finished: fixed the parser."),
        _task_row(T_OLD, "2026-09-19T09:00:00.000000", "Inspect the repo", "done", "success", old_plan,
                  old_log, delegated="claude_code", finished="2026-09-19T09:02:00"),
        _task_row(T_SHELL, "2026-09-18T08:00:00.000000", "Make a prime script", "failed", "failed",
                  shell_plan, shell_log, cost=0.001, pending=json.dumps({"kind": "confirm_command",
                                                                          "text": "run python x.py?"})),
    ]
    for r in rows:
        cols = ", ".join(r)
        mem.conn.execute(f"INSERT INTO tasks ({cols}) VALUES ({','.join('?' * len(r))})", list(r.values()))
    mem.conn.commit()
    mem.close()

    logs = home / "logs"
    (logs / "coding").mkdir(parents=True)
    (logs / "coding" / f"{T_CODE}-s1.log").write_text("Fixed the parser. All tests pass.", encoding="utf-8")
    (logs / "coding" / f"{T_CODE}-s1.json").write_text(json.dumps(
        {"result": "Fixed the parser.", "session_id": SESSION, "total_cost_usd": 1.25, "num_turns": 5,
         "duration_ms": 61000}), encoding="utf-8")
    legacy = home / "aiva_terminals"
    legacy.mkdir()
    (legacy / f"{T_OLD}-s1.log").write_text("# Report\nNothing changed.", encoding="utf-8")
    monkeypatch.setattr(dashboard, "LEGACY_JOBS_DIR", str(legacy))

    projects = home / "claude_projects"
    cwd = "C:\\Work\\Proj_1"
    folder = projects / "C--Work-Proj-1"
    folder.mkdir(parents=True)
    with open(folder / f"{SESSION}.jsonl", "w", encoding="utf-8") as f:
        for e in _transcript_lines("Fix the parser bug in src/a.py", cwd):
            f.write(json.dumps(e) + "\n")
        f.write("not json at all\n")
    # the old run's transcript: found only by prompt (whitespace-normalised) + time window
    old_start = datetime.fromisoformat("2026-09-19T09:00:06").astimezone()
    with open(folder / f"{OLD_SESSION}.jsonl", "w", encoding="utf-8") as f:
        for e in _transcript_lines("Inspect the repo and report", cwd, edit="b.py",
                                   ts=old_start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")):
            f.write(json.dumps(e) + "\n")
    t = datetime.fromisoformat("2026-09-19T09:01:00").timestamp()
    os.utime(folder / f"{OLD_SESSION}.jsonl", (t, t))
    # a decoy in the same folder: other prompt
    (folder / "00000000-0000-4000-8000-000000000000.jsonl").write_text(json.dumps(
        {"type": "queue-operation", "operation": "enqueue", "content": "something else"}) + "\n", encoding="utf-8")
    os.utime(folder / "00000000-0000-4000-8000-000000000000.jsonl", (t, t))
    monkeypatch.setattr(dashboard, "CLAUDE_PROJECTS_DIR", str(projects))
    return home


def test_tasks_tab_sits_between_conversations_and_settings(home):
    whos = [t["who"] for t in dashboard.tabs()]
    assert whos.index("tasks") == whos.index("conversations") + 1 == whos.index("settings") - 1


def test_tasks_list_merges_newest_first_with_tally(tasks_home):
    d = dashboard.collect("tasks")
    assert d["who"] == "tasks" and "household" in d
    tk = d["tasks"]
    assert [r["id"] for r in tk["rows"]] == [T_CODE, T_OLD, T_SHELL]
    top = tk["rows"][0]
    assert top["sister"] == "aiva" and top["sister_name"] == "Aiva" and top["bucket"] == "success"
    assert top["claude"] is True and top["claude_cost"] == 1.25 and top["steps_done"] == 1 == top["steps_total"]
    assert top["duration"] == "1m 12s" and top["summary"] == "Finished: fixed the parser."
    shell = tk["rows"][2]
    assert shell["bucket"] == "failed" and shell["steps_failed"] == 1 and shell["claude"] is False
    ta = tk["tally"]["aiva"]
    assert ta["total"] == 3 and ta["success"] == 2 and ta["failed"] == 1 and ta["delegated"] == 2
    assert tk["tally"]["saga"]["total"] == 0          # no db yet → an empty tally, not an error
    assert {s["slug"]: s["started"] for s in tk["sisters"]} == {"aiva": True, "saga": False}
    json.dumps(d, default=str)


def test_tasks_list_filters_and_paging(tasks_home):
    q = lambda **kw: dashboard.collect("tasks", **kw)["tasks"]
    assert [r["id"] for r in q(outcome="failed")["rows"]] == [T_SHELL]
    assert q(sister="saga")["rows"] == []
    assert [r["id"] for r in q(q="PARSER")["rows"]] == [T_CODE]
    assert q(outcome="bogus")["filters"]["outcome"] is None            # unknown filter → ignored
    p1 = q(limit=2)
    assert len(p1["rows"]) == 2 and p1["has_more"] and p1["next_before"]
    p2 = q(limit=2, before=p1["next_before"])
    assert [r["id"] for r in p2["rows"]] == [T_SHELL] and not p2["has_more"]
    assert len(q(before="'; DROP TABLE tasks; --")["rows"]) == 3      # malformed cursor → ignored
    code, payload = dashboard.api_response("who=tasks&outcome=success&limit=1")
    assert code == 200 and len(payload["tasks"]["rows"]) == 1 and payload["tasks"]["has_more"]


def test_task_detail_with_claude_event(tasks_home):
    code, t = dashboard.task_response(f"who=aiva&id={T_CODE}")
    assert code == 200 and t["id"] == T_CODE and t["goal"] == "Fix the parser bug"
    assert t["decision"] == "confirm_plan — never done this" and t["familiarity"].startswith("novel")
    assert t["steps"][0]["kind"] == "delegate" and t["steps"][0]["took"] == "1m 05s"
    assert t["steps"][0]["action"]["prompt"] == "Fix the parser bug in src/a.py"
    assert t["steps"][0]["evidence"] == ["Claude Code done: fixed", "grade good: ok"]
    tl = t["timeline"]
    assert [e["event"] for e in tl][:3] == ["start", "familiarity", "decision"]
    assert t["announcements"] == 1 and [e for e in tl if e["secondary"]][0]["event"] == "announce(question)"
    cl = next(e for e in tl if e["event"] == "claude")
    assert cl["claude"]["session"] == SESSION and cl["claude"]["cost"] == 1.25
    (job,) = t["claude_jobs"]
    assert job["job"] == f"{T_CODE}-s1" and job["n"] == 1 and job["prompt"].startswith("Fix the parser")
    assert job["result"]["text"] == "Fixed the parser. All tests pass." and job["result"]["legacy"] is False
    assert job["result"]["payload"]["num_turns"] == 5
    (run,) = job["runs"]
    assert run["session"] == SESSION and not run.get("inferred") and run["status"] == "done"
    assert run["took"] == "1m 05s" and run["transcript_path"].endswith(SESSION + ".jsonl")
    a = run["activity"]
    assert a["files_edited"] == [{"path": "src/a.py", "count": 2}]
    assert a["files_written"] == [{"path": "tests/test_a.py", "count": 1}]
    assert a["files_read"] == [{"path": "src/a.py", "count": 2}]
    assert a["commands"] == ["pytest -q"] and a["lines_added"] == 12
    assert t["claude_cost"] == 1.25
    json.dumps(t, default=str)


def test_task_detail_old_job_uses_legacy_log_and_infers_session(tasks_home):
    code, t = dashboard.task_response(f"who=aiva&id={T_OLD}")
    assert code == 200
    (job,) = t["claude_jobs"]
    assert job["result"]["legacy"] is True and job["result"]["text"].startswith("# Report")
    (run,) = job["runs"]
    assert run["session"] == OLD_SESSION and run["inferred"] is True
    assert run["activity"]["files_edited"] == [{"path": "src/b.py", "count": 2}]


def test_task_detail_steps_clip_args_and_show_pending(tasks_home):
    code, t = dashboard.task_response(f"who=aiva&id={T_SHELL}")
    assert code == 200 and t["claude_jobs"] == []
    s1, s2, s3 = t["steps"]
    assert s1["action"]["tool"] == "create_file" and len(s1["action"]["args"]["content"]) < 450
    assert "(+600 chars)" in s1["action"]["args"]["content"]
    assert s2["action"] == {"command": "python x.py", "purpose": "check"} and s2["failure"] == "rc=1"
    assert s2["attempts"] == 2 and s3["status"] == "pending"
    assert t["pending"]["kind"] == "confirm_command"
    assert next(e for e in t["timeline"] if e["event"] == "error")["tone"] == "bad"


def test_task_detail_rejects_bad_who_and_ids(tasks_home):
    assert dashboard.task_response(f"who=bogus&id={T_CODE}")[0] == 404
    assert dashboard.task_response(f"who=saga&id={T_CODE}")[0] == 404          # no db / not hers
    assert dashboard.task_response("who=aiva&id=t-20990101-000000-ffff")[0] == 404
    for bad in ("", "..%2F..%2Fmemory", "t-20260920-101500-abcd-s1", "t-20260920-101500-ABCD",
                "t-20260920-101500-abcd%00", "*"):
        code, payload = dashboard.task_response(f"who=aiva&id={bad}")
        assert code == 400 and "malformed" in payload["error"], bad


def test_job_and_transcript_lookups_refuse_traversal(tasks_home):
    assert dashboard.job_result("../../secrets") is None
    assert dashboard.job_result(f"{T_CODE}-s1/../../x") is None
    assert dashboard.job_result(f"{T_CODE}-s1")["text"].startswith("Fixed")
    assert dashboard.find_transcript("../*") is None
    assert dashboard.find_transcript("*") is None
    assert dashboard.find_transcript(SESSION).endswith(SESSION + ".jsonl")
    # a `claude` event carrying a hostile job/session is dropped / neutered
    assert dashboard.parse_claude_event(json.dumps({"job": "../../x", "session": SESSION})) is None
    c = dashboard.parse_claude_event(json.dumps({"job": f"{T_CODE}-s1", "session": "../*"}))
    assert c["session"] is None
    # the runner clips details at 300 chars: a cut-off JSON still yields what a regex can find
    cut = json.dumps({"job": f"{T_CODE}-s2", "session": SESSION, "cost": 0.5, "status": "done",
                      "log": "x" * 400})[:300]
    c = dashboard.parse_claude_event(cut)
    assert c["job"] == f"{T_CODE}-s2" and c["session"] == SESSION and c["cost"] == 0.5


def test_summarize_claude_transcript_pure():
    lines = [json.dumps(e) for e in _transcript_lines("do it", "C:\\p")] + ["", "{bad", "[1,2]"]
    a = dashboard.summarize_claude_transcript(lines)
    assert a["turns"] == 2 and a["model"] == "claude-x" and a["prompt"] == "do it"
    assert a["tools"] == {"Read": 2, "Grep": 1, "Edit": 1, "MultiEdit": 1, "Write": 1, "Bash": 1}
    assert a["searches"] == ["Grep def main  in src"] and a["tool_errors"] == 1
    assert a["final_text"] == "Done: fixed it." and a["cost_usd"] == 1.25 and a["duration_ms"] == 61000
    # a `result` event wins for the final answer and the turn count
    a = dashboard.summarize_claude_transcript(lines + [json.dumps(
        {"type": "result", "result": "Final.", "num_turns": 9, "total_cost_usd": 2.0})])
    assert a["final_text"] == "Final." and a["turns"] == 9 and a["cost_usd"] == 2.0
    assert dashboard.summarize_claude_transcript([])["turns"] == 0


def test_task_timeline_and_buckets_pure():
    tl = dashboard.task_timeline([{"ts": "2026-09-20T10:00:00", "event": "finish", "detail": "failed: x"},
                                  {"ts": "2026-09-20T10:00:01", "event": "announce(done)", "detail": "y"},
                                  "junk", {"event": "grade", "detail": "partial: meh"}])
    assert [(e["event"], e["tone"], e["secondary"]) for e in tl] == [
        ("finish", "bad", False), ("announce(done)", "info", True), ("grade", "warn", False)]
    assert tl[0]["time"] == "10:00:00" and tl[0]["day"] == "2026-09-20"
    assert dashboard.task_bucket("interrupted", None) == "interrupted"
    assert dashboard.task_bucket("interrupted", "cancelled") == "cancelled"
    assert dashboard.task_bucket("awaiting_user", None) == "active"
    assert dashboard.task_steps("not a list") == [] and dashboard.task_steps([None, {"n": 1}])[0]["status"] == "pending"


def test_http_task_route(tasks_home):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def get(path):
        try:
            with urllib.request.urlopen(base + path, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        code, body = get(f"/api/task?who=aiva&id={T_CODE}")
        assert code == 200 and body["claude_jobs"][0]["runs"][0]["session"] == SESSION
        assert get("/api/task?who=aiva&id=..%5C..%5Cx")[0] == 400
        assert get(f"/api/task?who=nobody&id={T_CODE}")[0] == 404
        code, body = get("/api/data?who=tasks")
        assert code == 200 and len(body["tasks"]["rows"]) == 3
    finally:
        srv.shutdown()
        srv.server_close()


def test_page_carries_the_tasks_tab(home):
    html = dashboard.page()
    assert '"who": "tasks"' in html and '"label": "Tasks"' in html
    assert "function renderTasks" in html and "/api/task?who=" in html and "d.who==='tasks'" in html
    assert "#tasks/" in html          # the sister tab's task rows link into it
