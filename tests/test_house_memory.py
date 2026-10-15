"""HouseMemory on an in-memory sqlite: transcript rows (user + said), paging/day filter, events
cursor, pruning, kv and the midnight-resetting day counters — with an injected clock that may
return either ISO strings or epoch seconds."""
import sqlite3
from datetime import datetime, timedelta

from modules.house_memory import HouseMemory

T0 = datetime(2026, 9, 11, 15, 0, 0)


class Clock:
    def __init__(self, t=T0, as_float=False):
        self.t = t
        self.as_float = as_float

    def __call__(self):
        return self.t.timestamp() if self.as_float else self.t.isoformat()

    def advance(self, **kw):
        self.t += timedelta(**kw)


def make(as_float=False):
    clock = Clock(as_float=as_float)
    return HouseMemory(sqlite3.connect(":memory:"), now=clock), clock


def test_tables_exist_and_pragmas_do_not_break_memory_db():
    hm, _ = make()
    names = {r[0] for r in hm.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"house_transcript", "house_events", "house_kv"} <= names


def test_transcript_rows_and_recent_order():
    hm, clock = make()
    a = hm.log_user_line(1, "Saga, weather?", "saga", kind="named")
    clock.advance(seconds=3)
    b = hm.log_said(1, "saga", "Sunny.", "him", audible=True, interrupted=False)
    clock.advance(seconds=3)
    c = hm.log_said(2, "aiva", "psst", "saga", audible=False, interrupted=True, kind="sister")
    rows = hm.recent_transcript()
    assert [r["id"] for r in rows] == [c, b, a]                      # newest first
    assert rows[2] == {"id": a, "ts": T0.isoformat(), "turn_id": 1, "speaker": "user", "addressee": "saga",
                       "text": "Saga, weather?", "audible": True, "kind": "named", "interrupted": False}
    assert rows[0]["audible"] is False and rows[0]["interrupted"] is True and rows[0]["kind"] == "sister"
    assert [r["id"] for r in hm.recent_transcript(limit=1)] == [c]
    assert [r["id"] for r in hm.recent_transcript(before_id=c)] == [b, a]


def test_recent_transcript_day_filter():
    hm, clock = make()
    hm.log_user_line(1, "today", "aiva")
    clock.advance(days=1)
    hm.log_user_line(2, "tomorrow", "aiva")
    assert [r["text"] for r in hm.recent_transcript(day="2026-09-11")] == ["today"]
    assert [r["text"] for r in hm.recent_transcript(day="2026-09-12")] == ["tomorrow"]


def test_events_cursor_and_relayed_to_list():
    hm, _ = make()
    e1 = hm.add_event("aiva", "tool_done", "looked something up", capability="web", relayed_to=["saga"])
    e2 = hm.add_event("saga", "said", "hello", relayed_to="aiva")
    assert e2 == e1 + 1
    evs = hm.events_since(0)
    assert [e["id"] for e in evs] == [e1, e2] and evs[0]["relayed_to"] == "saga" and evs[0]["capability"] == "web"
    assert [e["id"] for e in hm.events_since(e1)] == [e2]
    assert hm.events_since(e2) == []


def test_prune_removes_old_rows_only():
    hm, clock = make()
    hm.log_user_line(1, "old", "aiva")
    hm.add_event("aiva", "x", "old")
    clock.advance(days=40)
    hm.log_user_line(2, "new", "aiva")
    hm.add_event("aiva", "x", "new")
    assert hm.prune(days=30) == 2
    assert [r["text"] for r in hm.recent_transcript()] == ["new"]
    assert [e["summary"] for e in hm.events_since(0)] == ["new"]


def test_kv_and_day_counters():
    hm, clock = make()
    assert hm.kv_get("sister_talk") is None and hm.kv_get("sister_talk", "private") == "private"
    hm.kv_set("sister_talk", "audible")
    assert hm.kv_get("sister_talk") == "audible"
    hm.kv_set("sister_talk", "private")
    assert hm.kv_get("sister_talk") == "private"
    assert hm.day_count("sister_chats") == 0
    assert hm.bump_day_count("sister_chats") == 1
    assert hm.bump_day_count("sister_chats") == 2
    assert hm.day_count("sister_chats") == 2
    clock.advance(days=1)
    assert hm.day_count("sister_chats") == 0                      # midnight reset
    assert hm.bump_day_count("sister_chats") == 1


def test_float_clock_is_converted_to_iso():
    hm, clock = make(as_float=True)
    rid = hm.log_user_line(1, "hi", "aiva")
    assert hm.recent_transcript()[0]["ts"] == T0.isoformat()
    assert hm.day_count("k", now=T0.timestamp()) == 0
    hm.bump_day_count("k", now=T0.timestamp())
    assert hm.day_count("k", now=T0.timestamp()) == 1
    assert hm.day_count("k", now=(T0 + timedelta(days=1)).timestamp()) == 0
    assert rid == 1
