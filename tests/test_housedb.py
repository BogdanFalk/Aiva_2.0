"""housedb: the shared house.db opener (WAL, busy timeout) and the one-shot copy of a character's
spend-cap tables into it — idempotent, additive on usage, keeps house alert markers."""
import sqlite3

from modules.housedb import MIGRATION_KEY, migrate_limits_from, open_house_db
from modules.limits import ServiceLimits


def test_open_house_db_pragmas_and_multi_connection(tmp_path):
    path = str(tmp_path / "sub" / "house.db")
    a = open_house_db(path, timeout=2.0)
    b = open_house_db(path)
    assert a.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert a.execute("PRAGMA busy_timeout").fetchone()[0] == 2000
    assert b.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert a.execute("PRAGMA synchronous").fetchone()[0] == 1          # NORMAL
    # two connections, two ServiceLimits, one set of counters
    la = ServiceLimits(a, caps={"tavily": 10})
    lb = ServiceLimits(b, caps={"tavily": 10})
    a.execute("INSERT INTO service_usage (service, period, used) VALUES ('tavily', '2026-09', 3)")
    a.commit()
    assert lb.usage("tavily", "2026-09-11T10:00:00") == 3
    a.close()
    b.close()


def test_open_house_db_default_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AIVA_HOUSE_DB", str(tmp_path / "env.db"))
    conn = open_house_db()
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()
    assert (tmp_path / "env.db").exists()


async def test_migrate_limits_once_and_additive(tmp_path):
    mem = sqlite3.connect(str(tmp_path / "memory.db"))
    house = open_house_db(str(tmp_path / "house.db"))
    old = ServiceLimits(mem, caps={"tavily": 5}, now=lambda: "2026-09-11T10:00:00")
    for _ in range(5):
        await old.record("tavily")                                     # hits cap → paused + alert markers
    await old.record("tts", 120)
    assert old.is_paused("tavily", "2026-09-11T10:00:00")
    # the house already counted a little this month, and already warned once
    new = ServiceLimits(house, caps={"tavily": 5}, now=lambda: "2026-09-11T11:00:00")
    await new.record("tavily", 1)
    house.execute("INSERT INTO limits_kv (key, value) VALUES ('warn:tavily:2026-09', 'house-first')")
    house.commit()

    copied = migrate_limits_from(house, mem, now="2026-09-11T12:00:00")
    assert copied > 0
    assert new.usage("tavily", "2026-09-11T12:00:00") == 6              # 1 (house) + 5 (memory)
    assert new.usage("tts", "2026-09-11T12:00:00") == 120
    assert new.is_paused("tavily", "2026-09-11T12:00:00")               # pause came across
    assert house.execute("SELECT value FROM limits_kv WHERE key='warn:tavily:2026-09'").fetchone()[0] == "house-first"
    assert house.execute("SELECT 1 FROM limits_kv WHERE key='capped:tavily:2026-09'").fetchone()
    assert house.execute("SELECT value FROM limits_kv WHERE key=?", (MIGRATION_KEY,)).fetchone()[0] == "2026-09-11T12:00:00"
    # idempotent
    assert migrate_limits_from(house, mem) == 0
    assert new.usage("tavily", "2026-09-11T12:00:00") == 6
    mem.close()
    house.close()


def test_migrate_from_a_db_without_limits_tables(tmp_path):
    mem = sqlite3.connect(":memory:")
    house = open_house_db(str(tmp_path / "house.db"))
    assert migrate_limits_from(house, mem) == 0
    assert house.execute("SELECT 1 FROM limits_kv WHERE key=?", (MIGRATION_KEY,)).fetchone()
    assert migrate_limits_from(house, mem) == 0
    house.close()
