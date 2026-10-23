"""Memory as one db per character: backup snapshots named by stem (a sister's boot never prunes
Aiva's), store_meta identity/embedding guards, ':memory:' support, sister-fact filtering, freshness."""
import pytest

from modules.memory import FACT_EXTRACTION_PROMPT, Memory, misheard_list


def test_in_memory_db_skips_abspath_and_backup():
    m = Memory(":memory:")
    assert m.db_file == ":memory:" and m.in_memory and m.backup_stem == "memory"
    assert m.is_fresh()
    assert m.remember("Always say hi.") == 1
    assert not m.is_fresh()
    assert m.fact_prompt == FACT_EXTRACTION_PROMPT
    m.close()


def test_backup_stem_defaults_to_db_basename(tmp_path):
    a = Memory(str(tmp_path / "memory.db"))
    b = Memory(str(tmp_path / "saga.db"))
    c = Memory(str(tmp_path / "other.db"), backup_stem="sis")
    assert (a.backup_stem, b.backup_stem, c.backup_stem) == ("memory", "saga", "sis")
    for m in (a, b, c):
        m.close()


def test_backup_uses_stem_and_never_prunes_a_sisters_snapshots(tmp_path):
    db = tmp_path / "saga.db"
    m = Memory(str(db))                                   # fresh file: too small to snapshot
    m.conn.executemany("INSERT INTO messages (session_id, role, content, ts) VALUES (?,?,?,?)",
                       [("s", "user", "x" * 500, "t")] * 40)
    m.conn.commit()
    m.close()
    assert db.stat().st_size >= 4096
    bdir = tmp_path / "backups"
    bdir.mkdir(exist_ok=True)                             # the first boot may already have snapshotted
    for p in bdir.glob("saga-*.db"):
        p.unlink()
    aivas = [bdir / f"memory-20260101-0000{i:02d}.db" for i in range(12)]
    for p in aivas:
        p.write_bytes(b"aiva")                            # 12 fresh Aiva snapshots (> keep=10)
    m2 = Memory(str(db))
    snaps = sorted(bdir.glob("saga-*.db"))
    assert len(snaps) == 1 and snaps[0].stat().st_size > 0
    assert all(p.exists() for p in aivas)                 # one shared glob would have pruned two
    m2.close()
    m3 = Memory(str(db))                                  # a recent saga snapshot → no churn
    assert len(list(bdir.glob("saga-*.db"))) == 1
    m3.close()


def test_check_identity_stamps_then_refuses_a_mismatch():
    m = Memory(":memory:")
    assert m.meta_get("character") is None and m.meta_get("character", "x") == "x"
    assert m.check_identity("saga") == "saga"
    assert m.meta_get("character") == "saga"
    assert m.check_identity("saga") == "saga"             # idempotent
    with pytest.raises(RuntimeError, match="belongs to 'saga', not 'aiva'"):
        m.check_identity("aiva")
    m.meta_set("character", "aiva")                       # meta_set upserts
    assert m.meta_get("character") == "aiva"


def test_check_embeddings_stamps_then_nulls_every_nullable_embedding_column():
    m = Memory(":memory:")
    assert m.check_embeddings("text-embedding-3-small", 1536) == []
    assert (m.meta_get("embed_model"), m.meta_get("embed_dim")) == ("text-embedding-3-small", "1536")
    # tables created LATER by other stores on the same connection are discovered live
    m.conn.execute("CREATE TABLE finds (id INTEGER PRIMARY KEY, subject TEXT, embedding BLOB)")
    m.conn.execute("CREATE TABLE journal (id INTEGER PRIMARY KEY, entry TEXT, embedding BLOB)")
    m.conn.execute("CREATE TABLE speaker_samples (id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)")
    m.conn.execute("CREATE TABLE plain (id INTEGER PRIMARY KEY, note TEXT)")
    m.conn.execute("INSERT INTO finds (subject, embedding) VALUES ('a', ?)", (b"\x00\x01",))
    m.conn.execute("INSERT INTO journal (entry, embedding) VALUES ('b', ?)", (b"\x00\x02",))
    m.conn.execute("INSERT INTO speaker_samples (embedding) VALUES (?)", (b"\x00\x03",))
    m.conn.commit()
    assert m.check_embeddings("text-embedding-3-small", 1536) == []      # unchanged → untouched
    assert m.conn.execute("SELECT embedding FROM finds").fetchone()[0] is not None
    cleared = m.check_embeddings("text-embedding-3-large", 3072)
    assert set(cleared) == {"finds", "journal"}                            # voiceprints (NOT NULL) left alone
    assert m.conn.execute("SELECT embedding FROM finds").fetchone()[0] is None
    assert m.conn.execute("SELECT embedding FROM journal").fetchone()[0] is None
    assert m.conn.execute("SELECT embedding FROM speaker_samples").fetchone()[0] is not None
    assert (m.meta_get("embed_model"), m.meta_get("embed_dim")) == ("text-embedding-3-large", "3072")
    assert m.check_embeddings("text-embedding-3-large", 3072) == []      # re-stamped


def test_load_facts_hides_sister_facts_but_keeps_them():
    m = Memory(":memory:")
    m.upsert_facts([{"category": "instruction", "fact": "Always say hi."},
                    {"category": "sister", "fact": "Aiva can read books on her own."},
                    {"fact": "He likes tea."}])
    assert m.load_facts() == ["Always say hi.", "He likes tea."]
    assert m.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 3
    assert not m.is_fresh()


def test_retire_fact_only_takes_back_what_she_noted_herself():
    m = Memory(":memory:")
    m.upsert_facts([{"category": "project", "fact": "The witness lives in the Miso renderer."},
                    {"category": "instruction", "fact": "Never push to main."},
                    {"category": "person", "fact": "Frank is Romanian."}])
    assert m.retire_fact("The witness lives in the Miso renderer.") == "project"
    assert m.retire_fact("Never push to main.") is None          # his word: only he changes it
    assert m.retire_fact("Frank is Romanian.") is None
    assert m.retire_fact("no such fact") is None
    assert m.load_facts() == ["Never push to main.", "Frank is Romanian."]


def test_a_fact_that_stopped_being_true_moves_out_of_her_memory_with_its_replacement():
    m = Memory(":memory:")
    m.upsert_facts([{"category": "work", "fact": "He works at Fianu as an engineer."},
                    {"category": "instruction", "fact": "Always bring terminals to the front."}])
    # his word changes any category, even instructions; spacing/case/end punctuation don't matter
    m.upsert_facts([{"category": "work", "fact": "He leads the AI team at Fianu.",
                     "replaces": "he works at fianu as an engineer"},
                    {"category": "instruction", "fact": "Leave terminals in the background.",
                     "replaces": "Always bring terminals to the front"}])
    assert m.load_facts() == ["He leads the AI team at Fianu.", "Leave terminals in the background."]
    old = m.load_superseded()
    assert [(o, n) for o, n, _ in old] == [
        ("Always bring terminals to the front.", "Leave terminals in the background."),
        ("He works at Fianu as an engineer.", "He leads the AI team at Fianu.")]
    assert len(m.superseded) == 2


def test_a_replaces_that_names_nothing_known_changes_nothing():
    m = Memory(":memory:")
    m.upsert_facts([{"fact": "He likes tea."}])
    m.upsert_facts([{"fact": "He likes coffee.", "replaces": "He likes juice."},     # unknown
                    {"fact": "He likes tea.", "replaces": "He likes tea."},          # itself
                    {"fact": "He has a cat.", "replaces": None}])
    assert m.load_facts() == ["He likes tea.", "He likes coffee.", "He has a cat."]
    assert m.load_superseded() == [] and m.superseded == []


def test_misheard_list_rendering():
    assert misheard_list("Aiva", ()) == '"Aiva"'
    assert misheard_list("Saga", ("saga", "sega", "hey saga", "sagah")) == '"Saga", "Sega" or "Sagah"'
    assert misheard_list("Saga", ("a", "b", "c", "d", "e", "f", "g", "h", "i")).count('"') == 16  # capped at 8
