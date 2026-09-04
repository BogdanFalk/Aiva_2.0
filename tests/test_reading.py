import asyncio
import json
import math
import os
import sqlite3

import pytest

from modules import library, reading
from modules.reading import BookMemory, ReadingEngine, StudyLimits


# --- fakes -------------------------------------------------------------------------------

def fake_embed_factory():
    import zlib

    async def embed(text):
        # deterministic 512-d bag-of-words vector, normalised (crc32, not the per-run-salted hash())
        vec = [0.0] * reading.EMBED_DIMS
        for tok in (text or "").lower().split():
            vec[zlib.crc32(tok.encode()) % reading.EMBED_DIMS] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
    return embed


class FakeReply:
    def __init__(self, text, pt=500, ct=300):
        self.text = text
        self.tool_calls = []
        self.prompt_tokens = pt
        self.completion_tokens = ct


def note_json(summary="A thing happens.", quotes=None, feeling="curious", extra_key=None):
    d = {"summary": summary, "key_ideas": ["An idea worth keeping."],
         "quotes": quotes if quotes is not None else [],
         "nuances": {"tone": "wry", "style": "short sentences", "subtext": "more than it says",
                     "devices": ["irony"]},
         "questions": ["What next?"], "connections": [], "reflection": "It made me think about pacing.",
         "vocabulary": [], "feeling": feeling, "book_so_far": summary + " " + (extra_key or ""),
         "state": {"characters": [{"name": "Alice", "status": "curious"}], "threads": ["the door"],
                   "open_questions": [], "motifs": ["light"]}}
    return json.dumps(d)


class FakeLLM:
    """Keyed on which prompt constant leads the system message."""
    def __init__(self, notes=None, wrap=None, synth=None, delay=0.0, fail_json_once=False):
        self.notes = list(notes or [])
        self.wrap = wrap or {"struck": "The storm scene stayed with me. It made me want to keep going.",
                             "next_hope": "whether she opens the door", "mood": "hooked"}
        self.synth = synth
        self.delay = delay
        self.fail_json_once = fail_json_once
        self._failed = False
        self.calls = []

    async def __call__(self, messages, *, tools=None, json_mode=False, max_tokens=900, temperature=0.2):
        system = messages[0]["content"]
        if self.delay:
            await asyncio.sleep(self.delay)
        if system.startswith(reading.CLOSE_READING_PROMPT[:40]):
            self.calls.append("note")
            if self.fail_json_once and not self._failed:
                self._failed = True
                return FakeReply("not json")
            nxt = self.notes.pop(0) if self.notes else note_json()
            return FakeReply(nxt)
        if system.startswith(reading.SESSION_WRAP_PROMPT[:40]):
            self.calls.append("wrap")
            return FakeReply(json.dumps(self.wrap))
        if system.startswith(reading.SYNTHESIS_PROMPT[:40]):
            self.calls.append("synth")
            data = self.synth or {"summary": "A short book about a door.", "arc": "opens, closes",
                                  "themes": [{"theme": "curiosity", "how_it_shows": "the door"}],
                                  "people": [{"name": "Alice", "role": "reader", "arc_or_position": "curious"}],
                                  "passages": [], "style": {"voice": "wry", "sentence": "short",
                                                            "devices": ["irony"], "what_to_borrow": ["short verdicts"]},
                                  "takeaways": ["I want to notice small doors."], "disagreements": [],
                                  "verdict": "Slight but charming.", "talk_line": "It's about a door, and about not opening it."}
            return FakeReply(json.dumps(data))
        return FakeReply("{}")


def make_book_file(tmp_path, name="tale.txt", chapters=5, word="alpha", words=650):
    body = "\n\n".join(f"CHAPTER {'IVXLC'[i] if i < 5 else i}\n\n" + " ".join([f"{word}{i}"] * words) + "."
                       for i in range(chapters))
    f = tmp_path / name
    f.write_text(body, encoding="utf-8")
    return str(f)


def make_engine(tmp_path, llm=None, embed=True, awake=True, idle=True, limits=None):
    lib_dir = tmp_path / "library"
    lib_dir.mkdir(exist_ok=True)
    conn = sqlite3.connect(":memory:")
    bm = BookMemory(conn, embed=fake_embed_factory() if embed else None)
    llm = llm or FakeLLM()
    announces, quiets = [], []

    async def announce(text):
        announces.append(text)

    async def quiet(text):
        quiets.append(text)

    lim = limits or StudyLimits(milestone_min_gap_secs=0, announce_floor_secs=0)
    lim.announce_floor_secs = 0        # keep the offline suite fast regardless of caller
    eng = ReadingEngine(bm, llm_call=llm, synth_call=llm, client=None, library_dir=str(lib_dir),
                        model_name="gpt-5.6-luna", synth_model_name="gpt-5.6-terra",
                        limits=lim, log=lambda m: None)
    # client=None disables real embedding of chunks; recall/note embeds go through bm._embed
    eng.attach_voice(announce=announce, quiet_note=quiet, is_awake=lambda: awake, is_idle=lambda: idle)
    eng._announces = announces
    eng._quiets = quiets
    return eng, bm, str(lib_dir)


async def run_task(eng):
    if eng._task:
        await eng._task


# --- library scan / ingest ----------------------------------------------------------------

async def test_scan_ingests_and_dedupes(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path)
    make_book_file(tmp_path / "library" if False else None) if False else None
    # write two files, one a duplicate of the other's content
    p1 = os.path.join(lib_dir, "tale.txt")
    body = make_book_file(tmp_path, "src.txt")
    text = open(body, encoding="utf-8").read()
    open(p1, "w", encoding="utf-8").write(text)
    open(os.path.join(lib_dir, "copy.txt"), "w", encoding="utf-8").write(text)
    # backdate mtimes so the scan doesn't skip them as "still copying"
    old = 1000.0
    for n in ("tale.txt", "copy.txt"):
        os.utime(os.path.join(lib_dir, n), (old, old))
    result = await eng.scan_once()
    assert len(result["added"]) == 1 and len(result["duplicates"]) == 1
    assert len(bm.books()) == 1
    book = bm.books()[0]
    assert book["sections"] == 5 and book["status"] == "new"
    assert len(bm.sections(book["id"])) == 5
    # second scan with nothing new is a no-op
    assert (await eng.scan_once())["added"] == []


async def test_unreadable_recorded(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path)
    p = os.path.join(lib_dir, "empty.epub")
    import zipfile
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
    os.utime(p, (1000.0, 1000.0))
    await eng.scan_once()
    books = bm.books("unreadable")
    assert len(books) == 1 and books[0]["status_reason"]


# --- study loop ---------------------------------------------------------------------------

async def _ingest(tmp_path, eng, bm, **kw):
    lib_dir = os.path.join(str(tmp_path), "library")
    src = make_book_file(tmp_path, "src.txt", **kw)
    dst = os.path.join(lib_dir, "tale.txt")
    open(dst, "w", encoding="utf-8").write(open(src, encoding="utf-8").read())
    os.utime(dst, (1000.0, 1000.0))
    await eng.scan_once()
    return bm.books()[0]


async def test_session_reads_sections(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path, limits=StudyLimits(sections_per_session=2, milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm)
    res = await eng.start_session(book["id"], sections=2)
    assert res["success"] and res["sections_planned"] == 2
    await run_task(eng)
    b = bm.get_book(book["id"])
    assert b["cursor_section"] == 2 and b["status"] == "reading"
    notes = bm.notes_for(book["id"])
    assert len(notes) == 2 and notes[0]["nuances"]["tone"] == "wry"
    assert b["rolling_summary"] and b["state"]["characters"]
    assert any("starting to read" in q for q in eng._quiets)     # start is a SILENT note, not spoken
    assert not any("starting to read" in a for a in eng._announces)
    assert any("finished a reading session" in a for a in eng._announces)  # no becoming hook → engine speaks
    sess = bm.conn.execute("SELECT status FROM book_sessions").fetchone()[0]
    assert sess == "done"


async def test_quotes_verified(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path, limits=StudyLimits(sections_per_session=1, milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm)
    sec0 = bm.section(book["id"], 0)
    real = " ".join(sec0["text"].split()[:8])                     # a real verbatim phrase
    long_real = " ".join(sec0["text"].split()[:40])              # 40 words of real text
    quotes = [{"text": real, "why": "opening", "speaker": "narrator"},
              {"text": "a line that never appears in the book", "why": "x", "speaker": "narrator"},
              {"text": long_real, "why": "long", "speaker": "narrator"}]
    eng.llm_call = FakeLLM(notes=[note_json(quotes=quotes)])
    await run_task_after(eng, book["id"])
    note = bm.notes_for(book["id"])[0]
    texts = [q["text"] for q in note["quotes"]]
    assert real in texts                                         # verbatim kept
    assert "a line that never appears in the book" not in texts  # invented dropped
    assert any(len(q["text"].split()) == 25 and q.get("cut") for q in note["quotes"])  # 40 → 25


async def run_task_after(eng, book_id, **kw):
    res = await eng.start_session(book_id, **kw)
    assert res["success"], res
    await run_task(eng)


async def test_quiet_while_asleep(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path, awake=False,
                                   limits=StudyLimits(sections_per_session=2, milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm)
    await run_task_after(eng, book["id"], sections=2)
    assert eng._announces == []                                  # nothing spoken to an empty room
    assert any("starting to read" in q for q in eng._quiets)
    assert any("finished a reading session" in q for q in eng._quiets)


async def test_milestone_demoted_when_user_talking(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path, idle=False,
                                   limits=StudyLimits(sections_per_session=1, milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm)
    await run_task_after(eng, book["id"])
    assert not any("Reading: finished" in a for a in eng._announces)   # milestone not spoken mid-conversation
    assert any("Reading: finished" in q for q in eng._quiets)


async def test_front_matter_skipped_without_llm_call(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path, limits=StudyLimits(sections_per_session=6, milestone_min_gap_secs=0))
    # a book with a big front-matter block then chapters
    lib_dir_p = os.path.join(str(tmp_path), "library")
    text = ("just a title page\n\n" + " ".join(["frontish"] * 260) + ".\n\n"
            + "\n\n".join(f"CHAPTER {i}\n\n" + " ".join([f"body{i}"] * 650) + "." for i in range(1, 4)))
    dst = os.path.join(lib_dir_p, "fm.txt")
    open(dst, "w", encoding="utf-8").write(text)
    os.utime(dst, (1000.0, 1000.0))
    await eng.scan_once()
    book = bm.books()[0]
    llm = FakeLLM()
    eng.llm_call = eng.synth_call = llm
    await run_task_after(eng, book["id"])
    front = [n for n in bm.notes_for(book["id"]) if n["skipped"]]
    assert front and front[0]["skip_reason"].endswith("matter")


# --- resume / stop ------------------------------------------------------------------------

async def test_resume_after_restart(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path, limits=StudyLimits(sections_per_session=2, milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm)
    await run_task_after(eng, book["id"], sections=2)
    assert bm.get_book(book["id"])["cursor_section"] == 2
    # "restart": a fresh engine on the same connection
    eng2 = ReadingEngine(bm, llm_call=FakeLLM(), synth_call=FakeLLM(), client=None, library_dir=lib_dir,
                         model_name="gpt-5.6-luna", limits=StudyLimits(sections_per_session=2, milestone_min_gap_secs=0),
                         log=lambda m: None)
    eng2.attach_voice(announce=lambda t: asyncio.sleep(0), is_awake=lambda: True)
    bm.reconcile_stale()
    await run_task_after(eng2, book["id"], sections=2)
    b = bm.get_book(book["id"])
    assert b["cursor_section"] == 4
    assert len(bm.notes_for(book["id"])) == 4                    # no duplicate notes


async def test_stop_after_current_section(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path,
                                   limits=StudyLimits(sections_per_session=5, milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm)
    slow = FakeLLM(delay=0.05)
    eng.llm_call = eng.synth_call = slow
    res = await eng.start_session(book["id"])
    assert res["success"]
    await asyncio.sleep(0.02)
    await eng.stop()
    b = bm.get_book(book["id"])
    assert 0 < b["cursor_section"] < 5 and b["status"] == "reading"
    sess_status = bm.conn.execute("SELECT status FROM book_sessions ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert sess_status == "stopped"


async def test_stop_for_good_abandons(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path,
                                   limits=StudyLimits(sections_per_session=5, milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm)
    eng.llm_call = eng.synth_call = FakeLLM(delay=0.05)
    await eng.start_session(book["id"])
    await asyncio.sleep(0.02)
    await eng.stop(for_good=True)
    assert bm.get_book(book["id"])["status"] == "abandoned"


# --- finish / synthesis -------------------------------------------------------------------

async def test_until_done_finishes_and_synthesizes(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path, limits=StudyLimits(milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm, chapters=3)
    finished = {"book": None}

    async def on_finished(b, syn, notes):
        finished["book"] = (b, syn, notes)

    eng.on_book_finished = on_finished
    await run_task_after(eng, book["id"], until_done=True)
    b = bm.get_book(book["id"])
    assert b["status"] == "finished" and b["cursor_section"] == 3
    syn = bm.synthesis_for(book["id"])
    assert syn and syn["verdict"] and syn["talk_line"]
    assert finished["book"] is not None and len(finished["book"][2]) == 3


async def test_synthesis_passages_filtered_to_known_quotes(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path, limits=StudyLimits(milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm, chapters=3)
    sec0 = bm.section(book["id"], 0)
    real = " ".join(sec0["text"].split()[:6])
    notes = [note_json(quotes=[{"text": real, "why": "x", "speaker": "narrator"}]),
             note_json(), note_json()]
    synth = {"summary": "s", "arc": "a", "themes": [], "people": [], "verdict": "ok", "talk_line": "hi",
             "style": {}, "takeaways": [], "disagreements": [],
             "passages": [{"text": real, "section": "CHAPTER I", "why": "kept"},
                          {"text": "an invented passage never in notes", "section": "x", "why": "drop"}]}
    eng.llm_call = eng.synth_call = FakeLLM(notes=notes, synth=synth)
    await run_task_after(eng, book["id"], until_done=True)
    passages = bm.synthesis_for(book["id"])["passages"]
    assert len(passages) == 1 and real in passages[0]["text"]


async def test_bad_json_retry_then_fallback(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path, limits=StudyLimits(sections_per_session=1, milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm)

    class AllBadJSON(FakeLLM):
        async def __call__(self, messages, **kw):
            if messages[0]["content"].startswith(reading.CLOSE_READING_PROMPT[:40]):
                return FakeReply("still not json")
            return await super().__call__(messages, **kw)

    eng.llm_call = eng.synth_call = AllBadJSON()
    await run_task_after(eng, book["id"], sections=1)
    note = bm.notes_for(book["id"])[0]
    assert note["summary"] and bm.get_book(book["id"])["cursor_section"] == 1   # cursor still advanced


async def test_book_cost_cap(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path,
                                   limits=StudyLimits(sections_per_session=5, book_cost_usd=0.0000001,
                                                      milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm)
    res = await eng.start_session(book["id"])
    # first section commits (cost recorded), then the cap trips
    if res["success"]:
        await run_task(eng)
    b = bm.get_book(book["id"])
    assert b["cursor_section"] < 5 and b["status"] == "reading"


# --- recall / wishlist / library lines ----------------------------------------------------

async def test_recall_ranks_notes_and_filters_unread(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path, limits=StudyLimits(milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm, chapters=3)
    notes = [note_json(summary="The storm breaks over the harbour and the boats scatter."),
             note_json(summary="A quiet morning, tea and letters."),
             note_json(summary="The reckoning at the lighthouse.")]
    eng.llm_call = eng.synth_call = FakeLLM(notes=notes)
    await run_task_after(eng, book["id"], until_done=True)
    res = await bm.recall("what happened in the storm?")
    assert res["success"] and res["hits"]
    assert "storm" in res["hits"][0]["what"].lower()
    # a book still 'new' is not recalled
    other = await _ingest_named(tmp_path, eng, bm, "unread.txt")
    res2 = await bm.recall("tea and letters", book_id=other["id"])
    assert res2["hits"] == []


async def _ingest_named(tmp_path, eng, bm, name):
    lib_dir_p = os.path.join(str(tmp_path), "library")
    src = make_book_file(tmp_path, "src2.txt", word="beta")
    dst = os.path.join(lib_dir_p, name)
    open(dst, "w", encoding="utf-8").write(open(src, encoding="utf-8").read())
    os.utime(dst, (1000.0, 1000.0))
    await eng.scan_once()
    return [b for b in bm.books() if b["file_name"] == name][0]


async def test_wishlist_add_dedupe_and_fulfil(tmp_path):
    eng, bm, lib_dir = make_engine(tmp_path)
    assert bm.add_wish("A Small Tale", "Jane Doe", "curious")["added"]
    assert not bm.add_wish("a small tale")["added"]              # dedupe, case-insensitive
    assert [w["title"] for w in bm.wishlist()] == ["A Small Tale"]
    # a matching book arriving fulfils the wish
    lib_dir_p = os.path.join(str(tmp_path), "library")
    body = ("Title: A Small Tale\nAuthor: Jane Doe\n\n"
            + "\n\n".join(f"CHAPTER {i}\n\n" + " ".join([f"w{i}"] * 650) + "." for i in range(1, 4)))
    dst = os.path.join(lib_dir_p, "A Small Tale.txt")
    open(dst, "w", encoding="utf-8").write(body)
    os.utime(dst, (1000.0, 1000.0))
    res = await eng.scan_once()
    assert res["added"]
    assert bm.wishlist() == []                                  # fulfilled


def test_load_library_lines_shapes(tmp_path):
    conn = sqlite3.connect(":memory:")
    bm = BookMemory(conn, embed=None)
    bm.add_wish("Frankenstein")
    lines = bm.load_library_lines()
    assert any("Wishlist" in l for l in lines)


def test_find_book(tmp_path):
    conn = sqlite3.connect(":memory:")
    bm = BookMemory(conn, embed=None)
    import types
    parsed = library.ParsedBook("The Great Door", "A. Writer", "txt",
                                [library.Section("One", "x " * 300)], meta={})
    st = types.SimpleNamespace(st_size=100, st_mtime=1.0)
    bm.upsert_book(parsed, path="/x/door.txt", file_name="door.txt", sha256="a" * 64, fmt="txt", file_stat=st)
    assert bm.find_book("great door")["title"] == "The Great Door"
    assert bm.find_book("door.txt")["title"] == "The Great Door"
    assert bm.find_book("nonsense zzz") is None


def test_save_synthesis_coerces_list_text_fields():
    """The synth model sometimes returns summary/verdict as a list of sentences; sqlite can't
    bind a list, so save_synthesis must coerce plain-text fields to strings (regression)."""
    import sqlite3
    from modules.reading import BookMemory
    bm = BookMemory(sqlite3.connect(":memory:"), fake_embed_factory())
    bm.save_synthesis(1, {"summary": ["First point.", "Second point.", "Third."],
                          "arc": "opens then closes",
                          "verdict": ["Good.", "Worth it."],
                          "talk_line": "It's about a door.",
                          "themes": [{"theme": "curiosity", "how_it_shows": "the door"}]},
                      model="m", vec=None, cost_usd=0.01)
    row = bm.conn.execute(
        "SELECT summary, arc, verdict, talk_line FROM book_synthesis WHERE book_id = 1").fetchone()
    assert row[0] == "First point. Second point. Third."
    assert row[1] == "opens then closes"
    assert row[2] == "Good. Worth it."
    assert row[3] == "It's about a door."


async def test_start_session_finalizes_fully_read_unsynthesized_book(tmp_path):
    """A book read to the end but left without a synthesis (e.g. the synth call errored) must not
    dead-end at 'synthesis pending' — start_session should finalize it (regression)."""
    eng, bm, lib = make_engine(tmp_path, limits=StudyLimits(sections_per_session=5,
                                                            milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm, chapters=3)
    await eng.start_session(book["id"], until_done=True)
    await run_task(eng)
    assert bm.get_book(book["id"])["status"] == "finished"
    # simulate the crash's aftermath: synthesis gone, status rolled back, cursor still at the end
    bm.conn.execute("DELETE FROM book_synthesis WHERE book_id = ?", (book["id"],))
    bm.set_status(book["id"], "reading", None)
    assert bm.get_book(book["id"])["cursor_section"] >= bm.get_book(book["id"])["sections"]
    res = await eng.start_session(book["id"])
    assert res["success"] and res["status"] == "finishing"
    await run_task(eng)
    assert bm.get_book(book["id"])["status"] == "finished"
    assert bm.synthesis_for(book["id"]) is not None


async def test_unfinalized_books_detects_read_but_unsynthesized(tmp_path):
    """The auto-finalize-on-boot query must flag a book read to the end with no synthesis, and
    ignore both finished books and ones still mid-read (regression)."""
    eng, bm, lib = make_engine(tmp_path, limits=StudyLimits(sections_per_session=5,
                                                            milestone_min_gap_secs=0))
    book = await _ingest(tmp_path, eng, bm, chapters=3)
    # mid-read: not flagged
    await eng.start_session(book["id"])  # reads a few, not to the end? sections_per_session=5 > 3
    await run_task(eng)
    # that actually finishes it (3 <= 5) and synthesizes -> not flagged
    assert bm.unfinalized_books() == []
    # now simulate the crash aftermath: drop synthesis, roll back to 'reading' at the end
    bm.conn.execute("DELETE FROM book_synthesis WHERE book_id = ?", (book["id"],))
    bm.set_status(book["id"], "reading", None)
    flagged = bm.unfinalized_books()
    assert len(flagged) == 1 and flagged[0]["id"] == book["id"]
    # finalizing clears it
    await eng.start_session(book["id"]); await run_task(eng)
    assert bm.unfinalized_books() == []


async def test_status_includes_new_unstarted_books(tmp_path):
    """reading_status must surface NEW (unstarted) books he added, not only reading/finished ones
    (regression: 'she thinks only the finished book is in her library')."""
    eng, bm, lib = make_engine(tmp_path)
    book = await _ingest(tmp_path, eng, bm, chapters=3)  # lands as status 'new'
    st = eng.status()
    titles = [b["title"] for b in st["books"]]
    assert book["title"] in titles
    assert any(b["status"] == "new" for b in st["books"])


def test_plan_session_extent_length_based():
    from modules.reading import plan_session_extent
    sec = lambda w, t="Ch", k="chapter": {"words": w, "title": t, "kind": k}
    # long chapters: ~2 hit the 3500 target
    longs = [sec(2000) for _ in range(6)]
    assert plan_session_extent(longs, 0, 3500, 20) == 2
    # short poems: reads many to reach the target (not a fixed 4)
    poems = [sec(300, "Poem") for _ in range(30)]
    assert plan_session_extent(poems, 0, 3500, 20) == 12          # ceil(3500/300)
    # safety cap wins for a wall of tiny sections
    assert plan_session_extent(poems, 0, 100000, 5) == 5
    # at least one section even if the first is huge
    assert plan_session_extent([sec(50000)], 0, 3500, 20) == 1


def test_plan_session_extent_does_not_stop_before_a_continuation():
    from modules.reading import plan_session_extent
    secs = [{"words": 2000, "title": "Chapter 1", "kind": "chapter"},
            {"words": 2000, "title": "Chapter 2", "kind": "chapter"},
            {"words": 2000, "title": "Chapter 2 (part 2)", "kind": "split"},
            {"words": 2000, "title": "Chapter 3", "kind": "chapter"}]
    # target hit after idx1, but idx2 is a '(part 2)' continuation -> read through it, stop before ch3
    assert plan_session_extent(secs, 0, 3500, 20) == 3
