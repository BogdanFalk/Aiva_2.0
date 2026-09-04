"""Integration: reading a book to the end feeds the becoming layer, once."""
import asyncio
import json
import math
import os
import sqlite3

import pytest

from modules import reading
from modules.becoming import Becoming
from modules.persona import baseline_traits
from modules.reading import BookMemory, ReadingEngine, StudyLimits


def vec_of(text):
    v = [0.0] * 512
    for tok in (text or "").lower().split():
        v[hash(tok) % 512] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


async def fake_embed(text):
    return vec_of(text)


class FakeReply:
    def __init__(self, text):
        self.text, self.tool_calls, self.prompt_tokens, self.completion_tokens = text, [], 400, 200


class FakeStudyLLM:
    async def __call__(self, messages, *, tools=None, json_mode=False, max_tokens=900, temperature=0.2):
        sys_p = messages[0]["content"]
        if sys_p.startswith(reading.CLOSE_READING_PROMPT[:40]):
            return FakeReply(json.dumps({
                "summary": "The narrator learns to sit with silence.", "key_ideas": ["Silence can be kind."],
                "quotes": [], "nuances": {"tone": "calm", "style": "spare", "subtext": "restraint",
                                          "devices": []}, "questions": [], "connections": [],
                "reflection": "It made me want to leave more pauses.", "vocabulary": [], "feeling": "still",
                "book_so_far": "A quiet book about silence.", "state": {"characters": [], "threads": [],
                                                                       "open_questions": [], "motifs": ["silence"]}}))
        if sys_p.startswith(reading.SESSION_WRAP_PROMPT[:40]):
            return FakeReply(json.dumps({"struck": "The pauses stayed with me.", "next_hope": "the ending",
                                         "mood": "calm"}))
        if sys_p.startswith(reading.SYNTHESIS_PROMPT[:40]):
            return FakeReply(json.dumps({
                "summary": "A calm book about the value of silence.", "arc": "quiets down", "themes": [],
                "people": [], "passages": [], "style": {"voice": "spare", "sentence": "short",
                                                        "devices": [], "what_to_borrow": ["let a pause land"]},
                "takeaways": ["I want to leave more silence."], "disagreements": [],
                "verdict": "Quietly lovely.", "talk_line": "It's a book that taught me to shut up sometimes."}))
        return FakeReply("{}")


class FakeBecomingLLM:
    async def __call__(self, messages, max_tokens=1200, temperature=0.2):
        return json.dumps({
            "lessons": [{"lesson": "I've learned that a pause can say more than a sentence.",
                         "kind": "framing", "domains": ["people"], "strength": 0.8,
                         "evidence": "leave more pauses"}],
            "traits": [{"text": "I've come to value the silences in a conversation since reading this.",
                        "kind": "value", "strength": 0.7, "evidence": "leave more pauses",
                        "why_compatible": "adds a facet to my curiosity"}],
            "voice": [], "wants_next": None})


def make_book(tmp_path, chapters=3):
    lib = tmp_path / "library"
    lib.mkdir(exist_ok=True)
    body = "\n\n".join(f"CHAPTER {i}\n\n" + " ".join([f"word{i}"] * 650) + "." for i in range(1, chapters + 1))
    f = lib / "silence.txt"
    f.write_text(body, encoding="utf-8")
    os.utime(str(f), (1000.0, 1000.0))
    return str(lib)


async def test_finishing_a_book_grows_her_once(tmp_path):
    lib = make_book(tmp_path)
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, category TEXT, fact TEXT, "
                 "created_ts TEXT, last_confirmed_ts TEXT)")
    bm = BookMemory(conn, embed=fake_embed)
    eng = ReadingEngine(bm, llm_call=FakeStudyLLM(), synth_call=FakeStudyLLM(), client=None,
                        library_dir=lib, model_name="gpt-5.6-luna",
                        limits=StudyLimits(announce_floor_secs=0, milestone_min_gap_secs=0), log=lambda m: None)
    becoming = Becoming(conn, embed=fake_embed, llm=FakeBecomingLLM(), remember=lambda f, c: 0,
                        baseline_traits=baseline_traits(), log=lambda m: None, engine=eng)

    announces = []

    async def announce(text):
        announces.append(text)

    refreshed = []
    eng.attach_voice(announce=announce, is_awake=lambda: True)
    becoming.attach_voice(announce=announce, is_awake=lambda: True,
                          refresh_prompt=lambda: refreshed.append(1))
    eng.on_book_finished = becoming.on_book_finished
    eng.on_session_end = becoming.on_session_end

    await eng.scan_once()
    book = bm.books()[0]
    res = await eng.start_session(book["id"], until_done=True)
    assert res["success"]
    await eng._task

    # the book is finished and she was changed — once
    assert bm.get_book(book["id"])["status"] == "finished"
    traits = becoming._traits()
    assert len(traits) == 1 and "silences" in traits[0]["text"]
    assert becoming._lessons() and "pause" in becoming._lessons()[0]["lesson"]
    # the finish note carried the trait clause exactly once, and the prompt was refreshed
    finish_notes = [a for a in announces if "just finished" in a]
    assert len(finish_notes) == 1 and "changed something in you" in finish_notes[0]
    assert refreshed
    # the trait is marked announced, so it never resurfaces as news
    assert traits[0]["announced"] == 1
    # and it shows up in the prompt block
    blocks = becoming.prompt_blocks(eng.library_status_lines())
    assert blocks.get("traits") and "silences" in blocks["traits"][0]
    assert any("Finished" in l for l in eng.library_status_lines())
