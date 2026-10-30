"""Her drawing hand (modules/artist.py): a whole picture from one side-model call, drawn in the
background, and her voice kept from drawing a long run of single strokes.

2026-10-27 he asked for "more detail" and she drew 50 strokes, one voice round trip each, in
silence, until the 120 s turn cap cut her off."""
import asyncio
import json
import math

import pytest

from modules import artist as A
from modules import drawing
from modules.drawing import DrawingBoard
from modules.house_link import EnoughSaid
from modules.tools import STROKE_RUN_MAX, TOOL_SCHEMAS, make_drawing_handlers


@pytest.fixture(autouse=True)
def _journal_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(drawing, "JOURNAL_DIR", str(tmp_path / "logs"))


class Reply:
    def __init__(self, text):
        self.text = text


class FakeLLM:
    def __init__(self, text):
        self.text, self.calls = text, []

    async def __call__(self, messages, *, json_mode=False, max_tokens=900, temperature=0.2):
        self.calls.append({"messages": messages, "json_mode": json_mode, "max_tokens": max_tokens})
        return Reply(self.text)


class FakeBoard:
    author = "aiva"

    def __init__(self, on_canvas=(), fail_after=None):
        self.strokes, self.drawn, self.fail_after = list(on_canvas), [], fail_after

    async def on_canvas(self):
        return self.strokes

    async def stroke(self, points, color=None, width=None):
        if self.fail_after is not None and len(self.drawn) >= self.fail_after:
            return {"success": False, "error": "couldn't reach the drawing app"}
        self.drawn.append((points, color, width))
        return {"success": True, "stroke_id": f"aiva-{len(self.drawn)}"}


PICTURE = json.dumps({"strokes": [{"ellipse": [500, 300, 40, 40], "color": "#f4d35e", "width": 3},
                                  {"points": [[100, 700], [900, 700]]},
                                  {"bogus": True}],
                      "said": "I gave them a moon and a horizon."})


# --- pure helpers -------------------------------------------------------------------------------

def test_the_canvas_brief_shows_who_drew_what_and_where_in_a_few_points():
    assert "blank" in A.canvas_brief([])
    line = {"author": "user", "color": "#000000", "width": 4,
            "points": [[float(i), float(2 * i)] for i in range(100)]}
    brief = A.canvas_brief([line])
    assert brief.startswith("user #000000 w4: box 0,0-99,198 through 0,0 ")
    assert brief.endswith(" 99,198") and len(brief.split("through ")[1].split()) <= A.BRIEF_POINTS + 1
    many = A.canvas_brief([line] * 130, limit=120)
    assert many.startswith("(10 older strokes not listed)") and len(many.splitlines()) == 121


def test_shapes_expand_onto_the_canvas():
    pts, color, width = A.expand({"ellipse": [500, 300, 40, 40], "color": "#f4d35e", "width": 3})
    assert color == "#f4d35e" and width == 3.0
    assert pts[0] == pytest.approx(pts[-1]) and all(math.dist(p, (500, 300)) == pytest.approx(40) for p in pts)
    half, _, _ = A.expand({"ellipse": [500, 300, 40, 20], "arc": [0, 180]})
    assert half[0] == pytest.approx([540, 300]) and half[-1] == pytest.approx([460, 300])
    assert all(y >= 300 - 1e-9 for _, y in half)                          # 0..180 with y DOWN: the lower half
    off, color, width = A.expand({"points": [[-50, 10], [2000, 900]], "color": "red", "width": 999})
    assert off == [[0.0, 10.0], [1024.0, 768.0]] and color is None and width == A.MAX_WIDTH
    long, _, _ = A.expand({"points": [[i % 1000, 5] for i in range(3000)]})
    assert len(long) == A.MAX_POINTS and long[0] == [0.0, 5.0] and long[-1] == [999.0, 5.0]
    for bad in ({"ellipse": [1, 2, 3]}, {"ellipse": [1, 2, 3, 4], "arc": [10, 10]}, {"points": []},
                {"points": [["a", 1]]}, {"line": [[1, 1]]}, "dot", {"points": [[True, 1]]}):
        assert A.expand(bad) is None, bad


def test_a_reply_parses_tolerantly_and_keeps_only_drawable_shapes():
    pic = A.parse_picture(Reply("Here you go:\n```json\n" + PICTURE + "\n```"))
    assert len(pic["strokes"]) == 2 and pic["said"] == "I gave them a moon and a horizon."
    assert len(A.parse_picture(Reply(PICTURE), max_strokes=1)["strokes"]) == 1
    for junk in (Reply("no json"), Reply('{"strokes": []}'), Reply('{"strokes": [{"bogus": 1}]}'),
                 Reply("{broken"), Reply(None), Reply('{"said": "hi"}')):
        assert A.parse_picture(junk) is None


def test_the_prompt_carries_the_idea_the_canvas_and_her_ink_with_its_json_intact():
    msgs = A.artist_messages("Saga", "a moon above the cats", [], "#c0397a")
    system = msgs[0]["content"]
    assert "Saga's drawing hand" in system and "IDEA: a moon above the cats" in system
    assert "Saga's own ink, #c0397a" in system and "1024 pixels wide and 768 tall" in system
    assert '{"points": [[x, y], ...]}' in system and '{"strokes": [...], "said":' in system
    assert "{name}" not in system and "{brief}" not in system and "blank" in system


# --- the background job ---------------------------------------------------------------------------

async def test_she_hands_over_the_idea_and_hears_when_it_is_drawn():
    board, llm, notes = FakeBoard(on_canvas=[{"author": "saga", "color": "#c0397a", "width": 4,
                                              "points": [[10, 10], [20, 20]]}]), FakeLLM(PICTURE), []

    async def announce(text):
        notes.append(text)

    artist = A.Artist(board, llm, name="Aiva", announce=announce, log=lambda m: None, pace_s=0)
    out = artist.start("  a moon and a horizon  ")
    assert out["success"] and "background" in out["status"] and artist.busy     # back at once
    again = artist.start("a sun")
    assert again["success"] is False and again["drawing"] == "a moon and a horizon"
    await artist._job
    assert not artist.busy and len(board.drawn) == 2 and board.drawn[0][1] == "#f4d35e"
    call = llm.calls[0]
    assert call["json_mode"] and call["max_tokens"] == A.MAX_TOKENS
    assert "saga #c0397a w4: box 10,10-20,20" in call["messages"][0]["content"]   # it saw the canvas
    assert len(notes) == 1 and '"a moon and a horizon"' in notes[0]
    assert "I gave them a moon and a horizon." in notes[0] and "no stroke counts" in notes[0]
    assert artist.start("")["success"] is False


async def test_a_reply_with_nothing_drawable_or_a_dead_window_is_said_plainly():
    notes = []

    async def announce(text):
        notes.append(text)

    artist = A.Artist(FakeBoard(), FakeLLM("sorry, I can't"), announce=announce, log=lambda m: None, pace_s=0)
    artist.start("a cat")
    await artist._job
    assert "didn't happen" in notes[-1] and "nothing it could draw" in notes[-1]

    board = FakeBoard(fail_after=1)
    many = json.dumps({"strokes": [{"points": [[i, i]]} for i in range(10)], "said": ""})
    artist = A.Artist(board, FakeLLM(many), announce=announce, log=lambda m: None, pace_s=0)
    artist.start("dots")
    await artist._job
    assert len(board.drawn) == 1                          # stopped after three misses in a row
    assert "It stopped partway: couldn't reach the drawing app." in notes[-1]


# --- her tools ---------------------------------------------------------------------------------

class Params:
    def __init__(self, llm=None, **arguments):
        self.arguments, self.llm, self.result, self.props = arguments, llm, None, None

    async def result_callback(self, res, *, properties=None):
        self.result, self.props = res, properties


async def test_draw_picture_returns_at_once_and_needs_no_second_line():
    artist = A.Artist(FakeBoard(), FakeLLM(PICTURE), log=lambda m: None, pace_s=0)
    h = make_drawing_handlers(FakeBoard(), artist)
    assert "draw_picture" in h and "draw_picture" not in make_drawing_handlers(FakeBoard())
    p = Params(idea="a moon")
    await h["draw_picture"](p)
    assert p.result["success"] and isinstance(p.props, EnoughSaid)       # she said her line with it
    p = Params(idea="another")
    await h["draw_picture"](p)
    assert p.result["success"] is False and p.props is None
    await artist._job
    schema = next(s for s in TOOL_SCHEMAS.standard_tools if s.name == "draw_picture")
    assert schema.required == ["idea"] and "background" in schema.description


async def test_her_voice_is_sent_to_draw_picture_after_a_run_of_single_strokes():
    board = FakeBoard()
    h = make_drawing_handlers(board, A.Artist(board, FakeLLM(PICTURE), log=lambda m: None))
    for _ in range(STROKE_RUN_MAX):
        p = Params(llm=object(), points=[[1, 1]])
        await h["draw_stroke"](p)
        assert p.result["success"]
    p = Params(llm=object(), points=[[1, 1]])
    await h["draw_stroke"](p)
    assert p.result["success"] is False and "draw_picture" in p.result["note"]
    assert len(board.drawn) == STROKE_RUN_MAX
    for _ in range(STROKE_RUN_MAX + 3):                   # a task's strokes (no llm) are never stopped
        p = Params(points=[[1, 1]])
        await h["draw_stroke"](p)
        assert p.result["success"]
    plain = make_drawing_handlers(FakeBoard())            # no hand to send her to: no limit either
    for _ in range(STROKE_RUN_MAX + 3):
        p = Params(llm=object(), points=[[1, 1]])
        await plain["draw_stroke"](p)
        assert p.result["success"]


# --- what's on the canvas, read from its autosave ------------------------------------------------

async def test_the_canvas_is_read_from_its_autosave_without_opening_a_window(tmp_path):
    board = DrawingBoard(launch=lambda **k: pytest.fail("never opens a window"), save_dir=str(tmp_path))
    assert await board.on_canvas() == []
    (tmp_path / "recovery").mkdir()
    (tmp_path / "recovery" / "shared-autosave.aivadraw").write_text(json.dumps({
        "format": "aivadraw", "version": 1, "width": 1024, "height": 768, "background": "#ffffff",
        "strokes": [{"id": "user-1", "author": "user", "color": "#000000", "width": 4,
                     "points": [[1, 2], [3, 4]]}]}), encoding="utf-8")
    assert await board.on_canvas() == [{"author": "user", "color": "#000000", "width": 4.0,
                                     "points": [[1.0, 2.0], [3.0, 4.0]]}]
    (tmp_path / "recovery" / "shared-autosave.aivadraw").write_text("{damaged", encoding="utf-8")
    assert await board.on_canvas() == []


async def test_a_real_canvas_gets_the_whole_picture_and_reads_it_back(tmp_path):
    board = DrawingBoard(author="aiva", extra_env={"QT_QPA_PLATFORM": "offscreen"}, save_dir=str(tmp_path))
    notes = []

    async def announce(text):
        notes.append(text)

    try:
        artist = A.Artist(board, FakeLLM(PICTURE), announce=announce, log=lambda m: None, pace_s=0)
        assert artist.start("a moon and a horizon")["success"]
        await asyncio.wait_for(artist._job, 60)
        assert "Your drawing is done" in notes[-1], notes
        for _ in range(50):                               # the window autosaves after each stroke
            strokes = await board.on_canvas()
            if len(strokes) == 2:
                break
            await asyncio.sleep(0.1)
        assert [s["author"] for s in strokes] == ["aiva", "aiva"] and strokes[0]["color"] == "#f4d35e"
        assert strokes[1]["points"] == [[100.0, 700.0], [900.0, 700.0]]
    finally:
        await board.close()
