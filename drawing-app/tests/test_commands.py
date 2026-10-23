"""Drawing commands: parsing, validation and dispatch into the document — pure Python, no Qt."""
import json

import pytest

from drawapp import commands
from drawapp.commands import (ClearCommand, CommandError, CommandRunner, StrokeCommand,
                              UnsupportedCommand, parse)
from drawapp.model import AUTHOR_COLORS, Brush, Document
from drawapp.tools import BrushTool


def test_parse_a_full_stroke_from_a_json_line():
    line = json.dumps({"cmd": "stroke", "author": "saga", "points": [[1, 2], [3.5, 4]],
                       "color": "#00ff00", "width": 3})
    assert parse(line) == StrokeCommand("saga", ((1.0, 2.0), (3.5, 4.0)), "#00ff00", 3.0)
    assert parse(line.encode()) == parse(line)


def test_parse_defaults_author_color_and_width():
    assert parse({"cmd": "stroke", "points": [[0, 0]]}) == \
        StrokeCommand("user", ((0.0, 0.0),), Brush.color, Brush.width)
    assert parse({"cmd": "stroke", "author": "aiva", "points": [[0, 0]]}).color == AUTHOR_COLORS["aiva"]
    assert parse('{"cmd": "clear"}') == ClearCommand("user")
    assert parse({"cmd": "clear", "author": "saga"}) == ClearCommand("saga")


@pytest.mark.parametrize("raw, message", [
    ("{not json", "not valid JSON"),
    (b"\xff\xfe", "not valid JSON"),
    ('["stroke"]', "JSON object"),
    ("42", "JSON object"),
    ({}, 'missing "cmd"'),
    ({"cmd": 7}, 'missing "cmd"'),
    ({"cmd": "stroke", "points": [[0, 0]], "colour": "#000000"}, "unknown field"),
    ({"cmd": "clear", "points": [[0, 0]]}, "unknown field"),
    ({"cmd": "clear", "author": "Aiva"}, '"author"'),
    ({"cmd": "clear", "author": "a b"}, '"author"'),
    ({"cmd": "clear", "author": 3}, '"author"'),
    ({"cmd": "stroke", "points": [[0, 0]], "color": "red"}, '"color"'),
    ({"cmd": "stroke", "points": [[0, 0]], "color": "#12345"}, '"color"'),
    ({"cmd": "stroke", "points": [[0, 0]], "width": 0}, '"width"'),
    ({"cmd": "stroke", "points": [[0, 0]], "width": -2}, '"width"'),
    ({"cmd": "stroke", "points": [[0, 0]], "width": 500}, '"width"'),
    ({"cmd": "stroke", "points": [[0, 0]], "width": "4"}, '"width"'),
    ({"cmd": "stroke", "points": [[0, 0]], "width": True}, '"width"'),
    ('{"cmd": "stroke", "points": [[0, 0]], "width": NaN}', '"width"'),
    ({"cmd": "stroke"}, '"points"'),
    ({"cmd": "stroke", "points": []}, '"points"'),
    ({"cmd": "stroke", "points": "0,0"}, '"points"'),
    ({"cmd": "stroke", "points": [[0, 0, 0]]}, "points[0]"),
    ({"cmd": "stroke", "points": [[0, 0], [1]]}, "points[1]"),
    ({"cmd": "stroke", "points": [["1", 2]]}, "points[0]"),
    ({"cmd": "stroke", "points": [[None, 2]]}, "points[0]"),
    ({"cmd": "stroke", "points": [[False, 2]]}, "points[0]"),
    ('{"cmd": "stroke", "points": [[Infinity, 2]]}', "points[0]"),
    ({"cmd": "stroke", "points": [[10 ** 400, 2]]}, "points[0]"),
    ({"cmd": "stroke", "points": [[0, 0]] * (commands.MAX_POINTS + 1)}, "too many points"),
])
def test_malformed_commands_are_refused_with_a_reason(raw, message):
    with pytest.raises(CommandError) as err:
        parse(raw)
    assert message in str(err.value)
    assert not isinstance(err.value, UnsupportedCommand)


def test_absurdly_nested_json_is_refused_not_a_recursion_crash():
    # Not a parametrize case: pytest puts the test id in an env var, capped at 32 KB on Windows.
    with pytest.raises(CommandError, match="not valid JSON"):
        parse("[" * 100_000)


@pytest.mark.parametrize("cmd", ["circle", "redo", "STROKE", ""])
def test_unknown_cmds_are_unsupported_not_malformed(cmd):
    with pytest.raises(UnsupportedCommand, match="supported: stroke, clear"):
        parse({"cmd": cmd})


def test_a_stroke_command_becomes_one_finished_stroke():
    doc = Document(200, 100)
    seen = []
    doc.subscribe(lambda d, op: seen.append(type(op).__name__))
    result = CommandRunner(doc).run(parse(
        {"cmd": "stroke", "author": "aiva", "points": [[10, 10], [50, 20], [90, 30]], "width": 5}))

    assert len(doc.strokes) == 1
    s = doc.strokes[0]
    assert (s.author, s.brush, s.points, s.done) == \
        ("aiva", Brush(AUTHOR_COLORS["aiva"], 5.0), [(10, 10), (50, 20), (90, 30)], True)
    assert s.id.startswith("aiva-")
    assert result == {"stroke_id": s.id, "revision": doc.revision}
    # The same op sequence the mouse produces through BrushTool.
    assert seen == ["BeginStroke", "ExtendStroke", "ExtendStroke", "EndStroke"]


def test_a_single_point_is_a_dot_and_each_command_is_its_own_stroke():
    doc = Document(100, 100)
    runner = CommandRunner(doc)
    runner.run(parse({"cmd": "stroke", "author": "saga", "points": [[5, 5]]}))
    runner.run(parse({"cmd": "stroke", "author": "saga", "points": [[5, 5]], "color": "#000000"}))
    assert [(s.points, s.done) for s in doc.strokes] == [([(5, 5)], True), ([(5, 5)], True)]
    assert doc.strokes[0].id != doc.strokes[1].id
    assert [s.brush.color for s in doc.strokes] == [AUTHOR_COLORS["saga"], "#000000"]


def test_clear_command_empties_the_canvas():
    doc = Document(100, 100)
    runner = CommandRunner(doc)
    runner.run(parse({"cmd": "stroke", "points": [[1, 1], [9, 9]]}))
    assert runner.run(parse({"cmd": "clear", "author": "saga"})) == {"revision": doc.revision}
    assert doc.strokes == []


@pytest.mark.parametrize("point", [[-1, 5], [5, -0.5], [100.5, 5], [5, 51]])
def test_points_off_the_canvas_are_refused_before_anything_is_drawn(point):
    doc = Document(100, 50)
    with pytest.raises(CommandError, match="outside the 100x50 canvas"):
        CommandRunner(doc).run(parse({"cmd": "stroke", "points": [[10, 10], point]}))
    assert doc.strokes == [] and doc.revision == 0


def test_edges_of_the_canvas_are_inside():
    doc = Document(100, 50)
    CommandRunner(doc).run(parse({"cmd": "stroke", "points": [[0, 0], [100, 50]]}))
    assert doc.strokes[0].points == [(0, 0), (100, 50)]


def test_a_command_does_not_disturb_a_stroke_being_drawn_by_hand():
    doc = Document(200, 200)
    hand = BrushTool(doc, "aiva")
    hand.press(10, 10)
    hand.move(20, 10)
    CommandRunner(doc).run(parse({"cmd": "stroke", "author": "aiva", "points": [[100, 100], [150, 150]]}))
    hand.move(30, 10)
    hand.release(30, 10)

    by_hand, by_command = doc.strokes
    assert by_hand.points == [(10, 10), (20, 10), (30, 10)] and by_hand.done
    assert by_command.points == [(100, 100), (150, 150)] and by_command.done
    assert by_hand.id != by_command.id


def test_unsupported_command_objects_are_refused_by_the_runner():
    with pytest.raises(UnsupportedCommand):
        CommandRunner(Document()).run(object())


def test_handle_turns_errors_into_replies():
    doc = Document(100, 100)
    runner = CommandRunner(doc)
    ok = runner.handle('{"cmd": "stroke", "points": [[1, 1], [5, 5]]}')
    assert ok["ok"] is True and ok["stroke_id"] == doc.strokes[0].id

    bad = runner.handle("nope")
    assert (bad["ok"], bad["code"]) == (False, "invalid") and "not valid JSON" in bad["error"]
    assert runner.handle('{"cmd": "fill"}')["code"] == "unsupported"
    off = runner.handle({"cmd": "stroke", "points": [[500, 5]]})
    assert off["ok"] is False and off["code"] == "invalid" and "outside" in off["error"]
    assert len(doc.strokes) == 1


def test_undo_takes_back_only_that_authors_strokes_newest_first():
    runner = CommandRunner(Document(200, 100))
    a1 = runner.handle({"cmd": "stroke", "author": "aiva", "points": [[1, 1], [9, 9]]})["stroke_id"]
    s1 = runner.handle({"cmd": "stroke", "author": "saga", "points": [[5, 5], [50, 50]]})["stroke_id"]
    a2 = runner.handle({"cmd": "stroke", "author": "aiva", "points": [[20, 20], [30, 30]]})["stroke_id"]
    assert parse({"cmd": "undo", "author": "aiva"}) == commands.UndoCommand("aiva")
    assert parse({"cmd": "undo"}) == commands.UndoCommand("user")

    reply = runner.handle({"cmd": "undo", "author": "aiva"})
    assert reply["ok"] and reply["stroke_id"] == a2 and reply["remaining"] == 1
    assert runner.handle({"cmd": "undo", "author": "aiva"})["stroke_id"] == a1
    empty = runner.handle({"cmd": "undo", "author": "aiva"})
    assert empty["ok"] and empty["stroke_id"] is None and empty["remaining"] == 0
    assert [s.id for s in runner.document.strokes] == [s1]       # her sister's stroke is untouched
    assert runner.handle({"cmd": "undo", "author": "aiva", "points": [[0, 0]]})["code"] == "invalid"
