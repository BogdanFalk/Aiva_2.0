"""Document, ops and the brush tool — pure Python, no Qt."""
import dataclasses
import json

import pytest

from drawapp import ops
from drawapp.model import Brush, Document
from drawapp.tools import BrushTool


def test_brush_tool_makes_one_finished_stroke_per_gesture():
    doc = Document()
    tool = BrushTool(doc, "aiva", Brush("#ff0000", 3.0))
    tool.press(10, 10)
    tool.move(20, 10)
    tool.move(30, 15)
    tool.release(30, 15)

    assert len(doc.strokes) == 1
    s = doc.strokes[0]
    assert s.author == "aiva" and s.brush == Brush("#ff0000", 3.0)
    assert s.points == [(10, 10), (20, 10), (30, 15)]
    assert s.done and not tool.active


def test_moves_without_press_and_tiny_moves_are_ignored():
    doc = Document()
    tool = BrushTool(doc)
    tool.move(5, 5)
    assert doc.strokes == []
    tool.press(5, 5)
    tool.move(5.2, 5.2)  # below MIN_STEP
    tool.release(5.2, 5.2)
    assert doc.strokes[0].points == [(5, 5)]


def test_clear_empties_the_canvas_and_notifies():
    doc = Document()
    seen = []
    doc.subscribe(lambda d, op: seen.append(type(op).__name__))
    tool = BrushTool(doc)
    tool.press(1, 1)
    tool.release(9, 9)
    stroke_id = doc.strokes[0].id
    doc.apply(ops.ClearCanvas("user"))

    assert doc.strokes == [] and doc.stroke(stroke_id) is None
    assert seen == ["BeginStroke", "ExtendStroke", "EndStroke", "ClearCanvas"]
    assert doc.revision == 4


def test_clear_mid_stroke_drops_the_rest_of_that_gesture():
    doc = Document()
    aiva = BrushTool(doc, "aiva")
    aiva.press(0, 0)
    doc.apply(ops.ClearCanvas("saga"))  # her sister clears while she is drawing
    aiva.move(50, 50)
    aiva.release(60, 60)
    assert doc.strokes == [] and not aiva.active


def test_two_painters_interleave_on_one_document():
    doc = Document()
    aiva, saga = BrushTool(doc, "aiva"), BrushTool(doc, "saga")
    aiva.press(0, 0)
    saga.press(100, 100)
    aiva.move(10, 0)
    saga.move(110, 100)
    saga.release(120, 100)
    aiva.release(20, 0)
    assert [s.author for s in doc.strokes] == ["aiva", "saga"]
    assert doc.strokes[0].points == [(0, 0), (10, 0), (20, 0)]
    assert doc.strokes[1].points == [(100, 100), (110, 100), (120, 100)]


def test_finished_or_duplicate_strokes_are_rejected():
    doc = Document()
    doc.apply(ops.BeginStroke("s1", "user", "#000000", 2.0, (0, 0)))
    with pytest.raises(ValueError):
        doc.apply(ops.BeginStroke("s1", "user", "#000000", 2.0, (0, 0)))
    doc.apply(ops.EndStroke("s1"))
    with pytest.raises(ValueError):
        doc.apply(ops.ExtendStroke("s1", ((1, 1),)))


def test_ops_are_json_ready_and_replay_to_the_same_document():
    """The op log is the future save format / collaboration message."""
    log = []
    original = Document()
    original.subscribe(lambda d, op: log.append(op))
    tool = BrushTool(original, "saga")
    tool.press(1, 2)
    tool.move(30, 40)
    tool.release(30, 40)

    wire = json.dumps([{"type": type(op).__name__, **dataclasses.asdict(op)} for op in log])
    replica = Document()
    for d in json.loads(wire):
        cls = getattr(ops, d.pop("type"))
        fixed = {k: tuple(map(tuple, v)) if k == "points" else tuple(v) if k == "point" else v
                 for k, v in d.items()}
        replica.apply(cls(**fixed))
    assert [(s.id, s.author, s.points, s.done) for s in replica.strokes] == \
           [(s.id, s.author, s.points, s.done) for s in original.strokes]


def _finished(doc, author, x=10.0):
    tool = BrushTool(doc, author)
    tool.press(x, 10)
    tool.release(x + 20, 10)
    return doc.strokes[-1].id


def test_remove_stroke_takes_one_finished_stroke_off_and_bumps_the_revision():
    doc = Document()
    a, b = _finished(doc, "aiva"), _finished(doc, "aiva", 50)
    seen = []
    doc.subscribe(lambda _doc, op: seen.append(op))
    rev = doc.revision
    doc.apply(ops.RemoveStroke(a, "aiva"))
    assert [s.id for s in doc.strokes] == [b] and doc.stroke(a) is None
    assert doc.revision == rev + 1 and seen == [ops.RemoveStroke(a, "aiva")]
    with pytest.raises(KeyError):
        doc.apply(ops.RemoveStroke(a, "aiva"))                  # already gone


def test_last_stroke_is_the_authors_newest_finished_one_only():
    doc = Document()
    assert doc.last_stroke("aiva") is None
    mine = _finished(doc, "aiva")
    _finished(doc, "saga", 50)                                  # newer, but her sister's
    drawing = BrushTool(doc, "aiva")
    drawing.press(90, 90)                                       # still under the pointer
    assert doc.last_stroke("aiva").id == mine
    with pytest.raises(ValueError, match="still being drawn"):
        doc.apply(ops.RemoveStroke(doc.strokes[-1].id, "aiva"))
    drawing.release(95, 95)
    assert doc.last_stroke("aiva").id == doc.strokes[-1].id
