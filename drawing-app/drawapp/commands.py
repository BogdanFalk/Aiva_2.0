"""Drawing by command instead of by pointer. No Qt.

A command is one JSON object (one line on the wire, see server.py):

    {"cmd": "stroke", "author": "aiva", "points": [[10, 10], [80, 40]], "color": "#7b4fd6", "width": 6}
    {"cmd": "clear", "author": "saga"}
    {"cmd": "undo", "author": "aiva"}                     # takes back aiva's newest finished stroke
    {"cmd": "save", "author": "aiva", "name": "cat"}      # "name" optional (see storage.py)
    {"cmd": "open", "author": "aiva", "name": "cat"}      # "name" optional: the newest drawing
    {"cmd": "drawings"}                                   # what's saved, newest first
    {"cmd": "status"}                                     # this window: pid, author, revision

`author` defaults to "user"; a stroke's `color` defaults to the author's ink
(AUTHOR_COLORS) and `width` to the default brush. Unknown fields are refused,
so a typo ("colour") is an error rather than silently ignored.

`parse()` checks everything that doesn't depend on the canvas. `CommandRunner`
checks the points against the document size, then drives a BrushTool with
press/move/release — so a command stroke is exactly a mouse stroke: same ops,
same stroke ids, same repaint. Nothing here moves the real mouse or keyboard.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Union

from drawapp import ops
from drawapp.model import AUTHOR_COLORS, Brush, Document, Point
from drawapp.tools import BrushTool

DEFAULT_PORT = 8770  # the wire is server.py; the Aiva house listens on 8761 and her dashboard on 8760
MAX_POINTS = 10_000
MAX_WIDTH = 200.0
_COLOR = re.compile(r"#[0-9a-fA-F]{6}")
_AUTHOR = re.compile(r"[a-z][a-z0-9_]{0,31}")  # it ends up in stroke ids ("aiva-1f3c...")


class CommandError(ValueError):
    """The command is malformed or doesn't fit the canvas; nothing was applied."""
    code = "invalid"


class UnsupportedCommand(CommandError):
    """Well-formed, but a "cmd" this version doesn't know."""
    code = "unsupported"


@dataclass(frozen=True)
class StrokeCommand:
    author: str
    points: tuple[Point, ...]
    color: str
    width: float


@dataclass(frozen=True)
class ClearCommand:
    author: str


@dataclass(frozen=True)
class UndoCommand:
    author: str


@dataclass(frozen=True)
class SaveCommand:
    author: str
    name: str | None


@dataclass(frozen=True)
class OpenCommand:
    author: str
    name: str | None


@dataclass(frozen=True)
class DrawingsCommand:
    pass


@dataclass(frozen=True)
class StatusCommand:
    pass


Command = Union[StrokeCommand, ClearCommand, UndoCommand, SaveCommand, OpenCommand, DrawingsCommand,
                StatusCommand]

_FIELDS = {"stroke": {"cmd", "author", "points", "color", "width"}, "clear": {"cmd", "author"},
           "undo": {"cmd", "author"},
           "save": {"cmd", "author", "name"}, "open": {"cmd", "author", "name"},
           "drawings": {"cmd"}, "status": {"cmd"}}
MAX_NAME = 80


def parse(raw: str | bytes | dict) -> Command:
    """A JSON text or an already-decoded dict -> a validated Command, or CommandError."""
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except (ValueError, RecursionError) as e:  # bad JSON, bad UTF-8, absurd nesting
            raise CommandError(f"not valid JSON: {e}") from None
    if not isinstance(raw, dict):
        raise CommandError("a command must be a JSON object")
    cmd = raw.get("cmd")
    if not isinstance(cmd, str):
        raise CommandError(f'missing "cmd" (one of: {", ".join(_FIELDS)})')
    if cmd not in _FIELDS:
        raise UnsupportedCommand(f"unsupported cmd {cmd[:40]!r} (supported: {', '.join(_FIELDS)})")
    extra = set(raw) - _FIELDS[cmd]
    if extra:
        raise CommandError(f"unknown field(s) for {cmd}: {', '.join(sorted(map(str, extra)))[:200]}")

    author = raw.get("author", "user")
    if not isinstance(author, str) or not _AUTHOR.fullmatch(author):
        raise CommandError('"author" must be lowercase letters, digits or _ (e.g. "aiva")')
    if cmd == "clear":
        return ClearCommand(author)
    if cmd == "undo":
        return UndoCommand(author)
    if cmd in ("save", "open"):
        name = raw.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()
                                 or len(name) > MAX_NAME or not name.isprintable()):
            raise CommandError(f'"name" must be a short line of text (at most {MAX_NAME} characters)')
        return (SaveCommand if cmd == "save" else OpenCommand)(author, name)
    if cmd == "drawings":
        return DrawingsCommand()
    if cmd == "status":
        return StatusCommand()

    color = raw.get("color", AUTHOR_COLORS.get(author, Brush.color))
    if not isinstance(color, str) or not _COLOR.fullmatch(color):
        raise CommandError('"color" must be "#rrggbb"')
    width = raw.get("width", Brush.width)
    if not _finite(width) or not 0 < width <= MAX_WIDTH:
        raise CommandError(f'"width" must be a number in (0, {MAX_WIDTH:g}]')
    return StrokeCommand(author, _points(raw.get("points")), color, float(width))


def _points(value: Any) -> tuple[Point, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise CommandError('"points" must be a non-empty list of [x, y] pairs')
    if len(value) > MAX_POINTS:
        raise CommandError(f'too many points ({len(value)} > {MAX_POINTS}); split the stroke')
    out = []
    for i, p in enumerate(value):
        if not (isinstance(p, (list, tuple)) and len(p) == 2 and all(map(_finite, p))):
            raise CommandError(f"points[{i}] must be [x, y] with finite numbers, got {p!r:.60}")
        out.append((float(p[0]), float(p[1])))
    return tuple(out)


def _finite(v: Any) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:  # an int too big for a float
        return False


class CommandRunner:
    """Applies commands to one document, with one BrushTool per author.

    A command is applied whole inside one call, so on the GUI thread it never
    interleaves with the events of a mouse gesture; and it uses its own tools,
    so a stroke being drawn by hand at that moment is left alone.

    `session` (a storage.Session) is what save/open/drawings act on; without
    one those commands are refused as unsupported.
    """

    def __init__(self, document: Document, session=None):
        self.document = document
        self.session = session
        self._tools: dict[str, BrushTool] = {}

    def run(self, command: Command) -> dict:
        """Apply a parsed command; returns JSON-ready details. CommandError if it doesn't fit."""
        doc = self.document
        if isinstance(command, ClearCommand):
            doc.apply(ops.ClearCanvas(command.author))
            return {"revision": doc.revision}
        if isinstance(command, UndoCommand):
            return self.undo(command.author)
        if isinstance(command, StatusCommand):
            session = self.session
            return {"pid": os.getpid(), "author": session.author if session else None,
                    "revision": doc.revision, "strokes": len(doc.strokes),
                    "save_dir": session.save_dir if session else None,
                    "current": session.current_path if session else None,
                    "unsaved": session.dirty if session else None}
        if isinstance(command, (SaveCommand, OpenCommand, DrawingsCommand)):
            return self._storage(command)
        if not isinstance(command, StrokeCommand):
            raise UnsupportedCommand(f"unsupported command {type(command).__name__}")

        # Check every point before the first op, so a bad stroke leaves no half-stroke behind.
        for x, y in command.points:
            if not (0 <= x <= doc.width and 0 <= y <= doc.height):
                raise CommandError(
                    f"point ({x:g}, {y:g}) is outside the {doc.width}x{doc.height} canvas")
        tool = self._tools.get(command.author)
        if tool is None:
            tool = self._tools[command.author] = BrushTool(doc, command.author)
        tool.brush = Brush(command.color, command.width)
        first, *rest = command.points
        tool.press(*first)
        stroke_id = doc.strokes[-1].id
        for x, y in rest:
            tool.move(x, y)
        tool.release(*command.points[-1])
        return {"stroke_id": stroke_id, "revision": doc.revision}

    def undo(self, author: str) -> dict:
        """Take back `author`'s newest finished stroke. Only ever their own: a sister's undo never
        touches the other sister's strokes or the user's. Nothing of theirs left is not an error:
        `"stroke_id": null` says nothing was taken back."""
        doc = self.document
        stroke = doc.last_stroke(author)
        if stroke is not None:
            doc.apply(ops.RemoveStroke(stroke.id, author))
        return {"stroke_id": stroke.id if stroke else None, "revision": doc.revision,
                "remaining": sum(s.author == author for s in doc.strokes)}

    def _storage(self, command: Command) -> dict:
        session = self.session
        if session is None:
            raise UnsupportedCommand("this window has no save folder")
        if isinstance(command, DrawingsCommand):
            return session.drawings()
        if isinstance(command, SaveCommand):
            return {**session.save(command.name, author=command.author), "revision": self.document.revision}
        result = session.open(session.resolve(command.name), author=command.author)
        return {**result, "revision": self.document.revision}

    def handle(self, raw: str | bytes | dict) -> dict:
        """parse + run, as a reply: {"ok": true, ...} or {"ok": false, "code", "error"}."""
        try:
            return {"ok": True, **self.run(parse(raw))}
        except CommandError as e:
            return {"ok": False, "code": e.code, "error": str(e)}
