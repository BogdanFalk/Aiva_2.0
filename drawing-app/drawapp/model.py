"""The drawing as plain data: no Qt, no I/O.

Everything that changes a Document goes through `Document.apply(op)` (ops.py).
That single mutation path is the seam for undo (RemoveStroke), saving, and
replaying another painter's ops (another avatar, another process).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from drawapp import ops

Point = tuple[float, float]
Listener = Callable[["Document", ops.Op], None]

# Default ink per painter. Extension point — colors: a palette/picker sets
# `tool.brush`; this is only the starting color for whoever holds the pointer.
AUTHOR_COLORS = {"aiva": "#7b4fd6", "saga": "#d64f8a"}


@dataclass(frozen=True)
class Brush:
    """What a stroke looks like. Colors are '#rrggbb' strings so they survive JSON."""
    color: str = "#202020"
    width: float = 4.0


@dataclass
class Stroke:
    id: str
    author: str
    brush: Brush
    points: list[Point] = field(default_factory=list)
    done: bool = False


class Document:
    """A fixed-size canvas holding strokes in paint order (later = on top).

    Extension point — layers: today there is one implicit layer (`strokes`).
    Layers become `list[Layer]` each holding strokes, with ops gaining a
    `layer_id`; renderers walk layers bottom-up.
    """

    def __init__(self, width: int = 1024, height: int = 768, background: str = "#ffffff"):
        self.width = width
        self.height = height
        self.background = background
        self.strokes: list[Stroke] = []
        self.revision = 0
        self._by_id: dict[str, Stroke] = {}
        self._listeners: list[Listener] = []
        self._before: list[Listener] = []

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Call `listener(doc, op)` after every applied op; returns an unsubscribe."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def subscribe_before(self, listener: Listener) -> Callable[[], None]:
        """Call `listener(doc, op)` just before each op is applied, so a clear can
        still see what it is about to wipe (storage.py keeps a copy). Returns an unsubscribe."""
        self._before.append(listener)
        return lambda: self._before.remove(listener)

    def stroke(self, stroke_id: str) -> Stroke | None:
        return self._by_id.get(stroke_id)

    def last_stroke(self, author: str) -> Stroke | None:
        """`author`'s newest finished stroke: what their undo takes back. A stroke still being
        drawn is never it, and nobody else's stroke ever is."""
        for s in reversed(self.strokes):
            if s.author == author and s.done:
                return s
        return None

    def apply(self, op: ops.Op) -> None:
        for listener in list(self._before):
            listener(self, op)
        if isinstance(op, ops.BeginStroke):
            if op.stroke_id in self._by_id:
                raise ValueError(f"stroke {op.stroke_id!r} already exists")
            s = Stroke(op.stroke_id, op.author, Brush(op.color, op.width), [op.point])
            self.strokes.append(s)
            self._by_id[s.id] = s
        elif isinstance(op, ops.ExtendStroke):
            s = self._open_stroke(op.stroke_id)
            s.points.extend(op.points)
        elif isinstance(op, ops.EndStroke):
            self._open_stroke(op.stroke_id).done = True
        elif isinstance(op, ops.ClearCanvas):
            self.strokes.clear()
            self._by_id.clear()
        elif isinstance(op, ops.RemoveStroke):
            s = self._by_id.get(op.stroke_id)
            if s is None:
                raise KeyError(op.stroke_id)
            if not s.done:  # someone is still drawing it; their next move would have nowhere to go
                raise ValueError(f"stroke {op.stroke_id!r} is still being drawn")
            self.strokes.remove(s)
            del self._by_id[s.id]
        else:
            raise TypeError(f"unknown op {op!r}")
        self.revision += 1
        for listener in list(self._listeners):
            listener(self, op)

    def _open_stroke(self, stroke_id: str) -> Stroke:
        s = self._by_id.get(stroke_id)
        if s is None:
            # A clear can land between someone's press and release (a second
            # painter); their remaining points simply have nowhere to go.
            raise KeyError(stroke_id)
        if s.done:
            raise ValueError(f"stroke {stroke_id!r} is already finished")
        return s
