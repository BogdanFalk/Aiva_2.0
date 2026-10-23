"""Tools turn pointer gestures into ops. No Qt here either.

A tool gets press/move/release in document coordinates and applies ops to the
document it was given. The canvas widget only routes pointer events to the
current tool, so a new tool (eraser, line, fill, ...) is a new class with
these three methods — nothing in the widget changes. A non-mouse painter (an
avatar driven by code) can drive the same tool with the same calls.
"""
from __future__ import annotations

import math
import uuid
from typing import Protocol

from drawapp import ops
from drawapp.model import Brush, Document


class Tool(Protocol):
    def press(self, x: float, y: float) -> None: ...
    def move(self, x: float, y: float) -> None: ...
    def release(self, x: float, y: float) -> None: ...


class BrushTool:
    """Freehand brush: one stroke per press..release."""

    # Pointer events arrive far denser than the eye needs; points closer than
    # this (document px) to the previous one are dropped.
    MIN_STEP = 1.0

    def __init__(self, document: Document, author: str = "user", brush: Brush | None = None):
        self.document = document
        self.author = author
        self.brush = brush or Brush()
        self._stroke_id: str | None = None
        self._last: tuple[float, float] | None = None

    @property
    def active(self) -> bool:
        return self._stroke_id is not None

    def press(self, x: float, y: float) -> None:
        if self.active:  # a lost release (e.g. focus change) — close the old stroke first
            self.release(x, y)
        self._stroke_id = f"{self.author}-{uuid.uuid4().hex[:12]}"
        self._last = (x, y)
        self.document.apply(ops.BeginStroke(
            self._stroke_id, self.author, self.brush.color, self.brush.width, (x, y)))

    def move(self, x: float, y: float) -> None:
        if not self.active:
            return
        if math.dist(self._last, (x, y)) < self.MIN_STEP:
            return
        self._last = (x, y)
        self._apply_open(ops.ExtendStroke(self._stroke_id, ((x, y),)))

    def release(self, x: float, y: float) -> None:
        if not self.active:
            return
        self.move(x, y)
        if self.active:
            self._apply_open(ops.EndStroke(self._stroke_id))
        self._stroke_id = self._last = None

    def _apply_open(self, op: ops.Op) -> None:
        try:
            self.document.apply(op)
        except KeyError:
            # The canvas was cleared mid-stroke; drop the rest of this gesture.
            self._stroke_id = self._last = None
