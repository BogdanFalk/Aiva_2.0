"""Operations: the only way a Document changes.

Ops are frozen dataclasses of primitives (str, float, tuples), so
`dataclasses.asdict(op)` is JSON-ready. That is deliberate — it is what a save
file (an op log) or a collaboration channel would carry. Every op that starts
something names its `author` ("aiva", "saga", "user", ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

Point = tuple[float, float]


@dataclass(frozen=True)
class BeginStroke:
    stroke_id: str
    author: str
    color: str
    width: float
    point: Point


@dataclass(frozen=True)
class ExtendStroke:
    stroke_id: str
    points: tuple[Point, ...]


@dataclass(frozen=True)
class EndStroke:
    stroke_id: str


@dataclass(frozen=True)
class ClearCanvas:
    author: str


@dataclass(frozen=True)
class RemoveStroke:
    """Take one finished stroke off the canvas (undo). `author` is who asked."""
    stroke_id: str
    author: str


Op = Union[BeginStroke, ExtendStroke, EndStroke, ClearCanvas, RemoveStroke]
