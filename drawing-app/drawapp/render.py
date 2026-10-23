"""Painting a Document with QPainter — shared by the widget, export and tests.

Extension point — saving: `render_image(doc).save("x.png")` is already a PNG
export; a native format would store the op log instead (see ops.py).
"""
from __future__ import annotations

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPolygonF

from drawapp.model import Document, Stroke


def paint_document(painter: QPainter, doc: Document) -> None:
    painter.fillRect(0, 0, doc.width, doc.height, QColor(doc.background))
    painter.setRenderHint(QPainter.Antialiasing, True)
    for stroke in doc.strokes:
        paint_stroke(painter, stroke)


def paint_stroke(painter: QPainter, stroke: Stroke) -> None:
    pen = QPen(QColor(stroke.brush.color), stroke.brush.width,
               Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(pen)
    if len(stroke.points) == 1:  # a click is a dot (round cap gives it its width)
        painter.drawPoint(QPointF(*stroke.points[0]))
    else:
        painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in stroke.points]))


def render_image(doc: Document) -> QImage:
    image = QImage(doc.width, doc.height, QImage.Format_ARGB32)
    painter = QPainter(image)
    try:
        paint_document(painter, doc)
    finally:
        painter.end()
    return image
