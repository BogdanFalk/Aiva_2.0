"""The visible canvas: paints the Document and routes the pointer to a Tool.

The widget owns no drawing state. It repaints whenever the document applies an
op — from this pointer or from anyone else — so a remote painter's ops show up
here without any extra wiring.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent, QPainter
from PyQt5.QtWidgets import QWidget

from drawapp.model import Document
from drawapp.render import paint_document
from drawapp.tools import Tool


class CanvasWidget(QWidget):
    def __init__(self, document: Document, tool: Tool, parent: QWidget | None = None):
        super().__init__(parent)
        self.document = document
        self.tool = tool
        # Widget pixels == document pixels for now; zoom/pan would add a
        # transform here and in _doc_pos.
        self.setFixedSize(document.width, document.height)
        self.setCursor(Qt.CrossCursor)
        self._unsubscribe = document.subscribe(lambda _doc, _op: self.update())
        self.destroyed.connect(lambda *_: self._unsubscribe())

    def set_tool(self, tool: Tool) -> None:
        self.tool = tool

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            paint_document(painter, self.document)
        finally:
            painter.end()

    # Pen and touch input reach us as synthesized mouse events, so these three
    # handlers cover mouse, stylus and finger alike.
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.tool.press(*self._doc_pos(event))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.LeftButton:
            self.tool.move(*self._doc_pos(event))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.tool.release(*self._doc_pos(event))

    @staticmethod
    def _doc_pos(event: QMouseEvent) -> tuple[float, float]:
        p = event.localPos()
        return p.x(), p.y()
