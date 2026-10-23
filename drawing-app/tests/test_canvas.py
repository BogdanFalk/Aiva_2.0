"""The real widgets, driven offscreen (QT_QPA_PLATFORM=offscreen, see conftest)."""
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QColor, QMouseEvent
from PyQt5.QtWidgets import QApplication

from drawapp import app as drawapp_app
from drawapp.app import MainWindow
from drawapp.canvas import CanvasWidget
from drawapp.model import Brush, Document
from drawapp.render import render_image
from drawapp.tools import BrushTool

WHITE = QColor("#ffffff").rgb()


def _mouse(widget, kind, x, y):
    button = Qt.NoButton if kind == QEvent.MouseMove else Qt.LeftButton
    held = Qt.NoButton if kind == QEvent.MouseButtonRelease else Qt.LeftButton
    event = QMouseEvent(kind, QPointF(x, y), button, held, Qt.NoModifier)
    QApplication.sendEvent(widget, event)


def _drag(widget, points):
    _mouse(widget, QEvent.MouseButtonPress, *points[0])
    for p in points[1:]:
        _mouse(widget, QEvent.MouseMove, *p)
    _mouse(widget, QEvent.MouseButtonRelease, *points[-1])


def test_mouse_drag_draws_a_visible_stroke(qapp):
    doc = Document(200, 100)
    canvas = CanvasWidget(doc, BrushTool(doc, "aiva", Brush("#000000", 6.0)))
    _drag(canvas, [(20, 50), (100, 50), (180, 50)])

    assert len(doc.strokes) == 1 and doc.strokes[0].done
    shot = canvas.grab().toImage()
    assert shot.pixel(100, 50) != WHITE  # on the line
    assert shot.pixel(100, 10) == WHITE  # well off it


def test_hover_without_a_button_does_not_draw(qapp):
    doc = Document(100, 100)
    canvas = CanvasWidget(doc, BrushTool(doc))
    QApplication.sendEvent(canvas, QMouseEvent(
        QEvent.MouseMove, QPointF(50, 50), Qt.NoButton, Qt.NoButton, Qt.NoModifier))
    assert doc.strokes == []


def test_a_click_leaves_a_dot(qapp):
    doc = Document(50, 50)
    tool = BrushTool(doc, brush=Brush("#000000", 8.0))
    tool.press(25, 25)
    tool.release(25, 25)
    assert render_image(doc).pixel(25, 25) != WHITE


def test_window_clear_action_resets_the_canvas(qapp):
    doc = Document(160, 120)
    window = MainWindow(doc, author="saga")  # never shown
    _drag(window.canvas, [(10, 10), (150, 110)])
    assert len(doc.strokes) == 1
    assert doc.strokes[0].brush.color == drawapp_app.AUTHOR_COLORS["saga"]

    window.clear_action.trigger()
    assert doc.strokes == []
    shot = window.canvas.grab().toImage()
    assert all(shot.pixel(x, y) == WHITE for x in range(0, 160, 8) for y in range(0, 120, 8))
    window.deleteLater()


def test_render_demo_writes_a_png_without_a_window(qapp, tmp_path, capsys):
    out = tmp_path / "demo.png"
    assert drawapp_app.main(["--render-demo", str(out), "--width", "320", "--height", "200"]) == 0
    assert out.exists() and out.stat().st_size > 0
    assert "2 strokes" in capsys.readouterr().out
    assert not [w for w in QApplication.topLevelWidgets() if w.isVisible()]
