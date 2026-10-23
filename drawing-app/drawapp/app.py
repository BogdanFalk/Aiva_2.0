"""Main window + command line.

    python -m drawapp                      # open the drawing window
    python -m drawapp --author saga        # strokes are tagged as Saga's
    python -m drawapp --render-demo out.png  # headless: draw a demo, save a PNG, exit
    python -m drawapp --listen [PORT]      # also take JSON drawing commands on 127.0.0.1 (server.py)
    python -m drawapp --listen 0 --ready-file F  # write {"pid", "port"} to F once listening (launcher.py)
    python -m drawapp --no-activate        # show without taking the keyboard focus (launcher.py)
    python -m drawapp --save-dir DIR       # save/restore drawings in DIR (default: storage.default_save_dir())
    python -m drawapp --canvas shared --absorb aiva --absorb saga   # the sisters' one canvas

The window saves on its own (storage.py): every finished stroke rewrites its
author's recovery copy, and the next window for that author starts from it.
Ctrl+S saves a drawing (.aivadraw + .png) to the save folder, Ctrl+O opens one.
Ctrl+Z takes back the window author's last stroke (never anyone else's).

`--canvas NAME` makes the window one canvas several painters share (Aiva's and
Saga's is "shared"): its autosave, lock and window file are NAME's, while
`--author` stays who the mouse draws as, and every command stroke keeps its own
author. `--absorb NAME` adds NAME's own old canvas onto it (storage.py).
"""
from __future__ import annotations

import argparse
import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox,
                             QScrollArea, QWidget)

from drawapp import ops
from drawapp.canvas import CanvasWidget
from drawapp.commands import _AUTHOR, CommandError, CommandRunner
from drawapp.launcher import write_ready
from drawapp.model import AUTHOR_COLORS, Brush, Document
from drawapp.render import render_image
from drawapp.server import DEFAULT_PORT, CommandServer
from drawapp.storage import EXT, Session
from drawapp.tools import BrushTool


def save_png(doc: Document, path: str) -> bool:
    """The drawing as a picture next to its .aivadraw (atomic, like the drawing itself)."""
    tmp = f"{path}.tmp"
    if not render_image(doc).save(tmp, "PNG"):
        return False
    os.replace(tmp, path)
    return True


class MainWindow(QMainWindow):
    """The canvas window. With a `session` (storage.Session, already started or
    started right after) it also has Open/Save/Save As and shows where the
    drawing is saved; without one it draws only, as before."""

    def __init__(self, document: Document, author: str = "user", session: Session | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.document = document
        self.author = author
        self.session = session
        brush = Brush(color=AUTHOR_COLORS.get(author, Brush.color))
        self.canvas = CanvasWidget(document, BrushTool(document, author, brush))

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setStyleSheet("QScrollArea { background: #3a3a3f; }")
        self.setCentralWidget(scroll)

        toolbar = self.addToolBar("Canvas")
        toolbar.setMovable(False)
        if session is not None:
            self._add_storage(toolbar)
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.setToolTip(f"Take back {author}'s last stroke (Ctrl+Z); nobody else's")
        self.undo_action.triggered.connect(self.undo)
        toolbar.addAction(self.undo_action)
        self.clear_action = QAction("Clear", self)
        self.clear_action.setShortcut(QKeySequence("Ctrl+Shift+Del"))
        self.clear_action.setToolTip("Clear the canvas (Ctrl+Shift+Del)")
        self.clear_action.triggered.connect(self.clear)
        toolbar.addAction(self.clear_action)

        slot = session.author if session is not None else author   # the canvas, if it's shared
        self.setWindowTitle(f"Aiva & Saga — Drawing ({author if slot == author else slot + ' canvas'})")
        self.statusBar().showMessage(f"Drawing as {author}")
        self.resize(document.width + 40, document.height + 80)

    def clear(self) -> None:
        self.document.apply(ops.ClearCanvas(self.author))  # the session keeps a before-clear copy

    def undo(self) -> str | None:
        """Take back this window's author's newest finished stroke (the mouse draws as them); a
        stroke still under the pointer and everyone else's are left alone. The undone stroke id."""
        stroke = self.document.last_stroke(self.author)
        if stroke is None:
            self.show_notice(f"Nothing of {self.author}'s to undo")
            return None
        self.document.apply(ops.RemoveStroke(stroke.id, self.author))  # autosaved like a stroke
        return stroke.id

    # --- saving (only with a session) -------------------------------------------------------

    def _add_storage(self, toolbar) -> None:
        folder = self.session.save_dir
        self.open_action = QAction("Open…", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.setToolTip("Open a saved drawing (Ctrl+O); unsaved work is saved first")
        self.open_action.triggered.connect(self.open_drawing)
        self.save_action = QAction("Save", self)
        self.save_action.setShortcut(QKeySequence.Save)
        self.save_action.setToolTip(f"Save the drawing to {folder} (Ctrl+S)")
        self.save_action.triggered.connect(self.save_drawing)
        self.save_as_action = QAction("Save As…", self)
        self.save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.save_as_action.setToolTip("Save the drawing under a new name (Ctrl+Shift+S)")
        self.save_as_action.triggered.connect(self.save_drawing_as)
        for action in (self.open_action, self.save_action, self.save_as_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        self.save_label = QLabel(self.session.describe())
        self.save_label.setToolTip(f"Drawings: {folder}\nRecovery copies: {self.session.recovery_dir}")
        self.statusBar().addPermanentWidget(self.save_label)
        self.session.on_change = lambda session: self.save_label.setText(session.describe())

    def show_notice(self, text: str) -> None:
        """A line in the save label until the drawing next changes (e.g. what was restored)."""
        if self.session is not None and text:
            self.save_label.setText(text)

    def save_drawing(self) -> dict | None:
        result = self._storage(self.session.save)
        if result:
            self.show_notice(f"Saved {result['path']}")
        return result

    def save_drawing_as(self) -> dict | None:
        path, _ = QFileDialog.getSaveFileName(self, "Save drawing as", self.session.suggested_path(),
                                              f"Drawings (*{EXT})")
        if not path:
            return None
        result = self._storage(self.session.save_as, path)
        if result:
            self.show_notice(f"Saved {result['path']}")
        return result

    def open_drawing(self) -> dict | None:
        path, _ = QFileDialog.getOpenFileName(self, "Open a drawing", self.session.save_dir,
                                              f"Drawings (*{EXT})")
        if not path:
            return None
        result = self._storage(self.session.open, path, self.author)
        if result:
            first = (f" (your unsaved drawing was saved first as {result['saved_previous']})"
                     if result["saved_previous"] else "")
            self.show_notice(f"Opened {result['name']}{first}")
        return result

    def _storage(self, fn, *args) -> dict | None:
        try:
            return fn(*args)
        except CommandError as e:  # StorageError: the canvas is as it was
            QMessageBox.warning(self, "Drawing", str(e))
            return None

    def closeEvent(self, event) -> None:
        if self.session is not None:
            self.session.close()  # its autosave is already current
        super().closeEvent(event)


def start_listener(window: MainWindow, port: int) -> CommandServer:
    """Serve JSON drawing commands into `window`'s document on 127.0.0.1:`port`.

    The port goes in the title as well as the status bar, so a window that is
    listening can be told on sight from one that isn't. OSError if it's taken.
    """
    server = CommandServer(CommandRunner(window.document, window.session), parent=window)
    bound = server.start(port)
    where = f"commands on 127.0.0.1:{bound}"
    window.setWindowTitle(f"{window.windowTitle()} · {where}")
    window.statusBar().showMessage(f"Drawing as {window.author} · {where}")
    return server


def render_demo(path: str, width: int, height: int) -> None:
    """Draw with the real tools on a headless document and save it — no window."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv[:1])  # noqa: F841 (QPainter needs it)
    doc = Document(width, height)
    for author, y in (("aiva", height * 0.4), ("saga", height * 0.6)):
        tool = BrushTool(doc, author, Brush(AUTHOR_COLORS[author], 6.0))
        tool.press(width * 0.1, y)
        for i in range(1, 41):
            x = width * (0.1 + 0.8 * i / 40)
            tool.move(x, y + (height * 0.08) * (1 if i % 10 < 5 else -1) * (i % 5) / 5)
        tool.release(width * 0.9, y)
    if not render_image(doc).save(path):
        raise SystemExit(f"could not write {path}")
    print(f"wrote {path} ({width}x{height}, {len(doc.strokes)} strokes, revision {doc.revision})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="drawapp", description="A shared canvas for Aiva and Saga.")
    parser.add_argument("--author", default="user", help="who is drawing (aiva, saga, user, ...)")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--render-demo", metavar="PNG",
                        help="draw a demo headlessly, save it as PNG and exit (no window)")
    parser.add_argument("--listen", metavar="PORT", type=int, nargs="?", const=DEFAULT_PORT,
                        help=f"accept JSON drawing commands on 127.0.0.1:PORT "
                             f"(default {DEFAULT_PORT}; 0 = any free port)")
    parser.add_argument("--ready-file", metavar="JSON",
                        help='with --listen: write {"pid", "port"} here once the window is up '
                             "and listening; removed again on exit")
    parser.add_argument("--save-dir", metavar="DIR",
                        help="where drawings are saved and the canvas is restored from "
                             "(default: Pictures\\Aiva and Saga Drawings)")
    parser.add_argument("--canvas", metavar="NAME",
                        help="which canvas this window is: its autosave, lock and window file "
                             "(default: the author's own; 'shared' is the sisters' one canvas)")
    parser.add_argument("--absorb", metavar="NAME", action="append", default=[],
                        help="add NAME's own canvas onto this one at start if no window has it "
                             "open (repeatable; the old copy is kept as *.merged-*)")
    parser.add_argument("--no-activate", action="store_true",
                        help="show the window without taking the keyboard focus from whatever "
                             "has it (how code opens it, so nobody's typing is interrupted)")
    args = parser.parse_args(argv)
    if args.ready_file and args.listen is None:
        parser.error("--ready-file needs --listen")
    for name in ([args.canvas] if args.canvas else []) + args.absorb:  # they name files
        if not _AUTHOR.fullmatch(name.lower()):
            parser.error(f"--canvas/--absorb {name!r}: lowercase letters, digits or _")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.render_demo:
        render_demo(args.render_demo, args.width, args.height)
        return 0

    app = QApplication.instance() or QApplication(sys.argv[:1])
    document = Document(args.width, args.height)
    author = args.author.lower()
    canvas = (args.canvas or author).lower()
    session = Session(document, canvas, args.save_dir, png_writer=save_png,
                      absorb=tuple(name.lower() for name in args.absorb))
    window = MainWindow(document, author=author, session=session)
    try:
        return _run(app, args, window, session)
    finally:
        session.close()  # also when the event loop ends without a closeEvent, or we never showed


def _run(app: QApplication, args: argparse.Namespace, window: MainWindow, session: Session) -> int:
    server = None
    if args.listen is not None:
        # Bind before the window shows: a window on screen must already be listening. A taken
        # port ends it here, before it has touched the save folder. Commands wait for the event
        # loop, so none can land before the restore below.
        try:
            server = start_listener(window, args.listen)
        except OSError as e:
            print(f"drawapp: {e}", file=sys.stderr)
            return 1
    notice = session.start()  # restores this author's canvas before the window shows
    window.show_notice(notice)
    print(f"drawapp: {notice}", flush=True)
    if server is not None:
        session.publish(server.serverPort())  # launcher.find_running adopts this window
        print(f"drawapp: listening for drawing commands on 127.0.0.1:{server.serverPort()}",
              flush=True)
    if args.no_activate:
        window.setAttribute(Qt.WA_ShowWithoutActivating)  # SW_SHOWNOACTIVATE on Windows
    window.show()
    if not args.ready_file:
        return app.exec_()
    try:
        write_ready(args.ready_file, server.serverPort(), restored=session.restored)
    except OSError as e:
        print(f"drawapp: cannot write the ready file: {e}", file=sys.stderr)
        return 1
    try:
        return app.exec_()
    finally:
        try:
            os.remove(args.ready_file)
        except OSError:
            pass
