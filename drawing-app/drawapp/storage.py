"""Saving drawings to disk and getting them back, without ever losing the canvas. No Qt.

A drawing is a `.aivadraw` file: JSON holding the canvas size, background and
the strokes in paint order. It is a snapshot, not the op log: a log would keep
every cleared stroke forever. Loading replays the snapshot as ordinary ops
(ClearCanvas, then BeginStroke/ExtendStroke/EndStroke per stroke), so the one
mutation path still holds and the canvas repaints the usual way.

Everything lives in one folder, `~/Pictures/Aiva and Saga Drawings` by default
(`--save-dir`):

    aiva-20261020-153012.aivadraw   a saved drawing (Ctrl+S, the "save" command)
    aiva-20261020-153012.png        the same drawing as a picture, written next to it
    cat.aivadraw                    a named save ("name": "cat"); a taken name gets " (2)"
    recovery/aiva-autosave.aivadraw     Aiva's canvas, rewritten after every finished stroke/clear
    recovery/aiva-before-clear.aivadraw what was on it just before the last Clear
    recovery/aiva.lock, aiva-window.json   which live window owns Aiva's autosave, and its port

The slot (lock, autosave, window file) is named for the window's canvas: its
author by default, or `--canvas` for a canvas several painters share, such as
"shared" for both sisters (`recovery/shared-autosave.aivadraw`, ...). Each
stroke keeps its own author either way.

`Session` ties one Document to that folder:

- **Autosave.** Every finished stroke and every clear rewrites the author's
  autosave file at once (atomic: temp file + os.replace), so even a window
  ended with TerminateProcess (launcher.DrawApp.close) loses nothing finished.
- **Restore.** A new window for the same author starts from that autosave, so
  closing the window (or a crash) never loses the canvas. A file that can't be
  restored (corrupt, other canvas size) is moved aside as
  `*.unrecovered-<time>.aivadraw`, never overwritten, and the window starts blank.
- **One owner per author.** The first window holds `recovery/<author>.lock`
  (an OS lock, released by the OS when the process dies), restores and writes
  the shared autosave, and publishes its port in `<author>-window.json` for
  `launcher.find_running`. A second window for the same author starts blank
  and autosaves to `<author>-autosave-<pid>.aivadraw` instead, so neither
  overwrites the other.
- **A shared canvas takes in the old ones.** With `absorb` (`--absorb aiva`),
  the owner adds each named canvas's autosave onto its own when no live window
  holds that canvas, strokes keeping their authors, then moves the old copy
  aside as `<name>-autosave.merged-<time>.aivadraw`. So each is added once and
  never lost. A canvas whose window is still open is left alone until a later start.
- **Clear and Open keep what they replace.** A clear first copies the canvas to
  `<author>-before-clear`; opening a drawing over unsaved work saves that work
  as a drawing first. Saving over a file copies the old one to `<name>.aivadraw.bak`.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

from drawapp import ops
from drawapp.commands import _AUTHOR, _COLOR, MAX_WIDTH, CommandError, _finite
from drawapp.model import Document, Point

FORMAT = "aivadraw"
VERSION = 1
EXT = ".aivadraw"
RECOVERY = "recovery"
MAX_SIZE = 10_000            # px per side; drawapp's canvas is 1024x768
MAX_STROKES = 50_000
MAX_TOTAL_POINTS = 2_000_000

PngWriter = Callable[[Document, str], bool]


class StorageError(CommandError):
    """A drawing file couldn't be read or written. Nothing on the canvas changed."""
    code = "storage"


def default_save_dir() -> str:
    """Where drawings go unless --save-dir says otherwise: a folder in the user's Pictures."""
    return os.path.join(os.path.expanduser("~"), "Pictures", "Aiva and Saga Drawings")


def window_file(save_dir: str, author: str) -> str:
    """The owner window's {"pid", "port", "author"} (written by Session.publish)."""
    return os.path.join(save_dir, RECOVERY, f"{author}-window.json")


# --- the file format ------------------------------------------------------------------

@dataclass(frozen=True)
class SavedStroke:
    id: str
    author: str
    color: str
    width: float
    points: tuple[Point, ...]


@dataclass(frozen=True)
class Snapshot:
    width: int
    height: int
    background: str
    strokes: tuple[SavedStroke, ...]
    source: str | None = None  # autosave only: the file this canvas was last saved to or opened from
    unsaved: bool = False      # autosave only: changed since that file was written


def to_dict(doc: Document, **extra: Any) -> dict:
    return {"format": FORMAT, "version": VERSION, "width": doc.width, "height": doc.height,
            "background": doc.background, **extra,
            "strokes": [{"id": s.id, "author": s.author, "color": s.brush.color,
                         "width": s.brush.width, "points": [[x, y] for x, y in s.points]}
                        for s in doc.strokes if s.points]}


def from_dict(data: Any) -> Snapshot:
    """Validate a decoded file completely; StorageError names the first problem."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise StorageError("not a drawapp drawing")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise StorageError("the drawing has no valid version")
    if version > VERSION:
        raise StorageError(f"the drawing was saved by a newer drawapp (format {version})")
    width, height = data.get("width"), data.get("height")
    if not all(isinstance(v, int) and not isinstance(v, bool) and 0 < v <= MAX_SIZE
               for v in (width, height)):
        raise StorageError("the drawing has no valid canvas size")
    background = data.get("background", "#ffffff")
    if not isinstance(background, str) or not _COLOR.fullmatch(background):
        raise StorageError('"background" must be "#rrggbb"')
    raw = data.get("strokes")
    if not isinstance(raw, list) or len(raw) > MAX_STROKES:
        raise StorageError(f'"strokes" must be a list of at most {MAX_STROKES}')
    strokes, seen, total = [], set(), 0
    for i, s in enumerate(raw):
        strokes.append(stroke := _stroke(i, s))
        if stroke.id in seen:
            raise StorageError(f"strokes[{i}]: duplicate id {stroke.id!r}")
        seen.add(stroke.id)
        total += len(stroke.points)
        if total > MAX_TOTAL_POINTS:
            raise StorageError(f"more than {MAX_TOTAL_POINTS} points in one drawing")
    source = data.get("source")
    return Snapshot(width, height, background, tuple(strokes),
                    source if isinstance(source, str) and source else None,
                    data.get("unsaved") is True)


def _stroke(i: int, s: Any) -> SavedStroke:
    if not isinstance(s, dict):
        raise StorageError(f"strokes[{i}] must be an object")
    sid, author, color, width = s.get("id"), s.get("author"), s.get("color"), s.get("width")
    if not isinstance(sid, str) or not 0 < len(sid) <= 64:
        raise StorageError(f"strokes[{i}] has no valid id")
    if not isinstance(author, str) or not _AUTHOR.fullmatch(author):
        raise StorageError(f"strokes[{i}] has no valid author")
    if not isinstance(color, str) or not _COLOR.fullmatch(color):
        raise StorageError(f'strokes[{i}]: "color" must be "#rrggbb"')
    if not _finite(width) or not 0 < width <= MAX_WIDTH:
        raise StorageError(f'strokes[{i}]: "width" must be a number in (0, {MAX_WIDTH:g}]')
    pts = s.get("points")
    if not isinstance(pts, list) or not pts:
        raise StorageError(f"strokes[{i}] has no points")
    points = []
    for p in pts:
        if not (isinstance(p, list) and len(p) == 2 and all(map(_finite, p))):
            raise StorageError(f"strokes[{i}] has a point that isn't [x, y]")
        points.append((float(p[0]), float(p[1])))
    return SavedStroke(sid, author, color, float(width), tuple(points))


def read_drawing(path: str) -> Snapshot:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise StorageError(f"there is no drawing at {path}") from None
    except OSError as e:
        raise StorageError(f"couldn't read {path}: {e.strerror or e}") from None
    except (ValueError, RecursionError):  # bad JSON or bad UTF-8: a damaged file
        raise StorageError(f"{os.path.basename(path)} is damaged (not valid JSON)") from None
    return from_dict(data)


def write_drawing(path: str, doc: Document, *, backup: bool = False, **extra: Any) -> None:
    """Write atomically (temp file, fsync, os.replace), so a crash leaves the old file or the
    new one, never half of either. `backup` first copies an existing file to `<path>.bak`."""
    text = json.dumps(to_dict(doc, **extra), separators=(",", ":"))
    tmp = f"{path}.tmp"
    try:
        if backup and os.path.exists(path):
            shutil.copyfile(path, f"{path}.bak")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise StorageError(f"couldn't save to {path}: {e.strerror or e}") from None


def load_into(doc: Document, snap: Snapshot, author: str = "user") -> None:
    """Replace the canvas with `snap`, through ops. Checked first, so a mismatch changes nothing."""
    if (snap.width, snap.height) != (doc.width, doc.height):
        raise StorageError(f"that drawing is {snap.width}x{snap.height} and this canvas is "
                           f"{doc.width}x{doc.height}")
    doc.background = snap.background  # not an op: no tool changes it, it's part of the canvas
    doc.apply(ops.ClearCanvas(author))
    add_strokes(doc, snap.strokes)


def add_strokes(doc: Document, strokes) -> None:
    """Replay saved strokes on top of the canvas, through ops; each keeps its id and author."""
    for s in strokes:
        first, *rest = s.points
        doc.apply(ops.BeginStroke(s.id, s.author, s.color, s.width, first))
        if rest:
            doc.apply(ops.ExtendStroke(s.id, tuple(rest)))
        doc.apply(ops.EndStroke(s.id))


# --- one window's canvas and its folder -------------------------------------------------------

_NAME_JUNK = re.compile(r"[^\w \-]+")


class Session:
    """Autosave, restore, save, open and the owner lock for one window's Document.

    Call start() once (it claims the author's slot and restores the canvas) and
    close() when the window goes away. Every method that touches a file raises
    StorageError on failure, leaving the canvas as it was. Autosave never
    raises: it runs inside Document.apply, so a full disk must not stop the
    drawing. Its failure is kept in `last_error` and shown in the status bar.

    `author` names the slot: the window's author, or its `--canvas` when several
    painters share it. `absorb` names other canvases whose autosaves the owner
    takes in at start (see the module doc).
    """

    def __init__(self, document: Document, author: str = "user", save_dir: str | None = None,
                 png_writer: PngWriter | None = None, absorb: tuple[str, ...] = ()):
        self.document = document
        self.author = author
        self.absorb = tuple(name for name in absorb if name != author)
        self.save_dir = os.path.abspath(save_dir or default_save_dir())
        self.recovery_dir = os.path.join(self.save_dir, RECOVERY)
        self.png_writer = png_writer
        self.current_path: str | None = None
        self.dirty = False
        self.owner = False
        self.autosave_path: str | None = None
        self.last_error: str | None = None
        self.restored = 0          # strokes start() brought back
        self.on_change: Callable[[Session], None] | None = None
        self._lock = None
        self._loading = False
        self._closed = False
        self._unsubscribe = [document.subscribe_before(self._before), document.subscribe(self._after)]

    # --- lifecycle ---------------------------------------------------------------------------

    def start(self) -> str:
        """Claim this author's slot and restore the canvas; returns a line for the status bar."""
        try:
            os.makedirs(self.recovery_dir, exist_ok=True)
        except OSError as e:
            self.last_error = f"can't use {self.save_dir}: {e.strerror or e}"
            self._changed()
            return f"Saving is off: {self.last_error}"
        self._lock = _try_lock(os.path.join(self.recovery_dir, f"{self.author}.lock"))
        self.owner = self._lock is not None
        if not self.owner:
            self.autosave_path = os.path.join(
                self.recovery_dir, f"{self.author}-autosave-{os.getpid()}{EXT}")
            self._changed()
            return (f"Another {self.author} canvas is open, so this one starts blank and "
                    f"autosaves to {self.autosave_path}")
        self.autosave_path = os.path.join(self.recovery_dir, f"{self.author}-autosave{EXT}")
        message = self._restore()
        restored = self.restored
        joined = self._absorb_all()
        if joined:
            message = f"{message}; {joined}" if restored else joined[:1].upper() + joined[1:]
        self._changed()
        return message

    def _absorb_all(self) -> str:
        """Owner only: add each `absorb` canvas no live window holds onto this one, then move its
        autosave aside, so it's added once and kept. Returns a status-bar line, or ''."""
        added, notes = [], []
        for name in self.absorb:
            path = os.path.join(self.recovery_dir, f"{name}-autosave{EXT}")
            if not os.path.exists(path):
                continue
            held = _try_lock(os.path.join(self.recovery_dir, f"{name}.lock"))
            if held is None:  # its own window is still open and drawing there: a later start
                notes.append(f"{name}'s canvas is still open in its own window")
                continue
            try:
                count = self._absorb(name, path)
            except StorageError as e:
                notes.append(f"couldn't add {name}'s canvas ({e}); it stays at {path}")
                continue
            finally:
                _unlock(held)
            if count:
                added.append(f"{name}'s ({count})")
        if added:
            self.restored = len(self.document.strokes)
        parts = ([f"added {' and '.join(added)} strokes from the old canvases"] if added else []) + notes
        return "; ".join(parts)

    def _absorb(self, name: str, path: str) -> int:
        """Add `name`'s autosave at `path` onto the canvas and move it aside; the strokes added.
        The merged canvas is autosaved first, so the old copy only moves once its strokes are here."""
        snap = read_drawing(path)
        if (snap.width, snap.height) != (self.document.width, self.document.height):
            raise StorageError(f"it is {snap.width}x{snap.height} and this canvas is "
                               f"{self.document.width}x{self.document.height}")
        if not snap.strokes:
            return 0
        have = {s.id for s in self.document.strokes}  # a copy added before but never moved aside
        new = [s for s in snap.strokes if s.id not in have]
        if new:
            self._loading = True
            try:
                add_strokes(self.document, new)
            finally:
                self._loading = False
            self.dirty = True
            if not self.autosave():  # here but not on disk yet (last_error says why): the copy stays
                return len(new)
        aside = _unique(path[:-len(EXT)] + time.strftime(".merged-%Y%m%d-%H%M%S"), EXT)
        try:
            os.replace(path, aside)
        except OSError as e:  # added, but the copy stays: its ids keep a later start from doubling it
            self.last_error = f"couldn't move {path} aside: {e.strerror or e}"
        return len(new)

    def _restore(self) -> str:
        path = self.autosave_path
        if not os.path.exists(path):
            return f"Drawings save to {self.save_dir}"
        try:
            snap = read_drawing(path)
            if snap.strokes:
                self._load(snap)
        except StorageError as e:
            aside = _unique(path[:-len(EXT)] + time.strftime(".unrecovered-%Y%m%d-%H%M%S"), EXT)
            try:
                os.replace(path, aside)
            except OSError as move_error:  # keep it where it is; never autosave over it
                self.autosave_path = os.path.join(
                    self.recovery_dir, f"{self.author}-autosave-{os.getpid()}{EXT}")
                return (f"Couldn't restore the last drawing ({e}) or move it aside "
                        f"({move_error.strerror}); it stays at {path}")
            return f"Couldn't restore the last drawing ({e}); it is kept as {aside}"
        if not snap.strokes:
            return f"Drawings save to {self.save_dir}"
        self.restored = len(snap.strokes)
        source = snap.source if snap.source and os.path.exists(snap.source) else None
        self.current_path = source if source and not self._in_recovery(source) else None
        self.dirty = snap.unsaved or self.current_path is None
        return f"Restored your last drawing ({len(snap.strokes)} strokes)"

    def publish(self, port: int | None) -> None:
        """Owner only: tell launcher.find_running which port this window takes commands on."""
        if not self.owner:
            return
        path = window_file(self.save_dir, self.author)
        try:
            _write_text(path, json.dumps({"pid": os.getpid(), "port": port, "author": self.author}))
        except OSError as e:
            self.last_error = f"couldn't publish the window: {e.strerror or e}"
            self._changed()

    def close(self) -> None:
        """The window is going away: autosave is already current; give up the slot. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # The window's on_change refers back to the window: drop it, so the window is freed by
        # refcount when it goes, not later by the cyclic GC (tearing a QMainWindow down from the
        # GC, mid-whatever-runs-next, is how PyQt5 fail-fasts).
        self.on_change = None
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        self._unsubscribe = []
        if self.owner:
            try:
                os.remove(window_file(self.save_dir, self.author))
            except OSError:
                pass
        elif self.autosave_path and not self.dirty:  # a duplicate window's copy, already saved
            try:
                os.remove(self.autosave_path)
            except OSError:
                pass
        if self._lock is not None:
            _unlock(self._lock)
            self._lock = None
        self.owner = False

    # --- autosave (inside Document.apply: never raises) -----------------------------------------

    def _before(self, doc: Document, op: ops.Op) -> None:
        if isinstance(op, ops.ClearCanvas) and doc.strokes and not self._loading:
            self._quietly(os.path.join(self.recovery_dir, f"{self.author}-before-clear{EXT}"))

    def _after(self, doc: Document, op: ops.Op) -> None:
        if self._loading:
            return
        self.dirty = True
        # a stroke being drawn waits for its release; an undone one is gone from the copy at once
        if isinstance(op, (ops.EndStroke, ops.ClearCanvas, ops.RemoveStroke)):
            self.autosave()
        self._changed()

    def autosave(self) -> bool:
        if not self.autosave_path:
            return False
        ok = self._quietly(self.autosave_path, source=self.current_path, unsaved=self.dirty)
        if ok and self.last_error and self.last_error.startswith("autosave"):
            self.last_error = None
        return ok

    def _quietly(self, path: str, **extra: Any) -> bool:
        try:
            write_drawing(path, self.document, **extra)
        except StorageError as e:
            self.last_error = f"autosave failed: {e}"
            self._changed()
            return False
        return True

    # --- save / open ----------------------------------------------------------------------------

    def save(self, name: str | None = None, author: str | None = None) -> dict:
        """Save the canvas: to its own file if it has one and no `name` is given, else to a new
        file in the save folder (named `name`, or `<author>-<time>`; a taken name gets " (2)")."""
        if not self.document.strokes:
            raise StorageError("the canvas is empty, so there is nothing to save")
        if name is None and self.current_path and not self._in_recovery(self.current_path):
            path = self.current_path
        else:
            path = self._new_path(name, author or self.author)
        return self._write(path)

    def save_as(self, path: str) -> dict:
        """Save to a path the user picked (the dialog already asked before replacing a file)."""
        if not self.document.strokes:
            raise StorageError("the canvas is empty, so there is nothing to save")
        if not path.lower().endswith(EXT):
            path += EXT
        return self._write(os.path.abspath(path))

    def _write(self, path: str) -> dict:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except OSError as e:
            raise StorageError(f"couldn't save to {path}: {e.strerror or e}") from None
        write_drawing(path, self.document, backup=True)
        png = path[:-len(EXT)] + ".png"
        if self.png_writer is not None:
            try:
                png = png if self.png_writer(self.document, png) else None
            except Exception:  # the picture is a convenience; the drawing itself is saved
                png = None
        else:
            png = None
        self.current_path, self.dirty = path, False
        self.autosave()
        self._changed()
        return {"path": path, "png": png, "name": _stem(path), "strokes": len(self.document.strokes)}

    def open(self, path: str, author: str | None = None) -> dict:
        """Replace the canvas with a saved drawing. Unsaved work on it is saved as a drawing
        first, so nothing on the canvas is lost; a bad file changes nothing."""
        path = os.path.abspath(path)
        snap = read_drawing(path)
        if (snap.width, snap.height) != (self.document.width, self.document.height):
            raise StorageError(f"that drawing is {snap.width}x{snap.height} and this canvas is "
                               f"{self.document.width}x{self.document.height}")
        saved_previous = None
        if self.dirty and self.document.strokes:  # always a new file: nobody asked to overwrite one
            saved_previous = self._write(self._new_path(None, author or self.author))["path"]
        self._load(snap, author or self.author)
        rescued = self._in_recovery(path)  # a recovery copy is unsaved work, not a drawing file
        self.current_path = None if rescued else path
        self.dirty = rescued
        self.autosave()
        self._changed()
        return {"path": path, "name": _stem(path), "strokes": len(snap.strokes),
                "saved_previous": saved_previous}

    def _load(self, snap: Snapshot, author: str | None = None) -> None:
        self._loading = True
        try:
            load_into(self.document, snap, author or self.author)
        finally:
            self._loading = False

    # --- the folder -----------------------------------------------------------------------------

    def drawings(self) -> dict:
        """Saved drawings and recovery copies, newest first, by name."""
        return {"save_dir": self.save_dir,
                "drawings": _listing(self.save_dir, ""),
                "recovery": _listing(self.recovery_dir, RECOVERY + "/")}

    def resolve(self, name: str | None) -> str:
        """A drawing's path by its name as drawings() lists it; None means the newest saved one."""
        listed = self.drawings()
        if name is None:
            if not listed["drawings"]:
                raise StorageError(f"there are no saved drawings in {self.save_dir} yet")
            return listed["drawings"][0]["path"]
        want = name.strip().replace("\\", "/").lower()
        if want.endswith(EXT):
            want = want[:-len(EXT)]
        for entry in listed["drawings"] + listed["recovery"]:
            if entry["name"].lower() == want:
                return entry["path"]
        names = ", ".join(e["name"] for e in listed["drawings"][:8]) or "none yet"
        raise StorageError(f"there is no saved drawing called {name!r} (saved: {names})")

    def suggested_path(self) -> str:
        if self.current_path and not self._in_recovery(self.current_path):
            return self.current_path
        return os.path.join(self.save_dir, _stem(self._new_path(None, self.author)) + EXT)

    def describe(self) -> str:
        """One line for the status bar: where the drawing is, or will be, saved."""
        if self.last_error:
            return self.last_error[:1].upper() + self.last_error[1:]
        if not self.autosave_path:
            return ""
        if self.current_path and not self.dirty:
            return f"Saved as {os.path.basename(self.current_path)} in {self.save_dir}"
        if not self.document.strokes:
            return f"Drawings save to {self.save_dir}"
        return f"Not saved yet (a recovery copy is kept) · Ctrl+S saves to {self.save_dir}"

    def _new_path(self, name: str | None, author: str) -> str:
        stem = " ".join(_NAME_JUNK.sub(" ", name or "").split())[:60].strip(" -")
        if not stem:
            stem = f"{author}-{time.strftime('%Y%m%d-%H%M%S')}"
        return _unique(os.path.join(self.save_dir, stem), EXT)

    def _in_recovery(self, path: str) -> bool:
        return os.path.dirname(os.path.abspath(path)).lower() == self.recovery_dir.lower()

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change(self)


# --- small file helpers -------------------------------------------------------------------------

def _stem(path: str) -> str:
    return os.path.basename(path)[:-len(EXT)] if path.lower().endswith(EXT) else os.path.basename(path)


def _unique(base: str, ext: str) -> str:
    """`base+ext`, or `base (2)+ext`, ... — never a name whose drawing or picture already exists."""
    n, candidate = 1, base
    while os.path.exists(candidate + ext) or os.path.exists(candidate + ".png"):
        n += 1
        candidate = f"{base} ({n})"
    return candidate + ext


def _listing(folder: str, prefix: str) -> list[dict]:
    out = []
    for path in glob.glob(os.path.join(glob.escape(folder), "*" + EXT)):
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        out.append({"name": prefix + _stem(path), "path": path,
                    "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                    "_mtime": mtime})
    out.sort(key=lambda e: e["_mtime"], reverse=True)
    for e in out:
        del e["_mtime"]
    return out


def _write_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _try_lock(path: str):
    """An exclusive, non-blocking OS lock on `path`, held while the returned file stays open;
    None if another live process holds it. The OS drops it when the holder dies, however it dies."""
    try:
        f = open(path, "a+b")
    except OSError:
        return None
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def _unlock(f) -> None:
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    f.close()

