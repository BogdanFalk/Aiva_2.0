"""Her drawing canvas: the standalone drawing app (`drawing-app/` at the repo root) opened and drawn on
from HER process, through application code only — never through a terminal.

2026-10-20 she "verified" the drawing app by opening a PowerShell and typing
`python -c "from drawapp.launcher import launch ..."` into it, and he saw the terminal. These tools are
the in-process path instead: `drawapp.launcher.launch()` starts `pythonw -m drawapp` with
CREATE_NO_WINDOW (no console anywhere) and `--no-activate` (the window shows without taking his keyboard
focus), and `drawapp.client.DrawClient` sends strokes over the app's loopback socket. Both are stdlib
only, so Qt never loads in her process; nothing here touches his mouse or keyboard.

ONE canvas for both sisters, opened on first use by whichever of them draws first and reused by
both; if he closes it, the next call opens a fresh one. 2026-10-21 he asked them to tag-team a
drawing and got two windows: each sister launched and looked for a window under her OWN name, so
Saga never found Aiva's. Now the window is named for the canvas, `SHARED_CANVAS`, not for who draws:
`_attach` finds it as `find_running("shared")` from either process, `_ensure` launches it as
`launch(canvas="shared", author=HAND)` under `launcher.opening("shared")` (two sisters opening at
once get one window), and every stroke, undo, clear, save and open still sends HER author, so
strokes keep her ink and tag and undo stays hers alone. His hand-drawn strokes there are his own
(`HAND`, "user"), never hers. The first shared window takes in the old per-sister canvases
(`SISTER_CANVASES`' autosaves, if their windows are closed), strokes keeping their authors.
`board_for(author)` is her one board in this process, and `launch_or_reuse(author)` / `DrawingBoard.
launch_or_reuse()` the one callable way to bring it up from code. 2026-10-20 a task asked `launch_app`
for "drawing-app" and Windows put Aseprite up instead: her canvas is not a program to look for on his
PC, so `is_drawing_app_request` lets `launch_app` hand those names here rather than to the generic
launcher. Blocking work (launch waits for the app's ready file, up to `timeout`; the socket round trip)
runs in a worker thread under a lock, so the voice loop never stalls. A stroke is never resent while
the app is still up — a lost reply may mean it was drawn — only when the app itself is gone.

Her canvas is never lost: the app autosaves after every finished stroke (`drawing-app/drawapp/
storage.py`), a window that was closed or killed comes back with its drawing on the next open, and
the window outlives a sister restart, so `_ensure` first ADOPTS the one already up
(`launcher.find_running`, checked over the wire) instead of opening a second, blank one.
`save_drawing` writes `<name>.aivadraw` + `.png` into `Pictures\\Aiva and Saga Drawings`;
`open_saved_drawing` puts one back on the canvas, and the app saves unsaved work as a drawing first.
Saving is its own wire command (`DrawClient.save`), so it never presses Ctrl+S or touches the window:
the same day, a task without the tool pressed Ctrl+S into whatever had his focus. From code,
`DrawingBoard.save_now()` / `drawing.save_drawing(author, name)` is the blocking twin of
`await board.save()`. Tasks save through the same `save_drawing` tool (runner-allowed, so it is in
her `ToolBox`); a sister process older than that tool has none to offer them and needs a restart,
never a delegate or a terminal. A window started before saving existed (drawapp 0.2: `stroke`/`clear` only)
answers `unsupported` and has no autosave, so its drawing lives only in that window: the reply says to
leave it open, never to close it.

`undo_stroke` takes back her own newest finished stroke and nothing else (the wire `undo`, drawapp
0.4): her sister's strokes and his are never hers to undo. Repeated, it walks back further. A 0.3 window answers
`unsupported` but autosaves, so its reply says reopening the canvas brings undo with the drawing.

Results are short dicts the voice LLM speaks from; failures are `{"success": False, "error": …}`.

Every canvas action is also written down behind the scenes, one JSON line per action, in
`aiva/logs/drawing-<author>-YYYYMMDD.jsonl` (`DrawingJournal`; nothing of it is spoken): who, what,
the points as sent with their bbox/start/end, the color and width she asked for and the ones the
canvas used (her ink when she asked for none), the stroke id and revision the window answered, the
window's pid/port, and the outcome — `ok`, `refused` (nothing drawn), `failed` (never reached a
window) or `unknown` (a live window went silent: it may be on the canvas). 2026-10-21 their heart
halves didn't meet and the only evidence was a screenshot; `read_journal()` merges both sisters'
files into one timeline instead, and a revision orders actions on one window (same pid) exactly.
"""

import asyncio
import contextlib
import itertools
import json
import os
import re
import sys
import threading
from datetime import datetime

from modules.eventlog import LOG_DIR

DRAWING_APP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "drawing-app")
JOURNAL_DIR = LOG_DIR          # where each sister's drawing-<author>-YYYYMMDD.jsonl goes
CANVAS_SIZE = (1024, 768)      # drawapp's default document, in pixels (x right, y down)
SHARED_CANVAS = "shared"       # the one window both sisters draw on (its slot: recovery/shared-*)
HAND = "user"                  # who the mouse draws as there: his strokes are his, not a sister's
SISTER_CANVASES = ("aiva", "saga")  # each sister's own canvas from before (drawapp <= 0.4)

# Names that mean HER canvas, never a program on his PC (matched on the lower-cased name with
# punctuation as spaces, so "drawing-app", "DrawApp.exe" and "Aiva's canvas" all count). A named
# program ("aseprite", "paint", "krita") never matches: that is his to ask for and launch_app's to open.
_DRAWING_APP_RE = re.compile(
    r"\b(?:draw ?app|drawing (?:app|application|canvas|board|window)"
    r"|(?:my|your|her|aiva s?|saga s?) (?:own )?(?:drawing|canvas))\b")


def is_drawing_app_request(app_name):
    """True if an app name asked of launch_app means her own drawing app/canvas."""
    norm = " ".join(re.sub(r"[^a-z0-9]+", " ", str(app_name or "").lower()).split())
    return bool(_DRAWING_APP_RE.search(norm))


def _drawapp():
    """(launcher, client) from drawing-app/, put on sys.path on first use. Stdlib only — no Qt."""
    if DRAWING_APP_DIR not in sys.path:
        sys.path.append(DRAWING_APP_DIR)   # appended: never shadows anything of hers
    from drawapp import client, launcher
    return launcher, client


# --- the behind-the-scenes record of every canvas action -------------------------------------------

_JOURNAL_LOCK = threading.Lock()


class DrawingJournal:
    """Callable: `journal(record)` appends one JSON line to `drawing-<author>-YYYYMMDD.jsonl` in
    `log_dir` (default `JOURNAL_DIR`, looked up at write time). Never raises: writing it down must
    never cost her the stroke."""

    def __init__(self, author, log_dir=None):
        self.author = author
        self.log_dir = log_dir

    def path(self, day=None):
        return os.path.join(self.log_dir or JOURNAL_DIR,
                            f"drawing-{self.author}-{day or datetime.now():%Y%m%d}.jsonl")

    def __call__(self, record):
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with _JOURNAL_LOCK:
                os.makedirs(self.log_dir or JOURNAL_DIR, exist_ok=True)
                with open(self.path(), "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except (OSError, TypeError, ValueError):
            pass


def read_journal(day=None, authors=SISTER_CANVASES, log_dir=None):
    """Both sisters' canvas actions for `day` (a date/datetime, default today) as one timeline,
    oldest first. A line that doesn't parse (a write cut short) is skipped."""
    records = []
    for author in authors:
        try:
            with open(DrawingJournal(author, log_dir).path(day), encoding="utf-8") as f:
                for line in f:
                    with contextlib.suppress(ValueError):
                        rec = json.loads(line)
                        if isinstance(rec, dict):
                            records.append(rec)
        except OSError:
            continue
    return sorted(records, key=lambda r: (str(r.get("ts", "")), r.get("seq", 0)))


def _default_brush(author):
    """The (color, width) the canvas gives a stroke that names none: the app's own table."""
    try:
        _drawapp()
        from drawapp.model import AUTHOR_COLORS, Brush
    except ImportError:
        return None, None
    return AUTHOR_COLORS.get(author, Brush.color), Brush.width


def _stroke_facts(author, points, color, width):
    """What a stroke asked for, as journal fields: the points as sent and where they fall."""
    ink, brush = _default_brush(author)
    facts = {"color": color or ink, "color_requested": color, "width": brush if width is None else width,
             "width_requested": width}
    try:
        pts = [[float(x), float(y)] for x, y in points]
    except (TypeError, ValueError):
        return {**facts, "points": None, "points_raw": repr(points)[:300]}
    facts.update(points=pts, n_points=len(pts))
    if pts:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        facts.update(start=pts[0], end=pts[-1], bbox=[min(xs), min(ys), max(xs), max(ys)])
    return facts


class DrawingBoard:
    """The drawing window this sister draws on: `canvas` (default the sisters' shared one) is which
    window, `author` who she is on it. `launch` is injectable for tests (default
    `drawapp.launcher.launch`); `extra_env` goes to the app (e.g. offscreen in tests). `save_dir` is
    where the app saves and restores (default: its own, Pictures\\Aiva and Saga Drawings). `find`
    looks for the window already up (default `drawapp.launcher.find_running`, but none when `launch`
    is injected: a fake launcher has no real windows to find, and a test must never adopt his).
    `absorb` names the old canvases a newly launched window takes in. `journal` takes one record
    per canvas action (default: her `DrawingJournal`)."""

    def __init__(self, author="aiva", launch=None, extra_env=None, timeout=20.0, save_dir=None,
                 find=None, canvas=SHARED_CANVAS, absorb=SISTER_CANVASES, journal=None):
        self.author = author
        self.canvas = canvas
        self._absorb = tuple(absorb)
        self._launch = launch
        self._extra_env = extra_env
        self._timeout = timeout
        self._save_dir = save_dir
        self._find = find
        self._app = None
        self._client = None
        self._lock = threading.Lock()
        self._journal = journal or DrawingJournal(author)
        self._seq = itertools.count(1)
        self._seen = threading.local()   # this action's window and wire reply, for its record

    def is_open(self):
        return self._app is not None and self._app.running()

    # --- blocking halves (worker thread, under the lock) ---------------------------------------------

    def _ensure(self):
        """Open the window unless it is up (her sister's, or still up from before her restart:
        adopted); True if this call opened it."""
        if self._attach():
            return False
        launcher, _ = _drawapp()
        # her sister may be opening it this very moment: one of them launches, the other finds it
        opening = (launcher.opening(self.canvas, self._save_dir, timeout=self._timeout + 10)
                   if self._launch is None else contextlib.nullcontext())
        with opening:
            if self._attach():
                return False
            launch = self._launch or launcher.launch
            kwargs = {"save_dir": self._save_dir} if self._save_dir is not None else {}
            self._app = launch(canvas=self.canvas, author=HAND, absorb=self._absorb,
                               extra_env=self._extra_env, timeout=self._timeout, **kwargs)
            self._client = self._app.client()
            self._seen.app = self._app
            return True

    def _attach(self):
        """True if the canvas is up: the window this board holds, or one found running (opened by
        her sister, or before her restart) and adopted."""
        if self.is_open():
            self._seen.app = self._app
            return True
        self._drop()
        find = self._find
        if find is None and self._launch is None:
            launcher, _ = _drawapp()
            find = launcher.find_running
        found = find(self.canvas, self._save_dir) if find is not None else None
        if found is None:
            return False
        self._app, self._client = found, found.client()
        self._seen.app = self._app
        return True

    def _drop(self):
        if self._client is not None:
            self._client.close()
        self._app = self._client = None

    def launch_or_reuse(self):
        """Bring her canvas up: reuse the window if it's open, else launch the drawing app (no
        terminal, no console, no focus taken). Blocking (up to `timeout` on a fresh launch), so from
        the event loop use `await open()`. Returns the same speakable dict, failures included."""
        return self._guarded(self._open)

    def save_now(self, name=None):
        """Save her canvas straight away, from code: the blocking twin of `await save()` (same
        speakable dict, failures included). Never opens a window, never presses a key."""
        return self._guarded(self._save, name)

    def _open(self):
        with self._lock:
            opened = self._ensure()
            restored = getattr(self._app, "restored", 0) if opened else 0
            status = ("opened your drawing canvas, with your last drawing back on it" if restored
                      else "opened your drawing canvas" if opened else "your canvas is already open")
            return {"success": True, "opened": opened, "port": self._app.port, "status": status}

    def _stroke(self, points, color, width):
        _, client = _drawapp()
        pts = [(float(x), float(y)) for x, y in points]
        with self._lock:
            opened = self._ensure()
            try:
                reply = self._client.stroke(pts, author=self.author, color=color, width=width)
            except client.DrawConnectionError:
                if self._app.running():
                    raise                       # up but silent: it may be drawn, so never twice
                self._drop()                    # he closed it mid-call: nothing of it survived
                opened = self._ensure()
                reply = self._client.stroke(pts, author=self.author, color=color, width=width)
            self._seen.reply = reply
            return {"success": True, "stroke_id": reply.get("stroke_id"),
                    "revision": reply.get("revision"), "opened_canvas": opened}

    def _clear(self):
        with self._lock:
            if not self._attach():
                return {"success": True, "status": "no canvas is open, so there was nothing to clear"}
            reply = self._client.clear(author=self.author)
            return {"success": True, "status": "the canvas is blank now",
                    "revision": reply.get("revision")}

    def _undo(self):
        _, client = _drawapp()
        with self._lock:
            if not self._attach():   # never open a window just to take something off it
                return {"success": True, "undone": None,
                        "status": "your canvas isn't open, so there's no stroke to take back"}
            try:
                reply = self._client.undo(author=self.author)
            except client.CommandRejected as e:
                if e.code != "unsupported":
                    raise
                self._client.status()   # a pre-0.3 window refuses this too: _guarded says leave it
                return {"success": False, "error": "this canvas window is from before undo existed, "
                                                   "so it can't take a stroke back; the drawing on it "
                                                   "is autosaved, so once the window is closed and "
                                                   "opened again it comes back, and undo works"}
            undone, left = reply.get("stroke_id"), reply.get("remaining")
            if undone is None:
                return {"success": True, "undone": None, "status": "none of your strokes are on "
                                                                   "the canvas, so there was nothing to undo"}
            return {"success": True, "undone": undone, "remaining": left,
                    "revision": reply.get("revision"),
                    "status": "took back your last stroke" + (
                        "; none of yours are left on it" if left == 0
                        else f"; {left} of yours are still on it" if left else "")}

    def _save(self, name):
        with self._lock:
            if not self._attach():   # never open a window just to save it; it'd come back anyway
                return {"success": False, "error": "your canvas isn't open, so there's nothing to "
                                                   "save; your last drawing comes back when you open it"}
            reply = self._seen.reply = self._client.save(author=self.author, name=name or None)
            return {"success": True, "saved_as": reply.get("name"), "path": reply.get("path"),
                    "picture": reply.get("png"),
                    "status": f"saved it as {reply.get('name')} in your drawings folder"}

    def _open_saved(self, name):
        with self._lock:
            opened = self._ensure()
            reply = self._seen.reply = self._client.open(name=name or None, author=self.author)
            out = {"success": True, "opened_drawing": reply.get("name"), "strokes": reply.get("strokes"),
                   "opened_canvas": opened, "status": f"{reply.get('name')} is on your canvas now"}
            if reply.get("saved_previous"):
                out["saved_previous"] = os.path.splitext(os.path.basename(reply["saved_previous"]))[0]
                out["status"] += f"; what was on it is saved as {out['saved_previous']}"
            return out

    def _close(self):
        with self._lock:
            app, self._seen.app, self._seen.reply = self._app, self._app, None
            if app is not None:
                app.close()
            self._drop()
        with contextlib.suppress(Exception):
            self._report("close", (), {"success": True}, "ok", None)

    def _guarded(self, fn, *args):
        """fn(*args) as a speakable dict, failures included, with its journal record written."""
        self._seen.app = self._seen.reply = None
        out, outcome, why = self._call(fn, *args)
        with contextlib.suppress(Exception):   # writing it down never costs her the action
            self._report(fn.__name__.lstrip("_"), args, out, outcome, why)
        return out

    def _call(self, fn, *args):
        """(speakable dict, outcome, the exception behind a failure or None)."""
        try:
            launcher, client = _drawapp()
        except ImportError as e:              # drawing-app/ missing or broken: say so, don't raise
            return {"success": False, "error": f"couldn't load the drawing app: {e}"}, "failed", e
        try:
            out = fn(*args)
            return out, "ok" if out.get("success") else "refused", None
        except launcher.LaunchError as e:
            return {"success": False, "error": f"the drawing app didn't open: {e}"}, "failed", e
        except client.CommandRejected as e:
            if e.code == "storage":           # a drawing file: its own words say it best
                return {"success": False, "error": e.error}, "refused", e
            if e.code == "unsupported":       # a window from before saving existed: no autosave
                return {"success": False, "error": "this canvas window is from before saving "
                                                   "existed, so it can't save or open drawings, and "
                                                   "what's on it lives only in that window; leave it "
                                                   "open, closing it would lose the drawing"}, "refused", e
            return {"success": False, "error": f"the canvas refused it ({e.code}): {e.error}"}, "refused", e
        except client.DrawError as e:         # a window that went silent may have done it anyway
            outcome = "unknown" if isinstance(e, client.DrawConnectionError) else "failed"
            return {"success": False, "error": f"couldn't reach the drawing app: {e}"}, outcome, e
        except (TypeError, ValueError) as e:
            return {"success": False, "error": "points must be a list of [x, y] number pairs"}, "refused", e

    def _report(self, action, args, out, outcome, why):
        """One journal record for this action: enough to redraw it, or see why it wasn't drawn."""
        reply = getattr(self._seen, "reply", None) or {}
        app = getattr(self._seen, "app", None)
        rec = {"ts": datetime.now().isoformat(timespec="milliseconds"), "seq": next(self._seq),
               "process": os.getpid(), "author": self.author, "canvas": self.canvas,
               "action": action, "outcome": outcome}
        if action == "stroke":
            rec.update(_stroke_facts(self.author, *args))
        elif action in ("save", "open_saved"):
            rec["name"] = args[0] if args else None
        rec["stroke_id"] = out.get("undone") if action == "undo" else out.get("stroke_id")
        rec["revision"] = out.get("revision", reply.get("revision"))
        for key in ("remaining", "saved_as", "path", "opened_drawing", "strokes", "status"):
            if key in out:
                rec[key] = out[key]
        opened = out.get("opened_canvas", out.get("opened"))
        if opened is not None:
            rec["opened_canvas"] = opened
        rec["window"] = (None if app is None else
                         {"pid": getattr(app, "pid", None), "port": getattr(app, "port", None),
                          "adopted": getattr(app, "adopted", None)})
        if outcome != "ok":
            rec["error"] = out.get("error")
            rec["error_type"] = type(why).__name__ if why is not None else None
            if getattr(why, "code", None):
                rec["code"] = why.code
        self._journal(rec)

    # --- the async surface (tools call these) -------------------------------------------------------

    async def _run(self, fn, *args):
        return await asyncio.to_thread(self._guarded, fn, *args)

    async def open(self):
        """Launch or reuse her canvas, off the event loop."""
        return await self._run(self._open)

    async def stroke(self, points, color=None, width=None):
        return await self._run(self._stroke, points, color, width)

    async def clear(self):
        return await self._run(self._clear)

    async def undo(self):
        """Take back her own newest stroke (never anyone else's); never opens a window."""
        return await self._run(self._undo)

    async def save(self, name=None):
        """Save her canvas into the drawings folder (to its own file, or a new one called `name`)."""
        return await self._run(self._save, name)

    async def open_saved(self, name=None):
        """Put a saved drawing on her canvas (default: the newest), opening the canvas if needed."""
        return await self._run(self._open_saved, name)

    async def close(self):
        """End the window, for her sister too: it's their one canvas. Nothing of hers calls this."""
        await asyncio.to_thread(self._close)


# --- one board per sister in this process --------------------------------------------------------

_BOARDS = {}
_BOARDS_LOCK = threading.Lock()


def board_for(author="aiva", **kwargs):
    """The DrawingBoard for `author` in this process, made on first use (`kwargs` go to it then
    and are ignored after). Her tools and anything else in her process share it, and every board
    (her sister's, in her own process, too) draws on the one shared canvas window."""
    with _BOARDS_LOCK:
        if author not in _BOARDS:
            _BOARDS[author] = DrawingBoard(author=author, **kwargs)
        return _BOARDS[author]


def launch_or_reuse(author="aiva"):
    """Open `author`'s canvas or reuse the open one (blocking; see DrawingBoard.launch_or_reuse)."""
    return board_for(author).launch_or_reuse()


def save_drawing(author="aiva", name=None):
    """Save `author`'s open canvas into the drawings folder (blocking; see DrawingBoard.save_now)."""
    return board_for(author).save_now(name)
