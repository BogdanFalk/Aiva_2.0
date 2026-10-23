# drawapp — a shared canvas for Aiva and Saga

A small standalone desktop drawing app. It is **not part of the Aiva voice
app**: it does not import from `aiva/`, and `aiva/` reaches it only through
its stdlib launcher and client (`aiva/modules/drawing.py`, see *Drawing from
code*), never its Qt modules. It only shares the repo's Python venv.

v0.5 is a white canvas you draw on with a round brush (mouse, pen or touch),
an **Undo** button (`Ctrl+Z`, *Undo* below), a **Clear** button
(`Ctrl+Shift+Del`), and saving: **Open… / Save / Save As…**
in the toolbar, plus a canvas that keeps itself (*Saving and recovery*). Code
can open the window, draw on it and save it too, without a console
(*Drawing from code*).

## Run

From this directory, using the repo's venv (Python 3.11, PyQt5 5.15.11 — the
same Qt binding the avatar renderer uses):

```bat
..\.venv\Scripts\python -m drawapp                      :: open the window
..\.venv\Scripts\python -m drawapp --author saga        :: strokes tagged as Saga's (pink ink)
..\.venv\Scripts\python -m drawapp --render-demo out.png :: headless: draw a demo, save a PNG, exit
..\.venv\Scripts\python -m drawapp --listen             :: also take drawing commands on 127.0.0.1:8770
..\.venv\Scripts\python -m drawapp --save-dir D:\art    :: save and restore drawings there instead
..\.venv\Scripts\python -m drawapp --canvas shared      :: the sisters' one canvas (you draw as "user")
..\.venv\Scripts\python -m drawapp.verify              :: clean-launch check (see below); exit 0 = all good
..\.venv\Scripts\python -m pytest -q                    :: tests (offscreen, no window opens)
```

A separate venv works too: `pip install -r requirements.txt`.

## Saving and recovery

**Where drawings go:** `%USERPROFILE%\Pictures\Aiva and Saga Drawings\`
(change it with `--save-dir`, or `launch(save_dir=…)` from code). The status
bar shows the folder and whether the drawing is saved; hover **Save** for the
full path.

| control | what it does |
|---|---|
| **Save** (`Ctrl+S`) | First time: writes `<author>-<date>-<time>.aivadraw` plus a `.png` of it into the folder. After that it updates the same file, and the previous version is kept as `<name>.aivadraw.bak`. |
| **Save As…** (`Ctrl+Shift+S`) | Saves under a name you pick. The dialog asks before it replaces a file, and the replaced file is kept as `.bak` anyway. |
| **Open…** (`Ctrl+O`) | Puts a saved drawing on the canvas. If the canvas has unsaved work, that work is saved as a new drawing first, and the status bar names it. |

`.aivadraw` is plain JSON (canvas size, background, strokes in paint order),
and the `.png` next to it is for looking at. Saves are atomic (temp file +
`os.replace`), so a crash mid-save leaves the old file or the new one, never
a broken one.

**The canvas keeps itself.** Nobody has to press Save for a drawing to survive:

- Every finished stroke, every undo and every clear rewrites
  `recovery\<author>-autosave.aivadraw` straight away. So a window closed by
  hand, killed by `launch().close()` (TerminateProcess) or lost to a crash
  loses nothing but a stroke still being drawn.
- **The next window for the same author starts from that copy.** Close
  Aiva's canvas, open it again, and her drawing is back. The status bar says
  "Restored your last drawing (N strokes)". The restored canvas still knows
  its file, so `Ctrl+S` goes back to it. To start over, press Clear.
- **Clear keeps a copy.** Just before a clear, the canvas is copied to
  `recovery\<author>-before-clear.aivadraw`. Open it with **Open…** to get
  the drawing back (it opens as unsaved work, so `Ctrl+S` makes it a new drawing).
- **A copy that can't be restored is never thrown away.** A damaged autosave,
  or one from another canvas size, is renamed
  `<author>-autosave.unrecovered-<time>.aivadraw` and the window starts
  blank, saying so.
- **One canvas per author owns the autosave.** The first window for an
  author holds `recovery\<author>.lock`, an OS lock that goes away when its
  process does, however it ends. A second window for the same author starts
  blank and autosaves to `<author>-autosave-<pid>.aivadraw`, so neither ever
  overwrites the other. The owner also writes `recovery\<author>-window.json`
  (pid and port), which is how code finds the window again (below).
- **A shared canvas is one slot for several painters.** `--canvas shared`
  names the slot (`recovery\shared.lock`, `shared-autosave`,
  `shared-window.json`) apart from `--author`, who the mouse draws as. Aiva and
  Saga both draw on the `shared` canvas, and each command stroke keeps its own
  author. `--absorb aiva --absorb saga` takes in each sister's own old canvas:
  if no window holds it, its autosave is added on top (strokes keep their
  authors), the merged canvas is autosaved, and only then is the old copy
  renamed `<name>-autosave.merged-<time>.aivadraw`. So each one is added once
  and never deleted. An old canvas whose window is still open is left alone,
  and it is added on a later start once that window has closed.
- An autosave that fails (disk full, folder gone) never stops the drawing:
  the status bar says "Autosave failed: …" and the next stroke tries again.

## Undo

**Undo** (`Ctrl+Z`) takes back the newest finished stroke of the window's
author, and nothing else. Press it again to take back the one before. A
stroke still being drawn is never undone, and neither is anyone else's: in
Saga's window, Undo never removes an Aiva stroke sent there by command, and an
Aiva undo never touches Saga's. Mouse strokes are drawn as the window's
author, so in Aiva's own window your hand-drawn strokes count as hers. In the
sisters' shared window the mouse draws as `user`, so your strokes are yours. When
nothing of the author's is left, the status bar says so. An undo is autosaved
like a stroke, so a restored canvas comes back without the undone stroke.
There is no redo yet, and a Clear is not undone (its `before-clear` copy is
how to get it back).

To recover by hand, open the `recovery` folder in the **Open…** dialog and pick
the copy you want. Recovery copies are never listed as saved drawings.

## Drawing by command

`--listen [PORT]` (default 8770; `0` picks a free port and prints it) opens a
loopback-only socket. Send one JSON object per line and get one JSON reply line
per command.

Commands only reach a window started **with** `--listen`; a plain
`python -m drawapp` never listens. **From code, don't use a terminal at all**:
call `launch()` (see *Drawing from code*), which opens the window with no
console and returns once it listens. That's also what Aiva's drawing tools do.
Started by hand, wait for
`drawapp: listening for drawing commands on 127.0.0.1:8770` before sending.
A listening window says so in its title (`· commands on 127.0.0.1:8770`); if
the port is taken it exits with code 1 and names the port, showing no window.
Nothing touches the real mouse or keyboard, and a stroke someone is drawing
by hand at the time is left alone.

```json
{"cmd": "stroke", "author": "aiva", "points": [[100, 100], [300, 180], [500, 120]], "color": "#7b4fd6", "width": 6}
{"cmd": "clear", "author": "saga"}
{"cmd": "undo", "author": "aiva"}
```

`author` defaults to `"user"`, `color` to the author's ink, `width` to 4.
Points are document pixels and must lie on the canvas (default 1024x768).
Replies look like `{"ok": true, "stroke_id": "aiva-…", "revision": 3}` or
`{"ok": false, "code": "invalid" | "unsupported" | "storage" | "internal", "error": "…"}`.
A bad line never draws anything, and the connection stays open.
`undo` takes back that author's newest finished stroke (*Undo*) and replies
`{"stroke_id", "revision", "remaining"}`, where `remaining` counts that author's
strokes still on the canvas. With none left, `stroke_id` is `null`: the reply
is still `"ok": true`, and nothing changed. A v0.3 window answers
`"unsupported"`.

Saving goes over the same wire (*Saving and recovery* has the rules):

```json
{"cmd": "save", "author": "aiva"}                  // its own file, or a new <author>-<time> one
{"cmd": "save", "author": "aiva", "name": "cat"}   // a new cat.aivadraw (+ .png); a taken name gets " (2)"
{"cmd": "open", "author": "aiva", "name": "cat"}   // no "name": the newest saved drawing
{"cmd": "drawings"}                                // {"save_dir", "drawings": [{"name", "path", "modified"}], "recovery": [...]}
{"cmd": "status"}                                  // {"pid", "author", "revision", "strokes", "save_dir", "current", "unsaved"}
```

`save` replies `{"path", "png", "name", "strokes", …}`. `open` takes a name
exactly as `drawings` lists it (`"recovery/aiva-before-clear"` included), never
a path, and replies `"saved_previous"` with the file it saved unsaved work to
first (or `null`). A file problem (a damaged drawing, an unknown name, a full
disk) is `"code": "storage"` and leaves the canvas as it was.

```python
import json, socket
with socket.create_connection(("127.0.0.1", 8770)) as s:
    s.sendall(json.dumps({"cmd": "stroke", "author": "aiva", "points": [[50, 50], [400, 300]]}).encode() + b"\n")
    print(s.makefile().readline())
```

In-process, skip the socket: `CommandRunner(doc).handle({...})` (`commands.py`).

## Drawing from code

`launcher.py` and `client.py` are stdlib only (no Qt), so a process that never
loads PyQt5, such as a sister's, can open the window and draw on it:

```python
from drawapp.launcher import launch

app = launch(author="aiva")   # returns once the window is up and listening
with app.client() as draw:
    draw.stroke([(100, 100), (300, 180), (500, 120)], author="aiva", width=6)
    draw.clear(author="saga")
    draw.undo(author="aiva")                 # takes back aiva's newest stroke, nobody else's
    draw.save(author="aiva", name="waves")   # -> {"path": "...\\waves.aivadraw", "png": ...}
app.close()                   # or leave it open; closing the window ends it too

from drawapp.launcher import find_running
again = find_running("aiva")  # the window still up from before (e.g. her restart), or None

from drawapp.launcher import opening
with opening("shared"):       # the sisters' one canvas: whoever comes first launches it
    both = find_running("shared") or launch(canvas="shared", author="user",
                                            absorb=("aiva", "saga"))
```

- **No terminal.** The child is `pythonw.exe -m drawapp --listen 0 --ready-file … --no-activate`
  started with `CREATE_NO_WINDOW`. A GUI-subsystem interpreter never gets a
  console, so nothing flashes up and nothing waits on a terminal. Don't wrap
  this in a terminal either (`python -c "…launch()…"` typed into PowerShell
  still shows the PowerShell). Call it from the process that wants to draw.
- **Doesn't take the focus.** `--no-activate` shows the window with
  `WA_ShowWithoutActivating`, so whatever the user is typing into keeps the
  keyboard. `launch(activate=True)` gives the old behaviour.
- **No port clash.** `--listen 0` takes a free port. Once the window is shown
  and listening, the app writes `{"pid", "port"}` to the ready file, and
  `launch()` waits for that file. If the app exits first (say a fixed `port=`
  is taken), `launch()` raises `LaunchError` with the end of the app's log.
  The log lives in `%TEMP%\drawapp-*\drawapp.log`.
- **The right process.** A venv's `pythonw.exe` is a redirector that runs the
  real interpreter as its child, so `app.process.pid` is not the app.
  `app.pid` (from the ready file) is, and `close()` ends that one.
- **An existing canvas is found again, not doubled.** A window outlives the
  process that launched it. `find_running(canvas, save_dir=None)` reads that
  canvas's `recovery\<canvas>-window.json`, checks that the pid is alive, and
  asks the port for `status`. It returns a `DrawApp` for the window only if
  both pid and canvas match (`status` reports the canvas as its `"author"`),
  so a stale file, a reused pid or a stranger's port all give `None`. The canvas
  is the window's `--canvas`, or its author when it has its own. An adopted
  `DrawApp` has `process=None` (`.adopted`). `running()` and `close()` work on its pid.
- **One window for a shared canvas, even when two processes open it at once.**
  `opening(canvas, save_dir=None)` holds `recovery\<canvas>.opening.lock` (an OS
  lock) around find-then-launch. A second process waits, and then finds the
  first one's window instead of launching its own.
- **A new window comes back with its drawing.** `launch()` starts from the
  author's autosave, and `app.restored` says how many strokes came back.
- `DrawClient(port)` speaks the unchanged wire. `stroke()`/`clear()` return
  the reply, and an `"ok": false` reply raises `CommandRejected` (`.code`,
  `.error`). `send(dict)` returns any reply as-is. A dead, closed or
  timed-out connection raises `DrawConnectionError`, and the next call
  reconnects. Nothing is retried automatically: a stroke whose reply was lost
  may already be drawn. `undo()`, `save()`, `open()`, `drawings()` and
  `status()` wrap the other commands the same way.
- **In the Aiva app** this is already wired: `aiva/modules/drawing.py`
  (`DrawingBoard`) appends this directory to `sys.path` and gives each sister
  the tools `open_drawing`, `draw_stroke`, `clear_drawing`, `undo_stroke`,
  `save_drawing` and `open_saved_drawing`, in her own ink. `undo_stroke` takes
  back her own last stroke (the `undo` command). It never opens a window, and
  a v0.3 window that refuses it gets "reopen the canvas", which is safe there
  because v0.3 autosaves. **Both sisters draw on one window**, the `shared`
  canvas (`--canvas shared --author user --absorb aiva --absorb saga`). The first
  stroke from either of them opens it under `opening("shared")`, and every later
  call from either sister reuses it. Every command still names her own author,
  so each keeps her ink and her undo. After she restarts, she adopts the window that is still
  up (`find_running("shared")`) instead of opening a second one. A window the user closed
  is reopened on the next call, with their drawing back on it. Her task runner may use them too, so a
  "draw / test the canvas" task never opens a terminal. Each sister process has
  one board (`board_for(author)`), and `launch_or_reuse(author)` is the one call
  that brings the window up from her code. `save_drawing(author, name=None)` saves
  it the same way (the `save` command; never Ctrl+S, never a window opened just
  to save). A window from before v0.3 accepts only `stroke`/`clear` and has no
  autosave, so it can't be saved. Leave it open: closing it loses its drawing. Her generic `launch_app` tool hands
  this app's names ("drawing app", "drawapp", "your canvas") to that board, so
  it never goes looking for an installed program and opens Aseprite or Paint
  instead. Nothing here imports `aiva/`.

### Clean-launch check

`python -m drawapp.verify` runs the path above end to end. It launches with
`launch()`, checks that no console host (conhost/OpenConsole) sits under the
app, and checks that the window is on screen with its port in the title. Then
it checks that the window didn't take the keyboard focus. It draws an Aiva
stroke and a Saga stroke, checks that an off-canvas stroke is refused, that
the drawing saves (`.aivadraw` + `.png`) and that clear works, and closes the
app. It saves into a temp folder of its own (`save_dir` in the report), so it
never touches a sister's real canvas. It prints a JSON report and
exits 0 only if every check passed. It needs no terminal at all:

```bat
..\.venv\Scripts\pythonw -m drawapp.verify --report %TEMP%\drawapp-verify.json
```

Or open `verify_drawing.pyw` by double-clicking it, or from code with
`os.startfile` (Aiva's `open_file` tool does exactly that). `pyw.exe` runs it
with no console, the app runs on the repo `.venv`, and the window stays open
with the two strokes, without focus. The report lands in
`verify-report.json` next to the script. Close the window when you've seen it.

`--offscreen` runs without a window (`QT_QPA_PLATFORM=offscreen`), for when
nobody is at the screen. `--keep-open` leaves the drawing up to look at.

## Architecture

```
pointer ──► CanvasWidget ──► Tool ──► ops ──► Document.apply ──► listeners ──► CanvasWidget.update
 (Qt)        canvas.py      tools.py  ops.py    model.py                            │
                                                                         render.py ◄┘ (paint_document)
```

| module | Qt? | role |
|---|---|---|
| `model.py` | no | `Document` (size, background, strokes in paint order, `revision`), `Stroke`, `Brush`. Changes only through `apply(op)`; `subscribe()` notifies after each op. |
| `ops.py` | no | `BeginStroke`, `ExtendStroke`, `EndStroke`, `ClearCanvas`, `RemoveStroke` (undo) — frozen dataclasses of primitives, JSON-ready via `dataclasses.asdict`. Every op that starts something names its `author`. |
| `tools.py` | no | `Tool` protocol (`press`/`move`/`release` in document coordinates) and `BrushTool`, which turns one gesture into one stroke. |
| `commands.py` | no | `parse()` validates a JSON drawing command (`stroke`, `clear`, `undo`, the saving ones); `CommandRunner` bounds-checks it and drives a `BrushTool` per author, so a command stroke goes through the same path as a mouse stroke. |
| `server.py` | yes | `CommandServer`: opt-in (`--listen`) 127.0.0.1 JSON-lines socket. Python sockets on worker threads, with each line handed to the GUI thread by a queued signal (see *Why not QTcpServer*). |
| `client.py` | no | `DrawClient`: the wire as `stroke()`/`clear()`/`undo()`/`send()` and the saving calls. Stdlib only. |
| `launcher.py` | no | `launch()` -> `DrawApp`: starts the window with no console, waits for its ready file, and `close()` ends it. `find_running()` adopts a window already up. Stdlib only. |
| `storage.py` | no | The `.aivadraw` format (validated whole before anything is applied; loaded by replaying ops) and `Session`: autosave, restore, the per-author owner lock, save/open/list. |
| `verify.py` | no | `python -m drawapp.verify`: the clean-launch check. |
| `render.py` | yes | `paint_document(painter, doc)` and `render_image(doc) -> QImage`, shared by the widget, PNG export and tests. |
| `canvas.py` | yes | `CanvasWidget`: paints the document and forwards left-button pointer events to the current tool. It holds no drawing state. |
| `app.py` | yes | `MainWindow` (scrollable canvas + toolbar) and the CLI. |

The two rules that keep it extensible:

1. **One mutation path.** Only `Document.apply(op)` changes the drawing, and
   ops are plain serializable data. Undo, saving and collaboration all build
   on the op stream rather than on the widget.
2. **Painters are not widgets.** A tool takes coordinates, not Qt events, so
   code can drive it (an avatar drawing, a replayed log, a test) exactly the
   way the mouse does. The canvas repaints for *any* applied op, local or not.

### Why not QTcpServer

Up to v0.1 the listener used `QTcpServer`/`QTcpSocket`. On Windows, Qt
5.15.2's socket notifiers sometimes lost a connection's read notification
permanently. Winsock only posts `FD_READ` again after a `recv`, so the line
stayed unread and the client waited forever. Other connections kept working.
A client that sent its next command right after a stroke's reply (what
`DrawClient` does) hit this every time in some process layouts and never in
others; adding a timer or logging was enough to hide it. Plain sockets on
worker threads avoid Qt's notifiers entirely. The wire, the replies and "one
command is applied whole on the GUI thread" are unchanged.
`test_launch_draws_and_closes_a_real_child` is the regression: it fails
against the old server.

## Roadmap

Each step is additive. The seam it uses is in brackets.

1. **Colors and brush size**: a palette and size slider in the toolbar that
   set `canvas.tool.brush`. Per-author default ink is already in
   `AUTHOR_COLORS`. [`Brush`, `BrushTool.brush`]
2. **More tools**: eraser (a stroke with `composition: "clear"` in `Brush`),
   line, rectangle, fill. Each is a new `Tool` class plus a toolbar action
   that calls `canvas.set_tool(...)`. [`Tool` protocol]
3. **Undo/redo**: per-author undo is done in v0.4 (`RemoveStroke`,
   `Document.last_stroke`, the `undo` command, *Undo*). Next: redo, which keeps
   the removed strokes per author until that author draws again, and an
   undoable `ClearCanvas`, which keeps the cleared strokes in the history.
   [`ops`, `Document.apply`]
4. **Saving**: done in v0.3 (`storage.py`, *Saving and recovery*). `.aivadraw`
   is a snapshot of the strokes rather than the op log, because a log would
   keep every cleared stroke forever. Loading still replays ops. Next steps
   here are thumbnails in the Open dialog and a "New" action.
   [`storage.py`, `ops`, `render.py`]
5. **Layers**: `Document.layers: list[Layer]`, ops gain a `layer_id`, and
   `paint_document` walks layers bottom-up. Cache each layer to a `QImage`
   so a repaint doesn't re-stroke everything; v0.1 repaints all strokes per
   event, which is fine at sketch scale. [`model.py`, `render.py`]
6. **Multi-painter collaboration**: since v0.5, Aiva, Saga and the user already
   draw on one canvas from their own processes. It is one window (`--canvas
   shared`) that every process sends commands to. What's left is a small
   loopback hub (one owner `Document`, clients send and receive ops as JSON
   lines), so a painter can watch the others' ops live. Stroke ids are already
   `author-uuid`, so they don't collide across processes; a clear mid-stroke
   is already handled (the other painter's gesture is dropped). Each remote
   painter would get a cursor overlay in their author color.
   [`subscribe`, `apply`, ops]
7. **Avatar drawing**: a sister's LLM tool call drives a `BrushTool`. The
   first version exists: `draw_stroke` (`aiva/modules/drawing.py`) launches
   the window and sends point strokes, and both sisters share the window
   (v0.5). What's left is shapes, so she doesn't have to compute the points.
   The link stays at the edge, through the op protocol;
   this app never imports `aiva/`. [`Tool` protocol]
