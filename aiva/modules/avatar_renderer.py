"""Aiva's body — a self-owned Live2D renderer process (replaces VTube Studio,
the Spout overlay exe, and the VB-Cable lip-sync detour).

One transparent, click-through, always-on-top window renders ariu.moc3
directly via live2d-py (PyQt5 QOpenGLWidget). Aiva talks to it over a
localhost TCP socket (newline-JSON, see modules/avatar_ipc.py); the renderer
is the server and the sole writer of aiva/avatar_state.json, which is how a
restarted Aiva finds and adopts a body that never went away.

Architecture (all in this file):
  MouthEnvelope — lip-sync: RMS chunks queue up ahead of playback and are
                  consumed at wall-clock rate once speech actually starts.
                  Pure logic, unit-testable, no Qt/GL.
  Placement     — x/y (-1..1), rotation (deg), size (-100..100): the exact
                  value semantics VTube Studio's move API had, so Aiva's
                  learned habits keep working. Lerps over 0.2s like VTS did.
  Expressions   — the model's *.exp3.json toggles (Chinese names = file
                  stems); active params are re-applied every frame after
                  Update() because Cubism restores saved params each update.
  IpcServer     — daemon thread; mutations go to a queue the GL loop drains,
                  reads are answered from a lock-guarded snapshot.
  AvatarWindow  — the QOpenGLWidget; QTimer-paced paintGL owns ALL model
                  state. Runs on the Qt main thread of this process.

Standalone demo (never starts Aiva):   cd aiva && python -m modules.avatar_renderer --demo
  --demo      scripted sweep: mouth, all expressions, placement, sleep, layers
  --windowed  small opaque movable window (GL-vs-transparency triage)
  --selftest  load the model, render 3s, exit 0/1 (CI-ish smoke)
Quit a standalone renderer with Ctrl+C in its console or the `quit` IPC cmd.
"""

import argparse
import ctypes
import json
import math
import os
import queue
import socket
import sys
import threading
import time
import uuid
from collections import deque

from modules import avatar_ipc

# Cubism's view space is normalized by window HEIGHT: half-height = 1.0 view
# units, half-width = the aspect ratio. So y=±1 puts the model center at the
# top/bottom edge with a constant, while x needs the live aspect (portrait vs
# landscape monitors differ). BASE_SCALE calibrated against her old VTube
# Studio on-screen size: size=0 ≈ the size the user is used to (~26% of
# screen height; scale 1.0 would be ~96%).
BASE_SCALE = 0.3

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "avatar", "ariu", "ariu.model3.json")

_EXP_SUFFIX = ".exp3.json"

# live2d-py's renderer shows things VTube Studio hides (upstream issue #138):
# leftover artist layers render as stray lines, and mouth-interior crease art
# leaks through its clipping mask while the lips are closed. Per-model
# workarounds; parts named *.psd / PartSketch* are hidden for every model.
MODEL_FIXUPS = {
    "ariu": {
        # mouth crease/drool art (leaks below the closed lips). Painting it
        # flat skin colour hides the leak; teeth/tongue are other meshes, so
        # the open mouth is unaffected (verified against no-camo renders).
        "camo_drawables": [f"ArtMesh{n}" for n in range(79, 89)],
        "camo_skin": (0.988, 0.933, 0.894),
        # teeth/tongue are CLIPPED drawables — live2d-py never applies their
        # clip masks (nor colour overrides), so they shimmer through the shut
        # lips. Part opacity is the one control that reaches them: fade the
        # whole mouth part toward this floor as the mouth closes (the lip
        # line dims but stays readable, the leak drops below visibility) and
        # bring it back the moment she opens her mouth.
        "mouth_part": "Part20",
        "mouth_part_floor": 0.35,
    },
}


def _log(msg):
    print(f"[avatar] {msg}", flush=True)


# --- lip-sync envelope (no Qt/GL imports — unit-testable) --------------------

class MouthEnvelope:
    """Mouth-openness from streamed RMS chunks.

    TTS audio frames reach Aiva's tap seconds AHEAD of playback (the output
    transport does the pacing), so chunks queue here with their durations and
    are only consumed — at wall-clock rate — while speech is actually playing
    (`speech("start")` comes from BotStartedSpeakingFrame). An underrun closes
    the mouth, which is correct: the transport has nothing to play either.

    Chunks carry RAW speech RMS (int16 normalized to 0..1) plus an optional
    spectral brightness 0..1; the mapping lives here: a noise gate closes the
    mouth in the pauses between words (raw RMS never sits at true zero —
    breath and codec noise), a square-root curve lifts normal speech
    (RMS ~0.03-0.15) into a natural openness range, and brightness becomes a
    ParamMouthForm target — dark/rounded sounds (oo/oh) purse the mouth,
    bright ones (ee/eh, sibilants) widen it: viseme-ish shapes from audio
    alone. Both channels are independently smoothed.
    """

    NOISE_FLOOR = 0.006  # RMS below this is silence/breath — mouth shut

    def __init__(self, gain=2.5, attack_tau=0.03, release_tau=0.09,
                 form_scale=1.0, form_tau=0.06, max_queued_secs=120.0):
        self._lock = threading.Lock()
        self._queue = deque()           # [raw_rms, remaining_secs, brightness]
        self._queued_secs = 0.0
        self._max_queued_secs = max_queued_secs
        self._speaking = False
        self._value = 0.0               # smoothed openness, 0..1
        self._form = 0.0                # smoothed mouth form, -1..1
        self.gain = gain
        self.form_scale = form_scale    # 0 disables the shape channel
        self._attack_tau = attack_tau
        self._release_tau = release_tau
        self._form_tau = form_tau

    def _openness(self, rms):
        """Raw RMS -> target mouth openness 0..1."""
        if rms <= self.NOISE_FLOOR:
            return 0.0
        return min(1.0, self.gain * math.sqrt(rms - self.NOISE_FLOOR))

    def feed(self, chunks):
        """chunks: [[rms 0..1, duration_secs, brightness 0..1?], ...] — from
        the IPC thread. Brightness is optional (older senders omit it)."""
        with self._lock:
            for chunk in chunks:
                try:
                    level, dur = float(chunk[0]), float(chunk[1])
                    bright = float(chunk[2]) if len(chunk) > 2 else 0.5
                except (TypeError, ValueError, IndexError):
                    continue
                if dur <= 0 or self._queued_secs + dur > self._max_queued_secs:
                    continue
                self._queue.append([max(0.0, min(1.0, level)), dur,
                                    max(0.0, min(1.0, bright))])
                self._queued_secs += dur

    def speech(self, state):
        with self._lock:
            if state == "start":
                # queued chunks ARE the head of this utterance — keep them
                self._speaking = True
            elif state == "stop":
                self._speaking = False
                self._queue.clear()
                self._queued_secs = 0.0
            elif state == "interrupt":
                self._speaking = False
                self._queue.clear()
                self._queued_secs = 0.0
                self._value *= 0.5      # visually instant cut, then decay

    def step(self, dt):
        """Consume dt seconds of queued audio; return (openness 0..1,
        mouth form -1..1) for ParamMouthOpenY / ParamMouthForm."""
        with self._lock:
            target = 0.0
            form_target = 0.0
            if self._speaking and self._queue:
                budget = dt
                level = self._queue[0][0]
                bright = self._queue[0][2]
                while budget > 1e-9 and self._queue:
                    head = self._queue[0]
                    take = min(budget, head[1])
                    head[1] -= take
                    budget -= take
                    self._queued_secs -= take
                    level, bright = head[0], head[2]
                    if head[1] <= 1e-9:
                        self._queue.popleft()
                target = self._openness(level)
                if target > 0:
                    # shape only means something while sound is coming out;
                    # silence relaxes the mouth back to neutral
                    form_target = (bright - 0.5) * 2.0 * self.form_scale
            if dt > 0:
                tau = self._attack_tau if target > self._value else self._release_tau
                alpha = 1.0 - math.exp(-dt / tau)
                self._value += (target - self._value) * alpha
                f_alpha = 1.0 - math.exp(-dt / self._form_tau)
                self._form += (form_target - self._form) * f_alpha
            return min(1.0, self._value), max(-1.0, min(1.0, self._form))


# --- placement (VTS value semantics preserved) --------------------------------

class Placement:
    MOVE_SECS = 0.2  # matches VTS move_time so voice moves feel identical

    def __init__(self, saved=None):
        base = {"x": 0.0, "y": 0.0, "rotation": 0.0, "size": 0.0}
        if isinstance(saved, dict):
            for k in base:
                try:
                    base[k] = float(saved.get(k, base[k]))
                except (TypeError, ValueError):
                    pass
        self.cur = dict(base)
        self.target = dict(base)
        self._t_left = 0.0

    def set_targets(self, x=None, y=None, rotation=None, size=None):
        clamps = {"x": 2.0, "y": 2.0, "rotation": 360.0, "size": 100.0}
        for key, val in (("x", x), ("y", y), ("rotation", rotation), ("size", size)):
            if val is not None:
                lim = clamps[key]
                self.target[key] = max(-lim, min(lim, float(val)))
        self._t_left = self.MOVE_SECS

    def tick(self, dt):
        if self._t_left <= 0:
            self.cur = dict(self.target)
            return
        f = min(1.0, dt / self._t_left)
        for k in self.cur:
            self.cur[k] += (self.target[k] - self.cur[k]) * f
        self._t_left -= dt

    def apply(self, model, aspect):
        model.SetOffset(self.cur["x"] * aspect, self.cur["y"])
        model.SetScale(BASE_SCALE * 10 ** (self.cur["size"] / 100.0))
        model.Rotate(self.cur["rotation"])


# --- expression toggles --------------------------------------------------------

class Expressions:
    """The model's *.exp3.json files as VTS-style toggles.

    File stem = the exact (Chinese) hotkey name Aiva's prompt and tools use.
    Each file sets one param to 1.0 (Blend: Add on a 0-default param, i.e.
    plain overwrite). Active params must be re-set EVERY frame after Update().
    """

    def __init__(self, model_dir):
        self.params = {}   # name -> param id
        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith(_EXP_SUFFIX):
                continue
            name = fname[:-len(_EXP_SUFFIX)]
            try:
                with open(os.path.join(model_dir, fname), encoding="utf-8") as f:
                    data = json.load(f)
                params = data.get("Parameters") or []
                if params and params[0].get("Id"):
                    self.params[name] = params[0]["Id"]
            except (OSError, json.JSONDecodeError):
                _log(f"skipping unreadable expression file {fname}")
        self.active = set()

    def match(self, name):
        """Same matching vtube_studio.trigger_hotkey used: exact (casefold),
        then substring — she sometimes says a partial name."""
        want = name.casefold().strip()
        for n in self.params:
            if n.casefold() == want:
                return n
        for n in self.params:
            if want and want in n.casefold():
                return n
        return None

    def toggle(self, matched_name):
        if matched_name in self.active:
            self.active.discard(matched_name)
            return False
        self.active.add(matched_name)
        return True

    def apply(self, model):
        # Drive EVERY known expression param each frame — 1.0 when its toggle is
        # on, 0.0 when off. Only re-setting the active ones would leave a
        # toggled-off param frozen at its last 1.0 (SetParameterValue saves the
        # value), so e.g. her hood could go on but never come off.
        active_ids = {self.params[n] for n in self.active}
        for pid in set(self.params.values()):
            model.SetParameterValue(pid, 1.0 if pid in active_ids else 0.0)


# --- IPC server ----------------------------------------------------------------

class IpcServer(threading.Thread):
    """Owns the TCP socket. One client at a time — a new connection replaces
    the old one, so a restarted Aiva is never blocked by a dead socket.

    mouth/speech go straight to the (thread-safe) envelope; everything else is
    either answered from the window's lock-guarded snapshot or queued for the
    GL loop to apply within one frame.
    """

    def __init__(self, window):
        super().__init__(daemon=True, name="avatar-ipc")
        self._win = window
        self._stop = threading.Event()
        self._conn = None
        self._conn_lock = threading.Lock()
        self._srv = socket.create_server(("127.0.0.1", 0))
        self.port = self._srv.getsockname()[1]
        self.boot_id = uuid.uuid4().hex

    def stop(self):
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass

    def run(self):
        self._srv.listen(1)
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except OSError:
                break  # socket closed during shutdown
            with self._conn_lock:
                if self._conn is not None:
                    try:
                        self._conn.close()  # replaced by the new client
                    except OSError:
                        pass
                self._conn = conn
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        try:
            with conn, conn.makefile("r", encoding="utf-8") as reader:
                for line in reader:
                    reply = self._handle(line)
                    if reply is not None:
                        conn.sendall(avatar_ipc.encode(reply))
        except (OSError, ValueError):
            pass  # client vanished mid-line; the accept loop lives on

    def _handle(self, line):
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return None
        cmd = msg.get("cmd")
        msg_id = msg.get("id")

        # hot path: fire-and-forget, no reply
        if cmd == "mouth":
            self._win.envelope.feed(msg.get("chunks") or [])
            return None
        if cmd == "speech":
            self._win.envelope.speech(msg.get("state", ""))
            return None

        if msg_id is None:
            return None  # every remaining command requires an id

        if cmd == "hello":
            snap = self._win.snapshot()
            return {"id": msg_id, "ok": True, "version": avatar_ipc.PROTOCOL_VERSION,
                    "model": self._win.model_name,
                    "expressions": list(self._win.expressions.params),
                    "ready": snap["ready"]}
        if cmd == "ping":
            return {"id": msg_id, "ok": True}
        if cmd == "get_position":
            snap = self._win.snapshot()
            return {"id": msg_id, "ok": True, "x": snap["x"], "y": snap["y"],
                    "rotation": snap["rotation"], "size": snap["size"],
                    "monitor": snap["monitor"], "layer": snap["layer"]}
        if cmd == "expression":
            matched = self._win.expressions.match(str(msg.get("name", "")))
            if matched is None:
                return {"id": msg_id, "ok": False,
                        "error": f"no expression matching {msg.get('name')!r}"}
            self._win.cmd_q.put(("expression", matched))
            return {"id": msg_id, "ok": True, "toggled": matched}
        if cmd == "move":
            self._win.cmd_q.put(("move", {k: msg.get(k) for k in
                                          ("x", "y", "rotation", "size")}))
            return {"id": msg_id, "ok": True}
        if cmd == "monitor":
            idx = msg.get("index")
            monitors = avatar_ipc.list_monitors()
            if not isinstance(idx, int) or not 0 <= idx < len(monitors):
                return {"id": msg_id, "ok": False,
                        "error": f"monitor {idx} doesn't exist (have {len(monitors)})"}
            self._win.cmd_q.put(("monitor", idx))
            return {"id": msg_id, "ok": True}
        if cmd == "layer":
            mode = msg.get("mode")
            if mode not in ("top", "desktop"):
                return {"id": msg_id, "ok": False, "error": "mode must be top|desktop"}
            self._win.cmd_q.put(("layer", mode))
            return {"id": msg_id, "ok": True}
        if cmd == "sleeping":
            self._win.cmd_q.put(("sleeping", bool(msg.get("value"))))
            return {"id": msg_id, "ok": True}
        if cmd == "prop":
            self._win.cmd_q.put(("prop", {"show": bool(msg.get("show")), "path": msg.get("path"),
                                          "x": msg.get("x"), "y": msg.get("y"),
                                          "scale": msg.get("scale")}))
            return {"id": msg_id, "ok": True, "prop": bool(msg.get("show"))}
        if cmd == "quit":
            self._win.request_quit()
            return {"id": msg_id, "ok": True}
        return {"id": msg_id, "ok": False, "error": f"unknown cmd {cmd!r}"}


# --- the window ------------------------------------------------------------------

# Qt/GL imports live below the pure-logic classes so the envelope can be
# imported (for unit tests) even where PyQt5 isn't installed.
def _qt():
    from PyQt5.QtCore import Qt, QTimer, QMetaObject
    from PyQt5.QtGui import QSurfaceFormat
    from PyQt5.QtWidgets import QApplication, QOpenGLWidget
    return Qt, QTimer, QMetaObject, QSurfaceFormat, QApplication, QOpenGLWidget


_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_NOACTIVATE = 0x08000000
_HWND_TOPMOST = -1
_HWND_BOTTOM = 1
_SWP_ZONLY = 0x0010 | 0x0002 | 0x0001  # NOACTIVATE | NOMOVE | NOSIZE


def build_window_class():
    """AvatarWindow must be defined after QApplication's deps import cleanly;
    building it in a factory keeps module import light for unit tests."""
    Qt, QTimer, QMetaObject, _fmt, QApplication, QOpenGLWidget = _qt()
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QPainter, QPixmap
    from PyQt5.QtWidgets import QWidget
    import live2d.v3 as live2d

    class PropWindow(QWidget):
        """A frameless, translucent, click-through top-level window that shows a
        held prop (her book while reading) in front of the avatar. Kept OUT of
        the GL widget so it can never disturb the model's transparent render."""

        def __init__(self):
            super().__init__()
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setAttribute(Qt.WA_ShowWithoutActivating)
            self.setAttribute(Qt.WA_TransparentForMouseEvents)
            self._pix = None
            self._loaded_path = None

        def load(self, path):
            if path == self._loaded_path and self._pix is not None:
                return True
            try:
                if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    pix = QPixmap(path)
                    if pix.isNull():
                        return False
                else:  # SVG
                    from PyQt5.QtSvg import QSvgRenderer
                    r = QSvgRenderer(path)
                    if not r.isValid():
                        return False
                    size = r.defaultSize()
                    target_h = 1024
                    ratio = (size.width() / size.height()) if size.height() else 1.0
                    pix = QPixmap(int(target_h * ratio), target_h)
                    pix.fill(Qt.transparent)
                    p = QPainter(pix)
                    r.render(p)
                    p.end()
                self._pix = pix
                self._loaded_path = path
                return True
            except Exception as e:
                _log(f"prop load failed for {path}: {e}")
                return False

        def place(self, monitor_rect, frac_x=None, frac_y=None, frac_h=None):
            x, y, w, h = monitor_rect
            frac_h = frac_h if frac_h is not None else float(os.getenv("AIVA_READING_PROP_SCALE", "0.09"))
            frac_y = frac_y if frac_y is not None else float(os.getenv("AIVA_READING_PROP_Y", "0.80"))
            frac_x = frac_x if frac_x is not None else float(os.getenv("AIVA_READING_PROP_X", "0.85"))
            ph = int(h * frac_h)
            ratio = (self._pix.width() / self._pix.height()) if self._pix else 1.0
            pw = int(ph * ratio)
            px = x + int(w * frac_x) - pw // 2
            py = y + int(h * frac_y) - ph // 2
            self.setGeometry(px, py, pw, ph)

        def paintEvent(self, event):
            if self._pix is None:
                return
            p = QPainter(self)
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.drawPixmap(self.rect(), self._pix)
            p.end()

        def apply_click_through(self):
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            style = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            style |= (_WS_EX_LAYERED | _WS_EX_TRANSPARENT | _WS_EX_NOACTIVATE | _WS_EX_TOOLWINDOW)
            user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, style)
            user32.SetWindowPos(hwnd, _HWND_TOPMOST, 0, 0, 0, 0, _SWP_ZONLY)

        def showEvent(self, event):
            super().showEvent(event)
            self.apply_click_through()

    class AvatarWindow(QOpenGLWidget):
        def __init__(self, model_path, fps, sway, windowed, state):
            super().__init__()
            self.model_path = model_path
            self.model_name = os.path.basename(model_path).split(".")[0]
            self.model = None
            self.envelope = MouthEnvelope(
                gain=float(os.getenv("AIVA_LIPSYNC_GAIN", "2.5")),
                form_scale=float(os.getenv("AIVA_LIPSYNC_FORM", "1.0")))
            self.expressions = Expressions(os.path.dirname(model_path))
            self.placement = Placement((state or {}).get("placement"))
            self.cmd_q = queue.Queue()
            self._lock = threading.Lock()
            self._snapshot = {}
            self._windowed = windowed
            self._sway = sway
            self._fps = fps
            self._sleeping = False
            self._ready = False
            self._junk_parts = []
            self._camo_drawables = []
            self._camo_skin = (1.0, 1.0, 1.0)
            self._mouth_part = None
            self._mouth_floor = 1.0
            self._t0 = time.monotonic()
            self._last_frame = self._t0
            self._dirty_at = None      # debounced state persistence
            self._server = None        # set by main() before show()
            self._prop = None          # PropWindow, lazily created (her book while reading)
            _assets = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
            self._prop_path = os.getenv("AIVA_READING_PROP") or (
                os.path.join(_assets, "reading_prop.png")
                if os.path.isfile(os.path.join(_assets, "reading_prop.png"))
                else os.path.join(_assets, "reading_prop.svg"))

            monitors = avatar_ipc.list_monitors()
            saved_mon = (state or {}).get("monitor")
            self.monitor = saved_mon if isinstance(saved_mon, int) and \
                0 <= saved_mon < len(monitors) else 0
            self.layer = (state or {}).get("layer", "top")

            self.setWindowTitle("Aiva Avatar")
            if windowed:
                self.resize(480, 720)
            else:
                self.setWindowFlags(Qt.FramelessWindowHint
                                    | Qt.WindowStaysOnTopHint | Qt.Tool)
                self.setAttribute(Qt.WA_TranslucentBackground)
                self.setAttribute(Qt.WA_ShowWithoutActivating)
                self.setGeometry(*monitors[self.monitor])

            self._timer = QTimer(self)
            self._timer.timeout.connect(self.update)

        # -- thread-safe surface for the IPC thread --
        def snapshot(self):
            with self._lock:
                if self._snapshot:
                    return dict(self._snapshot)
            return {"ready": False, "x": 0, "y": 0, "rotation": 0, "size": 0,
                    "monitor": self.monitor, "layer": self.layer}

        def request_quit(self):
            QMetaObject.invokeMethod(QApplication.instance(), "quit",
                                     Qt.QueuedConnection)

        # -- GL lifecycle (Qt main thread) --
        def initializeGL(self):
            live2d.glInit()
            self.model = live2d.LAppModel()
            self.model.LoadModelJson(self.model_path)
            self.model.SetAutoBlinkEnable(True)
            self.model.SetAutoBreathEnable(True)
            self._compute_model_fixups()
            self._ready = True
            _log(f"model '{self.model_name}' loaded "
                 f"({len(self.expressions.params)} expressions)")
            self._timer.start(max(1, 1000 // self._fps))

        def _compute_model_fixups(self):
            """Resolve MODEL_FIXUPS + junk-part detection to runtime indices.
            Applied every frame in paintGL; failures degrade to no fixups."""
            self._junk_parts = []
            self._camo_drawables = []
            self._camo_skin = (1.0, 1.0, 1.0)
            self._mouth_part = None
            self._mouth_floor = 1.0
            try:
                names = {}
                cdi_path = self.model_path.replace(".model3.json", ".cdi3.json")
                if os.path.isfile(cdi_path):
                    with open(cdi_path, encoding="utf-8") as f:
                        cdi = json.load(f)
                    names = {p.get("Id"): p.get("Name", "")
                             for p in cdi.get("Parts", [])}
                for i, pid in enumerate(self.model.GetPartIds()):
                    if pid.startswith("PartSketch") or \
                            names.get(pid, "").lower().endswith(".psd"):
                        self._junk_parts.append(i)
                fixups = MODEL_FIXUPS.get(self.model_name, {})
                targets = set(fixups.get("camo_drawables", ()))
                if targets:
                    self._camo_skin = fixups.get("camo_skin", self._camo_skin)
                    self._camo_drawables = [
                        i for i, did in enumerate(self.model.GetDrawableIds())
                        if did in targets]
                mouth_pid = fixups.get("mouth_part")
                if mouth_pid:
                    part_ids = self.model.GetPartIds()
                    if mouth_pid in part_ids:
                        self._mouth_part = part_ids.index(mouth_pid)
                        self._mouth_floor = float(os.getenv(
                            "AIVA_MOUTH_PART_FLOOR",
                            str(fixups.get("mouth_part_floor", 0.4))))
                if self._junk_parts or self._camo_drawables or \
                        self._mouth_part is not None:
                    _log(f"model fixups: hiding {len(self._junk_parts)} junk "
                         f"part(s), camouflaging {len(self._camo_drawables)} "
                         f"leaky mouth mesh(es), mouth-part fade "
                         f"{'on' if self._mouth_part is not None else 'off'}")
            except Exception as exc:  # never let a fixup kill the renderer
                _log(f"model fixups skipped: {exc}")
                self._junk_parts = []
                self._camo_drawables = []
                self._mouth_part = None

        def resizeGL(self, w, h):
            dpr = self.devicePixelRatioF()
            if self.model:
                self.model.Resize(int(w * dpr), int(h * dpr))

        def showEvent(self, event):
            super().showEvent(event)
            if not self._windowed:
                self._apply_win32_styles()

        def _apply_win32_styles(self):
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            style = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            style |= (_WS_EX_LAYERED | _WS_EX_TRANSPARENT
                      | _WS_EX_NOACTIVATE | _WS_EX_TOOLWINDOW)
            user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, style)
            self._apply_layer()

        def _apply_layer(self):
            hwnd = int(self.winId())
            z = _HWND_TOPMOST if self.layer == "top" else _HWND_BOTTOM
            ctypes.windll.user32.SetWindowPos(hwnd, z, 0, 0, 0, 0, _SWP_ZONLY)

        def paintGL(self):
            now = time.monotonic()
            dt = max(0.0, min(0.1, now - self._last_frame))
            self._last_frame = now

            self._drain_commands()

            if self._windowed:
                live2d.clearBuffer(0.15, 0.15, 0.18, 1.0)
            else:
                live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)
            if not self.model:
                return

            for i in self._junk_parts:
                self.model.SetPartOpacity(i, 0.0)
            for i in self._camo_drawables:
                self.model.SetDrawableMultiplyColor(i, 0.0, 0.0, 0.0, 1.0)
                self.model.SetDrawableScreenColor(i, *self._camo_skin, 1.0)

            self.model.Update()
            if self._sway and not self._sleeping:
                t = now - self._t0
                self.model.SetParameterValue(
                    "ParamAngleX", 3.0 * math.sin(2 * math.pi * t / 9.3))
                self.model.SetParameterValue(
                    "ParamBodyAngleZ", 2.0 * math.sin(2 * math.pi * t / 12.7))
            self.expressions.apply(self.model)
            mouth_open, mouth_form = self.envelope.step(dt)
            self.model.SetParameterValue("ParamMouthOpenY", mouth_open)
            self.model.SetParameterValue("ParamMouthForm", mouth_form)
            if self._mouth_part is not None:
                t = min(1.0, mouth_open / 0.12)
                self.model.SetPartOpacity(
                    self._mouth_part,
                    self._mouth_floor + (1.0 - self._mouth_floor) * t)
            if self._sleeping:
                self.model.SetParameterValue("ParamEyeLOpen", 0.0)
                self.model.SetParameterValue("ParamEyeROpen", 0.0)
            self.placement.tick(dt)
            self.placement.apply(self.model, self.width() / max(1, self.height()))
            self.model.Draw()

            with self._lock:
                self._snapshot = {
                    "ready": self._ready,
                    "x": round(self.placement.cur["x"], 3),
                    "y": round(self.placement.cur["y"], 3),
                    "rotation": round(self.placement.cur["rotation"], 1),
                    "size": round(self.placement.cur["size"], 1),
                    "monitor": self.monitor, "layer": self.layer,
                }
            if self._dirty_at and now >= self._dirty_at:
                self._persist()

        def _drain_commands(self):
            while True:
                try:
                    kind, arg = self.cmd_q.get_nowait()
                except queue.Empty:
                    return
                if kind == "expression":
                    on = self.expressions.toggle(arg)
                    _log(f"expression {arg} -> {'on' if on else 'off'}")
                elif kind == "move":
                    self.placement.set_targets(**arg)
                    self._mark_dirty()
                elif kind == "monitor":
                    self.monitor = arg
                    if not self._windowed:
                        self.setGeometry(*avatar_ipc.list_monitors()[arg])
                        self._apply_layer()
                    self._mark_dirty()
                elif kind == "layer":
                    self.layer = arg
                    if not self._windowed:
                        self._apply_layer()
                    self._mark_dirty()
                elif kind == "sleeping":
                    self._sleeping = arg
                    if self.model:
                        self.model.SetAutoBlinkEnable(not arg)
                elif kind == "prop":
                    self._toggle_prop(arg)

        def _toggle_prop(self, arg):
            if self._windowed:
                return  # the GL-triage window has no monitor geometry to anchor to
            show = bool(arg.get("show"))
            path = arg.get("path") or self._prop_path
            if not show:
                if self._prop:
                    self._prop.hide()
                return
            try:
                if self._prop is None:
                    self._prop = PropWindow()
                if not self._prop.load(path):
                    _log(f"prop not shown: couldn't load {path}")
                    return
                monitors = avatar_ipc.list_monitors()
                rect = monitors[min(self.monitor, len(monitors) - 1)]
                self._prop.place(rect, arg.get("x"), arg.get("y"), arg.get("scale"))
                self._prop.show()
                self._prop.raise_()
                self._prop.update()
            except Exception as e:
                _log(f"prop show failed: {e}")

        def _mark_dirty(self):
            self._dirty_at = time.monotonic() + 2.0  # debounce disk writes

        def _persist(self):
            self._dirty_at = None
            if not self._server:
                return
            avatar_ipc.write_state_atomic({
                "port": self._server.port, "pid": os.getpid(),
                "boot_id": self._server.boot_id,
                "placement": dict(self.placement.target),
                "monitor": self.monitor, "layer": self.layer,
            })

    return AvatarWindow, live2d


# --- demo script -----------------------------------------------------------------

def _run_demo(window):
    """Scripted sweep through everything the IPC can do, using the same code
    paths real commands take. Runs on the Qt thread via QTimer.singleShot."""
    from PyQt5.QtCore import QTimer

    steps = []

    def at(ms, label, fn):
        steps.append((ms, label, fn))

    def mouth_sweep():
        # realistic speech RMS (0..0.15) incl. gate dips, with brightness
        # sweeping dark->bright so the viseme shapes show too
        chunks = [[0.075 + 0.075 * math.sin(i / 3.0), 0.05,
                   0.5 + 0.5 * math.sin(i / 7.0)] for i in range(80)]
        window.envelope.feed(chunks)
        window.envelope.speech("start")

    at(500, "mouth sweep (4s sine)", mouth_sweep)
    at(5000, "mouth stop", lambda: window.envelope.speech("stop"))
    t = 5500
    for name in list(window.expressions.params):
        at(t, f"expression on: {name}",
           lambda n=name: window.cmd_q.put(("expression", n)))
        at(t + 600, f"expression off: {name}",
           lambda n=name: window.cmd_q.put(("expression", n)))
        t += 1200
    corners = [(-0.8, -0.8, -40), (0.8, -0.8, 0), (0.8, 0.8, 40), (-0.8, 0.8, 0),
               (0.0, 0.0, 0)]
    for x, y, size in corners:
        at(t, f"move x={x} y={y} size={size}",
           lambda x=x, y=y, s=size: window.cmd_q.put(
               ("move", {"x": x, "y": y, "rotation": None, "size": s})))
        t += 1500
    at(t, "sleeping on", lambda: window.cmd_q.put(("sleeping", True)))
    at(t + 2000, "sleeping off", lambda: window.cmd_q.put(("sleeping", False)))
    at(t + 2500, "layer -> desktop", lambda: window.cmd_q.put(("layer", "desktop")))
    at(t + 4500, "layer -> top", lambda: window.cmd_q.put(("layer", "top")))
    at(t + 5000, "demo complete — window stays up, IPC live", lambda: None)

    for ms, label, fn in steps:
        QTimer.singleShot(ms, (lambda l=label, f=fn:
                               (_log(f"demo: {l}"), f())))


# --- process entry -----------------------------------------------------------------

def _hello_probe(state, timeout=2.0):
    """True if the renderer described by `state` answers a hello."""
    if not state or not state.get("port"):
        return False
    try:
        with socket.create_connection(("127.0.0.1", state["port"]), timeout=timeout) as s:
            s.sendall(avatar_ipc.encode({"id": 0, "cmd": "hello"}))
            s.settimeout(timeout)
            line = s.makefile("r", encoding="utf-8").readline()
            return bool(line) and json.loads(line).get("ok") is True
    except (OSError, json.JSONDecodeError, ValueError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aiva's Live2D avatar renderer")
    parser.add_argument("--model", default=os.getenv("AIVA_AVATAR_MODEL", DEFAULT_MODEL))
    parser.add_argument("--fps", type=int,
                        default=int(os.getenv("AIVA_AVATAR_FPS", "32")))
    parser.add_argument("--monitor", type=int, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    # single-instance guard: a live renderer already owns the desktop
    state = avatar_ipc.read_state()
    if state and state.get("pid") and avatar_ipc.pid_alive(state["pid"]) \
            and _hello_probe(state):
        _log(f"renderer already running (pid {state['pid']}) — exiting")
        return 0

    if not os.path.isfile(args.model):
        _log(f"model not found: {args.model}")
        return 3

    # DPI awareness BEFORE QApplication: physical pixels everywhere, matching
    # the EnumDisplayMonitors rects we position with
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        pass
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

    Qt, QTimer, _QMeta, QSurfaceFormat, QApplication, _QGL = _qt()
    fmt = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    fmt.setStencilBufferSize(8)
    fmt.setSwapInterval(0)  # the QTimer paces frames, not vsync
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv[:1])
    AvatarWindow, live2d = build_window_class()
    live2d.init()

    sway = os.getenv("AIVA_AVATAR_SWAY", "1") == "1"
    window = AvatarWindow(args.model, args.fps, sway, args.windowed, state)
    if args.monitor is not None:
        window.monitor = args.monitor
        if not args.windowed:
            monitors = avatar_ipc.list_monitors()
            window.setGeometry(*monitors[min(args.monitor, len(monitors) - 1)])

    server = IpcServer(window)
    window._server = server
    server.start()
    window._persist()  # publish port/pid/boot_id before anyone can look
    _log(f"IPC listening on 127.0.0.1:{server.port} (pid {os.getpid()})")

    window.show()
    if args.demo:
        _run_demo(window)
    if args.selftest:
        QTimer.singleShot(3000, app.quit)

    try:
        app.exec_()
    finally:
        server.stop()
        window._persist()
        try:
            live2d.dispose()
        except Exception:
            pass

    if args.selftest:
        ok = window.snapshot().get("ready")
        _log(f"selftest {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
