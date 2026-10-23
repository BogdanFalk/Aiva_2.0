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
  Activity      — (modules/avatar_activity.py) what her body is doing: the
                  book she pulls up while she reads, the work panel beside
                  her while she's on a task. Pure choreography; the book and
                  panel are painted by paint_book/paint_panel INTO the GL
                  frame after the model, anchored through her placement.
  Constellation — (modules/constellation.py) Saga's quiet witness: a faint sky of
                  stars to her upper left, one lit per lived moment. Pure logic; painted
                  by paint_sky in SkyWindow. Only an instance that has one (`enabled_for`).
  IpcServer     — daemon thread; mutations go to a queue the GL loop drains,
                  reads are answered from a lock-guarded snapshot.
  AvatarWindow  — the QOpenGLWidget; QTimer-paced paintGL owns ALL model
                  state. Runs on the Qt main thread of this process.

Standalone demo (never starts Aiva):   cd aiva && python -m modules.avatar_renderer --demo
  --demo      scripted sweep: mouth, all expressions, placement, sleep, layers
  --demo-activity  loop what her body does while she reads (the book) and works
                   (the work panel); pair with --instance demo next to a live sister
  --demo-witness   loop moments into Saga's constellation (one of each kind, a burst, the hush
                   while she speaks, sleep, a tilt, both layers); needs --instance saga, and uses a temp state file
                   unless --state is given, so made-up moments never reach her real sky
  --windowed  small opaque movable window (GL-vs-transparency triage)
  --selftest  load the model, render 3s, exit 0/1 (CI-ish smoke); a body with a witness
              must also paint its sky. Exits 4 without testing if a live renderer owns the
              state file — pass --state <a temp file> to run beside it
  --instance NAME   which sister this body belongs to (default AIVA_AVATAR_INSTANCE
                    or "aiva"); picks the state file, window title and log prefix
  --state PATH      explicit state file (default: avatar_ipc.state_path(instance))
  --write-model NAME  write NAME.model3.json beside --model, pointing at the
                    NAME.2048/ textures (a retexture of the same mesh), then exit
Quit a standalone renderer with Ctrl+C in its console or the `quit` IPC cmd.

One process = one instance = one model = one state file. Two sisters are two
renderer processes: `--instance saga --model .../saga.model3.json` owns
aiva/saga_avatar_state.json and never collides with Aiva's body.
"""

import argparse
import ctypes
import json
import math
import os
import queue
import socket
import sys
import tempfile
import threading
import time
import uuid
from collections import deque

from modules import avatar_activity
from modules import avatar_ipc
from modules import constellation

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
# Saga is a retexture of the same ariu.moc3 mesh (saga.2048/*), so every
# mesh/part fixup carries over verbatim — only the skin colour under the lips
# differs. camo_skin was sampled with sample_skin_rgb() on the face atlas
# (texture_06 — the only sheet with large flat skin fills; 229 dead-flat 32px
# blocks vs <=53 anywhere else) at box=(224, 96, 32, 32): ariu's sheet reads
# (0.988, 0.941, 0.898) there, within 0.008 of the hand-tuned ariu value above
# (the calibration check), and saga's sheet reads (1.000, 0.929, 0.929) —
# warmer/pinker, as the mean colour shift across her atlases suggested.
# Re-sample after any texture change; AIVA_CAMO_SKIN="r,g,b" overrides live.
MODEL_FIXUPS["saga"] = dict(MODEL_FIXUPS["ariu"], camo_skin=(1.0, 0.929, 0.929))

_INSTANCE = "aiva"   # set once by main(); every log line names the body


def _set_instance(name):
    global _INSTANCE
    _INSTANCE = name or "aiva"


def _log(msg):
    print(f"[avatar:{_INSTANCE}] {msg}", flush=True)


# --- pure helpers (no Qt/GL at import — unit-testable) --------------------------

def parse_rgb(text):
    """'r,g,b' -> normalised (r, g, b) or None. Accepts 0..1 floats or
    0..255 ints (AIVA_CAMO_SKIN override for tuning in --windowed)."""
    if not text:
        return None
    try:
        parts = [float(p) for p in str(text).replace(";", ",").split(",")]
    except ValueError:
        return None
    if len(parts) != 3 or any(p < 0 for p in parts):
        return None
    if max(parts) > 1.0:
        parts = [p / 255.0 for p in parts]
    if max(parts) > 1.0:
        return None
    return tuple(round(p, 3) for p in parts)


def sample_skin_rgb(png_path, box):
    """Mean colour of the fully opaque pixels inside box=(x, y, w, h) of a
    texture sheet, as a normalised (r, g, b) rounded to 3 dp.

    Used by hand, once per model, to derive its `camo_skin` (see the saga
    entry in MODEL_FIXUPS for the exact box). QImage needs no QApplication and
    is imported lazily so the module stays light for tests.
    """
    from PyQt5.QtGui import QImage
    img = QImage(str(png_path))
    if img.isNull():
        raise ValueError(f"can't read image {png_path}")
    img = img.convertToFormat(QImage.Format_RGBA8888)
    x, y, w, h = (int(v) for v in box)
    if w <= 0 or h <= 0 or x < 0 or y < 0 or \
            x + w > img.width() or y + h > img.height():
        raise ValueError(f"box {box} lies outside {img.width()}x{img.height()}")
    r = g = b = n = 0
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            c = img.pixelColor(xx, yy)
            if c.alpha() < 255:
                continue
            r += c.red()
            g += c.green()
            b += c.blue()
            n += 1
    if not n:
        raise ValueError(f"box {box} has no fully opaque pixels")
    return tuple(round(v / n / 255.0, 3) for v in (r, g, b))


def make_retexture_model3(src_model3, textures_dir):
    """A model3 dict for the SAME mesh with its textures in another folder.

    Copies FileReferences (Moc, Physics, DisplayInfo, ...) and Groups
    untouched; only each texture's directory component is rewritten to
    `textures_dir` (file names kept). The source dict is not mutated.
    """
    out = json.loads(json.dumps(src_model3))  # deep copy; model3 is plain JSON
    refs = out.setdefault("FileReferences", {})
    folder = str(textures_dir).replace("\\", "/").rstrip("/")
    rewritten = []
    for tex in refs.get("Textures") or []:
        name = str(tex).replace("\\", "/").rsplit("/", 1)[-1]
        rewritten.append(f"{folder}/{name}" if folder else name)
    refs["Textures"] = rewritten
    return out


def cdi3_path_for(model_path):
    """The model's display-info (cdi3) file: FileReferences.DisplayInfo from
    the model3 json when it names an existing file, else the historical
    <stem>.cdi3.json rule (kept as the fallback)."""
    try:
        with open(model_path, encoding="utf-8") as f:
            refs = json.load(f).get("FileReferences") or {}
        info = refs.get("DisplayInfo")
        if info:
            cand = os.path.join(os.path.dirname(model_path), str(info))
            if os.path.isfile(cand):
                return cand
    except (OSError, ValueError, AttributeError):
        pass
    return model_path.replace(".model3.json", ".cdi3.json")


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

    @property
    def speaking(self):
        """True between speech start and stop/interrupt (her voice is playing)."""
        return self._speaking

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

    def drives(self, param_id):
        """True while a toggled-on expression holds this param at 1 (e.g. 手柄 -> shoubing)."""
        return any(self.params.get(n) == param_id for n in self.active)


# --- activity props: her book and her work panel ---------------------------------------------
# Painted with QPainter INTO her GL scene right after model.Draw(), in model units mapped through
# her live placement (avatar_activity.model_to_px) — so they follow her wherever she stands and
# can't fall out of z-order with her body the way a separate top-level window did. Module-level
# (lazy Qt imports) so they can be exercised offscreen without a window.

def hue_rotate_rgb(rgb_array, degrees):
    """Rotate the hue of an (..., 3) float RGB array (W3C feColorMatrix hueRotate): greys —
    the book's white pages — stay put, the teal cover turns. Pure numpy."""
    import numpy as np
    a, b = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    m = np.array([
        [0.213 + a * 0.787 - b * 0.213, 0.715 - a * 0.715 - b * 0.715, 0.072 - a * 0.072 + b * 0.928],
        [0.213 - a * 0.213 + b * 0.143, 0.715 + a * 0.285 + b * 0.140, 0.072 - a * 0.072 - b * 0.283],
        [0.213 - a * 0.213 - b * 0.787, 0.715 - a * 0.715 + b * 0.715, 0.072 + a * 0.928 + b * 0.072],
    ])
    return np.clip(rgb_array @ m.T, 0.0, 255.0)


class PropArt:
    """The held-book image: loaded once (png/jpg/webp, or svg), shrunk to a working size,
    optionally hue-rotated for a sister, and cached smooth-scaled at the width she holds it."""

    MAX_W = 900

    def __init__(self, path, hue=0):
        self.path = path
        self.image = self._load(path, hue)
        self._scaled = {}

    @classmethod
    def _load(cls, path, hue):
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QImage, QPainter
        if not path or not os.path.isfile(path):
            return None
        if path.lower().endswith(".svg"):
            from PyQt5.QtSvg import QSvgRenderer
            r = QSvgRenderer(path)
            if not r.isValid():
                return None
            size = r.defaultSize()
            ratio = (size.width() / size.height()) if size.height() else 2.0
            img = QImage(cls.MAX_W, max(1, int(cls.MAX_W / ratio)), QImage.Format_ARGB32_Premultiplied)
            img.fill(Qt.transparent)
            p = QPainter(img)
            r.render(p)
            p.end()
        else:
            img = QImage(path)
            if img.isNull():
                return None
            if img.width() > cls.MAX_W:
                img = img.scaledToWidth(cls.MAX_W, Qt.SmoothTransformation)
        img = img.convertToFormat(QImage.Format_ARGB32_Premultiplied)
        if hue:
            img = cls._hue(img, hue)
        return img

    @staticmethod
    def _hue(img, hue):
        try:
            import numpy as np
            from PyQt5.QtGui import QImage
            w, h = img.width(), img.height()
            ptr = img.constBits()
            ptr.setsize(img.byteCount())
            arr = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine())[:, :w * 4]
            arr = arr.reshape(h, w, 4).astype(np.float32)        # BGRA, premultiplied
            rgb = hue_rotate_rgb(arr[..., 2::-1], hue)           # BGR -> RGB, rotate
            out = np.empty((h, w, 4), np.uint8)
            out[..., 2::-1] = np.minimum(rgb, arr[..., 3:4]).astype(np.uint8)  # stay premultiplied
            out[..., 3] = arr[..., 3].astype(np.uint8)
            buf = out.tobytes()                                  # QImage borrows it until copy()
            return QImage(buf, w, h, w * 4, QImage.Format_ARGB32_Premultiplied).copy()
        except Exception as e:  # a tint is cosmetic — never lose the book over it
            _log(f"book tint skipped: {e}")
            return img

    def at_width(self, width):
        from PyQt5.QtCore import Qt
        width = max(4, int(round(width)))
        img = self._scaled.get(width)
        if img is None and self.image is not None:
            if len(self._scaled) > 16:
                self._scaled.clear()
            img = self._scaled[width] = self.image.scaledToWidth(width, Qt.SmoothTransformation)
        return img


def paint_book(p, art, book, placement, w, h, base_scale):
    """Her book, centred on her hands. p: an active QPainter on her GL widget."""
    from PyQt5.QtCore import QPointF
    img = art.at_width(avatar_activity.model_len_px(book["w"], placement, h, base_scale))
    if img is None:
        return
    cx, cy = avatar_activity.model_to_px((book["x"], book["y"]), placement, w, h, base_scale)
    p.save()
    p.translate(cx, cy)
    # placement rotation is CCW on screen, QPainter.rotate is clockwise (y points down)
    p.rotate(-(float(placement.get("rotation", 0.0)) + book.get("rot", 0.0)))
    p.setOpacity(max(0.0, min(1.0, book["alpha"])))
    p.drawImage(QPointF(-img.width() / 2, -img.height() / 2), img)
    p.restore()


_PANEL_WORDS = {"planning": "planning…", "running": "working…", "checking": "checking…",
                "waiting": "waiting for you", "done": "done", "failed": "didn't work"}


def paint_panel(p, pv, placement, w, h, base_scale, accent, show_text=True):
    """Her work panel: a small glass screen beside her head that types while she works, fills a
    progress bar step by step, turns amber and pulses while a task waits on him, and flashes a
    check (or a cross) when it ends. pv: avatar_activity's panel view."""
    from PyQt5.QtCore import QRectF, QPointF, Qt
    from PyQt5.QtGui import QColor, QFont, QFontMetricsF, QLinearGradient, QPen

    A = avatar_activity
    state = pv["state"]
    acc = {"waiting": A.WAITING_ACCENT, "done": A.DONE_ACCENT, "failed": A.FAILED_ACCENT}.get(state, accent)
    t = pv["t"]

    def col(rgb, a):
        return QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255),
                      int(max(0.0, min(1.0, a)) * 255))

    cx, cy = A.model_to_px((pv["cx"], pv["cy"]), placement, w, h, base_scale)
    pw = A.model_len_px(pv["w"], placement, h, base_scale)
    ph = A.model_len_px(pv["h"], placement, h, base_scale)
    if pw < 8 or ph < 6:
        return
    pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 0.9) if state == "waiting" else 0.0

    p.save()
    p.translate(cx, cy)
    p.rotate(-float(placement.get("rotation", 0.0)))
    p.setOpacity(max(0.0, min(1.0, pv["alpha"])))
    p.scale(1.0, max(0.03, pv["open"]))   # it materialises out of a bright line
    r = QRectF(-pw / 2, -ph / 2, pw, ph)
    rad = ph * 0.09

    # glow, then glass, then the rim
    p.setBrush(Qt.NoBrush)
    for grow, a in ((ph * 0.09, 0.07), (ph * 0.05, 0.12), (ph * 0.025, 0.2)):
        p.setPen(QPen(col(acc, a * (1.0 + 0.8 * pulse)), grow * 2))
        p.drawRoundedRect(r, rad, rad)
    g = QLinearGradient(0, -ph / 2, 0, ph / 2)
    g.setColorAt(0.0, QColor(12, 32, 38, 200))
    g.setColorAt(1.0, QColor(6, 16, 22, 176))
    p.setBrush(g)
    p.setPen(QPen(col(acc, 0.85 + 0.15 * pulse), max(1.0, ph * 0.016)))
    p.drawRoundedRect(r, rad, rad)

    pad = ph * 0.09
    left, right, top, bottom = -pw / 2 + pad, pw / 2 - pad, -ph / 2 + pad, ph / 2 - pad
    icon = ph * 0.17
    icon_r = QRectF(left, top, icon, icon)
    # header icon: a spinner while she works, a pulsing "?" while she waits, nothing on a result
    if state in ("planning", "running", "checking"):
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(col(acc, 0.25), max(1.0, icon * 0.16)))
        p.drawEllipse(icon_r)
        p.setPen(QPen(col(acc, 0.95), max(1.0, icon * 0.16), Qt.SolidLine, Qt.RoundCap))
        p.drawArc(icon_r, int(-t * 330 * 16) % (360 * 16), 100 * 16)
    elif state == "waiting":
        p.setPen(Qt.NoPen)
        p.setBrush(col(acc, 0.55 + 0.45 * pulse))
        p.drawEllipse(icon_r)

    text_px = ph * 0.125
    if show_text and text_px >= 6:
        f = QFont("Segoe UI")
        f.setPixelSize(max(6, int(round(text_px))))
        f.setBold(True)
        small = QFont("Segoe UI")
        small.setPixelSize(max(6, int(round(ph * 0.1))))
        if state == "waiting":
            p.setFont(f)
            p.setPen(QColor(20, 14, 4, 230))
            p.drawText(icon_r, Qt.AlignCenter, "?")
        step = pv.get("step") or ""
        step_w = QFontMetricsF(small).horizontalAdvance(step) + pad * 0.4 if step else 0.0
        tx = left + icon + pad * 0.6
        title = pv.get("title") or _PANEL_WORDS.get(state, "")
        p.setFont(f)
        p.setPen(QColor(236, 248, 248, 242))
        p.drawText(QRectF(tx, top, right - tx - step_w, icon), Qt.AlignVCenter | Qt.AlignLeft,
                   QFontMetricsF(f).elidedText(title, Qt.ElideRight, max(1.0, right - tx - step_w)))
        if step:
            p.setFont(small)
            p.setPen(col(acc, 0.95))
            p.drawText(QRectF(right - step_w, top, step_w, icon), Qt.AlignVCenter | Qt.AlignRight, step)
        detail = pv.get("detail") if state == "running" and pv.get("detail") else _PANEL_WORDS.get(state, "")
        if detail == title:
            detail = ""
        if detail:
            p.setFont(small)
            p.setPen(QColor(200, 226, 226, 190) if state != "waiting" else col(acc, 0.95))
            dy = top + icon + pad * 0.15
            p.drawText(QRectF(left, dy, right - left, ph * 0.13), Qt.AlignVCenter | Qt.AlignLeft,
                       QFontMetricsF(small).elidedText(detail, Qt.ElideRight, right - left))
        code_top = top + icon + ph * 0.13 + pad * 0.35
    else:
        code_top = top + icon + pad * 0.5

    bar_h = max(1.5, ph * 0.035)
    code_bottom = bottom - bar_h - pad * 0.5
    code_r = QRectF(left, code_top, right - left, max(0.0, code_bottom - code_top))
    if state in ("done", "failed"):
        # the result, big and simple
        s = min(code_r.height(), code_r.width()) * 0.9
        c = code_r.center()
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(col(acc, 0.95), max(1.5, s * 0.12), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if state == "done":
            p.drawPolyline(QPointF(c.x() - s * 0.32, c.y() + s * 0.02), QPointF(c.x() - s * 0.08, c.y() + s * 0.26),
                           QPointF(c.x() + s * 0.36, c.y() - s * 0.24))
        else:
            d = s * 0.26
            p.drawLine(QPointF(c.x() - d, c.y() - d), QPointF(c.x() + d, c.y() + d))
            p.drawLine(QPointF(c.x() + d, c.y() - d), QPointF(c.x() - d, c.y() + d))
    elif code_r.height() > 2:
        # lines of "work" typing out, scrolling up; dimmed while she waits on him
        lh = ph * 0.045
        gap = ph * 0.085
        rows = max(1, int(code_r.height() // gap))
        lines = pv.get("lines") or []
        dim = 0.35 if state == "waiting" else 1.0
        p.setPen(Qt.NoPen)
        for i, (indent, width, typed, alt) in enumerate(lines[-rows:]):
            y = code_r.top() + i * gap + (gap - lh) / 2
            x0 = code_r.left() + indent * code_r.width()
            lw = max(0.0, (code_r.right() - x0) * width * typed)
            p.setBrush(QColor(230, 240, 240, int(110 * dim)) if alt else col(acc, 0.6 * dim))
            p.drawRoundedRect(QRectF(x0, y, lw, lh), lh / 2, lh / 2)
            if typed < 1.0 and state != "waiting" and int(t * 3) % 2 == 0:
                p.setBrush(col(acc, 0.95))
                p.drawRect(QRectF(x0 + lw + lh * 0.4, y - lh * 0.3, lh * 0.9, lh * 1.6))
        if state == "checking":
            sy = code_r.top() + (t * 0.8 % 1.0) * code_r.height()
            p.setPen(QPen(col(acc, 0.7), max(1.0, ph * 0.012)))
            p.drawLine(QPointF(code_r.left(), sy), QPointF(code_r.right(), sy))

    # progress
    bar = QRectF(left, bottom - bar_h, right - left, bar_h)
    p.setPen(Qt.NoPen)
    p.setBrush(col(acc, 0.16))
    p.drawRoundedRect(bar, bar_h / 2, bar_h / 2)
    prog = max(0.0, min(1.0, pv.get("progress") or 0.0))
    if prog > 0:
        fill = QRectF(bar.left(), bar.top(), bar.width() * prog, bar_h)
        p.setBrush(col(acc, 0.92))
        p.drawRoundedRect(fill, bar_h / 2, bar_h / 2)
    elif state in ("planning", "running", "checking"):
        # no steps known yet: a sliding sheen says "busy"
        x = bar.left() + ((t * 0.6) % 1.0) * bar.width() * 1.3 - bar.width() * 0.3
        p.setBrush(col(acc, 0.6))
        p.drawRoundedRect(QRectF(max(bar.left(), x), bar.top(),
                                 max(0.0, min(bar.right(), x + bar.width() * 0.3) - max(bar.left(), x)), bar_h),
                          bar_h / 2, bar_h / 2)
    p.restore()


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
        return dispatch(self._win, msg)


def dispatch(window, msg):
    """One decoded IPC message -> reply dict, or None when nothing is sent
    back (hot-path commands, id-less requests, non-dict junk).

    Module-level so it runs against any object with the window's surface
    (envelope, expressions, cmd_q, snapshot(), model_name, instance,
    request_quit()) — no socket, no Qt.
    """
    if not isinstance(msg, dict):
        return None
    cmd = msg.get("cmd")
    msg_id = msg.get("id")

    # hot path: fire-and-forget, no reply
    if cmd == "mouth":
        window.envelope.feed(msg.get("chunks") or [])
        return None
    if cmd == "speech":
        window.envelope.speech(msg.get("state", ""))
        return None

    if cmd == "witness":  # fire-and-forget like the hot path; a reply only if asked for
        kind = msg.get("kind")
        ok = isinstance(kind, str) and kind in constellation.KINDS
        if ok:
            window.cmd_q.put(("witness", kind))
        return {"id": msg_id, "ok": ok} if msg_id is not None else None

    # what her body is doing (avatar_activity): fire-and-forget, a reply only if asked for.
    # `activity` carries the whole state — reading {title} | null, working {label, detail, step,
    # total, state} | null; `cue` is a one-shot beat (page | done | failed | nod).
    if cmd == "activity":
        state = {k: msg.get(k) for k in ("reading", "working") if k in msg}
        if state:
            window.cmd_q.put(("activity", state))
        return {"id": msg_id, "ok": bool(state)} if msg_id is not None else None
    if cmd == "cue":
        kind = msg.get("kind")
        ok = kind in ("page", "done", "failed", "nod")
        if ok:
            window.cmd_q.put(("cue", kind))
        return {"id": msg_id, "ok": ok} if msg_id is not None else None
    if cmd == "prop":  # protocol v1 (a sister from before `activity`): show = she's reading
        window.cmd_q.put(("activity", {"reading": {"title": ""} if msg.get("show") else None}))
        return {"id": msg_id, "ok": True, "prop": bool(msg.get("show"))} if msg_id is not None else None

    if msg_id is None:
        return None  # every remaining command requires an id

    if cmd == "hello":
        snap = window.snapshot()
        return {"id": msg_id, "ok": True, "version": avatar_ipc.PROTOCOL_VERSION,
                "instance": getattr(window, "instance", "aiva"),
                "model": window.model_name,
                "expressions": list(window.expressions.params),
                "witness": bool(getattr(window, "witness_enabled", False)),
                "activity": getattr(window, "activity", None) is not None,
                "ready": snap["ready"]}
    if cmd == "ping":
        return {"id": msg_id, "ok": True}
    if cmd == "get_position":
        snap = window.snapshot()
        return {"id": msg_id, "ok": True, "x": snap["x"], "y": snap["y"],
                "rotation": snap["rotation"], "size": snap["size"],
                "monitor": snap["monitor"], "layer": snap["layer"]}
    if cmd == "expression":
        matched = window.expressions.match(str(msg.get("name", "")))
        if matched is None:
            return {"id": msg_id, "ok": False,
                    "error": f"no expression matching {msg.get('name')!r}"}
        window.cmd_q.put(("expression", matched))
        return {"id": msg_id, "ok": True, "toggled": matched}
    if cmd == "move":
        window.cmd_q.put(("move", {k: msg.get(k) for k in
                                   ("x", "y", "rotation", "size")}))
        return {"id": msg_id, "ok": True}
    if cmd == "monitor":
        idx = msg.get("index")
        monitors = avatar_ipc.list_monitors()
        if not isinstance(idx, int) or not 0 <= idx < len(monitors):
            return {"id": msg_id, "ok": False,
                    "error": f"monitor {idx} doesn't exist (have {len(monitors)})"}
        window.cmd_q.put(("monitor", idx))
        return {"id": msg_id, "ok": True}
    if cmd == "layer":
        mode = msg.get("mode")
        if mode not in ("top", "desktop"):
            return {"id": msg_id, "ok": False, "error": "mode must be top|desktop"}
        window.cmd_q.put(("layer", mode))
        return {"id": msg_id, "ok": True}
    if cmd == "sleeping":
        window.cmd_q.put(("sleeping", bool(msg.get("value"))))
        return {"id": msg_id, "ok": True}
    if cmd == "quit":
        window.request_quit()
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


def layer_z(mode):
    """Her Win32 z-order slot: over every window ("top") or behind them all ("desktop"). The body
    and her sky both take it, so the stars never float over windows she is sitting behind."""
    return _HWND_TOPMOST if mode == "top" else _HWND_BOTTOM


def persisted_sky(sky, saved):
    """The state file's `constellation` entry: the live sky, or — while this body's witness is
    switched off (SAGA_CONSTELLATION=off) — the saved one carried through untouched, so turning
    it off never erases the moments she has lived."""
    if sky is not None:
        return sky.to_dict()
    return saved if isinstance(saved, dict) else None


def next_sky_frame(sky, frame_no, fps, now, sleeping=False, speaking=False):
    """_tick_sky's frame to paint, or None when none is due. The sky only breathes, so ~8 fps is
    plenty — but while her voice is hushing it or it is easing back, every frame repaints so the
    fade is smooth. `speaking` is her voice actually playing (MouthEnvelope.speaking)."""
    if frame_no % max(1, fps // 8) == 0 or not sky.hush.settled(speaking):
        return sky.frame(now, sleeping, speaking)
    sky.hush.step(now, speaking)   # settled: moves nothing, but keeps its clock current
    return None


def _alpha8(a):
    return max(0, min(255, int(round(255 * a))))


def paint_sky(p, frame, w, h, tint):
    """Her constellation (Constellation.frame) on a w x h sky. p: an active QPainter. It paints
    exactly what the frame says: a star is one disc `size_px` across with no glow, a line is a
    LINE_PX hairline whose two ends carry the frame's two end alphas."""
    from PyQt5.QtCore import QLineF, QPointF, Qt
    from PyQt5.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
    p.setRenderHint(QPainter.Antialiasing, True)
    r, g, b = (int(c * 255) for c in tint)
    for x1, y1, x2, y2, a1, a2 in frame["edges"]:
        p1, p2 = QPointF(x1 * w, y1 * h), QPointF(x2 * w, y2 * h)
        grad = QLinearGradient(p1, p2)
        grad.setColorAt(0.0, QColor(r, g, b, _alpha8(a1)))
        grad.setColorAt(1.0, QColor(r, g, b, _alpha8(a2)))
        p.setPen(QPen(QBrush(grad), constellation.LINE_PX))
        # sub-pixel, like the stars: a drifting line glides with them, never steps
        p.drawLine(QLineF(p1, p2))
    p.setPen(Qt.NoPen)
    for x, y, a, size in frame["stars"]:
        p.setBrush(QColor(r, g, b, _alpha8(a)))
        p.drawEllipse(QPointF(x * w, y * h), size / 2.0, size / 2.0)


def selftest_verdict(ready, expect_sky=False, sky_paints=0, sky_failed=None):
    """--selftest's answer as (ok, why). A body with a witness passes only if its sky painted."""
    if not ready:
        return False, "the model never became ready"
    if expect_sky and sky_failed:
        return False, f"the constellation stopped drawing ({sky_failed})"
    if expect_sky and not sky_paints:
        return False, "the constellation never painted"
    return True, "ok"


def build_window_class():
    """AvatarWindow must be defined after QApplication's deps import cleanly;
    building it in a factory keeps module import light for unit tests."""
    Qt, QTimer, QMetaObject, _fmt, QApplication, QOpenGLWidget = _qt()
    from PyQt5.QtGui import QPainter
    from PyQt5.QtWidgets import QWidget
    import live2d.v3 as live2d

    class SkyWindow(QWidget):
        """Her constellation witness: a frameless, translucent, click-through
        window over the monitor, to her upper left. It lives OUTSIDE the GL
        widget so it can never disturb the model's transparent render. It only paints what Constellation.frame() hands it."""

        def __init__(self, tint, layer="top"):
            super().__init__()
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setAttribute(Qt.WA_ShowWithoutActivating)
            self.setAttribute(Qt.WA_TransparentForMouseEvents)
            self._tint = tint
            self._layer = layer        # follows her body's top|desktop layer
            self._frame = {"stars": [], "edges": []}
            self._rect = None
            self.paints = 0            # finished paints (the selftest asks)
            self.broken = None         # the paint error that switched drawing off

        def set_frame(self, frame):
            self._frame = frame
            self.update()

        def place(self, rect):
            if rect != self._rect:
                self._rect = rect
                self.setGeometry(*rect)

        def set_layer(self, mode):
            self._layer = mode
            if self.isVisible():
                ctypes.windll.user32.SetWindowPos(int(self.winId()), layer_z(mode), 0, 0, 0, 0, _SWP_ZONLY)

        def paintEvent(self, event):
            if self.broken:
                return
            p = QPainter(self)
            try:
                paint_sky(p, self._frame, self.width(), self.height(), self._tint)
                self.paints += 1
            except Exception as e:  # escaping a Qt virtual, PyQt5 aborts her whole body
                self.broken = repr(e)
            finally:
                p.end()

        def apply_click_through(self):
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            style = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            style |= (_WS_EX_LAYERED | _WS_EX_TRANSPARENT | _WS_EX_NOACTIVATE | _WS_EX_TOOLWINDOW)
            user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, style)
            user32.SetWindowPos(hwnd, layer_z(self._layer), 0, 0, 0, 0, _SWP_ZONLY)

        def showEvent(self, event):
            super().showEvent(event)
            self.apply_click_through()

    class AvatarWindow(QOpenGLWidget):
        def __init__(self, model_path, fps, sway, windowed, state,
                     instance="aiva", state_file=None):
            super().__init__()
            self.instance = instance or "aiva"
            self.state_file = state_file or avatar_ipc.state_path(self.instance)
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
            # what her body is doing: the book in her hands while she reads, the work panel
            # beside her while she's on a task (modules/avatar_activity.py). A mesh without a
            # rig entry simply doesn't do it.
            self.activity = avatar_activity.ActivityDirector(avatar_activity.RIGS.get(self.model_name))
            self._look = avatar_activity.LOOKS.get(self.model_name, avatar_activity.LOOKS["ariu"])
            self._task_text = os.getenv("AIVA_AVATAR_TASK_LABEL", "1") == "1"
            self._book_art = None      # PropArt, loaded the first time she reads
            self._held_part = None     # index of the rig's held-object part (resolved at GL init)
            self._held_hidden = False
            self._activity_errors = 0
            # her quiet constellation witness — only for an instance that has one (Saga)
            sky_cfg = constellation.WitnessConfig.from_env()
            self.witness_enabled = constellation.enabled_for(self.instance, sky_cfg)
            saved_sky = (state or {}).get("constellation")
            self.sky = constellation.Constellation(sky_cfg, saved_sky) if self.witness_enabled else None
            self._sky_saved = None if self.witness_enabled else saved_sky  # switched off: kept as-is
            self._sky_win = None       # SkyWindow, created on first tick (never in --windowed)
            self._sky_failed = None    # why drawing stopped; the sky itself keeps witnessing
            self._sky_frames = 0
            self._sky_mon = None       # (monitor index, its rect): looked up when she changes screens
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

            # "Aiva Avatar" for aiva (unchanged), "Saga Avatar" for saga
            self.setWindowTitle(f"{self.instance.capitalize()} Avatar")
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
            rig = self.activity.rig
            if rig is not None:
                part_ids = self.model.GetPartIds()
                self._held_part = part_ids.index(rig.held_part) if rig.held_part in part_ids else None
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
                cdi_path = cdi3_path_for(self.model_path)
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
                    self._camo_skin = tuple(fixups.get("camo_skin", self._camo_skin))
                    override = parse_rgb(os.getenv("AIVA_CAMO_SKIN"))
                    if override:
                        self._camo_skin = override
                        _log(f"camo skin overridden by AIVA_CAMO_SKIN -> {override}")
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
                         f"leaky mouth mesh(es) with skin {self._camo_skin}, "
                         f"mouth-part fade "
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
            ctypes.windll.user32.SetWindowPos(hwnd, layer_z(self.layer), 0, 0, 0, 0, _SWP_ZONLY)

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
            sway_x = None
            if self._sway and not self._sleeping:
                t = now - self._t0
                sway_x = 3.0 * math.sin(2 * math.pi * t / 9.3)
                self.model.SetParameterValue("ParamAngleX", sway_x)
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
            pose = self.activity.step(
                dt, now, talking=self.envelope.speaking, sleeping=self._sleeping,
                side_hint=avatar_activity.panel_side(self.placement.cur["x"]))
            self._apply_pose(pose, sway_x)
            self.placement.apply(self.model, self.width() / max(1, self.height()))
            self.model.Draw()
            if pose.book or pose.panel:
                self._paint_activity(pose)
            self._tick_sky(now)

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
                        if self._sky_win is not None:
                            self._sky_win.set_layer(arg)   # her stars sink behind windows with her
                    self._mark_dirty()
                elif kind == "sleeping":
                    self._sleeping = arg
                    if self.model:
                        self.model.SetAutoBlinkEnable(not arg)
                elif kind == "activity":
                    was = (bool(self.activity.reading), bool(self.activity.working))
                    self.activity.set_state(**arg)
                    now_ = (bool(self.activity.reading), bool(self.activity.working))
                    if now_ != was:
                        _log(f"activity: reading={'on' if now_[0] else 'off'} "
                             f"working={'on' if now_[1] else 'off'}")
                elif kind == "cue":
                    self.activity.cue(arg, time.monotonic())
                elif kind == "witness" and self.sky is not None:
                    if self.sky.witness(arg, time.monotonic()):   # a new star, or a burst brightened one
                        self._mark_dirty()

        def _tick_sky(self, now):
            """Repaint the witness at ~8 fps (it only breathes; every frame while her voice eases
            it) and keep it to her upper left."""
            if self.sky is None or self._windowed or self._sky_failed:
                return
            self._sky_frames += 1
            try:
                frame = next_sky_frame(self.sky, self._sky_frames, self._fps, now,
                                       self._sleeping, self.envelope.speaking)
                if frame is None:
                    return
                if self._sky_win is not None and self._sky_win.broken:
                    raise RuntimeError(f"paint failed: {self._sky_win.broken}")
                if self._sky_win is None:
                    tint = parse_rgb(os.getenv("SAGA_CONSTELLATION_TINT")) or constellation.TINT
                    self._sky_win = SkyWindow(tint, self.layer)
                if self._sky_mon is None or self._sky_mon[0] != self.monitor:
                    monitors = avatar_ipc.list_monitors()   # once per screen, not ~8x a second
                    self._sky_mon = (self.monitor, monitors[min(self.monitor, len(monitors) - 1)])
                self._sky_win.place(constellation.sky_rect(self._sky_mon[1], self.placement.cur,
                                                           self.sky.cfg.scale))
                self._sky_win.set_frame(frame)
                if not self._sky_win.isVisible():
                    self._sky_win.show()
            except Exception as e:  # a witness must never take her body down
                # stop DRAWING only: the sky keeps witnessing and stays in her state file, so a
                # paint failure can't erase it (it shows again when her body restarts)
                self._sky_failed = str(e) or repr(e)
                _log(f"constellation hidden: {self._sky_failed}")
                if self._sky_win is not None:
                    self._sky_win.hide()

        def _apply_pose(self, pose, sway_x):
            """Her activity on top of idle life: the hold (max'ed with a 手柄 toggle so the
            user's controller pose still wins), the held part hidden while a book is out, head
            and eyes. Runs after Update() and before Draw() like every other param here."""
            rig = self.activity.rig
            if rig is None:
                for pid, value in pose.params.items():     # a nod needs no rig
                    self.model.SetParameterValue(pid, value)
                return
            if pose.hold > 0:
                self.model.SetParameterValue(
                    rig.hold_param, max(pose.hold, 1.0 if self.expressions.drives(rig.hold_param) else 0.0))
            if self._held_part is not None:
                if pose.hide_held:
                    self.model.SetPartOpacity(self._held_part, 0.0)
                    self._held_hidden = True
                elif self._held_hidden:
                    self.model.SetPartOpacity(self._held_part, 1.0)   # the controller comes back
                    self._held_hidden = False
            for pid, value in pose.params.items():
                self.model.SetParameterValue(pid, value)
            if pose.head_turn:
                # absolute, never additive: a turn on top of the sway (or on its own)
                self.model.SetParameterValue("ParamAngleX", (sway_x or 0.0) + pose.head_turn)

        def _paint_activity(self, pose):
            """Her book and/or work panel, painted into this GL frame after the model."""
            p = QPainter(self)
            try:
                p.setRenderHint(QPainter.Antialiasing, True)
                p.setRenderHint(QPainter.SmoothPixmapTransform, True)
                p.setRenderHint(QPainter.TextAntialiasing, True)
                cur = self.placement.cur
                w, h = self.width(), self.height()
                if pose.book:
                    if self._book_art is None:
                        self._book_art = PropArt(self._prop_path, self._look.get("book_hue", 0))
                        if self._book_art.image is None:
                            _log(f"no book art at {self._prop_path} — reading without a book")
                    if self._book_art.image is not None:
                        paint_book(p, self._book_art, pose.book, cur, w, h, BASE_SCALE)
                if pose.panel:
                    paint_panel(p, pose.panel, cur, w, h, BASE_SCALE, self._look["accent"],
                                show_text=self._task_text)
            except Exception as e:  # a prop must never take her body down
                self._activity_errors += 1
                if self._activity_errors <= 3:
                    _log(f"activity paint failed: {e!r}")
                if self._activity_errors >= 30:
                    _log("activity props switched off after repeated failures")
                    self.activity = avatar_activity.ActivityDirector(None)
            finally:
                p.end()

        def _mark_dirty(self):
            self._dirty_at = time.monotonic() + 2.0  # debounce disk writes

        def _persist(self):
            self._dirty_at = None
            if not self._server:
                return
            sky = persisted_sky(self.sky, self._sky_saved)
            try:
                avatar_ipc.write_state_atomic({
                    "port": self._server.port, "pid": os.getpid(),
                    "boot_id": self._server.boot_id,
                    "instance": self.instance, "model": self.model_name,
                    "placement": dict(self.placement.target),
                    "monitor": self.monitor, "layer": self.layer,
                    **({"constellation": sky} if sky is not None else {}),
                }, self.state_file)
            except OSError as e:  # e.g. a reader holding the file; this runs inside paintGL,
                _log(f"state not saved ({e}) — retrying")  # where an escaped error aborts her body
                self._mark_dirty()

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


def _run_activity_demo(window):
    """--demo-activity: what her body does while she reads and works, through the same
    queue the IPC uses. Loops every ~40 s so it can be watched from any angle."""
    from PyQt5.QtCore import QTimer

    def act(**state):
        return lambda: window.cmd_q.put(("activity", state))

    def cue(kind):
        return lambda: window.cmd_q.put(("cue", kind))

    def talk(secs):
        def go():
            chunks = [[0.075 + 0.075 * math.sin(i / 3.0), 0.05, 0.5 + 0.5 * math.sin(i / 7.0)]
                      for i in range(int(secs / 0.05))]
            window.envelope.feed(chunks)
            window.envelope.speech("start")
            QTimer.singleShot(int(secs * 1000), lambda: window.envelope.speech("stop"))
        return go

    task = {"label": "Fix the dashboard chat box", "detail": "read dashboard.py", "step": 0,
            "total": 4, "state": "running"}
    script = [
        (500, "she pulls up a book", act(reading={"title": "Through the Looking-Glass"})),
        (6000, "a section ends: page turn", cue("page")),
        (9000, "she talks: looks up from the page", talk(2.5)),
        (14000, "she puts the book away", act(reading=None)),
        (16000, "a task starts: the work panel opens", act(working=dict(task, state="planning", total=0))),
        (18500, "step 1 of 4", act(working=task)),
        (22000, "step 2 of 4", act(working=dict(task, step=1, detail="edit the chat box"))),
        (25000, "it waits on you", act(working=dict(task, step=2, state="waiting", detail=""))),
        (29000, "checking", act(working=dict(task, step=3, state="checking", detail=""))),
        (32000, "done", lambda: (window.cmd_q.put(("cue", "done")),
                                 window.cmd_q.put(("activity", {"working": None})))),
        (35500, "reading and working at once", act(reading={"title": "x"}, working=dict(task, step=1))),
        (41000, "both away", act(reading=None, working=None)),
    ]
    for ms, label, fn in script:
        QTimer.singleShot(ms, (lambda l=label, f=fn: (_log(f"activity demo: {l}"), f())))
    QTimer.singleShot(44000, lambda: _run_activity_demo(window))


def witness_demo_script(gap):
    """--demo-witness as [(seconds, label, queued command)]: one moment of each kind, spaced past
    her burst gap so each lights a star (the first followed by one that only brightens it), then
    she speaks (the sky hushes) and stops (it eases back), the house sleeps and wakes, she tilts
    and moves, and her layer goes to the desktop and back.
    Pure, so a test can check it only queues what the IPC would."""
    step = gap + 1.0
    out, t = [], 1.0
    for i, kind in enumerate(constellation.KINDS):
        out.append((t, f"a moment ({kind}): a star lights", ("witness", kind)))
        if i == 0:
            out.append((t + 1.0, "another straight after: inside the gap it only brightens",
                        ("witness", "said")))
        t += step
    return out + [
        (t, "she speaks: stars and lines hush to about 30%", ("speech", "start")),
        (t + 4, "she stops: the sky eases back over a second", ("speech", "stop")),
        (t + 7, "the house sleeps: brighter, slower", ("sleeping", True)),
        (t + 13, "she wakes", ("sleeping", False)),
        (t + 15, "she tilts and moves: the sky follows",
         ("move", {"x": -0.4, "y": -0.3, "rotation": 25.0, "size": None})),
        (t + 19, "back", ("move", {"x": 0.0, "y": 0.0, "rotation": 0.0, "size": None})),
        (t + 21, "layer -> desktop: the stars sink behind windows with her", ("layer", "desktop")),
        (t + 25, "layer -> top", ("layer", "top")),
    ]


def _run_witness_demo(window):
    """--demo-witness: her constellation without a sister, through the same queue the IPC uses.
    Loops, so the sky fills to its cap and the oldest stars start dimming out."""
    from PyQt5.QtCore import QTimer

    if window.sky is None or window._windowed:
        _log("witness demo: this body draws no sky — only --instance saga has one "
             "(SAGA_CONSTELLATION on, not --windowed)")
        return
    def play(label, cmd):
        _log(f"witness demo: {label}")
        if cmd[0] == "speech":
            window.envelope.speech(cmd[1])   # where the IPC's `speech` lands, not the queue
        else:
            window.cmd_q.put(cmd)

    script = witness_demo_script(window.sky.cfg.min_gap)
    for secs, label, cmd in script:
        QTimer.singleShot(int(secs * 1000), (lambda l=label, c=cmd: play(l, c)))
    QTimer.singleShot(int((script[-1][0] + 2) * 1000), lambda: _run_witness_demo(window))


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


def write_model(src_model_path, name):
    """`--write-model NAME`: NAME.model3.json beside `src_model_path`, same
    mesh, textures from NAME.2048/. Prints the written path; 0 on success."""
    name = (name or "").strip()
    if not name or any(ch in name for ch in "/\\ .:"):
        _log(f"bad model name {name!r} (one word, no dots/slashes)")
        return 2
    if not os.path.isfile(src_model_path):
        _log(f"source model not found: {src_model_path}")
        return 3
    folder = os.path.dirname(os.path.abspath(src_model_path))
    textures_dir = f"{name}.2048"
    if not os.path.isdir(os.path.join(folder, textures_dir)):
        _log(f"textures folder missing: {os.path.join(folder, textures_dir)}")
        return 3
    try:
        with open(src_model_path, encoding="utf-8") as f:
            src = json.load(f)
    except (OSError, ValueError) as exc:
        _log(f"can't read {src_model_path}: {exc}")
        return 3
    out = make_retexture_model3(src, textures_dir)
    textures = out["FileReferences"]["Textures"]
    missing = [t for t in textures if not os.path.isfile(os.path.join(folder, t))]
    if missing:
        _log(f"{len(missing)} texture(s) missing under {textures_dir}: {missing[:3]}")
        return 3
    dest = os.path.join(folder, f"{name}.model3.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent="\t", ensure_ascii=False)
        f.write("\n")
    _log(f"wrote model '{name}' ({len(textures)} textures from {textures_dir}/, "
         f"mesh {out['FileReferences'].get('Moc')})")
    print(dest, flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Live2D avatar renderer (one per sister)")
    parser.add_argument("--model", default=os.getenv("AIVA_AVATAR_MODEL", DEFAULT_MODEL))
    parser.add_argument("--fps", type=int,
                        default=int(os.getenv("AIVA_AVATAR_FPS", "32")))
    parser.add_argument("--monitor", type=int, default=None)
    parser.add_argument("--instance", default=os.getenv("AIVA_AVATAR_INSTANCE", "aiva"),
                        help="which sister this body belongs to (state file, title, log prefix)")
    parser.add_argument("--state", default=None,
                        help="state file path (default: avatar_ipc.state_path(instance))")
    parser.add_argument("--write-model", metavar="NAME", default=None,
                        help="write NAME.model3.json beside --model using NAME.2048/ textures, then exit")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-activity", action="store_true",
                        help="loop the reading/working choreography (book, work panel)")
    parser.add_argument("--demo-witness", action="store_true",
                        help="loop moments into the constellation (needs --instance saga)")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    _set_instance(args.instance)
    if args.demo_witness and not args.state:
        # the demo's moments are made up: they must never land in her real sky
        args.state = os.path.join(tempfile.gettempdir(), f"{args.instance}_witness_demo_state.json")
    state_file = args.state or avatar_ipc.state_path(args.instance)

    if args.write_model:
        return write_model(args.model, args.write_model)

    # single-instance guard: a live renderer already owns THIS instance's body
    # (each sister has her own state file, so two sisters never collide here)
    state = avatar_ipc.read_state(state_file)
    if state and state.get("pid") and avatar_ipc.pid_alive(state["pid"]) \
            and _hello_probe(state):
        if args.selftest:  # nothing was rendered, so nothing may pass
            _log(f"selftest NOT RUN: a live renderer owns {state_file} (pid {state['pid']}) "
                 f"— pass --state <a temp file> to test beside it")
            return 4
        _log(f"renderer already running for instance {args.instance!r} "
             f"(pid {state['pid']}) — exiting")
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
    window = AvatarWindow(args.model, args.fps, sway, args.windowed, state,
                          instance=args.instance, state_file=state_file)
    if args.monitor is not None:
        window.monitor = args.monitor
        if not args.windowed:
            monitors = avatar_ipc.list_monitors()
            window.setGeometry(*monitors[min(args.monitor, len(monitors) - 1)])

    server = IpcServer(window)
    window._server = server
    server.start()
    window._persist()  # publish port/pid/boot_id before anyone can look
    _log(f"IPC listening on 127.0.0.1:{server.port} (pid {os.getpid()}, "
         f"model '{window.model_name}', state {state_file})")

    window.show()
    if args.demo:
        _run_demo(window)
    if args.demo_activity:
        _run_activity_demo(window)
    if args.demo_witness:
        _run_witness_demo(window)
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
        sky_win = window._sky_win
        expect_sky = window.sky is not None and not args.windowed
        ok, why = selftest_verdict(window.snapshot().get("ready"), expect_sky,
                                   sky_win.paints if sky_win else 0,
                                   window._sky_failed or (sky_win.broken if sky_win else None))
        sky_note = (f" — sky painted {sky_win.paints}x, {len(window.sky.stars)} star(s)"
                    if expect_sky and sky_win else "")
        _log(f"selftest {'OK' if ok else 'FAILED: ' + why}{sky_note}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
