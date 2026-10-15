"""Her body doing things — the choreography behind "she pulls up a book while she reads" and "a
work panel floats up beside her while she's on a task".

Pure (no Qt, no GL, clock injected), so every beat is unit-testable. The renderer feeds the
sister's state in (`set_state`, `cue`) and calls `step(dt, now, ...)` once per frame; the Pose it
gets back says which params to set, whether to hide the held-object part, and where the held book
and the work panel go — in MODEL units (the view units of an unscaled, unmoved model), which
`model_to_px` maps through her live placement. So the props follow her when she's moved, resized,
rotated or put on another monitor, and each sister's panel opens on her own side.

The hold is the model's own 手柄 (game-controller) pose: `shoubing`=1 raises both hands to her
chest. With the controller part hidden the hands are empty and the book is drawn over them.
`shoubing` CROSSFADES the two arm sets (it doesn't swing an arm), so the pull-up is staged: the
arms crossfade fast, the book rises into them from below and fades in, then her head tips down
and her eyes start running along lines. Putting it away runs the other way, hands last. While she
talks she looks up from the book; a finished section turns the page (`cue("page")`).

Offsets were measured on offscreen renders of ariu.moc3 (2026-10-13): the controller sits at
(-0.007, 0.463), 0.225 wide, just under her chin, so the book is centred a little lower to keep
her mouth (lip-sync) in view.
"""

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rig:
    """Where one mesh's hands, head and work panel are, in model units."""
    hold_param: str = "shoubing"   # the 手柄 pose: hands up at her chest
    held_part: str = "Part47"      # the controller in those hands (hidden while she holds a book)
    book_x: float = 0.0
    book_y: float = 0.42
    book_w: float = 0.30
    panel_dx: float = 0.76         # panel centre, beside her head, towards the screen's middle
    panel_y: float = 0.64
    panel_w: float = 0.74
    panel_h: float = 0.42


# keyed by model name like avatar_renderer.MODEL_FIXUPS; Saga is a retexture of the same mesh
RIGS = {"ariu": Rig()}
RIGS["saga"] = RIGS["ariu"]

# panel/book colours per body (0..1 rgb); book_hue rotates the stock teal book cover (degrees)
LOOKS = {
    "ariu": {"accent": (0.40, 0.88, 0.82), "book_hue": 0},
    "saga": {"accent": (0.98, 0.56, 0.58), "book_hue": 170},
}
WAITING_ACCENT = (1.0, 0.76, 0.32)
DONE_ACCENT = (0.46, 0.92, 0.62)
FAILED_ACCENT = (1.0, 0.47, 0.47)

READ_IN_SECS = 0.9      # hands up -> book risen -> head down
READ_OUT_SECS = 0.6
PANEL_IN_SECS = 0.45
PANEL_OUT_SECS = 0.35
RESULT_SECS = 2.4       # a finished task's ✓ / ✗ stays up this long
PAGE_SECS = 0.5
LINE_SECS = 2.2         # one line of reading: a slow sweep, then a quick return
LINES_PER_PAGE = 6
CODE_LINE_SECS = 0.5    # the panel types one line this often

WORK_STATES = ("planning", "running", "checking", "waiting")


# --- small pure helpers ------------------------------------------------------------------------

def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def smoothstep(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out_cubic(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_out_back(x, s=1.4):
    """Overshoots a little past 1 before settling — the panel 'pops' open."""
    x = clamp01(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def approach(cur, target, dt, tau):
    """Exponential smoothing toward target with time constant tau (seconds)."""
    if dt <= 0 or tau <= 0:
        return target if tau <= 0 else cur
    return cur + (target - cur) * (1 - math.exp(-dt / tau))


def _hash01(k, salt=0.0):
    """Deterministic 0..1 noise for integer k (the panel's typed-line widths)."""
    v = math.sin(k * 12.9898 + salt * 78.233) * 43758.5453
    return v - math.floor(v)


def model_to_px(pt, placement, w, h, base_scale):
    """A model-units point -> widget pixels, through her placement exactly as the renderer's
    Placement.apply sets it (offset x*aspect/y, scale base*10^(size/100), CCW rotation in degrees).
    Verified against offscreen renders: view = offset + s * R(rot) * pt."""
    aspect = w / max(1, h)
    s = base_scale * 10 ** (float(placement.get("size", 0.0)) / 100.0)
    a = math.radians(float(placement.get("rotation", 0.0)))
    x, y = pt
    vx = float(placement.get("x", 0.0)) * aspect + s * (x * math.cos(a) - y * math.sin(a))
    vy = float(placement.get("y", 0.0)) + s * (x * math.sin(a) + y * math.cos(a))
    return (vx / aspect + 1) / 2 * w, (1 - vy) / 2 * h


def model_len_px(length, placement, h, base_scale):
    """A model-units length -> pixels at her current size."""
    s = base_scale * 10 ** (float(placement.get("size", 0.0)) / 100.0)
    return length * s * h / 2


def panel_side(placement_x):
    """-1 = the panel opens on her left (screen left), +1 = her right: always towards the middle
    of the screen, so it never runs off an edge. Aiva stands right (x>0) -> -1; Saga left -> +1."""
    return -1 if float(placement_x) > 0.05 else 1


def _clean_text(v, limit=120):
    if not isinstance(v, str):
        return ""
    v = " ".join(v.split())
    return v[:limit]


def _clean_int(v):
    try:
        return max(0, min(999, int(v)))
    except (TypeError, ValueError):
        return 0


def clean_reading(v):
    """IPC payload -> {'title': str} | None (anything that isn't a dict means 'not reading')."""
    if not isinstance(v, dict):
        return None
    return {"title": _clean_text(v.get("title"))}


def clean_working(v):
    """IPC payload -> {'label','detail','step','total','state'} | None."""
    if not isinstance(v, dict):
        return None
    state = v.get("state") if v.get("state") in WORK_STATES else "running"
    return {"label": _clean_text(v.get("label")), "detail": _clean_text(v.get("detail")),
            "step": _clean_int(v.get("step")), "total": _clean_int(v.get("total")), "state": state}


# --- the director ------------------------------------------------------------------------------

@dataclass
class Pose:
    """One frame of what her body is doing on top of idle life (blink, breath, sway, lip-sync)."""
    params: dict = field(default_factory=dict)   # absolute values to set after expressions/mouth
    head_turn: float = 0.0     # degrees added to ParamAngleX (on top of the idle sway)
    hold: float = 0.0          # 0..1 for the rig's hold param (the renderer max()es it with a toggle)
    hide_held: bool = False    # hide the held-object part (the controller) this frame
    book: dict = None          # {"x","y","w","alpha","rot"} in model units / degrees, or None
    panel: dict = None         # what the work panel shows (see ActivityDirector._panel_view), or None

    @property
    def active(self):
        return bool(self.hold or self.book or self.panel or self.params or self.head_turn)


class ActivityDirector:
    def __init__(self, rig=None):
        self.rig = rig
        self.reading = None          # {"title"} while she reads
        self.working = None          # {"label","detail","step","total","state"} while she works
        self._last_working = None    # what the panel keeps showing while it closes / shows a result
        self._r = 0.0                # reading timeline 0..1 (hands, book, head)
        self._p = 0.0                # panel timeline 0..1
        self._u = 0.0                # looking up from the book because she's talking
        self._talk_until = -1e9
        self._read_t0 = None         # when the current page of eye-scanning started
        self._page_t0 = -1e9         # last page turn (book bounce)
        self._result = None          # ("done"|"failed", until)
        self._side = None            # latched while the panel is up
        self._work_clock = 0.0       # advances only while the panel types (frozen while waiting)
        self._work_t0 = None
        self._progress = 0.0
        self._gaze = 0.0             # smoothed eyes-to-panel, -1..1 (sign = side)
        self._turn = 0.0             # smoothed head turn, degrees

    # -- inputs (renderer's GL thread, from the IPC queue) --
    def set_state(self, **kw):
        """Partial update: only the keys given change (`reading=None` stops reading)."""
        if "reading" in kw:
            r = clean_reading(kw["reading"])
            if r and not self.reading:
                self._read_t0 = None
            self.reading = r
        if "working" in kw:
            self.working = clean_working(kw["working"])
            if self.working:
                self._last_working = self.working

    def cue(self, kind, now):
        """One-shot beats: 'page' (a section finished) or 'done'/'failed' (a task ended)."""
        if kind == "page":
            if self.reading:
                self._page_t0 = now
                self._read_t0 = now       # eyes back to the top of the new page
            return True
        if kind in ("done", "failed"):
            self._result = (kind, now + RESULT_SECS)
            self._progress = 1.0 if kind == "done" else self._progress
            return True
        return False

    # -- per frame --
    def step(self, dt, now, *, talking=False, sleeping=False, side_hint=-1):
        rig = self.rig
        pose = Pose()
        dt = max(0.0, min(0.25, dt))

        # timelines: reading in/out at different speeds; the panel likewise
        want_read = bool(self.reading) and rig is not None
        self._r = clamp01(self._r + (dt / READ_IN_SECS if want_read else -dt / READ_OUT_SECS))
        if self._result and now >= self._result[1]:
            self._result = None
        want_panel = (self.working is not None or self._result is not None) and rig is not None
        if want_panel and self._p == 0.0:
            self._side = side_hint                 # latch: the panel never hops sides while up
            self._work_t0 = now
        self._p = clamp01(self._p + (dt / PANEL_IN_SECS if want_panel else -dt / PANEL_OUT_SECS))
        if self._p == 0.0 and not want_panel:
            self._side = None
            self._progress = 0.0
            self._work_clock = 0.0

        if talking:
            self._talk_until = now + 0.8          # keep looking up a beat after she stops
        self._u = clamp01(self._u + (dt / 0.25 if now < self._talk_until else -dt / 0.6))
        u = smoothstep(self._u)

        if rig is None:
            return pose

        gaze_x = gaze_y = 0.0
        head_down = 0.0
        # --- the book ---------------------------------------------------------------------
        if self._r > 0.0:
            r = self._r
            pose.hold = smoothstep(r / 0.28)              # hands first in, last out (a quick crossfade)
            pose.hide_held = True
            b = clamp01((r - 0.15) / 0.85)                # the book rides on the hands' heels
            face = smoothstep((r - 0.25) / 0.75)
            if self._read_t0 is None:
                self._read_t0 = now
            if b > 0:
                rise = 0.10 * (1 - ease_out_cubic(b))
                bob = 0.004 * math.sin(2 * math.pi * now / 3.4)
                pt = (now - self._page_t0) / PAGE_SECS
                flip_y = 0.016 * math.sin(math.pi * pt) if 0 <= pt < 1 else 0.0
                flip_r = 3.0 * math.sin(2 * math.pi * pt) * (1 - pt) if 0 <= pt < 1 else 0.0
                pose.book = {"x": rig.book_x, "y": rig.book_y - rise - 0.035 * u + bob + flip_y,
                             "w": rig.book_w, "alpha": smoothstep(b * 2.6), "rot": flip_r}
            down = face * (1 - 0.85 * u)                  # talking: she looks up from the page
            head_down = down
            if not sleeping:
                sx, sy = self._scan(now)
                gaze_x += sx * down
                gaze_y += sy * down

        # --- the work panel -------------------------------------------------------------------
        to_panel = 0.0
        if self._p > 0.0:
            state = self._panel_state()
            if state not in ("waiting", "done", "failed"):
                self._work_clock += dt
            if self.working:
                tot = self.working["total"]
                target = (self.working["step"] / tot) if tot else 0.0
                self._progress = approach(self._progress, max(self._progress, target), dt, 0.4)
            pose.panel = self._panel_view(now, state)
            # glances: a long look at the panel, then back to the room. While she reads, only a
            # quick peek now and then; while it waits on him, she looks at him, not at it.
            if state == "waiting":
                look = 0.0
            elif self._r > 0.5:
                look = 1.0 if ((now - (self._work_t0 or now)) % 14.0) > 12.6 else 0.0
            else:
                look = 1.0 if ((now - (self._work_t0 or now)) % 9.0) < 5.5 else 0.0
            to_panel = look * smoothstep(self._p) * (1 - u)
        side = self._side if self._side is not None else side_hint
        self._gaze = approach(self._gaze, to_panel * side, dt, 0.14)
        self._turn = approach(self._turn, to_panel * side * 9.0, dt, 0.4)

        if self._r > 0.0 or self._p > 0.0 or abs(self._gaze) > 1e-3 or abs(self._turn) > 1e-2:
            p = pose.params
            p["ParamEyeBallX"] = max(-1.0, min(1.0, gaze_x * (1 - abs(self._gaze)) + 0.8 * self._gaze))
            p["ParamEyeBallY"] = max(-1.0, min(1.0, gaze_y * (1 - abs(self._gaze))))
            if head_down > 0:
                p["ParamAngleY"] = -12.0 * head_down - (6.0 if sleeping else 0.0) * head_down
            pose.head_turn = self._turn
        return pose

    def _scan(self, now):
        """Eyes running along lines of text: a slow left-to-right sweep, a quick return, one line
        lower each time; back to the top after a page's worth of lines."""
        t = max(0.0, now - (self._read_t0 or now))
        line = int(t // LINE_SECS) % LINES_PER_PAGE
        f = (t % LINE_SECS) / LINE_SECS
        x = -0.32 + 0.64 * (f / 0.86) if f < 0.86 else 0.32 - 0.64 * ((f - 0.86) / 0.14)
        y = -0.62 - 0.05 * line
        return x, y

    def _panel_state(self):
        if self._result:
            return self._result[0]
        w = self.working or self._last_working
        return w["state"] if w else "running"

    def _panel_view(self, now, state):
        rig = self.rig
        side = self._side if self._side is not None else -1
        w = self.working or self._last_working or {}
        k_now = int(self._work_clock // CODE_LINE_SECS)
        typed = (self._work_clock % CODE_LINE_SECS) / CODE_LINE_SECS
        lines = []
        for k in range(k_now - 3, k_now + 1):
            if k < 0:
                continue
            indent = (int(_hash01(k, 1.0) * 3)) * 0.08
            width = 0.30 + 0.58 * _hash01(k)
            lines.append((indent, width, typed if k == k_now else 1.0, _hash01(k, 2.0) < 0.3))
        tot = w.get("total") or 0
        return {
            "cx": side * rig.panel_dx, "cy": rig.panel_y, "w": rig.panel_w, "h": rig.panel_h,
            "side": side, "open": ease_out_back(self._p), "alpha": smoothstep(self._p * 1.5),
            "state": state, "title": w.get("label", ""), "detail": w.get("detail", ""),
            "step": f"{min(w.get('step', 0) + 1, tot)}/{tot}" if tot else "",
            "progress": 1.0 if state == "done" else self._progress,
            "lines": lines, "t": now,
        }
