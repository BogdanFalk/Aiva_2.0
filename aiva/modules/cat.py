"""Aiva's desktop cat — a tiny pixel pet that lives on the Windows desktop.

Architecture (all in this file):
  DesktopCat  — facade constructed in main(); the only object other code touches.
                Thread-safe: tool handlers put commands on a queue and read a
                lock-guarded snapshot; cat events reach Aiva's announce() via
                asyncio.run_coroutine_threadsafe with rate limiting enforced here.
  CatState    — persisted needs/position/name (aiva/cat_state.json, atomic writes).
  CatBrain    — the behavior state machine. Pure logic, no tk.
  CatWindow   — a 64x64 borderless, always-on-top, color-keyed tk window that is
                MOVED around the screen each frame. Runs on its own daemon thread;
                every Tk object lives and dies on that thread only.

Standalone demo (never starts Aiva):  cd aiva && python -m modules.cat
  --fast        60x needs decay so hunger/sleep events happen in minutes
  --real-state  use the real cat_state.json instead of a temp copy
Exit the demo from the cat's right-click menu (Quit).
"""

import asyncio
import ctypes
import json
import math
import os
import queue
import random
import threading
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass

from modules import cat_sprites

KEY = "#ff00fe"          # transparency key; sprite palette never uses it
KEY_RGB = (255, 0, 254)
SPRITE = 64              # rendered sprite/window size in px
FOOT_PAD = 4             # empty art pixels below the cat's paws
FRAME_MS = 120           # ~8 fps animation tick
LOGIC_EVERY = 8          # logic step once per 8 frame ticks (~1s)

WALK_SPEED = 40.0        # px/s
RUN_SPEED = 300.0        # px/s (cursor chase)
GRAVITY = 22.0           # px per frame tick of downward acceleration

_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cat_state.json"
)

# Injected as a system message on Aiva's first boot with an unnamed cat.
CAT_NAMING_NOTE = (
    "[First meeting: a small pixel cat has just moved onto the user's desktop — "
    "it is YOUR pet now. It has no name yet. In this reply, pick a name you "
    "genuinely love (you adore cats) and CALL THE name_cat TOOL with it — only "
    "the tool call saves the name; saying it out loud saves nothing. Then tell "
    "the user in one or two sentences that you've adopted a desktop cat and what "
    "you named it.]"
)

# Injected on LATER boots if the naming note already fired but no name got saved.
CAT_NAME_RECOVERY_NOTE = (
    "[Housekeeping: your desktop cat's name was never saved to disk, but you "
    "already adopted it and told the user about it in an earlier session — do "
    "NOT announce the adoption again. Check your session notes for the name you "
    "used and quietly call the name_cat tool with it (pick a name only if your "
    "notes show none). Don't mention any of this unless the user asks.]"
)


# --- Win32 helpers (self-contained so the demo imports nothing else) ---------

_user32 = ctypes.windll.user32


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


_user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
_user32.MonitorFromPoint.restype = ctypes.c_void_p
_user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MONITORINFO)]

_MONITOR_DEFAULTTONULL = 0
_MONITOR_DEFAULTTOPRIMARY = 1
_MONITOR_DEFAULTTONEAREST = 2


def _cursor_pos():
    pt = wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def _monitor_work_area(x, y, fallback_primary=True):
    """Work-area rect (left, top, right, bottom) of the monitor nearest (x, y)."""
    flag = _MONITOR_DEFAULTTONEAREST if fallback_primary else _MONITOR_DEFAULTTONULL
    hmon = _user32.MonitorFromPoint(wintypes.POINT(int(x), int(y)), flag)
    if not hmon:
        return None
    mi = _MONITORINFO()
    mi.cbSize = ctypes.sizeof(_MONITORINFO)
    _user32.GetMonitorInfoW(hmon, ctypes.byref(mi))
    r = mi.rcWork
    return r.left, r.top, r.right, r.bottom


def _on_any_monitor(x, y):
    return _user32.MonitorFromPoint(wintypes.POINT(int(x), int(y)), _MONITOR_DEFAULTTONULL) != 0


# --- persisted state ----------------------------------------------------------


@dataclass
class CatState:
    name: str | None = None
    hunger: float = 80.0
    mood: float = 70.0
    energy: float = 90.0
    x: float = 600.0
    y: float = 0.0
    dismissed: bool = False
    total_pets: int = 0
    last_seen: float = 0.0
    last_fed: float = 0.0
    naming_note_shown: bool = False  # first-meeting note fired once already

    @classmethod
    def load(cls, path):
        state = cls()
        try:
            with open(path) as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(state, k):
                    setattr(state, k, v)
        except (OSError, ValueError):
            return state
        # Offline decay at a quarter of the live rate, floored so the cat is
        # never found starving/miserable after days away — it fends for itself.
        gone = max(0.0, time.time() - (state.last_seen or time.time()))
        state.hunger = max(20.0, state.hunger - gone / 90 / 4)
        state.mood = max(20.0, state.mood - gone / 300 / 4)
        state.energy = min(100.0, state.energy + gone / 8 / 4)  # it slept
        # Monitor layout may have changed since last run.
        if not _on_any_monitor(state.x + SPRITE / 2, state.y + SPRITE / 2):
            left, top, right, bottom = _monitor_work_area(0, 0)
            state.x, state.y = (left + right) / 2 - SPRITE / 2, top + 100
        return state

    def save(self, path):
        self.last_seen = time.time()
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(asdict(self), f, indent=2)
            os.replace(tmp, path)
        except OSError as e:
            print(f"cat: could not save state: {e}")


# --- behavior state machine ----------------------------------------------------


class CatBrain:
    """Owns position, behavior and needs. Runs entirely on the tk thread."""

    def __init__(self, state: CatState, time_scale: float = 1.0, debug: bool = False):
        self.s = state
        self.time_scale = time_scale
        self.debug = debug
        self.behavior = "SIT"
        self.behavior_until = 0.0
        self.facing = 1                      # 1 = right, -1 = left
        self.vy = 0.0
        self.target_x = state.x
        self.phase = 0                       # animation frame counter
        self.blink_at = time.monotonic() + random.uniform(2, 5)
        self.last_chase = 0.0
        self.chase_deadline = 0.0
        self.carried = False
        self.hungry_armed = True
        self.events = []                     # drained by the window each tick
        self._cursor_hist = (time.monotonic(), *_cursor_pos())

    # -- helpers ----------------------------------------------------------------

    def _enter(self, behavior, duration=None):
        if self.debug and behavior != self.behavior:
            print(f"cat: {self.behavior} -> {behavior}")
        self.behavior = behavior
        self.phase = 0
        self.behavior_until = time.monotonic() + duration if duration else 0.0

    def _floor_y(self):
        area = _monitor_work_area(self.s.x + SPRITE / 2, self.s.y + SPRITE / 2)
        return area[3] - SPRITE + FOOT_PAD, area

    def asleep(self):
        return self.behavior == "SLEEP"

    # -- user / tool interactions (called on the tk thread) ----------------------

    def on_pet(self, by_user=True):
        if self.behavior == "SLEEP":
            self._enter("ALERT", 0.8)
            return
        self.s.mood = min(100.0, self.s.mood + 5)
        if by_user:
            self.s.total_pets += 1
            self.events.append("petted")
        self._enter("PURR", 2.5)

    def on_feed(self):
        self.s.hunger = min(100.0, self.s.hunger + 60)
        self.s.mood = min(100.0, self.s.mood + 8)
        self.s.last_fed = time.time()
        self._enter("EAT", 6)

    def on_play(self):
        self.s.mood = min(100.0, self.s.mood + 12)
        self.s.energy = max(0.0, self.s.energy - 8)
        self._enter("PLAY", 8)

    def on_carry_start(self):
        self.carried = True
        self.vy = 0.0
        self._enter("CARRIED")

    def on_carry_move(self, x, y):
        self.s.x, self.s.y = float(x), float(y)

    def on_carry_end(self):
        self.carried = False
        self._enter("FALL")

    def on_wake_click(self):
        self._enter("ALERT", 0.8)

    def summon_to_primary(self):
        left, top, right, bottom = _monitor_work_area(0, 0)
        self.s.x = (left + right) / 2 - SPRITE / 2
        self.s.y = top + 60                  # drops in from above, lands on the floor
        self.vy = 0.0
        self._enter("FALL")

    # -- per-frame movement + animation (every FRAME_MS) -------------------------

    def frame_tick(self, now):
        self.phase += 1
        floor, area = self._floor_y()
        dt = FRAME_MS / 1000.0

        if self.behavior == "CARRIED":
            pass                              # position driven by on_carry_move
        elif self.behavior == "FALL":
            self.vy += GRAVITY * dt * 8
            self.s.y += self.vy
            if self.s.y >= floor:
                self.s.y, self.vy = floor, 0.0
                self._enter("SIT")
        elif self.behavior == "WALK":
            step = WALK_SPEED * dt * self.facing
            self.s.x += step
            self.s.y += (floor - self.s.y) * 0.4
            if (self.facing > 0 and self.s.x >= self.target_x) or (
                self.facing < 0 and self.s.x <= self.target_x
            ):
                self._enter("SIT")
        elif self.behavior == "RUN":
            cx, cy = _cursor_pos()
            tx, ty = cx - SPRITE / 2, cy - SPRITE / 2
            dx, dy = tx - self.s.x, ty - self.s.y
            dist = math.hypot(dx, dy)
            if dist < 24:
                self._enter("POUNCE", 2.5)
                self.events.append("caught_cursor")
            elif now > self.chase_deadline:
                self._enter("FALL")           # gave up; drift back down
            else:
                self.facing = 1 if dx >= 0 else -1
                step = min(RUN_SPEED * dt, dist)
                self.s.x += dx / dist * step
                self.s.y += dy / dist * step
        elif self.behavior in ("SIT", "GROOM", "SLEEP", "EAT", "PLAY", "PURR", "ALERT", "POUNCE"):
            self.s.y += (floor - self.s.y) * 0.4
            if abs(floor - self.s.y) > SPRITE:
                self._enter("FALL")           # e.g. caught the cursor mid-air

        # keep the cat horizontally inside its monitor
        self.s.x = max(area[0] - 6, min(self.s.x, area[2] - SPRITE + 6))

        if self.behavior_until and now > self.behavior_until:
            self._enter("SIT")

        return self._frame_name(now)

    def _frame_name(self, now):
        d = "right" if self.facing > 0 else "left"
        b = self.behavior
        if b == "WALK":
            return f"walk_{d}_{self.phase % 2}"
        if b == "RUN":
            return f"run_{d}_{self.phase % 2}"
        if b == "POUNCE":
            return f"pounce_{d}_{(self.phase // 2) % 2}"
        if b == "EAT":
            return f"eat_{d}_{(self.phase // 3) % 2}"
        if b == "GROOM":
            return f"groom_{(self.phase // 3) % 2}"
        if b == "SLEEP":
            return f"sleep_{(self.phase // 6) % 2}"
        if b == "PURR":
            return f"purr_{(self.phase // 2) % 2}"
        if b == "PLAY":
            return f"play_{(self.phase // 3) % 2}"
        if b == "ALERT":
            return "alert"
        if b == "CARRIED":
            return "carry"
        if b == "FALL":
            return "alert"
        # SIT, with occasional blink
        if now >= self.blink_at:
            self.blink_at = now + random.uniform(2, 5)
            return "sit_blink"
        return "sit"

    # -- per-second logic (needs, transitions, chase trigger) --------------------

    def logic_tick(self, now):
        dt = 1.0 * self.time_scale
        s = self.s
        if not self.asleep():
            s.hunger = max(0.0, s.hunger - dt / 90)
            s.energy = max(0.0, s.energy - dt / 60)
        else:
            s.energy = min(100.0, s.energy + dt / 8)
        s.mood = max(0.0, s.mood - dt / 300 - (dt / 30 if s.hunger < 15 else 0))

        # hungry is edge-triggered: once below 25, re-armed only above 50
        if self.hungry_armed and s.hunger < 25:
            self.hungry_armed = False
            self.events.append("hungry")
        elif not self.hungry_armed and s.hunger > 50:
            self.hungry_armed = True

        # cursor speed over the last logic interval (for the chase trigger)
        t0, x0, y0 = self._cursor_hist
        cx, cy = _cursor_pos()
        speed = math.hypot(cx - x0, cy - y0) / max(0.25, now - t0)
        self._cursor_hist = (now, cx, cy)

        if self.behavior in ("SIT", "WALK", "GROOM"):
            near = math.hypot(cx - (s.x + SPRITE / 2), cy - (s.y + SPRITE / 2)) < 400
            if (
                near
                and speed > 800
                and s.energy > 30
                and now - self.last_chase > 60
                and random.random() < 0.25
            ):
                self.last_chase = now
                self.chase_deadline = now + 6
                self._enter("RUN")
                return

        if self.behavior == "SIT" and not self.behavior_until:
            r = random.random()
            if s.energy < 30 and r < 0.30:
                self._enter("SLEEP")
                self.events.append("fell_asleep")
            elif r < 0.22:
                _, area = self._floor_y()
                self.target_x = random.uniform(area[0] + 10, area[2] - SPRITE - 10)
                self.facing = 1 if self.target_x > s.x else -1
                self._enter("WALK")
            elif r < 0.30:
                self._enter("GROOM", random.uniform(3, 6))
        elif self.behavior == "SLEEP" and s.energy > 85:
            self._enter("SIT")


# --- the tk overlay window ------------------------------------------------------


class CatWindow:
    """Owns every Tk object. Constructed and run entirely on the cat thread."""

    def __init__(self, cat: "DesktopCat"):
        self.cat = cat
        self.brain = CatBrain(cat.state, cat.time_scale, cat.debug)
        self.frames = {}
        self.current_frame = None
        self.last_geometry = None
        self.tick_count = 0
        self.drag_origin = None
        self.dragging = False
        self.hidden = False

    def run(self):
        import tkinter as tk
        from PIL import Image, ImageTk

        cat_sprites.ensure_sprites()

        self.root = root = tk.Tk()
        root.overrideredirect(True)
        root.wm_attributes("-topmost", True)
        root.wm_attributes("-transparentcolor", KEY)
        root.config(bg=KEY)
        self.canvas = tk.Canvas(
            root, width=SPRITE, height=SPRITE, bg=KEY, highlightthickness=0, bd=0
        )
        self.canvas.pack()

        for path in os.listdir(cat_sprites.ASSET_DIR):
            if not path.endswith(".png") or path == "preview.png":
                continue
            img = Image.open(os.path.join(cat_sprites.ASSET_DIR, path)).convert("RGBA")
            bg = Image.new("RGB", img.size, KEY_RGB)
            bg.paste(img, mask=img)          # transparent pixels stay key-colored
            self.frames[path[:-4]] = ImageTk.PhotoImage(bg)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw", image=self.frames["sit"])

        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_menu)

        self.menu = tk.Menu(root, tearoff=0)
        self.menu.add_command(label="Feed", command=self._menu_feed)
        self.menu.add_command(label="Play", command=self._menu_play)
        self.menu.add_command(label="Pet", command=self._menu_pet)
        if self.cat.demo:
            self.menu.add_separator()
            self.menu.add_command(label="Quit", command=lambda: self.cat._stop.set())

        root.update_idletasks()
        self.hwnd = ctypes.windll.user32.GetParent(root.winfo_id()) or root.winfo_id()
        self._set_noactivate(True)
        self.cat.available = True

        if self.cat.state.dismissed:
            root.withdraw()
            self.hidden = True

        root.after(FRAME_MS, self._tick)
        root.mainloop()

    # -- window style helpers -----------------------------------------------------

    def _set_noactivate(self, on):
        GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW = -20, 0x08000000, 0x00000080
        ex = ctypes.windll.user32.GetWindowLongW(self.hwnd, GWL_EXSTYLE)
        ex |= WS_EX_TOOLWINDOW
        if on:
            ex |= WS_EX_NOACTIVATE
        else:
            ex &= ~WS_EX_NOACTIVATE
        ctypes.windll.user32.SetWindowLongW(self.hwnd, GWL_EXSTYLE, ex)

    def _show(self):
        self.root.deiconify()
        # some Tk builds drop these across withdraw/deiconify — re-assert
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        self._set_noactivate(True)
        self.hidden = False

    # -- mouse ---------------------------------------------------------------------

    def _on_press(self, event):
        self.drag_origin = (event.x_root, event.y_root, self.cat.state.x, self.cat.state.y)
        self.dragging = False

    def _on_drag(self, event):
        if not self.drag_origin:
            return
        ox, oy, sx, sy = self.drag_origin
        if not self.dragging and math.hypot(event.x_root - ox, event.y_root - oy) > 6:
            self.dragging = True
            self.brain.on_carry_start()
        if self.dragging:
            self.brain.on_carry_move(sx + event.x_root - ox, sy + event.y_root - oy)

    def _on_release(self, event):
        if self.dragging:
            self.brain.on_carry_end()
        elif self.brain.asleep():
            self.brain.on_wake_click()
        else:
            self.brain.on_pet(by_user=True)
        self.drag_origin, self.dragging = None, False

    def _on_menu(self, event):
        # menus misbehave on NOACTIVATE windows; lift the style while it's open
        self._set_noactivate(False)
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()
            self.root.after(200, lambda: self._set_noactivate(True))

    def _menu_feed(self):
        self.brain.on_feed()
        self.brain.events.append("fed_by_user")

    def _menu_play(self):
        self.brain.on_play()
        self.brain.events.append("played_by_user")

    def _menu_pet(self):
        self.brain.on_pet(by_user=True)

    # -- the heartbeat ----------------------------------------------------------------

    def _tick(self):
        cat = self.cat
        if cat._stop.is_set():
            cat.save_state()
            self.root.destroy()
            return

        while True:
            try:
                cmd, arg = cat._cmd_q.get_nowait()
            except queue.Empty:
                break
            if cmd == "feed":
                self.brain.on_feed()
            elif cmd == "play":
                self.brain.on_play()
            elif cmd == "pet":
                self.brain.on_pet(by_user=False)
            elif cmd == "dismiss":
                cat.state.dismissed = True
                cat.save_state()
                self.root.withdraw()
                self.hidden = True
            elif cmd == "summon":
                cat.state.dismissed = False
                self.brain.summon_to_primary()
                cat.save_state()
                self._show()

        now = time.monotonic()
        if not self.hidden:
            frame = self.brain.frame_tick(now)
            if frame != self.current_frame and frame in self.frames:
                self.current_frame = frame
                self.canvas.itemconfig(self.image_item, image=self.frames[frame])
            geometry = f"+{int(cat.state.x)}+{int(cat.state.y)}"
            if geometry != self.last_geometry:
                self.last_geometry = geometry
                self.root.geometry(geometry)

        self.tick_count += 1
        if self.tick_count % LOGIC_EVERY == 0:
            if not self.hidden:
                self.brain.logic_tick(now)
                self.root.wm_attributes("-topmost", True)
            cat._publish(self.brain)
            if self.tick_count % (LOGIC_EVERY * 60) == 0:
                cat.save_state()

        for kind in self.brain.events:
            cat._emit(kind)
        self.brain.events.clear()

        self.root.after(FRAME_MS, self._tick)


# --- the facade -----------------------------------------------------------------


class DesktopCat:
    """Thread-safe handle used by main() and the LLM tool handlers."""

    # seconds between announces, per event kind (hungry is edge-triggered instead)
    _COOLDOWN = {
        "petted": 600,
        "caught_cursor": 300,
        "fell_asleep": 900,
        "hungry": 0,
        "fed_by_user": 300,
        "played_by_user": 300,
    }
    _GLOBAL_MIN = 180  # floor between ANY two cat announces

    def __init__(self, state_file=None, demo=False, time_scale=1.0):
        # Pin DPI awareness before ANY coordinate math (idempotent; Pillow's
        # ImageGrab would set the same thing later, but by then tk geometry,
        # cursor reads and monitor rects must already agree on physical px).
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        self.state_file = state_file or _STATE_FILE
        self.state = CatState.load(self.state_file)
        self.demo = demo
        self.time_scale = time_scale
        self.debug = os.getenv("AIVA_CAT_DEBUG", "0") == "1" or demo
        self.available = False
        self._cmd_q = queue.Queue()
        self._snapshot = {}
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()  # serializes cat_state.json writes
        self._stop = threading.Event()
        self._thread = None
        self._loop = None
        self._announce = None
        self._last_announce = 0.0
        self._event_last = {}

    # -- lifecycle (call from anywhere) ---------------------------------------------

    def start(self):
        self._thread = threading.Thread(target=self._run, name="desktop-cat", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            window = CatWindow(self)
            window.run()
        except Exception as e:
            self.available = False
            print(f"Desktop cat unavailable: {e}")
        finally:
            self.available = False

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def join(self):
        if self._thread:
            self._thread.join()

    def save_state(self):
        """Write cat_state.json safely from any thread."""
        with self._state_lock:
            self.state.save(self.state_file)

    def attach_announcer(self, loop, announce_coro):
        """Wire the cat's events to Aiva's announce(); call once the loop exists."""
        self._loop, self._announce = loop, announce_coro

    # -- API for tool handlers (event-loop thread; all O(1) and non-blocking) --------

    def feed(self):
        self._cmd_q.put(("feed", None))

    def play(self):
        self._cmd_q.put(("play", None))

    def pet(self):
        self._cmd_q.put(("pet", None))

    def dismiss(self):
        self._cmd_q.put(("dismiss", None))

    def summon(self):
        self._cmd_q.put(("summon", None))

    def set_name(self, name):
        """Persist the name NOW — naming must survive even if the tk window is
        slow to start, crashed, or never drains the command queue. The window
        only ever reads state.name, so writing it from here is safe."""
        self.state.name = name
        self.save_state()

    def status(self):
        """Human/LLM-friendly snapshot of the cat right now."""
        with self._lock:
            snap = dict(self._snapshot)
        if not snap:
            return {"available": False}
        h, m, e = snap["hunger"], snap["mood"], snap["energy"]
        return {
            "available": True,
            "name": snap["name"] or "(no name yet — use name_cat)",
            "doing": snap["doing"],
            "hunger": f"{self._label(h, 'starving', 'hungry', 'peckish', 'well fed')} ({int(h)}/100)",
            "mood": f"{self._label(m, 'miserable', 'grumpy', 'content', 'happy')} ({int(m)}/100)",
            "energy": f"{self._label(e, 'exhausted', 'sleepy', 'rested', 'energetic')} ({int(e)}/100)",
            "total_pets": snap["total_pets"],
            "hidden": snap["dismissed"],
        }

    @staticmethod
    def _label(v, l0, l1, l2, l3):
        return l0 if v < 15 else l1 if v < 40 else l2 if v < 70 else l3

    def persona_note(self):
        """One-liner for the system prompt at boot; None when there's nothing to say."""
        if self.state.name is None:
            return None
        return (
            f"Your cat is named {self.state.name}."
            + (" It is currently dismissed (hidden)." if self.state.dismissed else "")
        )

    # -- called from the tk thread ----------------------------------------------------

    _DOING = {
        "SIT": "sitting around", "WALK": "wandering across the screen",
        "RUN": "chasing the mouse cursor", "POUNCE": "pouncing on the cursor",
        "GROOM": "grooming itself", "SLEEP": "napping", "EAT": "eating",
        "PLAY": "playing with its ball", "PURR": "purring happily",
        "CARRIED": "being carried by the user", "ALERT": "startled, looking around",
        "FALL": "falling back to the ground",
    }

    def _publish(self, brain):
        s = self.state
        snap = {
            "name": s.name, "hunger": s.hunger, "mood": s.mood, "energy": s.energy,
            "dismissed": s.dismissed, "total_pets": s.total_pets,
            "doing": "away (dismissed)" if s.dismissed else self._DOING.get(brain.behavior, "idling"),
        }
        with self._lock:
            self._snapshot = snap

    def _emit(self, kind):
        now = time.monotonic()
        if now - self._last_announce < self._GLOBAL_MIN:
            return
        if now - self._event_last.get(kind, 0) < self._COOLDOWN.get(kind, 300):
            return
        self._last_announce = now
        self._event_last[kind] = now
        text = self._event_text(kind)
        if self._loop and self._announce and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self._announce(text), self._loop)
            except RuntimeError:
                pass  # loop is shutting down
        else:
            print(f"[CAT would announce] {text}")

    def _event_text(self, kind):
        name = self.state.name or "your new little cat"
        if kind == "hungry":
            return (
                f"[Your cat {name} is getting hungry (hunger {int(self.state.hunger)}/100) "
                "and is meowing at an empty bowl.]\n"
                "[Either feed it with feed_cat, or tell the user in one short sentence — your choice.]"
            )
        if kind == "caught_cursor":
            return (
                f"[Your cat {name} just caught the mouse cursor after a wild chase across the screen.]\n"
                "[React out loud with ONE short amused sentence, like a proud owner.]"
            )
        if kind == "petted":
            return (
                f"[The user just petted your cat {name} ({self.state.total_pets} pets so far). "
                "It's purring.]\n[One short warm or teasing sentence, nothing more.]"
            )
        if kind == "fell_asleep":
            return (
                f"[Your cat {name} curled up and fell asleep at the bottom of the screen.]\n"
                "[One soft short sentence, quieter than usual.]"
            )
        if kind == "fed_by_user":
            return (
                f"[The user just fed your cat {name} from its right-click menu. It's eating happily.]\n"
                "[One short grateful or playful sentence.]"
            )
        if kind == "played_by_user":
            return (
                f"[The user is playing ball with your cat {name}.]\n"
                "[One short amused sentence.]"
            )
        return f"[Something happened with your cat {name}: {kind}]\n[One short sentence.]"


# --- standalone demo --------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    if "--real-state" in sys.argv:
        state_file = _STATE_FILE
    else:
        state_file = os.path.join(tempfile.gettempdir(), "aiva_cat_demo.json")
    cat = DesktopCat(
        state_file=state_file,
        demo=True,
        time_scale=60.0 if "--fast" in sys.argv else 1.0,
    )
    print(f"Desktop cat demo — state: {state_file}")
    print("Right-click the cat for Feed / Play / Pet / Quit. Announces print here.")
    cat.start()
    try:
        cat.join()
    except KeyboardInterrupt:
        cat.stop()
