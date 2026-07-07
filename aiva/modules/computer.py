"""Screen vision + mouse/keyboard control for Aiva (Tier 1 interactivity).

Three capabilities, deliberately kept independent so a failure in one never
takes down the voice loop:

- SIGHT: grab a screenshot (PIL, per-monitor) and let a vision model describe
  it. Uses Aiva's own OpenAI stack (gpt-4.1-mini accepts images), so "look at
  my screen" costs a fraction of a cent and adds ~1s to a turn.
- HANDS (reliable): focus a window and click a control BY NAME via the Windows
  UI Automation accessibility tree (pywinauto). No pixel guessing — "click Save"
  invokes the actual Save button, and can do so without moving the real cursor.
- HANDS (escape hatch): raw SendInput mouse + the `keyboard` library for typing
  and hotkeys. Works everywhere (games, canvas apps) but moves the real cursor
  and types into whatever is focused.

pywinauto is imported lazily inside click_ui_element so its absence degrades
only that one tool.
"""

import asyncio
import base64
import ctypes
import io
import os
from ctypes import wintypes

_openai_client = None


def _client():
    """Lazily build a shared AsyncOpenAI client (same key Aiva already uses)."""
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI

        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


# --- monitors --------------------------------------------------------------

def list_monitors():
    """Monitor rects sorted left-to-right: [(left, top, width, height), ...],
    in virtual-screen coordinates (the left monitor can have a negative left)."""
    user32 = ctypes.windll.user32
    monitors = []
    proc_type = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
                                   ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

    def _cb(hmon, hdc, rect, lparam):
        r = rect.contents
        monitors.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
        return 1

    user32.EnumDisplayMonitors(0, 0, proc_type(_cb), 0)
    monitors.sort()
    return monitors


# --- sight -----------------------------------------------------------------

def capture_screen(monitor_index=None, max_edge=1400):
    """Grab a screenshot as JPEG bytes. No monitor = the primary display;
    an index grabs that monitor from the left-to-right list. Downscaled so the
    long edge is at most max_edge px (keeps the vision call cheap and fast)."""
    from PIL import ImageGrab

    if monitor_index is None:
        img = ImageGrab.grab()  # primary display
    else:
        monitors = list_monitors()
        if not 0 <= monitor_index < len(monitors):
            monitor_index = 0
        x, y, w, h = monitors[monitor_index]
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)

    img = img.convert("RGB")
    long_edge = max(img.size)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        img = img.resize((int(img.width * scale), int(img.height * scale)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


async def describe_screen(looking_for=None, monitor_index=None):
    """Capture the screen and have the vision model describe it. Returns a
    short text description (verbatim on error dialogs / key labels)."""
    jpeg = await asyncio.to_thread(capture_screen, monitor_index)
    b64 = base64.b64encode(jpeg).decode("utf-8")
    model = os.getenv("AIVA_VISION_MODEL") or os.getenv("AIVA_LLM_MODEL", "gpt-4.1-mini")

    prompt = (
        "You are Aiva's eyes. Look at this screenshot of the user's Windows screen and "
        "describe what's relevant in 2-4 short sentences, plainly. Quote the exact text of "
        "any error messages, dialog titles, or important button/field labels verbatim. "
        "Name the app/window in focus if you can tell."
    )
    if looking_for:
        prompt += f" The user specifically wants to know: {looking_for}. If you can't find that, say so."

    resp = await _client().chat.completions.create(
        model=model,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    )
    return resp.choices[0].message.content.strip()


# --- window focus (Win32) --------------------------------------------------

def _visible_windows():
    """[(hwnd, title), ...] for visible top-level windows with a title."""
    user32 = ctypes.windll.user32
    out = []
    proc_type = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return 1
        length = user32.GetWindowTextLengthW(hwnd)
        if length:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if buf.value.strip():
                out.append((hwnd, buf.value))
        return 1

    user32.EnumWindows(proc_type(_cb), 0)
    return out


def focus_window(title_substring):
    """Bring the first visible window whose title contains title_substring
    (case-insensitive) to the foreground. Returns the matched title or None."""
    user32 = ctypes.windll.user32
    needle = title_substring.lower()
    match = next((w for w in _visible_windows() if needle in w[1].lower()), None)
    if not match:
        return None
    hwnd, title = match

    SW_RESTORE = 9
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    # SetForegroundWindow is refused unless our thread is "allowed"; attaching
    # to the target's input thread (as the launcher does) lifts that lock.
    fg = user32.GetForegroundWindow()
    our_tid = ctypes.windll.kernel32.GetCurrentThreadId()
    tgt_tid = user32.GetWindowThreadProcessId(fg, None)
    user32.AttachThreadInput(our_tid, tgt_tid, True)
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        user32.AttachThreadInput(our_tid, tgt_tid, False)
    return title


# --- reliable clicking (UI Automation via pywinauto) -----------------------

# Genuinely interactive control types, preferred over plain labels/custom.
_INTERACTIVE = {"Button", "MenuItem", "ListItem", "TreeItem", "Hyperlink",
                "TabItem", "CheckBox", "RadioButton", "SplitButton"}
# Weakly clickable (static labels, custom-drawn controls) — matched only as a
# fallback, since a Text label often shares a name with the real button.
_WEAK = {"Text", "Custom", "Group", "Pane"}


def click_ui_element(name, control_type=None):
    """Find a control BY NAME in the foreground window (via the accessibility
    tree) and invoke it — no pixel guessing. Returns (ok, detail)."""
    try:
        from pywinauto import Desktop
    except ImportError:
        return False, "pywinauto isn't installed, so I can't click things by name"

    try:
        win = Desktop(backend="uia").window(active_only=True)
        win.wait("exists", timeout=2)
    except Exception:
        return False, "couldn't attach to the focused window"

    needle = name.lower()
    try:
        descendants = win.descendants()
    except Exception as e:
        return False, f"couldn't read the window's controls ({e})"

    # Rank every match by (name exactness, control-type strength) and take the
    # best — so a real "Save" Button beats a static "Save" Text label.
    candidates = []
    for el in descendants:
        try:
            txt = (el.window_text() or "").strip()
            ct = el.friendly_class_name()
        except Exception:
            continue
        if not txt:
            continue
        low = txt.lower()
        if low == needle:
            name_rank = 0
        elif needle in low:
            name_rank = 1
        else:
            continue
        if control_type:
            if control_type.lower() not in ct.lower():
                continue
            type_rank = 0
        elif ct in _INTERACTIVE:
            type_rank = 0
        elif ct in _WEAK:
            type_rank = 1
        else:
            continue
        candidates.append(((name_rank, type_rank), el, txt))

    if not candidates:
        return False, f"couldn't find anything named '{name}' to click"

    candidates.sort(key=lambda c: c[0])
    _, el, label = candidates[0]
    try:
        el.invoke()  # accessibility invoke — doesn't move the real cursor
        return True, f"clicked '{label}'"
    except Exception:
        try:
            el.click_input()  # fallback: real click at the element's center
            return True, f"clicked '{label}'"
        except Exception as e:
            return False, f"found '{label}' but couldn't click it ({e})"


# --- raw input escape hatch (SendInput) ------------------------------------

_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79
_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


def _send_mouse(flags, dx=0, dy=0, data=0):
    inp = _INPUT(type=0, mi=_MOUSEINPUT(dx, dy, data, flags, 0, None))
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def _to_absolute(x, y):
    """Map virtual-screen pixel (x, y) to SendInput's 0..65535 range."""
    user32 = ctypes.windll.user32
    vx = user32.GetSystemMetrics(_SM_XVIRTUALSCREEN)
    vy = user32.GetSystemMetrics(_SM_YVIRTUALSCREEN)
    vw = user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN)
    vh = user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN)
    ax = int((x - vx) * 65535 / max(vw - 1, 1))
    ay = int((y - vy) * 65535 / max(vh - 1, 1))
    return ax, ay


def click_at(x, y, button="left", double=False):
    """Move the real cursor to (x, y) in virtual-screen pixels and click."""
    ax, ay = _to_absolute(int(x), int(y))
    _send_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, ax, ay)
    down, up = ((MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP) if button == "right"
                else (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP))
    clicks = 2 if double else 1
    for _ in range(clicks):
        _send_mouse(down)
        _send_mouse(up)
    return True


def scroll(direction="down", amount=3):
    """Mouse-wheel scroll at the current cursor position. amount = notches."""
    delta = 120 * int(amount)
    if direction == "down":
        delta = -delta
    _send_mouse(MOUSEEVENTF_WHEEL, data=delta)
    return True
