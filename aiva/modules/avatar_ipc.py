"""Shared bits between Aiva and her avatar renderer process.

Both sides import this, so it must stay light: stdlib + ctypes only, no Qt,
no live2d, no pipecat. The contract lives here — the state file the renderer
writes (and Aiva reads to adopt a running renderer across her self-restarts)
and the newline-JSON encoding both ends speak.
"""

import ctypes
import json
import os
from ctypes import wintypes

PROTOCOL_VERSION = 1

# Written ONLY by the renderer (atomic replace): {port, pid, boot_id,
# placement: {x, y, rotation, size}, monitor, layer}
STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "avatar_state.json")


def read_state():
    """The renderer's state file as a dict, or None if missing/corrupt."""
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_state_atomic(state: dict):
    """Atomic write so a reader can never see a half-written file."""
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, STATE_FILE)


def encode(msg: dict) -> bytes:
    """One IPC message: JSON (UTF-8, CJK kept readable) + newline."""
    return (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")


def list_monitors():
    """Monitor rects sorted left-to-right: [(left, top, width, height), ...]"""
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


def pid_alive(pid: int) -> bool:
    """True if a process with this pid exists (no psutil needed)."""
    SYNCHRONIZE = 0x00100000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return False
    # still-alive check: a reused pid could belong to a new process, but the
    # hello handshake right after this is what actually confirms identity
    STILL_ACTIVE = 259
    code = wintypes.DWORD()
    alive = True
    if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
        alive = code.value == STILL_ACTIVE
    kernel32.CloseHandle(handle)
    return alive
