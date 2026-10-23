"""Shared bits between a sister and her avatar renderer process.

Both sides import this, so it must stay light: stdlib + ctypes only, no Qt,
no live2d, no pipecat. The contract lives here — the state file the renderer
writes (and the character process reads to adopt a running renderer across
her self-restarts) and the newline-JSON encoding both ends speak.

Each renderer *instance* owns one state file: Aiva keeps the historical
`aiva/avatar_state.json`, every other instance (her sister Saga) gets
`aiva/<instance>_avatar_state.json` — see `state_path()`.
"""

import ctypes
import json
import os
from ctypes import wintypes

# 2 (2026-10-13): `activity` + `cue` (the book she reads from, the work panel). A sister that
# adopts a renderer reporting an older version replaces it — the body outlives her restarts, so
# without this a code update to the renderer would never reach the running body.
# 3 (2026-10-13): `witness` (Saga's constellation), its sky following her layer and tilt and
# surviving a switch-off. A v2 Saga body would otherwise be adopted as-is and never show it.
# 4 (2026-10-14): her sky hushes while she speaks. No new command, but a visible behaviour a
# running v3 body would never pick up.
# 5 (2026-10-14): her sky sits to her upper left and drifts instead of twinkling.
# 6 (2026-10-14): her sky as she described it: 2-4 px stars, lines barely brighter than them,
# shrunk to stay to her upper left near a screen edge.
# 7 (2026-10-14): her refinements: a 12.5% bigger sky (taller where a screen edge holds it narrow),
# a livelier drift, stars a touch less uniform, softer outermost lines.
# 8 (2026-10-16): `cue nod` — a small nod with a backchannel (mhm while he talks).
PROTOCOL_VERSION = 8

_AIVA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Written ONLY by the renderer (atomic replace): {port, pid, boot_id, instance,
# model, placement: {x, y, rotation, size}, monitor, layer[, constellation]}
STATE_FILE = os.path.join(_AIVA_DIR, "avatar_state.json")


def state_path(instance=None) -> str:
    """The state file a renderer instance owns.

    None / "aiva" -> STATE_FILE (Aiva's file, unchanged); anything else ->
    aiva/<instance>_avatar_state.json. Both the renderer (writer) and the
    client (reader) derive the path from the same instance name, so they can
    never disagree about where a body advertises itself.
    """
    if not instance or instance == "aiva":
        return STATE_FILE
    return os.path.join(_AIVA_DIR, f"{instance}_avatar_state.json")


def read_state(path=STATE_FILE):
    """The renderer's state file as a dict, or None if missing/corrupt."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_state_atomic(state: dict, path=STATE_FILE):
    """Atomic write so a reader can never see a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, path)


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
