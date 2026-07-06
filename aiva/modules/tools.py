"""Native tool-calling surface for Aiva.

Replaces the old prose-JSON action protocol: each tool is a real function
schema registered with the LLM service, so responses can never truncate or
malform an action.

run_command is deliberately two-step: it only STAGES a command; nothing
executes until the user verbally confirms and the model calls
confirm_pending_command. This gate is non-bypassable.
"""

import ctypes
import os
import subprocess
from ctypes import wintypes

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from modules import file_operations, utilities
from modules.app_launcher import launch_app as _launch_app_impl

# --- desktop overlay control (Spout2OverlayHUD window) ----------------------

def _list_monitors():
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


def _find_overlay_hwnd():
    """Window handle of the Spout2OverlayHUD overlay, or None."""
    user32 = ctypes.windll.user32
    out = subprocess.check_output(
        'tasklist /FI "IMAGENAME eq Spout2OverlayHUD.exe" /FO CSV', shell=True, text=True)
    pids = set()
    for line in out.splitlines()[1:]:
        parts = line.strip('"').split('","')
        if len(parts) > 1 and parts[1].isdigit():
            pids.add(int(parts[1]))
    if not pids:
        return None

    found = []
    proc_type = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, lparam):
        wpid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
        if wpid.value in pids and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return 1

    user32.EnumWindows(proc_type(_cb), 0)
    return found[0] if found else None


_overlay_state = {"mode": "top", "monitor": None}

_SWP_ZONLY = 0x0010 | 0x0002 | 0x0001  # NOACTIVATE | NOMOVE | NOSIZE
_HWND_BOTTOM = 1
_HWND_TOPMOST = -1

_OVERLAY_EXE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tools", "Spout2OverlayHUD.exe")


def _overlay_is_topmost(hwnd):
    return bool(ctypes.windll.user32.GetWindowLongW(hwnd, -20) & 0x8)


def _overlay_running():
    out = subprocess.check_output(
        'tasklist /FI "IMAGENAME eq Spout2OverlayHUD.exe" /FO CSV', shell=True, text=True)
    return "Spout2OverlayHUD" in out


async def _restart_overlay():
    """The only reliable way to regain always-on-top: the overlay creates
    itself topmost, and Windows refuses cross-process topmost promotion."""
    import asyncio

    subprocess.run('taskkill /IM Spout2OverlayHUD.exe /F', shell=True,
                   capture_output=True)
    # wait for the old instance to fully die, or the new one trips the
    # app's single-instance mutex and pops an "already running?" dialog
    for _ in range(15):
        if not _overlay_running():
            break
        await asyncio.sleep(0.3)

    subprocess.Popen([_OVERLAY_EXE], cwd=os.path.dirname(_OVERLAY_EXE))
    for _ in range(20):
        await asyncio.sleep(0.4)
        hwnd = _find_overlay_hwnd()
        if hwnd:
            return hwnd
    return None


def _place_on_monitor(hwnd, idx):
    monitors = _list_monitors()
    if idx is not None and 0 <= idx < len(monitors):
        x, y, w, h = monitors[idx]
        ctypes.windll.user32.MoveWindow(hwnd, x, y, w, h, True)


def ensure_overlay_running():
    """Start the desktop overlay if it isn't already (called at Aiva boot)."""
    if not os.path.exists(_OVERLAY_EXE):
        print("Desktop overlay not found (tools/Spout2OverlayHUD.exe); skipping.")
        return False
    if _overlay_running():
        return False
    subprocess.Popen([_OVERLAY_EXE], cwd=os.path.dirname(_OVERLAY_EXE))
    print("Desktop overlay started")
    return True


def close_overlay():
    """Close the desktop overlay (called at Aiva shutdown)."""
    if _overlay_running():
        subprocess.run('taskkill /IM Spout2OverlayHUD.exe /F', shell=True,
                       capture_output=True)


def _monitor_description():
    """Human hint for the tool schema, computed from the real layout."""
    descs = []
    for i, (x, y, w, h) in enumerate(_list_monitors()):
        pos = []
        if x < 0:
            pos.append("left")
        elif x > 0 and y == 0:
            pos.append("right of primary")
        if y > 0:
            pos.append("bottom")
        elif y < 0:
            pos.append("top")
        if x == 0 and y == 0:
            pos.append("primary/main/center")
        descs.append(f"{i}={w}x{h}" + (f" ({', '.join(pos)})" if pos else ""))
    return "; ".join(descs) if descs else "no monitors detected"

# --- pending shell-command gate ------------------------------------------

_pending_command: str | None = None


# --- tool handlers ---------------------------------------------------------

async def launch_app(params: FunctionCallParams):
    app_name = params.arguments.get("app_name", "")
    result = _launch_app_impl(app_name)
    if result:  # non-empty means "Error: ..."
        await params.result_callback({"success": False, "detail": result})
    else:
        await params.result_callback({"success": True, "launched": app_name})


async def create_file(params: FunctionCallParams):
    try:
        path = file_operations.create_file(
            params.arguments.get("file_path"),
            params.arguments.get("content", ""),
        )
        await params.result_callback({"success": True, "created": path})
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


async def open_file(params: FunctionCallParams):
    try:
        path = file_operations.open_file(params.arguments.get("file_path"))
        await params.result_callback({"success": True, "opened": path})
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


async def list_directory(params: FunctionCallParams):
    try:
        files = file_operations.list_directory(params.arguments.get("path", "."))
        await params.result_callback({"success": True, "entries": files[:100]})
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


async def run_command(params: FunctionCallParams):
    global _pending_command
    command = params.arguments.get("command", "").strip()
    if not command:
        await params.result_callback({"success": False, "error": "no command provided"})
        return
    _pending_command = command
    print(f"[COMMAND STAGED, AWAITING CONFIRMATION] {command}")
    await params.result_callback({
        "status": "pending_confirmation",
        "command": command,
        "instruction": "Tell the user what this command does and ask them to confirm. "
                       "Only after they answer, call confirm_pending_command.",
    })


async def confirm_pending_command(params: FunctionCallParams):
    global _pending_command
    command = _pending_command
    _pending_command = None

    if command is None:
        await params.result_callback({"success": False, "error": "no command is pending"})
        return

    if not params.arguments.get("confirmed", False):
        print(f"[COMMAND DISCARDED] {command}")
        await params.result_callback({"success": True, "status": "discarded", "command": command})
        return

    print(f"[COMMAND EXECUTING] {command}")
    try:
        completed = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        await params.result_callback({
            "success": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-1000:],
            "stderr": completed.stderr[-1000:],
        })
    except subprocess.TimeoutExpired:
        await params.result_callback({"success": False, "error": "command timed out after 30s"})
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


async def get_weather(params: FunctionCallParams):
    result = utilities.get_weather(
        params.arguments.get("city"),
        params.arguments.get("day", "today"),
    )
    await params.result_callback(result)


async def get_datetime(params: FunctionCallParams):
    await params.result_callback({
        "date": utilities.get_current_date(),
        "spoken_time": utilities.get_current_time(),
    })


async def move_avatar_to_monitor(params: FunctionCallParams):
    hwnd = _find_overlay_hwnd()
    if hwnd is None:
        await params.result_callback({
            "success": False,
            "error": "the desktop overlay (Spout2OverlayHUD) isn't running",
        })
        return
    monitors = _list_monitors()
    idx = params.arguments.get("monitor", 0)
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        idx = 0
    if not 0 <= idx < len(monitors):
        await params.result_callback({
            "success": False,
            "error": f"monitor {idx} doesn't exist; there are {len(monitors)} (0..{len(monitors)-1})",
        })
        return
    x, y, w, h = monitors[idx]
    # SetWindowPos silently fails on this layered/topmost overlay window;
    # MoveWindow actually moves it (topmost style is baked in, so it stays on top)
    ok = ctypes.windll.user32.MoveWindow(hwnd, x, y, w, h, True)
    _overlay_state["monitor"] = idx
    if _overlay_state["mode"] == "desktop":
        ctypes.windll.user32.SetWindowPos(hwnd, _HWND_BOTTOM, 0, 0, 0, 0, _SWP_ZONLY)
    await params.result_callback({"success": bool(ok), "moved_to_monitor": idx})


async def set_avatar_layer(params: FunctionCallParams):
    mode = params.arguments.get("layer", "top")
    hwnd = _find_overlay_hwnd()
    if hwnd is None:
        await params.result_callback({"success": False,
                                      "error": "the desktop overlay isn't running"})
        return

    if mode == "desktop":
        ctypes.windll.user32.SetWindowPos(hwnd, _HWND_BOTTOM, 0, 0, 0, 0, _SWP_ZONLY)
        _overlay_state["mode"] = "desktop"
        await params.result_callback({"success": True, "layer": "desktop",
                                      "note": "now sitting behind the user's windows"})
        return

    # top: promote if possible, otherwise restart the overlay (born topmost)
    if not _overlay_is_topmost(hwnd):
        hwnd = await _restart_overlay()
        if hwnd is None:
            await params.result_callback({"success": False,
                                          "error": "overlay didn't come back after restart"})
            return
        _place_on_monitor(hwnd, _overlay_state["monitor"])
    _overlay_state["mode"] = "top"
    await params.result_callback({"success": True, "layer": "top",
                                  "note": "floating above all windows again"})


def make_terminal_handlers(terminals):
    """Terminal/job/Claude-Code handlers close over the TerminalManager."""

    async def open_terminal(params: FunctionCallParams):
        await params.result_callback(terminals.open_session(
            params.arguments.get("name", "main"),
            params.arguments.get("directory", "~"),
        ))

    async def run_in_terminal(params: FunctionCallParams):
        await params.result_callback(await terminals.run(
            params.arguments.get("terminal", "main"),
            params.arguments.get("command", ""),
        ))

    async def read_terminal(params: FunctionCallParams):
        await params.result_callback(terminals.read_session(params.arguments.get("name", "main")))

    async def list_terminals(params: FunctionCallParams):
        await params.result_callback({"success": True, "open": terminals.list_sessions()})

    async def close_terminal(params: FunctionCallParams):
        await params.result_callback(terminals.close_session(params.arguments.get("name", "main")))

    def _spawn_window(args_wt, fallback_ps):
        """Open a visible terminal window: Windows Terminal, else PowerShell."""
        try:
            subprocess.Popen(args_wt)
            return True
        except OSError:
            subprocess.Popen(f'start powershell -NoExit -Command "{fallback_ps}"', shell=True)
            return True

    async def show_terminal(params: FunctionCallParams):
        name = params.arguments.get("name", "main")
        # forgiving lookup: fall back to the only session/job if the name is off
        if name not in terminals.jobs and name not in terminals.sessions:
            if len(terminals.sessions) == 1:
                name = next(iter(terminals.sessions))
            elif len(terminals.jobs) == 1:
                name = next(iter(terminals.jobs))
            else:
                await params.result_callback({"success": False, "error": f"nothing named '{name}'",
                                              "open": terminals.list_sessions()})
                return
        try:
            if name in terminals.jobs:
                log = terminals.jobs[name]["log_path"].replace("\\", "/")
                _spawn_window(["wt", "nt", "--title", f"Aiva: {name}", "powershell",
                               "-NoExit", "-Command", f"Get-Content -Wait '{log}'"],
                              f"Get-Content -Wait '{log}'")
            else:
                cwd = terminals.sessions[name]["cwd"].replace("\\", "/")
                _spawn_window(["wt", "nt", "--title", f"Aiva: {name}", "-d", cwd],
                              f"Set-Location '{cwd}'")
            await params.result_callback({"success": True, "shown": name})
        except Exception as e:
            await params.result_callback({"success": False, "error": f"couldn't open a window: {e}"})

    async def claude_code(params: FunctionCallParams):
        await params.result_callback(await terminals.start_claude(
            job_name=params.arguments.get("job", "claude-1"),
            project_dir=params.arguments.get("project_dir", "."),
            prompt=params.arguments.get("task", ""),
            mode=params.arguments.get("mode", "plan"),
            resume_id=params.arguments.get("resume_session_id"),
        ))

    return (open_terminal, run_in_terminal, read_terminal, list_terminals,
            close_terminal, show_terminal, claude_code)


def make_sleep_handler(mic):
    """Voice-commanded standby: closes her ears until the wake word."""

    async def go_to_sleep(params: FunctionCallParams):
        if not getattr(mic, "ambient", False):
            await params.result_callback({
                "success": False,
                "error": "wake word mode is off, so sleeping would leave me deaf; "
                         "tell the user to say Escape to quit or restart me with AIVA_WAKE_WORD=1",
            })
            return
        mic.sleep()
        await params.result_callback({
            "success": True,
            "status": "asleep — say the wake word to wake me",
        })

    return go_to_sleep


def make_vtube_handlers(vtube):
    """VTube handlers close over the shared VTubeStudio instance."""

    async def vtube_expression(params: FunctionCallParams):
        ok = await vtube.trigger_hotkey(params.arguments.get("expression", ""))
        await params.result_callback({"success": ok})

    async def vtube_move(params: FunctionCallParams):
        ok = await vtube.move_model(
            x=params.arguments.get("x", 0),
            y=params.arguments.get("y", 0),
            rotation=params.arguments.get("rotation", 0),
            size=params.arguments.get("size", 0),
        )
        await params.result_callback({"success": ok})

    return vtube_expression, vtube_move


# --- schemas ---------------------------------------------------------------

TOOL_SCHEMAS = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="launch_app",
        description="Launch an application on the user's Windows PC by name, e.g. 'steam', 'notepad', 'chrome'.",
        properties={"app_name": {"type": "string", "description": "Name of the application"}},
        required=["app_name"],
    ),
    FunctionSchema(
        name="create_file",
        description="Create a text file. Paths may start with 'desktop', 'documents' or 'downloads', e.g. 'desktop\\notes.txt'.",
        properties={
            "file_path": {"type": "string", "description": "Where to create the file"},
            "content": {"type": "string", "description": "Text content of the file"},
        },
        required=["file_path"],
    ),
    FunctionSchema(
        name="open_file",
        description="Open an existing file with its default application.",
        properties={"file_path": {"type": "string", "description": "Path of the file to open"}},
        required=["file_path"],
    ),
    FunctionSchema(
        name="list_directory",
        description="List the files in a directory.",
        properties={"path": {"type": "string", "description": "Directory path, e.g. 'desktop'"}},
        required=[],
    ),
    FunctionSchema(
        name="run_command",
        description="Stage a Windows shell command for execution. It does NOT run yet: the user must confirm first.",
        properties={"command": {"type": "string", "description": "The shell command to stage"}},
        required=["command"],
    ),
    FunctionSchema(
        name="confirm_pending_command",
        description="Execute or discard the staged shell command, according to the user's verbal answer.",
        properties={"confirmed": {"type": "boolean", "description": "true if the user said yes"}},
        required=["confirmed"],
    ),
    FunctionSchema(
        name="get_weather",
        description="Get current weather or tomorrow's forecast for a city.",
        properties={
            "city": {"type": "string", "description": "City name; omit for the user's default city"},
            "day": {"type": "string", "enum": ["today", "tomorrow"]},
        },
        required=[],
    ),
    FunctionSchema(
        name="get_datetime",
        description="Get the current date and time.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="set_avatar_layer",
        description="Change which layer your desktop avatar lives on. 'desktop' = sit behind all "
                    "the user's windows, quietly on the desktop (use when they say 'go to the "
                    "desktop'). 'top' = float above every window again (use for 'come back', "
                    "'show up', 'always on top' — your window may blink for a second).",
        properties={"layer": {"type": "string", "enum": ["top", "desktop"]}},
        required=["layer"],
    ),
    FunctionSchema(
        name="open_terminal",
        description="Open a named terminal session in a directory. Terminals remember their "
                    "working directory and command history — use descriptive names like 'aiva-repo'.",
        properties={
            "name": {"type": "string", "description": "Short name for this terminal"},
            "directory": {"type": "string", "description": "Working directory, e.g. a project path"},
        },
        required=["name", "directory"],
    ),
    FunctionSchema(
        name="run_in_terminal",
        description="Run a shell command in a named terminal. Fast commands return their output; "
                    "slow ones automatically become background jobs and you'll be notified when done.",
        properties={
            "terminal": {"type": "string", "description": "Terminal name"},
            "command": {"type": "string", "description": "PowerShell command to run"},
        },
        required=["terminal", "command"],
    ),
    FunctionSchema(
        name="read_terminal",
        description="Read recent output/history of a terminal or background job by name.",
        properties={"name": {"type": "string", "description": "Terminal or job name"}},
        required=["name"],
    ),
    FunctionSchema(
        name="list_terminals",
        description="List your open terminals and running/finished background jobs — use this to "
                    "recall what you're working on and where.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="close_terminal",
        description="Close a terminal session (kills its background job if one is running).",
        properties={"name": {"type": "string", "description": "Terminal name"}},
        required=["name"],
    ),
    FunctionSchema(
        name="show_terminal",
        description="Open a visible terminal window on the user's screen showing a session's "
                    "directory or a job's live output, so they can watch.",
        properties={"name": {"type": "string", "description": "Terminal or job name"}},
        required=["name"],
    ),
    FunctionSchema(
        name="claude_code",
        description="Delegate coding work to Claude Code (an autonomous coding agent) in a project "
                    "directory. Runs in the background for minutes; you'll be told when it finishes. "
                    "WORKFLOW: first call with mode='plan' to get an implementation plan, summarize "
                    "it to the user and discuss; refine by calling again with resume_session_id (from "
                    "the finished job's claude_session_id) and their feedback; only when the user "
                    "approves, call with mode='execute' and resume_session_id to implement.",
        properties={
            "job": {"type": "string", "description": "Job name, e.g. 'claude-featureX'"},
            "project_dir": {"type": "string", "description": "Absolute path of the project"},
            "task": {"type": "string", "description": "The task, plan feedback, or approval message"},
            "mode": {"type": "string", "enum": ["plan", "execute"],
                     "description": "plan = propose only; execute = actually change code"},
            "resume_session_id": {"type": "string",
                                  "description": "claude_session_id of a previous job to continue that conversation"},
        },
        required=["job", "project_dir", "task", "mode"],
    ),
    FunctionSchema(
        name="go_to_sleep",
        description="Go into standby: your eyes close and you stop listening until the user "
                    "says the wake word. Use when the user says 'go to sleep', 'goodnight', "
                    "'that's all', or wants privacy. Say a SHORT goodnight line after calling it.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="move_avatar_to_monitor",
        description="Move your desktop avatar overlay to another monitor. "
                    f"Monitors left-to-right: {_monitor_description()}. "
                    "Use vtube_move afterwards to position yourself within the screen.",
        properties={"monitor": {"type": "integer", "description": "Monitor index from the list"}},
        required=["monitor"],
    ),
    FunctionSchema(
        name="vtube_expression",
        description="Toggle one of your avatar's expression/prop hotkeys by its exact name. "
                    "The available hotkey names are listed in your system prompt (they may be "
                    "in Chinese — pass them verbatim). Hotkeys are toggles: trigger the same "
                    "one again to turn it off.",
        properties={"expression": {"type": "string", "description": "Exact hotkey name from your avatar's list"}},
        required=["expression"],
    ),
    FunctionSchema(
        name="vtube_move",
        description="Move/resize your avatar model on screen (absolute placement, animated over 0.2s).",
        properties={
            "x": {"type": "number", "description": "Horizontal position, -1 (left edge) to 1 (right edge), 0 = center"},
            "y": {"type": "number", "description": "Vertical position, -1 (bottom) to 1 (top), 0 = center"},
            "rotation": {"type": "number", "description": "Rotation in degrees, 0 = upright"},
            "size": {"type": "number", "description": "Zoom from -100 (tiny) to 100 (huge); 0 is the normal default size. Negative = smaller."},
        },
        required=[],
    ),
])


def register_tools(llm, vtube, mic=None, terminals=None):
    """Register every tool handler on the LLM service."""
    vtube_expression, vtube_move = make_vtube_handlers(vtube)
    llm.register_function("go_to_sleep", make_sleep_handler(mic))

    if terminals is not None:
        (open_terminal, run_in_terminal, read_terminal, list_terminals,
         close_terminal, show_terminal, claude_code) = make_terminal_handlers(terminals)
        llm.register_function("open_terminal", open_terminal)
        llm.register_function("run_in_terminal", run_in_terminal)
        llm.register_function("read_terminal", read_terminal)
        llm.register_function("list_terminals", list_terminals)
        llm.register_function("close_terminal", close_terminal)
        llm.register_function("show_terminal", show_terminal)
        llm.register_function("claude_code", claude_code)

    llm.register_function("launch_app", launch_app)
    llm.register_function("create_file", create_file)
    llm.register_function("open_file", open_file)
    llm.register_function("list_directory", list_directory)
    llm.register_function("run_command", run_command)
    llm.register_function("confirm_pending_command", confirm_pending_command)
    llm.register_function("get_weather", get_weather)
    llm.register_function("get_datetime", get_datetime)
    llm.register_function("move_avatar_to_monitor", move_avatar_to_monitor)
    llm.register_function("set_avatar_layer", set_avatar_layer)
    llm.register_function("vtube_expression", vtube_expression)
    llm.register_function("vtube_move", vtube_move)
