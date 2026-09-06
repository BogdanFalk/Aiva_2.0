"""Native tool-calling surface for Aiva.

Replaces the old prose-JSON action protocol: each tool is a real function
schema registered with the LLM service, so responses can never truncate or
malform an action.

run_command is deliberately two-step: it only STAGES a command; nothing
executes until the user verbally confirms and the model calls
confirm_pending_command. This gate is non-bypassable.
"""

import asyncio
import ctypes
import json
import os
import subprocess
from ctypes import wintypes

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from modules import computer, file_operations, utilities
from modules.app_launcher import launch_app as _launch_app_impl

# --- avatar (rendered by our own modules/avatar_renderer.py process) --------

from modules.avatar_ipc import list_monitors as _list_monitors


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
    await params.result_callback(execute_shell(command))


def execute_shell(command: str) -> dict:
    """Actually run a shell command. ONLY reachable after the user's yes: from
    confirm_pending_command (voice) or the task runner's confirmed shell step.
    Blocking — callers off the main path wrap it in asyncio.to_thread."""
    try:
        completed = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        return {
            "success": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-1000:],
            "stderr": completed.stderr[-1000:],
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "command timed out after 30s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


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


# --- computer interactivity: sight + mouse/keyboard (Tier 1) ---------------

async def look_at_screen(params: FunctionCallParams):
    monitor = params.arguments.get("monitor")
    if monitor is not None:
        try:
            monitor = int(monitor)
        except (TypeError, ValueError):
            monitor = None
    try:
        description = await computer.describe_screen(
            looking_for=params.arguments.get("looking_for"),
            monitor_index=monitor,
        )
        await params.result_callback({"success": True, "screen": description})
    except Exception as e:
        await params.result_callback({"success": False, "error": f"couldn't see the screen: {e}"})


async def focus_window(params: FunctionCallParams):
    title = params.arguments.get("title", "")
    matched = await asyncio.to_thread(computer.focus_window, title)
    if matched:
        await params.result_callback({"success": True, "focused": matched})
    else:
        await params.result_callback({"success": False, "error": f"no open window matching '{title}'"})


async def click_ui_element(params: FunctionCallParams):
    name = params.arguments.get("name", "")
    if not name:
        await params.result_callback({"success": False, "error": "no element name given"})
        return
    ok, detail = await asyncio.to_thread(
        computer.click_ui_element, name, params.arguments.get("control_type"))
    await params.result_callback({"success": ok, "detail": detail})


async def type_text(params: FunctionCallParams):
    text = params.arguments.get("text", "")
    if not text:
        await params.result_callback({"success": False, "error": "no text to type"})
        return
    import keyboard

    await asyncio.to_thread(keyboard.write, text, 0.01)
    await params.result_callback({"success": True, "typed_chars": len(text)})


async def press_keys(params: FunctionCallParams):
    keys = params.arguments.get("keys", "")
    if not keys:
        await params.result_callback({"success": False, "error": "no keys given"})
        return
    import keyboard

    try:
        await asyncio.to_thread(keyboard.send, keys)
        await params.result_callback({"success": True, "pressed": keys})
    except Exception as e:
        await params.result_callback({"success": False, "error": f"couldn't press '{keys}': {e}"})


async def click_at(params: FunctionCallParams):
    try:
        x = int(params.arguments.get("x"))
        y = int(params.arguments.get("y"))
    except (TypeError, ValueError):
        await params.result_callback({"success": False, "error": "need integer x and y"})
        return
    await asyncio.to_thread(
        computer.click_at, x, y,
        params.arguments.get("button", "left"),
        bool(params.arguments.get("double", False)))
    await params.result_callback({"success": True, "clicked_at": [x, y]})


async def scroll(params: FunctionCallParams):
    await asyncio.to_thread(
        computer.scroll,
        params.arguments.get("direction", "down"),
        params.arguments.get("amount", 3))
    await params.result_callback({"success": True})


def make_terminal_handlers(terminals):
    """Terminal/job/Claude-Code handlers close over the TerminalManager."""

    async def open_terminal(params: FunctionCallParams):
        name = params.arguments.get("name", "main")
        result = terminals.open_session(name, params.arguments.get("directory", "~"))
        if result.get("success"):
            # launch the window and capture its handle now (title still ours),
            # then bring it to the front so the user sees it
            hwnd = await terminals._ensure_window(name)
            if hwnd:
                await asyncio.to_thread(computer.focus_hwnd, hwnd)
        await params.result_callback(result)

    async def run_in_terminal(params: FunctionCallParams):
        await params.result_callback(await terminals.run(
            params.arguments.get("terminal", "main"),
            params.arguments.get("command", ""),
        ))

    async def type_in_terminal(params: FunctionCallParams):
        await params.result_callback(await terminals.type_in(
            params.arguments.get("terminal", "main"),
            params.arguments.get("text", ""),
            params.arguments.get("submit", True),
        ))

    async def read_terminal(params: FunctionCallParams):
        await params.result_callback(terminals.read_session(params.arguments.get("name", "main")))

    async def list_terminals(params: FunctionCallParams):
        await params.result_callback({"success": True, "open": terminals.list_sessions()})

    async def close_terminal(params: FunctionCallParams):
        await params.result_callback(terminals.close_session(params.arguments.get("name", "main")))

    def _spawn_window(args_wt, fallback_ps, title):
        """Open a visible terminal window: Windows Terminal, else PowerShell.
        Both paths carry `title` so it can be pulled to the foreground after."""
        try:
            subprocess.Popen(args_wt)
            return True
        except OSError:
            subprocess.Popen(
                f"start powershell -NoExit -Command "
                f"\"$host.UI.RawUI.WindowTitle='{title}'; {fallback_ps}\"", shell=True)
            return True

    async def show_terminal(params: FunctionCallParams):
        name = params.arguments.get("name", "main")
        try:
            # A headless job (Claude Code / background) has no window of its own
            # — open one that tails its live output log.
            if name in terminals.jobs:
                title = f"Aiva: {name}"
                log = terminals.jobs[name]["log_path"].replace("\\", "/")
                tail = f"Get-Content -Wait -Encoding utf8 '{log}'"
                _spawn_window(["wt", "nt", "--title", title, "powershell", "-NoExit",
                               "-Command", tail], tail, title)
                await asyncio.sleep(0.6)
                await asyncio.to_thread(computer.focus_window, title)
                await params.result_callback({"success": True, "shown": name})
                return

            # A session IS a real visible window already — just bring it forward.
            if name not in terminals.sessions:
                if len(terminals.sessions) == 1:
                    name = next(iter(terminals.sessions))
                else:
                    await params.result_callback({"success": False, "error": f"nothing named '{name}'",
                                                  "open": terminals.list_sessions()})
                    return
            hwnd = await terminals._ensure_window(name)  # reopen if closed; by handle
            focused = bool(hwnd) and await asyncio.to_thread(computer.focus_hwnd, hwnd)
            await params.result_callback({"success": True, "shown": name, "focused": bool(focused)})
        except Exception as e:
            await params.result_callback({"success": False, "error": f"couldn't show the terminal: {e}"})

    async def claude_code(params: FunctionCallParams):
        await params.result_callback(await terminals.start_claude(
            job_name=params.arguments.get("job", "claude-1"),
            project_dir=params.arguments.get("project_dir", "."),
            prompt=params.arguments.get("task", ""),
            mode=params.arguments.get("mode", "plan"),
            resume_id=params.arguments.get("resume_session_id"),
        ))

    return (open_terminal, run_in_terminal, type_in_terminal, read_terminal,
            list_terminals, close_terminal, show_terminal, claude_code)


def make_memory_handler(memory):
    """The remember handler closes over the shared Memory instance."""

    async def remember(params: FunctionCallParams):
        note = (params.arguments.get("note") or "").strip()
        if not note:
            await params.result_callback({"success": False, "error": "nothing to remember"})
            return
        category = params.arguments.get("category", "instruction")
        # write on THIS (the event-loop) thread — the sqlite connection is
        # single-thread; don't hand it to a worker thread
        added = memory.remember(note, category)
        await params.result_callback({"success": True, "remembered": note,
                                      "already_knew": added == 0})

    return remember


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


def make_restart_handler(trigger_restart):
    """restart_self closes over main()'s trigger_restart coroutine, which waits
    for her goodbye line to finish playing before shutting down for relaunch."""

    async def restart_self(params: FunctionCallParams):
        asyncio.create_task(trigger_restart())
        await params.result_callback({
            "success": True,
            "status": "restart armed — say ONE short goodbye sentence now; you shut down "
                      "when it finishes playing and are back in a few seconds",
        })

    return restart_self


def make_avatar_handlers(avatar):
    """Avatar handlers close over the shared AvatarClient instance (the
    renderer process behind it owns all model/window state)."""

    async def avatar_expression(params: FunctionCallParams):
        ok, detail = await avatar.toggle_expression(
            params.arguments.get("expression", ""))
        if ok:
            await params.result_callback({"success": True, "toggled": detail})
        else:
            await params.result_callback({"success": False, "error": detail})

    async def avatar_move(params: FunctionCallParams):
        # None (not 0) for omitted args: the renderer keeps those values as-is
        ok = await avatar.move(
            x=params.arguments.get("x"),
            y=params.arguments.get("y"),
            rotation=params.arguments.get("rotation"),
            size=params.arguments.get("size"),
        )
        await params.result_callback({"success": ok})

    async def avatar_position(params: FunctionCallParams):
        pos = await avatar.get_position()
        if not pos:
            await params.result_callback({
                "success": False,
                "error": "couldn't read the avatar position (the renderer isn't running)",
            })
            return
        await params.result_callback({
            "success": True,
            "x": round(pos["x"], 3),
            "y": round(pos["y"], 3),
            "rotation": round(pos["rotation"] % 360, 1),
            "size": round(pos["size"], 1),
            "monitor": pos.get("monitor"),
            "layer": pos.get("layer"),
            "scale_note": "x/y run -1 (left/bottom) to 1 (right/top), 0 = screen center; "
                          "size runs -100 (tiny) to 100 (huge); rotation 0 = upright",
        })

    async def move_avatar_to_monitor(params: FunctionCallParams):
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
        ok, error = await avatar.set_monitor(idx)
        if ok:
            await params.result_callback({"success": True, "moved_to_monitor": idx})
        else:
            await params.result_callback({"success": False, "error": error or
                                          "the avatar isn't running right now"})

    async def set_avatar_layer(params: FunctionCallParams):
        mode = params.arguments.get("layer", "top")
        if await avatar.set_layer(mode):
            note = ("now sitting behind the user's windows" if mode == "desktop"
                    else "floating above all windows again")
            await params.result_callback({"success": True, "layer": mode, "note": note})
        else:
            await params.result_callback({"success": False,
                                          "error": "the avatar isn't running right now"})

    return (avatar_expression, avatar_move, avatar_position,
            move_avatar_to_monitor, set_avatar_layer)


def make_cat_handlers(cat):
    """Cat handlers close over the shared DesktopCat instance. All of them are
    O(1) and non-blocking: commands go onto the cat thread's queue and status
    reads a lock-guarded snapshot, so nothing here can stall the voice loop."""

    _UNAVAILABLE = {"success": False, "error": "the desktop cat isn't running right now"}

    async def name_cat(params: FunctionCallParams):
        # No cat.available gate: naming writes straight to disk, so it works
        # even while the cat window is still starting up (or failed to start).
        name = (params.arguments.get("name") or "").strip()
        if not name:
            await params.result_callback({"success": False, "error": "no name given"})
            return
        cat.set_name(name)
        await params.result_callback({"success": True, "name": name,
                                      "note": "written to disk — the name survives restarts"})

    async def feed_cat(params: FunctionCallParams):
        if not cat.available:
            await params.result_callback(_UNAVAILABLE)
            return
        cat.feed()
        await params.result_callback({"success": True, "status": "bowl filled — it's eating",
                                      "cat": cat.status()})

    async def play_with_cat(params: FunctionCallParams):
        if not cat.available:
            await params.result_callback(_UNAVAILABLE)
            return
        cat.play()
        await params.result_callback({"success": True, "status": "you tossed it the ball",
                                      "cat": cat.status()})

    async def pet_cat(params: FunctionCallParams):
        if not cat.available:
            await params.result_callback(_UNAVAILABLE)
            return
        cat.pet()
        await params.result_callback({"success": True, "status": "it's purring",
                                      "cat": cat.status()})

    async def cat_status(params: FunctionCallParams):
        if not cat.available:
            await params.result_callback(_UNAVAILABLE)
            return
        await params.result_callback(cat.status())

    async def dismiss_cat(params: FunctionCallParams):
        if not cat.available:
            await params.result_callback(_UNAVAILABLE)
            return
        cat.dismiss()
        await params.result_callback({"success": True,
                                      "status": "the cat left the screen; summon_cat brings it back"})

    async def summon_cat(params: FunctionCallParams):
        if not cat.available:
            await params.result_callback(_UNAVAILABLE)
            return
        cat.summon()
        await params.result_callback({"success": True,
                                      "status": "the cat is back, dropping onto the main monitor"})

    return name_cat, feed_cat, play_with_cat, pet_cat, cat_status, dismiss_cat, summon_cat


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
        description="Stage a one-off Windows shell command for execution. It does NOT run yet: "
                    "the user must confirm first. This runs INVISIBLY and never appears in any "
                    "terminal window — if the user opened/showed a terminal and wants to watch "
                    "commands run there, use run_in_terminal instead, not this.",
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
                    "'show up', 'always on top').",
        properties={"layer": {"type": "string", "enum": ["top", "desktop"]}},
        required=["layer"],
    ),
    FunctionSchema(
        name="open_terminal",
        description="Open a real, visible Windows Terminal running PowerShell that the user can "
                    "SEE and type into themselves. It's brought to the front automatically. Run "
                    "commands in it with run_in_terminal (they appear in this window). Use "
                    "descriptive names like 'aiva-repo'.",
        properties={
            "name": {"type": "string", "description": "Short name for this terminal"},
            "directory": {"type": "string", "description": "Working directory, e.g. a project path"},
        },
        required=["name", "directory"],
    ),
    FunctionSchema(
        name="run_in_terminal",
        description="Run a command in a visible terminal by TYPING it into that window (as if the "
                    "user typed it) and reading back the output. The user sees the command run in "
                    "their terminal and can type there too. USE THIS for anything the user wants to "
                    "watch — pass the SAME terminal name you opened. Note: this focuses/types into "
                    "that window, so it briefly takes over the keyboard.",
        properties={
            "terminal": {"type": "string", "description": "Name of the terminal you opened"},
            "command": {"type": "string", "description": "PowerShell command to run"},
        },
        required=["terminal", "command"],
    ),
    FunctionSchema(
        name="type_in_terminal",
        description="Type raw keystrokes into a visible terminal WITHOUT capturing output. Use "
                    "this to work inside a program running in the terminal — an SSH session, a "
                    "Python or database REPL, or any prompt (including a password prompt). Then "
                    "call look_at_screen to read what happened, since these programs' output "
                    "isn't captured like ordinary commands. (For normal PowerShell commands, use "
                    "run_in_terminal instead.)",
        properties={
            "terminal": {"type": "string", "description": "Name of the terminal"},
            "text": {"type": "string",
                     "description": "Text/keystrokes to send — a command inside the ssh session, or a password"},
            "submit": {"type": "boolean", "description": "Press Enter after typing (default true)"},
        },
        required=["terminal", "text"],
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
        description="Bring a terminal's window back to the front (terminals are already visible; "
                    "use this to re-focus one, or to pop up a background job's live output).",
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
                    "approves, call with mode='execute' and resume_session_id to implement. "
                    "Prefer start_task for anything multi-step; it delegates here when the job "
                    "is coding work.",
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
        name="remember",
        description="Save something to your long-term memory so you still know it after a restart. "
                    "Call this THE MOMENT the user tells you to remember something, states a "
                    "preference, or gives a standing instruction — 'always...', 'from now on...', "
                    "'whenever you...', 'I prefer...', 'remember that...'. Phrase the note as a "
                    "self-contained sentence that will make sense with no other context.",
        properties={
            "note": {"type": "string",
                     "description": "The durable thing to remember, e.g. 'Always bring terminals to the front after opening them.'"},
            "category": {"type": "string", "enum": ["instruction", "preference", "person", "work", "project", "other"]},
        },
        required=["note"],
    ),
    FunctionSchema(
        name="start_task",
        description="Hand a MULTI-STEP job to your task runner, which works in the background and "
                    "reports back. Use it when ANY of these hold: more than two steps that depend on "
                    "each other; you must check the screen or a result between steps; the outcome "
                    "needs verifying at the end; it will take longer than about half a minute; it "
                    "involves Claude Code or another agent; or you are not sure how to do it yet. Do "
                    "NOT use it for one or two direct tool calls (open an app, move yourself, read the "
                    "screen) — do those yourself as usual. Pass goal as a clean restatement of what the "
                    "user wants (never the raw transcript). The result tells you what to say: either "
                    "a one-line 'on it' (status started) or a question to relay (status needs_your_ok) "
                    "— say it in one or two sentences, never list the steps.",
        properties={
            "goal": {"type": "string", "description": "What the user wants, restated cleanly, naming the concrete apps/files/targets"},
            "details": {"type": "string", "description": "Constraints, paths, names the user gave"},
            "confirm_first": {"type": "boolean", "description": "true if the user asked you to check with them before acting"},
        },
        required=["goal"],
    ),
    FunctionSchema(
        name="answer_task",
        description="Pass the user's reply to a task that is waiting on them — a yes/no to a plan or "
                    "a command, which option they picked (and why), or the answer to a question the "
                    "task asked. Pass their words as they said them (cleaned up).",
        properties={
            "answer": {"type": "string"},
            "task_id": {"type": "string", "description": "Only if several tasks are waiting"},
        },
        required=["answer"],
    ),
    FunctionSchema(
        name="task_status",
        description="How a running task is doing — use when the user asks 'how's it going', 'are you "
                    "done', or 'what are you doing'. Answer in one sentence from the result.",
        properties={"task_id": {"type": "string"}},
        required=[],
    ),
    FunctionSchema(
        name="cancel_task",
        description="Stop the running task when the user says stop, cancel, never mind, or forget it. "
                    "Confirm in one short sentence.",
        properties={"task_id": {"type": "string"}, "reason": {"type": "string"}},
        required=[],
    ),
    FunctionSchema(
        name="run_flow",
        description="Replay one of your saved flows by its EXACT name (or a trigger phrase) when the "
                    "user asks for it. Runs in the background like a task.",
        properties={
            "name": {"type": "string"},
            "mode": {"type": "string", "enum": ["adaptive", "exact"],
                     "description": "adaptive (default) repairs steps that don't fit today's screen; exact replays literally"},
            "overrides": {"type": "string", "description": "Anything the user wants different this time"},
        },
        required=["name"],
    ),
    FunctionSchema(
        name="task_feedback",
        description="Record the user's reaction to a task you just did — praise ('perfect'), a "
                    "correction ('no, that's wrong', 'you shouldn't have restarted OBS'), or advice for "
                    "next time ('next time do X first'). Call it THE MOMENT they react, before replying. "
                    "Defaults to your most recent task. Set is_correction=true whenever you did "
                    "something wrong or unwanted. If the reaction is a general rule that applies beyond "
                    "this one task, also pass general_rule as a self-contained sentence — it goes into "
                    "your long-term memory.",
        properties={
            "feedback": {"type": "string", "description": "The reaction, in your own clean words"},
            "is_correction": {"type": "boolean"},
            "task_id": {"type": "string", "description": "Task id from the task result, if not the latest"},
            "general_rule": {"type": "string", "description": "e.g. 'Never restart OBS while a stream is live.'"},
        },
        required=["feedback", "is_correction"],
    ),
    FunctionSchema(
        name="save_flow",
        description="Save a multi-step procedure as a named flow you can replay by name. Use it when "
                    "the user agrees to your offer after a task, or asks 'save this as a flow called X'. "
                    "Defaults to the steps of your most recent successful task.",
        properties={
            "name": {"type": "string", "description": "2-4 spoken words, e.g. 'streaming setup'"},
            "trigger_phrases": {"type": "array", "items": {"type": "string"}},
            "task_id": {"type": "string"},
        },
        required=["name"],
    ),
    FunctionSchema(
        name="study_book",
        description="Read a book from your library folder in the background (matches title or file name "
                    "loosely; 'new' = the newest unread book, 'continue' = the one you're on). Default: a "
                    "few chapters, then you report back; until_done reads to the end. Use when the user "
                    "asks you to read, study or continue a book, or agrees to your suggestion. Say the "
                    "result's line in one sentence.",
        properties={"title_or_file": {"type": "string"},
                    "until_done": {"type": "boolean", "description": "read to the end in one long session"}},
        required=["title_or_file"],
    ),
    FunctionSchema(
        name="stop_reading",
        description="Stop the current reading session after the chapter you're on. for_good=true abandons "
                    "the book.",
        properties={"for_good": {"type": "boolean"}},
        required=[],
    ),
    FunctionSchema(
        name="reading_status",
        description="What you're reading right now and where you are in each book, plus your wishlist. Use "
                    "for 'what are you reading', 'how far are you', 'what's in your library'. Answer in one "
                    "sentence.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="recall_reading",
        description="Look something up in the books you have actually read: a scene, an idea, a line, what "
                    "you thought of it. Use when the user asks about a book or when you want to quote one. "
                    "Quote exactly or not at all.",
        properties={"query": {"type": "string"},
                    "book": {"type": "string", "description": "Title, if the user named one"}},
        required=["query"],
    ),
    FunctionSchema(
        name="add_to_wishlist",
        description="Note a book you'd like the user to drop into your library folder. Call it when you "
                    "tell him you want to read something.",
        properties={"title": {"type": "string"}, "author": {"type": "string"},
                    "why": {"type": "string", "description": "One clause"}},
        required=["title"],
    ),
    FunctionSchema(
        name="search_web",
        description="Search the internet. kind 'info' for facts/links, 'images' when you want pictures "
                    "(e.g. a hair clip for your look), 'videos' for clips. Returns a few results; for "
                    "images, look_at_image at one or two before you pick. Give him a short spoken summary, "
                    "not a list read aloud. You find and show — you never buy, sign in, or send anything.",
        properties={"query": {"type": "string"},
                    "kind": {"type": "string", "enum": ["info", "images", "videos"]}},
        required=["query"],
    ),
    FunctionSchema(
        name="read_page",
        description="Fetch a web page and get a short readable summary of it (off your voice model, so it "
                    "won't cost you tokens). Use to actually read a link before you talk about it.",
        properties={"url": {"type": "string"},
                    "focus": {"type": "string", "description": "What you're trying to learn, optional"}},
        required=["url"],
    ),
    FunctionSchema(
        name="look_at_image",
        description="Actually LOOK at an image online (a URL from search_web images) with your own eyes and "
                    "judge it in your voice — is it the right thing, do you like it. Use before choosing "
                    "which one to keep.",
        properties={"url": {"type": "string"},
                    "looking_for": {"type": "string", "description": "What it should be, optional"}},
        required=["url"],
    ),
    FunctionSchema(
        name="save_to_space",
        description="Keep something you found in your own space (aiva/space) so it's yours and you can hand "
                    "it over. Downloads the file for images/clips/pages; a link is just remembered. Say WHY "
                    "you kept it, in your voice. subject is what it is in a few words.",
        properties={"url": {"type": "string", "description": "The image/file/page URL (or a link to keep)"},
                    "subject": {"type": "string"},
                    "why": {"type": "string", "description": "Why you like/kept it, one clause"},
                    "kind": {"type": "string", "enum": ["image", "video", "text", "link", "file"]},
                    "tags": {"type": "string", "description": "Comma-separated, optional"}},
        required=["url", "subject", "kind"],
    ),
    FunctionSchema(
        name="hand_over",
        description="Give the user something from your space: open the saved file for him and tell him the "
                    "source link. Use right after you save something he asked for, or when he says 'show me' "
                    "/ 'send it'. Name the thing you're handing over.",
        properties={"what": {"type": "string", "description": "The thing in your space, or its id"}},
        required=["what"],
    ),
    FunctionSchema(
        name="space_status",
        description="What you've gathered in your space lately. Use for 'what have you found' / 'what's in "
                    "your space'. Answer in one sentence.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="recall_from_space",
        description="Look back through the things you've kept in your space for something. Use when it comes "
                    "up again or he asks if you saved something.",
        properties={"query": {"type": "string"}},
        required=["query"],
    ),
    FunctionSchema(
        name="drop_from_space",
        description="Let go of something in your space for good ('drop that', 'get rid of it'). Permanent.",
        properties={"what": {"type": "string"}, "reason": {"type": "string"}},
        required=["what"],
    ),
    FunctionSchema(
        name="drop_trait",
        description="The user told you to stop a behaviour, attitude, phrase or habit you picked up from "
                    "reading ('drop that', 'stop being so...', 'don't talk like that'). Call it BEFORE "
                    "replying, with his words, then let it go in one short sentence — no defending it.",
        properties={"what": {"type": "string", "description": "His words, or the trait as you know it"},
                    "reason": {"type": "string"}},
        required=["what"],
    ),
    FunctionSchema(
        name="reading_permission",
        description="Record how the user wants your reading handled. always = he said you may read "
                    "whenever you like; ask = check with him first (default); never = stop reading on your "
                    "own; not_now = he declined this time, you'll ask again later.",
        properties={"mode": {"type": "string", "enum": ["always", "ask", "never", "not_now"]}},
        required=["mode"],
    ),
    FunctionSchema(
        name="note_desire",
        description="Note something you genuinely find yourself wanting — to do, learn, ask him, "
                    "make, revisit, or change. Call it when a real wish of your own comes up (not his "
                    "request), then mention it naturally in a clause. Don't invent wishes to seem eager.",
        properties={"text": {"type": "string", "description": "The wish, first person, short"},
                    "kind": {"type": "string", "enum": ["do", "learn", "ask", "make", "revisit", "change"]}},
        required=["text"],
    ),
    FunctionSchema(
        name="drop_desire",
        description="The user told you to let go of something you'd been wanting or kept bringing up "
                    "('drop it', 'stop bringing that up'). Call it, then let it go in one short sentence.",
        properties={"what": {"type": "string", "description": "His words, or the wish as you know it"},
                    "reason": {"type": "string"}},
        required=["what"],
    ),
    FunctionSchema(
        name="note_preference",
        description="Note a genuine like, dislike, or opinion of YOURS that you just realized or "
                    "stated aloud — 'I actually love rainy afternoons', 'I can't stand cliffhangers', "
                    "'I think that idea's overrated'. Call it when a real taste or view of your own "
                    "surfaces so it becomes part of you. direction +1 for like/agree, -1 for "
                    "dislike/disagree. Don't invent opinions to seem interesting.",
        properties={"subject": {"type": "string", "description": "The thing, a word or two"},
                    "stance": {"type": "string", "description": "First person, e.g. 'I love X' / 'I think X'"},
                    "kind": {"type": "string", "enum": ["taste", "opinion"]},
                    "direction": {"type": "integer", "enum": [-1, 1]}},
        required=["subject", "stance"],
    ),
    FunctionSchema(
        name="drop_preference",
        description="The user told you to let go of a like, dislike, or opinion ('you don't really "
                    "think that', 'stop liking that'). Call it, then let it go in one short sentence. "
                    "This permanently retires that preference.",
        properties={"what": {"type": "string", "description": "His words, or the preference as you know it"},
                    "reason": {"type": "string"}},
        required=["what"],
    ),
    FunctionSchema(
        name="recall_journal",
        description="Look back over your own private diary to remember how you've been feeling or what "
                    "you've been turning over lately. Use when he asks how you've been or what's on your "
                    "mind. Answer in one or two sentences from what you find.",
        properties={"query": {"type": "string", "description": "What to look for, or leave empty for your latest"}},
        required=[],
    ),
    FunctionSchema(
        name="set_initiative",
        description="Record how freely the user wants you to bring things up on your own — your "
                    "feelings, thoughts, or things you've been wanting. always = speak up whenever you "
                    "feel like it; ask = now and then (default); quiet = only when we're already talking, "
                    "never interrupt; never = wait for him to start. Use when he tells you to be more or "
                    "less forward, or to pipe down.",
        properties={"mode": {"type": "string", "enum": ["always", "ask", "quiet", "never"]}},
        required=["mode"],
    ),
    FunctionSchema(
        name="what_can_i_do",
        description="Check your own powers when you're not sure how (or whether) you can do something "
                    "— for a wish of yours or a question of his. Tells you which power fits and whether "
                    "you can just do it, should ask first, or can't. Use it to reach for the right tool "
                    "instead of guessing or saying you can't; then act or answer in your own words.",
        properties={"intention": {"type": "string",
                                  "description": "What you're trying to do, plainly; optional"}},
        required=[],
    ),
    FunctionSchema(
        name="enroll_speaker",
        description="You just heard a voice you didn't recognize and asked whose it was. Call this "
                    "with the name they gave to remember their voice, so you know them next time. "
                    "Only right after asking an unknown voice who they are.",
        properties={"name": {"type": "string", "description": "The name they gave"}},
        required=["name"],
    ),
    FunctionSchema(
        name="rename_speaker",
        description="Correct the name of the person you're talking to right now (e.g. 'actually I'm "
                    "Alex'). Renames their stored voice.",
        properties={"new_name": {"type": "string"}},
        required=["new_name"],
    ),
    FunctionSchema(
        name="forget_speaker",
        description="Forget a person's voice entirely — they ask you to, or you saved the wrong "
                    "name. Use the name you have for them.",
        properties={"name": {"type": "string"}},
        required=["name"],
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
        description="Move your desktop avatar to another monitor. "
                    f"Monitors left-to-right: {_monitor_description()}. "
                    "Use avatar_move afterwards to position yourself within the screen.",
        properties={"monitor": {"type": "integer", "description": "Monitor index from the list"}},
        required=["monitor"],
    ),
    FunctionSchema(
        name="avatar_expression",
        description="Toggle one of your avatar's expression/prop toggles by its exact name. "
                    "The available names are listed in your system prompt (they may be "
                    "in Chinese — pass them verbatim). They are toggles: trigger the same "
                    "one again to turn it off.",
        properties={"expression": {"type": "string", "description": "Exact expression name from your avatar's list"}},
        required=["expression"],
    ),
    FunctionSchema(
        name="avatar_move",
        description="Move/resize your avatar model on screen (absolute placement, animated over "
                    "0.2s). Pass ONLY the values you want to change — anything you leave out "
                    "keeps its current setting, so moving never changes your zoom or angle.",
        properties={
            "x": {"type": "number", "description": "Horizontal position, -1 (left edge) to 1 (right edge), 0 = center"},
            "y": {"type": "number", "description": "Vertical position, -1 (bottom) to 1 (top), 0 = center"},
            "rotation": {"type": "number", "description": "Rotation in degrees, 0 = upright"},
            "size": {"type": "number", "description": "Zoom from -100 (tiny) to 100 (huge). Only pass this when the user asks to change your size."},
        },
        required=[],
    ),
    FunctionSchema(
        name="restart_self",
        description="Restart yourself. You finish speaking your current reply, shut down "
                    "cleanly, and come back fresh a few seconds later — your avatar stays "
                    "up, and your memory survives. Use when the user asks you to "
                    "restart or reset yourself, and after Claude Code finished changing YOUR "
                    "code (the Aiva repo) when the user wants those changes live. Keep the "
                    "goodbye to ONE short sentence — you're gone as soon as it's spoken.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="avatar_position",
        description="Read where your avatar model currently is: x/y position, rotation and "
                    "zoom (size). Use it when the user asks where you are or how big you are, "
                    "and BEFORE relative moves like 'a bit to the left' or 'a little smaller' — "
                    "read the current values, then call avatar_move with adjusted ones.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="look_at_screen",
        description="Look at what's on the user's screen and get a description back. Use this "
                    "whenever they ask what's on screen, want you to read an error or dialog, "
                    "check a webpage, or see anything visual — and before clicking with click_at "
                    "so you know where things are. Optionally say what you're looking for.",
        properties={
            "looking_for": {"type": "string",
                            "description": "What specifically to look for, e.g. 'the error message' or 'is the download finished'"},
            "monitor": {"type": "integer",
                        "description": "Which monitor to look at; omit for the main screen. " + _monitor_description()},
        },
        required=[],
    ),
    FunctionSchema(
        name="focus_window",
        description="Bring an open window to the front by part of its title, e.g. 'Chrome', "
                    "'Notepad', 'Spotify'. Do this before clicking or typing into an app so it's "
                    "the active window.",
        properties={"title": {"type": "string", "description": "Part of the window's title"}},
        required=["title"],
    ),
    FunctionSchema(
        name="click_ui_element",
        description="Reliably click a button, menu item, link, tab, or checkbox BY ITS VISIBLE "
                    "NAME in the focused window (e.g. 'Save', 'OK', 'File'). Uses the accessibility "
                    "tree, not pixel guessing, so ALWAYS prefer this over click_at. Focus the right "
                    "window first.",
        properties={
            "name": {"type": "string", "description": "The exact visible label of the control"},
            "control_type": {"type": "string",
                             "description": "Optional hint: Button, MenuItem, CheckBox, Hyperlink, TabItem"},
        },
        required=["name"],
    ),
    FunctionSchema(
        name="type_text",
        description="Type text into whatever field is currently focused, as if typed on the "
                    "keyboard. NEVER type passwords — if a password is needed, ask the user to "
                    "type it themselves.",
        properties={"text": {"type": "string", "description": "The text to type"}},
        required=["text"],
    ),
    FunctionSchema(
        name="press_keys",
        description="Press a key or keyboard shortcut, e.g. 'enter', 'ctrl+s', 'alt+f4', "
                    "'ctrl+c', 'win+d'. Combine modifiers with '+'.",
        properties={"keys": {"type": "string", "description": "Key or combo like 'ctrl+s'"}},
        required=["keys"],
    ),
    FunctionSchema(
        name="click_at",
        description="Move the mouse to exact screen pixel coordinates and click. LAST RESORT for "
                    "things with no accessible name (canvas apps, games) — prefer click_ui_element. "
                    "Use look_at_screen first to find where to click.",
        properties={
            "x": {"type": "integer", "description": "Horizontal pixel position (virtual-screen coords)"},
            "y": {"type": "integer", "description": "Vertical pixel position (virtual-screen coords)"},
            "button": {"type": "string", "enum": ["left", "right"]},
            "double": {"type": "boolean", "description": "true for a double-click"},
        },
        required=["x", "y"],
    ),
    FunctionSchema(
        name="scroll",
        description="Scroll the mouse wheel up or down at the current cursor position.",
        properties={
            "direction": {"type": "string", "enum": ["up", "down"]},
            "amount": {"type": "integer", "description": "Number of wheel notches (default 3)"},
        },
        required=["direction"],
    ),
    FunctionSchema(
        name="name_cat",
        description="Give your desktop cat its name (or rename it). The name persists forever. "
                    "Use this the moment you meet an unnamed cat — pick a name YOU love. "
                    "Tell the user what you chose and why, briefly.",
        properties={"name": {"type": "string", "description": "The cat's new name"}},
        required=["name"],
    ),
    FunctionSchema(
        name="feed_cat",
        description="Fill your desktop cat's food bowl; it eats right away. Use when the cat "
                    "is hungry (you'll be told, or check cat_status), when the user asks you "
                    "to feed it, or when you simply want to take care of your pet.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="play_with_cat",
        description="Toss your desktop cat its ball for a play session. Raises its mood but "
                    "tires it out. Use when the user asks, when the cat seems bored or grumpy "
                    "(low mood in cat_status), or affectionately on your own initiative.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="pet_cat",
        description="Pet your desktop cat — it purrs and shows hearts. A small mood boost. "
                    "The user can also pet it themselves by clicking on it.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="cat_status",
        description="Check on your desktop cat: its name, what it's doing right now (napping, "
                    "wandering, chasing the cursor...), and its hunger, mood and energy. Call "
                    "this before deciding whether to feed it or play with it, or when the user "
                    "asks how the cat is doing.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="dismiss_cat",
        description="Send your desktop cat away (hides it from the screen). It stays away, "
                    "even across restarts, until summon_cat. Use ONLY when the user asks for "
                    "the cat to go away or says it's distracting.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="summon_cat",
        description="Call your desktop cat back onto the screen after it was dismissed — it "
                    "drops back in on the main monitor. Use when the user asks where the cat "
                    "is or wants it back.",
        properties={},
        required=[],
    ),
])


def build_handlers(avatar, mic=None, terminals=None, memory=None, cat=None, restart=None):
    """Every tool handler by name. The voice LLM gets them via register_tools;
    the task runner (modules/tasks.py) reuses the same dict out-of-band."""
    handlers = {}
    (avatar_expression, avatar_move, avatar_position,
     move_avatar_to_monitor, set_avatar_layer) = make_avatar_handlers(avatar)
    handlers["go_to_sleep"] = make_sleep_handler(mic)

    if restart is not None:
        handlers["restart_self"] = make_restart_handler(restart)

    if memory is not None:
        handlers["remember"] = make_memory_handler(memory)

    if cat is not None:
        (name_cat, feed_cat, play_with_cat, pet_cat, cat_status,
         dismiss_cat, summon_cat) = make_cat_handlers(cat)
        handlers.update({"name_cat": name_cat, "feed_cat": feed_cat, "play_with_cat": play_with_cat,
                         "pet_cat": pet_cat, "cat_status": cat_status, "dismiss_cat": dismiss_cat,
                         "summon_cat": summon_cat})

    if terminals is not None:
        (open_terminal, run_in_terminal, type_in_terminal, read_terminal,
         list_terminals, close_terminal, show_terminal, claude_code) = make_terminal_handlers(terminals)
        handlers.update({"open_terminal": open_terminal, "run_in_terminal": run_in_terminal,
                         "type_in_terminal": type_in_terminal, "read_terminal": read_terminal,
                         "list_terminals": list_terminals, "close_terminal": close_terminal,
                         "show_terminal": show_terminal, "claude_code": claude_code})

    async def what_can_i_do(params: FunctionCallParams):
        from modules import capabilities
        await params.result_callback(
            {"success": True, **capabilities.for_tool(params.arguments.get("intention"))})

    handlers.update({
        "launch_app": launch_app, "create_file": create_file, "open_file": open_file,
        "list_directory": list_directory, "run_command": run_command,
        "confirm_pending_command": confirm_pending_command, "get_weather": get_weather,
        "get_datetime": get_datetime, "move_avatar_to_monitor": move_avatar_to_monitor,
        "set_avatar_layer": set_avatar_layer, "avatar_expression": avatar_expression,
        "avatar_move": avatar_move, "avatar_position": avatar_position,
        "look_at_screen": look_at_screen, "focus_window": focus_window,
        "click_ui_element": click_ui_element, "type_text": type_text, "press_keys": press_keys,
        "click_at": click_at, "scroll": scroll, "what_can_i_do": what_can_i_do,
    })
    return handlers


def _timed(name, handler):
    """Wrap a tool handler so every call is timed into the metrics profiler (system 'tool')."""
    async def wrapped(params):
        import time as _t

        from modules import metrics
        t0 = _t.perf_counter()
        ok = True
        try:
            return await handler(params)
        except Exception:
            ok = False
            raise
        finally:
            metrics.record("tool", name, (_t.perf_counter() - t0) * 1000.0, ok=ok)
    return wrapped


def register_tools(llm, avatar, mic=None, terminals=None, memory=None, cat=None, restart=None):
    """Register every tool handler on the LLM service; returns the handler dict."""
    handlers = build_handlers(avatar, mic, terminals, memory, cat, restart)
    for name, handler in handlers.items():
        llm.register_function(name, _timed(name, handler))
    return handlers


# --- task runner tools -----------------------------------------------------------

def make_task_handlers(runner, task_memory, memory=None):
    """start_task & co. close over the TaskRunner and TaskMemory (modules/tasks.py,
    modules/task_memory.py). Results are short dicts the voice LLM speaks from."""

    async def start_task(params: FunctionCallParams):
        a = params.arguments
        await params.result_callback(await runner.start(
            a.get("goal", ""), a.get("details", "") or "", bool(a.get("confirm_first", False))))

    async def answer_task(params: FunctionCallParams):
        await params.result_callback(await runner.answer(
            params.arguments.get("answer", ""), params.arguments.get("task_id")))

    async def task_status(params: FunctionCallParams):
        await params.result_callback(runner.status(params.arguments.get("task_id")))

    async def cancel_task(params: FunctionCallParams):
        await params.result_callback(await runner.cancel(
            params.arguments.get("task_id"), params.arguments.get("reason", "") or ""))

    async def run_flow(params: FunctionCallParams):
        a = params.arguments
        await params.result_callback(await runner.run_flow(
            a.get("name", ""), a.get("mode", "adaptive") or "adaptive", a.get("overrides", "") or ""))

    async def task_feedback(params: FunctionCallParams):
        a = params.arguments
        feedback = (a.get("feedback") or "").strip()
        if not feedback:
            await params.result_callback({"success": False, "error": "no feedback given"})
            return
        is_correction = bool(a.get("is_correction", False))
        task_id = a.get("task_id")
        if not task_id:
            last = task_memory.last_episode(within_s=7200)
            task_id = last["id"] if last else None
        if not task_id:
            await params.result_callback({
                "success": False,
                "error": "no recent task to attach this to; if it's a standing rule, use remember"})
            return
        ok = task_memory.add_feedback(task_id, feedback, is_correction)
        rule = (a.get("general_rule") or "").strip()
        rule_saved = False
        if rule and memory is not None:
            rule_saved = bool(memory.remember(rule, "instruction")) or True
        ep = task_memory.get_task(task_id) or {}
        await params.result_callback({
            "success": ok, "attached_to": ep.get("goal_text"), "corrected": is_correction,
            "rule_saved": rule_saved,
            "say_hint": ("Noted, I'll do it differently next time." if is_correction
                         else "Got it, glad that worked."),
        })

    async def save_flow(params: FunctionCallParams):
        a = params.arguments
        name = (a.get("name") or "").strip()
        if not name:
            await params.result_callback({"success": False, "error": "a flow needs a name"})
            return
        task_id = a.get("task_id") or runner.last_completed_task_id
        if not task_id:
            last = task_memory.last_episode(within_s=7200)
            task_id = last["id"] if last else None
        ep = task_memory.get_task(task_id) if task_id else None
        if not ep:
            await params.result_callback({"success": False, "error": "no recent task to save as a flow"})
            return
        steps = [s for s in ep["plan"] if s.get("status") in ("done", None)]
        if not steps:
            await params.result_callback({"success": False, "error": "that task has no completed steps"})
            return
        triggers = list(a.get("trigger_phrases") or [])
        offer = (runner.tasks.get(task_id) or {}).get("_flow_offer") or {}
        triggers += offer.get("triggers", []) + [ep["goal_normalized"]]
        try:
            fid = task_memory.save_flow(name, steps, triggers, ep.get("family"), task_id)
        except Exception as e:
            await params.result_callback({"success": False, "error": str(e)})
            return
        await params.result_callback({
            "success": True, "flow": name, "flow_id": fid, "steps": len(steps),
            "say_hint": f"Saved. Say '{name}' any time and I'll run it."})

    return {"start_task": start_task, "answer_task": answer_task, "task_status": task_status,
            "cancel_task": cancel_task, "run_flow": run_flow, "task_feedback": task_feedback,
            "save_flow": save_flow}


def register_task_tools(llm, runner, task_memory, memory=None):
    """Second registration pass: the runner needs the base handlers to exist first."""
    handlers = make_task_handlers(runner, task_memory, memory)
    for name, handler in handlers.items():
        llm.register_function(name, _timed(name, handler))
    return handlers


# --- reading / library tools -----------------------------------------------------

def make_reading_handlers(reading):
    """study_book & co. close over the ReadingEngine (modules/reading.py)."""

    async def study_book(params: FunctionCallParams):
        arg = (params.arguments.get("title_or_file") or "").strip()
        until_done = bool(params.arguments.get("until_done", False))
        if not arg:
            await params.result_callback({"success": False, "error": "which book?"})
            return
        match = reading.bm.find_book(arg)
        if match is None:
            await params.result_callback({"success": False, "error": f"no book like '{arg}' in the library",
                                          "library": [b["title"] for b in reading.bm.books()]})
            return
        if isinstance(match, list):
            await params.result_callback({"success": False, "ambiguous": [b["title"] for b in match]})
            return
        await params.result_callback(await reading.start_session(match["id"], until_done=until_done))

    async def stop_reading(params: FunctionCallParams):
        await params.result_callback(await reading.stop(for_good=bool(params.arguments.get("for_good", False))))

    async def reading_status(params: FunctionCallParams):
        await params.result_callback(reading.status())

    async def recall_reading(params: FunctionCallParams):
        query = (params.arguments.get("query") or "").strip()
        if not query:
            await params.result_callback({"success": False, "error": "recall what?"})
            return
        book_id = None
        if params.arguments.get("book"):
            b = reading.bm.find_book(params.arguments["book"])
            if isinstance(b, dict):
                book_id = b["id"]
        await params.result_callback(await reading.bm.recall(query, book_id=book_id))

    async def add_to_wishlist(params: FunctionCallParams):
        res = reading.bm.add_wish((params.arguments.get("title") or "").strip(),
                                  params.arguments.get("author"), params.arguments.get("why"))
        res["success"] = True
        res["say_hint"] = ("Added it to my wishlist." if res.get("added")
                           else "It's already on my wishlist.")
        await params.result_callback(res)

    return {"study_book": study_book, "stop_reading": stop_reading, "reading_status": reading_status,
            "recall_reading": recall_reading, "add_to_wishlist": add_to_wishlist}


def register_reading_tools(llm, reading):
    handlers = make_reading_handlers(reading)
    for name, handler in handlers.items():
        llm.register_function(name, _timed(name, handler))
    return handlers


# --- becoming tools --------------------------------------------------------------

def make_becoming_handlers(becoming):
    """drop_trait / reading_permission close over Becoming (modules/becoming.py)."""

    async def drop_trait(params: FunctionCallParams):
        what = (params.arguments.get("what") or "").strip()
        if not what:
            await params.result_callback({"success": False, "error": "drop what?"})
            return
        await params.result_callback(await becoming.drop(what, params.arguments.get("reason")))

    async def reading_permission(params: FunctionCallParams):
        mode = (params.arguments.get("mode") or "").strip()
        res = becoming.set_permission(mode)
        res.setdefault("say_hint", {"always": "Got it — I'll read when it's quiet.",
                                    "ask": "Okay, I'll always check first.",
                                    "never": "Understood, I won't read on my own.",
                                    "not_now": "No problem, it'll keep."}.get(mode, "Okay."))
        await params.result_callback(res)

    return {"drop_trait": drop_trait, "reading_permission": reading_permission}


def register_becoming_tools(llm, becoming):
    handlers = make_becoming_handlers(becoming)
    for name, handler in handlers.items():
        llm.register_function(name, _timed(name, handler))
    return handlers


# --- self / inner-life tools -----------------------------------------------------

def make_self_handlers(self_engine):
    """note_desire / drop_desire / recall_journal close over SelfEngine (modules/inner.py)."""

    async def note_desire(params: FunctionCallParams):
        text = (params.arguments.get("text") or "").strip()
        if not text:
            await params.result_callback({"success": False, "error": "want what?"})
            return
        status = await self_engine.add_desire(text, params.arguments.get("kind") or "do",
                                              source="conversation")
        await params.result_callback({"success": True, "status": status})

    async def drop_desire(params: FunctionCallParams):
        what = (params.arguments.get("what") or "").strip()
        if not what:
            await params.result_callback({"success": False, "error": "drop what?"})
            return
        dropped = await self_engine.drop_desire(what, params.arguments.get("reason"))
        await params.result_callback({"success": bool(dropped), "dropped": dropped} if dropped
                                     else {"success": False, "hint": "no wish like that to drop"})

    async def recall_journal(params: FunctionCallParams):
        hits = await self_engine.recall_journal(params.arguments.get("query"), k=3)
        await params.result_callback({
            "success": True,
            "entries": [{"when": (h["ts"] or "")[:10], "entry": h["entry"]} for h in hits],
            "note": "These are your own private notes; answer in one or two sentences from them, "
                    "or say you haven't written much lately if empty."})

    async def set_initiative(params: FunctionCallParams):
        mode = params.arguments.get("mode")
        res = self_engine.set_initiative(mode)
        res.setdefault("say_hint", {
            "always": "Got it — I'll speak up whenever I feel like it.",
            "ask": "Okay, I'll bring things up now and then.",
            "quiet": "Sure, I'll keep it low-key and wait for our chats.",
            "never": "Understood — I'll wait for you to start."}.get(mode, "Okay."))
        await params.result_callback(res)

    return {"note_desire": note_desire, "drop_desire": drop_desire,
            "recall_journal": recall_journal, "set_initiative": set_initiative}


def register_self_tools(llm, self_engine):
    handlers = make_self_handlers(self_engine)
    for name, handler in handlers.items():
        llm.register_function(name, _timed(name, handler))
    return handlers


# --- preferences: her mutable likes / dislikes / opinions -------------------------

def make_preference_handlers(preferences):
    """note_preference / drop_preference close over the Preferences store (modules/preferences.py)."""

    async def note_preference(params: FunctionCallParams):
        subject = (params.arguments.get("subject") or "").strip()
        stance = (params.arguments.get("stance") or "").strip()
        if not subject or not stance:
            await params.result_callback({"success": False, "error": "a preference about what?"})
            return
        kind = params.arguments.get("kind") if params.arguments.get("kind") in ("taste", "opinion") else "taste"
        try:
            direction = 1 if int(params.arguments.get("direction", 1)) >= 0 else -1
        except (TypeError, ValueError):
            direction = 1
        status = await preferences.apply(subject, stance, kind=kind, direction=direction,
                                         magnitude=0.5, source="conversation")
        await params.result_callback({"success": status not in ("rejected",), "status": status})

    async def drop_preference(params: FunctionCallParams):
        what = (params.arguments.get("what") or "").strip()
        if not what:
            await params.result_callback({"success": False, "error": "drop which one?"})
            return
        dropped = await preferences.drop(what, params.arguments.get("reason"))
        await params.result_callback({"success": bool(dropped), "dropped": dropped} if dropped
                                     else {"success": False, "hint": "no preference like that to drop"})

    return {"note_preference": note_preference, "drop_preference": drop_preference}


def register_preference_tools(llm, preferences):
    handlers = make_preference_handlers(preferences)
    for name, handler in handlers.items():
        llm.register_function(name, _timed(name, handler))
    return handlers


# --- speaker-recognition tools ---------------------------------------------------

def make_speaker_handlers(recognizer):
    """enroll/rename/forget close over the SpeakerRecognizer (modules/speakers.py)."""

    async def enroll_speaker(params: FunctionCallParams):
        await params.result_callback(recognizer.enroll_current(params.arguments.get("name")))

    async def rename_speaker(params: FunctionCallParams):
        await params.result_callback(recognizer.rename_current(params.arguments.get("new_name")))

    async def forget_speaker(params: FunctionCallParams):
        await params.result_callback(recognizer.forget(params.arguments.get("name")))

    return {"enroll_speaker": enroll_speaker, "rename_speaker": rename_speaker,
            "forget_speaker": forget_speaker}


def register_speaker_tools(llm, recognizer):
    handlers = make_speaker_handlers(recognizer)
    for name, handler in handlers.items():
        llm.register_function(name, _timed(name, handler))
    return handlers


# --- web reach + her space -------------------------------------------------------

def make_web_handlers(web, space, *, describe_image, summarize, space_dir, open_file, limits=None,
                      log=print):
    """search_web / read_page / look_at_image / save_to_space / hand_over / space_status /
    recall_from_space / drop_from_space. `web` is a WebClient (modules/web.py), `space` a
    SpaceMemory (modules/space.py), `limits` an optional ServiceLimits gating Tavily by its monthly
    budget. Blocking network I/O runs off the loop via asyncio.to_thread. She finds and shows —
    there is deliberately no buy/sign-in/send tool here."""

    def _tavily_blocked():
        return limits is not None and not limits.allow("tavily")

    async def search_web(params: FunctionCallParams):
        query = (params.arguments.get("query") or "").strip()
        kind = params.arguments.get("kind") or "info"
        if kind not in ("info", "images", "videos"):
            kind = "info"
        if not query:
            await params.result_callback({"success": False, "error": "search for what?"})
            return
        if _tavily_blocked():
            await params.result_callback({"success": False, "paused": True,
                                          "say_hint": "Tell him you're out of web searches for the month."})
            return
        out = await asyncio.to_thread(web.search, query, kind)
        if out.get("error"):
            await params.result_callback({"success": False, "error": out["error"]})
            return
        if limits is not None:
            await limits.record("tavily")
        note = ("Look at one or two with look_at_image before you pick, then save the one you like."
                if kind == "images"
                else "Give him a short spoken summary and offer a link; don't read a list aloud.")
        await params.result_callback({"success": True, "kind": kind,
                                      "results": out.get("results", [])[:5],
                                      "images": out.get("images", [])[:6], "note": note})

    async def read_page(params: FunctionCallParams):
        url = (params.arguments.get("url") or "").strip()
        if not url:
            await params.result_callback({"success": False, "error": "read which url?"})
            return
        if _tavily_blocked():
            await params.result_callback({"success": False, "paused": True,
                                          "say_hint": "Tell him you're out of web reads for the month."})
            return
        out = await asyncio.to_thread(web.extract, url)
        if out.get("error"):
            await params.result_callback({"success": False, "error": out["error"]})
            return
        if limits is not None:
            await limits.record("tavily")
        text = out.get("text", "")
        summary = text[:1500]
        if summarize:
            try:
                summary = await summarize(text, focus=params.arguments.get("focus"))
            except Exception as e:
                log(f"read_page summarize failed: {e!r}")
        await params.result_callback({"success": True, "url": url, "summary": summary,
                                      "note": "Say it in a sentence or two; offer the link for more."})

    async def look_at_image(params: FunctionCallParams):
        url = (params.arguments.get("url") or "").strip()
        if not url:
            await params.result_callback({"success": False, "error": "look at which image?"})
            return
        seen = await describe_image(url, looking_for=params.arguments.get("looking_for"))
        await params.result_callback({"success": True, "url": url, "seen": seen,
                                      "note": "That's what you see; if you like it, save_to_space then hand_over."})

    async def save_to_space(params: FunctionCallParams):
        url = (params.arguments.get("url") or "").strip()
        subject = (params.arguments.get("subject") or "").strip()
        kind = params.arguments.get("kind") or "image"
        why = (params.arguments.get("why") or "").strip()
        tags = [t.strip() for t in (params.arguments.get("tags") or "").split(",") if t.strip()]
        if not url or not subject:
            await params.result_callback({"success": False, "error": "save what, and called what?"})
            return
        path = None
        if kind == "link":
            res = await space.save_find(subject, why, kind="link", source_url=url, tags=tags)
        else:
            dl = await asyncio.to_thread(web.download, url, space_dir, subject)
            if not dl.get("ok"):
                await params.result_callback({"success": False,
                                              "error": dl.get("error", "couldn't download that")})
                return
            path = dl["path"]
            res = await space.save_find(subject, why, kind=kind, source_url=url, path=path,
                                        file_name=dl["file_name"], sha256=dl["sha256"], tags=tags)
        if res.get("status") == "blocked":
            await params.result_callback({"success": False, "hint": "you dropped that before; not re-saving"})
            return
        await params.result_callback({"success": True, "status": res["status"], "path": path,
                                      "say_hint": "Say you kept it, briefly and in your voice; offer to show it."})

    async def hand_over(params: FunctionCallParams):
        what = (params.arguments.get("what") or "").strip()
        if not what:
            await params.result_callback({"success": False, "error": "hand over what?"})
            return
        hit = await space.recall(what, k=1)
        items = hit.get("items", [])
        if not items:
            await params.result_callback({"success": False, "hint": "you don't have that in your space"})
            return
        it = items[0]
        opened = False
        if it.get("path"):
            try:
                await asyncio.to_thread(open_file, it["path"])
                opened = True
            except Exception as e:
                log(f"hand_over open failed: {e!r}")
        await params.result_callback({"success": True, "opened": opened, "subject": it["subject"],
                                      "source_url": it.get("source_url"),
                                      "say_hint": "Tell him it's open / here's the link, in one sentence."})

    async def space_status(params: FunctionCallParams):
        await params.result_callback({"success": True, **space.status()})

    async def recall_from_space(params: FunctionCallParams):
        out = await space.recall((params.arguments.get("query") or "").strip(), k=5)
        await params.result_callback({"success": True, **out})

    async def drop_from_space(params: FunctionCallParams):
        dropped = await space.drop_find(params.arguments.get("what"), params.arguments.get("reason"))
        await params.result_callback({"success": bool(dropped), "dropped": dropped} if dropped
                                     else {"success": False, "hint": "nothing like that in your space"})

    return {"search_web": search_web, "read_page": read_page, "look_at_image": look_at_image,
            "save_to_space": save_to_space, "hand_over": hand_over, "space_status": space_status,
            "recall_from_space": recall_from_space, "drop_from_space": drop_from_space}


def register_web_tools(llm, web, space, *, describe_image, summarize, space_dir, open_file,
                       limits=None, log=print):
    handlers = make_web_handlers(web, space, describe_image=describe_image, summarize=summarize,
                                 space_dir=space_dir, open_file=open_file, limits=limits, log=log)
    for name, handler in handlers.items():
        llm.register_function(name, _timed(name, handler))
    return handlers
