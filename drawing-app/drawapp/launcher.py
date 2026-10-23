"""Start the drawing window from code, with no console, and know when it takes commands. No Qt.

    from drawapp.launcher import launch
    app = launch(author="aiva")        # returns once the window is up and listening
    with app.client() as draw:
        draw.stroke([(100, 100), (400, 300)], author="aiva")
    app.close()                        # or leave it; closing the window ends it too

The child is `pythonw -m drawapp --listen 0 --ready-file <tmp>/ready.json --no-activate`
started with CREATE_NO_WINDOW: a GUI-subsystem interpreter never gets a console,
so no terminal flashes up, and the window shows without taking the keyboard
focus, so whatever the user is typing into keeps it. `--listen 0` takes any free port, so a second window
or a leftover one never collides; the app writes {"pid", "port"} to the ready
file once it is listening and shown. Its output goes to <tmp>/drawapp.log.

A venv's python(w).exe is a small redirector that runs the real interpreter as
its child, so `process.pid` is not the app: the ready file's `pid` is, and
close() ends that one.

A window outlives the process that launched it (a sister restarts, the window
stays up). `find_running(author)` finds that window again through the
`<author>-window.json` its session publishes (storage.py) and checks it over
the wire, so the caller adopts it instead of opening a second, blank one.
Each window saves to `save_dir` (default storage.default_save_dir()) and starts
from that author's autosave, so a window that was closed comes back with its drawing.

A canvas several processes draw on (Aiva's and Saga's: `launch(canvas="shared",
author="user")`) is named by `canvas`, not by who draws: `find_running("shared")`
finds it from either process, and every command still names its own author.
`opening(canvas)` holds a lock across find-then-launch, so two processes opening
the same canvas at once get one window: the second waits, then finds the first's.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass

from drawapp.client import DrawClient, DrawError
from drawapp.storage import RECOVERY, _try_lock, _unlock, default_save_dir, window_file

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LaunchError(RuntimeError):
    """The drawing app didn't come up; the message carries the end of its log."""


def write_ready(path: str, port: int, **extra) -> None:
    """Called by the app once it listens; atomic, so a reader never sees half a file.
    `extra` rides along (the app adds "restored": how many strokes it brought back)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "port": port, **extra}, f)
    os.replace(tmp, path)


def read_ready(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            ready = json.load(f)
    except (OSError, ValueError):
        return None
    if isinstance(ready, dict) and isinstance(ready.get("pid"), int) and isinstance(ready.get("port"), int):
        return ready
    return None


def project_python() -> str | None:
    """The repo venv's interpreter (../.venv, the one with PyQt5), or None if there isn't one."""
    path = os.path.join(os.path.dirname(PROJECT_DIR), ".venv", "Scripts", "python.exe")
    return path if os.path.exists(path) else None


def gui_python(python: str | None = None) -> str:
    """`python` (default: this interpreter), swapped for its pythonw.exe sibling if there is one."""
    python = python or sys.executable
    folder, name = os.path.split(python)
    if name.lower() == "python.exe":
        windowed = os.path.join(folder, "pythonw.exe")
        if os.path.exists(windowed):
            return windowed
    return python


def launch_argv(python: str, ready_file: str, port: int = 0, author: str = "user",
                width: int | None = None, height: int | None = None,
                activate: bool = False, save_dir: str | None = None, *,
                canvas: str | None = None, absorb: tuple[str, ...] = ()) -> list[str]:
    """The child's command line. Pure, so tests can check it."""
    argv = [python, "-m", "drawapp", "--listen", str(port), "--ready-file", ready_file,
            "--author", author]
    if canvas is not None and canvas != author:
        argv += ["--canvas", canvas]
    for name in absorb:
        argv += ["--absorb", name]
    if not activate:
        argv.append("--no-activate")
    if save_dir is not None:
        argv += ["--save-dir", save_dir]
    if width is not None:
        argv += ["--width", str(width)]
    if height is not None:
        argv += ["--height", str(height)]
    return argv


def pid_alive(pid: int) -> bool:
    """Is process `pid` still running? Never signals it (on Windows os.kill(pid, 0) would end it)."""
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5       # access denied: it exists, it isn't ours
        try:
            code = ctypes.c_ulong()
            k32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
            return code.value == 259                  # STILL_ACTIVE
        finally:
            k32.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class DrawApp:
    """A launched drawing window, or (process None) one found running by find_running()."""
    process: subprocess.Popen | None  # with a venv interpreter, the redirector (see module doc)
    pid: int                          # the app itself
    port: int
    ready_file: str | None
    log_path: str | None
    restored: int = 0                 # strokes it brought back from its autosave at start

    @property
    def adopted(self) -> bool:
        return self.process is None

    def client(self, timeout: float = 5.0) -> DrawClient:
        return DrawClient(self.port, timeout=timeout)

    def running(self) -> bool:
        if self.process is None:
            return pid_alive(self.pid)
        return self.process.poll() is None

    def close(self, timeout: float = 5.0) -> None:
        """End the app (the user may already have closed its window). Leaves the log.
        Its autosave is current after every finished stroke, so ending it loses nothing finished."""
        if self.process is None:
            if pid_alive(self.pid):
                try:
                    os.kill(self.pid, signal.SIGTERM)  # TerminateProcess on Windows
                except OSError:
                    pass
            deadline = time.monotonic() + timeout
            while pid_alive(self.pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            return
        if self.process.poll() is None and self.pid != self.process.pid:
            try:
                os.kill(self.pid, signal.SIGTERM)  # TerminateProcess on Windows
            except OSError:
                pass
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout)
        try:
            os.remove(self.ready_file)
        except OSError:
            pass


def launch(*, author: str = "user", port: int = 0, width: int | None = None,
           height: int | None = None, python: str | None = None,
           extra_env: dict[str, str] | None = None, timeout: float = 20.0,
           run_dir: str | None = None, activate: bool = False,
           save_dir: str | None = None, canvas: str | None = None,
           absorb: tuple[str, ...] = ()) -> DrawApp:
    """Start a listening drawing window without a console; LaunchError if it doesn't come up.

    `port=0` (default) takes any free port; a fixed port that is taken fails
    fast with the app's own message. `extra_env` is added to this process's
    environment (e.g. {"QT_QPA_PLATFORM": "offscreen"} for a windowless check).
    The ready file and log go in `run_dir` (default: a new drawapp-* folder under %TEMP%).
    The window doesn't take the keyboard focus unless `activate` is true.
    Drawings save to `save_dir` (default: storage.default_save_dir()); the window
    starts from that author's autosave there, and `restored` says how many strokes came back.
    `canvas` makes it a canvas shared by name (its slot is `canvas`'s, the mouse draws as
    `author`); `absorb` names old canvases it takes in at start (storage.Session).
    """
    run_dir = run_dir or tempfile.mkdtemp(prefix="drawapp-")
    ready_file = os.path.join(run_dir, "ready.json")
    log_path = os.path.join(run_dir, "drawapp.log")
    argv = launch_argv(gui_python(python), ready_file, port, author, width, height, activate,
                       save_dir, canvas=canvas, absorb=tuple(absorb))
    env = {**os.environ, **(extra_env or {})}
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            argv, cwd=PROJECT_DIR, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    deadline = time.monotonic() + timeout
    while (ready := read_ready(ready_file)) is None:
        if process.poll() is not None:
            raise LaunchError(f"drawapp exited with code {process.returncode}: {_tail(log_path)}")
        if time.monotonic() > deadline:
            process.kill()
            process.wait(5)
            raise LaunchError(f"drawapp wasn't listening after {timeout:g}s: {_tail(log_path)}")
        time.sleep(0.05)
    restored = ready.get("restored")
    return DrawApp(process, ready["pid"], ready["port"], ready_file, log_path,
                   restored if isinstance(restored, int) else 0)


def find_running(canvas: str = "user", save_dir: str | None = None,
                 timeout: float = 2.0) -> DrawApp | None:
    """The live window that owns `canvas` in `save_dir`, or None. `canvas` is the window's
    --canvas ("shared" for the sisters' one), or its author for a window that has its own.

    Trusts nothing on disk alone: the window file's pid must be alive, and a
    "status" over its port must answer with the same pid and canvas (its "author",
    the session's slot; a stale file, a reused pid or someone else's port all come back None).
    """
    path = window_file(os.path.abspath(save_dir or default_save_dir()), canvas)
    try:
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(info, dict):
        return None
    pid, port = info.get("pid"), info.get("port")
    if not (isinstance(pid, int) and isinstance(port, int) and 0 < port < 65536):
        return None
    if not pid_alive(pid):
        return None
    try:
        with DrawClient(port, timeout=timeout) as draw:
            status = draw.status()
    except DrawError:
        return None
    if status.get("pid") != pid or status.get("author") != canvas:
        return None
    return DrawApp(None, pid, port, None, None)


@contextmanager
def opening(canvas: str = "user", save_dir: str | None = None, timeout: float = 30.0):
    """Hold `recovery/<canvas>.opening.lock` (an OS lock, gone with its process) around a
    find_running-then-launch, so a second process opening the same canvas waits and then finds
    the first one's window instead of launching its own. LaunchError after `timeout`."""
    folder = os.path.join(os.path.abspath(save_dir or default_save_dir()), RECOVERY)
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError:  # no folder, no saving at all: the window itself will say so
        yield
        return
    path = os.path.join(folder, f"{canvas}.opening.lock")
    deadline = time.monotonic() + timeout
    while (held := _try_lock(path)) is None:
        if time.monotonic() > deadline:
            raise LaunchError(f"another process has been opening the {canvas} canvas for "
                              f"over {timeout:g}s")
        time.sleep(0.05)
    try:
        yield
    finally:
        _unlock(held)


def _tail(path: str, chars: int = 600) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
    except OSError:
        return f"(no log at {path})"
    return (text[-chars:] or "(empty log)") + f" [log: {path}]"
