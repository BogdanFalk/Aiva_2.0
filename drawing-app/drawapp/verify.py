"""Clean-launch check: start the window the way a sister would, draw on it, close it. No Qt.

    python -m drawapp.verify                      # real window; exit 0 if every check passed
    python -m drawapp.verify --keep-open          # leave the drawing up to look at
    python -m drawapp.verify --offscreen          # no window at all (CI, nobody at the screen)
    pythonw -m drawapp.verify --report out.json   # no terminal anywhere; read the result from the file

It goes through launcher.launch() and client.DrawClient only, so it checks the
integration path itself: the child comes up with no console (on Windows: no
conhost/OpenConsole under it), its window is on screen and names its port
without taking the keyboard focus from whatever had it, commands draw and a bad one is refused,
the drawing saves (.aivadraw + .png, in a temp save folder of its own), and close() really ends it. The report
is JSON (stdout, and --report if given); nothing waits for a key or a prompt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time

from drawapp.client import CommandRejected, DrawError
from drawapp.launcher import DrawApp, LaunchError, launch

CONSOLE_HOSTS = {"conhost.exe", "openconsole.exe"}


class Report:
    def __init__(self):
        self.checks: list[dict] = []
        self.info: dict = {}

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c["ok"] for c in self.checks)

    def as_dict(self) -> dict:
        return {"ok": self.ok, **self.info, "checks": self.checks}


def run(offscreen: bool = False, keep_open: bool = False, timeout: float = 20.0,
        python: str | None = None) -> Report:
    report = Report()
    started = time.monotonic()
    # its own save folder: a check never restores, autosaves over or adopts a sister's real canvas
    save_dir = tempfile.mkdtemp(prefix="drawapp-verify-")
    report.info["save_dir"] = save_dir
    try:
        app = launch(author="aiva", extra_env={"QT_QPA_PLATFORM": "offscreen"} if offscreen else None,
                     timeout=timeout, python=python, save_dir=save_dir)
    except LaunchError as e:
        report.check("launch", False, str(e))
        return report
    report.check("launch", True, f"listening on 127.0.0.1:{app.port} after "
                                 f"{time.monotonic() - started:.1f}s")
    report.info.update(pid=app.pid, port=app.port, log=app.log_path, interpreter=app.process.args[0])
    try:
        _check_console_and_window(report, app, offscreen)
        _check_commands(report, app, clear=not keep_open)
    finally:
        if keep_open:
            report.info["left_open"] = True
        else:
            app.close()
            report.check("close", not app.running() and not _pid_alive(app.pid),
                         f"exit code {app.process.returncode}")
    return report


def _check_console_and_window(report: Report, app: DrawApp, offscreen: bool) -> None:
    if sys.platform != "win32":
        return
    hosts = [name for pid, parent, name in _processes()
             if parent in (app.pid, app.process.pid) and name.lower() in CONSOLE_HOSTS]
    report.check("no console", not hosts, ", ".join(hosts) or "no console host under the app")
    if offscreen:
        return
    deadline = time.monotonic() + 5  # shown before the ready file, but mapping is asynchronous
    while not (titles := _visible_window_titles(app.pid)) and time.monotonic() < deadline:
        time.sleep(0.1)
    report.check("window shown", any(f"127.0.0.1:{app.port}" in t for t in titles),
                 "; ".join(titles) or "no visible window")
    focused = _foreground_pid()  # whoever was typing somewhere else still is
    report.check("focus kept", focused != app.pid,
                 f"foreground window belongs to pid {focused}" if focused else "no foreground window")


def _check_commands(report: Report, app: DrawApp, clear: bool) -> None:
    try:
        _draw(report, app, clear)
    except DrawError as e:  # unreachable, dropped or timed out: a failed check, not a crash
        report.check("commands", False, f"{type(e).__name__}: {e}")


def _draw(report: Report, app: DrawApp, clear: bool) -> None:
    with app.client() as draw:
        wave = [(112 + 8 * i, 300 + 60 * math.sin(i / 6)) for i in range(101)]
        aiva = draw.stroke(wave, author="aiva", width=6)
        report.check("aiva stroke", aiva["stroke_id"].startswith("aiva-"), json.dumps(aiva))
        ring = [(512 + 120 * math.cos(t / 30 * math.pi), 470 + 120 * math.sin(t / 30 * math.pi))
                for t in range(61)]
        saga = draw.stroke(ring, author="saga", width=6)
        report.check("saga stroke", saga["stroke_id"].startswith("saga-")
                     and saga["revision"] > aiva["revision"], json.dumps(saga))
        try:
            draw.stroke([(-50, 10), (20, 20)], author="aiva")
            report.check("bad command refused", False, "an off-canvas stroke was accepted")
        except CommandRejected as e:
            report.check("bad command refused", e.code == "invalid", str(e))
        saved = draw.save(author="aiva", name="verify")
        report.check("save", os.path.isfile(saved["path"]) and bool(saved["png"])
                     and os.path.isfile(saved["png"]), json.dumps(saved))
        if clear:  # --keep-open leaves the drawing up to look at
            cleared = draw.clear(author="saga")
            report.check("clear", cleared["revision"] == saga["revision"] + 1, json.dumps(cleared))


def _processes() -> list[tuple[int, int, str]]:
    """(pid, parent pid, exe name) for every process (Windows Toolhelp snapshot)."""
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W))
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if snap in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_last_error(), "CreateToolhelp32Snapshot failed")
    out = []
    try:
        entry = PROCESSENTRY32W(dwSize=ctypes.sizeof(PROCESSENTRY32W))
        more = k32.Process32FirstW(snap, ctypes.byref(entry))
        while more:
            out.append((entry.th32ProcessID, entry.th32ParentProcessID, entry.szExeFile))
            more = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return out


def _visible_window_titles(pid: int) -> list[str]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    titles = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _lparam):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, buf, 512)
            if buf.value:
                titles.append(buf.value)
        return True

    user32.EnumWindows(visit, 0)
    return titles


def _foreground_pid() -> int:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.restype = wintypes.HWND
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return 0
    owner = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
    return owner.value


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        return any(p == pid for p, _parent, _name in _processes())
    import os
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drawapp.verify", description=__doc__.splitlines()[0])
    parser.add_argument("--offscreen", action="store_true", help="no window (QT_QPA_PLATFORM=offscreen)")
    parser.add_argument("--keep-open", action="store_true", help="leave the drawing window open")
    parser.add_argument("--report", metavar="JSON", help="also write the report to this file")
    parser.add_argument("--timeout", type=float, default=20.0, help="seconds to wait for the window")
    parser.add_argument("--python", help="interpreter with PyQt5 for the app (default: this one)")
    args = parser.parse_args(argv)
    try:
        report = run(args.offscreen, args.keep_open, args.timeout, args.python)
    except Exception as e:  # a crash is a failed check, and still a report
        report = Report()
        report.check("verify", False, f"{type(e).__name__}: {e}")
    text = json.dumps(report.as_dict(), indent=2)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)  # a no-op under pythonw, where sys.stdout is None
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
