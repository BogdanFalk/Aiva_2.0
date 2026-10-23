"""Aiva's workbench: real, user-visible PowerShell terminals plus headless
Claude Code runs.

A "session" is a genuine Windows Terminal + PowerShell window the user can see
AND type in themselves. Aiva runs commands by focusing that window and typing
into it like a person (via computer.send_keys_to_hwnd). Each command is wrapped so
its output is teed to a per-run file and ends with a unique done-marker, and the
result is read back from that file. This is deliberately NOT PowerShell's
Start-Transcript: on Windows PowerShell 5.1 that buffers small output in memory and
often never flushes it to disk while the window is alive, so normal commands read
back empty. Tee-Object still shows the output live in the window (the user watches
and controls the same terminal) but closes/flushes the file when the command ends.

Claude Code still runs headless through its print mode with JSON output (a
background "job"), announced out loud on completion via the `announce`
callback wired in main.py.
"""

import asyncio
import ctypes
import json
import os
import subprocess
import tempfile
import time
from ctypes import wintypes

from modules import computer

OUTPUT_TAIL = 2500       # chars of output surfaced to the LLM
CAPTURE_TIMEOUT = 20     # max seconds to wait for a typed command's output
CAPTURE_TAIL_BYTES = 262144   # only the tail of a capture is read (marker + surfaced text live there)
WM_CLOSE = 0x0010
JOBS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "coding")
# Claude Code's permission mode for a job that may change code. Headless (`-p`) nobody can answer a
# prompt, so in acceptEdits every git/pytest/listing command was simply denied (14 in one job on
# 2026-10-14) and the job ended asking him for approvals. "auto" lets Claude Code's own safety
# classifier approve routine commands and block risky ones, with no prompts. Deliberately no
# bypassPermissions: the classifier is what stands between a model-written command and his machine.
EXECUTE_MODES = ("auto", "acceptEdits")


CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")   # claude --effort


def execute_permission_mode():
    mode = os.getenv("AIVA_CLAUDE_PERMISSION_MODE", "auto").strip()
    return mode if mode in EXECUTE_MODES else "auto"


def resolve_dir(directory):
    """A directory the way her file tools mean it: "~", absolute, or RELATIVE TO HIS HOME
    ("desktop/foo" is ~/Desktop/foo). Resolving against the process cwd once put a folder
    create_file had just made out of reach of open_terminal."""
    directory = os.path.expanduser(str(directory or "~").strip() or "~")
    if not os.path.isabs(directory):
        directory = os.path.join(os.path.expanduser("~"), directory)
    return os.path.normpath(directory)


def is_whole_home(directory):
    """"." / "~" / "" all resolve to his home — never a project (a job once ran across all of it)."""
    home = os.path.normcase(os.path.normpath(os.path.expanduser("~")))
    return os.path.normcase(resolve_dir(directory)) == home


class TerminalManager:
    def __init__(self, announce=None, name="Aiva"):
        # name -> {cwd, history, created, title, log_path, startup}
        self.sessions = {}
        self.jobs = {}       # name -> headless Claude/background job
        self._active = None  # the terminal most recently opened/used
        self._announce = announce
        self.name = name     # the character whose windows these are ("Saga: main")
        self._log_dir = os.path.join(tempfile.gettempdir(), "aiva_terminals")
        os.makedirs(self._log_dir, exist_ok=True)
        # job results are the record of her coding work (the dashboard's Tasks tab reads them), so
        # they live with the other logs, not in %TEMP%
        self.jobs_dir = JOBS_DIR

    # --- visible terminal windows ------------------------------------------

    def _title(self, name):
        return f"{self.name}: {name}"

    def _launch_window(self, name, directory, log_path):
        """Open a real Windows Terminal + PowerShell window at `directory` that
        logs its whole session to `log_path`. Startup goes in a .ps1 file, not
        on the wt command line — wt treats ';' as its own tab separator and
        would chop a multi-statement -Command."""
        startup = os.path.join(self._log_dir, f"startup-{name}.ps1")
        with open(startup, "w", encoding="utf-8") as f:
            f.write(f"$host.UI.RawUI.WindowTitle = '{self._title(name)}'\n"
                    f"Set-Location -LiteralPath '{directory}'\n"
                    f"Start-Transcript -Path '{log_path}' -Append | Out-Null\n"
                    f"Clear-Host\n")
        try:
            # -w new forces its OWN window (not a tab in a shared one) so its
            # handle is unambiguous
            subprocess.Popen(["wt", "-w", "new", "new-tab", "--title", self._title(name),
                              "powershell", "-NoExit", "-NoProfile", "-File", startup])
        except OSError:  # Windows Terminal not installed — plain PowerShell console
            subprocess.Popen(f'start powershell -NoExit -NoProfile -File "{startup}"',
                             shell=True)
        return startup

    async def _ensure_window(self, name):
        """Return a LIVE window handle for the session, launching the window if
        needed and capturing its handle while the title is still ours (before
        ssh/REPLs rename it). Tracked by handle thereafter, so a retitled window
        is never mistaken for a closed one — the bug that spawned a new terminal
        on every post-ssh command."""
        s = self.sessions[name]
        if computer.window_alive(s.get("hwnd")):
            return s["hwnd"]
        # no live handle yet: adopt an existing window with our title, else launch
        if not computer.hwnd_for_title(self._title(name)):
            self._launch_window(name, s["cwd"], s["log_path"])
        for _ in range(20):
            await asyncio.sleep(0.3)
            hwnd = computer.hwnd_for_title(self._title(name))
            if hwnd:
                s["hwnd"] = hwnd
                return hwnd
        s["hwnd"] = None
        return None

    def _read_capture(self, path, tail_bytes=CAPTURE_TAIL_BYTES):
        """Read a per-command capture file, tolerant of the encoding Tee-Object writes (UTF-16LE with
        a BOM on PowerShell 5.1, UTF-8 elsewhere). Only the tail is read for big outputs — the
        done-marker and the bit we surface both live at the end."""
        try:
            with open(path, "rb") as f:
                head = f.read(3)
                if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
                    full_enc, width = "utf-16", 2
                elif head[:3] == b"\xef\xbb\xbf":
                    full_enc, width = "utf-8-sig", 1
                else:
                    full_enc, width = "utf-8", 1
                f.seek(0, os.SEEK_END)
                size = f.tell()
                start = 0
                if tail_bytes and size > tail_bytes:
                    start = size - tail_bytes
                    if width == 2 and start % 2:      # keep UTF-16 code units aligned
                        start += 1
                f.seek(start)
                raw = f.read()
        except OSError:
            return ""
        # a tail slice has no BOM, so decode it BOM-free; a full read uses the detected encoding
        dec = full_enc if start == 0 else ("utf-16-le" if width == 2 else "utf-8")
        return raw.decode(dec, "replace")

    def _clean_output(self, text):
        """Strip PowerShell prompt/echo lines and Start-Transcript banners from
        a transcript slice, leaving just command output."""
        keep = []
        for ln in text.splitlines():
            s = ln.strip()
            if s.startswith("PS ") and s.endswith(">"):
                continue  # bare prompt
            if s.startswith("PS ") and ">" in s:
                continue  # prompt + echoed command
            if s.startswith("**********") or s.startswith("Transcript "):
                continue  # Start-Transcript banner lines
            keep.append(ln)
        return "\n".join(keep).strip()

    def _finish_capture(self, text, marker):
        """Drop the done-marker line and clean the captured output."""
        return self._clean_output(text.replace(marker, "")) if marker else self._clean_output(text)

    async def _await_capture(self, out_path, marker):
        """Poll the capture file until the done-marker appears (the command finished AND its output
        flushed to disk), or we time out. Returns (cleaned_output, done)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < CAPTURE_TIMEOUT:
            await asyncio.sleep(0.2)
            text = self._read_capture(out_path)
            if marker and marker in text:
                return self._finish_capture(text, marker), True
        return self._finish_capture(self._read_capture(out_path), marker), False

    # --- sessions ----------------------------------------------------------

    def open_session(self, name, directory):
        directory = resolve_dir(directory)
        if not os.path.isdir(directory):
            return {"success": False, "error": f"directory doesn't exist: {directory}"}
        log_path = os.path.join(self._log_dir, f"session-{name}.log")
        try:  # start each window with a fresh transcript
            open(log_path, "w").close()
        except OSError:
            pass
        # the window is launched (and its handle captured) lazily by
        # _ensure_window, so open + first-use never double-launch
        self.sessions[name] = {"cwd": directory, "history": [], "created": time.time(),
                               "title": self._title(name), "log_path": log_path, "hwnd": None}
        self._active = name
        return {"success": True, "terminal": name, "cwd": directory,
                "note": "opened a real PowerShell window the user can see and type in too"}

    def _resolve(self, name):
        """Map a name to an EXISTING session — run_in_terminal / type_in_terminal
        act on a terminal that's already open, so this never spawns a new window
        when one exists. Prefer the exact match, then the terminal she's actively
        using (handles 'the same terminal' / a mis-remembered name), then any
        open one. Only open a fresh terminal if there are none at all."""
        if name in self.sessions:
            resolved = name
        elif self._active and self._active in self.sessions:
            resolved = self._active
        elif self.sessions:
            resolved = next(reversed(self.sessions))  # most recently opened
        else:
            self.open_session(name, "~")
            resolved = name
        self._active = resolved
        return resolved

    def close_session(self, name):
        s = self.sessions.pop(name, None)
        if s:
            try:
                hwnd = s.get("hwnd")
                if computer.window_alive(hwnd):  # by handle — title may have changed (ssh)
                    ctypes.windll.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
                else:  # fall back to title match if we never captured a handle
                    for h, t in computer._visible_windows():
                        if s["title"].lower() in t.lower():
                            ctypes.windll.user32.PostMessageW(h, WM_CLOSE, 0, 0)
            except Exception:
                pass
        if self._active == name:  # don't leave _active pointing at a dead session
            self._active = next(reversed(self.sessions)) if self.sessions else None
        job = self.jobs.get(name)
        if job and job["status"] == "running":
            job["proc"].kill()
            job["status"] = "killed"
        return {"success": True, "closed": name}

    def list_sessions(self):
        out = []
        for name, s in self.sessions.items():
            last = s["history"][-1][0] if s["history"] else None
            out.append({"terminal": name, "cwd": s["cwd"], "last_command": last})
        for name, j in self.jobs.items():
            out.append({"job": name, "status": j["status"], "command": j["cmd"][:80]})
        return out

    def read_session(self, name):
        if name in self.jobs:
            return self._job_output(name)
        s = self.sessions.get(name)
        if not s:
            return {"success": False, "error": f"no terminal named '{name}'"}
        out_path = s.get("last_out")
        if not out_path:
            return {"success": True, "terminal": name, "cwd": s["cwd"],
                    "recent_output": "(nothing run in this terminal yet)"}
        raw = self._read_capture(out_path)
        marker = s.get("last_marker", "")
        done = (marker in raw) if marker else True
        text = self._finish_capture(raw, marker)
        note = "" if done else "\n(still running — read again shortly)"
        return {"success": True, "terminal": name, "cwd": s["cwd"],
                "recent_output": (text[-OUTPUT_TAIL:] or "(no output captured)") + note}

    # --- commands -----------------------------------------------------------

    def _wrap(self, command, out_path, marker):
        """Wrap the command so ALL its output is teed to `out_path` and it ends with `marker`.
        Dot-sourced (`. { }`, not `& { }`) so cd / $vars persist to the session; `2>&1` so errors
        are captured too; Tee-Object shows it live in the window and flushes the file when it ends."""
        return ". { " + command + " ; '" + marker + "' } 2>&1 | Tee-Object -FilePath '" + out_path + "'"

    async def run(self, name, command):
        """Type a command into the session's visible terminal and read back its output. Deterministic:
        the command is teed to a per-run file ending in a unique marker (no dependence on
        Start-Transcript, which doesn't flush small output in time). The user sees it run in the window."""
        name = self._resolve(name)
        s = self.sessions[name]
        hwnd = await self._ensure_window(name)
        if not hwnd:
            return {"success": False, "error": "couldn't open or find the terminal window"}

        s["cmd_n"] = s.get("cmd_n", 0) + 1
        n = s["cmd_n"]
        out_path = os.path.join(self._log_dir, f"cmd-{name}-{n}.out")
        marker = f"AIVADONE{n}X{int(time.time() * 1000) % 1000000:06d}"
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass
        ok = await asyncio.to_thread(computer.send_keys_to_hwnd, hwnd,
                                     self._wrap(command, out_path, marker), 0.35, True)
        if not ok:
            return {"success": False,
                    "error": "couldn't focus the terminal window to type the command"}
        s["last_out"] = out_path
        s["last_marker"] = marker

        output, done = await self._await_capture(out_path, marker)
        s["history"].append((command, output))
        tail = output[-OUTPUT_TAIL:]
        if done:
            return {"success": True, "output": tail or "(command finished with no output)"}
        return {"success": True, "still_running": True,
                "output": (tail + "\n(still running — read the terminal again shortly)").strip()
                if tail else "(command sent; still running — read the terminal again shortly)"}

    async def type_in(self, name, text, submit=True):
        """Type raw keystrokes into the visible terminal WITHOUT waiting to
        capture output — for interactive programs (ssh sessions, REPLs, prompts,
        passwords) whose output PowerShell's transcript can't see. The user
        reads the result on screen; Aiva reads it with look_at_screen.

        Deliberately does not echo `text` back (it may be a password)."""
        name = self._resolve(name)
        hwnd = await self._ensure_window(name)
        if not hwnd:
            return {"success": False, "error": "couldn't open or find the terminal window"}
        ok = await asyncio.to_thread(computer.send_keys_to_hwnd, hwnd, text, 0.35, submit)
        if not ok:
            return {"success": False, "error": "couldn't focus the terminal window to type"}
        return {"success": True,
                "note": "typed into the terminal — use look_at_screen to read what it shows"}

    def _track_job(self, job_name, session, cmd, proc, announce_done, claude=False):
        job = {"proc": proc, "cmd": cmd, "session": session, "status": "running", "claude": bool(claude),
               "result": "", "claude_session_id": None, "cost": None, "denied": [], "started": time.time(),
               "log_path": os.path.join(self.jobs_dir, f"{job_name}.log"),
               "done_event": asyncio.Event()}
        self.jobs[job_name] = job
        asyncio.create_task(self._watch_job(job_name, announce_done, claude))
        return job

    async def _watch_job(self, job_name, announce_done, claude):
        job = self.jobs[job_name]
        out, _ = await job["proc"].communicate()
        text = out.decode("utf-8", errors="replace")
        if job["status"] != "killed":  # kill_job already labelled it
            job["status"] = "done" if job["proc"].returncode == 0 else f"failed ({job['proc'].returncode})"
        job["result"] = text

        payload = None
        if claude:
            try:
                payload = json.loads(text.strip().splitlines()[-1])
                job["claude_session_id"] = payload.get("session_id")
                job["cost"] = payload.get("total_cost_usd")
                job["result"] = payload.get("result", text)
                # what it wanted to do but wasn't allowed (headless: nobody to approve the prompt)
                for d in payload.get("permission_denials") or []:
                    inp = d.get("tool_input") or {}
                    what = inp.get("command") or inp.get("file_path") or ""
                    job["denied"].append(f"{d.get('tool_name')}: {str(what)[:120]}".strip(": "))
            except (json.JSONDecodeError, IndexError):
                pass

        try:
            os.makedirs(os.path.dirname(job["log_path"]), exist_ok=True)
            with open(job["log_path"], "w", encoding="utf-8") as f:
                f.write(job["result"])
            if isinstance(payload, dict):  # session_id, cost, turns, duration — the raw record
                with open(job["log_path"][:-4] + ".json", "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=1)
        except OSError:
            pass  # a log we can't write must never stall the waiting task
        job["done_event"].set()  # wait_job() callers (the task runner) wake up here

        if announce_done and self._announce:
            mins = (time.time() - job["started"]) / 60
            summary = job["result"][-1200:]
            kind = "Claude Code" if claude else "background command"
            await self._announce(
                f"[{kind} job '{job_name}' finished after {mins:.1f} min, status: {job['status']}]\n"
                f"Output (tail):\n{summary}\n"
                f"[Briefly tell the user it's done and summarize the outcome in one or two spoken sentences.]"
            )

    async def wait_job(self, name, timeout=None):
        """Await a job's completion. Returns {status, result, cost, claude_session_id}
        or None if it isn't known / didn't finish within `timeout` seconds."""
        job = self.jobs.get(name)
        if not job:
            return None
        try:
            await asyncio.wait_for(job["done_event"].wait(), timeout)
        except asyncio.TimeoutError:
            return None
        return {"status": job["status"], "result": job["result"], "cost": job["cost"],
                "claude_session_id": job["claude_session_id"], "log_path": job.get("log_path"),
                "denied": job.get("denied") or []}

    def kill_job(self, name):
        """Kill a running job (task cancelled / timed out). True if something died."""
        job = self.jobs.get(name)
        if not job or job["status"] != "running":
            return False
        job["status"] = "killed"
        try:
            job["proc"].kill()
        except ProcessLookupError:
            pass
        return True

    def _job_output(self, name):
        j = self.jobs.get(name)
        if not j:
            return {"success": False, "error": f"no job named '{name}'"}
        return {"success": True, "job": name, "status": j["status"],
                "claude_session_id": j["claude_session_id"], "cost_usd": j["cost"],
                "output_tail": (j["result"] or "(no output yet)")[-OUTPUT_TAIL:]}

    # --- Claude Code ---------------------------------------------------------

    async def start_claude(self, job_name, project_dir, prompt, mode="plan", resume_id=None,
                           announce_done=True, effort=None):
        if is_whole_home(project_dir):
            return {"success": False, "error": "won't run Claude Code across his whole home folder; "
                                               "give the project's own folder"}
        project_dir = resolve_dir(project_dir)
        if not os.path.isdir(project_dir):
            return {"success": False, "error": f"project dir doesn't exist: {project_dir}"}
        if job_name in self.jobs and self.jobs[job_name]["status"] == "running":
            return {"success": False, "error": f"job '{job_name}' is already running"}

        args = ["claude", "-p", "--output-format", "json"]
        if mode == "plan":
            args += ["--permission-mode", "plan"]
        else:
            args += ["--permission-mode", execute_permission_mode()]
        if resume_id:
            args += ["--resume", resume_id]
        if effort in CLAUDE_EFFORTS:
            # how hard this one job thinks (tasks.job_effort); None leaves his own Claude Code setting
            args += ["--effort", effort]

        proc = await asyncio.create_subprocess_exec(
            *args, cwd=project_dir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()

        self._track_job(job_name, None, f"claude[{mode}]: {prompt[:60]}", proc,
                        announce_done=announce_done, claude=True)
        return {"success": True, "status": "started", "job": job_name, "mode": mode,
                "note": "Claude Code is working — this takes minutes; you'll be told when it's done"}

    def shutdown(self):
        for j in self.jobs.values():
            if j["status"] == "running":
                try:
                    j["proc"].kill()
                except ProcessLookupError:
                    pass
