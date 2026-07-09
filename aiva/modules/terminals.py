"""Aiva's workbench: real, user-visible PowerShell terminals plus headless
Claude Code runs.

A "session" is a genuine Windows Terminal + PowerShell window the user can see
AND type in themselves. Aiva runs commands by focusing that window and typing
into it like a person (via computer.send_command_to_window), and reads results
back from a PowerShell transcript the window writes (flushes in ~0.2s). So the
user watches — and controls — the same terminal Aiva uses.

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
CAPTURE_QUIET = 0.7      # seconds of no new transcript output => command done
WM_CLOSE = 0x0010


class TerminalManager:
    def __init__(self, announce=None):
        # name -> {cwd, history, created, title, log_path, startup}
        self.sessions = {}
        self.jobs = {}       # name -> headless Claude/background job
        self._announce = announce
        self._log_dir = os.path.join(tempfile.gettempdir(), "aiva_terminals")
        os.makedirs(self._log_dir, exist_ok=True)

    # --- visible terminal windows ------------------------------------------

    def _title(self, name):
        return f"Aiva: {name}"

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
            subprocess.Popen(["wt", "new-tab", "--title", self._title(name),
                              "powershell", "-NoExit", "-NoProfile", "-File", startup])
        except OSError:  # Windows Terminal not installed — plain PowerShell console
            subprocess.Popen(f'start powershell -NoExit -NoProfile -File "{startup}"',
                             shell=True)
        return startup

    def _window_open(self, name):
        title = self._title(name).lower()
        try:
            return any(title in t.lower() for _, t in computer._visible_windows())
        except Exception:
            return False

    def _transcript_len(self, log_path):
        try:
            with open(log_path, encoding="utf-8-sig", errors="replace") as f:
                return len(f.read())
        except OSError:
            return 0

    def _read_transcript(self, log_path):
        try:
            with open(log_path, encoding="utf-8-sig", errors="replace") as f:
                return f.read()
        except OSError:
            return ""

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

    async def _capture_delta(self, log_path, before):
        """Wait for the transcript to grow past `before` and settle, then return
        the new (cleaned) output. Times out for long/interactive commands."""
        last = before
        stable = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < CAPTURE_TIMEOUT:
            await asyncio.sleep(0.25)
            cur = self._transcript_len(log_path)
            if cur > last:
                last = cur
                stable = time.monotonic()
            elif stable and cur > before and (time.monotonic() - stable) >= CAPTURE_QUIET:
                break
        full = self._read_transcript(log_path)
        delta = full[before:] if len(full) > before else ""
        return self._clean_output(delta)

    # --- sessions ----------------------------------------------------------

    def open_session(self, name, directory):
        directory = os.path.expanduser(directory or "~")
        if not os.path.isdir(directory):
            return {"success": False, "error": f"directory doesn't exist: {directory}"}
        log_path = os.path.join(self._log_dir, f"session-{name}.log")
        try:  # start each window with a fresh transcript
            open(log_path, "w").close()
        except OSError:
            pass
        self._launch_window(name, directory, log_path)
        self.sessions[name] = {"cwd": directory, "history": [], "created": time.time(),
                               "title": self._title(name), "log_path": log_path}
        return {"success": True, "terminal": name, "cwd": directory,
                "note": "opened a real PowerShell window the user can see and type in too"}

    def _resolve(self, name):
        """Map a possibly-wrong name to a real session so a command never runs
        outside a visible terminal: exact match, else the sole session, else a
        freshly opened one."""
        if name in self.sessions:
            return name
        if len(self.sessions) == 1:
            return next(iter(self.sessions))
        self.open_session(name, "~")
        return name

    def close_session(self, name):
        s = self.sessions.pop(name, None)
        if s:
            try:
                for hwnd, t in computer._visible_windows():
                    if s["title"].lower() in t.lower():
                        ctypes.windll.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            except Exception:
                pass
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
        text = self._clean_output(self._read_transcript(s["log_path"]))
        return {"success": True, "terminal": name, "cwd": s["cwd"],
                "recent_output": text[-OUTPUT_TAIL:]}

    # --- commands -----------------------------------------------------------

    async def run(self, name, command):
        """Type a command into the session's visible terminal and read back its
        output from the transcript. The user sees it run in that window."""
        name = self._resolve(name)
        s = self.sessions[name]

        if not self._window_open(name):  # user closed it — reopen
            self._launch_window(name, s["cwd"], s["log_path"])
            await asyncio.sleep(1.3)

        before = self._transcript_len(s["log_path"])
        ok = await asyncio.to_thread(computer.send_command_to_window, s["title"], command)
        if not ok:
            return {"success": False,
                    "error": "couldn't focus the terminal window to type the command"}

        output = await self._capture_delta(s["log_path"], before)
        s["history"].append((command, output))
        return {"success": True, "output": output[-OUTPUT_TAIL:] or
                "(command sent to the terminal; still running or no output — read it again shortly)"}

    async def type_in(self, name, text, submit=True):
        """Type raw keystrokes into the visible terminal WITHOUT waiting to
        capture output — for interactive programs (ssh sessions, REPLs, prompts,
        passwords) whose output PowerShell's transcript can't see. The user
        reads the result on screen; Aiva reads it with look_at_screen.

        Deliberately does not echo `text` back (it may be a password)."""
        name = self._resolve(name)
        s = self.sessions[name]
        if not self._window_open(name):
            self._launch_window(name, s["cwd"], s["log_path"])
            await asyncio.sleep(1.3)
        ok = await asyncio.to_thread(
            computer.send_command_to_window, s["title"], text, 0.35, submit)
        if not ok:
            return {"success": False, "error": "couldn't focus the terminal window to type"}
        return {"success": True,
                "note": "typed into the terminal — use look_at_screen to read what it shows"}

    def _track_job(self, job_name, session, cmd, proc, announce_done, claude=False):
        job = {"proc": proc, "cmd": cmd, "session": session, "status": "running",
               "result": "", "claude_session_id": None, "cost": None, "started": time.time(),
               "log_path": os.path.join(self._log_dir, f"{job_name}.log")}
        self.jobs[job_name] = job
        asyncio.create_task(self._watch_job(job_name, announce_done, claude))
        return job

    async def _watch_job(self, job_name, announce_done, claude):
        job = self.jobs[job_name]
        out, _ = await job["proc"].communicate()
        text = out.decode("utf-8", errors="replace")
        job["status"] = "done" if job["proc"].returncode == 0 else f"failed ({job['proc'].returncode})"
        job["result"] = text

        if claude:
            try:
                payload = json.loads(text.strip().splitlines()[-1])
                job["claude_session_id"] = payload.get("session_id")
                job["cost"] = payload.get("total_cost_usd")
                job["result"] = payload.get("result", text)
            except (json.JSONDecodeError, IndexError):
                pass

        with open(job["log_path"], "w", encoding="utf-8") as f:
            f.write(job["result"])

        if announce_done and self._announce:
            mins = (time.time() - job["started"]) / 60
            summary = job["result"][-1200:]
            kind = "Claude Code" if claude else "background command"
            await self._announce(
                f"[{kind} job '{job_name}' finished after {mins:.1f} min, status: {job['status']}]\n"
                f"Output (tail):\n{summary}\n"
                f"[Briefly tell the user it's done and summarize the outcome in one or two spoken sentences.]"
            )

    def _job_output(self, name):
        j = self.jobs.get(name)
        if not j:
            return {"success": False, "error": f"no job named '{name}'"}
        return {"success": True, "job": name, "status": j["status"],
                "claude_session_id": j["claude_session_id"], "cost_usd": j["cost"],
                "output_tail": (j["result"] or "(no output yet)")[-OUTPUT_TAIL:]}

    # --- Claude Code ---------------------------------------------------------

    async def start_claude(self, job_name, project_dir, prompt, mode="plan", resume_id=None):
        project_dir = os.path.expanduser(project_dir)
        if not os.path.isdir(project_dir):
            return {"success": False, "error": f"project dir doesn't exist: {project_dir}"}
        if job_name in self.jobs and self.jobs[job_name]["status"] == "running":
            return {"success": False, "error": f"job '{job_name}' is already running"}

        args = ["claude", "-p", "--output-format", "json"]
        if mode == "plan":
            args += ["--permission-mode", "plan"]
        else:
            args += ["--permission-mode", "acceptEdits"]
        if resume_id:
            args += ["--resume", resume_id]

        proc = await asyncio.create_subprocess_exec(
            *args, cwd=project_dir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()

        self._track_job(job_name, None, f"claude[{mode}]: {prompt[:60]}", proc,
                        announce_done=True, claude=True)
        return {"success": True, "status": "started", "job": job_name, "mode": mode,
                "note": "Claude Code is working — this takes minutes; you'll be told when it's done"}

    def shutdown(self):
        for j in self.jobs.values():
            if j["status"] == "running":
                try:
                    j["proc"].kill()
                except ProcessLookupError:
                    pass
