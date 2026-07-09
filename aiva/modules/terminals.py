"""Aiva's workbench: named terminal sessions, background jobs, and headless
Claude Code runs.

Sessions are named working contexts (cwd + command history). Short commands
run inline; anything slow becomes a background job whose completion is
announced out loud via the `announce` callback (wired to the voice pipeline
in main.py). Claude Code runs through its headless print mode with JSON
output, so every run yields a resumable session_id for the plan -> refine ->
execute loop.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time

FOREGROUND_TIMEOUT = 15  # seconds before a command is converted to a job
OUTPUT_TAIL = 2500       # chars of output surfaced to the LLM


class TerminalManager:
    def __init__(self, announce=None):
        self.sessions = {}   # name -> {cwd, history: [(cmd, tail)], created}
        self.jobs = {}       # name -> {proc, cmd, session, log_path, status, result, claude_session_id, cost}
        self._announce = announce
        self._log_dir = os.path.join(tempfile.gettempdir(), "aiva_terminals")
        os.makedirs(self._log_dir, exist_ok=True)

    # --- sessions ----------------------------------------------------------

    def open_session(self, name, directory):
        directory = os.path.expanduser(directory or "~")
        if not os.path.isdir(directory):
            return {"success": False, "error": f"directory doesn't exist: {directory}"}
        # Each session keeps a live transcript log. run() appends every command
        # and its output here; show_terminal tails it, so the window the user
        # watches actually mirrors what Aiva runs in this session.
        log_path = os.path.join(self._log_dir, f"session-{name}.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"# Aiva terminal '{name}'  ({directory})\n"
                        f"# live transcript of what Aiva runs here\n\n")
        except OSError:
            pass
        self.sessions[name] = {"cwd": directory, "history": [], "created": time.time(),
                               "log_path": log_path}
        return {"success": True, "terminal": name, "cwd": directory}

    def _append_session_log(self, name, command, output):
        """Mirror a command + its output into the session's transcript log."""
        s = self.sessions.get(name)
        if not s or not s.get("log_path"):
            return
        try:
            with open(s["log_path"], "a", encoding="utf-8") as f:
                f.write(f"PS {s['cwd']}> {command}\n{output.rstrip()}\n\n")
        except OSError:
            pass

    def close_session(self, name):
        self.sessions.pop(name, None)
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
        return {"success": True, "terminal": name, "cwd": s["cwd"],
                "recent": [{"cmd": c, "output": o[-800:]} for c, o in s["history"][-3:]]}

    # --- commands -----------------------------------------------------------

    async def run(self, name, command):
        """Run in the named session; auto-converts to a job if slow.

        Forgiving about the name so a command never silently runs outside a
        visible terminal: an unknown name falls back to the only open session,
        or opens a fresh one — it never errors back (which used to make Aiva
        retry via the invisible run_command path)."""
        s = self.sessions.get(name)
        if not s:
            if len(self.sessions) == 1:
                name = next(iter(self.sessions))
            else:
                self.open_session(name, "~")
            s = self.sessions[name]

        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", command,
            cwd=s["cwd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=FOREGROUND_TIMEOUT)
            text = out.decode("utf-8", errors="replace")
            s["history"].append((command, text))
            self._append_session_log(name, command, text)
            return {"success": proc.returncode == 0, "returncode": proc.returncode,
                    "output": text[-OUTPUT_TAIL:]}
        except asyncio.TimeoutError:
            job_name = f"{name}-{len(self.jobs) + 1}"
            self._track_job(job_name, name, command, proc, announce_done=True)
            s["history"].append((command, "(still running as background job)"))
            self._append_session_log(
                name, command, f"(long-running — moved to background job '{job_name}')")
            return {"success": True, "status": "running_in_background", "job": job_name,
                    "note": "taking a while — moved to a background job; you'll be told when it finishes"}

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
