"""Aiva's task runner: multi-step jobs that run in the background.

The voice LLM hands a goal to `TaskRunner.start()` (via the start_task tool)
and goes back to being conversational. The runner plans (one JSON call),
decides how much autonomy the user's rule allows (task_memory.decide_autonomy),
then works step by step: act with the SAME tool handlers the voice LLM uses
(ToolBox adapter), observe, verify, repair, and report back through short
system notes — the same `announce()` path Claude Code jobs already use.

Hard rules baked in here:
- Ordered execution lives inside this module, never as N parallel tool calls
  (pipecat's run_in_parallel must stay True).
- Shell commands NEVER run without the user's yes: a `shell` step pauses the
  task and waits for answer_task. Same for handing code to Claude Code on a
  path the user didn't already approve.
- Every state transition is persisted (task_memory `tasks` row) so a restart
  can report what was interrupted.
- Nothing the runner "thinks" reaches the voice context; only the notes.
- All task_memory calls happen on the event-loop thread (single-thread sqlite).
"""

import asyncio
import copy
import glob
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime

from modules import task_memory as tm

from modules.persona import named
from modules.utilities import LLM_RATES, llm_kwargs

RUNNER_TOOL_ALLOW = frozenset({
    "launch_app", "create_file", "open_file", "list_directory",
    "look_at_screen", "focus_window", "click_ui_element", "type_text", "press_keys",
    "click_at", "scroll",
    "open_terminal", "run_in_terminal", "type_in_terminal", "read_terminal",
    "list_terminals", "show_terminal", "get_datetime",
    # web reach + her space: read-only / reversible (download to her own space), so a "go find me X"
    # task can run autonomously. Buy/sign-in/send have no tool at all, so they can't happen here.
    "search_web", "read_page", "look_at_image", "save_to_space", "hand_over",
    "recall_from_space", "space_status",
    # what she has seen her sister do — read-only, her own observations
    "recall_about_sister",
    # her drawing canvas, opened and drawn on from her own process (modules/drawing.py). A task that
    # draws or checks the drawing app uses these, never a terminal (2026-10-20: he saw the PowerShell)
    "open_drawing", "draw_stroke", "clear_drawing", "undo_stroke", "save_drawing",
    "open_saved_drawing",
})

DESTRUCTIVE_RE = re.compile(
    r"(Remove-Item[^|;]*-Recurse|\brm\s+-r|\brmdir\s+/s|\bdel\s+/s|\bformat\s+[a-z]:|\bdiskpart\b|"
    r"\bshutdown\b|\breg\s+delete|git\s+push\s+[^|;]*--force|git\s+reset\s+--hard|"
    r"Stop-Computer|Restart-Computer|\bmkfs\b)", re.I)

# a delegate step that only looks: it runs in Claude Code's plan mode (no edits) and is graded on its
# report. The planner sets action.read_only; this catches a step with NO flag whose wording rules out
# every change — never a partial one ("without changing any implementation files" while writing a note)
READ_ONLY_RE = re.compile(r"\b(do not|don'?t|without) (modify|modifying|change|changing|edit|editing|touch|"
                          r"touching) (anything|any files?|any code|files|the files|code|the code|the repo\w*|"
                          r"the project)\b|\bread[- ]only\b|\bno (edits|changes)\b", re.I)


def _flag(v):
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def delegate_read_only(action):
    """Plan mode or not for a delegate step. The planner's own read_only flag decides whenever it set
    one: 2026-10-14 a retry the planner marked read_only false — he'd said "retry outside plan mode
    and write the note" — still ran in plan mode because its prompt said "without changing any
    implementation files", and the note was refused. Only a step with no flag is read from its words."""
    action = action or {}
    if action.get("read_only") is not None:
        return _flag(action["read_only"])
    return bool(READ_ONLY_RE.search(action.get("prompt") or ""))


# How hard a Claude Code job thinks (`claude --effort`). Every job used to inherit his own xhigh: a
# read-only README review spent 18k of its 31k output tokens thinking, ~5 min and ~$2 (2026-10-17).
# The planner sizes each job; a job that came back short on the same project goes one level up.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def job_effort(asked, read_only, misses=0, forced=None):
    """PURE: the --effort for one Claude Code job, or None to leave his own Claude Code setting.
    `asked` is the planner's (or the agent's) level; missing or unknown → medium for a look, high
    for a change. A look never goes past high, a change past xhigh (max only when forced). Each
    earlier job on this project in this task that came back partial or bad (`misses`) is one level
    up: it wasn't enough the first time. `forced` (AIVA_CLAUDE_EFFORT) is auto, inherit, or a level."""
    forced = (forced or "auto").strip().lower()
    if forced == "inherit":
        return None
    if forced in EFFORT_LEVELS:
        return forced
    level = (asked or "").strip().lower()
    if level not in EFFORT_LEVELS[:4]:
        level = "medium" if read_only else "high"
    top = EFFORT_LEVELS.index("high" if read_only else "xhigh")
    return EFFORT_LEVELS[min(EFFORT_LEVELS.index(level) + max(0, int(misses or 0)), top)]
# her own code (and her sister's, the house's): the planner is told, so work on herself gets this
# folder and not "." — which resolves to his home (2026-10-14: a job ran across all of C:\Users\Extre)
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def is_whole_home(project_dir):
    from modules.terminals import is_whole_home as whole  # lazy: terminals pulls in computer/ctypes
    return whole(project_dir)


# files her own running processes write while a job runs — never the agent's doing
WORKTREE_NOISE_RE = re.compile(r"(^|/)(logs|backups|__pycache__)/|\.(log|db|db-wal|db-shm|jsonl|pyc)$|"
                               r"(^|/)[^/]*state\.json$|\.restart_marker$", re.I)

YES_WORDS = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok|okay|go|go ahead|do it|please do|"
                       r"of course|absolutely|fine|alright|correct|right|confirmed|proceed)\b", re.I)
NO_WORDS = re.compile(r"^\W*(no|nope|nah|don'?t|do not|stop|wait|not now|cancel|never mind|"
                      r"hold on|abort|forget it)\b", re.I)


class TaskCancelled(Exception):
    pass


class BudgetExceeded(Exception):
    pass


@dataclass
class Limits:
    max_steps: int = 12
    max_tool_calls: int = 80
    max_llm_calls: int = 90
    max_wall_secs: float = 20 * 60
    step_tool_calls: int = 12          # moves per step (a move may make several look calls); 6 ran out mid-look
    tool_timeout: float = 60
    claude_wait_secs: float = 30 * 60
    answer_reminder_secs: float = 180
    max_cost_usd: float = 1.50
    milestone_min_gap_secs: float = 30
    plan_wait_secs: float = 12
    max_repairs_per_step: int = 1
    max_repairs_per_task: int = 2
    max_rethinks: int = 1
    test_timeout: float = 300

    # env key -> (field, cast, scale): the dashboard's Settings tab edits these in .env; minutes are
    # friendlier than seconds for the long ones. Read once at boot (TaskRunner(limits=Limits.from_env())).
    ENV_KEYS = {
        "AIVA_TASK_MAX_STEPS": ("max_steps", int, 1),
        "AIVA_TASK_MAX_TOOL_CALLS": ("max_tool_calls", int, 1),
        "AIVA_TASK_MAX_LLM_CALLS": ("max_llm_calls", int, 1),
        "AIVA_TASK_MAX_WALL_MIN": ("max_wall_secs", float, 60),
        "AIVA_TASK_STEP_TOOL_CALLS": ("step_tool_calls", int, 1),
        "AIVA_TASK_TOOL_TIMEOUT_S": ("tool_timeout", float, 1),
        "AIVA_TASK_CLAUDE_WAIT_MIN": ("claude_wait_secs", float, 60),
        "AIVA_TASK_ANSWER_REMINDER_S": ("answer_reminder_secs", float, 1),
        "AIVA_TASK_MAX_COST_USD": ("max_cost_usd", float, 1),
        "AIVA_TASK_MILESTONE_GAP_S": ("milestone_min_gap_secs", float, 1),
        "AIVA_TASK_PLAN_WAIT_S": ("plan_wait_secs", float, 1),
        "AIVA_TASK_MAX_REPAIRS_PER_STEP": ("max_repairs_per_step", int, 1),
        "AIVA_TASK_MAX_REPAIRS_PER_TASK": ("max_repairs_per_task", int, 1),
        "AIVA_TASK_MAX_RETHINKS": ("max_rethinks", int, 1),
        "AIVA_TASK_TEST_TIMEOUT_S": ("test_timeout", float, 1),
    }

    @classmethod
    def from_env(cls, env=None):
        """The defaults above, each overridable by its `AIVA_TASK_*` key; a blank or unparsable
        value keeps the default (a typo in .env must never stop her booting)."""
        env = os.environ if env is None else env
        kw = {}
        for key, (field_name, cast, scale) in cls.ENV_KEYS.items():
            raw = (env.get(key) or "").strip()
            if not raw:
                continue
            try:
                kw[field_name] = cast(float(raw) * scale) if cast is int else float(raw) * scale
            except ValueError:
                continue
        return cls(**kw)


@dataclass
class LLMReply:
    text: str | None
    tool_calls: list = field(default_factory=list)  # [{"id","name","arguments"(dict),"arguments_json"}]
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def assistant_message(self):
        msg = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc.get("arguments_json")
                              or json.dumps(tc.get("arguments", {}), ensure_ascii=False)}}
                for tc in self.tool_calls]
        return msg


def openai_llm_call(client, model, label="llm"):
    """LLMCall over a real AsyncOpenAI client. Signature shared with test fakes:
    call(messages, *, tools=None, json_mode=False, max_tokens=900, temperature=0.2).
    `label` names the side-call in the metrics profiler (e.g. 'task', 'reading', 'self')."""
    import time as _t

    from modules import metrics
    from modules.utilities import LLM_RATES

    async def call(messages, *, tools=None, json_mode=False, max_tokens=900, temperature=0.2):
        kwargs = dict(model=model, messages=messages,
                      **llm_kwargs(model, max_tokens=max_tokens, temperature=temperature))
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = _t.perf_counter()
        resp = await client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {"_raw": raw}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args, "arguments_json": raw})
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        rin, rout = LLM_RATES.get(model, LLM_RATES.get("gpt-5.6-luna", (0.2, 1.2)))
        metrics.record("llm", label, (_t.perf_counter() - t0) * 1000.0,
                       tokens=pt + ct, cost=(pt * rin + ct * rout) / 1e6)
        return LLMReply(msg.content, calls, pt, ct)

    return call


# --- tool adapter ------------------------------------------------------------------

def schema_to_openai(fs):
    """pipecat FunctionSchema -> OpenAI tool dict (no adapter internals needed)."""
    return {"type": "function", "function": {
        "name": fs.name, "description": fs.description,
        "parameters": {"type": "object", "properties": dict(fs.properties or {}),
                       "required": list(fs.required or [])}}}


def _tool(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


PRIVATE_TOOLS = [
    _tool("shell", "Run a PowerShell command invisibly (Windows PowerShell 5.1: no `&&`, use `;`). The USER must confirm it first — the "
                   "task pauses until they answer, so use it only to CHANGE things no other tool can "
                   "(install, move, delete); never to read, list, search, check git, run tests or write "
                   "a file — read_file / find_files / search_text / repo_status / run_tests / create_file "
                   "do that with no wait. Use run_in_terminal (a visible window) only when the user asked to see a "
                   "terminal or the task is about one.",
          {"command": {"type": "string"}, "purpose": {"type": "string",
           "description": "One plain sentence: what this command is for"}}, ["command", "purpose"]),
    _tool("delegate_to_claude", "Hand coding work (multi-file edits, bug fixes, features, tests) in a "
                                "project directory to Claude Code. Takes minutes; the result is graded. "
                                "Reading or searching a few files yourself is faster.",
          {"project_dir": {"type": "string"}, "task": {"type": "string"},
           "why": {"type": "string", "description": "Why this needs an agent rather than you"},
           "effort": {"type": "string", "enum": ["low", "medium", "high", "xhigh"],
                      "description": "How hard it should think: low = confirm one thing, medium = a "
                                     "review or small edit, high = a real change, xhigh = build from scratch"}},
          ["project_dir", "task", "why"]),
    # looking needs no yes: these only read (2026-10-14: with nothing but the shell to read a file,
    # search or run the tests, every look was a PowerShell command that waited on him — a text-file
    # task ran out of moves asking, a verification became nine approvals)
    _tool("read_file", "Read a text file (no confirmation needed). Lines come numbered.",
          {"path": {"type": "string"}, "start_line": {"type": "integer"},
           "max_lines": {"type": "integer", "description": "default 200, at most 400"}}, ["path"]),
    _tool("find_files", "Find files under a folder by name pattern, e.g. '*.py' or '*constellation*' "
                        "(no confirmation needed; skips .git, virtualenvs and caches).",
          {"directory": {"type": "string"}, "pattern": {"type": "string"}}, ["directory", "pattern"]),
    _tool("search_text", "Search the text files under a folder for a word or phrase, case-insensitive "
                         "(no confirmation needed). Returns path:line: text.",
          {"directory": {"type": "string"}, "text": {"type": "string"},
           "file_pattern": {"type": "string", "description": "e.g. '*.py'; default every text file"}},
          ["directory", "text"]),
    _tool("repo_status", "A git project's state: changed files, the last commits, the diff size "
                         "(no confirmation needed).", {"project_dir": {"type": "string"}}, ["project_dir"]),
    _tool("run_tests", "Run a project's own test suite (pytest from its own .venv), or just the given "
                       "test files (no confirmation needed). This is how to test — never pytest via the shell.",
          {"project_dir": {"type": "string"},
           "tests": {"type": "array", "items": {"type": "string"},
                     "description": "test files inside the project, e.g. tests/test_x.py; empty = all"}},
          ["project_dir"]),
    _tool("ask_user", "Ask the user one short question when you genuinely can't proceed without "
                      "their answer. The task pauses until they reply.",
          {"question": {"type": "string"}}, ["question"]),
    _tool("wait", "Pause a few seconds for something to load or settle (max 30).",
          {"seconds": {"type": "number"}}, ["seconds"]),
    _tool("step_done", "Declare this step complete, with what you observed as evidence.",
          {"summary": {"type": "string"}, "evidence": {"type": "string"}}, ["summary", "evidence"]),
    _tool("step_failed", "Declare this step impossible or failed, with the reason and a suggestion.",
          {"reason": {"type": "string"}, "suggestion": {"type": "string"}}, ["reason"]),
]
PRIVATE_TOOL_NAMES = {t["function"]["name"] for t in PRIVATE_TOOLS}
LOOK_TOOLS = frozenset({"read_file", "find_files", "search_text", "repo_status", "run_tests"})


# --- git and the project's own tests (how she checks a coding agent's work) -------------------

def run_git(project_dir, *args, timeout=30):
    """git's WHOLE output. Not execute_shell: that keeps the last 1000 characters, and with his
    weeks of uncommitted work (~2,500 characters of status) the worktree snapshot only ever saw the
    tail — every tracked file an agent edited went unseen (2026-10-13: a job that fixed the renderer
    and passed 837 tests was graded "nothing in the project changed")."""
    try:
        done = subprocess.run(["git", "-C", project_dir, *args], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout, stdin=subprocess.DEVNULL,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"success": False, "returncode": None, "stdout": "", "stderr": str(e)}
    return {"success": done.returncode == 0, "returncode": done.returncode,
            "stdout": done.stdout or "", "stderr": done.stderr or ""}


def project_test_command(project_dir):
    """The project's own test command, or None — never one a model wrote. Only pytest, and only
    through the project's own virtualenv with pytest installed in it: another interpreter would fail
    on missing packages, and that would read as broken tests."""
    has_tests = (any(os.path.isdir(os.path.join(project_dir, d)) for d in ("tests", "test"))
                 or any(os.path.isfile(os.path.join(project_dir, f)) for f in ("pytest.ini", "conftest.py")))
    if not has_tests:
        return None
    for venv in (".venv", "venv"):
        root = os.path.join(project_dir, venv)
        if not (glob.glob(os.path.join(root, "Lib", "site-packages", "pytest"))
                or glob.glob(os.path.join(root, "lib", "python*", "site-packages", "pytest"))):
            continue
        for exe in (os.path.join(root, "Scripts", "python.exe"), os.path.join(root, "bin", "python")):
            if os.path.isfile(exe):
                return [exe, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    return None


PYTEST_TALLY_RE = re.compile(r"\b\d+ (passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b")


def tests_result(returncode, output):
    """pytest's exit code + output -> the evidence her grader reads. Exit 5 is "no tests collected":
    nothing ran, which is not a failure."""
    if returncode == 5:
        return {"ran": False, "why": "no tests collected"}
    lines = [ln.strip() for ln in (output or "").splitlines() if ln.strip()]
    tally = next((ln.strip("= ") for ln in reversed(lines) if PYTEST_TALLY_RE.search(ln)), "")
    return {"ran": True, "returncode": returncode, "passed": returncode == 0,
            "summary": tally or f"exit code {returncode}",
            "failing": [ln for ln in lines if ln.startswith(("FAILED ", "ERROR "))][:8]}


def run_test_command(command, project_dir, timeout):
    """Run the project's tests (blocking; callers use asyncio.to_thread)."""
    try:
        done = subprocess.run(command, cwd=project_dir, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, stdin=subprocess.DEVNULL,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return {"ran": False, "why": f"the tests ran longer than {timeout:.0f} s"}
    except OSError as e:
        return {"ran": False, "why": f"couldn't run the tests: {e}"}
    return tests_result(done.returncode, (done.stdout or "") + "\n" + (done.stderr or ""))


# --- looking without asking: the task agent's read-only tools ---------------------------------
# Pure Python (git and the tests excepted, both fixed commands), fenced to his home like her file
# tools. Blocking — the runner calls them through asyncio.to_thread.

# logs/backups too: a code search drowned in old log lines (searching the logs folder itself still works)
SKIP_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
                       ".pytest_cache", "site-packages", ".tox", "dist", "build", "logs", "backups"})
MAX_WALK_FILES = 40000           # a walk this big is the wrong folder, not a search
MAX_READ_BYTES = 2_000_000
# never opened by a text search: 2026-10-14 her first search of the repo took two minutes reading
# ~1 GB of wake-word training audio (20,000 .wav files) only to find each one binary
BINARY_EXTS = frozenset({
    ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wem", ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".psd", ".moc3", ".pdf",
    ".onnx", ".tflite", ".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".pkl", ".bin", ".h5",
    ".parquet", ".arrow", ".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm", ".pyc", ".pyd",
    ".zip", ".7z", ".rar", ".gz", ".tar", ".exe", ".dll", ".so", ".lib", ".obj", ".lnk",
    ".woff", ".woff2", ".ttf", ".otf"})


def _home_path(path):
    from modules.file_operations import resolve_path
    return resolve_path(str(path or "").strip() or "~")


def _walk_files(root):
    seen = 0
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for f in sorted(files):
            seen += 1
            if seen > MAX_WALK_FILES:
                return
            yield os.path.join(d, f)


def _read_text(path):
    if os.path.getsize(path) > MAX_READ_BYTES:
        return None
    with open(path, "rb") as fh:
        head = fh.read(8192)
        if b"\x00" in head:
            return None                               # binary: don't read the rest
        data = head + fh.read()
    return data.decode("utf-8", errors="replace")


def read_file_lines(path, start_line=1, max_lines=200):
    try:
        path = _home_path(path)
        if not os.path.isfile(path):
            return {"success": False, "error": f"no such file: {path}"}
        text = _read_text(path)
    except (ValueError, OSError) as e:
        return {"success": False, "error": str(e)}
    if text is None:
        return {"success": False, "error": "not a text file (binary or over 2 MB)"}
    lines = text.splitlines()
    start = max(1, int(start_line or 1))
    count = max(1, min(400, int(max_lines or 200)))
    chunk = lines[start - 1:start - 1 + count]
    return {"success": True, "path": path, "total_lines": len(lines),
            "lines": "\n".join(f"{start + i}: {ln}" for i, ln in enumerate(chunk)),
            "more": start - 1 + count < len(lines)}


def find_files(directory, pattern, limit=100):
    import fnmatch
    try:
        root = _home_path(directory)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    if not os.path.isdir(root):
        return {"success": False, "error": f"no such folder: {root}"}
    pat = (pattern or "*").strip()
    hits = []
    for p in _walk_files(root):
        rel = os.path.relpath(p, root)
        if fnmatch.fnmatch(os.path.basename(p).lower(), pat.lower()) or fnmatch.fnmatch(rel.lower(), pat.lower()):
            hits.append(rel)
            if len(hits) >= limit:
                break
    return {"success": True, "root": root, "files": hits, "truncated": len(hits) >= limit}


def search_text(directory, text, file_pattern="*", limit=60):
    import fnmatch
    needle = (text or "").strip().lower()
    if not needle:
        return {"success": False, "error": "nothing to search for"}
    try:
        root = _home_path(directory)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    if not os.path.isdir(root):
        return {"success": False, "error": f"no such folder: {root}"}
    pat = (file_pattern or "*").strip().lower() or "*"
    hits = []
    for p in _walk_files(root):
        if not fnmatch.fnmatch(os.path.basename(p).lower(), pat) or \
                os.path.splitext(p)[1].lower() in BINARY_EXTS:
            continue
        try:
            body = _read_text(p)
        except OSError:
            continue
        if body is None or needle not in body.lower():
            continue
        rel = os.path.relpath(p, root)
        here = 0
        for i, ln in enumerate(body.splitlines(), 1):
            if needle in ln.lower():
                hits.append(f"{rel}:{i}: {ln.strip()[:200]}")
                here += 1
                if here >= 5:                         # one file can't eat the whole answer
                    break
                if len(hits) >= limit:
                    return {"success": True, "root": root, "matches": hits, "truncated": True}
    return {"success": True, "root": root, "matches": hits, "truncated": False}


def pick_test_targets(project_dir, tests):
    """The test files a model asked for, kept only when each is a real path inside the project
    ('tests/test_x.py' or 'tests/test_x.py::test_name'). Never options: nothing here starts with '-'."""
    root = os.path.normcase(os.path.realpath(project_dir))
    out = []
    for t in tests or []:
        t = str(t or "").strip()
        if not t or t.startswith("-"):
            continue
        file_part, _, node = t.partition("::")
        full = os.path.realpath(os.path.join(project_dir, file_part))
        if not (os.path.normcase(full) + os.sep).startswith(root + os.sep) or not os.path.exists(full):
            continue
        out.append(os.path.relpath(full, project_dir) + (f"::{node}" if node else ""))
    return out


def _same_dir(a, b):
    def norm(p):
        return os.path.normcase(os.path.normpath(os.path.expanduser(str(p or "").strip()))).rstrip("\\/")
    return bool(a) and bool(b) and norm(a) == norm(b)


_RISK_ORDER = ("none", "reversible", "irreversible")


def one_job_per_project(steps):
    """Consecutive delegate steps on the same project become ONE Claude Code job. Split into
    inspect / implement / test / verify, every job started from nothing, re-read the repo and could
    fail its grading on its own (2026-10-13: four jobs for one renderer fix; the inspection's
    findings never reached the job that made the change). Pure; returns a new list."""
    def project(s):
        return (s.get("action") or {}).get("project_dir") if isinstance(s, dict) and s.get("kind") == "delegate" else None

    groups = []
    for s in steps:
        if groups and project(s) and _same_dir(project(groups[-1][0]), project(s)):
            groups[-1].append(s)
        else:
            groups.append([s])
    return [g[0] if len(g) == 1 else _merged_delegate(g) for g in groups]


def _merged_delegate(parts):
    acts = [p.get("action") or {} for p in parts]
    flags = [delegate_read_only(a) for a in acts]
    read_only = all(flags)
    doing = next((p for p, f in zip(parts, flags) if not f), parts[0])
    prompt = ("Do all of this in this one job, in order. Each part builds on the one before; a part that "
              "says only to look, or not to change anything, means only that part.\n\n"
              + "\n\n".join(f"{i}. {p.get('title')}:\n{a.get('prompt')}" for i, (p, a) in enumerate(zip(parts, acts), 1)))
    risks = [p.get("risk") if p.get("risk") in _RISK_ORDER else "none" for p in parts]
    # the one job does every part, so it thinks as hard as its hardest part asked
    efforts = [str(a.get("effort") or "").strip().lower() for a in acts]
    efforts = [e for e in efforts if e in EFFORT_LEVELS]
    return {"title": doing.get("title"), "kind": "delegate",
            "action": {"project_dir": acts[0].get("project_dir"), "prompt": prompt, "read_only": read_only,
                       "effort": max(efforts, key=EFFORT_LEVELS.index) if efforts else None,
                       "merged": True},
            "expect": "; ".join(str(p.get("expect") or p.get("title")) for p in parts),
            "verify": "result", "milestone": False, "risk": max(risks, key=_RISK_ORDER.index),
            "merged_from": [p.get("title") for p in parts]}


class ToolBox:
    """The voice LLM's own tool handlers, callable out-of-band with a fake
    FunctionCallParams. Results are always non-empty dicts."""

    def __init__(self, handlers, schemas=None, allow=RUNNER_TOOL_ALLOW, name="Aiva"):
        self._handlers = dict(handlers)
        self._allow = frozenset(allow)
        self.name = name                 # whose tools these are (schema-less fallback descriptions)
        self._schemas = {}
        if schemas is not None:
            for fs in getattr(schemas, "standard_tools", schemas):
                self._schemas[fs.name] = fs

    def add_handlers(self, mapping):
        """Merge in tool handlers registered after construction (e.g. web/space, whose stores are
        built later in boot). Only allow-listed names ever become callable; this just makes the
        handler reachable. Without this, an allow-listed tool with no handler here silently fails."""
        self._handlers.update(mapping or {})

    @property
    def names(self):
        return sorted(n for n in self._handlers if n in self._allow)

    def openai_tools(self, extra=()):
        tools = []
        for name in self.names:
            fs = self._schemas.get(name)
            if fs is not None:
                tools.append(schema_to_openai(fs))
            else:
                tools.append(_tool(name, f"{self.name} tool {name}", {}, []))
        tools.extend(extra)
        return tools

    def describe(self):
        lines = []
        for name in self.names:
            fs = self._schemas.get(name)
            desc = (fs.description if fs is not None else "").split(". ")[0][:160]
            args = ", ".join((fs.properties or {}).keys()) if fs is not None else ""
            lines.append(f"- {name}({args}): {desc}")
        return "\n".join(lines)

    async def call(self, name, args, *, task_id="t", step_n=0, timeout=60):
        if name not in self._allow or name not in self._handlers:
            return {"success": False, "error": f"tool {name} is not available to tasks"}
        from pipecat.services.llm_service import FunctionCallParams

        box = {}

        async def cb(result, *, properties=None):
            box["result"] = result

        params = FunctionCallParams(
            function_name=name, tool_call_id=f"{task_id}-s{step_n}-{int(time.time() * 1000) % 100000}",
            arguments=dict(args or {}), llm=None, pipeline_worker=None, context=None,
            result_callback=cb)
        try:
            await asyncio.wait_for(self._handlers[name](params), timeout)
        except asyncio.TimeoutError:
            return {"success": False, "error": f"{name} timed out after {timeout:.0f}s"}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return {"success": False, "error": f"{name} raised: {e}"}
        result = box.get("result")
        if not result:
            return {"success": False, "error": f"{name} returned nothing"}
        if not isinstance(result, dict):
            return {"success": True, "result": str(result)[:2000]}
        return result


# --- prompts -----------------------------------------------------------------------

# Templates carry {name} (rendered once per character in TaskRunner) and are .replace()d later for
# {tools}/{families}/{approaches}/{budget} — never .format()ed: they carry literal JSON braces.
PLANNER_PROMPT_T = """You are the planner inside {name}, a voice assistant that controls the user's \
Windows PC. Turn the goal into a short, concrete plan of steps. Reply with JSON only.

Step kinds:
- "tool": exactly one call of a listed tool with fixed args (use when you know the exact args). \
action MUST be {"tool": "<name from the tool list>", "args": {<that tool's arguments>}}. Example step: \
{"n": 1, "title": "Create hello.txt on the desktop", "kind": "tool", "action": {"tool": "create_file", \
"args": {"file_path": "desktop/hello.txt", "content": "hello"}}, "expect": "hello.txt exists on the \
desktop", "verify": "file", "milestone": false, "risk": "reversible"}
- "agent": a small goal for a sub-agent that may use several tools and must look at results \
(use when args depend on what's on screen, or the step needs a look-then-act).
- "shell": an invisible PowerShell command; the user will be asked to confirm it. action: \
{"command", "purpose"}; its verify is "result" (the command's own output is the evidence). Work in \
the BACKGROUND by default: never open a visible terminal (open_terminal/run_in_terminal) unless \
the user asked to see one or the task is about a terminal. Looking never needs a shell step: an \
"agent" step can read files, find and search them, check a git repo and run a project's tests \
(read_file, find_files, search_text, repo_status, run_tests) without asking anyone. A shell step \
stops the task until the user answers, so use one only to change things (delete, move, install) \
that no tool can; writing a file is create_file, which needs no yes and never replaces an existing \
file unless told to. Put the check and the change in ONE command where you can (e.g. test the path, \
then remove it) so the user confirms once. Plan only the steps the goal needs: when he asks for a \
file, writing it IS the task — any looking it needs goes in the same agent step that writes it, \
not a research step in front that can fail on its own.
- "delegate": hand CODING work (multi-file edits, bug fixes, features, tests) in a project \
directory to Claude Code. action: {"project_dir", "prompt", "read_only", "effort"}. A question about \
one or a few files (what does it say, is X set, does Y pass) is NOT a delegate: an "agent" step \
answers it with read_file / search_text / run_tests in seconds, while every Claude Code job costs \
minutes and a fixed ~$0.25 before it even starts. effort is how hard the job thinks: "low" = find \
or confirm one thing in the code; "medium" = a review or inspection across several files, or a \
small contained edit; "high" = a real change across files, a bug that needs understanding, tests \
to fix; "xhigh" = building something new from the ground up, a redesign, or a hard bug that already \
beat a first try. Pick the lowest that will do it well. ONE piece of coding work \
is ONE delegate step: Claude Code looks at the code, makes the change and runs the project's tests \
itself in that one job, so never split it into separate inspect / implement / test / verify delegate \
steps (back-to-back delegate steps on the same project are merged into one job anyway). read_only \
true only when the user asked just to look, review or plan (the agent then cannot edit; its report is \
the result). A later delegate step on the same project continues the same Claude Code session. \
Never for desktop/UI work. \
project_dir MUST be the folder the code lives in (Claude Code can only touch files inside it) — for \
a new standalone script or program use its own new folder, e.g. "desktop/prime_project" (it is \
created if missing), never the assistant's own repo unless the work is on that repo. The brief's \
your_own_code is that repo (your code, your sister's, the house's): use its full path for work on \
any of you. Never "." or "~" — that is his whole home folder, not a project.
- "ask": ask the user one question you cannot proceed without. action: {"question"}.

{name}'s own drawing app (her drawing canvas, drawapp) opens with ONE "tool" step, open_drawing \
(it reuses the window if it's up), is drawn on with draw_stroke and saved with save_drawing, \
never Ctrl+S or press_keys (open_saved_drawing brings a saved one back). It is not a program on the \
PC: never launch_app, a terminal, Aseprite, Paint or any other drawing program for it.

Every step: {"n", "title" (<= 12 words, spoken to the user), "kind", "action", "expect" (plain-words \
success criterion), "verify": "result"|"screen"|"terminal"|"file"|"none", "milestone": bool, \
"risk": "none"|"reversible"|"irreversible"}. Prefer "file"/"result"/"terminal" verification over \
"screen". Mark at most 3 milestones (only for plans longer than 3 steps). 1 to 12 steps. Mark \
a step "irreversible" ONLY when it destroys or changes something that cannot be recovered - deleting, overwriting an EXISTING file, uninstalling, sending a message, paying; creating a new file, opening or focusing apps, reading, or typing into an empty document is "reversible" or "none". If a similar past task or a \
correction is listed, follow what worked and avoid what the user complained about. If \
previous_attempt is given, the first plan for this task already failed: the agent's report in it is \
ground truth about the project, so build on what it found, never on what it showed doesn't exist, \
and don't repeat the failed step as it was.

Also return: "summary" (<= 15 words, names the concrete apps/files/targets), "family" (one of \
{families} or "other:<two words>"), "approach" (one of {approaches}), "trivial" (true if a single \
tool call would do), "est_minutes", "delegate" (true if any step delegates), "same_as_match" \
(false ONLY if the closest listed past task is clearly a different target), "risks" (short \
list), "suggested_flow_name" (2-4 spoken words) and "suggested_triggers" (2 short phrases). \
When two approaches genuinely differ in risk or effort (UI vs terminal vs Claude Code, quick vs \
thorough), also return "alternatives": [{"label", "summary", "tradeoff", "est_minutes", \
"irreversible", "steps": [...]}] (2 items) and "options_distinct": true; otherwise omit them.

Available tools (name(args): purpose):
{tools}"""

RUNNER_SYSTEM_PROMPT_T = """You are the hands of {name}, a voice assistant, carrying out ONE step of a \
larger task on the user's Windows PC. Use the tools to act, then LOOK at results before deciding. \
Prefer click_ui_element by visible name over click_at; use look_at_screen to find things or read \
errors. Never type a password the user hasn't given. Work in the background: do not open or type \
into a visible terminal unless the step is about one or the user asked to see it. LOOK with \
read_file, find_files, search_text, list_directory and repo_status, TEST with run_tests, and WRITE a \
file with create_file — none of them waits for the user, and several can go in one move. Writing the \
file he asked for is already approved: create_file never replaces an existing file unless you pass \
overwrite true (a taken name gets ' (2)' and the result says so), so there is nothing to check first. \
The shell tool stops the whole task until the user answers, so it is only for what no tool here can \
do (delete, move, install); never for reading, searching, writing a file, git or tests. Your own \
drawing canvas is open_drawing, draw_stroke and save_drawing, never launch_app, another drawing program \
or Ctrl+S. Never try a \
destructive command any other way. Use the tools before you give up: step_failed is for a tool that \
really failed or a step none of them can do. A \
look-around is a means, not the goal: once you know enough, finish. Do ONLY this step: never do the work of later \
steps (don't delete while the step says inspect). When the step is achieved, call step_done with what you observed \
as evidence. If it can't be done, call step_failed with the reason. Do not call the same tool with \
the same arguments twice in a row. Be economical: at most {budget} tool calls for this step."""

PLANNER_PROMPT = named(PLANNER_PROMPT_T, "Aiva")
RUNNER_SYSTEM_PROMPT = named(RUNNER_SYSTEM_PROMPT_T, "Aiva")

VERIFIER_PROMPT = """You verify one step of a task done by an assistant on a Windows PC. Given the \
step's success criterion and the observation, reply JSON: {"ok": true|false, "confidence": 0..1, \
"reason": "<one plain sentence>", "repair_hint": "<what to try, or null>"}. Be strict: if the \
observation doesn't show the criterion is met, ok is false. One exception: if the observation \
shows the whole task's goal (task_goal) is ALREADY achieved, so this step has become moot (e.g. \
"inspect the folder before deleting it" when the folder is already gone and deleting it was the \
goal), reply ok true AND "goal_done": true, and the reason says the goal is already done."""

SUMMARY_PROMPT = """Summarize what a task achieved for a voice assistant to say aloud: two plain \
sentences, past tense, naming the concrete results (files, apps, values). No lists, no markdown. \
If something was skipped or partial, say so plainly."""

ANSWER_PROMPT = """The user was asked a question by a voice assistant and replied. Classify the \
reply as JSON: {"decision": "yes"|"no"|"option:<label>"|"other", "reason": "<their why, if any>"}. \
Options, when offered, are listed; match the closest label."""

# {name}: whose second thoughts these are. Shown the agent's report in full — the report may know
# what she got wrong (2026-10-13: twice she sent Claude Code after a "Miso renderer" that never
# existed; it said so, and she never read it).
RETHINK_PROMPT_T = """You are {name}'s second thoughts about one step of her task: a coding agent \
(Claude Code) just did it and it did not go as planned. Read the agent's report carefully: it looked \
at the real project, so where it contradicts what the task assumed, the report is right and the task \
was wrong. Work out what really happened; don't blame the agent for doing what it was asked. Reply \
with JSON only:
{"what_went_wrong": "<one plain spoken sentence, e.g. 'There is no Miso renderer; Miso is the desktop cat.'>",
 "beliefs": [{"fact": "<a lasting, self-contained truth about the user's PC or projects that the \
report establishes>", "evidence": "<5 to 25 words copied VERBATIM from the report that show it>", \
"replaces": ["<a line from what_you_believe, copied EXACTLY, that this fact proves wrong>"]}],
 "lesson": "<one sentence: what to do differently on tasks like this, or null>",
 "next": "retry" | "ask" | "give_up",
 "revised_goal": "<when next is retry: the goal restated with the wrong part corrected, achievable \
with what the report shows exists>",
 "question": "<when next is ask: ONE short spoken question for the user that settles it>"}
retry only when the report itself settles what to do instead. When the right reading is a choice \
only the user can make (what they meant, which of several things, whether to go ahead with something \
big), ask. give_up when it simply can't be done. At most 3 beliefs, only facts the report states \
that will still be true next week (what exists, what something is called, where it lives) — never \
guesses, never the state of this run (git status, what was changed or couldn't run) and nothing \
about the agent itself. Never invent a feature, file or component the report didn't find. \
what_you_believe is what she currently holds true: list in "replaces" only lines the report \
directly contradicts (often the very belief the task was built on); never a line that is merely \
about the same topic."""


# --- the runner --------------------------------------------------------------------

class TaskRunner:
    def __init__(self, *, task_memory, toolbox, llm_call, terminals=None, limits=None,
                 log=print, execute_shell=None, describe_screen=None, list_monitors=None,
                 resolve_path=None, facts=None, model_name="gpt-4.1", name="Aiva", on_done=None,
                 free_range=None, on_outcome=None, git=None, test_command=None, test_runner=None,
                 verify_tests=None, on_result=None):
        self.tm = task_memory
        # free range (AIVA_TASK_FREE_RANGE=1, user decision 2026-09-22): no plan, shell or
        # delegate confirmations — see task_memory.decide_autonomy
        if free_range is None:
            free_range = os.getenv("AIVA_TASK_FREE_RANGE", "0") == "1"
        self.free_range = bool(free_range)
        # after a coding job she runs the project's own tests herself and grades on what they say,
        # not on the agent's word (user decision 2026-10-13). A fixed command from
        # project_test_command, never one a model wrote; AIVA_TASK_RUN_TESTS=0 turns it off.
        if verify_tests is None:
            verify_tests = os.getenv("AIVA_TASK_RUN_TESTS", "1") == "1"
        self.verify_tests = bool(verify_tests)
        self._git = git or run_git
        self._test_command = test_command or project_test_command
        self._test_runner = test_runner or run_test_command
        self.toolbox = toolbox
        self.llm_call = llm_call
        self.name = name
        # on_done(summary, family): fired when a task's outcome is recorded as a full success — the
        # house tells her sister what she got done (kin). Sync or async; never breaks the finish.
        self._on_done = on_done
        # on_outcome(outcome, goal): EVERY finish (success|partial|failed|cancelled) — what it felt
        # like to get it done or not (v3 drives: competence). Sync or async; never breaks the finish.
        self._on_outcome = on_outcome
        # on_result(goal, summary): a task that got done (success|partial), with its WHOLE goal and
        # what came of it — what she wanted is now done, so the wishes it settles can close
        # (SelfEngine.settle_wishes). Sync or async; never breaks the finish.
        self._on_result = on_result
        self.planner_prompt = named(PLANNER_PROMPT_T, name)
        self.runner_prompt = named(RUNNER_SYSTEM_PROMPT_T, name)
        self.rethink_prompt = named(RETHINK_PROMPT_T, name)
        self.private_tools = PRIVATE_TOOLS
        self.terminals = terminals
        self.limits = limits or Limits()
        self.log = log
        self._execute_shell = execute_shell
        self._describe_screen = describe_screen
        self._list_monitors = list_monitors
        self._resolve_path = resolve_path
        self._facts = facts or (lambda: [])
        self.model_name = model_name
        self.tasks = {}                 # id -> live task dict
        self._events = {}               # id -> asyncio.Event (answer arrived)
        self._step_tasks = {}           # id -> asyncio.Task of the running step
        self._run_tasks = {}            # id -> asyncio.Task of the whole task (_run)
        self._planned = {}              # id -> Future(first result for start_task)
        self._announce_cb = None
        self._quiet_cb = None
        self._is_awake = lambda: True
        self._spawn = asyncio.ensure_future
        self._announce_lock = asyncio.Lock()
        self._last_announce = 0.0
        self._last_milestone = 0.0
        self.last_completed_task_id = None
        self.session_task_ids = []

    def attach_voice(self, *, announce, quiet_note=None, is_awake=None, spawn=None):
        self._announce_cb = announce
        self._quiet_cb = quiet_note
        if is_awake:
            self._is_awake = is_awake
        if spawn:
            self._spawn = spawn

    # --- voice-facing entry points --------------------------------------------

    async def start(self, goal, details="", confirm_first=False, *, flow=None, replay="adaptive"):
        goal = (goal or "").strip()
        if not goal:
            return {"success": False, "error": "no goal given"}
        running = [t for t in self.tasks.values() if t["status"] in tm.LIVE_STATUSES]
        if len(running) >= 2:
            return {"success": False, "error": "two tasks are already running; finish or cancel one first",
                    "tasks": [t["goal"] for t in running]}
        task = self._new_task(goal, details, confirm_first, flow, replay)
        self.tasks[task["id"]] = task
        self.session_task_ids.append(task["id"])
        self._events[task["id"]] = asyncio.Event()
        fut = asyncio.get_running_loop().create_future()
        self._planned[task["id"]] = fut
        self.tm.record_episode(task["id"], goal, details, status="planning",
                               flow_id=flow["id"] if flow else None,
                               plan_steps=task["plan"])
        self._log(task, "start", goal)
        self._run_tasks[task["id"]] = self._spawn(self._run(task))
        try:
            first = await asyncio.wait_for(asyncio.shield(fut), self.limits.plan_wait_secs)
            task["_delivered_in_turn"] = True
            return first
        except asyncio.TimeoutError:
            task["_in_turn_missed"] = True
            return {"success": True, "task_id": task["id"], "status": "started",
                    "say": "On it — this one needs a moment to plan.",
                    "note": "Say one short sentence that you're on it. If the plan needs a yes, "
                            "you'll get a note to relay."}

    async def answer(self, answer, task_id=None):
        task = self._find_task(task_id, waiting=True)
        if task is None:
            return {"success": False, "error": "no task is waiting for an answer"}
        pending = task.get("pending") or {}
        kind = pending.get("kind")
        text = (answer or "").strip()
        decision = None
        reason = None
        if YES_WORDS.match(text):
            decision = "yes"
        elif NO_WORDS.match(text):
            decision = "no"
        if kind == "choose_option":
            opts = pending.get("payload", {}).get("options") or []
            picked = self._match_option(text, opts)
            if picked:
                decision, reason = f"option:{picked['label']}", text
            elif decision != "no":
                data = await self._classify_answer(task, text, opts)
                decision, reason = data.get("decision"), data.get("reason")
        elif kind == "clarify":
            decision = "text"
        elif decision is None:
            data = await self._classify_answer(task, text, [])
            decision = data.get("decision")
        if decision in (None, "other"):
            return {"success": False, "error": "couldn't tell if that was a yes or a no — ask again plainly",
                    "task_id": task["id"], "question": pending.get("text")}
        task["_answer"] = {"decision": decision, "text": text, "reason": reason}
        self._events[task["id"]].set()
        self._log(task, "answer", f"{kind}: {decision}")
        understood = decision if not decision.startswith("option:") else decision.split(":", 1)[1]
        return {"success": True, "task_id": task["id"], "understood": understood,
                "note": "Say one short sentence that you're continuing (or that you've stopped)."}

    def status(self, task_id=None):
        tasks = [self.tasks[task_id]] if task_id in self.tasks else \
            sorted(self.tasks.values(), key=lambda t: t["started_ts"], reverse=True)[:3]
        out = []
        for t in tasks:
            done = sum(1 for s in t["plan"] if s.get("status") == "done")
            current = next((s for s in t["plan"] if s.get("status") in ("running", "awaiting_user")), None)
            elapsed = (time.time() - t["_started_mono"]) / 60 if t.get("_started_mono") else 0.0
            out.append({
                "id": t["id"], "goal": t["goal"], "status": t["status"],
                "step": f"{done}/{len(t['plan'])}" if t["plan"] else "planning",
                "current": current.get("title") if current else None,
                "elapsed_min": round(elapsed, 1),
                "pending_question": (t.get("pending") or {}).get("text"),
                "last_event": t["log"][-1]["detail"] if t["log"] else None,
                "result": t.get("result_summary"),
            })
        return {"success": True, "tasks": out,
                "note": "Answer in one sentence." if out else "Nothing is running."}

    def busy_within(self, seconds, now=None):
        """True while a task is live or one of this session's finished in the last `seconds`: the
        stretch counts as work for modules/pacing.py. A pure read of self.tasks."""
        if any(t["status"] in tm.LIVE_STATUSES for t in self.tasks.values()):
            return True
        now = now or datetime.now()
        for t in self.tasks.values():
            try:
                if t.get("finished_ts") and (now - datetime.fromisoformat(t["finished_ts"])).total_seconds() <= seconds:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    _BODY_STATES = {"planning": "planning", "awaiting_user": "waiting", "verifying": "checking"}

    def activity(self):
        """What her body shows while she works (the avatar's work panel), or None when no task
        is live: {label, detail, step, total, state}. A task waiting on him wins over one that's
        merely running — that's the one he needs to see. A pure read of self.tasks."""
        live = [t for t in self.tasks.values() if t["status"] in tm.LIVE_STATUSES]
        if not live:
            return None
        waiting = [t for t in live if t["status"] == "awaiting_user"]
        t = max(waiting or live, key=lambda t: t.get("started_ts") or 0)
        plan = t.get("plan") or []
        current = next((s for s in plan if s.get("status") in ("running", "awaiting_user")), None)
        return {"label": t.get("goal") or "", "detail": (current or {}).get("title") or "",
                "step": sum(1 for s in plan if s.get("status") == "done"), "total": len(plan),
                "state": self._BODY_STATES.get(t["status"], "running")}

    async def cancel(self, task_id=None, reason=""):
        task = self._find_task(task_id, waiting=False)
        if task is None:
            return {"success": False, "error": "no task is running"}
        task["_cancel"] = True
        task["_answer"] = {"decision": "cancel", "text": reason}
        self._events[task["id"]].set()
        step_task = self._step_tasks.get(task["id"])
        if step_task and not step_task.done():
            step_task.cancel()
        job = task.get("_claude_job")
        if job and self.terminals is not None and hasattr(self.terminals, "kill_job"):
            try:
                self.terminals.kill_job(job)
            except Exception:
                pass
        self._log(task, "cancel_requested", reason or "user asked")
        return {"success": True, "cancelled": task["id"], "goal": task["goal"]}

    async def run_flow(self, name, mode="adaptive", overrides=""):
        flow = self.tm.get_flow(name) or self.tm.find_flow(name)
        if not flow:
            fuzzy = await self.tm.find_flow_fuzzy(name)
            flow = fuzzy[0] if fuzzy else None
        if not flow:
            known = [f["name"] for f in self.tm.flows()]
            return {"success": False, "error": f"no flow called '{name}'", "flows": known}
        goal = flow["name"] + (f" ({overrides})" if overrides else "")
        return await self.start(goal, overrides, False, flow=flow, replay=mode)

    def recover_orphans(self):
        return self.tm.reconcile_stale()

    async def join(self, task_id, timeout=None):
        """Wait until a task's background coroutine has fully finished
        (including its final announce). Returns the task dict."""
        rt = self._run_tasks.get(task_id)
        if rt is not None and not rt.done():
            try:
                await asyncio.wait_for(asyncio.shield(rt), timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        return self.tasks.get(task_id)

    async def shutdown(self):
        """Stop everything and persist while the database is still open."""
        for tid, step_task in list(self._step_tasks.items()):
            if not step_task.done():
                step_task.cancel()
        for tid, rt in list(self._run_tasks.items()):
            if not rt.done():
                rt.cancel()
        await asyncio.sleep(0)  # let cancellations land before we persist
        for task in self.tasks.values():
            if task["status"] in tm.LIVE_STATUSES:
                task["status"] = "interrupted"
                task["_finished"] = True
                self._persist(task)

    # --- task construction ------------------------------------------------------

    def _new_task(self, goal, details, confirm_first, flow, replay):
        now = datetime.now().isoformat(timespec="seconds")
        plan = []
        if flow:
            for i, s in enumerate(copy.deepcopy(flow["steps"]), 1):
                s.setdefault("n", i)
                s.update({"status": "pending", "attempts": 0, "evidence": []})
                plan.append(s)
        return {
            "id": tm.new_task_id(), "goal": goal, "details": details or "",
            "status": "planning", "family": flow.get("family") if flow else None,
            "approach": None, "options": None, "pending": None, "plan": plan, "log": [],
            "result_summary": None, "grade": None, "flow_id": flow["id"] if flow else None,
            "replay": replay if flow else None, "plan_confirmed": bool(flow),
            "confirm_first": bool(confirm_first),
            "llm_calls": 0, "tool_calls": 0, "cost_usd": 0.0,
            "started_ts": now, "updated_ts": now, "finished_ts": None,
            "_started_mono": time.monotonic(), "_repairs": 0, "_rethinks": 0, "_cancel": False,
            "delegated_to": "none", "summary": flow["name"] if flow else None,
        }

    def _find_task(self, task_id, waiting):
        if task_id and task_id in self.tasks:
            t = self.tasks[task_id]
            return t if (not waiting or t["status"] == "awaiting_user") else (t if not waiting else None)
        live = [t for t in self.tasks.values() if t["status"] in tm.LIVE_STATUSES]
        if waiting:
            live = [t for t in live if t["status"] == "awaiting_user"]
        if not live:
            return None
        return sorted(live, key=lambda t: t["updated_ts"], reverse=True)[0]

    # --- bookkeeping -------------------------------------------------------------

    def _log(self, task, event, detail=""):
        entry = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event, "detail": str(detail)[:300]}
        task["log"].append(entry)
        task["log"] = task["log"][-80:]
        try:
            self.log(f"task {task['id']} {event}: {str(detail)[:160]}")
        except Exception:
            pass

    def _persist(self, task):
        task["updated_ts"] = datetime.now().isoformat(timespec="seconds")
        try:
            self.tm.update_task(
                task["id"], status=task["status"], plan=task["plan"], log=task["log"],
                pending=task.get("pending"), family=task.get("family"), approach=task.get("approach"),
                llm_calls=task["llm_calls"], tool_calls=task["tool_calls"],
                cost_usd=round(task["cost_usd"], 4), delegated_to=task.get("delegated_to", "none"),
                details=task.get("details", ""))
        except Exception as e:  # persistence must never take a task down
            self.log(f"task {task['id']} persist failed: {e}")

    def _check_budget(self, task):
        lim = self.limits
        if task.get("_cancel"):
            raise TaskCancelled()
        if task["llm_calls"] >= lim.max_llm_calls:
            raise BudgetExceeded(f"used {lim.max_llm_calls} LLM calls")
        if task["tool_calls"] >= lim.max_tool_calls:
            raise BudgetExceeded(f"used {lim.max_tool_calls} tool calls")
        if time.monotonic() - task["_started_mono"] > lim.max_wall_secs:
            raise BudgetExceeded(f"ran longer than {lim.max_wall_secs / 60:.0f} minutes")
        if task["cost_usd"] > lim.max_cost_usd:
            raise BudgetExceeded(f"cost passed ${lim.max_cost_usd:.2f}")

    async def _llm(self, task, messages, **kw):
        self._check_budget(task)
        reply = await self.llm_call(messages, **kw)
        task["llm_calls"] += 1
        rate_in, rate_out = LLM_RATES.get(self.model_name, LLM_RATES.get("gpt-4.1-mini", (0.4, 1.6)))
        task["cost_usd"] += (reply.prompt_tokens * rate_in + reply.completion_tokens * rate_out) / 1e6
        return reply

    async def _llm_json(self, task, system, user, max_tokens=900):
        reply = await self._llm(task, [{"role": "system", "content": system},
                                       {"role": "user", "content": user}],
                                json_mode=True, max_tokens=max_tokens, temperature=0.2)
        try:
            return json.loads(reply.text or "{}")
        except json.JSONDecodeError:
            return None

    async def _tool(self, task, step, name, args):
        self._check_budget(task)
        task["tool_calls"] += 1
        result = await self.toolbox.call(name, args, task_id=task["id"], step_n=step.get("n", 0),
                                         timeout=self.limits.tool_timeout)
        self._log(task, "tool", f"{name}({json.dumps(args, ensure_ascii=False)[:120]}) -> "
                                f"{'ok' if result.get('success', True) else result.get('error')}")
        return result

    # --- announcing --------------------------------------------------------------

    async def _announce(self, task, text, kind):
        """kind: question | milestone | done | failed | cancelled | reminder"""
        now = time.monotonic()
        if kind == "milestone":
            if now - self._last_milestone < self.limits.milestone_min_gap_secs:
                self._log(task, "milestone_dropped", text.splitlines()[0])
                return
            self._last_milestone = now
            if not self._is_awake():
                if self._quiet_cb:
                    await self._quiet_cb(text.splitlines()[0])
                return
        if self._announce_cb is None:
            self._log(task, f"announce({kind})", text.splitlines()[0])
            return
        async with self._announce_lock:
            gap = time.monotonic() - self._last_announce
            if gap < 3.0:
                await asyncio.sleep(3.0 - gap)
            self._last_announce = time.monotonic()
            self._log(task, f"announce({kind})", text.splitlines()[0])
            await self._announce_cb(text)

    def _short_goal(self, task):
        return (task.get("summary") or task["goal"])[:70]

    # --- the loop ------------------------------------------------------------------

    async def _run(self, task):
        fut = self._planned.get(task["id"])
        try:
            if not task["plan"]:
                await self._plan_and_decide(task, fut)
            else:
                task["status"] = "running"
                self._resolve(fut, {"success": True, "task_id": task["id"], "status": "started",
                                    "say": f"On it — {self._short_goal(task)}.",
                                    "first_step": task["plan"][0].get("title")})
            task["status"] = "running"
            self._persist(task)
            i = 0
            while i < len(task["plan"]):        # by index: a rethink appends a new plan mid-loop
                step = task["plan"][i]
                i += 1
                if step.get("status") in ("done", "skipped", "not_needed", "replaced"):
                    continue
                if task.get("_cancel"):
                    raise TaskCancelled()
                if step.get("n", 0) - task.get("_step_base", 0) > self.limits.max_steps:
                    raise BudgetExceeded(f"more than {self.limits.max_steps} steps")
                await self._do_step(task, step)
                if step.get("status") == "failed":
                    if await self._rethink(task, step):
                        continue                # a corrected plan was appended; carry on with it
                    for later in task["plan"]:
                        if later.get("status") == "pending":
                            later["status"] = "skipped"
                    wrong = (step.get("_rethink") or {}).get("what_went_wrong") or task.get("_rethought")
                    await self._finish(task, "failed",
                                       f"stopped at step {step['n']} ({step['title']}): "
                                       f"{step.get('failure', 'it did not succeed')}"
                                       + (f" What I worked out: {wrong}" if wrong else ""))
                    return
                if task.get("_goal_done"):
                    for later in task["plan"]:
                        if later.get("status") == "pending":
                            later["status"] = "not_needed"     # not "skipped": that reads as partial
                    break
                if step.get("milestone") and step is not task["plan"][-1]:
                    nxt = next((s for s in task["plan"] if s.get("status") == "pending"), None)
                    done = sum(1 for s in task["plan"] if s.get("status") == "done")
                    await self._announce(task,
                        f"[Task '{self._short_goal(task)}' progress: step {done} of {len(task['plan'])} "
                        f"done — {step['title']}." + (f" Next: {nxt['title']}." if nxt else "") +
                        "]\n[One short spoken sentence, no details unless asked.]", "milestone")
            await self._finish(task, "success", None)
        except TaskCancelled:
            await self._finish(task, "cancelled", "you stopped it")
        except BudgetExceeded as e:
            await self._finish(task, "failed", f"hit a safety limit: {e}")
        except asyncio.CancelledError:
            if task.get("_cancel"):
                await self._finish(task, "cancelled", "you stopped it")
            elif not task.get("_finished"):
                task["status"] = "interrupted"
                self._persist(task)
                raise
        except Exception as e:
            self._log(task, "error", repr(e))
            await self._finish(task, "failed", f"something broke: {e}")
        finally:
            self._resolve(fut, {"success": True, "task_id": task["id"], "status": task["status"],
                                "say": task.get("result_summary") or "That's done."})

    @staticmethod
    def _resolve(fut, value):
        if fut is not None and not fut.done():
            fut.set_result(value)

    async def _plan_and_decide(self, task, fut):
        vec = await self.tm.embed(task["goal"])
        fam = await self.tm.familiarity(task["goal"], vec=vec)
        self._log(task, "familiarity", f"{fam['level']} ({fam['method']}, {fam['score']})")
        if vec:
            self.tm.update_task(task["id"], embedding=vec)
        plan = await self._plan(task, fam)
        if plan is None:
            task["status"] = "failed"
            task["result_summary"] = "I couldn't work out a plan for that."
            task["_finished"] = True  # _finish() becomes a no-op on unwind
            self._persist(task)
            self.tm.finish_episode(task["id"], "failed", task["result_summary"], 0.0)
            self._resolve(fut, {"success": False, "task_id": task["id"], "status": "failed",
                                "say": "I couldn't work out a plan for that — can you tell me more?"})
            await asyncio.sleep(0.02)
            if not task.get("_delivered_in_turn"):
                await self._announce(task, f"[Task '{self._short_goal(task)}' failed: I couldn't work out a "
                                           f"plan.]\n[Tell the user in one sentence and ask what they meant.]",
                                     "failed")
            raise TaskCancelled()  # unwinds _run
        task.update({"plan": plan["steps"], "summary": plan.get("summary"),
                     "family": plan.get("family"), "approach": plan.get("approach"),
                     "options": plan.get("alternatives"), "_plan_meta": plan})
        if plan.get("same_as_match") is False and fam["level"] in ("routine", "seen"):
            fam["level"] = "similar"
        pref = self.tm.preference_for(task["family"]) if task.get("family") else None
        deleg = self.tm.delegation_record(task["family"]) if task.get("family") else None
        decision = tm.decide_autonomy(task["goal"], plan, fam, pref, deleg,
                                      confirm_first=task.get("confirm_first", False),
                                      free_range=self.free_range)
        self._log(task, "decision", f"{decision['mode']} — {decision['why']}")
        task["_decision"] = decision
        if decision["mode"] == "just_do":
            task["plan_confirmed"] = False
            self._resolve(fut, {"success": True, "task_id": task["id"], "status": "started",
                                "say": decision["say"], "first_step": task["plan"][0]["title"],
                                "note": "Say this one line (in your own words), then wait for notes."})
            return
        kind = "choose_option" if decision["mode"] == "offer_options" else "confirm_plan"
        payload = {"options": plan.get("alternatives") or []} if kind == "choose_option" else {}
        # the task must ALREADY be waiting when start_task returns: the voice
        # LLM may relay the question and get the user's yes in the same turn
        pending = self._prepare_wait(task, {"kind": kind, "text": decision["say"], "payload": payload})
        first = {"success": True, "task_id": task["id"], "status": "needs_your_ok", "say": decision["say"],
                 "note": "Ask this in one or two sentences, then call answer_task with the user's reply."}
        if kind == "choose_option":
            first["options"] = [{"label": o.get("label"), "summary": o.get("summary"),
                                 "tradeoff": o.get("tradeoff")} for o in payload["options"][:3]]
        self._resolve(fut, first)
        for _ in range(8):  # let start_task pick the result up (or note it missed the window)
            if task.get("_delivered_in_turn") or task.get("_in_turn_missed"):
                break
            await asyncio.sleep(0.01)
        note = None
        if not task.get("_delivered_in_turn"):
            note = (f"[Task '{self._short_goal(task)}' planned. {decision['say']}]\n"
                    f"[Relay this to the user in one or two spoken sentences; then call answer_task "
                    f"with their reply.]")
        ans = await self._wait_user(task, pending, note, prepared=True)
        if ans["decision"] == "no":
            raise TaskCancelled()
        if ans["decision"].startswith("option:"):
            label = ans["decision"].split(":", 1)[1]
            chosen = next((o for o in payload["options"] if (o.get("label") or "").lower() == label.lower()), None)
            if chosen and chosen.get("steps"):
                task["plan"] = self._normalize_steps(chosen["steps"])
                task["approach"] = chosen.get("label")
                task["summary"] = chosen.get("summary") or task.get("summary")
            rejected = [o.get("label") for o in payload["options"] if o is not chosen]
            pref_out = self.tm.record_preference(task.get("family"), label, rejected, ans.get("reason"),
                                                 task["id"], remember=getattr(self, "remember", None))
            if pref_out.get("pattern_line"):
                task["_pattern_line"] = pref_out["pattern_line"]
        task["plan_confirmed"] = True

    async def _plan(self, task, fam, *, goal=None, previous=None):
        matches = []
        for ep, score in (fam.get("matches") or [])[:3]:
            matches.append({"goal": ep["goal_text"], "outcome": ep["outcome"], "score": round(score, 2),
                            "approach": ep.get("approach"), "corrected": ep.get("corrected"),
                            "user_feedback": (ep.get("user_feedback") or "")[:200],
                            "steps": [s.get("title") for s in ep.get("plan", [])][:8]})
        facts = list(self._facts() or [])[:20]
        monitors = []
        if self._list_monitors:
            try:
                monitors = self._list_monitors()
            except Exception:
                monitors = []
        sessions = []
        if self.terminals is not None and hasattr(self.terminals, "list_sessions"):
            try:
                sessions = self.terminals.list_sessions()
            except Exception:
                sessions = []
        pref = self.tm.preference_for(fam.get("best", {}).get("family")) if fam.get("best") else None
        brief = {
            "goal": goal or task["goal"], "details": task["details"],
            "now": datetime.now().strftime("%A %H:%M"), "monitors": monitors,
            "open_terminals": sessions, "your_own_code": REPO_DIR, "standing_instructions": facts,
            "similar_past_tasks": matches, "user_preference": pref,
            "avoid_delegation": bool(fam.get("best") and self.tm.delegation_record(
                fam["best"].get("family")).get("bad", 0) >= 3),
        }
        if previous:
            brief["previous_attempt"] = previous
        user = json.dumps(brief, ensure_ascii=False)
        system = (self.planner_prompt.replace("{tools}", self.toolbox.describe())
                  .replace("{families}", ", ".join(tm.FAMILIES))
                  .replace("{approaches}", ", ".join(tm.APPROACHES)))
        plan = await self._llm_json(task, system, user, max_tokens=1600)
        problems = self._validate_plan(plan)
        if problems:
            self._log(task, "plan_invalid", "; ".join(problems))
            plan = await self._llm_json(task, system, user + "\n\nYour previous plan was invalid: "
                                        + "; ".join(problems) + ". Return a corrected plan.", max_tokens=1600)
            problems = self._validate_plan(plan)
            if problems:
                self._log(task, "plan_invalid", "; ".join(problems))
                return None
        merged = one_job_per_project(plan["steps"])
        if len(merged) < len(plan["steps"]):
            self._log(task, "merged", f"{len(plan['steps'])} steps -> {len(merged)}: back-to-back coding "
                                      f"steps on one project run as one Claude Code job")
        plan["steps"] = self._normalize_steps(merged)
        for alt in plan.get("alternatives") or []:
            if alt.get("steps"):
                alt["steps"] = self._normalize_steps(one_job_per_project(alt["steps"]))
        self._log(task, "plan", f"{len(plan['steps'])} steps: " +
                  " | ".join(s["title"] for s in plan["steps"]))
        return plan

    @staticmethod
    def _coerce_step(s):
        """Accept the shapes LLMs actually produce for a step's action: tool
        name/args under other keys or hoisted to the step level."""
        if not isinstance(s, dict):
            return s
        action = s.get("action")
        if not isinstance(action, dict):
            action = {} if action is None else {"value": action}
        tool = (action.get("tool") or action.get("name") or action.get("tool_name")
                or s.get("tool") or s.get("tool_name"))
        args = action.get("args") if isinstance(action.get("args"), dict) else None
        if args is None:
            for key in ("arguments", "parameters", "params", "input"):
                if isinstance(action.get(key), dict):
                    args = action[key]
                    break
        if args is None and isinstance(s.get("args"), dict):
            args = s["args"]
        kind = s.get("kind")
        if kind is None and tool:
            kind = "tool"
        if kind == "tool":
            action = {"tool": tool, "args": args or {}}
        elif kind == "shell":
            action = {"command": action.get("command") or action.get("cmd") or s.get("command"),
                      "purpose": action.get("purpose") or action.get("why") or s.get("purpose", "")}
        elif kind == "delegate":
            action = {"project_dir": action.get("project_dir") or action.get("dir") or s.get("project_dir"),
                      "prompt": action.get("prompt") or action.get("task") or s.get("prompt"),
                      # None = the planner didn't say; delegate_read_only then reads the wording
                      "read_only": None if (ro := action.get("read_only", s.get("read_only"))) is None
                      else _flag(ro),
                      "effort": action.get("effort") or s.get("effort")}
        elif kind == "ask":
            action = {"question": action.get("question") or action.get("text") or s.get("question")}
        s["kind"] = kind
        s["action"] = action
        return s

    def _validate_plan(self, plan):
        problems = []
        if not isinstance(plan, dict):
            return ["not a JSON object"]
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return ["no steps"]
        plan["steps"] = steps = [self._coerce_step(st) for st in steps]
        for alt in plan.get("alternatives") or []:
            if isinstance(alt, dict) and isinstance(alt.get("steps"), list):
                alt["steps"] = [self._coerce_step(st) for st in alt["steps"]]
        if len(steps) > self.limits.max_steps:
            problems.append(f"more than {self.limits.max_steps} steps")
        allowed = set(self.toolbox.names)
        for i, s in enumerate(steps, 1):
            if not isinstance(s, dict):
                problems.append(f"step {i} is not an object")
                continue
            kind = s.get("kind")
            if kind not in ("tool", "agent", "shell", "delegate", "ask"):
                problems.append(f"step {i} has unknown kind {kind!r}")
            action = s.get("action") or {}
            if kind == "tool":
                if action.get("tool") not in allowed and action.get("tool") not in LOOK_TOOLS:
                    problems.append(f"step {i} uses unavailable tool {action.get('tool')!r}")
            elif kind == "shell" and not action.get("command"):
                problems.append(f"step {i} shell step has no command")
            elif kind == "delegate" and not (action.get("project_dir") and action.get("prompt")):
                problems.append(f"step {i} delegate step needs project_dir and prompt")
            elif kind == "delegate" and is_whole_home(action["project_dir"]):
                problems.append(f"step {i} delegate project_dir {action['project_dir']!r} is his whole home "
                                f"folder: give the full path of the project's own folder (your own code is "
                                f"{REPO_DIR})")
            elif kind == "ask" and not action.get("question"):
                problems.append(f"step {i} ask step has no question")
            if s.get("verify") not in (None, "result", "screen", "terminal", "file", "none"):
                problems.append(f"step {i} has unknown verify {s.get('verify')!r}")
            if not s.get("title"):
                problems.append(f"step {i} has no title")
        return problems

    @staticmethod
    def _normalize_steps(steps):
        out = []
        for i, s in enumerate(steps, 1):
            s = dict(s)
            s["n"] = i
            s.setdefault("verify", "result")
            s.setdefault("risk", "none")
            s.setdefault("milestone", False)
            if len(steps) <= 3:
                s["milestone"] = False  # short plans: the done note is enough
            s.setdefault("expect", s.get("title", ""))
            s.update({"status": "pending", "attempts": 0, "evidence": []})
            if s["kind"] == "delegate":
                s["verify"] = "result"
            out.append(s)
        return out

    # --- steps -------------------------------------------------------------------

    async def _do_step(self, task, step):
        step["status"] = "running"
        step["started"] = datetime.now().isoformat(timespec="seconds")
        task["status"] = "running"
        self._persist(task)
        self._log(task, "step", f"{step['n']}/{len(task['plan'])} {step['title']}")
        # the spawner may be pipecat's create_task, which LOGS AND SWALLOWS exceptions — a
        # BudgetExceeded inside a step once vanished, the step stayed "running" and the task
        # reported success having done nothing. Carry the exception out by value instead.
        async def guarded():
            try:
                await self._execute_and_verify(task, step)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                return e
            return None

        st = self._spawn(guarded())
        self._step_tasks[task["id"]] = st
        try:
            err = await st
        except asyncio.CancelledError:
            if task.get("_cancel"):
                raise TaskCancelled()
            raise
        finally:
            self._step_tasks.pop(task["id"], None)
        if isinstance(err, Exception):
            raise err
        if step.get("status") == "running":  # never a silent pass
            step["status"] = "failed"
            step["failure"] = "the step never finished"
        step["finished"] = datetime.now().isoformat(timespec="seconds")
        self._persist(task)

    async def _execute_and_verify(self, task, step):
        outcome = await self._execute_step(task, step)
        if outcome.get("failed"):
            step["status"] = "failed"
            step["failure"] = outcome.get("reason", "it did not succeed")
            return
        verdict = await self._verify_step(task, step)
        if verdict["ok"]:
            step["status"] = "done"
            self._note_goal_done(task, verdict)
            return
        self._log(task, "verify_failed", verdict.get("reason", ""))
        if step["attempts"] <= self.limits.max_repairs_per_step and \
                task["_repairs"] < self.limits.max_repairs_per_task and step["kind"] != "ask":
            task["_repairs"] += 1
            step["attempts"] += 1
            repaired = await self._agent_step(task, step, repair_hint=verdict)
            if not repaired.get("failed"):
                verdict = await self._verify_step(task, step)
                if verdict["ok"]:
                    step["status"] = "done"
                    self._note_goal_done(task, verdict)
                    return
            step["failure"] = repaired.get("reason") or verdict.get("reason") or "verification failed"
        else:
            step["failure"] = verdict.get("reason") or "verification failed"
        step["status"] = "failed"

    def _note_goal_done(self, task, verdict):
        """The verifier saw the whole goal already achieved (a step did more than its share, e.g.
        deleted while it was meant to inspect): the rest of the plan is moot, not failed."""
        if verdict.get("goal_done"):
            task["_goal_done"] = True
            self._log(task, "goal_done", verdict.get("reason", ""))

    async def _execute_step(self, task, step):
        kind = step["kind"]
        action = step.get("action") or {}
        step["attempts"] += 1
        if kind == "tool":
            if action.get("tool") in LOOK_TOOLS:
                result = await self._look(task, action["tool"], action.get("args") or {})
            else:
                result = await self._tool(task, step, action.get("tool"), action.get("args") or {})
            step["evidence"].append(json.dumps(result, ensure_ascii=False)[:800])
            if result.get("success") is False:
                return {"failed": True, "reason": result.get("error", "the tool reported a failure")}
            return {"failed": False}
        if kind == "agent":
            return await self._agent_step(task, step)
        if kind == "shell":
            return await self._shell_step(task, step, action.get("command", ""), action.get("purpose", ""))
        if kind == "delegate":
            return await self._delegate_step(task, step, action)
        if kind == "ask":
            ans = await self._wait_user(task, {"kind": "clarify", "text": action.get("question", ""), "payload": {}},
                                        f"[Task '{self._short_goal(task)}' has a question: {action.get('question', '')}]\n"
                                        f"[Ask the user in one sentence; then call answer_task with their reply.]")
            if ans["decision"] == "cancel":
                raise TaskCancelled()
            step["evidence"].append(f"user said: {ans['text']}")
            task["details"] = (task["details"] + f"\nUser said: {ans['text']}").strip()
            return {"failed": False}
        return {"failed": True, "reason": f"unknown step kind {kind}"}

    async def _agent_step(self, task, step, repair_hint=None):
        brief = {
            "task_goal": task["goal"], "details": task["details"],
            "plan": [f"{s['n']}. {s['title']} [{s.get('status')}]" for s in task["plan"]],
            "this_step": {"n": step["n"], "title": step["title"], "expect": step.get("expect"),
                          "action": step.get("action")},
            "evidence_so_far": [e for s in task["plan"] for e in s.get("evidence", [])][-6:],
        }
        if repair_hint:
            brief["previous_attempt_failed"] = {"reason": repair_hint.get("reason"),
                                                "hint": repair_hint.get("repair_hint")}
        messages = [{"role": "system", "content": self.runner_prompt.replace("{budget}", str(self.limits.step_tool_calls))},
                    {"role": "user", "content": json.dumps(brief, ensure_ascii=False)}]
        tools = self.toolbox.openai_tools(self.private_tools)
        last_call = None
        tried = pushed_back = False
        moves = self.limits.step_tool_calls + 2
        for move in range(moves):
            if move == moves - 1:
                # 2026-10-14: a look-around step spent its moves and the whole task failed with what
                # it had already found in hand — finishing with that is better than nothing
                messages.append({"role": "user", "content": "This is your last move for this step: call "
                                 "step_done with what you have found so far, or step_failed."})
            reply = await self._llm(task, messages, tools=tools, max_tokens=700, temperature=0.2)
            if not reply.tool_calls:
                if reply.text:
                    step["evidence"].append(reply.text[:400])
                messages.append({"role": "assistant", "content": reply.text or ""})
                messages.append({"role": "user", "content": "Use step_done or step_failed to finish this step."})
                continue
            messages.append(reply.assistant_message())
            for tc in reply.tool_calls:
                name, args = tc["name"], tc.get("arguments") or {}
                sig = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                if sig == last_call and name not in ("step_done", "step_failed"):
                    return {"failed": True, "reason": "it kept repeating the same action"}
                last_call = sig
                if name == "step_done":
                    step["evidence"].append(f"{args.get('summary', '')} — {args.get('evidence', '')}"[:600])
                    return {"failed": False}
                if name == "step_failed":
                    if tried or pushed_back:
                        return {"failed": True, "reason": args.get("reason", "the step could not be done")}
                    # 2026-10-14: handed everything it needed, it gave up on its first move — "creating a
                    # file needs the shell confirmation" — with create_file in its hands. Once per step.
                    pushed_back = True
                    self._log(task, "pushed_back", str(args.get("reason", ""))[:160])
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({
                        "not_accepted": "You haven't tried a single tool in this step. Look again at what "
                                        "your tools do (create_file writes a file and needs no yes; the look "
                                        "tools read, find and search) and try. Call step_failed again only if "
                                        "a tool really fails or none of them can do this."})})
                    continue
                tried = True
                if name == "ask_user":
                    ans = await self._wait_user(task, {"kind": "clarify", "text": args.get("question", ""), "payload": {}},
                                                f"[Task '{self._short_goal(task)}' has a question: {args.get('question', '')}]\n"
                                                f"[Ask the user in one sentence; then call answer_task with their reply.]")
                    if ans["decision"] == "cancel":
                        raise TaskCancelled()
                    result = {"user_answer": ans["text"]}
                elif name == "wait":
                    await asyncio.sleep(max(0.0, min(30.0, float(args.get("seconds", 2) or 2))))
                    result = {"success": True, "waited": True}
                elif name in LOOK_TOOLS:
                    result = await self._look(task, name, args)
                    step["evidence"].append(f"{name}: {json.dumps(result, ensure_ascii=False)[:300]}")
                elif name == "shell":
                    out = await self._shell_step(task, step, args.get("command", ""), args.get("purpose", ""))
                    result = out.get("result") or {"success": not out.get("failed"), "error": out.get("reason")}
                elif name == "delegate_to_claude":
                    out = await self._delegate_step(task, step, {"project_dir": args.get("project_dir"),
                                                                 "prompt": args.get("task"),
                                                                 "effort": args.get("effort")
                                                                 or (step.get("action") or {}).get("effort")})
                    result = out.get("result") or {"success": not out.get("failed"), "error": out.get("reason")}
                else:
                    result = await self._tool(task, step, name, args)
                    step["evidence"].append(f"{name}: {json.dumps(result, ensure_ascii=False)[:300]}")
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps(result, ensure_ascii=False)[:3000]})
            if task["tool_calls"] > self.limits.max_tool_calls:
                raise BudgetExceeded(f"used {self.limits.max_tool_calls} tool calls")
        return {"failed": True, "reason": "it ran out of moves for this step"}

    async def _shell_step(self, task, step, command, purpose):
        command = (command or "").strip()
        if not command:
            return {"failed": True, "reason": "empty command"}
        destructive = bool(DESTRUCTIVE_RE.search(command))
        text = (f"{'DESTRUCTIVE: ' if destructive else ''}run `{command}` — {purpose or 'no reason given'}")
        note = (f"[Task '{self._short_goal(task)}' step {step['n']} wants to {text}]\n"
                f"[Tell the user what the command does and ask for a clear yes or no in one sentence; "
                f"then call answer_task with their reply. Nothing runs until they say yes.]")
        ans = await self._wait_user(task, {"kind": "confirm_command", "text": text,
                                           "payload": {"command": command, "purpose": purpose,
                                                       "step_n": step["n"], "destructive": destructive}}, note)
        if ans["decision"] == "cancel":
            raise TaskCancelled()
        if ans["decision"] != "yes":
            return {"failed": True, "reason": "you declined the command"}
        return await self._shell_result(task, step, command)

    async def _shell_result(self, task, step, command):
        result = await self._run_shell(task, command)
        stderr = (result.get("stderr") or "").strip()
        step["evidence"].append(f"shell `{command[:80]}` -> rc={result.get('returncode')} "
                                f"{(result.get('stdout') or result.get('error') or '')[:1000]}"
                                + (f" stderr: {stderr[:300]}" if stderr else ""))
        reason = None
        if result.get("success") is False:
            # the error text is what lets her (and the repair step) say WHY, not just "exit code 1"
            reason = result.get("error") or (f"exit code {result.get('returncode')}"
                                             + (f": {stderr.splitlines()[0][:200]}" if stderr else ""))
        return {"failed": result.get("success") is False, "reason": reason, "result": result}

    async def _run_shell(self, task, command):
        task["tool_calls"] += 1
        fn = self._execute_shell
        if fn is None:
            from modules.tools import execute_shell as fn  # lazy: tools imports pipecat
        result = fn(command)
        if asyncio.iscoroutine(result):
            result = await result
        self._log(task, "shell", f"{command[:100]} -> rc={result.get('returncode')}")
        return result

    async def _delegate_step(self, task, step, action):
        from modules.terminals import resolve_dir  # lazy: terminals pulls in computer/ctypes
        project_dir = resolve_dir(action.get("project_dir") or "~")
        prompt = str(action.get("prompt") or action.get("task") or "").strip()
        if self.terminals is None or not hasattr(self.terminals, "start_claude"):
            return {"failed": True, "reason": "Claude Code isn't available"}
        if is_whole_home(project_dir):
            return {"failed": True, "reason": "won't run Claude Code across your whole home folder; "
                                              "it needs the project's own folder"}
        if not os.path.isdir(project_dir):
            # a new project's folder: Claude Code is fenced to its cwd, so it must exist first.
            # Same fence as her file tools — only ever inside his home.
            home = os.path.normcase(os.path.expanduser("~"))
            if not os.path.normcase(project_dir).startswith(home + os.sep):
                return {"failed": True, "reason": f"won't create a project folder outside your home: {project_dir}"}
            try:
                os.makedirs(project_dir, exist_ok=True)
            except OSError as e:
                return {"failed": True, "reason": f"couldn't create {project_dir}: {e}"}
            self._log(task, "mkdir", project_dir)
        if not task.get("plan_confirmed") and not self.free_range:
            note = (f"[Task '{self._short_goal(task)}' step {step['n']} wants Claude Code to CHANGE code in "
                    f"{project_dir}: {prompt[:160]}]\n[Ask for a clear yes or no in one sentence; then call answer_task.]")
            ans = await self._wait_user(task, {"kind": "confirm_delegate", "text": f"hand this to Claude Code: {prompt[:120]}",
                                               "payload": {"project_dir": project_dir, "prompt": prompt}}, note)
            if ans["decision"] == "cancel":
                raise TaskCancelled()
            if ans["decision"] != "yes":
                return {"failed": True, "reason": "you declined the delegation"}
        # the flag is the truth, not the wording (a merged step's prompt quotes its look-only part)
        read_only = delegate_read_only({"read_only": action.get("read_only"), "prompt": prompt})
        where = os.path.normcase(os.path.normpath(project_dir))
        sessions = task.setdefault("_claude_sessions", {})
        resume = sessions.get(where)
        brief = prompt
        earlier = task.get("_reports") or []
        if resume:
            # the SAME Claude Code session as the last job on this project: it remembers what it read,
            # found and changed (2026-10-13: a fresh job couldn't open the inspection's plan file and
            # redid the whole inspection)
            brief += "\n\n(This continues your earlier work in this session: build on what you already found and did.)"
        elif earlier:
            # a fresh Claude Code session knows nothing of the step before ("using the inspection
            # findings" once meant nothing to it): hand it the previous report verbatim
            last = earlier[-1]
            brief += (f"\n\nWhat the earlier step ({last['title']}) found, in its own words:\n"
                      + tm.clip_middle(last["report"], 2500))
        job = f"{task['id']}-s{step['n']}"
        task["delegated_to"] = "claude_code"
        task["_claude_job"] = job
        self._persist(task)
        is_git = os.path.isdir(os.path.join(project_dir, ".git"))
        before = await self._worktree(task, project_dir) if is_git else None
        # a step is judged on everything IT has changed, from its first job on: a repair re-run that
        # found the work already done was graded "nothing in the project changed"
        if before is not None:
            task.setdefault("_baselines", {}).setdefault((step["n"], where), before)
        base = (task.get("_baselines") or {}).get((step["n"], where), before)
        test_cmd = self._test_command(project_dir) if self.verify_tests and not read_only else None
        tests_before = task.setdefault("_tests_before", {})
        if test_cmd and where not in tests_before:
            # once per project per task, before the agent touches anything: tests he left failing
            # are never blamed on the agent
            tests_before[where] = await self._run_tests(task, test_cmd, project_dir, "before")
        misses = (task.get("_misses") or {}).get(where, 0)
        effort = job_effort(action.get("effort"), read_only, misses, os.getenv("AIVA_CLAUDE_EFFORT"))
        started = await self.terminals.start_claude(job_name=job, project_dir=project_dir, prompt=brief,
                                                    mode="plan" if read_only else "execute",
                                                    resume_id=resume, announce_done=False, effort=effort)
        if not started.get("success"):
            return {"failed": True, "reason": started.get("error", "Claude Code didn't start")}
        self._log(task, "delegate", f"{job} in {project_dir}" + (f" (continuing session {resume})" if resume else "")
                  + (f", effort {effort}" + (f" (up {misses} after a short try)" if misses else "") if effort else ""))
        waited = 0.0
        result = None
        while waited < self.limits.claude_wait_secs:
            result = await self.terminals.wait_job(job, timeout=180)
            if result is not None:
                break
            waited += 180
            if task.get("_cancel"):
                raise TaskCancelled()
            await self._announce(task, f"[Task '{self._short_goal(task)}': Claude Code is still working on step "
                                       f"{step['n']}.]\n[One short spoken sentence.]", "milestone")
        task["_claude_job"] = None
        self._log_claude(task, job, result, effort)
        if result is None:
            try:
                self.terminals.kill_job(job)
            except Exception:
                pass
            return {"failed": True, "reason": "Claude Code took too long"}
        if result.get("claude_session_id"):
            sessions[where] = result["claude_session_id"]
        elif resume:
            sessions.pop(where, None)            # that session couldn't be continued: start fresh next time
        report = result.get("result") or ""
        evidence = {"job_status": result.get("status"), "mode": "inspect" if read_only else "execute",
                    "result_text": tm.clip_middle(report, tm.JUDGE_REPORT_CHARS), "cost_usd": result.get("cost"),
                    "project_dir": project_dir}
        if result.get("denied"):
            evidence["not_allowed_to_run"] = result["denied"][:8]
        if is_git:
            after = await self._worktree(task, project_dir)
            if base is not None and after is not None:
                changed = self._worktree_delta(base, after)
                if base["head"] and after["head"] and base["head"] != after["head"]:
                    out = await self._run_git(task, project_dir, "-c", "core.quotepath=false", "diff",
                                              "--name-only", base["head"], after["head"])
                    changed |= {p.strip() for p in (out.get("stdout") or "").splitlines() if p.strip()}
                found = sorted(p for p in changed if not WORKTREE_NOISE_RE.search(p))[:40]
                if read_only:
                    # plan mode can't edit the project, so whatever changed while it ran was someone
                    # else's (2026-10-17: a README review failed "for modifying tasks.py" — fixes to
                    # the runner made in Frank's own Claude Code session while it ran)
                    evidence["agent_changed"] = []
                    if found:
                        evidence["changed_by_others_meanwhile"] = found
                else:
                    evidence["agent_changed"] = found
                evidence["user_uncommitted_before"] = len(base["files"])
            out = await self._run_git(task, project_dir, "log", "-3", "--oneline")
            evidence["git_log"] = (out.get("stdout") or "")[-600:]
        # "nothing changed" is only a finding when she could actually see what changed
        evidence["diff_checked"] = "agent_changed" in evidence
        if test_cmd and (evidence.get("agent_changed") or not is_git):
            evidence["tests"] = await self._run_tests(task, test_cmd, project_dir, "after")
            if (tests_before.get(where) or {}).get("ran"):
                evidence["tests_before"] = tests_before[where]
        later = [s for s in task["plan"] if s.get("n", 0) > step["n"] and s.get("status") in (None, "pending")]
        grade = await self.tm.grade_agent_work(
            getattr(self, "openai_client", None), self.model_name, task["goal"], evidence,
            step={"title": step["title"], "expect": step.get("expect"), "prompt": prompt[:600],
                  "read_only": read_only, "later_steps": [s.get("title") for s in later]})
        if read_only and grade.get("blocked") == "premise" and grade["grade"] != "bad" \
                and not any(s.get("kind") == "delegate" for s in later):
            # 2026-10-14: "check the speech dimming" found there is none — that IS what the look was
            # for, and the next step was to write it into his note; as a "broken premise" it stopped
            # the task to ask him instead. A premise needs second thoughts only while coding is still
            # to be built on it (the Miso renderer that never existed).
            self._log(task, "finding", grade["reason"])
            grade = {**grade, "blocked": None}
        task["grade"] = grade
        if grade["grade"] in ("partial", "bad") and not grade.get("blocked"):
            # came back short: the next job on this project in this task thinks one level harder
            # (a question or a broken premise is not a matter of thinking harder)
            task.setdefault("_misses", {})[where] = task.get("_misses", {}).get(where, 0) + 1
        task.setdefault("_reports", []).append({"n": step["n"], "title": step["title"], "report": report})
        # the raw report stays on the live task (not the persisted plan, which can become a flow)
        task.setdefault("_step_reports", {})[step["n"]] = {"report": report, "grade": grade, "denied": result.get("denied") or [],
                           "agent_changed": evidence.get("agent_changed"), "read_only": read_only,
                           "tests": evidence.get("tests")}
        step["evidence"].append(f"Claude Code {result.get('status')}: {tm.clip_middle(report, 600)}")
        if evidence.get("tests"):
            step["evidence"].append(f"tests {self.name} ran herself: {self._tests_line(evidence['tests'])}")
        if result.get("denied"):
            step["evidence"].append("not allowed to run: " + "; ".join(result["denied"][:4]))
        step["evidence"].append(f"grade {grade['grade']}: {grade['reason']}"
                                + (f" (blocked: {grade['blocked']})" if grade.get("blocked") else ""))
        self._log(task, "grade", f"{grade['grade']}: {grade['reason']}"
                                 + (f" [blocked: {grade['blocked']}]" if grade.get("blocked") else ""))
        if grade["grade"] == "bad" or grade.get("blocked"):
            why = (f"Claude Code stopped to ask: {grade['question']}" if grade.get("blocked") == "question"
                   and grade.get("question") else f"Claude Code's work didn't check out: {grade['reason']}")
            return {"failed": True, "reason": why,
                    "result": {"success": False, "error": grade["reason"], "job": job}}
        return {"failed": False, "result": {"success": True, "job": job, "grade": grade["grade"],
                                            "summary": evidence["result_text"][:400]}}

    def _log_claude(self, task, job, result, effort=None):
        """The link from a task to its Claude Code run: session id (its full transcript lives in
        ~/.claude/projects/*/<session>.jsonl), cost, status and where the result was saved."""
        log = (result or {}).get("log_path")
        if log:
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            try:
                log = os.path.relpath(log, here).replace("\\", "/")
            except ValueError:  # another drive
                pass
        self._log(task, "claude", json.dumps({
            "job": job, "session": (result or {}).get("claude_session_id"),
            "cost": (result or {}).get("cost"),
            "status": (result or {}).get("status") or "timed out", "log": log, "effort": effort},
            ensure_ascii=False))

    async def _look(self, task, name, args):
        """The read-only private tools (LOOK_TOOLS). Nothing here changes anything, so nothing waits
        on him; git and the tests are the runner's own fixed commands, never model-written."""
        self._check_budget(task)
        task["tool_calls"] += 1

        def num(v, default):
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        if name == "read_file":
            result = await asyncio.to_thread(read_file_lines, args.get("path"), num(args.get("start_line"), 1),
                                             num(args.get("max_lines"), 200))
        elif name == "find_files":
            result = await asyncio.to_thread(find_files, args.get("directory"), args.get("pattern"))
        elif name == "search_text":
            result = await asyncio.to_thread(search_text, args.get("directory"), args.get("text"),
                                             args.get("file_pattern") or "*")
        elif name == "repo_status":
            result = await self._repo_status(task, args.get("project_dir"))
        else:
            tests = args.get("tests") or []
            result = await self._tests_tool(task, args.get("project_dir"), tests if isinstance(tests, list) else [tests])
        self._log(task, "look", f"{name}({json.dumps(args, ensure_ascii=False)[:120]}) -> "
                                + ("ok" if result.get("success", True) else str(result.get("error"))[:120]))
        return result

    @staticmethod
    def _project_path(project_dir):
        from modules.terminals import resolve_dir  # lazy: terminals pulls in computer/ctypes
        where = resolve_dir(project_dir or "~")
        home = os.path.normcase(os.path.expanduser("~"))
        return where if (os.path.normcase(where) + os.sep).startswith(home + os.sep) else None

    async def _repo_status(self, task, project_dir):
        where = self._project_path(project_dir)
        if not where or not os.path.isdir(os.path.join(where, ".git")):
            return {"success": False, "error": f"not a git project in his home: {project_dir}"}
        status = await self._run_git(task, where, "-c", "core.quotepath=false", "status", "--short")
        log = await self._run_git(task, where, "log", "-5", "--oneline")
        diff = await self._run_git(task, where, "diff", "--stat")
        changed = (status.get("stdout") or "").splitlines()
        return {"success": bool(status.get("success")), "project_dir": where, "changed_files": len(changed),
                "status": "\n".join(changed[:150]) + (f"\n... and {len(changed) - 150} more" if len(changed) > 150 else ""),
                "last_commits": (log.get("stdout") or "").strip(),
                "diff_size": ((diff.get("stdout") or "").strip().splitlines() or [""])[-1]}

    async def _tests_tool(self, task, project_dir, tests):
        if not self.verify_tests:
            return {"success": False, "error": "running the tests is switched off (AIVA_TASK_RUN_TESTS)"}
        where = self._project_path(project_dir)
        cmd = self._test_command(where) if where and os.path.isdir(where) else None
        if not cmd:
            return {"success": False, "error": f"no test suite to run in {project_dir} (it needs tests/ "
                                               f"and pytest in the project's own .venv)"}
        picked = pick_test_targets(where, tests)
        if tests and not picked:
            return {"success": False, "error": "none of those test files are in the project"}
        result = await self._run_tests(task, cmd + picked, where, "asked")
        return dict(result, success=bool(result.get("ran")), tests=picked or "all")

    async def _run_git(self, task, project_dir, *args):
        """git for her own bookkeeping: the whole output (see run_git), off the event loop, and not
        counted as a tool call — six of these per job once ate into the task's tool budget."""
        result = await asyncio.to_thread(self._git, project_dir, *args)
        if asyncio.iscoroutine(result):
            result = await result
        if not result.get("success"):
            self._log(task, "git_failed", f"git {' '.join(args)[:80]}: {(result.get('stderr') or '').strip()[:160]}")
        return result

    async def _run_tests(self, task, command, project_dir, when):
        result = await asyncio.to_thread(self._test_runner, command, project_dir, self.limits.test_timeout)
        if asyncio.iscoroutine(result):
            result = await result
        self._log(task, "tests", f"{when}: {self._tests_line(result)}")
        return result

    @staticmethod
    def _tests_line(res):
        if not res.get("ran"):
            return f"didn't run ({res.get('why') or 'unknown'})"
        line = res.get("summary") or f"exit code {res.get('returncode')}"
        return line + ("" if res.get("passed") else " — " + "; ".join(res.get("failing") or [])[:300])

    async def _worktree(self, task, project_dir):
        """{head, files: {path: (mtime_ns, size) | None}} for the uncommitted files, taken before and
        after a job: the difference is what the AGENT changed. Without it his own weeks of
        uncommitted work read as the agent's "large unrelated diff" and a clean job was graded bad."""
        st = await self._run_git(task, project_dir, "-c", "core.quotepath=false", "status", "--porcelain", "-uall")
        if not st.get("success"):
            return None
        head = await self._run_git(task, project_dir, "rev-parse", "HEAD")
        files = {}
        for path in tm.parse_porcelain(st.get("stdout")):
            try:
                info = os.stat(os.path.join(project_dir, path))
                files[path] = (info.st_mtime_ns, info.st_size)
            except OSError:
                files[path] = None                       # deleted in the worktree
        return {"head": (head.get("stdout") or "").strip() or None, "files": files}

    @staticmethod
    def _worktree_delta(before, after):
        changed = {p for p, sig in after["files"].items() if before["files"].get(p, "absent") != sig}
        return changed | {p for p in before["files"] if p not in after["files"]}

    # --- second thoughts: read the agent's report, learn, re-plan or ask -------------

    async def _rethink(self, task, step):
        """A delegated step failed or the agent stopped to ask. Read its whole report the way he
        would: what went wrong, what it shows to be TRUE about the project (kept as facts, only
        when the report really says so), then — once per task — re-plan around it, or put the
        agent's question to the user and re-plan with the answer. True when a new plan was added."""
        info = (task.get("_step_reports") or {}).get(step["n"])
        if not info or task["_rethinks"] >= self.limits.max_rethinks:
            return False
        task["_rethinks"] += 1
        grade = info.get("grade") or {}
        try:
            believed = [str(f) for f in (self._facts() or [])][-120:]
        except Exception:
            believed = []
        try:
            refl = await self._llm_json(task, self.rethink_prompt, json.dumps({
                "what_you_believe": believed,
                "task_goal": task["goal"], "details": task["details"],
                "plan": [f"{s['n']}. {s['title']} [{s.get('status')}]" for s in task["plan"]],
                "this_step": {"title": step["title"], "expect": step.get("expect"),
                              "prompt": (step.get("action") or {}).get("prompt")},
                "grade": {k: grade.get(k) for k in ("grade", "reason", "blocked", "question")},
                "not_allowed_to_run": info.get("denied"), "files_agent_changed": info.get("agent_changed"),
                "tests_you_ran": info.get("tests"),
                "agent_report": tm.clip_middle(info.get("report"), tm.JUDGE_REPORT_CHARS),
            }, ensure_ascii=False), max_tokens=700)
        except BudgetExceeded:
            return False
        if not isinstance(refl, dict) or not (refl.get("what_went_wrong") or "").strip():
            return False
        wrong = refl["what_went_wrong"].strip()
        nxt = refl.get("next")
        step["_rethink"] = {"what_went_wrong": wrong, "next": nxt}
        step["evidence"].append(f"second thoughts: {wrong}")
        self._log(task, "rethink", f"{nxt}: {wrong}")
        self._learn(task, refl, info.get("report") or "", believed)
        previous = {"failed_step": step["title"], "what_went_wrong": wrong,
                    "agent_report": tm.clip_middle(info.get("report"), 2500)}
        revised = (refl.get("revised_goal") or "").strip()
        question = (refl.get("question") or "").strip() or (grade.get("question") or "").strip()
        if nxt == "retry" and revised:
            goal = revised
        elif nxt == "ask" and question:
            ans = await self._wait_user(task, {"kind": "clarify", "text": question, "payload": {}},
                                        f"[Task '{self._short_goal(task)}' hit a snag: {wrong}]\n"
                                        f"[Tell the user what you found in one sentence and ask: {question} "
                                        f"Then call answer_task with their reply.]")
            if ans["decision"] == "cancel":
                raise TaskCancelled()
            if NO_WORDS.match(ans.get("text") or ""):
                self._log(task, "rethink_dropped", "the user said to leave it")
                return False
            task["details"] = (task["details"] + f"\nUser said: {ans['text']}").strip()
            previous["user_answer"] = ans["text"]
            goal = task["goal"]
        else:
            return False
        return await self._replan(task, goal, previous)

    def _learn(self, task, refl, report, believed=()):
        """Keep what the report proved (verbatim-checked against it — a paraphrase or a guess is
        dropped and logged) as facts, retire the old facts a PROVED belief replaces (only exact
        lines she held, only retirable kinds — see Memory.retire_fact), and put the lesson on the
        task row for LESSONS FROM PAST TASKS. Adding the truth beside the old belief isn't enough:
        three "it lives in the Miso renderer" project facts re-seeded every plan."""
        kept = 0
        believed = set(believed or ())
        retire = getattr(self, "retire_fact", None)
        for b in (refl.get("beliefs") or [])[:3]:
            if not isinstance(b, dict):
                continue
            fact = (b.get("fact") or "").strip()
            if not fact or len(fact) > 240:
                continue
            if not tm.found_in(b.get("evidence"), report):
                self._log(task, "belief_unverified", fact)
                continue
            remember = getattr(self, "remember", None)
            if remember is not None:
                try:
                    remember(f"Checked myself: {fact}", "learned")
                except Exception as e:
                    self.log(f"tasks: remember failed ({e!r})")
                    continue
            kept += 1
            self._log(task, "learned", fact)
            for old in (b.get("replaces") or [])[:5]:
                old = str(old).strip()
                if retire is None or old not in believed or old == f"Checked myself: {fact}":
                    continue
                try:
                    if retire(old):
                        self._log(task, "unlearned", old)
                except Exception as e:
                    self.log(f"tasks: retire_fact failed ({e!r})")
        lesson = (refl.get("lesson") or "").strip()
        if lesson and lesson.lower() not in ("null", "none"):
            try:
                self.tm.update_task(task["id"], self_lesson=lesson[:300])
            except Exception as e:
                self.log(f"tasks: self_lesson failed ({e!r})")
            task["self_lesson"] = lesson[:300]
            self._log(task, "lesson", lesson)
        return kept

    async def _replan(self, task, goal, previous):
        plan = await self._plan(task, {"level": "attempted", "matches": []}, goal=goal, previous=previous)
        if plan is None:
            self._log(task, "replan_failed", goal)
            return False
        decision = tm.decide_autonomy(goal, plan, {"level": "attempted",
                                                   "last_failure": previous["what_went_wrong"]},
                                      confirm_first=task.get("confirm_first", False),
                                      free_range=self.free_range)
        if decision["mode"] != "just_do":
            ans = await self._wait_user(task, {"kind": "confirm_plan", "text": decision["say"], "payload": {}},
                                        f"[Task '{self._short_goal(task)}' rethought: {decision['say']}]\n"
                                        f"[Ask in one or two spoken sentences; then call answer_task.]")
            if ans["decision"] == "cancel":
                raise TaskCancelled()
            if ans["decision"] != "yes":
                self._log(task, "replan_declined", ans.get("text", ""))
                return False
            task["plan_confirmed"] = True
        else:
            await self._announce(task, f"[Task '{self._short_goal(task)}': the first try didn't work — "
                                       f"{previous['what_went_wrong']} Trying again: "
                                       f"{plan.get('summary') or goal}.]\n[One short spoken sentence.]", "rethink")
        base = max(s.get("n", 0) for s in task["plan"])
        for s in task["plan"]:
            if s.get("status") == "pending":
                s["status"] = "replaced"
        for s in plan["steps"]:
            s["n"] += base
            s["replanned"] = True
        task["_step_base"] = base
        task["plan"].extend(plan["steps"])
        task["revised_goal"] = goal
        task["_rethought"] = previous["what_went_wrong"]
        self._log(task, "replan", f"{len(plan['steps'])} steps: " + " | ".join(s["title"] for s in plan["steps"]))
        self._persist(task)
        return True

    async def _verify_step(self, task, step):
        mode = step.get("verify") or "result"
        if mode == "none":
            return {"ok": True, "confidence": 1.0, "reason": "no verification requested"}
        observation = None
        if step.get("kind") == "shell" and mode in ("terminal", "screen"):
            # a shell step runs invisibly: its own output (in the evidence) is the only observation.
            # Reading a terminal here looked for one called "main" that never existed, failed a
            # command that had worked, and sent the repair off to open a visible terminal.
            mode = "result"
        if mode == "file":
            path = self._path_from_step(step)
            if path:
                try:
                    resolved = self._resolve_path(path) if self._resolve_path else os.path.expanduser(path)
                    exists = os.path.exists(resolved)
                except Exception as e:
                    return {"ok": False, "confidence": 1.0, "reason": f"bad path: {e}", "repair_hint": None}
                if exists:
                    return {"ok": True, "confidence": 1.0, "reason": f"{path} exists"}
                observation = f"{path} does not exist"
        elif mode == "screen":
            if self._describe_screen is None:
                from modules.computer import describe_screen as ds
                self._describe_screen = ds
            try:
                observation = await self._describe_screen(looking_for=step.get("expect"))
            except Exception as e:
                observation = f"could not look at the screen: {e}"
            task["tool_calls"] += 1
        elif mode == "terminal":
            name = ((step.get("action") or {}).get("args") or {}).get("terminal") or "main"  # the tools' default
            if self.terminals is not None and hasattr(self.terminals, "read_session"):
                try:
                    obs = self.terminals.read_session(name)
                    if obs.get("success") is not False:       # no such terminal → use the evidence
                        observation = json.dumps(obs, ensure_ascii=False)[-1500:]
                except Exception as e:
                    observation = f"could not read the terminal: {e}"
        if observation is None:
            # a coding job leaves report, tests, denials and grade: all four are the observation
            keep = 4 if step.get("kind") == "delegate" else 3
            observation = "\n".join(step.get("evidence", [])[-keep:]) or "(no evidence recorded)"
        elif step.get("kind") == "agent" and step.get("evidence"):
            # 2026-10-14: "the observation is recorded" was judged on a fresh screenshot alone, which
            # can't show what the step wrote down — failed, and spent the task's only repair
            observation = (f"What it shows now: {str(observation)[:1900]}\n\nWhat the step itself reported "
                           f"(its own claim; what is visible above is the ground truth): {step['evidence'][-1][:800]}")
        verdict = await self._llm_json(
            task, VERIFIER_PROMPT,
            json.dumps({"task_goal": task.get("goal"), "step": step["title"], "criterion": step.get("expect"),
                        "observation": observation[:3000]}, ensure_ascii=False), max_tokens=200)
        if not isinstance(verdict, dict):
            return {"ok": False, "confidence": 0.0, "reason": "the check came back garbled", "repair_hint": None}
        ok = bool(verdict.get("ok")) and float(verdict.get("confidence", 0) or 0) >= 0.6
        return {"ok": ok, "confidence": float(verdict.get("confidence", 0) or 0),
                "reason": verdict.get("reason", ""), "repair_hint": verdict.get("repair_hint"),
                "goal_done": ok and verdict.get("goal_done") is True}

    @staticmethod
    def _path_from_step(step):
        args = ((step.get("action") or {}).get("args") or {})
        for key in ("file_path", "path", "directory", "target"):
            if args.get(key):
                return str(args[key])
        m = re.search(r"([A-Za-z]:\\[^\s\"']+|~[\\/][^\s\"']+|\b(?:desktop|documents|downloads)[\\/][^\s\"']+)",
                      step.get("expect") or "", re.I)
        return m.group(1) if m else None

    # --- waiting on the user --------------------------------------------------------

    def _prepare_wait(self, task, pending):
        """Put the task into awaiting_user NOW (answers may arrive before the
        coroutine that will wait gets to run)."""
        pending = dict(pending, asked_ts=datetime.now().isoformat(timespec="seconds"), reminded=False)
        task["pending"] = pending
        task["status"] = "awaiting_user"
        task["_answer"] = None
        self._events[task["id"]].clear()
        self._persist(task)
        return pending

    async def _wait_user(self, task, pending, note=None, prepared=False):
        if not prepared:
            pending = self._prepare_wait(task, pending)
        event = self._events[task["id"]]
        waited_from = time.monotonic()
        try:
            if note:
                await self._announce(task, note, "question")
            try:
                await asyncio.wait_for(event.wait(), self.limits.answer_reminder_secs)
            except asyncio.TimeoutError:
                pending["reminded"] = True
                await self._announce(task, f"[Task '{self._short_goal(task)}' is still waiting for your answer: "
                                           f"{pending['text']}]\n[Ask once more, briefly.]", "reminder")
                await event.wait()
        finally:
            # the wall-clock limit is for her working, not for him answering: a 21-minute wait
            # for a yes once tripped it on the first step
            if "_started_mono" in task:
                task["_started_mono"] += time.monotonic() - waited_from
        ans = task.get("_answer") or {"decision": "cancel", "text": ""}
        task["pending"] = None
        task["status"] = "running"
        self._persist(task)
        if task.get("_cancel"):
            raise TaskCancelled()
        return ans

    @staticmethod
    def _match_option(text, options):
        low = (text or "").lower()
        for o in options:
            label = (o.get("label") or "").lower()
            if label and (label in low or any(w in low for w in label.split("_") if len(w) > 3)):
                return o
        if re.search(r"\b(first|former|the one|number one|option one|a)\b", low) and options:
            return options[0]
        if re.search(r"\b(second|latter|other|number two|option two|b)\b", low) and len(options) > 1:
            return options[1]
        return None

    async def _classify_answer(self, task, text, options):
        payload = {"reply": text, "question": (task.get("pending") or {}).get("text"),
                   "options": [o.get("label") for o in options]}
        try:
            data = await self._llm_json(task, ANSWER_PROMPT, json.dumps(payload, ensure_ascii=False), max_tokens=80)
        except (BudgetExceeded, TaskCancelled):
            return {"decision": "other"}
        return data if isinstance(data, dict) else {"decision": "other"}

    # --- finishing -----------------------------------------------------------------------

    async def _finish(self, task, outcome, reason):
        if task.get("_finished"):
            return
        task["_finished"] = True
        duration = time.monotonic() - task["_started_mono"]
        mins = duration / 60
        goal = self._short_goal(task)
        if outcome == "success":
            partial = any(s.get("status") == "skipped" for s in task["plan"])
            summary = await self._summarize(task)
            outcome = "partial" if partial else "success"
            task["result_summary"] = summary
            checked = "; ".join(s["title"] for s in task["plan"] if s.get("status") == "done")[:200]
            note = (f"[Task '{goal}' finished after {mins:.1f} min, status: {outcome}]\n"
                    f"Result: {summary}\nChecked: {checked}\n")
            if task.get("grade"):
                note += f"Claude Code's work checked: grade {task['grade']['grade'].upper()} — {task['grade']['reason']}.\n"
            reports = task.get("_reports") or []
            if reports:
                # what the job actually said, so she can tell him (2026-10-17: "take a look and tell me
                # what you'd change" came back as two sentences made from 300 characters of the report)
                note += ("What Claude Code reported, in its own words: "
                         + tm.clip_middle(reports[-1]["report"], 3000) + "\n")
            if task.get("_pattern_line"):
                note += f"Pattern noticed: {task['_pattern_line']}\n"
            if task.get("_rethought"):
                note += (f"The first try didn't work ({task['_rethought']}); you worked out why from the "
                         f"agent's report and re-planned.\n")
            task["status"] = "done"
        elif outcome == "cancelled":
            task["result_summary"] = "cancelled"
            last = next((s["title"] for s in reversed(task["plan"]) if s.get("status") == "done"), None)
            note = (f"[Task '{goal}' cancelled" + (f"; last completed: {last}" if last else "") +
                    ".]\n[Confirm in one short sentence.]")
            task["status"] = "cancelled"
        else:
            task["result_summary"] = reason or "it failed"
            failed = next((s for s in reversed(task["plan"]) if s.get("status") == "failed"), None)
            where = f" at step {failed['n']} of {len(task['plan'])} ({failed['title']})" if failed else ""
            # what it found is not lost with the task: she gets it, and can finish straight away
            # (2026-10-14: a text-file task failed in its look-around step and the file never got written)
            gathered = [str(e) for s in task["plan"] for e in s.get("evidence", [])
                        if not str(e).startswith("grade ")][-4:]
            note = (f"[Task '{goal}' failed{where} after {mins:.1f} min: {reason}]\n"
                    + ("[What it had found before it stopped:\n"
                       + "\n".join(f"- {tm.clip_middle(e, 400)}" for e in gathered) + "]\n" if gathered else "")
                    + "[Tell the user plainly what went wrong in one sentence, then offer the most direct way "
                      "to still get what they asked for — usually doing the rest yourself right now with what "
                      "was found. Don't make them start over.]")
            task["status"] = "failed"
        task["pending"] = None
        task["finished_ts"] = datetime.now().isoformat(timespec="seconds")
        self._persist(task)
        grade = task.get("grade") or {}
        self.tm.finish_episode(task["id"], outcome, task["result_summary"], round(duration, 1),
                               agent_grade=grade.get("grade"), agent_grade_reason=grade.get("reason"))
        if task.get("flow_id"):
            self.tm.bump_flow_stats(task["flow_id"], outcome == "success")
        if self._on_outcome is not None:
            try:
                res = self._on_outcome(outcome, goal)
                if hasattr(res, "__await__"):
                    await res
            except Exception as e:
                self.log(f"tasks: on_outcome hook failed ({e!r})")
        if outcome in ("success", "partial") and self._on_result is not None:
            try:
                res = self._on_result(task["goal"], task["result_summary"])
                if hasattr(res, "__await__"):
                    await res
            except Exception as e:
                self.log(f"tasks: on_result hook failed ({e!r})")
        if outcome == "success" and self._on_done is not None:
            try:
                res = self._on_done(task["result_summary"], task.get("family"))
                if hasattr(res, "__await__"):
                    await res
            except Exception as e:
                self.log(f"tasks: on_done hook failed ({e!r})")
        if outcome in ("success", "partial"):
            self.last_completed_task_id = task["id"]
            if outcome == "success" and not task.get("flow_id"):
                cand = self.tm.flow_candidate(task["id"])
                if cand:
                    meta = task.get("_plan_meta") or {}
                    name = meta.get("suggested_flow_name") or cand["name"]
                    trig = (meta.get("suggested_triggers") or cand["triggers"])[:2]
                    task["_flow_offer"] = {"name": name, "triggers": trig}
                    note += (f"Flow suggestion: offer to keep this as a flow called '{name}' "
                             f"(trigger: '{trig[0] if trig else name}'); if the user agrees call save_flow.\n")
            note += "[Briefly tell the user it's done and summarize the outcome in one or two spoken sentences.]"
        self._log(task, "finish", f"{outcome}: {task['result_summary']}")
        await self._announce(task, note, "done" if outcome in ("success", "partial") else outcome)

    async def _summarize(self, task):
        evidence = [f"{s['n']}. {s['title']} [{s.get('status')}]: " + " / ".join(s.get("evidence", [])[-2:])[:300]
                    for s in task["plan"]]
        try:
            reply = await self._llm(task, [{"role": "system", "content": SUMMARY_PROMPT},
                                           {"role": "user", "content": json.dumps(
                                               {"goal": task["goal"], "steps": evidence}, ensure_ascii=False)}],
                                    max_tokens=120, temperature=0.3)
            text = (reply.text or "").strip()
        except Exception:
            text = ""
        return text or f"Finished: {self._short_goal(task)}."
