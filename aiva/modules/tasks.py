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
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from modules import task_memory as tm

from modules.utilities import LLM_RATES, llm_kwargs

RUNNER_TOOL_ALLOW = frozenset({
    "launch_app", "create_file", "open_file", "list_directory",
    "look_at_screen", "focus_window", "click_ui_element", "type_text", "press_keys",
    "click_at", "scroll",
    "open_terminal", "run_in_terminal", "type_in_terminal", "read_terminal",
    "list_terminals", "show_terminal", "get_datetime",
})

DESTRUCTIVE_RE = re.compile(
    r"(Remove-Item[^|;]*-Recurse|\brm\s+-r|\brmdir\s+/s|\bdel\s+/s|\bformat\s+[a-z]:|\bdiskpart\b|"
    r"\bshutdown\b|\breg\s+delete|git\s+push\s+[^|;]*--force|git\s+reset\s+--hard|"
    r"Stop-Computer|Restart-Computer|\bmkfs\b)", re.I)

READONLY_EVIDENCE_CMDS = ("git -C {dir} status --short", "git -C {dir} diff --stat",
                          "git -C {dir} log -3 --oneline")

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
    max_tool_calls: int = 40
    max_llm_calls: int = 60
    max_wall_secs: float = 20 * 60
    step_tool_calls: int = 6
    tool_timeout: float = 60
    claude_wait_secs: float = 30 * 60
    answer_reminder_secs: float = 180
    max_cost_usd: float = 1.50
    milestone_min_gap_secs: float = 30
    plan_wait_secs: float = 12
    max_repairs_per_step: int = 1
    max_repairs_per_task: int = 2


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
    _tool("shell", "Run a PowerShell/cmd command invisibly. The USER must confirm it first — the "
                   "task pauses until they answer. Prefer run_in_terminal (visible) for ordinary "
                   "commands; use this for quick checks or when no terminal is wanted.",
          {"command": {"type": "string"}, "purpose": {"type": "string",
           "description": "One plain sentence: what this command is for"}}, ["command", "purpose"]),
    _tool("delegate_to_claude", "Hand coding work (multi-file edits, bug fixes, features, tests) in a "
                                "project directory to Claude Code. Takes minutes; the result is graded.",
          {"project_dir": {"type": "string"}, "task": {"type": "string"},
           "why": {"type": "string", "description": "Why this needs an agent rather than you"}},
          ["project_dir", "task", "why"]),
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


class ToolBox:
    """The voice LLM's own tool handlers, callable out-of-band with a fake
    FunctionCallParams. Results are always non-empty dicts."""

    def __init__(self, handlers, schemas=None, allow=RUNNER_TOOL_ALLOW):
        self._handlers = dict(handlers)
        self._allow = frozenset(allow)
        self._schemas = {}
        if schemas is not None:
            for fs in getattr(schemas, "standard_tools", schemas):
                self._schemas[fs.name] = fs

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
                tools.append(_tool(name, f"Aiva tool {name}", {}, []))
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

PLANNER_PROMPT = """You are the planner inside Aiva, a voice assistant that controls the user's \
Windows PC. Turn the goal into a short, concrete plan of steps. Reply with JSON only.

Step kinds:
- "tool": exactly one call of a listed tool with fixed args (use when you know the exact args). \
action MUST be {"tool": "<name from the tool list>", "args": {<that tool's arguments>}}. Example step: \
{"n": 1, "title": "Create hello.txt on the desktop", "kind": "tool", "action": {"tool": "create_file", \
"args": {"file_path": "desktop/hello.txt", "content": "hello"}}, "expect": "hello.txt exists on the \
desktop", "verify": "file", "milestone": false, "risk": "reversible"}
- "agent": a small goal for a sub-agent that may use several tools and must look at results \
(use when args depend on what's on screen, or the step needs a look-then-act).
- "shell": an invisible command; the user will be asked to confirm it. action: {"command", "purpose"}.
- "delegate": hand CODING work (multi-file edits, bug fixes, features, tests) in a project \
directory to Claude Code. action: {"project_dir", "prompt"}. Never for desktop/UI work.
- "ask": ask the user one question you cannot proceed without. action: {"question"}.

Every step: {"n", "title" (<= 12 words, spoken to the user), "kind", "action", "expect" (plain-words \
success criterion), "verify": "result"|"screen"|"terminal"|"file"|"none", "milestone": bool, \
"risk": "none"|"reversible"|"irreversible"}. Prefer "file"/"result"/"terminal" verification over \
"screen". Mark at most 3 milestones (only for plans longer than 3 steps). 1 to 12 steps. Mark \
a step "irreversible" ONLY when it destroys or changes something that cannot be recovered - deleting, overwriting an EXISTING file, uninstalling, sending a message, paying; creating a new file, opening or focusing apps, reading, or typing into an empty document is "reversible" or "none". If a similar past task or a \
correction is listed, follow what worked and avoid what the user complained about.

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

RUNNER_SYSTEM_PROMPT = """You are the hands of Aiva, a voice assistant, carrying out ONE step of a \
larger task on the user's Windows PC. Use the tools to act, then LOOK at results before deciding. \
Prefer click_ui_element by visible name over click_at; use look_at_screen to find things or read \
errors. Never type a password the user hasn't given. Never run destructive commands without the \
shell tool (which asks the user). When the step is achieved, call step_done with what you observed \
as evidence. If it can't be done, call step_failed with the reason. Do not call the same tool with \
the same arguments twice in a row. Be economical: at most {budget} tool calls for this step."""

VERIFIER_PROMPT = """You verify one step of a task done by an assistant on a Windows PC. Given the \
step's success criterion and the observation, reply JSON: {"ok": true|false, "confidence": 0..1, \
"reason": "<one plain sentence>", "repair_hint": "<what to try, or null>"}. Be strict: if the \
observation doesn't show the criterion is met, ok is false."""

SUMMARY_PROMPT = """Summarize what a task achieved for a voice assistant to say aloud: two plain \
sentences, past tense, naming the concrete results (files, apps, values). No lists, no markdown. \
If something was skipped or partial, say so plainly."""

ANSWER_PROMPT = """The user was asked a question by a voice assistant and replied. Classify the \
reply as JSON: {"decision": "yes"|"no"|"option:<label>"|"other", "reason": "<their why, if any>"}. \
Options, when offered, are listed; match the closest label."""


# --- the runner --------------------------------------------------------------------

class TaskRunner:
    def __init__(self, *, task_memory, toolbox, llm_call, terminals=None, limits=None,
                 log=print, execute_shell=None, describe_screen=None, list_monitors=None,
                 resolve_path=None, facts=None, model_name="gpt-4.1"):
        self.tm = task_memory
        self.toolbox = toolbox
        self.llm_call = llm_call
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
            "_started_mono": time.monotonic(), "_repairs": 0, "_cancel": False,
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
            for step in task["plan"]:
                if step.get("status") in ("done", "skipped"):
                    continue
                if task.get("_cancel"):
                    raise TaskCancelled()
                if step.get("n", 0) > self.limits.max_steps:
                    raise BudgetExceeded(f"more than {self.limits.max_steps} steps")
                await self._do_step(task, step)
                if step.get("status") == "failed":
                    for later in task["plan"]:
                        if later.get("status") == "pending":
                            later["status"] = "skipped"
                    await self._finish(task, "failed",
                                       f"stopped at step {step['n']} ({step['title']}): "
                                       f"{step.get('failure', 'it did not succeed')}")
                    return
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
                                      confirm_first=task.get("confirm_first", False))
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

    async def _plan(self, task, fam):
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
        user = json.dumps({
            "goal": task["goal"], "details": task["details"],
            "now": datetime.now().strftime("%A %H:%M"), "monitors": monitors,
            "open_terminals": sessions, "standing_instructions": facts,
            "similar_past_tasks": matches, "user_preference": pref,
            "avoid_delegation": bool(fam.get("best") and self.tm.delegation_record(
                fam["best"].get("family")).get("bad", 0) >= 3),
        }, ensure_ascii=False)
        system = (PLANNER_PROMPT.replace("{tools}", self.toolbox.describe())
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
        plan["steps"] = self._normalize_steps(plan["steps"])
        for alt in plan.get("alternatives") or []:
            if alt.get("steps"):
                alt["steps"] = self._normalize_steps(alt["steps"])
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
                      "prompt": action.get("prompt") or action.get("task") or s.get("prompt")}
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
                if action.get("tool") not in allowed:
                    problems.append(f"step {i} uses unavailable tool {action.get('tool')!r}")
            elif kind == "shell" and not action.get("command"):
                problems.append(f"step {i} shell step has no command")
            elif kind == "delegate" and not (action.get("project_dir") and action.get("prompt")):
                problems.append(f"step {i} delegate step needs project_dir and prompt")
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
        coro = self._execute_and_verify(task, step)
        st = self._spawn(coro)
        self._step_tasks[task["id"]] = st
        try:
            await st
        except asyncio.CancelledError:
            if task.get("_cancel"):
                raise TaskCancelled()
            raise
        finally:
            self._step_tasks.pop(task["id"], None)
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
                    return
            step["failure"] = repaired.get("reason") or verdict.get("reason") or "verification failed"
        else:
            step["failure"] = verdict.get("reason") or "verification failed"
        step["status"] = "failed"

    async def _execute_step(self, task, step):
        kind = step["kind"]
        action = step.get("action") or {}
        step["attempts"] += 1
        if kind == "tool":
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
        messages = [{"role": "system", "content": RUNNER_SYSTEM_PROMPT.replace("{budget}", str(self.limits.step_tool_calls))},
                    {"role": "user", "content": json.dumps(brief, ensure_ascii=False)}]
        tools = self.toolbox.openai_tools(PRIVATE_TOOLS)
        last_call = None
        for _ in range(self.limits.step_tool_calls + 2):
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
                    return {"failed": True, "reason": args.get("reason", "the step could not be done")}
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
                elif name == "shell":
                    out = await self._shell_step(task, step, args.get("command", ""), args.get("purpose", ""))
                    result = out.get("result") or {"success": not out.get("failed"), "error": out.get("reason")}
                elif name == "delegate_to_claude":
                    out = await self._delegate_step(task, step, {"project_dir": args.get("project_dir"),
                                                                 "prompt": args.get("task")})
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
        result = await self._run_shell(task, command)
        step["evidence"].append(f"shell `{command[:80]}` -> rc={result.get('returncode')} "
                                f"{(result.get('stdout') or result.get('error') or '')[:300]}")
        return {"failed": result.get("success") is False, "reason": result.get("error") or
                (f"exit code {result.get('returncode')}" if result.get("success") is False else None),
                "result": result}

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
        project_dir = os.path.expanduser(str(action.get("project_dir") or "."))
        prompt = str(action.get("prompt") or action.get("task") or "").strip()
        if self.terminals is None or not hasattr(self.terminals, "start_claude"):
            return {"failed": True, "reason": "Claude Code isn't available"}
        if not task.get("plan_confirmed"):
            note = (f"[Task '{self._short_goal(task)}' step {step['n']} wants Claude Code to CHANGE code in "
                    f"{project_dir}: {prompt[:160]}]\n[Ask for a clear yes or no in one sentence; then call answer_task.]")
            ans = await self._wait_user(task, {"kind": "confirm_delegate", "text": f"hand this to Claude Code: {prompt[:120]}",
                                               "payload": {"project_dir": project_dir, "prompt": prompt}}, note)
            if ans["decision"] == "cancel":
                raise TaskCancelled()
            if ans["decision"] != "yes":
                return {"failed": True, "reason": "you declined the delegation"}
        job = f"{task['id']}-s{step['n']}"
        task["delegated_to"] = "claude_code"
        task["_claude_job"] = job
        self._persist(task)
        started = await self.terminals.start_claude(job_name=job, project_dir=project_dir, prompt=prompt,
                                                    mode="execute", announce_done=False)
        if not started.get("success"):
            return {"failed": True, "reason": started.get("error", "Claude Code didn't start")}
        self._log(task, "delegate", f"{job} in {project_dir}")
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
        if result is None:
            try:
                self.terminals.kill_job(job)
            except Exception:
                pass
            return {"failed": True, "reason": "Claude Code took too long"}
        evidence = {"job_status": result.get("status"), "mode": "execute",
                    "result_text": (result.get("result") or "")[-1500:], "cost_usd": result.get("cost"),
                    "project_dir": project_dir}
        diff_checked = False
        if os.path.isdir(os.path.join(project_dir, ".git")):
            for tmpl in READONLY_EVIDENCE_CMDS:
                cmd = tmpl.format(dir=project_dir)
                out = await self._run_shell(task, cmd)
                key = "diff_stat" if "diff" in cmd else ("git_status" if "status" in cmd else "git_log")
                evidence[key] = (out.get("stdout") or "")[-1200:]
            diff_checked = True
        evidence["diff_checked"] = diff_checked
        grade = await self.tm.grade_agent_work(getattr(self, "openai_client", None), self.model_name,
                                               task["goal"], evidence)
        task["grade"] = grade
        step["evidence"].append(f"Claude Code {result.get('status')}: {evidence['result_text'][:300]}")
        step["evidence"].append(f"grade {grade['grade']}: {grade['reason']}")
        self._log(task, "grade", f"{grade['grade']}: {grade['reason']}")
        if grade["grade"] == "bad":
            return {"failed": True, "reason": f"Claude Code's work didn't check out: {grade['reason']}",
                    "result": {"success": False, "error": grade["reason"], "job": job}}
        return {"failed": False, "result": {"success": True, "job": job, "grade": grade["grade"],
                                            "summary": evidence["result_text"][:400]}}

    async def _verify_step(self, task, step):
        mode = step.get("verify") or "result"
        if mode == "none":
            return {"ok": True, "confidence": 1.0, "reason": "no verification requested"}
        observation = None
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
            name = ((step.get("action") or {}).get("args") or {}).get("terminal")
            if self.terminals is not None and hasattr(self.terminals, "read_session"):
                try:
                    obs = self.terminals.read_session(name or "main")
                    observation = json.dumps(obs, ensure_ascii=False)[-1500:]
                except Exception as e:
                    observation = f"could not read the terminal: {e}"
        if observation is None:
            observation = "\n".join(step.get("evidence", [])[-3:]) or "(no evidence recorded)"
        verdict = await self._llm_json(
            task, VERIFIER_PROMPT,
            json.dumps({"step": step["title"], "criterion": step.get("expect"),
                        "observation": observation[:3000]}, ensure_ascii=False), max_tokens=200)
        if not isinstance(verdict, dict):
            return {"ok": False, "confidence": 0.0, "reason": "the check came back garbled", "repair_hint": None}
        ok = bool(verdict.get("ok")) and float(verdict.get("confidence", 0) or 0) >= 0.6
        return {"ok": ok, "confidence": float(verdict.get("confidence", 0) or 0),
                "reason": verdict.get("reason", ""), "repair_hint": verdict.get("repair_hint")}

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
        if note:
            await self._announce(task, note, "question")
        try:
            await asyncio.wait_for(event.wait(), self.limits.answer_reminder_secs)
        except asyncio.TimeoutError:
            pending["reminded"] = True
            await self._announce(task, f"[Task '{self._short_goal(task)}' is still waiting for your answer: "
                                       f"{pending['text']}]\n[Ask once more, briefly.]", "reminder")
            await event.wait()
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
            if task.get("_pattern_line"):
                note += f"Pattern noticed: {task['_pattern_line']}\n"
            task["status"] = "done"
        elif outcome == "cancelled":
            task["result_summary"] = "cancelled"
            last = next((s["title"] for s in reversed(task["plan"]) if s.get("status") == "done"), None)
            note = (f"[Task '{goal}' cancelled" + (f"; last completed: {last}" if last else "") +
                    ".]\n[Confirm in one short sentence.]")
            task["status"] = "cancelled"
        else:
            task["result_summary"] = reason or "it failed"
            failed = next((s for s in task["plan"] if s.get("status") == "failed"), None)
            where = f" at step {failed['n']} of {len(task['plan'])} ({failed['title']})" if failed else ""
            note = (f"[Task '{goal}' failed{where} after {mins:.1f} min: {reason}]\n"
                    f"[Tell the user plainly what went wrong and what you'd try next, in one or two sentences.]")
            task["status"] = "failed"
        task["pending"] = None
        task["finished_ts"] = datetime.now().isoformat(timespec="seconds")
        self._persist(task)
        grade = task.get("grade") or {}
        self.tm.finish_episode(task["id"], outcome, task["result_summary"], round(duration, 1),
                               agent_grade=grade.get("grade"), agent_grade_reason=grade.get("reason"))
        if task.get("flow_id"):
            self.tm.bump_flow_stats(task["flow_id"], outcome == "success")
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
