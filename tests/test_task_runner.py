import asyncio
import json
import os
import sqlite3

import pytest

from modules import tasks as T
from modules.task_memory import TaskMemory
from modules.tasks import DESTRUCTIVE_RE, Limits, LLMReply, TaskRunner, ToolBox


# --- ToolBox handler-merge (regression: web/space verbs were allow-listed but unreachable) ------

async def test_toolbox_add_handlers_makes_allowlisted_tool_reachable():
    tb = ToolBox({}, schemas=None)                      # no handlers yet
    blocked = await tb.call("search_web", {"query": "x"})
    assert blocked["success"] is False and "not available" in blocked["error"]

    async def fake_search(params):
        await params.result_callback({"success": True, "results": []})
    tb.add_handlers({"search_web": fake_search})        # merged after construction, like main.py does
    ok = await tb.call("search_web", {"query": "x"})
    assert ok["success"] is True                        # allow-listed AND now has a handler

    # a non-allow-listed tool stays blocked even with a handler present
    tb.add_handlers({"restart_self": fake_search})
    assert (await tb.call("restart_self", {}))["success"] is False


# --- fakes --------------------------------------------------------------------------

def step(n, title, kind="tool", tool="launch_app", args=None, verify="result", **extra):
    s = {"n": n, "title": title, "kind": kind, "expect": f"{title} happened", "verify": verify,
         "milestone": False, "risk": "none"}
    if kind == "tool":
        s["action"] = {"tool": tool, "args": args or {"app_name": "notepad"}}
    elif kind == "shell":
        s["action"] = {"command": args or "dir", "purpose": "list files"}
    elif kind == "delegate":
        s["action"] = {"project_dir": args or ".", "prompt": "add a docstring"}
    elif kind == "ask":
        s["action"] = {"question": "which file?"}
    else:
        s["action"] = None
    s.update(extra)
    return s


def plan(steps, **meta):
    p = {"summary": "open Notepad and save hello.txt", "family": "app-launch", "approach": "ui",
         "trivial": False, "est_minutes": 1, "same_as_match": True, "steps": steps,
         "suggested_flow_name": "hello note", "suggested_triggers": ["hello note", "make the hello note"]}
    p.update(meta)
    return p


class FakeLLM:
    """Scripted by which system prompt is used."""

    def __init__(self, plans=None, verdicts=None, agent_script=None, plan_delay=0.0, rethinks=None):
        self.plans = list(plans or [])
        self.verdicts = list(verdicts or [])
        self.agent_script = list(agent_script or [])
        self.rethinks = list(rethinks or [])
        self.plan_delay = plan_delay
        self.calls = []
        self.planner_inputs = []
        self.rethink_inputs = []

    async def __call__(self, messages, *, tools=None, json_mode=False, max_tokens=900, temperature=0.2):
        system = messages[0]["content"]
        self.calls.append(system[:25])
        if "second thoughts" in system[:60]:
            self.rethink_inputs.append(json.loads(messages[1]["content"]))
            r = self.rethinks.pop(0) if self.rethinks else {}
            return LLMReply(json.dumps(r), [], 200, 80)
        if system.startswith("You are the planner"):
            self.planner_inputs.append(json.loads(messages[1]["content"].split("\n\nYour previous plan")[0]))
            await asyncio.sleep(self.plan_delay)
            p = self.plans.pop(0) if self.plans else plan([step(1, "Open Notepad")])
            return LLMReply(json.dumps(p), [], 100, 50)
        if system.startswith("You verify one step"):
            v = self.verdicts.pop(0) if self.verdicts else {"ok": True, "confidence": 0.9, "reason": "done"}
            return LLMReply(json.dumps(v), [], 50, 10)
        if system.startswith("Summarize what a task"):
            return LLMReply("Opened Notepad and saved the note. All good.", [], 40, 20)
        if system.startswith("The user was asked"):
            return LLMReply(json.dumps({"decision": "yes"}), [], 10, 5)
        if system.startswith("You are the hands"):
            item = self.agent_script.pop(0) if self.agent_script else \
                [("step_done", {"summary": "ok", "evidence": "saw it"})]
            if isinstance(item, str):
                return LLMReply(item, [], 30, 10)
            return LLMReply(None, [{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(item)],
                            30, 10)
        return LLMReply("{}", [], 1, 1)


class FakeTerminals:
    def __init__(self):
        self.started = []
        self.killed = []
        self.job_result = {"status": "done", "result": "Added the docstring.", "cost": 0.12}

    def list_sessions(self):
        return []

    def read_session(self, name):
        return {"success": True, "output": "PS> dir\nhello.txt"}

    async def start_claude(self, **kw):
        self.started.append(kw)
        return {"success": True, "status": "started", "job": kw["job_name"]}

    async def wait_job(self, name, timeout):
        await asyncio.sleep(0.01)
        return self.job_result

    def kill_job(self, name):
        self.killed.append(name)


def make_handlers(tmp_path, calls):
    async def launch_app(params):
        calls.append(("launch_app", dict(params.arguments)))
        await params.result_callback({"success": True, "launched": params.arguments.get("app_name")})

    async def create_file(params):
        p = tmp_path / os.path.basename(params.arguments["file_path"])
        p.write_text(params.arguments.get("content", ""))
        calls.append(("create_file", str(p)))
        await params.result_callback({"success": True, "path": str(p)})

    async def look_at_screen(params):
        await params.result_callback({"success": True, "description": "Notepad is open with hello world"})

    async def focus_window(params):  # slow tool, for cancellation tests
        await asyncio.sleep(3)
        await params.result_callback({"success": True})

    async def type_text(params):  # returns nothing
        return None

    async def press_keys(params):  # raises
        raise RuntimeError("kaboom")

    async def remember(params):  # NOT in the allow list
        await params.result_callback({"success": True})

    return {"launch_app": launch_app, "create_file": create_file, "look_at_screen": look_at_screen,
            "focus_window": focus_window, "type_text": type_text, "press_keys": press_keys,
            "remember": remember}


def make_runner(tmp_path, llm, announces, *, limits=None, awake=True, quiet=None, shell=None,
                terminals=None, free_range=False, **extra):
    extra.setdefault("test_command", lambda project_dir: None)   # hermetic: never runs a real suite
    tmem = TaskMemory(sqlite3.connect(":memory:"))
    calls = []
    toolbox = ToolBox(make_handlers(tmp_path, calls))
    shell_calls = []

    def fake_shell(command):
        shell_calls.append(command)
        return {"success": True, "returncode": 0, "stdout": "ok", "stderr": ""}

    async def describe(looking_for=None, monitor_index=None):
        return f"The screen shows Notepad. ({looking_for})"

    runner = TaskRunner(task_memory=tmem, toolbox=toolbox, llm_call=llm, terminals=terminals or FakeTerminals(),
                        limits=limits or Limits(plan_wait_secs=2, answer_reminder_secs=5),
                        log=lambda m: None, execute_shell=shell or fake_shell, describe_screen=describe,
                        resolve_path=lambda p: str(tmp_path / os.path.basename(p)),
                        free_range=free_range, **extra)

    async def announce(text):
        announces.append(text)

    runner.attach_voice(announce=announce, quiet_note=quiet, is_awake=lambda: awake)
    runner.shell_calls = shell_calls
    return runner, tmem, calls


async def wait_status(runner, tid, *statuses, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        t = runner.tasks[tid]
        if t["status"] in statuses:
            return t
        await asyncio.sleep(0.01)
    raise AssertionError(f"task stuck in {runner.tasks[tid]['status']}")


TERMINAL = ("done", "failed", "cancelled", "interrupted")


async def wait_pending(runner, tid, kind, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        t = runner.tasks[tid]
        if t["status"] == "awaiting_user" and (t.get("pending") or {}).get("kind") == kind:
            return t
        if t["status"] in TERMINAL:
            raise AssertionError(f"task ended {t['status']} before pending {kind}: {t.get('result_summary')}")
        await asyncio.sleep(0.01)
    raise AssertionError(f"no pending {kind}; task is {runner.tasks[tid]['status']}")


async def wait_announce(announces, needle, timeout=2.0):
    for _ in range(int(timeout / 0.01)):
        if any(needle in a for a in announces):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"no announce containing {needle!r}: {announces}")


# --- ToolBox ----------------------------------------------------------------------------

async def test_toolbox_edge_cases(tmp_path):
    calls = []
    box = ToolBox(make_handlers(tmp_path, calls))
    assert "remember" not in box.names and "launch_app" in box.names
    assert (await box.call("remember", {}))["success"] is False
    assert (await box.call("type_text", {}))["error"] == "type_text returned nothing"
    assert "kaboom" in (await box.call("press_keys", {}))["error"]
    assert "timed out" in (await box.call("focus_window", {}, timeout=0.05))["error"]
    ok = await box.call("launch_app", {"app_name": "notepad"})
    assert ok == {"success": True, "launched": "notepad"} and calls == [("launch_app", {"app_name": "notepad"})]
    names = [t["function"]["name"] for t in box.openai_tools(T.PRIVATE_TOOLS)]
    assert "shell" in names and "launch_app" in names and "remember" not in names


# --- happy paths ------------------------------------------------------------------------

async def test_novel_task_confirms_in_turn_then_runs(tmp_path):
    announces = []
    llm = FakeLLM(plans=[plan([step(1, "Open Notepad"),
                               step(2, "Save hello.txt", tool="create_file",
                                    args={"file_path": "desktop\\hello.txt", "content": "hello"}, verify="file")])])
    runner, tmem, calls = make_runner(tmp_path, llm, announces)
    first = await runner.start("open notepad and save hello.txt on the desktop")
    assert first["status"] == "needs_your_ok" and "New one for me" in first["say"]
    tid = first["task_id"]
    await wait_status(runner, tid, "awaiting_user")
    assert announces == []                                       # asked in-turn, no announce round-trip
    res = await runner.answer("yes please")
    assert res["understood"] == "yes"
    t = await wait_status(runner, tid, *TERMINAL)
    assert t["status"] == "done"
    assert [c[0] for c in calls] == ["launch_app", "create_file"]
    assert (tmp_path / "hello.txt").read_text() == "hello"
    assert len(announces) == 1 and "finished after" in announces[0] and "status: success" in announces[0]
    row = tmem.get_task(tid)
    assert row["outcome"] == "success" and [s["status"] for s in row["plan"]] == ["done", "done"]
    assert row["llm_calls"] >= 2 and row["cost_usd"] > 0
    assert runner.last_completed_task_id == tid


async def test_seen_task_just_does_it(tmp_path):
    announces = []
    llm = FakeLLM(plans=[plan([step(1, "Open Notepad")]), plan([step(1, "Open Notepad")])])
    runner, tmem, calls = make_runner(tmp_path, llm, announces)
    first = await runner.start("open notepad")
    await runner.answer("yes")
    await wait_status(runner, first["task_id"], *TERMINAL)
    second = await runner.start("open notepad")
    assert second["status"] == "started" and "worked last time" in second["say"]
    t = await wait_status(runner, second["task_id"], *TERMINAL)
    assert t["status"] == "done" and len(calls) == 2


async def test_a_task_that_got_done_hands_its_whole_goal_and_result_to_on_result(tmp_path):
    # what she wanted is now done: main.py closes the wishes it settles (SelfEngine.settle_wishes)
    got = []
    llm = FakeLLM(plans=[plan([step(1, "Open Notepad")]), plan([step(1, "Open Notepad")])])
    runner, tmem, calls = make_runner(tmp_path, llm, [],
                                      on_result=lambda goal, summary: got.append((goal, summary)))
    goal = "open notepad and keep it open for the whole afternoon so I can jot things down"
    first = await runner.start(goal)
    await runner.answer("yes")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "done" and len(got) == 1
    assert got[0][0] == goal and got[0][1] == t["result_summary"]
    # a cancelled one settles nothing
    second = await runner.start("rename every photo in the pictures folder by the date it was taken")
    await wait_status(runner, second["task_id"], "awaiting_user")
    await runner.answer("no, don't")
    assert (await wait_status(runner, second["task_id"], *TERMINAL))["status"] == "cancelled"
    assert len(got) == 1


async def test_no_cancels_the_plan(tmp_path):
    announces = []
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(), announces)
    first = await runner.start("open notepad")
    await wait_status(runner, first["task_id"], "awaiting_user")
    await runner.answer("no, don't")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "cancelled" and calls == []
    assert "cancelled" in announces[-1]
    assert tmem.get_task(first["task_id"])["outcome"] == "cancelled"


async def test_unclear_answer_keeps_waiting(tmp_path):
    announces = []

    class NoLLM(FakeLLM):
        async def __call__(self, messages, **kw):
            if messages[0]["content"].startswith("The user was asked"):
                return LLMReply(json.dumps({"decision": "other"}), [], 1, 1)
            return await super().__call__(messages, **kw)

    runner, tmem, calls = make_runner(tmp_path, NoLLM(), announces)
    first = await runner.start("open notepad")
    await wait_status(runner, first["task_id"], "awaiting_user")
    res = await runner.answer("the weather is nice")
    assert res["success"] is False and runner.tasks[first["task_id"]]["status"] == "awaiting_user"
    await runner.answer("yes")
    assert (await wait_status(runner, first["task_id"], *TERMINAL))["status"] == "done"


async def test_slow_plan_returns_started_then_announces_question(tmp_path):
    announces = []
    llm = FakeLLM(plan_delay=0.3)
    runner, tmem, calls = make_runner(tmp_path, llm, announces,
                                      limits=Limits(plan_wait_secs=0.05, answer_reminder_secs=5))
    first = await runner.start("open notepad")
    assert first["status"] == "started"
    await wait_status(runner, first["task_id"], "awaiting_user")
    await wait_announce(announces, "call answer_task")
    assert any("planned" in a for a in announces)
    await runner.answer("yes")
    assert (await wait_status(runner, first["task_id"], *TERMINAL))["status"] == "done"


# --- shell gate --------------------------------------------------------------------------

async def test_shell_step_pauses_and_runs_only_after_yes(tmp_path):
    announces = []
    llm = FakeLLM(plans=[plan([step(1, "Count files", kind="shell", args="dir C:\\Users\\me\\Downloads")])])
    runner, tmem, calls = make_runner(tmp_path, llm, announces)
    first = await runner.start("count the files in downloads")
    assert first["status"] == "needs_your_ok"
    await runner.answer("yes")
    t = await wait_pending(runner, first["task_id"], "confirm_command")
    assert runner.shell_calls == []
    assert any("wants to run `dir" in a for a in announces)
    await runner.answer("go ahead")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "done" and runner.shell_calls == ["dir C:\\Users\\me\\Downloads"]


async def test_shell_step_is_verified_by_its_own_output_not_a_terminal(tmp_path):
    # 2026-09-26: a shell step planned with verify "terminal" was checked against a terminal
    # called "main" that didn't exist — the command had worked, the step failed, and the repair
    # opened a visible terminal
    class Terms(FakeTerminals):
        def read_session(self, name):
            return {"success": False, "error": f"no terminal named '{name}'"}

    class LLM(FakeLLM):
        observations = []

        async def __call__(self, messages, **kw):
            if messages[0]["content"].startswith("You verify one step"):
                self.observations.append(json.loads(messages[1]["content"])["observation"])
            return await super().__call__(messages, **kw)

    announces = []
    llm = LLM(plans=[plan([step(1, "Inspect the folder", kind="shell", args="Get-Item x", verify="terminal")])])
    runner, tmem, calls = make_runner(tmp_path, llm, announces, terminals=Terms())
    first = await runner.start("inspect the folder")
    await runner.answer("yes")
    await wait_pending(runner, first["task_id"], "confirm_command")
    await runner.answer("yes")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "done"
    assert llm.observations and "rc=0 ok" in llm.observations[0]
    assert "no terminal" not in llm.observations[0]


async def test_goal_already_done_makes_the_rest_of_the_plan_moot_not_a_failure(tmp_path):
    # 2026-09-26: step 1 (open a terminal) went on to delete the folder; step 2 "inspect before
    # deleting" then failed because the folder was gone — the task reported failure after succeeding
    announces = []
    llm = FakeLLM(plans=[plan([step(1, "Open Notepad"), step(2, "Inspect it"), step(3, "Delete it")])],
                  verdicts=[{"ok": True, "confidence": 0.9, "reason": "folder already gone", "goal_done": True}])
    runner, tmem, calls = make_runner(tmp_path, llm, announces)
    first = await runner.start("delete the folder")
    await runner.answer("yes")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    row = tmem.get_task(first["task_id"])
    assert t["status"] == "done" and row["outcome"] == "success"             # not "partial"
    assert [s["status"] for s in row["plan"]] == ["done", "not_needed", "not_needed"]
    assert len(calls) == 1                                          # steps 2-3 never ran


async def test_shell_step_declined_fails_gracefully(tmp_path):
    announces = []
    llm = FakeLLM(plans=[plan([step(1, "Open Notepad"), step(2, "Wipe temp", kind="shell",
                                                                 args="Remove-Item -Recurse C:\\temp")])])
    runner, tmem, calls = make_runner(tmp_path, llm, announces)
    first = await runner.start("open notepad then wipe temp")
    await runner.answer("yes")
    t = await wait_pending(runner, first["task_id"], "confirm_command")
    assert t["pending"]["payload"]["destructive"] is True
    assert any("DESTRUCTIVE" in a for a in announces)
    await runner.answer("no")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "failed" and "declined" in t["result_summary"] and runner.shell_calls == []
    assert t["plan"][0]["status"] == "done" and t["plan"][1]["status"] == "failed"


def test_destructive_lint():
    assert DESTRUCTIVE_RE.search("Remove-Item -Recurse -Force C:\\x")
    assert DESTRUCTIVE_RE.search("git push origin main --force")
    assert DESTRUCTIVE_RE.search("git reset --hard HEAD~1")
    assert not DESTRUCTIVE_RE.search("Get-ChildItem C:\\x | Measure-Object")
    assert not DESTRUCTIVE_RE.search("git status")


# --- agent steps, verification, repair, loops, budgets ---------------------------------

async def test_agent_step_uses_tools_then_step_done(tmp_path):
    announces = []
    llm = FakeLLM(plans=[plan([step(1, "Find and open Notepad", kind="agent", verify="screen")])],
                  agent_script=[[("look_at_screen", {}), ("launch_app", {"app_name": "notepad"})],
                                [("step_done", {"summary": "opened", "evidence": "Notepad visible"})]])
    runner, tmem, calls = make_runner(tmp_path, llm, announces)
    first = await runner.start("open notepad")
    await wait_status(runner, first["task_id"], "awaiting_user")
    await runner.answer("yes")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "done" and [c[0] for c in calls] == ["launch_app"]
    assert any("Notepad visible" in e for e in t["plan"][0]["evidence"])


async def test_verify_failure_then_repair(tmp_path):
    announces = []
    llm = FakeLLM(plans=[plan([step(1, "Open Notepad")])],
                  verdicts=[{"ok": False, "confidence": 0.9, "reason": "no window", "repair_hint": "launch again"},
                            {"ok": True, "confidence": 0.9, "reason": "window is up"}],
                  agent_script=[[("launch_app", {"app_name": "notepad"})],
                                [("step_done", {"summary": "opened", "evidence": "second try worked"})]])
    runner, tmem, calls = make_runner(tmp_path, llm, announces)
    first = await runner.start("open notepad")
    await runner.answer("yes")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "done" and t["plan"][0]["attempts"] == 2 and t["_repairs"] == 1
    assert len(calls) == 2


async def test_verify_failure_exhausts_repairs(tmp_path):
    announces = []
    llm = FakeLLM(plans=[plan([step(1, "Open Notepad")])],
                  verdicts=[{"ok": False, "confidence": 0.9, "reason": "no window"},
                            {"ok": False, "confidence": 0.9, "reason": "still no window"}])
    runner, tmem, calls = make_runner(tmp_path, llm, announces)
    first = await runner.start("open notepad")
    await runner.answer("yes")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "failed" and "still no window" in t["result_summary"]
    assert "failed at step 1 of 1" in announces[-1]


async def test_loop_detector(tmp_path):
    announces = []
    llm = FakeLLM(plans=[plan([step(1, "Poke at it", kind="agent")])],
                  agent_script=[[("launch_app", {"app_name": "x"})], [("launch_app", {"app_name": "x"})]])
    runner, tmem, calls = make_runner(tmp_path, llm, announces)
    first = await runner.start("poke")
    await runner.answer("yes")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "failed" and "repeating" in t["result_summary"]


async def test_budget_limit(tmp_path):
    announces = []
    llm = FakeLLM(plans=[plan([step(i, f"Step {i}") for i in range(1, 6)])])
    runner, tmem, calls = make_runner(tmp_path, llm, announces,
                                      limits=Limits(plan_wait_secs=2, answer_reminder_secs=5, max_llm_calls=3))
    first = await runner.start("do five things")
    await runner.answer("yes")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "failed" and "safety limit" in t["result_summary"]


async def test_invalid_plan_retried_then_accepted(tmp_path):
    announces = []
    bad = plan([step(1, "Nuke", tool="delete_everything")])
    llm = FakeLLM(plans=[bad, plan([step(1, "Open Notepad")])])
    runner, tmem, calls = make_runner(tmp_path, llm, announces)
    first = await runner.start("open notepad")
    assert first["status"] == "needs_your_ok"
    assert any(e["event"] == "plan_invalid" for e in runner.tasks[first["task_id"]]["log"])
    await runner.answer("yes")
    assert (await wait_status(runner, first["task_id"], *TERMINAL))["status"] == "done"


async def test_plan_invalid_twice_fails_in_turn(tmp_path):
    announces = []
    bad = plan([step(1, "Nuke", tool="delete_everything")])
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[bad, bad]), announces)
    first = await runner.start("open notepad")
    assert first["status"] == "failed" and "couldn't work out a plan" in first["say"]
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert tmem.get_task(first["task_id"])["outcome"] == "failed"


# --- cancel, status, milestones, quiet notes ---------------------------------------------

async def test_cancel_mid_step(tmp_path):
    announces = []
    llm = FakeLLM(plans=[plan([step(1, "Focus it", tool="focus_window", args={"title": "x"}),
                               step(2, "Open Notepad")])])
    runner, tmem, calls = make_runner(tmp_path, llm, announces)
    first = await runner.start("focus then open")
    await runner.answer("yes")
    await wait_status(runner, first["task_id"], "running")
    await asyncio.sleep(0.05)
    status = runner.status()["tasks"][0]
    assert status["current"] == "Focus it" and status["step"] == "0/2"
    res = await runner.cancel(reason="user said stop")
    assert res["success"] and res["cancelled"] == first["task_id"]
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "cancelled" and calls == [] and "cancelled" in announces[-1]
    assert tmem.get_task(first["task_id"])["outcome"] == "cancelled"
    assert (await runner.cancel())["success"] is False


async def test_milestones_throttled_and_quiet_when_asleep(tmp_path):
    announces, quiet = [], []

    async def quiet_note(text):
        quiet.append(text)

    steps = [step(i, f"Step {i}", milestone=True) for i in range(1, 5)]
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan(steps)]), announces, quiet=quiet_note)
    first = await runner.start("four steps")
    await runner.answer("yes")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    progress = [a for a in announces if "progress" in a]
    assert len(progress) == 1                                    # 30 s gap: only the first milestone spoke
    assert any(e["event"] == "milestone_dropped" for e in t["log"])

    announces2 = []
    runner2, _, _ = make_runner(tmp_path, FakeLLM(plans=[plan(steps)]), announces2, awake=False, quiet=quiet_note)
    first2 = await runner2.start("four steps")
    await runner2.answer("yes")
    await wait_status(runner2, first2["task_id"], *TERMINAL)
    assert not any("progress" in a for a in announces2) and any("progress" in q for q in quiet)
    assert any("finished" in a for a in announces2)              # done still wakes her


# --- ask step, delegate step, flows, persistence ------------------------------------

async def test_ask_step_pauses_for_clarification(tmp_path):
    announces = []
    llm = FakeLLM(plans=[plan([step(1, "Which file", kind="ask"), step(2, "Open Notepad")])])
    runner, tmem, calls = make_runner(tmp_path, llm, announces)
    first = await runner.start("open the file")
    await runner.answer("yes")
    t = await wait_pending(runner, first["task_id"], "clarify")
    assert any("has a question" in a for a in announces)
    await runner.answer("the notes file")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "done" and "User said: the notes file" in t["details"]


async def test_delegate_step_runs_and_grades(tmp_path):
    announces = []
    terminals = FakeTerminals()
    llm = FakeLLM(plans=[plan([step(1, "Hand to Claude Code", kind="delegate", args=str(tmp_path))],
                              family="repo-edit", delegate=True)])
    runner, tmem, calls = make_runner(tmp_path, llm, announces, terminals=terminals)
    first = await runner.start("add a docstring to utilities.py in the repo")
    await runner.answer("yes")                                   # plan confirmed -> no second delegate confirm
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "done" and terminals.started[0]["announce_done"] is False
    assert t["grade"]["grade"] == "good"
    assert "grade GOOD" in announces[-1]
    row = tmem.get_task(first["task_id"])
    assert row["agent_grade"] == "good" and row["delegated_to"] == "claude_code"


async def test_delegate_bad_grade_fails_task(tmp_path):
    announces = []
    terminals = FakeTerminals()
    terminals.job_result = {"status": "failed (1)", "result": "boom", "cost": 0.0}
    llm = FakeLLM(plans=[plan([step(1, "Hand to Claude Code", kind="delegate", args=str(tmp_path))])])
    runner, tmem, calls = make_runner(tmp_path, llm, announces, terminals=terminals)
    first = await runner.start("fix the bug")
    await runner.answer("yes")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "failed" and "didn't check out" in t["result_summary"]


async def test_run_flow_skips_planning_and_records_run(tmp_path):
    announces = []
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(), announces)
    fid = tmem.save_flow("hello note", [step(1, "Open Notepad"), step(2, "Save hello.txt", tool="create_file",
                                                                    args={"file_path": "hello.txt"}, verify="file")],
                         trigger_phrases=["make the hello note"], family="app-launch")
    res = await runner.run_flow("make the hello note")
    assert res["status"] == "started" and "On it" in res["say"]
    t = await wait_status(runner, res["task_id"], *TERMINAL)
    assert t["status"] == "done" and not any("planner" in c for c in runner.llm_call.calls)
    assert tmem.get_flow(fid)["run_count"] == 1 and tmem.get_flow(fid)["success_count"] == 1
    assert tmem.get_task(res["task_id"])["flow_id"] == fid
    missing = await runner.run_flow("no such flow")
    assert missing["success"] is False and missing["flows"] == ["hello note"]


async def test_flow_offer_after_second_success(tmp_path):
    announces = []
    plans = [plan([step(1, "Focus OBS"), step(2, "Start scene")], family="streaming-setup")]
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=plans), announces)
    first = await runner.start("focus obs and start the gaming scene")
    await runner.answer("yes")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "done" and "Flow suggestion" in announces[-1] and "hello note" in announces[-1]
    assert t["_flow_offer"]["name"] == "hello note"


async def test_two_running_tasks_max(tmp_path):
    announces = []
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(), announces)
    a = await runner.start("one")
    b = await runner.start("two")
    c = await runner.start("three")
    assert a["status"] == b["status"] == "needs_your_ok" and c["success"] is False


# --- regressions from the 2026-09-20 prime-number runs --------------------------------------

async def test_step_exception_survives_a_swallowing_spawner(tmp_path):
    # pipecat's create_task logs and swallows exceptions; a step that raised once left the step
    # "running" and the task reported success having done nothing
    announces = []
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([step(1, "Open Notepad"),
                                                                      step(2, "Open Notepad again")])]),
                                      announces)

    def swallowing_spawn(coro):
        async def run():
            try:
                return await coro
            except asyncio.CancelledError:
                raise
            except Exception:
                return None
        return asyncio.ensure_future(run())

    runner.attach_voice(announce=runner._announce_cb, spawn=swallowing_spawn)

    async def boom(task, step):
        raise T.BudgetExceeded("ran longer than 20 minutes")
    runner._execute_and_verify = boom
    first = await runner.start("open notepad")
    await runner.answer("yes")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "failed" and "safety limit" in t["result_summary"]
    assert tmem.get_task(first["task_id"])["outcome"] == "failed"


async def test_waiting_for_the_user_does_not_count_against_wall_clock(tmp_path):
    announces = []
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(), announces,
                                      limits=Limits(plan_wait_secs=2, answer_reminder_secs=5, max_wall_secs=0.3))
    first = await runner.start("open notepad")
    await wait_status(runner, first["task_id"], "awaiting_user")
    await asyncio.sleep(0.5)                       # he takes longer to answer than the whole budget
    await runner.answer("yes")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "done" and calls == [("launch_app", {"app_name": "notepad"})]


def test_resolve_dir_is_relative_to_home_like_create_file():
    from modules.file_operations import resolve_path
    from modules.terminals import resolve_dir
    home = os.path.expanduser("~")
    assert resolve_dir("desktop/prime_numbers_app").lower() == \
        os.path.dirname(resolve_path("desktop/prime_numbers_app/x.py")).lower()
    assert resolve_dir(None) == os.path.normpath(home)
    assert resolve_dir("~/Documents") == os.path.normpath(os.path.join(home, "Documents"))
    assert resolve_dir("C:\Temp") == "C:\Temp"


# --- free range (AIVA_TASK_FREE_RANGE, user decision 2026-09-22) ------------------------------

def test_free_range_decision_skips_confirmation_except_sending():
    from modules.task_memory import decide_autonomy
    novel = {"level": "novel", "matches": []}
    risky = plan([step(1, "Build it", kind="delegate"), step(2, "Wipe", kind="shell", risk="irreversible")])
    assert decide_autonomy("g", risky, novel)["mode"] == "confirm_plan"
    d = decide_autonomy("g", risky, novel, free_range=True)
    assert d["mode"] == "just_do" and d["say"].startswith("On it")
    assert decide_autonomy("g", risky, {"level": "attempted", "last_failure": "x"}, free_range=True)["mode"] == "just_do"
    sending = plan([step(1, "Mail him", tool="send_email")])
    assert decide_autonomy("g", sending, novel, free_range=True)["mode"] == "confirm_plan"
    assert decide_autonomy("g", plan([step(1, "x")]), novel, confirm_first=True, free_range=True)["mode"] == "confirm_plan"


async def test_free_range_starts_and_delegates_without_asking(tmp_path):
    announces = []
    terms = FakeTerminals()
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([step(1, "Write the script", kind="delegate",
                                                                          args=str(tmp_path))])]),
                                      announces, terminals=terms)
    runner.free_range = True
    first = await runner.start("code me a prime number script")
    assert first["status"] == "started"
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "done" and len(terms.started) == 1
    assert not any("answer_task" in a for a in announces)


def test_free_range_defaults_off_and_reads_env(monkeypatch, tmp_path):
    monkeypatch.delenv("AIVA_TASK_FREE_RANGE", raising=False)
    assert make_runner(tmp_path, FakeLLM(), [], free_range=None)[0].free_range is False
    monkeypatch.setenv("AIVA_TASK_FREE_RANGE", "1")
    assert make_runner(tmp_path, FakeLLM(), [], free_range=None)[0].free_range is True


async def test_delegate_logs_claude_link_for_the_dashboard(tmp_path):
    announces = []
    terms = FakeTerminals()
    terms.job_result = {"status": "done", "result": "Wrote primes.py.", "cost": 0.2,
                        "claude_session_id": "0b1c2d3e-0000-4000-8000-000000000001",
                        "log_path": str(tmp_path / "t-x-s1.log")}
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([step(1, "Write it", kind="delegate",
                                                                          args=str(tmp_path))])]),
                                      announces, terminals=terms, free_range=True)
    first = await runner.start("code me a prime script")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    ev = [e for e in tmem.get_task(first["task_id"])["log"] if e["event"] == "claude"]
    assert t["status"] == "done" and len(ev) == 1
    d = json.loads(ev[0]["detail"])
    assert d["session"] == "0b1c2d3e-0000-4000-8000-000000000001" and d["cost"] == 0.2
    assert d["job"].endswith("-s1") and d["status"] == "done" and d["log"]


async def test_the_done_note_carries_what_the_agent_reported(tmp_path):
    # 2026-10-17: "take a look and tell me what you'd change" — she only had a two-sentence summary
    announces = []
    terms = FakeTerminals()
    terms.job_result = {"status": "done", "cost": 0.2,
                        "result": "README gaps: no Saga section. Add the house and both launchers."}
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([step(1, "Review it", kind="delegate",
                                                                          args=str(tmp_path))])]),
                                      announces, terminals=terms, free_range=True)
    first = await runner.start("look at the readme and tell me what you'd change")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "done"
    assert "What Claude Code reported, in its own words: README gaps: no Saga section." in announces[-1]


async def test_delegate_creates_a_new_project_folder_inside_home_only(tmp_path):
    new_dir = tmp_path / "prime_project"                      # tmp_path lives under the profile
    terms = FakeTerminals()
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([step(1, "Write it", kind="delegate",
                                                                          args=str(new_dir))])]),
                                      [], terminals=terms, free_range=True)
    t = await wait_status(runner, (await runner.start("code me a prime script"))["task_id"], *TERMINAL)
    assert t["status"] == "done" and new_dir.is_dir() and len(terms.started) == 1

    outside = "C:\aiva_should_never_create_this"
    terms2 = FakeTerminals()
    runner2, _, _ = make_runner(tmp_path, FakeLLM(plans=[plan([step(1, "Write it", kind="delegate",
                                                                   args=outside)])]),
                                [], terminals=terms2, free_range=True)
    t2 = await wait_status(runner2, (await runner2.start("code it"))["task_id"], *TERMINAL)
    assert t2["status"] == "failed" and not os.path.exists(outside) and terms2.started == []


# --- second thoughts (2026-10-13: twice sent Claude Code after a "Miso renderer" that never existed) -----

MISO_REPORT = ("I found nothing called a \"Miso renderer\". Miso is the name of Aiva's **desktop pixel cat**. "
               "The thing that actually renders an avatar is the **Live2D renderer**.\n\n"
               + "## Details\n" + "- a long middle section about the repo\n" * 120 +
               "\nWhich one did you mean? And should I make the safety commit before starting?")


def miso_runner(tmp_path, rethinks, plans, announces, remembered, *, free_range=True, result=MISO_REPORT):
    terms = FakeTerminals()
    terms.job_result = {"status": "done", "result": result, "cost": 0.77}
    llm = FakeLLM(plans=plans, rethinks=rethinks)
    runner, tmem, calls = make_runner(tmp_path, llm, announces, terminals=terms, free_range=free_range)
    runner.remember = lambda fact, category: remembered.append((fact, category))
    return runner, tmem, llm, terms


async def test_rethink_reads_the_report_learns_and_retries(tmp_path):
    announces, remembered = [], []
    old = "Aiva is implementing the witness in the existing Miso renderer."
    rethink = {"what_went_wrong": "There is no Miso renderer; Miso is the desktop cat.",
               "beliefs": [{"fact": "Miso is the desktop pixel cat, not a renderer.",
                            "evidence": "Miso is the name of Aiva's desktop pixel cat",
                            "replaces": [old, "a line she never held"]},
                           {"fact": "The constellation lives in stars.py.",          # not in the report
                            "evidence": "the constellation witness is in stars.py",
                            "replaces": ["Frank likes tea."]}],
               "lesson": "Check a component exists before asking an agent to rebuild it.",
               "next": "retry", "revised_goal": "Build the constellation witness on the Live2D avatar"}
    runner, tmem, llm, terms = miso_runner(
        tmp_path, [rethink],
        [plan([step(1, "Rebuild it in the Miso renderer", kind="delegate", args=str(tmp_path))]),
         plan([step(1, "Open Notepad")], summary="build it on the avatar")], announces, remembered)
    retired = []
    runner._facts = lambda: [old, "Frank likes tea."]
    runner.retire_fact = lambda fact: retired.append(fact) or "project"
    t = await wait_status(runner, (await runner.start("rebuild the witness in the Miso renderer"))["task_id"],
                          *TERMINAL)
    await runner.join(t["id"], timeout=10)                      # announces are spaced 3 s apart
    assert t["status"] == "done"
    # her second thoughts saw the WHOLE report — the start ("Miso is the cat"), not just the closing question
    seen = llm.rethink_inputs[0]
    assert "desktop pixel cat" in seen["agent_report"] and "Which one did you mean?" in seen["agent_report"]
    assert seen["grade"]["blocked"] == "question"
    # only the belief the report really states is kept
    assert remembered == [("Checked myself: Miso is the desktop pixel cat, not a renderer.", "learned")]
    assert any(e["event"] == "belief_unverified" for e in t["log"])
    # the old belief the PROVED fact replaces is retired; an unproved belief retires nothing
    assert seen["what_you_believe"] == [old, "Frank likes tea."] and retired == [old]
    assert any(e["event"] == "unlearned" for e in t["log"])
    row = tmem.get_task(t["id"])
    assert row["self_lesson"].startswith("Check a component exists")
    assert any("(you found this yourself)" in line for line in tmem.load_lessons())
    # the re-plan was told what happened and built on the corrected goal
    prev = llm.planner_inputs[1]
    assert prev["goal"] == "Build the constellation witness on the Live2D avatar"
    assert "Miso is the desktop cat" in prev["previous_attempt"]["what_went_wrong"]
    assert [(s["n"], s["status"]) for s in t["plan"]] == [(1, "failed"), (2, "done")]
    assert any("first try didn't work" in a for a in announces)
    assert "worked out why" in announces[-1]


async def test_rethink_puts_the_agents_question_to_the_user(tmp_path):
    announces, remembered = [], []
    rethink = {"what_went_wrong": "There is no Miso renderer.", "beliefs": [], "lesson": None,
               "next": "ask", "question": "Should the constellation live on your avatar or be its own widget?"}
    runner, tmem, llm, terms = miso_runner(
        tmp_path, [rethink],
        [plan([step(1, "Rebuild it", kind="delegate", args=str(tmp_path))]), plan([step(1, "Open Notepad")])],
        announces, remembered)
    first = await runner.start("rebuild the witness in the Miso renderer")
    t = await wait_pending(runner, first["task_id"], "clarify")
    assert "own widget" in t["pending"]["text"] and any("hit a snag" in a for a in announces)
    await runner.answer("on the avatar")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "done" and "User said: on the avatar" in t["details"]
    assert llm.planner_inputs[1]["previous_attempt"]["user_answer"] == "on the avatar"
    assert remembered == [] and tmem.get_task(t["id"])["self_lesson"] is None


async def test_rethink_happens_once_then_fails_with_what_she_worked_out(tmp_path):
    announces, remembered = [], []
    retry = {"what_went_wrong": "There is no Miso renderer.", "next": "retry", "revised_goal": "try the avatar"}
    deleg = plan([step(1, "Rebuild it", kind="delegate", args=str(tmp_path))])
    runner, tmem, llm, terms = miso_runner(tmp_path, [retry, retry], [deleg, deleg], announces, remembered)
    t = await wait_status(runner, (await runner.start("rebuild the witness"))["task_id"], *TERMINAL)
    assert t["status"] == "failed" and len(llm.rethink_inputs) == 1 and len(terms.started) == 2
    assert "step 2" in t["result_summary"] and "Claude Code stopped to ask" in t["result_summary"]
    assert "What I worked out: There is no Miso renderer." in t["result_summary"]


async def test_rethink_give_up_still_says_why(tmp_path):
    announces, remembered = [], []
    runner, tmem, llm, terms = miso_runner(
        tmp_path, [{"what_went_wrong": "The project folder is empty.", "next": "give_up"}],
        [plan([step(1, "Rebuild it", kind="delegate", args=str(tmp_path))])], announces, remembered)
    t = await wait_status(runner, (await runner.start("rebuild the witness"))["task_id"], *TERMINAL)
    assert t["status"] == "failed" and "The project folder is empty." in t["result_summary"]
    assert len(llm.planner_inputs) == 1


async def test_rethink_needs_a_yes_outside_free_range(tmp_path):
    announces, remembered = [], []
    retry = {"what_went_wrong": "There is no Miso renderer.", "next": "retry", "revised_goal": "use the avatar"}
    runner, tmem, llm, terms = miso_runner(
        tmp_path, [retry], [plan([step(1, "Rebuild it", kind="delegate", args=str(tmp_path))]),
                            plan([step(1, "Open Notepad")])], announces, remembered, free_range=False)
    first = await runner.start("rebuild the witness")
    await runner.answer("yes")                                    # the original plan
    t = runner.tasks[first["task_id"]]
    for _ in range(500):                                          # until the RE-plan is the one waiting
        if "didn't work out" in ((t.get("pending") or {}).get("text") or ""):
            break
        assert t["status"] not in TERMINAL, t.get("result_summary")
        await asyncio.sleep(0.01)
    assert t["status"] == "awaiting_user" and "didn't work out" in t["pending"]["text"]
    await runner.answer("no")
    t = await wait_status(runner, first["task_id"], *TERMINAL)
    assert t["status"] == "failed" and len(t["plan"]) == 1


# --- one job per coding goal (2026-10-13: four jobs for one renderer fix) --------------------------

async def test_back_to_back_coding_steps_on_one_project_run_as_one_job(tmp_path):
    announces = []
    terms = FakeTerminals()
    look = step(1, "Inspect the renderer", kind="delegate", args=str(tmp_path))
    look["action"].update(read_only=True, prompt="Inspect the renderer. Do not modify files yet.")
    build = step(2, "Fix the renderer gaps", kind="delegate", args=str(tmp_path) + os.sep)  # same dir, other spelling
    check = step(3, "Run the tests", kind="delegate", args=str(tmp_path), risk="reversible")
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([look, build, check])]), announces,
                                      terminals=terms, free_range=True)
    t = await wait_status(runner, (await runner.start("finish the witness"))["task_id"], *TERMINAL)
    assert t["status"] == "done" and len(terms.started) == 1
    job = terms.started[0]
    # the look-only part's "do not modify" wording doesn't turn the merged job read-only
    assert job["mode"] == "execute"
    assert "1. Inspect the renderer:" in job["prompt"] and "3. Run the tests:" in job["prompt"]
    only = t["plan"][0]
    assert len(t["plan"]) == 1 and only["title"] == "Fix the renderer gaps" and only["risk"] == "reversible"
    assert only["merged_from"] == ["Inspect the renderer", "Fix the renderer gaps", "Run the tests"]
    assert any(e["event"] == "merged" for e in t["log"])


def test_one_job_per_project_only_merges_neighbours_on_the_same_project():
    a = step(1, "Look at A", kind="delegate", args="desktop/a")
    a["action"]["read_only"] = True
    a2 = step(2, "Look again at A", kind="delegate", args="desktop/a")
    a2["action"]["read_only"] = True
    b = step(3, "Fix B", kind="delegate", args="desktop/b")
    tool = step(4, "Open Notepad")
    b2 = step(5, "Test B", kind="delegate", args="desktop/b")
    out = T.one_job_per_project([a, a2, b, tool, b2])
    assert [s["title"] for s in out] == ["Look at A", "Fix B", "Open Notepad", "Test B"]
    assert out[0]["action"]["read_only"] is True             # all look-only: still look-only
    assert T.one_job_per_project([b]) == [b]


async def test_a_later_job_on_the_same_project_continues_the_claude_session(tmp_path):
    announces = []
    terms = FakeTerminals()
    terms.job_result = {"status": "done", "result": "The renderer is aiva/modules/avatar_renderer.py.",
                        "cost": 0.1, "claude_session_id": "0b1c2d3e-0000-4000-8000-000000000001"}
    look = step(1, "Inspect the renderer", kind="delegate", args=str(tmp_path))
    look["action"]["read_only"] = True
    build = step(3, "Build it", kind="delegate", args=str(tmp_path))
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([look, step(2, "Open Notepad"), build])]),
                                      announces, terminals=terms, free_range=True)
    t = await wait_status(runner, (await runner.start("build the witness"))["task_id"], *TERMINAL)
    assert t["status"] == "done"
    assert [s["mode"] for s in terms.started] == ["plan", "execute"]
    assert terms.started[0]["resume_id"] is None
    assert terms.started[1]["resume_id"] == "0b1c2d3e-0000-4000-8000-000000000001"
    # the session remembers its own findings: no second-hand summary of them
    assert "What the earlier step" not in terms.started[1]["prompt"]
    assert any("continuing session" in e["detail"] for e in t["log"] if e["event"] == "delegate")


async def test_without_a_session_the_next_job_gets_the_earlier_report(tmp_path):
    terms = FakeTerminals()
    terms.job_result = {"status": "done", "result": "The renderer is aiva/modules/avatar_renderer.py.", "cost": 0.1}
    look = step(1, "Inspect the renderer", kind="delegate", args=str(tmp_path))
    look["action"]["read_only"] = True
    runner, tmem, calls = make_runner(
        tmp_path, FakeLLM(plans=[plan([look, step(2, "Open Notepad"), step(3, "Build it", kind="delegate",
                                                                          args=str(tmp_path))])]),
        [], terminals=terms, free_range=True)
    t = await wait_status(runner, (await runner.start("build the witness"))["task_id"], *TERMINAL)
    assert t["status"] == "done" and terms.started[1]["resume_id"] is None
    assert "What the earlier step (Inspect the renderer) found" in terms.started[1]["prompt"]
    assert "avatar_renderer.py" in terms.started[1]["prompt"]


# --- what the agent changed, seen in full (2026-10-13: a 1000-character tail hid every edit) -------

class EditingTerminals(FakeTerminals):
    """Claude Code jobs that really edit files before they report: one path (or None) per job."""

    def __init__(self, edits, result="Fixed the four renderer gaps; the suite passes."):
        super().__init__()
        self.edits = list(edits)
        self.job_result = {"status": "done", "result": result, "cost": 1.0,
                           "claude_session_id": "11111111-0000-4000-8000-000000000001"}

    async def wait_job(self, name, timeout):
        path = self.edits.pop(0) if self.edits else None
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write("# edited by the agent\n")
        return await super().wait_job(name, timeout)


def git_repo_with_his_work(root, n=120):
    """A repo whose `git status` lists the edited file FIRST, then n more of his uncommitted files —
    far past 1000 characters, like the real Aiva repo."""
    (root / ".git").mkdir()
    paths = ["aiva/modules/avatar_renderer.py"] + [f"aiva/modules/his_work_{i:03}.py" for i in range(n)]
    for p in paths:
        f = root / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x = 1\n", encoding="utf-8")
    status = "".join(f" M {p}\n" for p in paths)
    assert len(status) > 3000

    def git(project_dir, *args):
        args = [a for a in args if a != "-c" and not a.startswith("core.")]
        out = {"status": status, "rev-parse": "abc123\n"}.get(args[0], "")
        return {"success": True, "returncode": 0, "stdout": out, "stderr": ""}
    return git, root / paths[0]


async def test_snapshot_sees_an_edit_listed_before_the_last_1000_characters(tmp_path):
    git, edited = git_repo_with_his_work(tmp_path)
    terms = EditingTerminals([edited])
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([step(1, "Fix the renderer", kind="delegate",
                                                                          args=str(tmp_path))])]),
                                      [], terminals=terms, free_range=True, git=git)
    t = await wait_status(runner, (await runner.start("fix the renderer"))["task_id"], *TERMINAL)
    assert t["status"] == "done" and t["grade"]["grade"] == "good"
    assert t["_step_reports"][1]["agent_changed"] == ["aiva/modules/avatar_renderer.py"]
    # her own git bookkeeping isn't billed as tool calls
    assert t["tool_calls"] == 0 and not any(e["event"] == "shell" for e in t["log"])


async def test_what_changes_while_a_plan_mode_job_runs_is_not_the_agents(tmp_path):
    # 2026-10-17: a README review failed "for modifying tasks.py": runner fixes being made in his own
    # Claude Code session while it ran. Plan mode can't edit, so nothing that changed is the agent's.
    git, edited = git_repo_with_his_work(tmp_path)
    terms = EditingTerminals([edited])                  # someone edits the renderer mid-job
    look = step(1, "Review the readme", kind="delegate", args=str(tmp_path))
    look["action"].update(read_only=True, prompt="Review the README and say what to change.")
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([look])]), [], terminals=terms,
                                      free_range=True, git=git)
    t = await wait_status(runner, (await runner.start("review the readme"))["task_id"], *TERMINAL)
    assert terms.started[0]["mode"] == "plan"
    assert t["status"] == "done" and t["_step_reports"][1]["agent_changed"] == []
    assert t["grade"]["grade"] == "good"


async def test_a_repair_rerun_is_judged_on_everything_its_step_changed(tmp_path):
    # the 18:45 run: the first job did the work, the re-run found it done and changed nothing
    git, edited = git_repo_with_his_work(tmp_path)
    terms = EditingTerminals([edited, None])
    llm = FakeLLM(plans=[plan([step(1, "Fix the renderer", kind="delegate", args=str(tmp_path))])],
                  verdicts=[{"ok": False, "confidence": 0.9, "reason": "not evidenced", "repair_hint": "check"}],
                  agent_script=[[("delegate_to_claude", {"project_dir": str(tmp_path), "task": "make sure it's done",
                                                         "why": "it's code"})],
                                [("step_done", {"summary": "already in place", "evidence": "report"})]])
    runner, tmem, calls = make_runner(tmp_path, llm, [], terminals=terms, free_range=True, git=git)
    t = await wait_status(runner, (await runner.start("fix the renderer"))["task_id"], *TERMINAL)
    assert t["status"] == "done" and len(terms.started) == 2
    assert t["grade"]["grade"] == "good"                       # not "nothing in the project changed"
    assert t["_step_reports"][1]["agent_changed"] == ["aiva/modules/avatar_renderer.py"]
    assert terms.started[1]["resume_id"] == "11111111-0000-4000-8000-000000000001"


# --- she runs the project's tests herself ----------------------------------------------------------

def scripted_tests(*results):
    ran = []

    def runner(command, project_dir, timeout):
        ran.append(command)
        return results[min(len(ran) - 1, len(results) - 1)]
    runner.ran = ran
    return runner


GREEN = {"ran": True, "returncode": 0, "passed": True, "summary": "837 passed, 1 skipped", "failing": []}
RED = {"ran": True, "returncode": 1, "passed": False, "summary": "1 failed, 836 passed",
       "failing": ["FAILED tests/test_constellation.py::test_sky - assert 0 == 5"]}


# --- how hard a job thinks (2026-10-17: every job inherited his xhigh) -----------------------------

def test_job_effort_sizes_the_job_and_climbs_after_a_short_try():
    from modules.tasks import job_effort
    assert job_effort("low", read_only=True) == "low"
    assert job_effort(None, read_only=True) == "medium" and job_effort("bogus", read_only=False) == "high"
    assert job_effort("xhigh", read_only=True) == "high"            # a look never needs xhigh
    assert job_effort("max", read_only=False) == "high"             # max only when he forces it
    assert job_effort("medium", read_only=False, misses=1) == "high"
    assert job_effort("medium", read_only=False, misses=5) == "xhigh"
    assert job_effort("low", read_only=True, misses=5) == "high"
    assert job_effort("low", read_only=True, forced="xhigh") == "xhigh"   # AIVA_CLAUDE_EFFORT=<level>
    assert job_effort("low", read_only=True, forced="inherit") is None   # his own Claude Code setting
    assert job_effort("low", read_only=True, forced="auto") == "low"


def test_merged_job_thinks_as_hard_as_its_hardest_part(tmp_path):
    from modules.tasks import one_job_per_project
    a = step(1, "Look", kind="delegate", args=str(tmp_path))
    a["action"].update(effort="low", read_only=True)
    b = step(2, "Build it", kind="delegate", args=str(tmp_path))
    b["action"].update(effort="xhigh")
    assert one_job_per_project([a, b])[0]["action"]["effort"] == "xhigh"


async def test_the_planners_effort_reaches_claude_code_and_climbs_after_a_partial(tmp_path, monkeypatch):
    monkeypatch.delenv("AIVA_CLAUDE_EFFORT", raising=False)
    git, edited = git_repo_with_his_work(tmp_path)
    first = step(1, "Fix the renderer", kind="delegate", args=str(tmp_path))
    first["action"]["effort"] = "medium"
    between = step(2, "Open Notepad")
    again = step(3, "Finish the renderer", kind="delegate", args=str(tmp_path))
    again["action"]["effort"] = "medium"
    terms = EditingTerminals([edited, edited])
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([first, between, again])]), [],
                                      terminals=terms, free_range=True, git=git,
                                      test_command=lambda d: ["pytest"],
                                      test_runner=scripted_tests(GREEN, RED, GREEN))
    t = await wait_status(runner, (await runner.start("fix the renderer"))["task_id"], *TERMINAL)
    assert [s["effort"] for s in terms.started] == ["medium", "high"]   # partial → one level up
    assert any("effort high (up 1 after a short try)" in e["detail"] for e in t["log"] if e["event"] == "delegate")
    claude = [json.loads(e["detail"]) for e in t["log"] if e["event"] == "claude"]
    assert [c["effort"] for c in claude] == ["medium", "high"]


async def test_inherit_leaves_his_own_claude_code_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("AIVA_CLAUDE_EFFORT", "inherit")
    look = step(1, "Review the readme", kind="delegate", args=str(tmp_path))
    look["action"].update(read_only=True, effort="low")
    terms = FakeTerminals()
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([look])]), [], terminals=terms,
                                      free_range=True)
    await wait_status(runner, (await runner.start("review the readme"))["task_id"], *TERMINAL)
    assert terms.started[0]["effort"] is None


async def test_failing_tests_she_runs_cap_the_grade(tmp_path):
    git, edited = git_repo_with_his_work(tmp_path)
    tests = scripted_tests(GREEN, RED)
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([step(1, "Fix the renderer", kind="delegate",
                                                                          args=str(tmp_path))])]),
                                      [], terminals=EditingTerminals([edited]), free_range=True, git=git,
                                      test_command=lambda d: ["pytest"], test_runner=tests)
    t = await wait_status(runner, (await runner.start("fix the renderer"))["task_id"], *TERMINAL)
    assert len(tests.ran) == 2                                 # before the job, after it
    assert t["grade"]["grade"] == "partial" and "tests are failing" in t["grade"]["reason"]
    ev = t["plan"][0]["evidence"]
    assert any(e.startswith("tests Aiva ran herself: 1 failed, 836 passed") and "test_sky" in e for e in ev)
    assert [e["detail"] for e in t["log"] if e["event"] == "tests"] == \
        ["before: 837 passed, 1 skipped", "after: 1 failed, 836 passed — FAILED tests/test_constellation.py::test_sky - assert 0 == 5"]


async def test_tests_he_already_had_failing_are_not_blamed_on_the_agent(tmp_path):
    git, edited = git_repo_with_his_work(tmp_path)
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([step(1, "Fix the renderer", kind="delegate",
                                                                          args=str(tmp_path))])]),
                                      [], terminals=EditingTerminals([edited]), free_range=True, git=git,
                                      test_command=lambda d: ["pytest"], test_runner=scripted_tests(RED, RED))
    t = await wait_status(runner, (await runner.start("fix the renderer"))["task_id"], *TERMINAL)
    assert t["status"] == "done" and t["grade"]["grade"] == "good"


async def test_no_test_run_for_a_look_only_job_or_when_nothing_changed(tmp_path):
    git, edited = git_repo_with_his_work(tmp_path)
    look = step(1, "Inspect the renderer", kind="delegate", args=str(tmp_path))
    look["action"]["read_only"] = True
    tests = scripted_tests(GREEN)
    runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([look])]), [],
                                      terminals=EditingTerminals([None]), free_range=True, git=git,
                                      test_command=lambda d: ["pytest"], test_runner=tests)
    t = await wait_status(runner, (await runner.start("look at the renderer"))["task_id"], *TERMINAL)
    assert t["status"] == "done" and tests.ran == []

    tests2 = scripted_tests(GREEN)
    runner2, _, _ = make_runner(tmp_path, FakeLLM(plans=[plan([step(1, "Fix it", kind="delegate", args=str(tmp_path))])]),
                                [], terminals=EditingTerminals([None]), free_range=True, git=git,
                                test_command=lambda d: ["pytest"], test_runner=tests2)
    t2 = await wait_status(runner2, (await runner2.start("fix it"))["task_id"], *TERMINAL)
    assert t2["grade"]["reason"] == "it says it's done but nothing in the project changed"
    assert len(tests2.ran) == 1                                # only the before-run


async def test_tests_are_off_with_the_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("AIVA_TASK_RUN_TESTS", "0")
    assert make_runner(tmp_path, FakeLLM(), [])[0].verify_tests is False
    monkeypatch.delenv("AIVA_TASK_RUN_TESTS")
    assert make_runner(tmp_path, FakeLLM(), [])[0].verify_tests is True


def test_tests_result_reads_pytest_output():
    out = ("....F\n=========================== short test summary info ===========================\n"
           "FAILED tests/test_x.py::test_y - AssertionError\n1 failed, 4 passed in 0.31s\n")
    r = T.tests_result(1, out)
    assert r["ran"] and not r["passed"] and r["summary"] == "1 failed, 4 passed in 0.31s"
    assert r["failing"] == ["FAILED tests/test_x.py::test_y - AssertionError"]
    assert T.tests_result(0, "...\n= 3 passed in 0.1s =\n")["summary"] == "3 passed in 0.1s"
    assert T.tests_result(5, "no tests ran") == {"ran": False, "why": "no tests collected"}


def test_project_test_command_only_uses_the_projects_own_venv(tmp_path):
    assert T.project_test_command(str(tmp_path)) is None                     # no tests at all
    (tmp_path / "tests").mkdir()
    assert T.project_test_command(str(tmp_path)) is None                     # tests, but no venv
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_text("")
    assert T.project_test_command(str(tmp_path)) is None                     # venv without pytest
    (tmp_path / ".venv" / "Lib" / "site-packages" / "pytest").mkdir(parents=True)
    cmd = T.project_test_command(str(tmp_path))
    assert cmd[0] == str(scripts / "python.exe") and cmd[1:4] == ["-B", "-m", "pytest"]


def test_run_git_returns_the_whole_output(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for i in range(80):
        (tmp_path / f"untracked_file_number_{i:03}.txt").write_text("x")
    out = T.run_git(str(tmp_path), "status", "--porcelain", "-uall")
    assert out["success"] and len(out["stdout"]) > 2000
    assert "untracked_file_number_000.txt" in out["stdout"] and "untracked_file_number_079.txt" in out["stdout"]


def test_read_only_prompt_wording_and_worktree_delta():
    assert T.READ_ONLY_RE.search("Inspect the repo. Do not modify files yet.")
    assert T.READ_ONLY_RE.search("a read-only review")
    assert not T.READ_ONLY_RE.search("Fix the bug and modify the tests")
    assert not T.READ_ONLY_RE.search("without changing any implementation files, write the note")
    before = {"head": "a", "files": {"his.py": (1, 10), "gone.py": (1, 1)}}
    after = {"head": "a", "files": {"his.py": (1, 10), "new.py": (2, 5)}}
    assert TaskRunner._worktree_delta(before, after) == {"new.py", "gone.py"}
    assert T.WORKTREE_NOISE_RE.search("aiva/logs/aiva-20261013.log")
    assert T.WORKTREE_NOISE_RE.search("aiva/avatar_state.json")
    assert not T.WORKTREE_NOISE_RE.search("aiva/modules/avatar_renderer.py")


# --- what her body shows while she works (avatar work panel) ------------------------------------

def test_activity_is_none_until_a_task_is_live_then_shows_the_newest():
    r = TaskRunner(task_memory=None, toolbox=None, llm_call=None)
    assert r.activity() is None
    r.tasks = {
        "old": {"goal": "Tidy downloads", "status": "done", "started_ts": 1, "plan": []},
        "a": {"goal": "Fix the chat box", "status": "running", "started_ts": 5, "plan": [
            {"title": "read dashboard.py", "status": "done"},
            {"title": "edit the input row", "status": "running"},
            {"title": "check it", "status": "pending"}]},
        "b": {"goal": "Plan a trip", "status": "planning", "started_ts": 3, "plan": []},
    }
    assert r.activity() == {"label": "Fix the chat box", "detail": "edit the input row",
                            "step": 1, "total": 3, "state": "running"}


def test_activity_shows_the_task_waiting_on_him_first():
    r = TaskRunner(task_memory=None, toolbox=None, llm_call=None)
    r.tasks = {
        "a": {"goal": "Newer, running", "status": "running", "started_ts": 9, "plan": []},
        "b": {"goal": "Needs a yes", "status": "awaiting_user", "started_ts": 2,
              "plan": [{"title": "run the build", "status": "awaiting_user"}]},
    }
    act = r.activity()
    assert act["label"] == "Needs a yes" and act["state"] == "waiting"
    r.tasks["b"]["status"] = "verifying"
    r.tasks["a"]["status"] = "cancelled"
    assert r.activity()["state"] == "checking"


async def test_a_coding_step_in_his_whole_home_is_sent_back_to_the_planner(tmp_path):
    # 2026-10-14: the planner wrote project_dir "." and Claude Code ran across all of C:\Users\Extre
    from modules.tasks import REPO_DIR
    terms = FakeTerminals()
    llm = FakeLLM(plans=[plan([step(1, "Fix the renderer", kind="delegate", args=".")]),
                         plan([step(1, "Fix the renderer", kind="delegate", args=str(tmp_path))])])
    runner, tmem, calls = make_runner(tmp_path, llm, [], terminals=terms, free_range=True)
    t = await wait_status(runner, (await runner.start("fix the renderer"))["task_id"], *TERMINAL)
    assert any(e["event"] == "plan_invalid" and "whole home" in e["detail"] for e in t["log"])
    assert [s["project_dir"] for s in terms.started] == [str(tmp_path)]
    assert llm.planner_inputs[0]["your_own_code"] == REPO_DIR


# --- looking without asking (2026-10-14: nine approvals for a verification; a text-file task ran
# --- out of moves asking to search) -----------------------------------------------------------

def a_project(tmp_path):
    proj = tmp_path / "proj"
    (proj / "tests").mkdir(parents=True)
    (proj / ".git").mkdir()
    (proj / "notes.md").write_text("the witness should glow softly\n", encoding="utf-8")
    (proj / "tests" / "test_sky.py").write_text("def test_ok():\n    pass\n", encoding="utf-8")
    (proj / ".venv" / "Lib").mkdir(parents=True)
    (proj / ".venv" / "Lib" / "hidden.md").write_text("glow", encoding="utf-8")      # never searched
    return proj


async def test_looking_and_testing_never_wait_on_him(tmp_path):
    proj = a_project(tmp_path)
    ran = []

    def test_runner(command, project_dir, timeout):
        ran.append(command)
        return {"ran": True, "returncode": 0, "passed": True, "summary": "1 passed", "failing": []}

    def git(project_dir, *args):
        return {"success": True, "returncode": 0, "stderr": "",
                "stdout": " M notes.md\n" if "status" in args else "abc123 first\n"}

    llm = FakeLLM(plans=[plan([step(1, "Check the witness", kind="agent")])],
                  agent_script=[[("search_text", {"directory": str(proj), "text": "GLOW"}),
                                 ("find_files", {"directory": str(proj), "pattern": "*.md"}),
                                 ("read_file", {"path": str(proj / "notes.md")}),
                                 ("repo_status", {"project_dir": str(proj)}),
                                 ("run_tests", {"project_dir": str(proj),
                                                "tests": ["tests/test_sky.py", "-k evil", "../outside.py"]})],
                                [("step_done", {"summary": "checked", "evidence": "1 passed"})]])
    runner, tmem, calls = make_runner(tmp_path, llm, [], free_range=True, git=git,
                                      test_command=lambda d: ["pytest"], test_runner=test_runner)
    t = await wait_status(runner, (await runner.start("check the witness"))["task_id"], *TERMINAL)
    assert t["status"] == "done"
    assert not any(e["event"] in ("shell", "answer") for e in t["log"]) and runner.shell_calls == []
    assert ran == [["pytest", os.path.join("tests", "test_sky.py")]]      # options and outside paths dropped
    ev = " ".join(t["plan"][0]["evidence"])
    assert "notes.md:1: the witness should glow softly" in ev and "hidden.md" not in ev
    assert "1 passed" in ev and "M notes.md" in ev


def test_the_look_tools_stay_inside_his_home_and_read_text_only(tmp_path):
    from modules.tasks import find_files, pick_test_targets, read_file_lines, search_text
    proj = a_project(tmp_path)
    assert read_file_lines(r"C:\Windows\win.ini")["success"] is False
    (proj / "blob.bin").write_bytes(b"\x00\x01glow")
    assert read_file_lines(str(proj / "blob.bin"))["success"] is False
    r = read_file_lines(str(proj / "tests" / "test_sky.py"), start_line=2, max_lines=1)
    assert r["lines"] == "2:     pass" and r["total_lines"] == 2 and r["more"] is False
    assert find_files(str(proj), "*.md")["files"] == ["notes.md"]
    assert search_text(str(proj), "glow")["matches"] == ["notes.md:1: the witness should glow softly"]
    assert pick_test_targets(str(proj), ["tests/test_sky.py::test_ok", "--co", "../x.py", "tests/nope.py"]) \
        == [os.path.join("tests", "test_sky.py") + "::test_ok"]


async def test_a_failed_task_hands_her_what_it_found_and_asks_for_the_direct_way(tmp_path):
    proj = a_project(tmp_path)
    llm = FakeLLM(plans=[plan([step(1, "Gather the notes", kind="agent"), step(2, "Write the file")])],
                  agent_script=[[("read_file", {"path": str(proj / "notes.md")})],
                                [("step_failed", {"reason": "the notes are scattered"})]])
    announces = []
    runner, tmem, calls = make_runner(tmp_path, llm, announces, free_range=True,
                                      limits=Limits(plan_wait_secs=2, answer_reminder_secs=5, max_repairs_per_step=0))
    t = await wait_status(runner, (await runner.start("write my vision file"))["task_id"], *TERMINAL)
    assert t["status"] == "failed"
    note = announces[-1]
    assert "What it had found" in note and "glow softly" in note and "doing the rest yourself" in note


async def test_only_he_can_answer_a_tasks_question():
    from modules.tools import make_task_handlers

    class Runner:
        answered = []

        async def answer(self, answer, task_id=None):
            self.answered.append(answer)
            return {"success": True}

    class Params:
        def __init__(self, arguments):
            self.arguments, self.results = arguments, []

        async def result_callback(self, r):
            self.results.append(r)

    spoke = {"him": False}
    runner = Runner()
    h = make_task_handlers(runner, None, heard_him=lambda: spoke["him"])
    p = Params({"answer": "yes"})
    await h["answer_task"](p)                    # her own news turn, or her sister's
    assert p.results[0]["success"] is False and runner.answered == []
    spoke["him"] = True
    p = Params({"answer": "yes"})
    await h["answer_task"](p)
    assert p.results[0] == {"success": True} and runner.answered == ["yes"]


# --- 2026-10-14: Saga's agent gave up on his note on its first move ("creating a file needs the
# --- shell confirmation") with create_file in its hands -------------------------------------------

async def test_giving_up_without_trying_is_sent_back_once(tmp_path):
    llm = FakeLLM(plans=[plan([step(1, "Write his note", kind="agent")])],
                  agent_script=[[("step_failed", {"reason": "creating a file needs the shell confirmation"})],
                                [("create_file", {"file_path": "desktop/note.txt", "content": "the sky"})],
                                [("step_done", {"summary": "wrote it", "evidence": "note.txt"})]])
    runner, tmem, calls = make_runner(tmp_path, llm, [], free_range=True)
    t = await wait_status(runner, (await runner.start("write my note"))["task_id"], *TERMINAL)
    assert t["status"] == "done" and (tmp_path / "note.txt").read_text() == "the sky"
    assert any(e["event"] == "pushed_back" for e in t["log"]) and runner.shell_calls == []


async def test_giving_up_twice_is_accepted(tmp_path):
    reason = {"reason": "there is no such window"}
    llm = FakeLLM(plans=[plan([step(1, "Close the window", kind="agent")])],
                  agent_script=[[("step_failed", reason)], [("step_failed", reason)]])
    runner, tmem, calls = make_runner(tmp_path, llm, [], free_range=True,
                                      limits=Limits(plan_wait_secs=2, answer_reminder_secs=5, max_repairs_per_step=0))
    t = await wait_status(runner, (await runner.start("close it"))["task_id"], *TERMINAL)
    assert t["status"] == "failed" and "no such window" in t["result_summary"]
    assert llm.calls.count("You are the hands of Aiva,"[:25]) == 2


def test_the_hands_know_writing_a_file_needs_no_yes():
    p = T.RUNNER_SYSTEM_PROMPT
    assert "WRITE a" in p and "create_file" in p and "already approved" in p
    shell = next(t for t in T.PRIVATE_TOOLS if t["function"]["name"] == "shell")["function"]["description"]
    assert "write a file" in shell and "create_file" in shell


def test_create_file_never_replaces_his_file_unless_told(tmp_path, monkeypatch):
    from modules import file_operations as F
    monkeypatch.setattr(F, "resolve_path", lambda p: str(p))
    target = tmp_path / "note.txt"
    target.write_text("his words", encoding="utf-8")
    second = F.create_file(str(target), "new words")
    assert target.read_text(encoding="utf-8") == "his words" and second == str(tmp_path / "note (2).txt")
    assert F.create_file(str(target), "again") == str(tmp_path / "note (3).txt")
    assert F.create_file(str(target), "replaced", overwrite=True) == str(target)
    assert target.read_text(encoding="utf-8") == "replaced"


async def test_create_file_tool_says_where_the_text_went(tmp_path, monkeypatch):
    from modules import file_operations as F
    from modules import tools as TL
    monkeypatch.setattr(F, "resolve_path", lambda p: str(p))
    (tmp_path / "note.txt").write_text("his words", encoding="utf-8")

    class Params:
        def __init__(self, arguments):
            self.arguments, self.results = arguments, []

        async def result_callback(self, r):
            self.results.append(r)

    p = Params({"file_path": str(tmp_path / "note.txt"), "content": "x"})
    await TL.create_file(p)
    assert p.results[0]["created"].endswith("note (2).txt") and "already existed" in p.results[0]["note"]


def test_search_never_opens_audio_or_models(tmp_path, monkeypatch):
    proj = a_project(tmp_path)
    (proj / "clip.wav").write_bytes(b"RIFF glow " * 10)       # no NUL in the head, still never read
    opened = []
    real_open = open

    def spy(path, *a, **kw):
        opened.append(str(path))
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", spy)
    r = T.search_text(str(proj), "glow")
    assert r["matches"] == ["notes.md:1: the witness should glow softly"]
    assert not any(p.endswith(".wav") for p in opened)


async def test_a_screen_check_sees_what_the_step_reported(tmp_path):
    llm = FakeLLM(plans=[plan([step(1, "Look at the left monitor", kind="agent", verify="screen")])],
                  agent_script=[[("look_at_screen", {})],
                                [("step_done", {"summary": "observed", "evidence": "no stars visible"})]])
    seen = []
    orig = llm.__call__

    async def spy(messages, **kw):
        if messages[0]["content"].startswith("You verify one step"):
            seen.append(json.loads(messages[1]["content"])["observation"])
        return await orig(messages, **kw)

    llm.__call__ = spy
    runner, tmem, calls = make_runner(tmp_path, spy, [], free_range=True)
    t = await wait_status(runner, (await runner.start("look at the sky"))["task_id"], *TERMINAL)
    assert t["status"] == "done"
    assert "The screen shows Notepad" in seen[0] and "no stars visible" in seen[0]


# --- 2026-10-14: Aiva's verification note. The look-only job found "there is no speech dimming",
# --- which stopped the task as a broken premise; the retry he asked for "outside plan mode" still ran
# --- in plan mode, because its prompt said "without changing any implementation files" ----------

def delegate_step(n, title, project_dir, prompt, **action):
    s = step(n, title, kind="delegate", args=str(project_dir))
    s["action"].update(prompt=prompt, **action)
    return s


async def test_the_planners_read_only_flag_beats_the_wording(tmp_path):
    words = "Run the check without changing any implementation files, then write the note on his desktop."
    for flag, mode in ((False, "execute"), ("false", "execute"), (True, "plan"), ("true", "plan")):
        terms = FakeTerminals()
        llm = FakeLLM(plans=[plan([delegate_step(1, "Check and write the note", tmp_path, words, read_only=flag)])])
        runner, tmem, calls = make_runner(tmp_path, llm, [], terminals=terms, free_range=True)
        await wait_status(runner, (await runner.start("verify and note it"))["task_id"], *TERMINAL)
        assert terms.started[0]["mode"] == mode, flag


async def test_without_a_flag_only_a_whole_no_changes_means_plan_mode(tmp_path):
    for prompt, mode in (("Inspect the renderer. Do not modify files yet.", "plan"),
                         ("Look around, but don't change anything.", "plan"),
                         ("Fix the bug without changing the public API.", "execute"),
                         ("Do not modify implementation files; write the report to his desktop.", "execute")):
        terms = FakeTerminals()
        s = delegate_step(1, "Do it", tmp_path, prompt)
        s["action"].pop("read_only", None)
        runner, tmem, calls = make_runner(tmp_path, FakeLLM(plans=[plan([s])]), [], terminals=terms, free_range=True)
        await wait_status(runner, (await runner.start("do it"))["task_id"], *TERMINAL)
        assert terms.started[0]["mode"] == mode, prompt


def premise_grader(tmem, seen):
    async def grade(client, model, goal, evidence, step=None):
        seen.append(step)
        return {"grade": "partial", "reason": "speech dimming does not exist in the code",
                "blocked": "premise", "question": None}
    tmem.grade_agent_work = grade


async def test_a_look_that_finds_it_missing_has_its_answer_and_the_note_still_gets_written(tmp_path):
    terms = FakeTerminals()
    terms.job_result = {"status": "done", "result": "Tracing passes 35 tests; there is no speech dimming.", "cost": 1.0}
    look = delegate_step(1, "Inspect and verify", tmp_path, "Verify the trace and the speech dimming.", read_only=True)
    note = step(2, "Write the note", kind="agent")
    llm = FakeLLM(plans=[plan([look, note])],
                  agent_script=[[("create_file", {"file_path": "desktop/verify.txt", "content": "no dimming"})],
                                [("step_done", {"summary": "wrote it", "evidence": "verify.txt"})]])
    runner, tmem, calls = make_runner(tmp_path, llm, [], terminals=terms, free_range=True)
    seen = []
    premise_grader(tmem, seen)
    t = await wait_status(runner, (await runner.start("verify the witness and note it"))["task_id"], *TERMINAL)
    assert t["status"] == "done" and (tmp_path / "verify.txt").read_text() == "no dimming"
    assert any(e["event"] == "finding" for e in t["log"]) and not any(e["event"] == "rethink" for e in t["log"])
    assert seen[0]["later_steps"] == ["Write the note"]


async def test_a_premise_still_stops_coding_built_on_it(tmp_path):
    terms = FakeTerminals()
    look = delegate_step(1, "Inspect the Miso renderer", tmp_path, "Inspect the Miso renderer.", read_only=True)
    build = delegate_step(2, "Fix the Miso renderer", tmp_path.parent, "Fix it.", read_only=False)
    llm = FakeLLM(plans=[plan([look, build])], rethinks=[{"what_went_wrong": "There is no Miso renderer.",
                                                         "next": "give_up"}])
    runner, tmem, calls = make_runner(tmp_path, llm, [], terminals=terms, free_range=True)
    premise_grader(tmem, [])
    t = await wait_status(runner, (await runner.start("fix the miso renderer"))["task_id"], *TERMINAL)
    assert t["status"] == "failed" and len(terms.started) == 1
    assert not any(e["event"] == "finding" for e in t["log"])
