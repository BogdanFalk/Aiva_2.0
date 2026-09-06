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

    def __init__(self, plans=None, verdicts=None, agent_script=None, plan_delay=0.0):
        self.plans = list(plans or [])
        self.verdicts = list(verdicts or [])
        self.agent_script = list(agent_script or [])
        self.plan_delay = plan_delay
        self.calls = []

    async def __call__(self, messages, *, tools=None, json_mode=False, max_tokens=900, temperature=0.2):
        system = messages[0]["content"]
        self.calls.append(system[:25])
        if system.startswith("You are the planner"):
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
                terminals=None):
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
                        resolve_path=lambda p: str(tmp_path / os.path.basename(p)))

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
