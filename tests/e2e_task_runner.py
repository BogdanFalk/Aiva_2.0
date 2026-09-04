"""Semi-live task runner check: REAL OpenAI + REAL desktop tools, NO Aiva pipeline.

    cd aiva && ..\\.venv\\Scripts\\python ..\\tests\\e2e_task_runner.py "open notepad, type hello world, save it to the desktop as hello.txt and check the file exists"

Prints every announce the runner would have spoken through Aiva and the persisted
task row. Confirmations are asked on the console; `--yes` auto-approves PLAN
confirmations only — shell commands and Claude Code delegation always ask.
Uses a throwaway memory database (e2e_memory.db next to this file).
"""
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "aiva"))
os.chdir(os.path.join(os.path.dirname(HERE), "aiva"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(HERE), ".env"))

from openai import AsyncOpenAI  # noqa: E402

from modules import computer, file_operations  # noqa: E402
from modules.memory import Memory  # noqa: E402
from modules.task_memory import TaskMemory, embed_text  # noqa: E402
from modules.tasks import TaskRunner, ToolBox, openai_llm_call  # noqa: E402
from modules.terminals import TerminalManager  # noqa: E402
from modules.tools import TOOL_SCHEMAS, build_handlers, execute_shell  # noqa: E402


class _NoAvatar:
    expressions = {}

    def __getattr__(self, name):
        async def missing(*a, **k):
            return {"success": False, "error": "no avatar in e2e"}
        return missing


async def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    auto_yes = "--yes" in sys.argv
    goal = " ".join(args) or "open notepad, type hello world, save it to the desktop as hello.txt and check the file exists"

    oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("AIVA_TASK_MODEL") or os.getenv("AIVA_LLM_MODEL", "gpt-4.1")
    memory = Memory(os.path.join(HERE, "e2e_memory.db"))
    task_memory = TaskMemory(memory.conn, embed=lambda t: embed_text(oai, t))
    terminals = TerminalManager()
    handlers = build_handlers(_NoAvatar(), None, terminals, memory, None, None)

    announces = []

    async def announce(text):
        announces.append(text)
        print("\n[ANNOUNCE]\n" + text + "\n")

    runner = TaskRunner(task_memory=task_memory, toolbox=ToolBox(handlers, TOOL_SCHEMAS),
                        llm_call=openai_llm_call(oai, model), terminals=terminals, log=print,
                        execute_shell=lambda c: asyncio.to_thread(execute_shell, c),
                        describe_screen=computer.describe_screen, list_monitors=computer.list_monitors,
                        resolve_path=file_operations.resolve_path, facts=memory.load_facts, model_name=model)
    runner.openai_client = oai
    runner.attach_voice(announce=announce, quiet_note=None, is_awake=lambda: True)

    print(f"GOAL: {goal}\nMODEL: {model}\n")
    first = await runner.start(goal)
    print("[start_task ->]", json.dumps(first, ensure_ascii=False, indent=1))
    tid = first["task_id"]

    while runner.tasks[tid]["status"] not in ("done", "failed", "cancelled", "interrupted"):
        t = runner.tasks[tid]
        if t["status"] == "awaiting_user":
            pending = t["pending"] or {}
            print(f"\n[QUESTION ({pending.get('kind')})] {pending.get('text')}")
            if auto_yes and pending.get("kind") == "confirm_plan":
                reply = "yes"
                print("-> auto: yes")
            elif not sys.stdin.isatty():
                reply = "no"  # unattended run: never approve commands/delegation blind
                print("-> no console: answering no")
            else:
                reply = await asyncio.to_thread(input, "your answer> ")
            print("[answer_task ->]", await runner.answer(reply, tid))
            await asyncio.sleep(0.2)
            continue
        await asyncio.sleep(0.5)

    await runner.join(tid, timeout=10)  # let the final announce flush before we close the db
    row = task_memory.get_task(tid)
    row.pop("embedding", None)
    print("\n[TASK ROW]\n" + json.dumps(row, ensure_ascii=False, indent=1, default=str))
    print(f"\nDONE: status={row['status']} outcome={row['outcome']} llm_calls={row['llm_calls']} "
          f"tool_calls={row['tool_calls']} cost=${row['cost_usd']:.3f}")
    terminals.shutdown()
    memory.close()


if __name__ == "__main__":
    asyncio.run(main())
