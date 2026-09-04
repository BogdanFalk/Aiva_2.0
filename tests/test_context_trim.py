import sys

import main  # import-safe: definitions only (sys.argv guard below)

sys.argv = ["main.py"]


class Ctx:
    def __init__(self, msgs):
        self._m = list(msgs)

    def get_messages(self):
        return self._m

    def set_messages(self, msgs):
        self._m[:] = msgs


class Mem:
    def __init__(self):
        self.forgotten = 0

    def forget_saved(self, n):
        self.forgotten += n


def turn(i, size=400):
    return [{"role": "user", "content": f"u{i} " + "x" * size},
            {"role": "assistant", "content": f"a{i} " + "y" * size}]


def test_keeps_system_block_and_recent_turns():
    msgs = [{"role": "system", "content": "persona"}, {"role": "system", "content": "intro"}]
    for i in range(20):
        msgs += turn(i)
    ctx, mem = Ctx(msgs), Mem()
    dropped = main.trim_context(ctx, mem, budget_chars=4000, keep_min=8)
    kept = ctx.get_messages()
    assert kept[0]["content"] == "persona" and kept[1]["content"] == "intro"
    assert dropped > 0 and len(kept) >= 2 + 8
    assert kept[2]["content"].startswith("u")                # oldest surviving turn starts with the user
    assert kept[-1]["content"].startswith("a19")             # newest turn intact
    assert mem.forgotten == dropped                          # all dropped messages were spoken ones
    assert sum(len(m["content"]) for m in kept[2:]) <= 4000 + 810


def test_tool_exchange_dropped_together_and_in_progress_protected():
    msgs = [{"role": "system", "content": "p"}]
    msgs += [{"role": "user", "content": "open it " + "x" * 500},
             {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                                                   "function": {"name": "launch_app", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "c1", "content": "{\"success\": true}" + "z" * 500},
             {"role": "assistant", "content": "done " + "y" * 500}]
    for i in range(6):
        msgs += turn(i, 300)
    ctx = Ctx(msgs)
    dropped = main.trim_context(ctx, None, budget_chars=3000, keep_min=4)
    kept = ctx.get_messages()
    assert dropped >= 3 and not any(m.get("tool_call_id") == "c1" for m in kept)
    assert not any(m.get("tool_calls") for m in kept[:2])   # no orphaned tool_calls left at the front

    # an in-flight tool exchange at the front blocks trimming
    msgs2 = [{"role": "system", "content": "p"},
             {"role": "assistant", "tool_calls": [{"id": "c9", "type": "function",
                                                   "function": {"name": "look_at_screen", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "c9", "content": "IN_PROGRESS"}]
    for i in range(6):
        msgs2 += turn(i, 500)
    ctx2 = Ctx(msgs2)
    assert main.trim_context(ctx2, None, budget_chars=1000, keep_min=2) == 0
    assert len(ctx2.get_messages()) == len(msgs2)


def test_under_budget_is_a_noop():
    ctx, mem = Ctx([{"role": "system", "content": "p"}] + turn(1)), Mem()
    assert main.trim_context(ctx, mem, budget_chars=24000) == 0 and mem.forgotten == 0
