"""Name parametrization: every module-level *_PROMPT still renders "Aiva" with no slot left (imports
and prefix-dispatching fakes are unchanged), and a store built with name="Saga" speaks as Saga.
`persona.named` is str.replace, never .format — the prompts carry literal JSON braces."""
import inspect
import json
import sqlite3

import pytest

from modules import becoming, inner, memory, persona, reading, task_memory, tasks, terminals

AIVA_PROMPTS = {
    "fact": memory.FACT_EXTRACTION_PROMPT, "recap": memory.RECAP_PROMPT,
    "journal": inner.JOURNAL_PROMPT, "becoming": becoming.BECOMING_PROMPT,
    "close_reading": reading.CLOSE_READING_PROMPT, "session_wrap": reading.SESSION_WRAP_PROMPT,
    "synthesis": reading.SYNTHESIS_PROMPT, "grade": task_memory.GRADE_PROMPT,
    "feedback": task_memory.FEEDBACK_EXTRACTION_PROMPT, "planner": tasks.PLANNER_PROMPT,
    "runner": tasks.RUNNER_SYSTEM_PROMPT,
}
SLOTS = ("{name}", "{name_variants}", "{sister_note}", "{self_hint}", "{temperament_clause}",
         "{sister_clause}", "{baseline_desc}")


@pytest.mark.parametrize("key", sorted(AIVA_PROMPTS))
def test_module_constants_render_aiva_with_no_slot_left(key):
    prompt = AIVA_PROMPTS[key]
    assert "Aiva" in prompt
    for slot in SLOTS:
        assert slot not in prompt, f"{key} still carries {slot}"


def test_named_is_replace_not_format():
    t = 'You are {name}. Reply {"facts": [{"fact": "..."}]} using {tools}; {name} again, {x}.'
    out = persona.named(t, "Saga", x="here")
    assert out == 'You are Saga. Reply {"facts": [{"fact": "..."}]} using {tools}; Saga again, here.'
    assert persona.named("a{k}b", "X", k=None) == "ab"


def test_task_prompts_keep_their_late_slots():
    for slot in ("{tools}", "{families}", "{approaches}"):
        assert slot in tasks.PLANNER_PROMPT
    assert "{budget}" in tasks.RUNNER_SYSTEM_PROMPT


def test_memory_prompts_per_character():
    a = memory.Memory(":memory:")
    assert a.fact_prompt == memory.FACT_EXTRACTION_PROMPT and a.recap_prompt == memory.RECAP_PROMPT
    s = memory.Memory(":memory:", name="Saga", name_variants=("saga", "sagah", "hey saga"),
                      sisters=("Aiva",))
    assert "memory of Saga" in s.fact_prompt and "about Saga's behavior" in s.fact_prompt
    assert '"Saga" or "Sagah"' in s.fact_prompt and "Hey Saga" not in s.fact_prompt
    assert "(her sister is Aiva)" in s.fact_prompt and 'category "sister"' in s.fact_prompt
    assert "notes for Saga" in s.recap_prompt and "Aiva" not in s.recap_prompt
    for slot in SLOTS:
        assert slot not in s.fact_prompt and slot not in s.recap_prompt


def test_self_engine_renders_name_and_self_hint():
    con = sqlite3.connect(":memory:")
    default = inner.SelfEngine(con, inner.ExperienceBus(con), inner.InnerBrain(inner.InnerConfig()))
    assert default.journal_prompt == inner.JOURNAL_PROMPT and default.name == "Aiva"
    con2 = sqlite3.connect(":memory:")
    saga = inner.SelfEngine(con2, inner.ExperienceBus(con2), inner.InnerBrain(inner.InnerConfig()),
                            name="Saga", self_forming=True)
    assert saga.journal_prompt.startswith("You are Saga, writing privately")
    assert inner.FORMING_SELF_HINT in saga.journal_prompt and "cheeky" not in saga.journal_prompt
    assert "Aiva" not in saga.journal_prompt and "{temperament_clause}" not in saga.journal_prompt


def test_becoming_renders_name_baseline_desc_and_permission_fact():
    remembered = []
    b = becoming.Becoming(sqlite3.connect(":memory:"), name="Saga", baseline_traits=[],
                          remember=lambda f, c: remembered.append(f) or 1, log=lambda m: None)
    assert b.becoming_prompt.startswith("You are the part of Saga, a voice companion")
    assert becoming.SELF_FORMING_BASELINE_DESC in b.becoming_prompt
    assert "smartass" not in b.becoming_prompt and "Saga's own voice" in b.becoming_prompt
    b.set_permission("always")
    assert remembered and remembered[0].startswith("Saga may read on her own")
    a = becoming.Becoming(sqlite3.connect(":memory:"), log=lambda m: None)
    assert a.becoming_prompt == becoming.BECOMING_PROMPT
    # an authored (non-Aiva) baseline gets the generic sentence, a custom one wins outright
    c = becoming.Becoming(sqlite3.connect(":memory:"), name="Nova", baseline_traits=["I'm calm"],
                          log=lambda m: None)
    assert becoming.AUTHORED_BASELINE_DESC in c.becoming_prompt
    d = becoming.Becoming(sqlite3.connect(":memory:"), name="Nova", baseline_desc="She is stern.",
                          log=lambda m: None)
    assert "She is stern. Reading GROWS her" in d.becoming_prompt


def test_reading_engine_renders_name(tmp_path):
    async def llm(*a, **k):
        return None
    bm = reading.BookMemory(sqlite3.connect(":memory:"), embed=None)
    eng = reading.ReadingEngine(bm, llm_call=llm, client=None, library_dir=str(tmp_path),
                                model_name="m", name="Saga")
    assert eng.close_reading_prompt.startswith("You are Saga, and you are reading")
    assert eng.session_wrap_prompt.startswith("You are Saga. You just put a book down")
    assert eng.synthesis_prompt.startswith("You are Saga, and you have just finished")
    for p in (eng.close_reading_prompt, eng.session_wrap_prompt, eng.synthesis_prompt):
        assert "Aiva" not in p and "{name}" not in p
    default = reading.ReadingEngine(bm, llm_call=llm, client=None, library_dir=str(tmp_path),
                                    model_name="m")
    assert default.close_reading_prompt == reading.CLOSE_READING_PROMPT


def test_task_memory_toolbox_and_runner_render_name():
    tmem = task_memory.TaskMemory(sqlite3.connect(":memory:"), name="Saga")
    assert tmem.grade_prompt.startswith("You audit an autonomous coding agent for Saga")
    assert "Saga's task memory" in tmem.feedback_prompt and "Aiva" not in tmem.feedback_prompt

    async def noop(params):
        return {"success": True}
    box = tasks.ToolBox({"launch_app": noop}, name="Saga")          # no schema → fallback description
    assert "Saga tool launch_app" in json.dumps(box.openai_tools())
    runner = tasks.TaskRunner(task_memory=tmem, toolbox=box, llm_call=None, log=lambda m: None,
                              name="Saga")
    assert runner.planner_prompt.startswith("You are the planner inside Saga")
    assert runner.runner_prompt.startswith("You are the hands of Saga")
    assert "{tools}" in runner.planner_prompt and "{budget}" in runner.runner_prompt
    default = tasks.TaskRunner(task_memory=task_memory.TaskMemory(sqlite3.connect(":memory:")),
                               toolbox=tasks.ToolBox({}), llm_call=None, log=lambda m: None)
    assert default.planner_prompt == tasks.PLANNER_PROMPT


def test_terminal_titles_and_vision_helpers_take_a_name():
    assert terminals.TerminalManager(name="Saga")._title("main") == "Saga: main"
    assert terminals.TerminalManager()._title("main") == "Aiva: main"
    from modules import computer
    for fn in (computer.describe_screen, computer.describe_image):
        assert inspect.signature(fn).parameters["name"].default == "Aiva"
