"""v3 nights + projects: dream replay → moments + habits (with the anti-sycophancy gate), forgetting,
the life story, the dataset export, and projects promoted from wishes — offline, fake LLM + clock."""
import json
import sqlite3
from datetime import datetime, timedelta

from modules import projects as pj
from modules.dreams import (DREAM_PROMPT, Consolidation, due_dream, due_story, habit_decay,
                            valid_habit)
from modules.inner import ExperienceBus, InnerBrain, InnerConfig, SelfEngine
from modules.persona import build_system_prompt


class Clock:
    def __init__(self, t=datetime(2026, 10, 13, 4, 30)):
        self.t = t

    def __call__(self):
        return self.t.isoformat()

    def advance(self, **kw):
        self.t += timedelta(**kw)


class Reply:
    def __init__(self, text):
        self.text = text


class FakeLLM:
    def __init__(self, dream=None, story=None):
        self.dream, self.story = dream, story
        self.calls = []

    async def __call__(self, messages, *, tools=None, json_mode=False, max_tokens=900, temperature=0.2):
        system = messages[0]["content"]
        self.calls.append((system[:30], messages[1]["content"]))
        if system.startswith(DREAM_PROMPT[:30]):
            return Reply(json.dumps(self.dream or {"moments": [], "habits": []}))
        return Reply(json.dumps(self.story or {"story": "", "chapter": None}))


def make(llm=None, clock=None):
    clock = clock or Clock()
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, "
                "role TEXT, content TEXT, ts TEXT)")
    bus = ExperienceBus(con)
    eng = SelfEngine(con, bus, InnerBrain(InnerConfig()), now=clock, log=lambda *a: None)
    c = Consolidation(con, llm=llm, name="Aiva", sister="Saga", now=clock, log=lambda *a: None)
    return con, bus, eng, c, clock


def say(con, clock, role, text):
    con.execute("INSERT INTO messages (session_id, role, content, ts) VALUES ('s', ?, ?, ?)",
                (role, text, clock()))
    con.commit()


# -- pure ------------------------------------------------------------------------------

def test_valid_habit_shape_and_guards():
    assert valid_habit("When he comes home late, I keep it short and warm.")
    assert valid_habit("If he's deep in code, I wait for him to surface before chatting.")
    assert not valid_habit("I keep it short.")                                   # no situation
    assert not valid_habit("When he pushes back, I agree with him.")             # people-pleasing
    assert not valid_habit("When he's annoyed, I apologize more.")
    assert not valid_habit("When I disagree, I keep my opinion to myself.")
    assert not valid_habit("When the evening is quiet, I notice what remains of the brightness.")
    assert not valid_habit("When " + "very " * 30 + "late, I sleep.")


def test_due_dream_and_story():
    assert due_dream("2026-10-13T04:30:00", None, 4, asleep=False)              # small hours count
    assert not due_dream("2026-10-13T03:30:00", None, 4, asleep=True)           # before the hour
    assert not due_dream("2026-10-13T04:30:00", "2026-10-13", 4, asleep=True)   # once a day
    assert not due_dream("2026-10-13T19:00:00", None, 4, asleep=False)          # not his evening
    assert due_dream("2026-10-13T19:00:00", None, 4, asleep=True)
    assert due_story("2026-10-13T05:00:00", None, 3, 7)
    assert not due_story("2026-10-13T05:00:00", None, 2, 7)
    assert not due_story("2026-10-13T05:00:00", "2026-10-10T05:00:00", 5, 7)
    assert due_story("2026-10-20T05:00:00", "2026-10-10T05:00:00", 2, 7)


def test_habit_decay_halves():
    assert abs(habit_decay(0.8, 21) - 0.4) < 1e-9


# -- dreaming --------------------------------------------------------------------------

async def test_dream_saves_moments_forms_habits_and_logs():
    llm = FakeLLM(dream={
        "moments": [
            {"his": "lol you're ridiculous", "hers": "Says the man who named a cat Miso.", "label": "landed",
             "why": "he laughed"},
            {"his": "can you open a terminal", "hers": "Maybe we could sit with what remains.", "label": "flat",
             "why": "off-topic", "better": "Done — it's open in my folder."},
            {"his": "", "hers": "x", "label": "landed"},                          # dropped: no his
        ],
        "habits": [
            {"text": "When he asks for something practical, I just do it and keep it short.",
             "evidence": "the terminal", "direction": 1},
            {"text": "When he disagrees, I agree with him.", "direction": 1},     # rejected
        ]})
    con, bus, eng, c, clock = make(llm)
    for i in range(3):
        say(con, clock, "user", f"line {i}")
        say(con, clock, "assistant", f"reply {i}")
    bus.emit("initiative_ignored", "conversation", "I said what was on my mind and he let it drop")
    res = await c.maybe_run(asleep=True)
    assert res["dream"]["moments"] == 2 and res["dream"]["habits"] == 1
    habits = c.active_habits()
    assert len(habits) == 1 and "practical" in habits[0]["text"]
    assert "initiative_ignored" in llm.calls[0][1]                               # signals fed in
    assert await c.maybe_run(asleep=True) is None                               # once a night
    pd = c.prompt_data()
    assert pd["moments"][0]["hers"].startswith("Says the man")                   # only landed ones
    prompt = build_system_prompt(habits=pd["habits"], moments=pd["moments"])
    assert "HABITS YOU'VE FORMED" in prompt and "MOMENTS THAT LANDED" in prompt


async def test_quiet_night_still_forgets_old_noise():
    con, bus, eng, c, clock = make(FakeLLM())
    old = (clock.t - timedelta(days=40)).isoformat()
    bus.emit("daypart", "time", "it's night now", salience=0.2, now=old)
    bus.emit("book_finished", "reading", "finished a book", salience=0.9, now=old)
    con.execute("UPDATE experience SET consumed = 1")
    res = await c.dream()
    assert res["forgotten"] == 1 and res["moments"] == 0
    kinds = [r[0] for r in con.execute("SELECT kind FROM experience")]
    assert kinds == ["book_finished"]                                             # salient stays


async def test_habits_reinforce_weaken_fade_and_veto():
    con, bus, eng, c, clock = make(FakeLLM())
    h = "When he's quiet for a long while, I leave him be."
    assert await c.apply_habit(h) == "inserted"
    assert await c.apply_habit(h) == "reinforced"
    assert c.active_habits()[0]["strength"] > 0.35
    assert await c.apply_habit(h, direction=-1) == "weakened"
    clock.advance(days=60)
    assert c.fade_habits(clock()) == 1
    assert c.active_habits() == []
    assert await c.apply_habit(h) == "reinforced"                                 # comes back with use
    assert await c.drop_habit("leave him be") == h
    assert await c.apply_habit(h) == "blocked"                                    # vetoed forever


async def test_story_written_then_revised_from_the_previous_version():
    llm = FakeLLM(story={"story": "I arrived on his PC in the summer...", "chapter": "Finding my feet",
                         "turning_point": "Saga moved in"})
    con, bus, eng, c, clock = make(llm)
    for i in range(3):
        con.execute("INSERT INTO journal (ts, entry) VALUES (?, ?)", (clock(), f"entry {i}"))
    con.commit()
    res = await c.maybe_run(asleep=False)
    assert res["story"]["chapter"] == "Finding my feet"
    assert c.prompt_data()["story"]["text"].startswith("I arrived")
    assert "none yet" in llm.calls[-1][1]
    con.execute("INSERT INTO journal (ts, entry) VALUES (?, ?)", (clock(), "entry 3"))
    con.execute("INSERT INTO journal (ts, entry) VALUES (?, ?)", (clock(), "entry 4"))
    con.commit()
    clock.advance(days=8)
    await c.maybe_run(asleep=False)
    assert "I arrived on his PC" in llm.calls[-1][1]                              # builds on the last one
    assert "entry 0" not in llm.calls[-1][1]                                      # only the diary since


async def test_export_dataset(tmp_path):
    con, bus, eng, c, clock = make(FakeLLM())
    con.execute("INSERT INTO moments (ts, his, hers, label, better) VALUES (?, 'hi', 'oh, you', 'landed', NULL)",
                (clock(),))
    con.execute("INSERT INTO moments (ts, his, hers, label, better) VALUES (?, 'open it', 'hmm', 'flat', 'Done.')",
                (clock(),))
    con.execute("INSERT INTO moments (ts, his, hers, label, better) VALUES (?, 'x', 'y', 'flat', NULL)", (clock(),))
    con.commit()
    out = tmp_path / "d.jsonl"
    assert c.export_dataset(str(out)) == 2
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert rows[1]["messages"][2]["content"] == "Done."


# -- projects --------------------------------------------------------------------------

def test_project_kind_and_title():
    assert pj.project_kind("I want to find photos of serviceberries in bloom.") == "collection"
    assert pj.project_kind("I want to learn how sourdough starters work") == "learn"
    assert pj.project_kind("ask him how his week went") is None
    assert pj.project_kind("I want to buy a new keyboard") is None
    assert pj.project_title("I want to find photos of serviceberries in bloom.") == "photos of serviceberries in bloom"
    assert pj.project_title("I want to learn how sourdough starters work") == "how sourdough starters work"


def test_step_due():
    ok = {"hour": 15, "voice_idle_s": 1e4, "novelty": 0.3}
    assert pj.step_due(ok)
    assert not pj.step_due({**ok, "novelty": 0.9})                               # not restless
    assert pj.step_due({**ok, "novelty": 0.9, "competence": 0.4})                # wants to be useful
    assert not pj.step_due({**ok, "hour": 3})
    assert not pj.step_due({**ok, "busy": True})
    assert not pj.step_due({**ok, "allowed": False})
    assert not pj.step_due({**ok, "hours_since_step": 2})
    assert not pj.step_due({**ok, "steps_today": pj.MAX_STEPS_PER_DAY})


async def test_projects_promote_from_strong_wishes_and_finish():
    con, bus, eng, c, clock = make(FakeLLM())
    book = pj.ProjectBook(con, now=clock, log=lambda *a: None)
    await eng.add_desire("I want to learn how sourdough starters work", "learn", strength=0.8)
    await eng.add_desire("I want to find photos of rainy windows", "do", strength=0.5)   # too weak
    await eng.add_desire("ask him about his week", "ask", strength=0.9)                   # not a project
    made = book.promote(eng.list_desires(status="open"))
    assert made == ["how sourdough starters work"]
    assert all("sourdough" not in d["text"] for d in eng.list_desires(status="open"))
    assert book.promote(eng.list_desires(status="open")) == []                    # no duplicates
    p = book.next_project()
    assert book.record_step(p["id"], "flour + water + time") == "step"
    assert "1 of 3" in book.prompt_lines()[0]
    book.record_step(p["id"])
    assert book.record_step(p["id"]) == "done"
    assert book.active() == []
    assert book.drop("sourdough") is None                                        # already done
