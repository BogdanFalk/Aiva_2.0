import sqlite3

import pytest

from modules import task_memory as tm
from modules.task_memory import TaskMemory, decide_autonomy, normalize_goal


@pytest.fixture
def mem():
    return TaskMemory(sqlite3.connect(":memory:"), name_variants=("aiva", "ava"))


def finished(mem, goal, outcome="success", family="other", steps=2, corrected=None, embedding=None):
    tid = tm.new_task_id() + goal[:3]
    plan = [{"n": i + 1, "title": f"step {i + 1}", "kind": "tool", "status": "done"} for i in range(steps)]
    mem.record_episode(tid, goal, family=family, plan_steps=plan, embedding=embedding)
    mem.update_task(tid, plan=plan)
    mem.finish_episode(tid, outcome, f"{outcome} summary", 5.0)
    if corrected:
        mem.add_feedback(tid, corrected, True)
    return tid


# --- normalization ------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Hey Aiva, could you please set up my stream now?", "set up stream"),
    ("eva eva open open the aiva repo in a terminal", "open repo in terminal"),
    ("Ayva, count files in C:\\Users\\me\\Downloads", "count files in c:\\users\\me\\downloads"),
    ("", ""),
])
def test_normalize_goal(text, expected):
    assert normalize_goal(text, ("aiva", "ava")) == expected


# --- familiarity (lexical path) ---------------------------------------------------

async def test_novel_when_nothing_recorded(mem):
    fam = await mem.familiarity("open notepad and write hello")
    assert fam["level"] == "novel" and fam["method"] == "lexical"


async def test_seen_then_routine(mem):
    finished(mem, "open notepad and type hello world")
    fam = await mem.familiarity("open notepad and type hello world please")
    assert fam["level"] == "seen" and fam["successes"] == 1
    finished(mem, "open notepad and type hello world")
    fam = await mem.familiarity("Aiva open notepad and type hello world")
    assert fam["level"] == "routine" and fam["successes"] == 2


async def test_attempted_when_only_failures(mem):
    finished(mem, "restart obs and switch scene", outcome="failed")
    fam = await mem.familiarity("restart obs and switch scene")
    assert fam["level"] == "attempted"
    assert fam["last_failure"] == "failed summary"


async def test_correction_sticks_until_a_clean_success(mem):
    finished(mem, "restart obs and switch to gaming scene", corrected="never restart OBS while live")
    fam = await mem.familiarity("restart obs and switch to gaming scene")
    assert fam["corrected"] is True and fam["level"] == "seen"
    assert "never restart OBS" in fam["last_correction"]
    finished(mem, "restart obs and switch to gaming scene")  # a clean run afterwards
    fam = await mem.familiarity("restart obs and switch to gaming scene")
    assert fam["corrected"] is False and fam["level"] == "routine"


async def test_similar_is_same_family_shape(mem):
    finished(mem, "open notepad type hello save desktop hello.txt")
    fam = await mem.familiarity("open notepad type goodbye save desktop bye.txt")
    assert fam["level"] == "similar"


async def test_flow_trigger_short_circuits_to_routine(mem):
    mem.save_flow("streaming setup", [{"title": "focus OBS", "kind": "tool"}],
                  trigger_phrases=["start my stream"], family="streaming-setup")
    fam = await mem.familiarity("hey aiva start my stream")
    assert fam["level"] == "routine" and fam["flow"]["name"] == "streaming setup"


# --- familiarity (embedding path with fake vectors) ------------------------------

async def test_embedding_levels():
    vectors = {"same": [1.0, 0.0, 0.0], "near": [0.95, 0.312, 0.0], "far": [0.0, 1.0, 0.0],
               "mid": [0.8, 0.6, 0.0]}

    async def embed(text):
        return vectors[text.split()[0]]

    mem = TaskMemory(sqlite3.connect(":memory:"), embed=embed)
    finished(mem, "same task alpha", embedding=vectors["same"])
    fam = await mem.familiarity("near task alpha")      # cosine 0.95 -> same task
    assert fam["method"] == "embedding" and fam["level"] == "seen"
    fam = await mem.familiarity("mid task alpha")       # cosine 0.80 -> similar
    assert fam["level"] == "similar"
    fam = await mem.familiarity("far task alpha")       # cosine 0 -> novel
    assert fam["level"] == "novel"


async def test_embedding_failure_falls_back_to_lexical():
    async def embed(text):
        raise RuntimeError("offline")

    mem = TaskMemory(sqlite3.connect(":memory:"), embed=embed)
    finished(mem, "open notepad and type hello")
    fam = await mem.familiarity("open notepad and type hello")
    assert fam["method"] == "lexical" and fam["level"] == "seen"


# --- decide_autonomy truth table -----------------------------------------------

def plan(steps=2, **meta):
    p = {"summary": "open Notepad and save hello.txt", "family": "app-launch", "approach": "ui",
         "steps": [{"n": i + 1, "title": f"step {i + 1}", "kind": "tool", "risk": "none"} for i in range(steps)]}
    p.update(meta)
    return p


def fam(level, corrected=False, **extra):
    d = {"level": level, "method": "lexical", "score": 0.9, "best": {"goal_text": "open notepad and save"},
         "matches": [], "successes": 2 if level == "routine" else 1, "corrected": corrected,
         "last_correction": None, "last_failure": None, "flow": None}
    d.update(extra)
    return d


@pytest.mark.parametrize("level", ["routine", "seen", "similar"])
def test_familiar_means_just_do(level):
    d = decide_autonomy("g", plan(), fam(level))
    assert d["mode"] == "just_do"


def test_similar_non_trivial_still_just_do_per_user_rule():
    d = decide_autonomy("g", plan(steps=5), fam("similar"))
    assert d["mode"] == "just_do" and "Like" in d["say"]


def test_novel_trivial_confirms():
    d = decide_autonomy("g", plan(), fam("novel"))
    assert d["mode"] == "confirm_plan" and "New one for me" in d["say"]


def test_novel_with_distinct_options_offers_them():
    alts = [{"label": "terminal", "summary": "do it in the terminal", "tradeoff": "fast"},
            {"label": "ui", "summary": "click through the app", "tradeoff": "slow but visible"}]
    d = decide_autonomy("g", plan(steps=4, alternatives=alts, options_distinct=True), fam("novel"))
    assert d["mode"] == "offer_options" and "two ways" in d["say"]


def test_novel_without_distinct_flag_never_offers():
    alts = [{"label": "a", "summary": "x"}, {"label": "b", "summary": "y"}]
    d = decide_autonomy("g", plan(steps=4, alternatives=alts), fam("novel"))
    assert d["mode"] == "confirm_plan"


def test_pattern_streak_suppresses_options():
    alts = [{"label": "terminal"}, {"label": "ui"}]
    d = decide_autonomy("g", plan(steps=4, alternatives=alts, options_distinct=True), fam("novel"),
                        pref={"approach": "terminal", "streak": 3})
    assert d["mode"] == "confirm_plan" and d["approach_hint"] == "terminal"


def test_correction_confirms_and_states_change():
    d = decide_autonomy("g", plan(), fam("routine", corrected=True, last_correction="never restart OBS while live"))
    assert d["mode"] == "confirm_plan" and "never restart OBS while live" in d["say"]


def test_attempted_mentions_failure():
    d = decide_autonomy("g", plan(), fam("attempted", last_failure="the scene did not switch"))
    assert d["mode"] == "confirm_plan" and "the scene did not switch" in d["say"] and "Want me to?" in d["say"]


@pytest.mark.parametrize("step", [{"kind": "shell", "risk": "none"}, {"kind": "tool", "risk": "irreversible"}])
def test_always_confirm_overrides_familiarity(step):
    p = plan()
    p["steps"][0].update(step)
    d = decide_autonomy("g", p, fam("routine"))
    assert d["mode"] == "confirm_plan" and d["must_confirm_steps"] == [1]


def test_bad_delegation_record_escalates():
    p = plan(steps=3)
    p["steps"][1].update({"kind": "delegate"})
    d = decide_autonomy("g", p, fam("routine"), delegation={"last_two_bad": True})
    assert d["mode"] == "confirm_plan" and "botched" in d["say"]


def test_confirm_first_only_escalates():
    assert decide_autonomy("g", plan(), fam("routine"), confirm_first=True)["mode"] == "confirm_plan"
    assert decide_autonomy("g", plan(), fam("novel"), confirm_first=True)["mode"] == "confirm_plan"


# --- flows ----------------------------------------------------------------------------

async def test_flow_candidate_rules(mem):
    first = finished(mem, "open notepad and save a note", family="other")
    assert mem.flow_candidate(first) is None                      # one-off until repeated
    second = finished(mem, "open notepad and save a note", family="other")
    cand = mem.flow_candidate(second)
    assert cand and cand["suggest"]
    ritual = finished(mem, "focus obs and start the gaming scene", family="streaming-setup")
    assert mem.flow_candidate(ritual)                             # ritual families offer at once
    one_step_delegate = tm.new_task_id()
    mem.record_episode(one_step_delegate, "fix the bug in utils", delegated_to="claude_code",
                       plan_steps=[{"n": 1, "title": "delegate", "kind": "delegate", "status": "done"}])
    mem.finish_episode(one_step_delegate, "success", "ok", 60)
    assert mem.flow_candidate(one_step_delegate) is None


async def test_save_flow_and_prompt_block(mem):
    tid = finished(mem, "focus obs and start the gaming scene", family="streaming-setup")
    fid = mem.save_flow("streaming setup", mem.get_task(tid)["plan"],
                        trigger_phrases=["start my stream"], family="streaming-setup", source_task_id=tid)
    assert mem.get_task(tid)["flow_id"] == fid
    mem.bump_flow_stats(fid, True)
    lines = mem.load_flows()
    assert len(lines) == 1 and lines[0].startswith("streaming setup (say: 'start my stream')")
    assert "1 run, last today" in lines[0]
    assert mem.get_flow("Streaming Setup")["steps"][0]["title"] == "step 1"
    assert "status" not in mem.get_flow("streaming setup")["steps"][0]


# --- preferences, delegation record, lessons, grading -----------------------------

def test_preference_pattern_spoken_once(mem):
    remembered = []

    def remember(fact, category):
        remembered.append(fact)
        return 1 if len(remembered) == 1 else 0

    for _ in range(2):
        out = mem.record_preference("repo-ops", "terminal", ["ui"], "I like watching it", remember=remember)
        assert out["pattern_new"] is False
    out = mem.record_preference("repo-ops", "terminal", ["ui"], "I like watching it", remember=remember)
    assert out["pattern_new"] is True and "terminal route 3 times" in out["pattern_line"]
    assert remembered == ["For repo-ops tasks, default to the terminal approach; the user chose it 3 times "
                          "(I like watching it)."]
    out = mem.record_preference("repo-ops", "terminal", ["ui"], None, remember=remember)
    assert out["pattern_new"] is False                            # already known: never re-announced
    mem.record_preference("repo-ops", "ui", ["terminal"], None)
    assert mem.preference_for("repo-ops")["streak"] == 1


def test_delegation_record(mem):
    for grade in ("good", "bad", "bad"):
        tid = tm.new_task_id() + grade
        mem.record_episode(tid, "fix bug", family="repo-edit", delegated_to="claude_code")
        mem.finish_episode(tid, "success", "ok", 1, agent_grade=grade, agent_grade_reason=f"{grade} reason")
    rec = mem.delegation_record("repo-edit")
    assert rec == {"good": 1, "partial": 0, "bad": 2, "last_bad_reason": "bad reason", "last_two_bad": True}


def test_lessons_and_feedback(mem):
    tid = finished(mem, "restart obs")
    assert mem.last_episode()["id"] == tid
    mem.add_feedback(tid, "you should not restart OBS while streaming", True)
    assert mem.load_lessons() == [f"{mem.get_task(tid)['created_ts'][:10]}, \"restart obs\": "
                                  f"you should not restart OBS while streaming"]
    mem.add_feedback(tid, "otherwise fine", False)
    assert mem.get_task(tid)["corrected"] is True                # a later non-correction doesn't clear it


async def test_grade_pre_rules(mem):
    bad = await mem.grade_agent_work(None, "m", "g", {"job_status": "failed (1)"})
    assert bad["grade"] == "bad"
    empty = await mem.grade_agent_work(None, "m", "g", {"job_status": "done", "mode": "execute",
                                                         "diff_checked": True, "diff_stat": ""})
    assert empty["grade"] == "bad" and "nothing" in empty["reason"]
    tests_fail = await mem.grade_agent_work(None, "m", "g", {"job_status": "done", "diff_checked": True,
                                                              "diff_stat": " 1 file changed",
                                                              "tests": {"returncode": 1}})
    assert tests_fail["grade"] == "partial"
    good = await mem.grade_agent_work(None, "m", "g", {"job_status": "done", "diff_checked": True,
                                                        "diff_stat": " 1 file changed"})
    assert good["grade"] == "good"


def test_reconcile_stale(mem):
    mem.record_episode("t-1", "half done", status="running")
    mem.record_episode("t-2", "finished", status="done")
    mem.finish_episode("t-2", "success", "ok", 1)
    stale = mem.reconcile_stale()
    assert [t["id"] for t in stale] == ["t-1"]
    assert mem.get_task("t-1")["status"] == "interrupted"
    assert mem.get_task("t-1")["result_summary"] == "lost at shutdown"
    assert mem.reconcile_stale() == []
