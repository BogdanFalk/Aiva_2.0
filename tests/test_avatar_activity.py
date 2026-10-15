"""avatar_activity: the choreography of her body reading (the book) and working (the panel) —
pure, clock-injected, no Qt."""
import math

import pytest

from modules import avatar_activity as A

DT = 1 / 32


def run(d, start, secs, **kw):
    """Step the director at 32 fps from `start` for `secs`; returns (poses, end_time)."""
    poses, t = [], start
    for _ in range(int(round(secs / DT))):
        t += DT
        poses.append(d.step(DT, t, **kw))
    return poses, t


def reader():
    d = A.ActivityDirector(A.RIGS["ariu"])
    d.set_state(reading={"title": "Through the Looking-Glass"})
    return d


TASK = {"label": "Fix the chat box", "detail": "edit dashboard.py", "step": 1, "total": 4,
        "state": "running"}


# --- reading: pulling up the book --------------------------------------------------------------

def test_hands_come_up_before_the_book_and_the_book_rises_into_them():
    d = reader()
    poses, _ = run(d, 0.0, A.READ_IN_SECS + 0.2)
    first_book = next(i for i, p in enumerate(poses) if p.book)
    assert first_book > 0 and poses[first_book - 1].hold > 0      # hands already on their way
    assert all(p.hide_held for p in poses)                         # the controller never shows
    ys = [p.book["y"] for p in poses if p.book]
    assert ys[0] < ys[-1]                                          # it rose from below
    assert ys[-1] == pytest.approx(A.RIGS["ariu"].book_y, abs=0.02)
    last = poses[-1]
    assert last.hold == pytest.approx(1.0) and last.book["alpha"] == pytest.approx(1.0)
    assert last.params["ParamAngleY"] < -8                         # head tipped down to the page


def test_her_eyes_run_along_lines_while_she_reads():
    d = reader()
    poses, _ = run(d, 0.0, 6.0)
    xs = [p.params["ParamEyeBallX"] for p in poses[48:]]
    assert max(xs) > 0.2 and min(xs) < -0.2                        # left-to-right sweeps
    assert all(p.params["ParamEyeBallY"] < -0.3 for p in poses[48:])  # looking down at the book


def test_putting_the_book_away_runs_backwards_hands_last():
    d = reader()
    _, t = run(d, 0.0, 2.0)
    d.set_state(reading=None)
    poses, t = run(d, t, A.READ_OUT_SECS + 0.2)
    gone = next(i for i, p in enumerate(poses) if not p.book)
    assert poses[gone].hold > 0                                    # book gone, hands still up
    assert poses[-1].hold == 0 and not poses[-1].hide_held and not poses[-1].book
    more, _ = run(d, t, 1.0)
    assert not more[-1].active                                     # back to plain idle life


def test_she_looks_up_from_the_book_while_she_talks():
    d = reader()
    _, t = run(d, 0.0, 2.0)
    reading_pose = d.step(DT, t + DT)
    talking, t = run(d, t + DT, 0.6, talking=True)
    up = talking[-1]
    assert up.params["ParamAngleY"] > reading_pose.params["ParamAngleY"] + 6
    assert up.book["y"] < reading_pose.book["y"]                   # book lowered a little
    after, _ = run(d, t, 2.5)                                      # stops talking: back to the page
    assert after[-1].params["ParamAngleY"] == pytest.approx(reading_pose.params["ParamAngleY"], abs=1.0)


def test_a_page_cue_bounces_the_book_once():
    d = reader()
    _, t = run(d, 0.0, 2.0)
    rest = d.step(DT, t + DT).book
    assert d.cue("page", t + DT) is True
    poses, t = run(d, t + DT, A.PAGE_SECS)
    assert max(abs(p.book["rot"]) for p in poses) > 1.0
    assert max(p.book["y"] for p in poses) > rest["y"] + 0.005
    later = run(d, t, 0.5)[0][-1].book
    assert later["rot"] == 0.0


def test_page_cue_without_a_book_is_a_no_op():
    d = A.ActivityDirector(A.RIGS["ariu"])
    assert d.cue("page", 1.0) is True
    assert not d.step(DT, 1.0 + DT).book


def test_a_body_without_a_rig_never_does_any_of_it():
    d = A.ActivityDirector(None)
    d.set_state(reading={"title": "x"}, working=TASK)
    d.cue("done", 0.0)
    poses, _ = run(d, 0.0, 2.0)
    assert not any(p.active for p in poses)


# --- working: the panel -------------------------------------------------------------------------

def test_panel_opens_beside_her_towards_the_middle_and_stays_on_that_side():
    d = A.ActivityDirector(A.RIGS["ariu"])
    d.set_state(working=TASK)
    poses, t = run(d, 0.0, A.PANEL_IN_SECS + 0.1, side_hint=-1)
    pv = poses[-1].panel
    assert pv["side"] == -1 and pv["cx"] < 0
    assert pv["alpha"] == pytest.approx(1.0) and pv["open"] == pytest.approx(1.0)
    assert pv["title"] == "Fix the chat box" and pv["step"] == "2/4"
    moved, t = run(d, t, 0.5, side_hint=1)                         # she's dragged across mid-task
    assert moved[-1].panel["side"] == -1                           # ...the panel doesn't hop
    d.set_state(working=None)
    closed, t = run(d, t, A.PANEL_OUT_SECS + 0.1, side_hint=1)
    assert closed[-1].panel is None
    d.set_state(working=TASK)
    reopened, _ = run(d, t, 0.3, side_hint=1)
    assert reopened[-1].panel["side"] == 1


def test_panel_keeps_the_last_task_while_it_closes():
    d = A.ActivityDirector(A.RIGS["ariu"])
    d.set_state(working=TASK)
    _, t = run(d, 0.0, 1.0)
    d.set_state(working=None)
    closing = d.step(DT, t + DT).panel
    assert closing is not None and closing["title"] == "Fix the chat box"


def test_she_glances_at_the_panel_while_she_works():
    d = A.ActivityDirector(A.RIGS["ariu"])
    d.set_state(working=TASK)
    poses, _ = run(d, 0.0, 9.0, side_hint=-1)
    xs = [p.params.get("ParamEyeBallX", 0.0) for p in poses]
    turns = [p.head_turn for p in poses]
    assert min(xs) < -0.5 and min(turns) < -5                      # looks (and turns) to her left
    assert max(xs) > min(xs) + 0.4                                 # ...and back to the room


def test_waiting_on_him_freezes_the_typing_and_she_looks_at_him():
    d = A.ActivityDirector(A.RIGS["ariu"])
    d.set_state(working=dict(TASK, state="waiting"))
    poses, _ = run(d, 0.0, 4.0, side_hint=-1)
    assert poses[-1].panel["state"] == "waiting"
    assert poses[-1].panel["lines"] == poses[-30].panel["lines"]   # nothing typing
    assert abs(poses[-1].params.get("ParamEyeBallX", 0.0)) < 0.05  # eyes on him, not the panel


def test_a_finished_task_flashes_its_result_then_the_panel_goes():
    d = A.ActivityDirector(A.RIGS["ariu"])
    d.set_state(working=TASK)
    _, t = run(d, 0.0, 1.0)
    d.cue("done", t)
    d.set_state(working=None)
    poses, t = run(d, t, A.RESULT_SECS - 0.2)
    assert all(p.panel and p.panel["state"] == "done" for p in poses)
    assert poses[-1].panel["progress"] == 1.0
    after, _ = run(d, t, 0.2 + A.PANEL_OUT_SECS + 0.1)
    assert after[-1].panel is None


def test_a_result_shows_even_when_the_task_was_too_quick_to_see():
    d = A.ActivityDirector(A.RIGS["ariu"])
    d.cue("failed", 0.0)
    poses, _ = run(d, 0.0, 1.0)
    assert poses[-1].panel["state"] == "failed"


def test_progress_fills_smoothly_and_never_runs_backwards():
    d = A.ActivityDirector(A.RIGS["ariu"])
    d.set_state(working=dict(TASK, step=3))
    poses, t = run(d, 0.0, 3.0)
    assert poses[-1].panel["progress"] == pytest.approx(0.75, abs=0.02)
    assert poses[2].panel["progress"] < 0.5                        # it filled, didn't jump
    d.set_state(working=dict(TASK, step=1))                        # a re-plan shrinks the count
    later, _ = run(d, t, 1.0)
    assert later[-1].panel["progress"] >= 0.74


def test_reading_and_working_at_once():
    d = reader()
    d.set_state(working=TASK)
    poses, _ = run(d, 0.0, 3.0)
    assert poses[-1].book and poses[-1].panel


# --- inputs and geometry --------------------------------------------------------------------------

def test_clean_working_and_reading_sanitise_ipc_payloads():
    w = A.clean_working({"label": "  a\n b  " + "x" * 300, "step": "2", "total": -4, "state": "rm -rf"})
    assert w["label"].startswith("a b") and len(w["label"]) == 120
    assert w["step"] == 2 and w["total"] == 0 and w["state"] == "running"
    assert A.clean_working("yes") is None and A.clean_working(None) is None
    assert A.clean_reading({"title": 5}) == {"title": ""}
    assert A.clean_reading([1]) is None


def test_set_state_is_partial():
    d = A.ActivityDirector(A.RIGS["ariu"])
    d.set_state(reading={"title": "a"}, working=TASK)
    d.set_state(working=None)
    assert d.reading == {"title": "a"} and d.working is None


def test_unknown_cues_are_refused():
    assert A.ActivityDirector(A.RIGS["ariu"]).cue("dance", 0.0) is False


def test_model_to_px_follows_her_placement():
    home = {"x": 0.0, "y": 0.0, "rotation": 0.0, "size": 0.0}
    assert A.model_to_px((0, 0), home, 1600, 900, 0.3) == pytest.approx((800, 450))
    # a point one model-unit up sits 0.3 view units up: 0.3 * 450 px
    assert A.model_to_px((0, 1), home, 1600, 900, 0.3) == pytest.approx((800, 450 - 135))
    moved = dict(home, x=0.5, y=-0.5)
    assert A.model_to_px((0, 0), moved, 1600, 900, 0.3) == pytest.approx((1200, 675))
    doubled = dict(home, size=100 * math.log10(2))
    assert A.model_to_px((0, 1), doubled, 1600, 900, 0.3)[1] == pytest.approx(450 - 270)
    assert A.model_len_px(1.0, doubled, 900, 0.3) == pytest.approx(270)
    turned = dict(home, rotation=90)                               # CCW: "up" points left
    x, y = A.model_to_px((0, 1), turned, 1600, 900, 0.3)
    assert x == pytest.approx(800 - 135) and y == pytest.approx(450)


def test_panel_side_points_to_the_middle_of_the_screen():
    assert A.panel_side(0.8) == -1
    assert A.panel_side(-0.7) == 1
    assert A.panel_side(0.0) == 1


def test_every_rig_names_a_look():
    assert set(A.RIGS) <= set(A.LOOKS)
