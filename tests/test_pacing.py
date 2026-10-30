"""Pacing (modules/pacing.py): one "read how he's doing" note after a long stretch of task work.

2026-10-27: 78 minutes of checklist, his cold not quite gone, and he had to call it himself."""
from datetime import datetime

from modules.pacing import PacingConfig, WorkStreak, daypart, pacing_note
from modules.tasks import TaskRunner

TUE_EVENING = datetime(2026, 10, 27, 18, 9)


def _lines(streak, minutes, working=True, local=TUE_EVENING):
    return [streak.heard(m * 60.0, working, local) for m in minutes]


def test_a_long_stretch_of_work_gets_one_note():
    s = WorkStreak()
    notes = _lines(s, [0, 8, 15, 22, 30, 38])            # six lines, but only 38 minutes
    assert notes == [None] * 6
    note = s.heard(45 * 60.0, True, TUE_EVENING)
    assert note and "about 45 minutes" in note and "7 of his lines" in note
    assert "Tuesday evening, 18:09" in note and "offer him a break or something lighter" in note
    assert "carry on and don't mention this" in note
    assert _lines(s, [50, 60, 72, 85, 99]) == [None] * 5  # once per hour of the same stretch
    assert s.heard(106 * 60.0, True, TUE_EVENING) is not None


def test_a_quiet_gap_ends_the_stretch_and_chat_does_not_count():
    s = WorkStreak()
    _lines(s, [0, 10, 20, 30])
    assert s.heard(40 * 60.0, False) is None             # chatting between tasks: not counted ...
    assert s.turns == 4 and s.since == 0.0               # ... and not an end either
    assert s.heard(44 * 60.0, True, TUE_EVENING) is None  # five lines in 44 minutes
    assert s.heard(75 * 60.0, False) is None             # 31 min since the last work: over
    assert s.since is None and s.turns == 0
    assert _lines(s, [76, 85, 95, 105, 115]) == [None] * 5
    assert s.heard(121 * 60.0, True, TUE_EVENING) is not None   # a new stretch: 45 min, six lines


def test_the_evening_of_2026_10_27_gets_its_note_at_18_11():
    # his lines that evening, from aiva-20261027.log, and whether a task was live or had finished
    # in the last 15 minutes; the 17:39 -> 18:00 lull (him away, nothing running) is not a break
    evening = [("17:22", 0), ("17:22", 0), ("17:23", 0), ("17:24", 0), ("17:24", 0), ("17:25", 1),
               ("17:27", 1), ("17:28", 1), ("17:30", 1), ("17:31", 1), ("17:32", 1), ("17:33", 1),
               ("17:39", 1), ("18:00", 0), ("18:07", 1), ("18:09", 1), ("18:11", 1), ("18:16", 1),
               ("18:19", 1), ("18:31", 1), ("18:40", 1), ("18:43", 1)]
    def replay(cfg):
        s, fired = WorkStreak(cfg), []
        for hm, working in evening:
            local = datetime.fromisoformat(f"2026-10-27T{hm}")
            if s.heard(local.timestamp(), bool(working), local):
                fired.append(hm)
        return fired

    assert replay(PacingConfig()) == ["18:11"]           # 46 minutes in; he called it himself at 18:40
    assert replay(PacingConfig(gap_minutes=15)) == []    # the lull read as a fresh start: never


def test_switched_off_never_notes_and_the_thresholds_are_settings():
    s = WorkStreak(PacingConfig(enabled=False))
    assert _lines(s, range(0, 200, 5)) == [None] * 40
    notes = _lines(WorkStreak(PacingConfig(min_minutes=10, min_turns=2)), [0, 5, 11])
    assert notes[:2] == [None, None] and notes[2] is not None


def test_config_and_wording():
    cfg = PacingConfig.from_env({"AIVA_PACING": "0", "AIVA_PACING_MIN_MINUTES": "30", "AIVA_PACING_MIN_TURNS": "4"})
    assert (cfg.enabled, cfg.min_minutes, cfg.min_turns) == (False, 30.0, 4)
    assert PacingConfig.from_env({}) == PacingConfig()
    assert [daypart(datetime(2026, 1, 1, h)) for h in (3, 9, 14, 19, 23)] == \
           ["night", "morning", "afternoon", "evening", "night"]
    note = pacing_note(61.4, 9)
    assert note.startswith("[") and note.endswith("]") and "about 61 minutes" in note and "It's" not in note


class _Runner:
    def __init__(self, tasks):
        self.tasks = {t["id"]: t for t in tasks}


def test_a_task_live_or_just_finished_counts_as_work():
    now = datetime(2026, 10, 27, 18, 0, 0)
    busy = TaskRunner.busy_within
    assert busy(_Runner([]), 900, now) is False
    assert busy(_Runner([{"id": "a", "status": "running", "finished_ts": None}]), 900, now) is True
    done = {"id": "b", "status": "done", "finished_ts": "2026-10-27T17:50:00"}
    assert busy(_Runner([done]), 900, now) is True and busy(_Runner([done]), 300, now) is False
    assert busy(_Runner([{"id": "c", "status": "failed", "finished_ts": "junk"}]), 900, now) is False
