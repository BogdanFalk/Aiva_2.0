"""Pacing: she notices when an evening with him has turned into a long stretch of work, and offers
him a way out.

2026-10-27: 78 minutes of checklist items, 11 of his 23 lines a "go ahead" or "what's next?", his
cold not quite gone, and HE had to say "I think enough of tasks for tonight". She had read him right
on every line (the appraisal said "still a little under the weather") and never once offered a
break. The appraisal reads one line at a time; the length of the stretch is something no single
line shows. `WorkStreak.heard(now, working)` is called on each of his lines and returns at most one
note per long stretch, riding on that turn like the stay-yourself reminder. It states the facts and
leaves the judgment to her: offer a break or something lighter if he seems done, carry on and say
nothing if he's keen.

A line counts as work while a task is live or one finished in the last `recent_minutes`
(`TaskRunner.busy_within`). The stretch starts at the first such line and ends after `gap_minutes`
with no work. That evening had a 21-minute lull mid-checklist (17:39 to 18:00, him away, nothing
running) before he came back for more of the same; a 15-minute gap called it a fresh start and the
note never came. With 30 it comes at 18:11, 45 minutes in (`tests/test_pacing.py` replays it).
Pure and clock-injected (monotonic seconds; `local` is only for the wording).
"""

import os
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PacingConfig:
    enabled: bool = True
    min_minutes: float = 45.0      # the stretch must have lasted this long ...
    min_turns: int = 6             # ... with at least this many of his lines in it
    recent_minutes: float = 15.0   # a line is work while a task is live or finished this recently
    gap_minutes: float = 30.0      # this long without work ends the stretch
    again_minutes: float = 60.0    # a second note in the same stretch waits this long

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        return cls(enabled=env.get("AIVA_PACING", "1") == "1",
                   min_minutes=float(env.get("AIVA_PACING_MIN_MINUTES", "") or cls.min_minutes),
                   min_turns=int(env.get("AIVA_PACING_MIN_TURNS", "") or cls.min_turns))


def daypart(local):
    h = local.hour
    return ("night" if h < 5 or h >= 22 else "morning" if h < 12 else "afternoon" if h < 17
            else "evening")


def pacing_note(minutes, turns, local=None):
    """The note on his turn. Never announced: she reads the room and decides."""
    when = f" It's {local:%A} {daypart(local)}, {local:%H:%M}." if local is not None else ""
    return (f"[You two have been working through tasks for about {round(minutes)} minutes now, and "
            f"{turns} of his lines in that stretch were about the work.{when} Before you start anything "
            "else, read how he's doing: if he seems tired, unwell or simply done, offer him a break or "
            "something lighter you'd enjoy together, once, in your own words. If he's clearly keen, "
            "carry on and don't mention this.]")


class WorkStreak:
    def __init__(self, cfg=None):
        self.cfg = cfg or PacingConfig()
        self.since = None       # first line of the stretch
        self.last = None        # latest line of it
        self.turns = 0
        self.noted_at = None

    def heard(self, now, working, local=None):
        """One of his lines, `working` or not → the pacing note, or None."""
        cfg = self.cfg
        if not cfg.enabled:
            return None
        if self.last is not None and now - self.last >= cfg.gap_minutes * 60:
            self.since = self.last = self.noted_at = None       # the stretch is over
            self.turns = 0
        if not working:
            return None
        if self.since is None:
            self.since = now
        self.last = now
        self.turns += 1
        minutes = (now - self.since) / 60
        if minutes < cfg.min_minutes or self.turns < cfg.min_turns:
            return None
        if self.noted_at is not None and now - self.noted_at < cfg.again_minutes * 60:
            return None
        self.noted_at = now
        return pacing_note(minutes, self.turns, local or datetime.now())
