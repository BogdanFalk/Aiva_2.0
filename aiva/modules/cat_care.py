"""Aiva looks after her cat on her own — the caretaker that keeps an eye on the desktop cat.

The cat (modules/cat.py) lives on its tk thread and can only *ask* (a rate-limited 'hungry'
announce). This is the other half: a loop-thread watcher that reads the cat's published snapshot
about once a minute and decides — with no LLM call, so no tokens — whether she feeds it, plays with
it or pets it, the way an owner who glances over now and then would.

    care_decision(snap, ctx)    PURE: None | (action, why). Need first (feed > play > pet), then a
                                pet just because when she's in a warm mood. Never a cat that's busy
                                (eating, carried, chasing), never wakes a napping one unless it's
                                starving, never repeats what he just did himself, only feeding at
                                night, and her optional extras (play/pet) are capped per day.
    should_tell(ctx)            PURE: whether this one is worth a spoken heads-up (rarely — a
                                turn costs voice-model tokens and he doesn't need a feed log).
    action_for_wish(text, snap) PURE: which care a wish like "play with Miso" means, or None when
                                the cat can't take it right now (so the wish waits).
    CatCaretaker                the I/O: tick() acts through DesktopCat's own command queue (which
                                records it on the cat, so cat_status tells her later), emits
                                `cat_care` on the experience bus — on the loop thread, the sqlite
                                rule the tk thread can't keep — relays a public act to her sister
                                once a day, and only sometimes announces.
"""

import re
import time
from datetime import datetime

CARE = {
    "feed_below": 35.0,             # hunger under this → fill the bowl (the cat's own 'hungry' is 25)
    "starving": 12.0,               # ...under this she'd even wake a napping cat to eat
    "play_below": 45.0,             # mood under this, with energy to spare → toss the ball
    "play_energy": 35.0,            # play costs the cat 8 energy; a tired cat gets a pet instead
    "pet_below": 60.0,              # mood under this → a pet
    "fondness_valence": 0.30,       # she's feeling good (valence) ...
    "fondness_social": 0.30,        # ... or wanting company → a pet just because
    "fondness_mood_below": 90.0,    # (a cat already over the moon doesn't need it)
    "fondness_gap_s": 3 * 3600,     # at most this often
    "gap_s": {"feed": 20 * 60, "play": 45 * 60, "pet": 30 * 60},   # per action, hers
    "min_gap_s": 5 * 60,            # between any two of her cares
    "user_grace_s": 10 * 60,        # he just did it himself → leave it to him
    "max_extras_per_day": 8,        # play + pet; feeding is need-driven and self-limiting
    "quiet_hours": (1, 8),          # [1:00, 8:00): she only feeds
    "boot_grace_s": 2 * 60,         # let the cat window come up and settle first
    "tells_per_day": 2,             # spoken heads-ups
    "tell_gap_s": 3 * 3600,
}

# behaviours where the cat is already in the middle of something
BUSY = ("CARRIED", "FALL", "RUN", "POUNCE", "EAT", "PLAY", "PURR", "ALERT")
ACTIONS = ("feed", "play", "pet")

TELL_NOTE = ("[On your own, you just {did} {name}{why}. If it fits the moment, mention it to him in "
             "ONE short, natural sentence, the way an owner does in passing; if not, say nothing.]")
_DID = {"feed": "filled the bowl for", "play": "tossed the ball for", "pet": "gave a pet to"}
_WHY = {"hungry": " — it was getting hungry", "starving": " — it was really hungry",
        "down": " — it seemed a bit down", "fondness": ", just because you felt like it",
        "wish": ", because you wanted to"}
_FELT = {"hungry": "it was getting hungry", "starving": "it was really hungry",
         "down": "it seemed a bit down", "fondness": "just because", "wish": "because I wanted to"}
_PAST = {"feed": "fed", "play": "played ball with", "pet": "petted"}

_WISH_WORDS = (
    ("feed", re.compile(r"\b(feed|fed|food|bowl|hungry|eat|eats|eating|treats?|snack)\b", re.I)),
    ("play", re.compile(r"\b(play|playing|ball|toy|toys|game)\b", re.I)),
    ("pet", re.compile(r"\b(pet|petting|cuddle|stroke|scratch|snuggle|pat)\b", re.I)),
)


def _since(now, ts):
    return now - ts if ts else 1e9


def _quiet_hour(hour, cfg):
    lo, hi = cfg["quiet_hours"]
    return lo <= hour < hi


def care_decision(snap, ctx, cfg=None):
    """Pure: None | (action, why). `snap` is DesktopCat.care_snapshot() (hunger/mood/energy
    0..100, behavior, dismissed, last_fed, self_care/user_care {action: wall ts}); `ctx` carries
    now (wall s), hour, secs_since_boot, last_care_ts, extras_today and her valence/social."""
    cfg = cfg or CARE
    if not snap or snap.get("dismissed"):
        return None
    if ctx.get("secs_since_boot", 1e9) < cfg["boot_grace_s"]:
        return None
    behavior = snap.get("behavior") or "SIT"
    if behavior in BUSY:
        return None
    now = ctx["now"]
    if _since(now, ctx.get("last_care_ts")) < cfg["min_gap_s"]:
        return None
    hunger, mood, energy = snap.get("hunger", 100.0), snap.get("mood", 100.0), snap.get("energy", 100.0)
    mine = snap.get("self_care") or {}
    his = snap.get("user_care") or {}

    def free(action):
        return (_since(now, mine.get(action)) >= cfg["gap_s"][action]
                and _since(now, his.get(action)) >= cfg["user_grace_s"])

    # feeding is need-driven: last_fed covers his feeds as well as hers
    if (hunger < cfg["feed_below"] and _since(now, snap.get("last_fed")) >= cfg["gap_s"]["feed"]
            and free("feed") and (behavior != "SLEEP" or hunger < cfg["starving"])):
        return ("feed", "starving" if hunger < cfg["starving"] else "hungry")
    if behavior == "SLEEP" or _quiet_hour(ctx.get("hour", 12), cfg):
        return None
    if ctx.get("extras_today", 0) >= cfg["max_extras_per_day"]:
        return None
    if mood < cfg["play_below"] and energy >= cfg["play_energy"] and free("play"):
        return ("play", "down")
    if mood < cfg["pet_below"] and free("pet"):
        return ("pet", "down")
    warm = (ctx.get("valence", 0.0) >= cfg["fondness_valence"]
            or ctx.get("social", 0.0) >= cfg["fondness_social"])
    if (warm and mood < cfg["fondness_mood_below"]
            and _since(now, mine.get("pet")) >= cfg["fondness_gap_s"] and free("pet")):
        return ("pet", "fondness")
    return None


def should_tell(ctx, cfg=None):
    """Pure: worth a spoken heads-up? Only to him at the PC, while she's awake and not mid-turn,
    outside quiet hours, unless initiative is 'never' — and at most `tells_per_day`, spaced out."""
    cfg = cfg or CARE
    if ctx.get("permission", "ask") == "never":
        return False
    if not ctx.get("awake") or ctx.get("busy") or ctx.get("present") is not True:
        return False
    if _quiet_hour(ctx.get("hour", 12), cfg):
        return False
    if ctx.get("tells_today", 0) >= cfg["tells_per_day"]:
        return False
    return _since(ctx["now"], ctx.get("last_tell_ts")) >= cfg["tell_gap_s"]


def action_for_wish(text, snap, cfg=None):
    """Pure: the care a wish means — the one it names, else whatever the cat needs most — or None
    when the cat can't take it right now (hidden, busy, or napping and not starving)."""
    cfg = cfg or CARE
    if not snap or snap.get("dismissed"):
        return None
    behavior = snap.get("behavior") or "SIT"
    if behavior in BUSY:
        return None
    hunger, mood, energy = snap.get("hunger", 100.0), snap.get("mood", 100.0), snap.get("energy", 100.0)
    named = next((a for a, rx in _WISH_WORDS if rx.search(text or "")), None)
    if named is None:
        named = "feed" if hunger < 70 else "play" if mood < 90 else "pet"
    if named == "play" and energy < cfg["play_energy"]:
        named = "pet"                       # too tired for the ball
    if behavior == "SLEEP" and not (named == "feed" and hunger < cfg["starving"]):
        return None
    return named


class CatCaretaker:
    """Owns the I/O around the pure decisions. Runs on the event loop (main.py's `cat_care_watch`
    ticks it); every collaborator is injected and optional so it tests with fakes."""

    def __init__(self, cat, *, bus=None, announce=None, house_emit=None, mood=None, tell_ctx=None,
                 enabled=True, log=print, clock=time.time, cfg=None):
        self.cat = cat
        self.bus = bus                      # ExperienceBus — loop thread only
        self._announce = announce           # async/sync (text) -> None
        self._house_emit = house_emit       # (summary) -> None: a public act for her sister
        self._mood = mood or (lambda: {})   # -> {"valence", "social"}
        self._tell_ctx = tell_ctx or (lambda: {})   # -> {"permission", "awake", "busy", "present"}
        self.enabled = enabled
        self.log = log
        self.clock = clock
        self.cfg = cfg or CARE
        self._boot = clock()
        self._last_care = 0.0
        self._last_tell = 0.0
        self._day = None
        self._extras = 0
        self._tells = 0
        self._house_day = None

    def _roll_day(self, now):
        day = datetime.fromtimestamp(now).date().isoformat()
        if day != self._day:
            self._day, self._extras, self._tells = day, 0, 0

    def _snapshot(self):
        if not self.enabled or not getattr(self.cat, "available", False):
            return None
        try:
            return self.cat.care_snapshot()
        except Exception:
            return None

    async def tick(self):
        """One glance at the cat. Returns the (action, why) she took, or None."""
        snap = self._snapshot()
        if not snap:
            return None
        now = self.clock()
        self._roll_day(now)
        ctx = {"now": now, "hour": datetime.fromtimestamp(now).hour,
               "secs_since_boot": now - self._boot, "last_care_ts": self._last_care,
               "extras_today": self._extras}
        try:
            ctx.update(self._mood() or {})
        except Exception:
            pass
        pick = care_decision(snap, ctx, self.cfg)
        if not pick:
            return None
        action, why = pick
        await self._care(action, why, snap, now, feel=True)
        return pick

    def can_act_on_wish(self, text):
        """For main.py's `_self_actionable`: can she act on this cat wish right now?"""
        return action_for_wish(text, self._snapshot(), self.cfg) is not None

    async def act_on_wish(self, text):
        """Initiative 'act' on a tend_cat wish. The self engine already felt it (self_act) and told
        her sister, so this only does it and tells him. Returns the action or None."""
        snap = self._snapshot()
        action = action_for_wish(text, snap, self.cfg)
        if not action:
            return None
        now = self.clock()
        self._roll_day(now)
        await self._care(action, "wish", snap, now, feel=False, tell=True)
        return action

    async def _care(self, action, why, snap, now, *, feel, tell=None):
        getattr(self.cat, action)()         # DesktopCat.feed/play/pet: queued + recorded on the cat
        self._last_care = now
        if action != "feed":
            self._extras += 1
        name = snap.get("name") or "your cat"
        self.log(f"cat care: {action} ({why}) — hunger {int(snap.get('hunger', 0))} "
                 f"mood {int(snap.get('mood', 0))} energy {int(snap.get('energy', 0))}")
        if feel:
            if self.bus is not None:
                try:
                    self.bus.emit("cat_care", "cat", f"{_PAST[action]} {name} — {_FELT[why]}",
                                  valence=0.2 if why == "fondness" or action == "play" else 0.12,
                                  social=0.1, salience=0.3, payload={"action": action, "why": why})
                except Exception as e:
                    self.log(f"cat care emit failed: {e!r}")
            day = datetime.fromtimestamp(now).date().isoformat()
            if self._house_emit is not None and self._house_day != day:
                self._house_day = day
                try:
                    res = self._house_emit(f"{_PAST[action]} {name} on her own")
                    if hasattr(res, "__await__"):
                        await res
                except Exception as e:
                    self.log(f"cat care house emit failed: {e!r}")
        if tell is None:
            try:
                tctx = dict(self._tell_ctx() or {})
            except Exception:
                tctx = {}
            tctx.update(now=now, hour=datetime.fromtimestamp(now).hour,
                        tells_today=self._tells, last_tell_ts=self._last_tell)
            tell = should_tell(tctx, self.cfg)
        if tell and self._announce is not None:
            self._tells += 1
            self._last_tell = now
            note = TELL_NOTE.format(did=_DID[action], name=name, why=_WHY.get(why, ""))
            try:
                res = self._announce(note)
                if hasattr(res, "__await__"):
                    await res
            except Exception as e:
                self.log(f"cat care announce failed: {e!r}")
