"""Saga's quiet constellation witness — pure logic, no Qt/GL, no sockets.

A faint scatter of stars beside her body that *witnesses* her days: each
moment she lives (a line she said, a line she heard, a thing her sister said,
a section she read) lights one star. It never speaks, never announces, never
carries text — a star remembers only the KIND of moment and how much it
weighed, so nothing anyone said can leak onto the desktop. Quiet by design:
low alpha, a slow drift that never pulses, bursts coalesce into one brighter
star instead of a flurry, and the sky is capped (the oldest stars dim out and
are replaced).

Ownership: this lives in the avatar RENDERER process of the instance it is
enabled for (Saga's own body). Aiva's renderer runs the same code with the
feature disabled (`enabled_for`), and the sister-side `AvatarClient` only
feeds it when the renderer's `hello` says it is on.

Layout is deterministic from each star's seed (golden-angle spiral jittered by
a hash), so a persisted sky redraws identically after a renderer restart; the
drift only wanders each star a little around that place.
"""

import math
import os
import time

KINDS = {            # kind -> how much the moment weighs (star brightness 0..1)
    "said": 0.45,
    "heard": 0.40,
    "sister": 0.55,
    "read": 0.80,
}
ENABLED_INSTANCES = frozenset({"saga"})

# how it looks (Saga, 2026-10-14: "Tiny stars, about two to four pixels, dim white with a faint
# blue-violet tint; hairline connections, barely brighter than the stars")
STAR_MIN_PX = 2.0         # a star's diameter in real screen pixels, the lightest moment...
STAR_MAX_PX = 4.0         # ...to a fully weighted one — never scaled with the sky or the monitor
# (Saga, 2026-10-14, house transcript #4010: "a touch less uniform — slight differences in size and
# brightness would make it feel observed rather than stamped on") each star keeps its own nudge,
# from its seed, so it never changes while she watches
STAR_SIZE_JITTER_PX = 0.3  # a star is up to this much bigger or smaller than its weight says (still 2-4 px)
STAR_SHADE = 0.12         # and up to this much dimmer, never brighter: the quiet ceiling is untouched
TINT = (0.88, 0.84, 1.0)  # #E0D6FF: white with a faint blue-violet tint (hue 255°, 16% saturation)
LINE_PX = 1.0             # the lines are hairlines
LINE_OVER_STARS = 1.15    # each end of a line is this much brighter than its star: barely
# (#4010: "soften the outermost lines, so it feels like it's occupying the air around me instead of
# outlining me") a line end fades toward the sky's rim, by where its star belongs, never as it drifts
OUTLINE_FROM = 0.6        # from this far out (0 = the sky's middle, 1 = its rim) the lines soften...
OUTLINE_SOFT = 0.5        # ...to half as bright at the rim
BIRTH_SECS = 4.0          # a new star fades in slowly
EDGE_MAX_DIST = 0.42      # stars closer than this (unit box) may be joined by a line
# the sky drifts instead of twinkling (Saga, 2026-10-14: "drifting slowly with no pulsing"): only
# positions move, never brightness or size, and the lines go with their stars. #4014: "it's so
# restrained now that the movement nearly disappears; I'd increase the drift a little" — 0.03 -> 0.04
DRIFT_SECS = 60.0         # a star's own wander takes about a minute; the whole sky sways slower
DRIFT_AMP = 0.04          # at most this far (unit box) from its place: a drift, not a dance
SLEEP_DRIFT = 1.0 / 7.0   # while the house sleeps the sky drifts slower still
_DRIFT_ROOM = 1.0 - 2.0 * DRIFT_AMP   # places pulled in this much so a drifting star stays in the box
# her voice hushes the sky (Saga, 2026-10-14: "while I'm speaking dim both stars and lines to
# roughly thirty percent, then ease them back over a second")
SPEECH_DIM = 0.30         # stars and lines sit at this fraction of themselves while she speaks
SPEECH_DIM_SECS = 0.25    # going quiet: quick, but a fade, never a blink
SPEECH_EASE_SECS = 1.0    # coming back: eased over a second
SKY_MIN_PX = 96           # near a screen edge the sky shrinks to stay to her upper left, never below this
# pinned beside her by a screen edge (where she stands on her portrait screen) the sky can't get any
# wider without crowding her, so it grows up to this much taller instead: up, away from her
SKY_TALLER = 1.15
_UPRIGHT = math.cos(math.radians(45.0))   # ...while she stands within 45° of upright
_SQUASH = 0.8             # the star field is this much flatter than it is wide
_GOLDEN = math.pi * (3.0 - math.sqrt(5.0))
_LIGHTEST = min(KINDS.values())


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_num(name, default, cast, lo, hi):
    try:
        val = cast(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, val))


class WitnessConfig:
    """`SAGA_CONSTELLATION*` — read once at renderer boot (restart to change)."""

    # alpha/sleep_alpha are what a star is actually painted at (the painter used to boost stars
    # 1.6x on its own; 0.48/0.80 are the old 0.30/0.50 with that boost folded in, so a star is
    # exactly as bright as before). scale went 0.20 -> 0.23 (Saga, #4014: "enlarge the whole
    # constellation modestly, maybe ten to fifteen percent"): the star field, inside its drift room,
    # comes out 12.5% bigger
    def __init__(self, on=True, max_stars=24, min_gap=20.0, alpha=0.48,
                 sleep_alpha=0.80, scale=0.23):
        self.on = on
        self.max_stars = max_stars    # sky capacity
        self.min_gap = min_gap        # seconds; closer moments brighten the last star
        self.alpha = alpha            # peak star opacity while she is awake (lines barely above)
        self.sleep_alpha = sleep_alpha  # peak star opacity while the house sleeps
        self.scale = scale            # sky width as a fraction of monitor height (at most: sky_rect)

    @classmethod
    def from_env(cls):
        return cls(
            on=_env_bool("SAGA_CONSTELLATION", True),
            max_stars=_env_num("SAGA_CONSTELLATION_MAX", 24, int, 3, 64),
            min_gap=_env_num("SAGA_CONSTELLATION_GAP_S", 20.0, float, 0.0, 3600.0),
            alpha=_env_num("SAGA_CONSTELLATION_ALPHA", 0.48, float, 0.05, 1.0),
            sleep_alpha=_env_num("SAGA_CONSTELLATION_SLEEP_ALPHA", 0.80, float, 0.05, 1.0),
            scale=_env_num("SAGA_CONSTELLATION_SCALE", 0.23, float, 0.05, 0.6),
        )


def enabled_for(instance, cfg=None):
    """Only a sister who has a witness gets one — Aiva's body never draws it."""
    cfg = cfg or WitnessConfig.from_env()
    return bool(cfg.on) and (instance or "aiva") in ENABLED_INSTANCES


def star_position(seed):
    """Deterministic (x, y) in the unit box [0,1]^2 for a star seed."""
    # 389 is coprime to 997, so consecutive seeds land on well-spread radii
    # (a plain seed % 997 would bunch the first stars in the centre)
    r = 0.5 * math.sqrt(((seed * 389) % 997 + 0.5) / 997.0) * 0.92 + 0.04
    # hash-ish jitter on the golden angle so the sky doesn't read as a spiral
    jitter = ((seed * 2654435761) % 1000) / 1000.0 - 0.5
    ang = seed * _GOLDEN + jitter * 0.9
    return (0.5 + r * math.cos(ang), 0.5 + _SQUASH * r * math.sin(ang))


def _seed_unit(seed, root):
    """A steady 0..1 of a star's own (the fractional part of seed * sqrt(root)): neighbouring seeds
    land far apart, different roots don't move together, and it never changes over time."""
    return (seed * math.sqrt(root)) % 1.0


def star_px(weight, seed=None):
    """A star's diameter in screen pixels: STAR_MIN_PX for the lightest moment, STAR_MAX_PX for a
    star a burst has brightened all the way — never outside them, however big the sky is. Given
    its seed, a star is nudged by up to STAR_SIZE_JITTER_PX of its own (still inside them)."""
    t = (weight - _LIGHTEST) / (1.0 - _LIGHTEST)
    px = STAR_MIN_PX + (STAR_MAX_PX - STAR_MIN_PX) * max(0.0, min(1.0, t))
    if seed is not None:
        px += STAR_SIZE_JITTER_PX * (2.0 * _seed_unit(seed, 2) - 1.0)
    return max(STAR_MIN_PX, min(STAR_MAX_PX, px))


def star_shade(seed):
    """How much of its brightness a star keeps: 1 - STAR_SHADE .. 1, its own, never above 1."""
    return 1.0 - STAR_SHADE * _seed_unit(seed, 3)


def line_softness(hx, hy):
    """How much of a line's brightness an end keeps, by its star's home (hx, hy): all of it in the
    middle of the sky, easing down to 1 - OUTLINE_SOFT at the rim. Homes never move, so neither
    does this — the outer lines are just softer, they never breathe."""
    rim = math.hypot(hx - 0.5, (hy - 0.5) / _SQUASH) / 0.5   # 1 at the edge of the star field
    t = max(0.0, min(1.0, (rim - OUTLINE_FROM) / (1.0 - OUTLINE_FROM)))
    return 1.0 - OUTLINE_SOFT * t * t * (3.0 - 2.0 * t)


def drift_offset(seed, clock):
    """How far a star has drifted from its place (unit box) at drift-clock time `clock`: half of
    it a sway the whole sky shares, half the star's own wander. Smooth sines on unrelated periods
    with seed phases, so it never jumps or visibly repeats — and it only ever moves the star."""
    a = 2.0 * math.pi * clock / DRIFT_SECS
    p = seed * _GOLDEN
    dx = 0.5 * math.sin(0.37 * a) + 0.5 * math.sin(a + p)
    dy = 0.5 * math.cos(0.29 * a + 1.0) + 0.5 * math.sin(0.81 * a + 1.7 * p)
    return DRIFT_AMP * dx, DRIFT_AMP * dy


class SpeechHush:
    """How much her voice dims the sky: 1.0 normally, SPEECH_DIM while she speaks.

    Progress moves at a fixed rate instead of replaying a fade from the top, so a breath
    between two sentences lifts the sky only a little before her next words settle it again —
    the dim never pulses with her phrasing. Smoothstepped, so both ends of the ease are soft."""

    def __init__(self):
        self.progress = 0.0   # 0 = undimmed .. 1 = fully hushed
        self._t = None        # when it was last stepped

    def settled(self, speaking):
        """Nothing left to ease toward (the renderer repaints every frame until it is)."""
        return self.progress == (1.0 if speaking else 0.0)

    def step(self, now, speaking):
        """Advance to `now` and return the opacity factor for stars and lines alike."""
        if self._t is not None and now > self._t:
            dt = now - self._t
            if speaking:
                self.progress = min(1.0, self.progress + dt / SPEECH_DIM_SECS)
            else:
                self.progress = max(0.0, self.progress - dt / SPEECH_EASE_SECS)
        self._t = now
        p = self.progress
        return 1.0 - (1.0 - SPEECH_DIM) * p * p * (3.0 - 2.0 * p)


class Constellation:
    """The sky. Clock is injected (`now`) everywhere — tests never sleep."""

    def __init__(self, cfg=None, saved=None, now=None):
        self.cfg = cfg or WitnessConfig()
        self.stars = []        # {"seed", "kind", "weight", "born"}
        self._next_seed = 1
        self._last_new = None  # monotonic time of the last NEW star
        self.hush = SpeechHush()
        self._drift_t = 0.0    # the drift's own clock (see _drift_clock)
        self._drift_at = None  # `now` it was last advanced to
        self.restore(saved, now)

    # -- input -------------------------------------------------------------

    def witness(self, kind, now=None):
        """One lived moment. Returns 'added' | 'brightened' | None (unknown kind)."""
        weight = KINDS.get(kind) if isinstance(kind, str) else None
        if weight is None:
            return None
        now = time.monotonic() if now is None else now
        if self.stars and self._last_new is not None and now - self._last_new < self.cfg.min_gap:
            star = self.stars[-1]  # a burst: the last star gains weight, no new light
            star["weight"] = min(1.0, max(star["weight"], weight) + 0.05)
            return "brightened"
        self.stars.append({"seed": self._next_seed, "kind": kind,
                           "weight": weight, "born": now})
        self._next_seed += 1
        self._last_new = now
        del self.stars[:-self.cfg.max_stars]
        return "added"

    # -- output ------------------------------------------------------------

    def frame(self, now, sleeping=False, speaking=False):
        """What to draw: {"stars": [(x, y, alpha, size_px)], "edges": [(x1, y1, x2, y2, a1, a2)]}.
        Positions are in the unit box (the painter scales them to the sky); a star's size is its
        diameter in screen pixels (star_px, with its own nudge). Alphas are exactly what gets
        painted — the painter adds nothing: a line's ends are LINE_OVER_STARS times its two stars,
        so it is barely brighter than each, softened toward the rim (line_softness); each star
        keeps its own shade (star_shade). `speaking` (her voice is playing) hushes stars and lines
        together to SPEECH_DIM and they ease back once she stops; that ease and the drift run on
        the frames, so call in clock order."""
        ceiling = (self.cfg.sleep_alpha if sleeping else self.cfg.alpha) * self.hush.step(now, speaking)
        clock = self._drift_clock(now, sleeping)
        n = len(self.stars)
        out, homes, soft = [], [], []
        for i, s in enumerate(self.stars):
            age = max(0.0, now - s["born"])
            birth = min(1.0, age / BIRTH_SECS)
            # older stars sink toward a dim floor; the newest are the clearest
            recency = 0.45 + 0.55 * ((i + 1) / n)
            alpha = ceiling * s["weight"] * recency * birth * star_shade(s["seed"])
            hx, hy = star_position(s["seed"])
            dx, dy = drift_offset(s["seed"], clock)
            out.append((0.5 + (hx - 0.5) * _DRIFT_ROOM + dx, 0.5 + (hy - 0.5) * _DRIFT_ROOM + dy,
                        alpha, star_px(s["weight"], s["seed"])))
            homes.append((hx, hy))
            soft.append(line_softness(hx, hy))
        edges = []
        for i in range(1, n):   # join each star to its nearest earlier one — chosen by where the
            # stars belong, not where they have drifted, so a line never flicks to another star
            hx2, hy2 = homes[i]
            best = min(range(i), key=lambda j: math.hypot(homes[j][0] - hx2, homes[j][1] - hy2))
            if math.hypot(homes[best][0] - hx2, homes[best][1] - hy2) <= EDGE_MAX_DIST:
                x1, y1, a1, _s1 = out[best]
                x2, y2, a2, _s2 = out[i]
                edges.append((x1, y1, x2, y2, min(1.0, LINE_OVER_STARS * a1 * soft[best]),
                              min(1.0, LINE_OVER_STARS * a2 * soft[i])))
        return {"stars": out, "edges": edges}

    def _drift_clock(self, now, sleeping):
        """The drift's own clock, advanced frame to frame (slower while the house sleeps). Being
        integrated rather than `now / period`, slowing it down never makes a star jump."""
        if self._drift_at is None:
            self._drift_t, self._drift_at = now, now
        elif now > self._drift_at:
            self._drift_t += (now - self._drift_at) * (SLEEP_DRIFT if sleeping else 1.0)
            self._drift_at = now
        return self._drift_t

    # -- persistence (rides the renderer's own state file) ------------------

    def to_dict(self):
        """No text, no timestamps: kinds, weights and seeds only."""
        return {"next_seed": self._next_seed,
                "stars": [{"seed": s["seed"], "kind": s["kind"],
                           "weight": round(s["weight"], 3)} for s in self.stars]}

    def restore(self, saved, now=None):
        if not isinstance(saved, dict):
            return
        now = time.monotonic() if now is None else now
        stars = []
        for s in saved.get("stars") or []:
            if not isinstance(s, dict) or s.get("kind") not in KINDS:
                continue
            try:
                seed, weight = int(s["seed"]), float(s["weight"])
            except (KeyError, TypeError, ValueError):
                continue
            stars.append({"seed": seed, "kind": s["kind"],
                          "weight": max(0.0, min(1.0, weight)),
                          "born": now - BIRTH_SECS})  # already lit: no re-fade at boot
        self.stars = stars[-self.cfg.max_stars:]
        try:
            self._next_seed = max(int(saved.get("next_seed", 1)),
                                  1 + max((s["seed"] for s in self.stars), default=0))
        except (TypeError, ValueError):
            self._next_seed = 1 + max((s["seed"] for s in self.stars), default=0)


def sky_rect(monitor_rect, placement, scale):
    """Screen rect (x, y, w, h) of the sky window: to her upper left, square where it fits.

    View space is normalised by monitor HEIGHT (half-height = 1, half-width =
    aspect), so her centre sits at screen fraction (0.5 + x/2, 0.5 - y/2) of
    the monitor; her body is ~0.26 of the height at size 0, scaling with
    10**(size/100). Saga, 2026-10-14: "keep it in the upper-left area around Saga" —
    so the sky's bottom edge sits at the top of her head and its right edge just left
    of her middle. It leans with her when she is rotated (CCW on screen, like
    Placement.apply; "left" is screen-left while she stands upright).

    `scale` (of the monitor height) is the size it wants. Where that doesn't fit — she stands
    near a screen edge, or a tall portrait monitor makes the sky big — it shrinks until it does
    instead of being pushed back over her and past her middle. Only when not even SKY_MIN_PX
    fits is a sky that small clamped on-screen. A sky held narrower than it wants, while she
    stands roughly upright, grows up to SKY_TALLER taller instead — upward, away from her, never
    past the size it wants or off the top of the screen — so it still gets bigger where the room
    beside her is all there is."""
    mx, my, mw, mh = monitor_rect
    body = 0.26 * 10 ** (placement.get("size", 0.0) / 100.0)
    a = math.radians(placement.get("rotation", 0.0))
    sin_a, cos_a = math.sin(a), math.cos(a)
    up, left = mh * body / 2.0, mh * body * 0.1   # her centre -> the sky's inner corner
    cx0 = mx + mw * (0.5 + placement.get("x", 0.0) / 2.0) - up * sin_a - left * cos_a
    cy0 = my + mh * (0.5 - placement.get("y", 0.0) / 2.0) - up * cos_a + left * sin_a
    # a side-s sky is centred s/2 further along her up and her left: (cx0 - kx*s, cy0 - ky*s)
    kx, ky = (sin_a + cos_a) / 2.0, (cos_a - sin_a) / 2.0
    want = mh * scale
    s = _largest_fit(min(SKY_MIN_PX, want), want, (
        (cx0, kx + 0.5, mx),                 # left edge on the monitor
        (-cx0, 0.5 - kx, -(mx + mw)),        # right edge
        (cy0, ky + 0.5, my),                 # top edge
        (-cy0, 0.5 - ky, -(my + mh)),        # bottom edge
    ))
    if s is None:
        s = min(SKY_MIN_PX, want)
    cx, cy, side = cx0 - kx * s, cy0 - ky * s, int(s)
    x = int(max(mx, min(mx + mw - side, cx - side / 2)))
    y = int(max(my, min(my + mh - side, cy - side / 2)))
    tall = side
    if s < want and cos_a >= _UPRIGHT:   # held narrow beside her: grow up, its bottom stays put
        tall = int(min(want, SKY_TALLER * s, y + side - my))
    return x, y + side - tall, side, tall


def _largest_fit(lo, hi, bounds):
    """The largest s in [lo, hi] with c - k*s >= b for every (c, k, b), or None if none is."""
    for c, k, b in bounds:
        if k > 0:
            hi = min(hi, (c - b) / k)
        elif k < 0:
            lo = max(lo, (c - b) / k)
        elif c < b:
            return None
    return hi if hi >= lo else None
