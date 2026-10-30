"""Her drawing hand: a whole picture drawn in the background while she keeps talking.

2026-10-27 he asked the sisters for "more detail" and Aiva drew 50 strokes with draw_stroke, one
voice-model round trip each: two minutes of silence until the house's 120 s turn cap cut her off
mid-drawing, and the stroke calls pushed 33 messages of the evening out of her context. Now
`draw_picture(idea)` hands the idea to `Artist`: ONE side-model call (the task model, its own token
bucket) reads what's on the canvas (`DrawingBoard.on_canvas()`, the window's autosave: his strokes and
her sister's too) and answers with the strokes as JSON. They're drawn in her ink at a watchable pace
through the same `DrawingBoard.stroke` (journaled like any other), and an announce tells her when
it's done. The tool returns at once, so she says one line and keeps talking, and none of the points
ever enter her voice context.

Shapes the model may use, expanded and clamped to the canvas here: `{"points": [[x, y], ...]}` is a
line through the points (one point is a dot), `{"ellipse": [cx, cy, rx, ry]}` an ellipse or circle,
and `"arc": [from°, to°]` beside it part of one. Models draw curves badly as raw points, so the
circles are ours. `color` (#rrggbb) and `width` are optional on each.

`canvas_brief`, `expand`, `parse_picture` and `artist_messages` are pure; `Artist` does the I/O.
"""

import asyncio
import json
import math
import re

from modules.drawing import CANVAS_SIZE, ink
from modules.persona import named

MAX_STROKES = 150         # one picture
MAX_POINTS = 400          # one stroke
MAX_WIDTH = 60.0
BRIEF_STROKES = 120       # the newest strokes the model is shown
BRIEF_POINTS = 10         # sampled points per stroke it's shown
PACE_S = 0.12             # between strokes: he watches it being drawn
MAX_TOKENS = 12000
_COLOR = re.compile(r"#[0-9a-fA-F]{6}")

ARTIST_PROMPT_T = """You are {name}'s drawing hand. {name} draws on a canvas she shares with her sister and with him, and she has asked you to draw this:

IDEA: {idea}

The canvas is {w} pixels wide and {h} tall; x goes right and y goes DOWN from the top-left corner (0, 0), and every point must be on it. The background is white. This is what's on it now, oldest first; strokes by "user" are his, the others are the sisters':
{brief}

Draw the idea as strokes in drawing order, building on what's there: put things where they make sense next to what's already drawn, and never draw over his strokes unless the idea asks for it. Use as many strokes as real detail needs (at most {max_strokes}): big shapes first, then features, then texture and small touches. Each stroke is one of:
- {"points": [[x, y], ...]}: a line through the points (2 to 200 points; one point is a dot)
- {"ellipse": [cx, cy, rx, ry]}: an ellipse, or a circle when rx equals ry
- {"ellipse": [cx, cy, rx, ry], "arc": [from_degrees, to_degrees]}: part of one (0 is to the right, 90 is DOWN)
Each may add "color": "#rrggbb" (otherwise {name}'s own ink, {ink}) and "width": 1 to 60 pixels (default 4).

Reply with JSON only: {"strokes": [...], "said": "one short sentence in {name}'s spoken voice about what she added"}"""


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _on_canvas(x, y):
    w, h = CANVAS_SIZE
    return [min(max(float(x), 0.0), float(w)), min(max(float(y), 0.0), float(h))]


def canvas_brief(strokes, limit=BRIEF_STROKES, sample=BRIEF_POINTS):
    """What's on the canvas, for the prompt: one line per stroke (the newest `limit`), who drew it,
    its ink and width, its box and a few of its points in order. Enough to see the shapes and draw
    around them, for a fraction of the tokens."""
    if not strokes:
        return "(nothing yet: the canvas is blank)"
    shown = strokes[-limit:]
    lines = [f"({len(strokes) - len(shown)} older strokes not listed)"] if len(strokes) > len(shown) else []
    for s in shown:
        pts = s.get("points") or []
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        step = max(1, math.ceil(len(pts) / sample))
        picked = pts[::step]
        if picked[-1] is not pts[-1]:
            picked.append(pts[-1])
        through = " ".join(f"{round(x)},{round(y)}" for x, y in picked)
        lines.append(f"{s.get('author')} {s.get('color')} w{s.get('width', 4):g}: "
                     f"box {round(min(xs))},{round(min(ys))}-{round(max(xs))},{round(max(ys))} "
                     f"through {through}")
    return "\n".join(lines)


def expand(item):
    """One shape from the model → (points, color, width) on the canvas, or None if it's unusable."""
    if not isinstance(item, dict):
        return None
    color = item.get("color")
    color = color if isinstance(color, str) and _COLOR.fullmatch(color) else None
    width = item.get("width")
    width = min(max(float(width), 1.0), MAX_WIDTH) if _num(width) else None
    if "ellipse" in item:
        e = item["ellipse"]
        if not (isinstance(e, list) and len(e) == 4 and all(map(_num, e))):
            return None
        cx, cy, rx, ry = e[0], e[1], max(abs(e[2]), 0.5), max(abs(e[3]), 0.5)
        arc = item.get("arc")
        a0, a1 = (arc if isinstance(arc, list) and len(arc) == 2 and all(map(_num, arc)) else (0, 360))
        span = max(-360.0, min(360.0, float(a1) - float(a0)))
        if span == 0:
            return None
        around = 2 * math.pi * math.sqrt((rx * rx + ry * ry) / 2) * abs(span) / 360
        n = int(min(120, max(8, around / 8)))
        points = [_on_canvas(cx + rx * math.cos(math.radians(a0 + span * i / n)),
                             cy + ry * math.sin(math.radians(a0 + span * i / n))) for i in range(n + 1)]
    elif "points" in item:
        raw = item["points"]
        if not isinstance(raw, list):
            return None
        pts = [p for p in raw if isinstance(p, list) and len(p) == 2 and all(map(_num, p))]
        if not pts:
            return None
        if len(pts) > MAX_POINTS:                       # thinned evenly, the ends kept
            step = len(pts) / MAX_POINTS
            pts = [pts[int(i * step)] for i in range(MAX_POINTS - 1)] + [pts[-1]]
        points = [_on_canvas(x, y) for x, y in pts]
    else:
        return None
    return points, color, width


def parse_picture(raw, max_strokes=MAX_STROKES):
    """The model's reply → {"strokes": [(points, color, width), ...], "said": str}, or None when
    nothing drawable came back. Tolerant: code fences and prose around the JSON; a shape that doesn't
    parse is dropped, not fatal."""
    text = getattr(raw, "text", raw)
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    items = data.get("strokes") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    strokes = [s for s in map(expand, items[:max_strokes]) if s is not None]
    if not strokes:
        return None
    said = data.get("said")
    return {"strokes": strokes, "said": said.strip()[:200] if isinstance(said, str) else ""}


def artist_messages(name, idea, strokes, own_ink=None, max_strokes=MAX_STROKES):
    w, h = CANVAS_SIZE
    prompt = named(ARTIST_PROMPT_T, name, idea=idea.strip(), w=w, h=h, brief=canvas_brief(strokes),
                   max_strokes=max_strokes, ink=own_ink or "her default ink")
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": "Draw it now. JSON only."}]


class Artist:
    """Draws one picture at a time in the background on `board` (a `DrawingBoard`). `llm_call` is
    the side model (`tasks.openai_llm_call` on the task model); `announce(text)` tells her it's done
    (wired after boot through `attach_voice`); `spawn(coro)` starts the job (default: the running
    loop's `create_task`)."""

    def __init__(self, board, llm_call, *, name="Aiva", announce=None, log=print, spawn=None,
                 pace_s=PACE_S, max_strokes=MAX_STROKES, timeout_s=180.0):
        self.board = board
        self._llm = llm_call
        self._name = name
        self._announce = announce
        self._log = log
        self._spawn = spawn
        self._pace = pace_s
        self._max = max_strokes
        self._timeout = timeout_s
        self._job = None
        self._idea = None

    def attach_voice(self, announce):
        self._announce = announce

    @property
    def busy(self):
        return self._job is not None and not self._job.done()

    def start(self, idea):
        """Start drawing `idea` and return at once with a speakable dict."""
        idea = (idea or "").strip()
        if not idea:
            return {"success": False, "error": "say what to draw"}
        if self.busy:
            return {"success": False, "error": "your hand is still drawing the last picture; you'll "
                                               "get a note when it's done", "drawing": self._idea}
        self._idea = idea
        spawn = self._spawn or asyncio.get_running_loop().create_task
        self._job = spawn(self._run(idea))
        self._log(f"artist: drawing {idea[:160]!r}")
        return {"success": True, "status": "drawing it now, in the background",
                "note": "Say one short line about what you're drawing, then carry on talking; "
                        "you'll get a note when it's finished."}

    async def _run(self, idea):
        try:
            done = await self._draw(idea)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            done = {"drawn": 0, "error": str(e) or type(e).__name__}
        self._log(f"artist: {done.get('drawn', 0)}/{done.get('planned', 0)} strokes"
                  + (f", error: {done['error']}" if done.get("error") else ""))
        if self._announce is not None:
            await self._announce(self._note(idea, done))

    async def _draw(self, idea):
        strokes = await self.board.on_canvas()
        reply = await asyncio.wait_for(
            self._llm(artist_messages(self._name, idea, strokes, ink(self.board.author), self._max),
                      json_mode=True, max_tokens=MAX_TOKENS, temperature=0.7), self._timeout)
        picture = parse_picture(reply, self._max)
        if picture is None:
            return {"drawn": 0, "error": "your drawing hand came back with nothing it could draw"}
        drawn, misses, error = 0, 0, None
        for points, color, width in picture["strokes"]:
            out = await self.board.stroke(points, color=color, width=width)
            if out.get("success"):
                drawn, misses = drawn + 1, 0
            else:
                misses, error = misses + 1, out.get("error")
                if misses >= 3:                 # the window is gone or refusing: stop, don't flail
                    break
            await asyncio.sleep(self._pace)
        return {"drawn": drawn, "planned": len(picture["strokes"]), "said": picture["said"],
                "error": error if drawn < len(picture["strokes"]) else None}

    @staticmethod
    def _note(idea, done):
        if not done.get("drawn"):
            return (f"[Your drawing of \"{idea}\" didn't happen: {done.get('error') or 'nothing was drawn'}. "
                    "Tell him in one sentence.]")
        said = f" Your hand's words for it: \"{done['said']}\"." if done.get("said") else ""
        partly = f" It stopped partway: {done['error']}." if done.get("error") else ""
        return (f"[Your drawing is done: \"{idea}\".{said}{partly} Tell him in one short sentence what "
                "you added, in your own words: no stroke counts, no coordinates.]")
