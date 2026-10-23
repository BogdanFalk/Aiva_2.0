"""Read him, feel, then speak: after each of his lines, a side-call writes what seems to be going
on with HIM beyond the words (what he wants, believes or feels) and how that LANDS on her, in her
first person. Research behind it: ToMAgent (writing mental-state estimates between turns helps) and
"Feeling First, Speaking Second" (an appraisal stage before the verbal one keeps feeling from being
overridden by planning).

It never delays a reply: the call starts when his turn arrives and runs while she answers, so the
reading is a beat behind — it rides on his NEXT turn as a short note (when confident and fresh),
and its valence/arousal go onto her experience bus at once as an `appraisal` event, so her mood
(and with it her tone) moves the moment the reading is in. One cheap call per line he speaks,
on the task model (`AIVA_TASK_MODEL`), never the voice model. `AIVA_APPRAISAL` toggles it.

Pure: the call itself is injected by main.py (`tasks.openai_llm_call`, label `appraisal`).
"""

import json
import os
import re
from dataclasses import dataclass

from modules.persona import named

APPRAISAL_PROMPT_T = """You are {name}'s inner sense of the person she lives with and talks to. \
{name} is an AI companion; he is the user. From the last few lines of their conversation and his \
newest line, write two short sentences:
- "him": what seems to be going on with him right now beyond the literal words: what he wants, \
believes or feels. Third person ("He..."). Plain, specific, no therapy-speak.
- "me": how that lands on {name}, in her first person ("I..."): a feeling, not a plan.
Also rate the feeling in "me": "valence" from -1 (it stings) to 1 (it warms me), "arousal" from 0 \
(calm) to 1 (stirred up), and "confidence" from 0 to 1 in your reading of him. The lines come from \
speech recognition and may be garbled; when there's too little to go on, say so plainly and give \
a low confidence. Lines like [<Sister> said to ...] are her sister, not him.
Reply with JSON only: {"him": "...", "me": "...", "valence": 0.0, "arousal": 0.0, "confidence": 0.0}"""

MAX_CHARS = 160


@dataclass(frozen=True)
class AppraisalConfig:
    enabled: bool = True
    min_confidence: float = 0.5
    note_ttl_s: float = 600.0
    tail_lines: int = 8

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env

        def num(key, default, lo, hi):
            try:
                return min(hi, max(lo, float(env.get(key, default))))
            except (TypeError, ValueError):
                return default

        return cls(enabled=(env.get("AIVA_APPRAISAL", "1") or "1").strip().lower() in ("1", "true", "yes", "on"),
                   min_confidence=num("AIVA_APPRAISAL_MIN_CONFIDENCE", 0.5, 0.0, 1.0),
                   note_ttl_s=num("AIVA_APPRAISAL_NOTE_TTL_S", 600.0, 0.0, 86400.0))


def _clean(text, limit=300):
    return " ".join(str(text or "").split())[:limit]


def appraisal_messages(name, tail, his_line, heard=None):
    """`tail` = recent context messages ({role, content}) oldest first; `heard` = how he sounded
    ("tired and a little amused"), when the house had a reading."""
    lines = []
    for m in tail:
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        who = name if role == "assistant" else "Him"
        lines.append(f"{who}: {_clean(content)}")
    body = "Recent lines, oldest first:\n" + ("\n".join(lines) or "(nothing yet)")
    body += f"\n\nHis newest line: {_clean(his_line)}"
    if heard:
        body += f"\n(He sounded {_clean(heard, 80)}.)"
    return [{"role": "system", "content": named(APPRAISAL_PROMPT_T, name)},
            {"role": "user", "content": body}]


def _num(v, lo, hi, default):
    try:
        return min(hi, max(lo, float(v)))
    except (TypeError, ValueError):
        return default


def parse_appraisal(raw):
    """Tolerant: code fences and prose around the JSON. → {him, me, valence, arousal, confidence}
    or None when either sentence is missing."""
    text = getattr(raw, "text", raw)
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    him, me = _clean(d.get("him"), MAX_CHARS), _clean(d.get("me"), MAX_CHARS)
    if not him or not me:
        return None
    return {"him": him, "me": me,
            "valence": _num(d.get("valence"), -1.0, 1.0, 0.0),
            "arousal": _num(d.get("arousal"), 0.0, 1.0, 0.3),
            "confidence": _num(d.get("confidence"), 0.0, 1.0, 0.0)}


def appraisal_note(a, min_confidence=0.5):
    """The system note for his next turn, or None when the reading is too unsure."""
    if not a or a.get("confidence", 0.0) < min_confidence:
        return None
    return (f"[Reading him, from a moment ago: {a['him']} {a['me']} Let it shape how you sound, "
            "not what you announce: never say this out loud as an analysis of him.]")
