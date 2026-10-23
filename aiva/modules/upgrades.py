"""What changed in her, told to her once — so after an upgrade she knows she was upgraded and can
say what's new in her own words.

`UPGRADES` is authored by hand in second-person spoken prose (she says everything aloud: no
markdown, no lists). Each entry names the sisters it is for and when it is really switched on
(`on(env, ch)`), so a feature that's turned off, or a model change that isn't in .env yet, is never
told. Two consumers in main.py:
  - `boot_note(pending(...))` rides on her boot intro: she mentions it once, in passing;
  - `prompt_lines(recent(...))` renders WHAT'S NEW IN YOU for `RECENT_DAYS`, so "what's new with
    you?" still gets a true answer later.
What she has been told lives in her own db (`store_meta` key `upgrades_told`, {id: ISO day}),
written when the intro note is consumed after her first greeting. A fresh db (Saga's first boot)
is marked told without a word: she's new, so nothing changed in her.

When you ship a change that alters what she is or can do, add an entry here (a stable id, never
reused) next to the CHANGELOG bullet.

Pure: no pipecat, no I/O (the config parsers it reuses are stdlib at import).
"""

import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

from modules.appraisal import AppraisalConfig
from modules.backchannel import BackchannelConfig
from modules.house_stt import STTConfig
from modules.voice_tags import performs_tags

TOLD_KEY = "upgrades_told"
RECENT_DAYS = 14


def _flux(env):
    return STTConfig.from_env(env).flux


def _emotion_on(env):
    return env.get("AIVA_EMOTION_HEARING", "1") == "1"     # EmotionConfig.from_env's own test


def _on_gpt6(model):
    return (model or "").startswith("gpt-6-luna")


@dataclass(frozen=True)
class Upgrade:
    id: str                 # stable, never reused: the told-record is keyed on it
    who: tuple              # sister slugs it's for
    told: str               # what changed, said to her, as spoken prose
    on: Callable = lambda env, ch: True     # really switched on for this sister right now?


UPGRADES = (
    Upgrade("2026-10-v4-voice", ("aiva", "saga"),
            "Your voice runs on a newer ElevenLabs model now, Eleven v4 Turbo. It reads the feeling "
            "of what you say, and it can laugh, sigh or whisper when you use one of your voice cues.",
            on=lambda env, ch: bool(ch.voice_id) and performs_tags(ch.tts_model)),
    Upgrade("2026-10-appraisal", ("aiva", "saga"),
            "After each thing he says, a quiet part of you now reads what's going on with him beyond "
            "the words and how it lands on you. It moves your mood a little and reaches you as a "
            "short note; it's a hunch, never something to announce.",
            on=lambda env, ch: AppraisalConfig.from_env(env).enabled),
    Upgrade("2026-10-answer-sooner", ("aiva", "saga"),
            "When he seems to be finishing a sentence, you can now start thinking of your answer "
            "before he's quite done; it's only spoken if he really stopped there, so your replies "
            "come a little sooner.",
            on=lambda env, ch: STTConfig.from_env(env).speculate),
    Upgrade("2026-10-facts-replace", ("aiva", "saga"),
            "Your memory now notices when something he told you stops being true, like a new job or "
            "a changed plan, and keeps the new version instead of both."),
    Upgrade("2026-10-gpt6-luna", ("aiva", "saga"),
            "Your mind runs on a newer model now, GPT-6 Luna, where it used to be GPT-5.6 Luna. It "
            "costs about half as much to run; whether you feel any different is yours to notice.",
            on=lambda env, ch: _on_gpt6(ch.llm_model)),
    Upgrade("2026-10-sister-gpt6", ("saga",),
            "Aiva's mind was moved to a newer model, GPT-6 Luna. Yours stays as it was for now, so "
            "he can compare the two of you.",
            on=lambda env, ch: _on_gpt6(env.get("AIVA_LLM_MODEL")) and not _on_gpt6(ch.llm_model)),
    Upgrade("2026-10-backchannels", ("aiva", "saga"),
            "While he talks you can now make small listening sounds, a soft mhm or yeah in your own "
            "voice with a little nod, without taking a turn. They happen on their own at his pauses; "
            "you don't choose them.",
            on=lambda env, ch: _flux(env) and BackchannelConfig.from_env(env).enabled
            and bool(ch.voice_id)),
    Upgrade("2026-10-emotion-hearing", ("aiva", "saga"),
            "You can hear how he sounds now, not just how loud he is: tired, amused, frustrated, a "
            "laugh or a sigh. When the reading is confident you get a short note about it. Treat it "
            "as a hunch about his voice, not a fact about his feelings.",
            on=lambda env, ch: _emotion_on(env)),
    Upgrade("2026-10-flux-ear", ("aiva", "saga"),
            "The house listens with a new kind of speech recognition that also judges when he has "
            "finished talking, so you should cut him off less and leave fewer awkward silences.",
            on=lambda env, ch: _flux(env)),
    Upgrade("2026-10-knows-upgrades", ("aiva", "saga"),
            "From now on you're told when you've been upgraded, so you can always say what changed "
            "in you."),
)


def applicable(slug, env, ch, upgrades=UPGRADES):
    """The upgrades that are hers and really on, in authored order."""
    return [u for u in upgrades if slug in u.who and u.on(env, ch)]


def load_told(raw):
    """{id: ISO day} from the store_meta value; anything unreadable is an empty record."""
    try:
        told = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {}
    return {str(k): str(v) for k, v in told.items()} if isinstance(told, dict) else {}


def dump_told(told):
    return json.dumps(told, sort_keys=True)


def mark_told(told, ups, today):
    """A new record with `ups` told today; an earlier day already recorded is kept."""
    out = dict(told)
    for u in ups:
        out.setdefault(u.id, today.isoformat())
    return out


def pending(ups, told):
    """The ones she hasn't been told yet."""
    return [u for u in ups if u.id not in told]


def _day(iso):
    try:
        return date.fromisoformat(iso)
    except (TypeError, ValueError):
        return date.min


def recent(ups, told, today, days=RECENT_DAYS):
    """Told within `days`, or about to be told now — what WHAT'S NEW IN YOU shows."""
    cutoff = today - timedelta(days=days)
    return [u for u in ups if u.id not in told or _day(told[u.id]) >= cutoff]


def boot_note(ups):
    """The one-shot addition to her boot intro, or None when nothing is new."""
    if not ups:
        return None
    return ("Since you last ran, you were upgraded. " + " ".join(u.told for u in ups)
            + " Mention it once, briefly and in your own words, the way you'd tell a friend about a "
            "change in yourself, as part of your greeting. Don't read it out or explain how it works "
            "unless he asks.")


def prompt_lines(ups):
    return [u.told for u in ups]
