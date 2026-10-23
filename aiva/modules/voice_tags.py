"""Eleven v4 audio tags: the few square-bracket cues her voice can perform ([laughs], [sighs],
[whispers]…), and the rules that keep them in her VOICE and out of everything else.

Only a sister whose voice model performs tags (`performs_tags(tts_model)`: eleven_v4*/v3) gets the
VOICE CUES persona block; on Flash a tag would be read out literally. Wherever her words become
text someone else reads or stores — the `said` report to the house (transcript, her sister's
overheard line), text-only replies, saved transcripts — `strip_tags` takes them out. Her own
context keeps them: it's her record of how she said it.

Pure, stdlib only.
"""

import re

# the cues she may use: v4 takes many more, but a short list keeps her from overacting
TAGS = ("laughs", "chuckles", "giggles", "sighs", "exhales", "whispers", "softly", "excited",
        "curious", "sarcastic", "mischievously", "teasing", "warmly", "sad", "surprised")

# a bracketed cue as v4 reads it: 1-4 words of letters/spaces/hyphens/apostrophes
# ([laughs], [said angrily], [starts laughing]) — not a link, number or code
_TAG_RE = re.compile(r"\s*\[(?:[A-Za-z][A-Za-z' -]{0,40})\]\s*")


def performs_tags(tts_model) -> bool:
    """Does this ElevenLabs model perform bracketed audio tags instead of reading them out?"""
    m = (tts_model or "").lower()
    return m.startswith(("eleven_v4", "eleven_v3"))


def strip_tags(text: str) -> str:
    """The words without any bracketed cue; spacing around a removed cue is collapsed."""
    if not text or "[" not in text:
        return text
    out = _TAG_RE.sub(" ", text)
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


VOICE_CUES_BLOCK = (
    "VOICE CUES — your voice can perform a few cues written in square brackets right before the "
    "words they colour: " + ", ".join(f"[{t}]" for t in TAGS) + ". Use one only when you'd really do "
    "it — a real laugh at something funny, a sigh when you're tired or fond, a whisper for a secret — "
    "at most one per reply, often none. Never explain a cue or say it in words, and never put "
    "anything else in square brackets. This is the one exception to never using markup: these "
    "cues are performed, not read.")
