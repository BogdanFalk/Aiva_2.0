"""Who a process IS: one `Character` per sister (Aiva, Saga) — her name and how STT mangles it,
her persona floor, and every path/voice/config that must differ between two cores on one PC.

Stdlib-only (persona is stdlib-only too), so the dashboard and tests import it without pipecat.
`Character.from_env("aiva")` reproduces today's unprefixed env keys and defaults EXACTLY (Aiva's
files never move); `from_env("saga")` reads `SAGA_*`, falling back to shared keys only for the TTS
model, the LLM model and the library. main.py loads .env — this module just reads os.environ.
"""

import os
import re
from dataclasses import dataclass

from modules import persona

AIVA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWN = ("aiva", "saga")

# How Deepgram tends to spell each name (keyterm boosting helps, but not always). Extend per
# character without code changes via {PREFIX}_NAME_VARIANTS (comma list). The loose regexes catch
# the vowel/consonant SHAPE of a name that no common English word has ("eyvah", "sagah") — and are
# disjoint from each other (sage/sugar/sarah/stage are not Saga; avocado is not Aiva).
AIVA_NAME_VARIANTS = ("aiva", "ava", "eva", "iva", "ayva", "eiva", "aeva", "aiwa",
                      "avah", "evah", "aivah", "hey va", "hey iva")
AIVA_LOOSE = r"[aeiy]{1,3}[vwf]ah?"
SAGA_NAME_VARIANTS = ("saga", "sagah", "sarga", "sagar", "sager", "sega", "seiga", "sa ga", "hey saga")
SAGA_LOOSE = r"s[ae]+r?g+a+h?r?"

# a sister's spoken line as it lands in the OTHER sister's context (house protocol) — prompts that
# must tell her words from the user's match on this
OVERHEARD_RE = re.compile(r'^\[(Aiva|Saga) said to (him|you): ')

# Deepgram keyterms every sister wants boosted besides her own name (today's list, minus "Aiva")
SHARED_STT_KEYTERMS = ("Claude Code", "PowerShell", "Frank", "Bogdan", "Fianu")

_DISPLAY = {"aiva": "Aiva", "saga": "Saga"}

# InnerConfig's knobs and Aiva's defaults (inner.InnerConfig.from_env reads the same AIVA_MOOD_*
# keys); a self-forming sister starts from a neutral baseline unless her own SAGA_MOOD_* say otherwise
_MOOD_DEFAULTS = {"base_valence": 0.15, "base_arousal": 0.50, "base_social": 0.10, "halflife_h": 6.0,
                  "salience_floor": 0.15, "gain": 0.60, "material_eps": 0.05}
_NEUTRAL_MOOD_BASE = {"base_valence": 0.0, "base_arousal": 0.5, "base_social": 0.0}
_TTS_MODEL_DEFAULT = "eleven_flash_v2_5"
_LLM_MODEL_DEFAULT = "gpt-5.6-luna"      # was gpt-4.1-mini, which OpenAI shuts down 2027-01-14


def display_name(slug):
    """'aiva' -> 'Aiva'."""
    s = (slug or "").strip().lower()
    return _DISPLAY.get(s, s.capitalize())


def _here(*parts):
    return os.path.join(AIVA_DIR, *parts)


def _env_bool(key, default):
    v = os.getenv(key)
    if v is None or not v.strip():
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


def _env_float(key, default):
    v = os.getenv(key)
    if v is None or not v.strip():
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _csv(key):
    return tuple(v.strip() for v in os.getenv(key, "").split(",") if v.strip())


def _variants(base, key):
    return tuple(base) + tuple(v.lower() for v in _csv(key))


def _mood_cfg(prefix, base_overrides):
    out = {}
    for k, d in _MOOD_DEFAULTS.items():
        out[k] = _env_float(f"{prefix}_MOOD_{k.upper()}", base_overrides.get(k, d))
    return out


@dataclass(frozen=True)
class Character:
    name: str                    # "Aiva"
    slug: str                    # "aiva" — db stamp, log prefix, protocol id
    sisters: tuple               # display names of the other sisters, e.g. ("Saga",)
    name_variants: tuple         # lowercase STT spellings of her name (+ env extras)
    loose_regex: str             # fullmatch on a single word → plausibly her name
    self_forming: bool           # temperament forms from experience (no authored bullets)
    personality: str             # her persona floor (persona.PERSONALITY / SAGA_FLOOR)
    baseline_traits: tuple       # becoming's anti-contradiction floor (() when self-forming)
    baseline_tastes: bool        # seed preferences from the baseline at boot
    identity_tag: str            # "the same cheeky Aiva" / "still Saga" in persona guards
    db_path: str
    backup_stem: str             # backups/<stem>-YYYYMMDD-HHMMSS.db
    inner_state_path: str
    cat_state_path: str | None   # only the sister who owns the desktop cat
    avatar_state_path: str
    avatar_model: str
    voice_id: str | None         # ElevenLabs voice; None → she can only talk text-only
    tts_model: str
    llm_model: str               # the model she thinks and talks with (Saga may differ: SAGA_LLM_MODEL)
    space_dir: str
    library_dir: str             # shared between sisters (rows/cursors live per db)
    capability_exclude: tuple    # capabilities.CAPABILITIES names she doesn't have
    mood_cfg: dict               # kwargs for inner.InnerConfig
    speaker_seed_primary: bool   # first voice ever is silently the primary user (Aiva) or asked (Saga)
    speaker_primary_name: str
    wake_model: str
    stt_keyterms: tuple

    @property
    def prefix(self):
        return self.slug.upper()

    @property
    def sister(self):
        """Her (first) sister's display name, or None."""
        return self.sisters[0] if self.sisters else None

    @classmethod
    def from_env(cls, slug="aiva"):
        slug = (slug or "").strip().lower()
        library = os.getenv("AIVA_LIBRARY_DIR") or _here("library")
        if slug == "aiva":
            return cls(
                name="Aiva", slug="aiva", sisters=("Saga",),
                name_variants=_variants(AIVA_NAME_VARIANTS, "AIVA_NAME_VARIANTS"),
                loose_regex=AIVA_LOOSE,
                self_forming=False,
                personality=persona.PERSONALITY,
                baseline_traits=tuple(persona.baseline_traits()),
                baseline_tastes=True,
                identity_tag="the same cheeky Aiva",
                db_path=_here("memory.db"),
                backup_stem="memory",
                inner_state_path=os.getenv("AIVA_INNER_STATE") or _here("inner_state.json"),
                cat_state_path=_here("cat_state.json"),
                avatar_state_path=_here("avatar_state.json"),
                avatar_model=os.getenv("AIVA_AVATAR_MODEL")
                or _here("assets", "avatar", "ariu", "ariu.model3.json"),
                voice_id=os.getenv("ELEVENLABS_VOICE_ID") or None,
                tts_model=os.getenv("ELEVENLABS_MODEL") or _TTS_MODEL_DEFAULT,
                llm_model=os.getenv("AIVA_LLM_MODEL") or _LLM_MODEL_DEFAULT,
                space_dir=os.getenv("AIVA_SPACE_DIR") or _here("space"),
                library_dir=library,
                capability_exclude=(),
                mood_cfg=_mood_cfg("AIVA", {}),
                speaker_seed_primary=_env_bool("AIVA_SPEAKER_SEED_PRIMARY", True),
                speaker_primary_name=os.getenv("AIVA_SPEAKER_PRIMARY_NAME") or "Frank",
                wake_model=os.getenv("AIVA_WAKE_MODEL", "hey_jarvis"),
                stt_keyterms=("Aiva",) + SHARED_STT_KEYTERMS + _csv("AIVA_STT_KEYTERMS"),
            )
        if slug == "saga":
            return cls(
                name="Saga", slug="saga", sisters=("Aiva",),
                name_variants=_variants(SAGA_NAME_VARIANTS, "SAGA_NAME_VARIANTS"),
                loose_regex=SAGA_LOOSE,
                self_forming=True,
                personality=persona.SAGA_FLOOR,
                baseline_traits=(),
                baseline_tastes=False,
                identity_tag="still Saga",
                db_path=os.getenv("SAGA_DB") or _here("saga.db"),
                backup_stem="saga",
                inner_state_path=os.getenv("SAGA_INNER_STATE") or _here("saga_inner_state.json"),
                cat_state_path=None,
                avatar_state_path=_here("saga_avatar_state.json"),
                avatar_model=os.getenv("SAGA_AVATAR_MODEL")
                or _here("assets", "avatar", "ariu", "saga.model3.json"),
                voice_id=os.getenv("SAGA_ELEVENLABS_VOICE_ID") or None,
                tts_model=os.getenv("SAGA_ELEVENLABS_MODEL") or os.getenv("ELEVENLABS_MODEL")
                or _TTS_MODEL_DEFAULT,
                llm_model=os.getenv("SAGA_LLM_MODEL") or os.getenv("AIVA_LLM_MODEL") or _LLM_MODEL_DEFAULT,
                space_dir=os.getenv("SAGA_SPACE_DIR") or _here("space_saga"),
                library_dir=library,
                capability_exclude=("tend_cat",),
                mood_cfg=_mood_cfg("SAGA", _NEUTRAL_MOOD_BASE),
                speaker_seed_primary=_env_bool("SAGA_SPEAKER_SEED_PRIMARY", False),
                speaker_primary_name=os.getenv("SAGA_SPEAKER_PRIMARY_NAME") or "Frank",
                wake_model=os.getenv("SAGA_WAKE_MODEL") or _here("models", "hey_saga.onnx"),
                stt_keyterms=("Saga",) + SHARED_STT_KEYTERMS + _csv("SAGA_STT_KEYTERMS"),
            )
        raise ValueError(f"unknown character {slug!r}; known: {', '.join(KNOWN)}")
