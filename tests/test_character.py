"""Character: one sister's whole config. `from_env("aiva")` must reproduce today's paths and
defaults exactly (Aiva's files never move); `from_env("saga")` reads SAGA_* with her own defaults."""
import os
import re

import pytest

from modules import persona
from modules.character import (AIVA_LOOSE, AIVA_NAME_VARIANTS, KNOWN, OVERHEARD_RE, SAGA_LOOSE,
                               SAGA_NAME_VARIANTS, Character, display_name)
from modules.inner import InnerConfig

AIVA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aiva")


def here(*parts):
    return os.path.join(AIVA_DIR, *parts)


@pytest.fixture
def clean_env(monkeypatch):
    """No AIVA_/SAGA_/ELEVENLABS_ keys (the .env may have been loaded by another import)."""
    for k in list(os.environ):
        if k.startswith(("AIVA_", "SAGA_", "ELEVENLABS_")):
            monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_aiva_reproduces_todays_paths_and_defaults(clean_env):
    ch = Character.from_env("aiva")
    assert (ch.name, ch.slug, ch.sisters) == ("Aiva", "aiva", ("Saga",))
    assert ch.db_path == here("memory.db") and ch.backup_stem == "memory"
    assert ch.inner_state_path == here("inner_state.json")
    assert ch.cat_state_path == here("cat_state.json")
    assert ch.avatar_state_path == here("avatar_state.json")
    assert ch.avatar_model == here("assets", "avatar", "ariu", "ariu.model3.json")
    assert ch.space_dir == here("space") and ch.library_dir == here("library")
    assert ch.voice_id is None and ch.tts_model == "eleven_flash_v2_5"
    assert ch.wake_model == "hey_jarvis"                       # main.py's default when unset
    assert ch.name_variants == AIVA_NAME_VARIANTS and ch.loose_regex == AIVA_LOOSE
    assert ch.personality is persona.PERSONALITY
    assert ch.baseline_traits == tuple(persona.baseline_traits()) and ch.baseline_tastes is True
    assert ch.self_forming is False and ch.identity_tag == "the same cheeky Aiva"
    assert ch.capability_exclude == ()
    assert InnerConfig(**ch.mood_cfg) == InnerConfig()         # same knobs, same defaults
    assert ch.speaker_seed_primary is True and ch.speaker_primary_name == "Frank"
    assert ch.stt_keyterms[:6] == ("Aiva", "Claude Code", "PowerShell", "Frank", "Bogdan", "Fianu")
    assert ch.prefix == "AIVA" and ch.sister == "Saga"
    assert os.path.isabs(ch.db_path) and os.path.isabs(ch.space_dir)


def test_aiva_honours_the_unprefixed_env_keys(clean_env):
    clean_env.setenv("AIVA_NAME_VARIANTS", "Ayvuh, eyva")
    clean_env.setenv("AIVA_MOOD_BASE_VALENCE", "0.3")
    clean_env.setenv("ELEVENLABS_VOICE_ID", "voice-a")
    clean_env.setenv("ELEVENLABS_MODEL", "eleven_v3")
    clean_env.setenv("AIVA_STT_KEYTERMS", "Foo,Bar")
    clean_env.setenv("AIVA_INNER_STATE", "x/inner.json")
    clean_env.setenv("AIVA_SPEAKER_SEED_PRIMARY", "0")
    ch = Character.from_env("aiva")
    assert ch.name_variants == AIVA_NAME_VARIANTS + ("ayvuh", "eyva")
    assert ch.mood_cfg["base_valence"] == 0.3 and ch.mood_cfg["base_arousal"] == 0.5
    assert ch.voice_id == "voice-a" and ch.tts_model == "eleven_v3"
    assert ch.stt_keyterms[-2:] == ("Foo", "Bar")
    assert ch.inner_state_path == "x/inner.json"
    assert ch.speaker_seed_primary is False


def test_saga_defaults(clean_env):
    ch = Character.from_env("saga")
    assert (ch.name, ch.slug, ch.sisters) == ("Saga", "saga", ("Aiva",))
    assert ch.db_path == here("saga.db") and ch.backup_stem == "saga"
    assert ch.inner_state_path == here("saga_inner_state.json")
    assert ch.cat_state_path is None                            # no cat for Saga
    assert ch.avatar_state_path == here("saga_avatar_state.json")
    assert ch.avatar_model == here("assets", "avatar", "ariu", "saga.model3.json")
    assert ch.space_dir == here("space_saga")
    assert ch.library_dir == Character.from_env("aiva").library_dir     # shared library
    assert ch.voice_id is None and ch.tts_model == "eleven_flash_v2_5"
    assert ch.wake_model == here("models", "hey_saga.onnx")
    assert ch.name_variants == SAGA_NAME_VARIANTS and ch.loose_regex == SAGA_LOOSE
    assert ch.personality is persona.SAGA_FLOOR
    assert ch.baseline_traits == () and ch.baseline_tastes is False
    assert ch.self_forming is True and ch.identity_tag == "still Saga"
    assert ch.capability_exclude == ("tend_cat",)
    assert (ch.mood_cfg["base_valence"], ch.mood_cfg["base_arousal"], ch.mood_cfg["base_social"]) == (0.0, 0.5, 0.0)
    assert ch.mood_cfg["halflife_h"] == 6.0                     # the non-baseline knobs are shared defaults
    assert ch.speaker_seed_primary is False and ch.speaker_primary_name == "Frank"
    assert ch.stt_keyterms[0] == "Saga" and "Aiva" not in ch.stt_keyterms
    assert ch.sister == "Aiva"


def test_saga_reads_saga_keys_and_falls_back_only_for_shared_ones(clean_env):
    clean_env.setenv("SAGA_DB", "D:/sisters/saga.db")
    clean_env.setenv("SAGA_INNER_STATE", "D:/sisters/inner.json")
    clean_env.setenv("SAGA_SPACE_DIR", "D:/sisters/space")
    clean_env.setenv("SAGA_ELEVENLABS_VOICE_ID", "uSbTIocs0ieRu70Acnzm")
    clean_env.setenv("ELEVENLABS_MODEL", "eleven_v3")           # shared fallback for the TTS model
    clean_env.setenv("ELEVENLABS_VOICE_ID", "aivas-voice")      # NOT a fallback for her voice
    clean_env.setenv("SAGA_AVATAR_MODEL", "D:/sisters/saga.model3.json")
    clean_env.setenv("SAGA_NAME_VARIANTS", "Sagga")
    clean_env.setenv("SAGA_MOOD_BASE_VALENCE", "0.2")
    clean_env.setenv("AIVA_MOOD_BASE_SOCIAL", "0.9")            # Aiva's mood knob must not leak
    clean_env.setenv("SAGA_SPEAKER_PRIMARY_NAME", "Bogdan")
    clean_env.setenv("SAGA_SPEAKER_SEED_PRIMARY", "1")
    clean_env.setenv("SAGA_WAKE_MODEL", "D:/sisters/hey_saga.onnx")
    ch = Character.from_env("saga")
    assert ch.db_path == "D:/sisters/saga.db" and ch.inner_state_path == "D:/sisters/inner.json"
    assert ch.space_dir == "D:/sisters/space"
    assert ch.voice_id == "uSbTIocs0ieRu70Acnzm" and ch.tts_model == "eleven_v3"
    assert ch.avatar_model == "D:/sisters/saga.model3.json"
    assert ch.name_variants == SAGA_NAME_VARIANTS + ("sagga",)
    assert ch.mood_cfg["base_valence"] == 0.2 and ch.mood_cfg["base_social"] == 0.0
    assert ch.speaker_primary_name == "Bogdan" and ch.speaker_seed_primary is True
    assert ch.wake_model == "D:/sisters/hey_saga.onnx"
    clean_env.setenv("SAGA_ELEVENLABS_MODEL", "eleven_flash_v2_5")
    assert Character.from_env("saga").tts_model == "eleven_flash_v2_5"   # her own key wins


def test_unknown_slug_and_display_names():
    assert KNOWN == ("aiva", "saga")
    assert display_name("aiva") == "Aiva" and display_name("SAGA") == "Saga" and display_name("x") == "X"
    with pytest.raises(ValueError):
        Character.from_env("nova")


@pytest.mark.parametrize("word,aiva,saga", [
    ("eyva", True, False), ("ayvah", True, False), ("iva", True, False),
    ("avocado", False, False), ("eve", False, False),
    ("saga", False, True), ("sagah", False, True), ("sarga", False, True), ("seega", False, True),
    ("sage", False, False), ("sugar", False, False), ("sarah", False, False), ("stage", False, False),
])
def test_loose_regexes_are_disjoint_and_reject_common_words(word, aiva, saga):
    assert bool(re.fullmatch(AIVA_LOOSE, word)) is aiva
    assert bool(re.fullmatch(SAGA_LOOSE, word)) is saga


def test_overheard_re_matches_only_a_sisters_bracketed_line():
    assert OVERHEARD_RE.match('[Saga said to him: "not that one"]')
    assert OVERHEARD_RE.match('[Aiva said to you: "your turn"]')
    assert OVERHEARD_RE.match('[Saga said to him: "x"]').group(1) == "Saga"
    assert not OVERHEARD_RE.match('[He said to Saga: "hi"]')
    assert not OVERHEARD_RE.match('Saga said to him: "hi"')
    assert not OVERHEARD_RE.match('[Speaking now: Frank.]')
