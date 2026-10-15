"""The settings registry (offline, tmp .env files only — the real .env is never written):

- coverage: every env key the code reads (grepped from aiva/*.py and aiva/modules/*.py, incl. the
  dynamic `{PREFIX}_…` spellings) is registered, and every registered env key is actually read;
- the registered defaults are the ones the code uses (from_env with a clean environment);
- validate: types, choices, ranges, reset, secrets/readonly refused;
- write_env: in-place edit that keeps comments, blank lines, order, unknown lines, inline comments,
  per-line endings and a BOM; removal; one dashboard marker; .env.bak; atomic round-trip;
- secrets are masked everywhere and can't be written.
"""
import glob
import os
import re

import pytest

from modules import settings_registry as reg

AIVA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aiva")
PREFIXES = ("AIVA_", "SAGA_", "NTFY_", "ELEVENLABS_", "DEEPGRAM_", "OPENAI_", "TAVILY_", "OPENWEATHER_",
            "SEARCH_", "DEFAULT_CITY")


def _sources():
    files = glob.glob(os.path.join(AIVA_DIR, "*.py")) + glob.glob(os.path.join(AIVA_DIR, "modules", "*.py"))
    return {f: open(f, encoding="utf-8").read() for f in files
            if os.path.basename(f) != "settings_registry.py"}


def _keys_in_source():
    found = set()
    pref = "|".join(re.escape(p) for p in PREFIXES)
    for path, src in _sources().items():
        # every quoted env-looking key (os.getenv("X"), _env_bool("X"), device_index("X", …), g("X"), …)
        found.update(re.findall(rf'"((?:{pref})[A-Z0-9_]*)"', src))
        # f"{ch.slug.upper()}_AVATAR_X" / f"{prefix}_MOOD_…" → Saga's spelling (Aiva's is literal)
        for suffix in re.findall(r'f"\{[^}"]*(?:upper\(\)|prefix|p)\}_([A-Z0-9_]+)"', src):
            found.add(f"SAGA_{suffix}")
        if path.endswith("speakers.py"):   # shared("T_HIGH", …) → AIVA_SPEAKER_T_HIGH
            found.update(f"AIVA_SPEAKER_{k}" for k in re.findall(r'shared\("([A-Z0-9_]+)"', src))
    from modules.character import _MOOD_DEFAULTS
    for k in _MOOD_DEFAULTS:                          # _mood_cfg("AIVA"/"SAGA") → {P}_MOOD_{K}
        found.add(f"AIVA_MOOD_{k.upper()}")
        found.add(f"SAGA_MOOD_{k.upper()}")
    from modules.tasks import Limits
    found.update(Limits.ENV_KEYS)
    # C-constant variable names that happen to look like keys (AIVA_NAME_VARIANTS = (...) is also a key)
    found.discard("AIVA_")
    return found


def test_every_env_key_in_the_code_is_registered():
    missing = sorted(_keys_in_source() - set(reg.BY_KEY))
    assert not missing, f"env keys read by the code but missing from settings_registry: {missing}"


def test_every_registered_env_key_is_read_somewhere():
    stale = sorted(reg.env_keys() - _keys_in_source())
    assert not stale, f"registered but never read by the code: {stale}"


def test_registry_shape():
    assert len(reg.SETTINGS) > 100 and len(reg.BY_KEY) == len(reg.SETTINGS)
    for s in reg.SETTINGS:
        assert s.group in reg.GROUPS and s.type in reg.TYPES, s.key
        assert s.description.endswith(".") and "\n" not in s.description, s.key
        assert s.applies_to in ("aiva", "saga", "sisters", "house", "all", "dashboard"), s.key
        assert s.restart_targets and set(s.restart_targets) <= set(reg.PROCESS_LABELS), s.key
        if s.type == "choice":
            assert s.choices and s.default in s.choices, s.key
        if s.default_from:
            assert s.default_from in reg.BY_KEY, s.key
        if s.secret:
            assert not s.editable
    assert {s.key for s in reg.SETTINGS if s.secret} >= {"OPENAI_API_KEY", "ELEVENLABS_API_KEY", "DEEPGRAM_API_KEY",
                                                         "TAVILY_API_KEY", "OPENWEATHER_API_KEY"}
    assert [g for g in reg.GROUPS if any(s.group == g for s in reg.SETTINGS)] == list(reg.GROUPS)


@pytest.fixture
def clean_env(monkeypatch):
    for k in reg.env_keys():
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _d(key):
    return reg.BY_KEY[key].default


def test_registered_defaults_are_the_codes_defaults(clean_env):
    from modules.tasks import Limits
    lim = Limits.from_env()
    assert lim == Limits()
    for key, (field, _cast, scale) in Limits.ENV_KEYS.items():
        assert getattr(lim, field) == pytest.approx(_d(key) * scale), key

    from modules.sisters import ExchangeConfig
    ex = ExchangeConfig.from_env()
    assert (ex.max_rounds, ex.backoff_s, ex.max_per_hour, ex.chats_per_day, ex.sticky_s, ex.speech_grace_s,
            ex.announce_quiet_s) == (_d("AIVA_SISTER_MAX_ROUNDS"), _d("AIVA_SISTER_BACKOFF_S"),
                                     _d("AIVA_SISTER_MAX_PER_HOUR"), _d("AIVA_SISTER_CHATS_PER_DAY"),
                                     _d("AIVA_ADDRESS_STICKY_S"), _d("AIVA_SPEECH_GRACE_S"), _d("AIVA_ANNOUNCE_QUIET_S"))

    from modules.speakers import SpeakerConfig
    sp = SpeakerConfig.from_env()
    for field in ("t_high", "t_low", "margin", "sticky_keep", "sticky_secs", "min_secs", "max_secs", "embed_timeout_ms"):
        assert getattr(sp, field) == pytest.approx(_d(f"AIVA_SPEAKER_{field.upper()}")), field
    assert sp.primary_name == _d("AIVA_SPEAKER_PRIMARY_NAME") and sp.seed_primary is _d("AIVA_SPEAKER_SEED_PRIMARY")
    saga_sp = SpeakerConfig.from_env(prefix="SAGA")
    assert saga_sp.primary_name == _d("SAGA_SPEAKER_PRIMARY_NAME")

    from modules.inner import InnerConfig
    ic = InnerConfig.from_env()
    for field in ("base_valence", "base_arousal", "base_social", "halflife_h", "salience_floor", "gain", "material_eps"):
        assert getattr(ic, field) == pytest.approx(_d(f"AIVA_MOOD_{field.upper()}")), field

    from modules.character import Character
    aiva, saga = Character.from_env("aiva"), Character.from_env("saga")
    for field, v in saga.mood_cfg.items():
        assert v == pytest.approx(_d(f"SAGA_MOOD_{field.upper()}")), field
    assert aiva.wake_model == _d("AIVA_WAKE_MODEL") and aiva.tts_model == _d("ELEVENLABS_MODEL")
    assert saga.speaker_seed_primary is _d("SAGA_SPEAKER_SEED_PRIMARY")
    assert aiva.speaker_seed_primary is _d("AIVA_SPEAKER_SEED_PRIMARY")

    from modules.paralinguistics import AffectConfig
    af = AffectConfig.from_env()
    for field in ("ema", "gain", "salience_base", "salience_max", "notable"):
        assert getattr(af, field) == pytest.approx(_d(f"AIVA_AFFECT_{field.upper()}")), field

    from modules.reading import StudyLimits
    st = StudyLimits.from_env()
    assert (st.session_target_words, st.max_sections_per_session, st.sections_per_session) == (
        _d("AIVA_STUDY_WORDS"), _d("AIVA_STUDY_MAX_SECTIONS"), _d("AIVA_STUDY_SECTIONS"))
    assert st.book_cost_usd == pytest.approx(_d("AIVA_BOOK_MAX_COST_USD"))

    from modules.web import WebConfig
    w = WebConfig.from_env()
    assert (w.provider, w.enabled, w.max_results) == (_d("AIVA_WEB_PROVIDER"), _d("AIVA_WEB_SEARCH"), _d("AIVA_WEB_MAX_RESULTS"))
    assert w.timeout == pytest.approx(_d("AIVA_WEB_TIMEOUT"))

    from modules.notify import NotifyConfig
    n = NotifyConfig.from_env()
    assert n.enabled is _d("AIVA_NTFY") and n.base_url == _d("NTFY_URL")

    from modules.limits import caps_from_env
    caps = caps_from_env()
    assert caps["tavily"] == _d("AIVA_TAVILY_MONTHLY_CAP") and caps["tts"] == _d("AIVA_TTS_MONTHLY_CHARS")
    assert caps["stt"] == _d("AIVA_STT_MONTHLY_MINUTES")


def test_readonly_constants_match_their_modules():
    import importlib
    ro = [s for s in reg.SETTINGS if s.source]
    assert ro and all(s.readonly and not s.editable for s in ro)
    for s in ro:
        mod, name, *sub = s.key.split(".")
        value = getattr(importlib.import_module(f"modules.{mod}"), name)
        for k in sub:
            value = value[k]
        assert value == s.default, s.key


def test_limits_from_env_scales_and_tolerates_typos():
    from modules.tasks import Limits
    lim = Limits.from_env({"AIVA_TASK_MAX_WALL_MIN": "5", "AIVA_TASK_MAX_STEPS": "oops",
                           "AIVA_TASK_MAX_COST_USD": "2.5", "AIVA_TASK_CLAUDE_WAIT_MIN": " ",
                           "AIVA_TASK_MAX_REPAIRS_PER_TASK": "3"})
    assert lim.max_wall_secs == 300 and lim.max_cost_usd == 2.5 and lim.max_repairs_per_task == 3
    assert lim.max_steps == Limits().max_steps and lim.claude_wait_secs == Limits().claude_wait_secs
    assert isinstance(lim.max_repairs_per_task, int)


# --- validate -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("key,raw,expected", [
    ("AIVA_WAKE_WORD", True, True), ("AIVA_WAKE_WORD", "off", False), ("AIVA_WAKE_WORD", "1", True),
    ("AIVA_WAKE_WORD", 0, False),
    ("AIVA_SISTER_MAX_ROUNDS", "5", 5), ("AIVA_SISTER_MAX_ROUNDS", 5.0, 5), ("AIVA_SISTER_MAX_ROUNDS", " 7 ", 7),
    ("AIVA_WAKE_THRESHOLD", "0.45", 0.45), ("AIVA_WAKE_THRESHOLD", 1, 1.0),
    ("AIVA_SISTER_TALK", "Audible", "audible"),
    ("AIVA_LLM_MODEL", "  gpt-5.6-luna ", "gpt-5.6-luna"),
    ("AIVA_LLM_MODEL", "", None), ("AIVA_SISTER_MAX_ROUNDS", "", None), ("AIVA_WAKE_WORD", None, None),
])
def test_validate_coerces(key, raw, expected):
    ok, val = reg.validate(key, raw)
    assert ok, val
    assert val == expected and type(val) is type(expected)


@pytest.mark.parametrize("key,raw,needle", [
    ("NOPE_KEY", "1", "unknown"),
    ("OPENAI_API_KEY", "sk-x", "secret"),
    ("becoming.MAX_ACTIVE_TRAITS", 3, "read-only"),
    ("AIVA_CHARACTER", "saga", "read-only"),
    ("AIVA_WAKE_WORD", "maybe", "on/off"),
    ("AIVA_SISTER_MAX_ROUNDS", "2.5", "whole number"),
    ("AIVA_SISTER_MAX_ROUNDS", "abc", "whole number"),
    ("AIVA_SISTER_MAX_ROUNDS", True, "not on/off"),
    ("AIVA_WAKE_THRESHOLD", "1.5", "at most"),
    ("AIVA_WAKE_THRESHOLD", "-1", "at least"),
    ("AIVA_WAKE_THRESHOLD", "nan", "number"),
    ("AIVA_SISTER_TALK", "loud", "one of"),
    ("AIVA_LLM_MODEL", "gpt\nEVIL=1", "one line"),
    ("AIVA_LLM_MODEL", "x" * 1000, "too long"),
    ("AIVA_LLM_MODEL", ["a"], "single"),
])
def test_validate_rejects(key, raw, needle):
    ok, err = reg.validate(key, raw)
    assert not ok and needle in err


# --- write_env ----------------------------------------------------------------------------------------

ORIGINAL = (
    "# my keys\n"
    "OPENAI_API_KEY=sk-secret-abcd1234\n"
    "\n"
    "# the voice model — keep it fast\n"
    "AIVA_LLM_MODEL=gpt-5.6-luna\n"
    "SOMETHING_ELSE=keep me   # not ours\n"
    "AIVA_NTFY_ON_START=1          # ping on boot\n"
    "# AIVA_TTS_MONTHLY_CHARS=0    (0 = off)\n"
    "AIVA_OUTPUT_DEVICE_NAME=Speakers (Realtek\n"
    'NTFY_URL="https://ntfy.example"\n'
    "export AIVA_SELF_TICK_S=60\n"
    "AIVA_STUDY_SECTIONS=4\n"
    "last line without newline=ok"
)


def test_parse_env_text_follows_dotenv_rules():
    vals = reg.parse_env_text(ORIGINAL)
    assert vals["AIVA_NTFY_ON_START"] == "1" and vals["SOMETHING_ELSE"] == "keep me"
    assert vals["AIVA_OUTPUT_DEVICE_NAME"] == "Speakers (Realtek" and vals["NTFY_URL"] == "https://ntfy.example"
    assert vals["AIVA_SELF_TICK_S"] == "60" and "AIVA_TTS_MONTHLY_CHARS" not in vals
    assert reg.parse_env_text("A=1\nA=2\nB=x#y\nC=\"q # r\" # c")== {"A": "2", "B": "x#y", "C": "q # r"}


def test_write_env_edits_in_place_and_keeps_everything_else(tmp_path):
    p = tmp_path / ".env"
    p.write_bytes(ORIGINAL.encode("utf-8"))
    res = reg.write_env(str(p), {"AIVA_LLM_MODEL": "gpt-4.1", "AIVA_NTFY_ON_START": "0",
                                 "NTFY_URL": "https://x.example", "AIVA_SELF_TICK_S": "30",
                                 "AIVA_STUDY_SECTIONS": None, "AIVA_TASK_MAX_COST_USD": "2.5",
                                 "AIVA_TTS_MONTHLY_CHARS": "10000"})
    assert res == {"changed": ["AIVA_LLM_MODEL", "AIVA_NTFY_ON_START", "NTFY_URL", "AIVA_SELF_TICK_S"],
                   "added": ["AIVA_TASK_MAX_COST_USD", "AIVA_TTS_MONTHLY_CHARS"], "removed": ["AIVA_STUDY_SECTIONS"]}
    out = p.read_text(encoding="utf-8")
    assert out == (
        "# my keys\n"
        "OPENAI_API_KEY=sk-secret-abcd1234\n"
        "\n"
        "# the voice model — keep it fast\n"
        "AIVA_LLM_MODEL=gpt-4.1\n"
        "SOMETHING_ELSE=keep me   # not ours\n"
        "AIVA_NTFY_ON_START=0          # ping on boot\n"
        "# AIVA_TTS_MONTHLY_CHARS=0    (0 = off)\n"
        "AIVA_OUTPUT_DEVICE_NAME=Speakers (Realtek\n"
        'NTFY_URL="https://x.example"\n'
        "export AIVA_SELF_TICK_S=30\n"
        "last line without newline=ok\n"
        "\n"
        f"{reg.DASHBOARD_MARKER}\n"
        "AIVA_TASK_MAX_COST_USD=2.5\n"
        "AIVA_TTS_MONTHLY_CHARS=10000\n")
    assert (tmp_path / ".env.bak").read_bytes() == ORIGINAL.encode("utf-8")
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    # a second write reuses the marker instead of adding another one
    reg.write_env(str(p), {"AIVA_WEB_MAX_RESULTS": "4", "AIVA_TASK_MAX_COST_USD": "3"})
    out2 = p.read_text(encoding="utf-8")
    assert out2.count(reg.DASHBOARD_MARKER) == 1
    assert out2.endswith("AIVA_TASK_MAX_COST_USD=3\nAIVA_TTS_MONTHLY_CHARS=10000\nAIVA_WEB_MAX_RESULTS=4\n")
    vals = reg.read_env(str(p))
    assert vals["AIVA_LLM_MODEL"] == "gpt-4.1" and vals["AIVA_WEB_MAX_RESULTS"] == "4"
    assert "AIVA_STUDY_SECTIONS" not in vals and vals["SOMETHING_ELSE"] == "keep me"


def test_write_env_keeps_mixed_line_endings_and_bom(tmp_path):
    p = tmp_path / ".env"
    raw = "﻿A_KEEP=1\r\nAIVA_LLM_MODEL=x\r\n# note\nAIVA_SELF_TICK_S=60\n".encode("utf-8")
    p.write_bytes(raw)
    reg.write_env(str(p), {"AIVA_LLM_MODEL": "y", "AIVA_SELF_TICK_S": "30", "AIVA_WEB_TIMEOUT": "9"})
    # each old line keeps its own ending; new lines use the file's first ending (CRLF here)
    assert p.read_bytes() == ("﻿A_KEEP=1\r\nAIVA_LLM_MODEL=y\r\n# note\nAIVA_SELF_TICK_S=30\n"
                              f"\r\n{reg.DASHBOARD_MARKER}\r\nAIVA_WEB_TIMEOUT=9\r\n").encode("utf-8")
    assert reg.read_env(str(p)) == {"A_KEEP": "1", "AIVA_LLM_MODEL": "y", "AIVA_SELF_TICK_S": "30", "AIVA_WEB_TIMEOUT": "9"}


def test_write_env_quotes_when_needed_and_creates_a_missing_file(tmp_path):
    p = tmp_path / "sub" / ".env"
    reg.write_env(str(p), {"AIVA_HOTKEY_SISTER_TALK": "ctrl+alt+p", "AIVA_STT_KEYTERMS": "Frank, #hash tag",
                           "DEFAULT_CITY": ' "Brad" '})
    vals = reg.read_env(str(p))
    assert vals == {"AIVA_HOTKEY_SISTER_TALK": "ctrl+alt+p", "AIVA_STT_KEYTERMS": "Frank, #hash tag",
                    "DEFAULT_CITY": ' "Brad" '}
    assert not (tmp_path / "sub" / ".env.bak").exists()      # nothing to back up the first time


def test_round_trip_through_dotenv(tmp_path):
    dotenv = pytest.importorskip("dotenv")
    p = tmp_path / ".env"
    p.write_text(ORIGINAL, encoding="utf-8")
    reg.write_env(str(p), {"AIVA_STT_KEYTERMS": "a #b", "AIVA_LLM_MODEL": "gpt-4.1", "DEFAULT_CITY": "Brad, RO"})
    parsed = dotenv.dotenv_values(str(p))
    for k, v in reg.read_env(str(p)).items():
        assert parsed.get(k) == v, k


# --- current values, masking, apply -----------------------------------------------------------------

def test_current_values_sources_defaults_and_masking(tmp_path):
    p = tmp_path / ".env"
    p.write_text("OPENAI_API_KEY=sk-secret-abcd1234\nDEEPGRAM_API_KEY=\nAIVA_LLM_MODEL=gpt-5.6-luna\n"
                 "AIVA_WAKE_THRESHOLD=0.5\nAIVA_SISTER_MAX_ROUNDS=lots\n", encoding="utf-8")
    v = reg.current_values(str(p))
    assert v["OPENAI_API_KEY"]["value"] == "••••1234" and v["OPENAI_API_KEY"]["is_set"] is True
    assert "sk-secret" not in repr(v)
    assert v["DEEPGRAM_API_KEY"]["value"] is None and v["DEEPGRAM_API_KEY"]["is_set"] is False
    assert v["AIVA_LLM_MODEL"] == {**v["AIVA_LLM_MODEL"], "value": "gpt-5.6-luna", "source": ".env", "modified": True}
    assert v["AIVA_TASK_MODEL"]["source"] == "default" and v["AIVA_TASK_MODEL"]["value"] == "gpt-5.6-luna"
    assert v["AIVA_WEB_MODEL"]["value"] == "gpt-5.6-luna"             # default_from chains through the task model
    assert v["AIVA_WEB_MODEL"]["default_text"] == "same as AIVA_TASK_MODEL"
    assert v["AIVA_WAKE_THRESHOLD"]["modified"] is False               # set, but to the default
    assert v["AIVA_SISTER_MAX_ROUNDS"]["invalid"] is True and v["AIVA_SISTER_MAX_ROUNDS"]["value"] == "lots"
    assert v["AIVA_SELF_TICK_S"] == {**v["AIVA_SELF_TICK_S"], "value": 60, "source": "default"}
    assert v["becoming.MAX_ACTIVE_TRAITS"]["source"] == "code"
    assert reg.mask("abc") == "••••" and reg.mask("") == "" and reg.mask(None) == ""


def test_missing_env_file_is_all_defaults(tmp_path):
    v = reg.current_values(str(tmp_path / "none.env"))
    assert all(r["source"] in ("default", "code") for r in v.values())


def test_apply_updates_validates_everything_before_writing(tmp_path):
    p = tmp_path / ".env"
    p.write_text("OPENAI_API_KEY=sk-secret-abcd1234\nAIVA_WAKE_THRESHOLD=0.4\n", encoding="utf-8")
    before = p.read_bytes()
    ok, payload = reg.apply_updates(str(p), {"AIVA_WAKE_THRESHOLD": "0.45", "OPENAI_API_KEY": "sk-new",
                                             "AIVA_SISTER_TALK": "loud"})
    assert not ok and set(payload["errors"]) == {"OPENAI_API_KEY", "AIVA_SISTER_TALK"}
    assert p.read_bytes() == before and not (tmp_path / ".env.bak").exists()
    assert reg.apply_updates(str(p), {})[0] is False
    assert reg.apply_updates(str(p), ["x"])[0] is False

    ok, payload = reg.apply_updates(str(p), {"AIVA_WAKE_THRESHOLD": 0.45, "AIVA_SELF_TICK_S": "30",
                                             "AIVA_AVATAR_FPS": 30, "AIVA_WAKE_WORD": True})
    assert ok
    assert payload["restart"] == ["house", "aiva", "saga", "body"]
    assert payload["message"].startswith("saved — restart the house, Aiva, Saga and ")
    vals = reg.read_env(str(p))
    assert vals["AIVA_WAKE_THRESHOLD"] == "0.45" and vals["AIVA_WAKE_WORD"] == "1" and vals["AIVA_AVATAR_FPS"] == "30"
    assert vals["OPENAI_API_KEY"] == "sk-secret-abcd1234"                 # the secret was never touched
    ok, payload = reg.apply_updates(str(p), {"AIVA_WAKE_THRESHOLD": None})
    assert ok and "AIVA_WAKE_THRESHOLD" not in reg.read_env(str(p)) and payload["restart"] == ["house"]


def test_restart_text_and_grouped_payload(tmp_path):
    assert reg.restart_text([]) == "saved"
    assert reg.restart_text(["aiva"]) == "saved — restart Aiva to apply"
    groups = reg.grouped(str(tmp_path / ".env"))
    assert [g["group"] for g in groups] == list(reg.GROUPS)
    row = next(s for g in groups for s in g["settings"] if s["key"] == "AIVA_SISTER_TALK")
    assert row["choices"] == ["private", "audible"] and row["editable"] and row["restart"] == ["house", "aiva", "saga"]
    qh = next(s for g in groups for s in g["settings"] if s["key"] == "becoming.CURIOSITY.max_asks_per_day")
    assert qh["readonly"] and qh["source_file"].endswith("becoming.py CURIOSITY")
