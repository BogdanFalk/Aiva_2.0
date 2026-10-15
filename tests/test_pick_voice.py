"""tools/pick_voice.py: the pure parts — .env upsert, label similarity and
ranking, filters/search, CLI parsing, WAV wrapping. No network."""
import importlib.util
import io
import os
import wave

import pytest

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
_spec = importlib.util.spec_from_file_location("pick_voice", os.path.join(_TOOLS, "pick_voice.py"))
pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv)

KEY = "SAGA_ELEVENLABS_VOICE_ID"


# --- upsert_env -------------------------------------------------------------------

def test_upsert_replaces_existing_line_in_place():
    text = "A=1\nSAGA_ELEVENLABS_VOICE_ID=old\nB=2\n"
    assert pv.upsert_env(text, KEY, "new") == "A=1\nSAGA_ELEVENLABS_VOICE_ID=new\nB=2\n"


def test_upsert_uncomments_a_commented_line_in_place():
    text = "A=1\n# SAGA_ELEVENLABS_VOICE_ID=old\nB=2\n"
    assert pv.upsert_env(text, KEY, "v") == "A=1\nSAGA_ELEVENLABS_VOICE_ID=v\nB=2\n"
    text2 = "#SAGA_ELEVENLABS_VOICE_ID=\n"
    assert pv.upsert_env(text2, KEY, "v") == "SAGA_ELEVENLABS_VOICE_ID=v\n"


def test_upsert_active_line_wins_over_commented_one():
    text = "# SAGA_ELEVENLABS_VOICE_ID=old\nSAGA_ELEVENLABS_VOICE_ID=cur\n"
    assert pv.upsert_env(text, KEY, "v") == "# SAGA_ELEVENLABS_VOICE_ID=old\nSAGA_ELEVENLABS_VOICE_ID=v\n"


def test_upsert_inserts_right_after_aivas_voice_line():
    text = "ELEVENLABS_API_KEY=k\nELEVENLABS_VOICE_ID=EZ\nELEVENLABS_MODEL=m\n"
    out = pv.upsert_env(text, KEY, "v")
    assert out == "ELEVENLABS_API_KEY=k\nELEVENLABS_VOICE_ID=EZ\nSAGA_ELEVENLABS_VOICE_ID=v\nELEVENLABS_MODEL=m\n"


def test_upsert_anchor_as_newline_less_last_line():
    out = pv.upsert_env("X=1\nELEVENLABS_VOICE_ID=EZ", KEY, "v")
    assert out == "X=1\nELEVENLABS_VOICE_ID=EZ\nSAGA_ELEVENLABS_VOICE_ID=v\n"


def test_upsert_appends_when_nothing_matches():
    assert pv.upsert_env("X=1\n", KEY, "v") == "X=1\nSAGA_ELEVENLABS_VOICE_ID=v\n"
    assert pv.upsert_env("X=1", KEY, "v") == "X=1\nSAGA_ELEVENLABS_VOICE_ID=v\n"
    assert pv.upsert_env("", KEY, "v") == "SAGA_ELEVENLABS_VOICE_ID=v\n"


def test_upsert_is_idempotent():
    text = "A=1\nELEVENLABS_VOICE_ID=EZ\nB=2\n"
    once = pv.upsert_env(text, KEY, "v")
    assert pv.upsert_env(once, KEY, "v") == once
    assert pv.upsert_env(once, KEY, "w") == once.replace("=v\n", "=w\n")
    assert once.count(KEY) == 1


def test_upsert_preserves_other_bytes_and_trailing_newline_state():
    # comments, blank lines, odd spacing, and NO trailing newline: all kept
    text = "# keys\n\n  A = 1  \nSAGA_ELEVENLABS_VOICE_ID = old  # note\nB=2"
    out = pv.upsert_env(text, KEY, "new")
    assert out == "# keys\n\n  A = 1  \nSAGA_ELEVENLABS_VOICE_ID=new\nB=2"
    # replaced key on the last line without newline stays newline-less
    assert pv.upsert_env("A=1\nSAGA_ELEVENLABS_VOICE_ID=old", KEY, "n") == "A=1\nSAGA_ELEVENLABS_VOICE_ID=n"
    # trailing newline kept when present
    assert pv.upsert_env("A=1\nSAGA_ELEVENLABS_VOICE_ID=old\n", KEY, "n").endswith("=n\n")


def test_upsert_keeps_crlf_line_endings():
    text = "A=1\r\nELEVENLABS_VOICE_ID=EZ\r\nB=2\r\n"
    assert pv.upsert_env(text, KEY, "v") == "A=1\r\nELEVENLABS_VOICE_ID=EZ\r\nSAGA_ELEVENLABS_VOICE_ID=v\r\nB=2\r\n"
    assert pv.upsert_env("A=1\r\n", KEY, "v") == "A=1\r\nSAGA_ELEVENLABS_VOICE_ID=v\r\n"


def test_upsert_does_not_touch_longer_keys_with_the_same_prefix():
    text = "SAGA_ELEVENLABS_VOICE_ID_BACKUP=zzz\nELEVENLABS_VOICE_ID=EZ\n"
    out = pv.upsert_env(text, KEY, "v")
    assert out == "SAGA_ELEVENLABS_VOICE_ID_BACKUP=zzz\nELEVENLABS_VOICE_ID=EZ\nSAGA_ELEVENLABS_VOICE_ID=v\n"


# --- similarity / ranking / filters ---------------------------------------------------

def _voice(vid, name, category="premade", **labels):
    return {"voice_id": vid, "name": name, "category": category, "labels": labels,
            "preview_url": f"https://example.invalid/{vid}.mp3"}


AIVA = _voice("EZ", "Aiva", "cloned", gender="female", age="young", accent="american",
              description="warm", use_case="conversational")


def test_labels_of_normalises_case_and_drops_blanks():
    v = {"labels": {"Gender": " Female ", "age": "", "accent": None, "Use_Case": "Narration"}}
    assert pv.labels_of(v) == {"gender": "female", "use_case": "narration"}
    assert pv.labels_of({}) == {}


def test_similarity_weighted_and_bounded():
    a = pv.labels_of(AIVA)
    assert pv.similarity(a, a) == 1.0
    assert pv.similarity({}, {}) == 0.0
    assert pv.similarity(a, {"gender": "male"}) == 0.0
    # gender agrees (3), age disagrees (2): 3 / 5
    assert pv.similarity({"gender": "female", "age": "young"},
                         {"gender": "female", "age": "old"}) == pytest.approx(0.6)
    # a key only one side has still counts against the score
    assert pv.similarity({"gender": "female"}, {"gender": "female", "accent": "irish"}) == pytest.approx(3 / 5)


def test_rank_like_orders_best_first_with_stable_ties():
    voices = [
        _voice("m1", "Zed", gender="male", age="young", accent="american"),
        _voice("f2", "Bea", gender="female", age="young", accent="american", description="warm"),
        _voice("f1", "Ann", gender="female", age="young", accent="american", description="warm"),
        _voice("f3", "Cal", gender="female", age="young", accent="british", description="warm"),
    ]
    ranked = pv.rank_like(voices, pv.labels_of(AIVA))
    # Ann/Bea match everything but use_case (8/9, tie broken by name), Cal loses
    # accent too (6/9), Zed keeps only age+accent (4/9)
    assert [v["name"] for _, v in ranked] == ["Ann", "Bea", "Cal", "Zed"]
    assert ranked[0][0] == ranked[1][0] == pytest.approx(8 / 9, abs=1e-3)
    assert ranked[2][0] == pytest.approx(6 / 9, abs=1e-3)
    assert ranked[3][0] == pytest.approx(4 / 9, abs=1e-3)


def test_apply_filters_substring_on_labels_and_top_level_fields():
    voices = [
        _voice("a", "A", "cloned", gender="female", accent="british"),
        _voice("b", "B", "premade", gender="female", accent="american"),
        _voice("c", "C", "premade", gender="male", accent="british"),
    ]
    assert [v["voice_id"] for v in pv.apply_filters(voices, {"gender": "female"})] == ["a", "b"]
    assert [v["voice_id"] for v in pv.apply_filters(voices, {"gender": "female", "accent": "brit"})] == ["a"]
    assert [v["voice_id"] for v in pv.apply_filters(voices, {"category": "premade"})] == ["b", "c"]
    assert pv.apply_filters(voices, {"accent": "irish"}) == []
    assert pv.apply_filters(voices, {}) == voices


def test_search_matches_name_description_labels():
    voices = [_voice("a", "Saga", description="calm"), _voice("b", "Other", accent="nordic")]
    voices[0]["description"] = "a calm, bright voice"
    assert [v["voice_id"] for v in pv.search(voices, "saga")] == ["a"]
    assert [v["voice_id"] for v in pv.search(voices, "bright")] == ["a"]
    assert [v["voice_id"] for v in pv.search(voices, "NORDIC")] == ["b"]
    assert pv.search(voices, "") == voices


def test_parse_filters_and_cli_repeatable_flag():
    args = pv.build_parser().parse_args(["--filter", "gender=female", "--filter", "Accent=British"])
    assert args.filter == ["gender=female", "Accent=British"]
    assert pv.parse_filters(args.filter) == {"gender": "female", "accent": "british"}
    assert pv.parse_filters([]) == {}
    with pytest.raises(SystemExit):
        pv.parse_filters(["gender"])
    with pytest.raises(SystemExit):
        pv.parse_filters(["=female"])
    args2 = pv.build_parser().parse_args(["--set", "abc", "--env", "x.env"])
    assert args2.set == "abc" and args2.env == "x.env" and args2.key == KEY
    assert pv.build_parser().parse_args([]).env.endswith(".env")


def test_fmt_row_survives_missing_labels():
    row = pv.fmt_row({"voice_id": "v1", "name": "Nameless"}, idx=2, score=0.5)
    assert "v1" in row and "Nameless" in row and row.startswith("  2.") and "0.50" in row
    assert "[-]" in row
    full = pv.fmt_row(AIVA)
    assert "female" in full and "american" in full and "[cloned]" in full


def test_api_key_only_travels_to_elevenlabs_hosts():
    assert pv.is_api_url("https://api.us.elevenlabs.io/v1/voices/x/previews/audio?payload=abc")
    assert pv.is_api_url("https://api.elevenlabs.io/v1/voices/x")
    assert not pv.is_api_url("https://storage.googleapis.com/eleven-public-prod/premade/x.mp3")
    assert not pv.is_api_url("https://elevenlabs.io.evil.example/v1/voices")
    assert not pv.is_api_url("http://api.elevenlabs.io/v1/voices")  # never in the clear
    assert not pv.is_api_url("")
    assert not pv.is_api_url(None)


def test_download_preview_uses_session_only_for_api_urls(tmp_path, monkeypatch):
    calls = []

    class Resp:
        content = b"mp3"

        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, url, timeout):
            calls.append(("session", url))
            return Resp()

    def fake_requests_get(url, timeout):
        calls.append(("bare", url))
        return Resp()

    monkeypatch.setattr(pv.requests, "get", fake_requests_get)
    monkeypatch.setattr(pv, "PREVIEW_DIR", str(tmp_path))
    api_voice = {"voice_id": "gen1", "preview_url": "https://api.us.elevenlabs.io/v1/voices/gen1/previews/audio?payload=x"}
    pub_voice = {"voice_id": "pre1", "preview_url": "https://storage.googleapis.com/eleven-public-prod/pre1.mp3"}
    assert pv.download_preview(api_voice, FakeSession()) == str(tmp_path / "gen1.mp3")
    assert pv.download_preview(pub_voice, FakeSession()) == str(tmp_path / "pre1.mp3")
    assert calls == [("session", api_voice["preview_url"]), ("bare", pub_voice["preview_url"])]
    # cached: a second call downloads nothing
    assert pv.download_preview(api_voice, FakeSession()) == str(tmp_path / "gen1.mp3")
    assert len(calls) == 2
    assert pv.download_preview({"voice_id": "none"}, FakeSession()) is None


def test_credits_estimate_and_wav_wrapper():
    assert pv.credits_for("abcd", "eleven_flash_v2_5") == 2.0
    assert pv.credits_for("abcd", "eleven_turbo_v2_5") == 2.0
    assert pv.credits_for("abcd", "eleven_multilingual_v2") == 4.0
    pcm = b"\x00\x00" * 2205  # 0.1 s of silence at 22.05 kHz
    wav = pv.pcm16_to_wav(pcm, 22050)
    with wave.open(io.BytesIO(wav), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 22050, 2205)
    assert pv.SAY_MAX_CHARS == 200
