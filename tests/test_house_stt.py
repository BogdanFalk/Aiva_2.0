"""House STT switch (AIVA_STT=flux|nova): config parsing, the idle-close rule, building the service
and the stage aggregator params offline (nothing connects), Flux's lazy reopen after an idle close,
and the empty-final guard in ActivityWatcher."""
import pytest
from pipecat.frames.frames import TranscriptionFrame
from pipecat.tests.utils import SleepFrame, run_test
from websockets.protocol import State

from modules import settings_registry as reg
from modules.ear import ActivityWatcher, MicController
from modules.house_stt import FLUX_MODEL, STTConfig, build_stt, idle_close_due, stage_params


def test_config_defaults_match_registry():
    c = STTConfig.from_env({})
    assert c.engine == reg.BY_KEY["AIVA_STT"].default == "flux" and c.flux
    assert c.eot_threshold == reg.BY_KEY["AIVA_FLUX_EOT_THRESHOLD"].default
    assert c.eot_timeout_ms == reg.BY_KEY["AIVA_FLUX_EOT_TIMEOUT_MS"].default
    assert c.idle_close_s == reg.BY_KEY["AIVA_FLUX_IDLE_CLOSE_S"].default
    assert c.note == ""


def test_config_parses_and_falls_back():
    c = STTConfig.from_env({"AIVA_STT": " NOVA ", "AIVA_FLUX_EOT_THRESHOLD": "0.8",
                            "AIVA_FLUX_EOT_TIMEOUT_MS": "3000", "AIVA_FLUX_IDLE_CLOSE_S": "0"})
    assert (c.engine, c.flux, c.eot_threshold, c.eot_timeout_ms, c.idle_close_s) == ("nova", False, 0.8, 3000, 0.0)
    bad = STTConfig.from_env({"AIVA_STT": "whisper", "AIVA_FLUX_EOT_THRESHOLD": "0.99",
                              "AIVA_FLUX_EOT_TIMEOUT_MS": "oops", "AIVA_FLUX_IDLE_CLOSE_S": "-1"})
    assert bad.engine == "flux" and "whisper" in bad.note
    assert (bad.eot_threshold, bad.eot_timeout_ms, bad.idle_close_s) == (0.7, 5000, 3.0)
    assert "Flux" in c.__class__.from_env({}).describe() and "nova" in c.describe()


def test_idle_close_rule():
    assert not idle_close_due(active=False, user_speaking=False, idle_s=99, close_after_s=3)
    assert not idle_close_due(active=True, user_speaking=False, idle_s=99, close_after_s=0)
    assert not idle_close_due(active=True, user_speaking=False, idle_s=2.9, close_after_s=3)
    assert idle_close_due(active=True, user_speaking=False, idle_s=3.0, close_after_s=3)
    # mid-turn: wait much longer (Flux ends a dangling turn itself on the watchdog's silence)
    assert not idle_close_due(active=True, user_speaking=True, idle_s=10, close_after_s=3)
    assert idle_close_due(active=True, user_speaking=True, idle_s=15, close_after_s=3)


def test_build_flux_offline():
    cfg = STTConfig.from_env({"AIVA_FLUX_EOT_THRESHOLD": "0.75", "AIVA_FLUX_EOT_TIMEOUT_MS": "4000"})
    stt = build_stt(cfg, api_key="dummy", keyterms=["Aiva", "Saga"], log=lambda m: None)
    s = stt._settings
    assert (s.model, s.eot_threshold, s.eot_timeout_ms, s.keyterm) == (FLUX_MODEL, 0.75, 4000, ["Aiva", "Saga"])
    assert s.eager_eot_threshold is None and not stt.eager_end_of_turn_enabled
    assert stt._init_sample_rate == 16000 and stt._websocket is None
    stt._sample_rate = 16000
    q = stt._build_query_string()
    assert "model=flux-general-en" in q and "eot_threshold=0.75" in q and "eager_eot" not in q
    assert "sample_rate=16000" in q


def test_build_nova_offline():
    from pipecat.services.deepgram.stt import DeepgramSTTService
    stt = build_stt(STTConfig.from_env({"AIVA_STT": "nova"}), api_key="dummy", keyterms=["Aiva"])
    assert isinstance(stt, DeepgramSTTService)


def test_stage_params_per_engine():
    from pipecat.turns.user_start.external_user_turn_start_strategy import ExternalUserTurnStartStrategy
    from pipecat.turns.user_stop.external_user_turn_stop_strategy import ExternalUserTurnStopStrategy
    mute = object()
    p = stage_params(STTConfig.from_env({}), [mute])
    assert p.vad_analyzer is None and p.user_mute_strategies == [mute]
    assert isinstance(p.user_turn_strategies.start[0], ExternalUserTurnStartStrategy)
    assert not p.user_turn_strategies.start[0]._enable_interruptions
    assert isinstance(p.user_turn_strategies.stop[0], ExternalUserTurnStopStrategy)
    n = stage_params(STTConfig.from_env({"AIVA_STT": "nova"}), [mute])
    assert n.vad_analyzer is not None and n.user_mute_strategies == [mute]


class _WS:
    def __init__(self):
        self.state = State.OPEN
        self.sent = []

    async def send(self, data):
        self.sent.append(data)


async def test_flux_reopens_after_idle_close():
    logs = []
    stt = build_stt(STTConfig.from_env({}), api_key="dummy", keyterms=[], log=logs.append)
    stt._sample_rate = 16000
    calls = []

    async def fake_disconnect():
        calls.append("disconnect")
        stt._websocket = None

    async def fake_connect():
        calls.append("connect")
        stt._websocket = _WS()

    stt._disconnect, stt._connect = fake_disconnect, fake_connect
    chunk = b"\x01\x00" * 640
    out = [f async for f in stt.run_stt(chunk)]           # socket was closed (asleep) → reopened
    assert calls == ["disconnect", "connect"] and out == [None]
    assert stt._websocket.sent == [chunk]
    assert any("reopened" in m for m in logs)
    _ = [f async for f in stt.run_stt(chunk)]             # open → just sends
    assert calls == ["disconnect", "connect"] and len(stt._websocket.sent) == 2


async def test_flux_reopen_failure_backs_off():
    stt = build_stt(STTConfig.from_env({}), api_key="dummy", keyterms=[], log=lambda m: None)
    stt._sample_rate = 16000
    calls = []

    async def noop():
        calls.append(1)

    stt._disconnect, stt._connect = noop, noop             # connect fails: socket stays None
    assert [f async for f in stt.run_stt(b"\x01\x00")] == [None]
    n = len(calls)
    assert [f async for f in stt.run_stt(b"\x01\x00")] == [None]
    assert len(calls) == n                                 # backing off: no handshake per chunk


async def test_empty_final_neither_rejects_a_wake_nor_passes():
    mic = MicController(ambient=True, log=lambda m: None)
    mic.wake("test", preroll=b"\x00\x00" * 10, verify=True)
    assert mic.verifying
    finals = []

    async def on_final(text):
        finals.append(text)

    w = ActivityWatcher(mic, lambda t: "aiva" in t.lower(), on_user_final=on_final, log=lambda m: None)
    await run_test(w, frames_to_send=[TranscriptionFrame(text="  ", user_id="", timestamp="t"), SleepFrame(0.05)],
                   expected_down_frames=[])
    assert mic.verifying and finals == []
    await run_test(w, frames_to_send=[TranscriptionFrame(text="hey Aiva", user_id="", timestamp="t")],
                   expected_down_frames=[TranscriptionFrame])
    assert not mic.verifying and mic.awake and finals == ["hey Aiva"]
