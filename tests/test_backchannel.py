"""Backchannels (modules/backchannel.py + the house/sister plumbing), all offline:
the pure decision and its gates, the per-turn pause tracker (clock + dice injected), target choice,
clip choice, echo removal from his final line, the clip cache with a fake ElevenLabs, the sister's
handler with fakes, the proto message, the Flux turn-info hook, the player's frames and HouseLink
swallowing the clip's speaking reports, and the avatar nod."""
import asyncio
import os

import pytest
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, TTSAudioRawFrame,
                                   TTSStartedFrame, TTSStoppedFrame)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test

from modules import backchannel as bc
from modules import house_proto as proto
from modules import settings_registry as reg
from modules.avatar_activity import NOD_DEG, NOD_SECS, ActivityDirector
from modules.house_link import BackchannelPlayer, HouseLink

CFG = bc.BackchannelConfig()
LINE = "so I was walking home from the station yesterday"


def _decide(**kw):
    base = dict(in_turn=True, speech_s=5.0, pause_s=0.8, text=LINE, eot=0.3, eot_ceiling=0.6,
                since_last_s=60.0, blocked=None, roll=0.1)
    base.update(kw)
    return bc.decide(CFG, **base)


# --- config -----------------------------------------------------------------------------------------

def test_config_defaults_match_registry_and_parse():
    c = bc.BackchannelConfig.from_env({})
    d = lambda k: reg.BY_KEY[k].default   # noqa: E731
    assert c.enabled is d("AIVA_BACKCHANNEL") is True
    assert (c.cooldown_s, c.probability, c.min_speech_s, c.pause_s, c.volume) == (
        d("AIVA_BACKCHANNEL_COOLDOWN_S"), d("AIVA_BACKCHANNEL_PROBABILITY"), d("AIVA_BACKCHANNEL_MIN_SPEECH_S"),
        d("AIVA_BACKCHANNEL_PAUSE_S"), d("AIVA_BACKCHANNEL_VOLUME"))
    off = bc.BackchannelConfig.from_env({"AIVA_BACKCHANNEL": "0", "AIVA_BACKCHANNEL_PROBABILITY": "2",
                                         "AIVA_BACKCHANNEL_COOLDOWN_S": "x", "AIVA_BACKCHANNEL_PAUSE_S": "1.2"})
    assert not off.enabled and off.probability == 0.5 and off.cooldown_s == 6.0 and off.pause_s == 1.2


# --- the pure decision --------------------------------------------------------------------------

def test_decide_fires_at_a_good_pause():
    assert _decide() == (True, "pause")


@pytest.mark.parametrize("kw,reason", [
    (dict(in_turn=False), "no turn"),
    (dict(blocked="floor busy"), "floor busy"),
    (dict(speech_s=2.0), "too early in his turn"),
    (dict(since_last_s=3.0), "cooldown"),
    (dict(pause_s=0.3), "no pause yet"),
    (dict(eot=0.65), "he's finishing"),
    (dict(text="so I was"), "too few words"),
    (dict(text="what do you think about the new layout"), "a question"),
    (dict(text="I moved the couch over, right?"), "a question"),
    (dict(text="and then I went over to the"), "mid-phrase"),
    (dict(text="I was trying to fix the, um"), "mid-phrase"),
    (dict(roll=0.5), "dice"),
])
def test_decide_gates(kw, reason):
    assert _decide(**kw) == (False, reason)


def test_decide_disabled_and_unknown_eot():
    assert bc.decide(bc.BackchannelConfig(enabled=False), in_turn=True, speech_s=9, pause_s=1, text=LINE,
                     eot=None, eot_ceiling=0.6, since_last_s=99, blocked=None, roll=0) == (False, "disabled")
    assert _decide(eot=None) == (True, "pause")          # no confidence: the text heuristics decide


# --- tracker --------------------------------------------------------------------------------------

def _tracker(rolls=(0.1,), **cfg):
    seq = list(rolls)
    return bc.Backchanneler(bc.BackchannelConfig(**cfg), eot_ceiling=0.6,
                            rand=lambda: seq.pop(0) if len(seq) > 1 else seq[0], log=lambda m: None)


def test_tracker_judges_each_pause_once_and_respects_cooldown():
    b = _tracker()
    b.observe("StartOfTurn", "so", None, 0.0)
    b.observe("Update", "so I was walking home", 0.1, 1.0)
    assert b.tick(1.5) is None                            # still talking / too early
    b.observe("Update", LINE, 0.2, 3.5)
    assert b.tick(3.9) is None                            # 0.4 s: no pause yet
    clip, reason, info = b.tick(4.2)
    assert clip in bc.PHRASES and reason == "pause" and info["words"] == len(bc.words(LINE))
    assert b.tick(4.5) is None                            # judged already
    b.observe("Update", LINE + " and it was raining hard", 0.1, 5.0)
    res = b.tick(6.0)
    assert res[0] is None and res[1] == "cooldown"
    b.observe("Update", LINE + " and it was raining hard so I ran", 0.1, 10.0)
    assert b.tick(10.7)[0] is not None                    # cooldown over: fires again
    assert b.partial.endswith("so I ran")


def test_tracker_blocked_is_lazy_and_end_of_turn_resets():
    b = _tracker()
    calls = []
    b.observe("StartOfTurn", LINE, None, 0.0)
    b.observe("Update", LINE, 0.2, 3.5)
    res = b.tick(4.2, blocked=lambda: calls.append(1) or "asleep")
    assert res[0] is None and res[1] == "asleep" and calls == [1]
    b.observe("EndOfTurn", LINE, 0.9, 4.5)
    assert not b.in_turn and b.tick(9.0) is None


def test_tracker_undo_fire_forgets_cooldown_and_echo():
    b = _tracker()
    b.observe("StartOfTurn", LINE, None, 0.0)
    b.observe("Update", LINE, 0.2, 3.5)
    assert b.tick(4.2)[0] is not None
    b.undo_fire()
    b.observe("Update", LINE + " and then", 0.2, 4.4)
    b.observe("Update", LINE + " and then the bus came", 0.2, 4.6)
    assert b.tick(5.3)[0] is not None                     # no cooldown from the unsent one


# --- target + clip choice -------------------------------------------------------------------------

def test_target_order():
    both = ["aiva", "saga"]
    assert bc.target({"saga"}, "aiva", 5, "aiva", both, 60) == "saga"            # his words name her
    assert bc.target({"saga", "aiva"}, "aiva", 5, "aiva", both, 60) is None       # names both
    assert bc.target(set(), "saga", 10, "aiva", both, 60) == "saga"              # sticky addressee
    assert bc.target(set(), "saga", 120, "aiva", both, 60) == "aiva"             # stale → last speaker
    assert bc.target(set(), "both", 5, "saga", both, 60) == "saga"               # "both" → last speaker
    assert bc.target(set(), None, None, None, ["aiva"], 60) == "aiva"            # only one here
    assert bc.target(set(), None, None, None, both, 60) is None
    assert bc.target({"saga"}, None, None, None, ["aiva"], 60) == "aiva"         # named one isn't here


def test_choose_clip_groups_and_no_repeat():
    assert bc.choose_clip(LINE, roll=0.1) in bc.AFFIRMS
    assert bc.choose_clip(LINE, roll=0.9) in bc.CONTINUERS
    assert bc.choose_clip("it was like um kind of weird", roll=0.1) in bc.CONTINUERS   # hesitant
    for pick in (0.0, 0.5, 0.99):
        assert bc.choose_clip(LINE, recent=["mhm"], roll=0.9, pick=pick) != "mhm"
    assert bc.choose_clip(LINE, roll=0.9, available={"uh_huh"}) == "uh_huh"


# --- echo -------------------------------------------------------------------------------------------

def test_strip_echo_removes_only_her_clip_at_the_fire_point():
    n = len(bc.words(LINE))
    final = LINE + " mm-hmm and it was raining"
    assert bc.strip_echo(final, [(n, "mm_hmm")]) == LINE + " and it was raining"
    assert bc.strip_echo(LINE + " Mhm. And it rained.", [(n, "mhm")]) == LINE + ". And it rained."
    # "yeah" said by him elsewhere stays; only the one at the fire point goes
    line = "yeah so I was walking home from the station"
    k = len(bc.words(line))
    assert bc.strip_echo(line + " yeah and then", [(k, "yeah")]) == line + " and then"
    # nothing at that spot → untouched
    assert bc.strip_echo(LINE + " and it was raining", [(n, "mhm")]) == LINE + " and it was raining"
    assert bc.strip_echo("hmm " + LINE, [(n, "mhm")]) == "hmm " + LINE


def test_tracker_strips_echo_from_the_matching_final_only():
    b = _tracker()
    b.observe("StartOfTurn", LINE, None, 0.0)
    b.observe("Update", LINE, 0.2, 3.5)
    clip = b.tick(4.2)[0]
    b.observe("EndOfTurn", LINE, 0.9, 6.0)
    other = "turn on the lights mhm please"
    assert b.strip_final(other, 6.1) == other                       # a different turn: untouched
    spoken = {"mhm": "mhm", "mm_hmm": "mm-hmm", "uh_huh": "uh-huh", "yeah": "yeah", "right": "right"}[clip]
    assert b.strip_final(f"{LINE} {spoken} and then", 6.2) == f"{LINE} and then"
    assert b.strip_final(f"{LINE} {spoken} and then", 6.3) == f"{LINE} {spoken} and then"   # used once


# --- clip cache --------------------------------------------------------------------------------------

class _Resp:
    def __init__(self, content, status=200):
        self.content, self.status_code = content, status


def _tone(secs, amp=8000, lead=0.2):
    import array
    import math
    sr = bc.SAMPLE_RATE
    a = array.array("h", [0] * int(lead * sr))
    a.extend(int(amp * math.sin(2 * math.pi * 220 * i / sr)) for i in range(int(secs * sr)))
    a.extend([0] * int(lead * sr))
    return a.tobytes()


def test_clip_cache_synthesises_once_keyed_by_voice(tmp_path):
    posts = []

    def post(url, **kw):
        posts.append((url, kw["json"], kw["headers"]["xi-api-key"]))
        return _Resp(_tone(0.4))

    c = bc.ClipCache("saga", "VOICE1", "eleven_flash_v2_5", "KEY", root=str(tmp_path), post=post, log=lambda m: None)
    assert c.get("mhm") is None and set(c.missing()) == set(bc.PHRASES)
    made = c.ensure()
    assert sorted(made) == sorted(bc.PHRASES) and len(posts) == len(bc.PHRASES)
    assert all("VOICE1" in u and "pcm_24000" in u and j["model_id"] == "eleven_flash_v2_5" and k == "KEY"
               for u, j, k in posts)
    pcm, rate = c.get("mhm")
    assert rate == bc.SAMPLE_RATE and 0.4 <= len(pcm) / 2 / rate <= 0.5      # leading/trailing silence trimmed
    assert c.ensure() == [] and len(posts) == len(bc.PHRASES)                  # cached: no second call
    c2 = bc.ClipCache("saga", "VOICE2", "eleven_flash_v2_5", "KEY", root=str(tmp_path), post=post, log=lambda m: None)
    assert set(c2.missing()) == set(bc.PHRASES)                                # a new voice regenerates
    assert os.path.isdir(os.path.join(str(tmp_path), "saga"))


def test_clip_cache_failures_and_no_voice(tmp_path):
    logs = []
    c = bc.ClipCache("aiva", "V", "m", "K", root=str(tmp_path), log=logs.append,
                     post=lambda url, **kw: _Resp(b"", 401))
    assert c.ensure() == [] and any("HTTP 401" in m for m in logs)
    silent = bc.ClipCache("aiva", "V", "m", "K", root=str(tmp_path), log=logs.append,
                          post=lambda url, **kw: _Resp(b"\x00\x00" * 24000))
    assert silent.ensure(["mhm"]) == [] and any("silent" in m for m in logs)

    def boom(url, **kw):
        raise OSError("offline")
    assert bc.ClipCache("aiva", "V", "m", "K", root=str(tmp_path), post=boom, log=logs.append).ensure() == []
    assert bc.ClipCache("aiva", None, "m", "K", root=str(tmp_path), post=boom).ensure() == []


def test_trim_caps_long_clips_and_scale_saturates():
    long = bc.trim_silence(_tone(3.0), max_s=1.2)
    assert len(long) / 2 / bc.SAMPLE_RATE == pytest.approx(1.2, abs=0.01)
    import array
    a = array.array("h", [20000, -20000, 100])
    out = array.array("h")
    out.frombytes(bc.scale_pcm(a.tobytes(), 2.0))
    assert list(out) == [32767, -32768, 200]
    assert bc.scale_pcm(a.tobytes(), 1.0) == a.tobytes()


# --- the sister's handler ------------------------------------------------------------------------

class _Avatar:
    def __init__(self):
        self.cues = []

    def send_cue(self, kind):
        self.cues.append(kind)


class _Player:
    def __init__(self, ok=True):
        self.played, self.ok = [], ok

    async def play(self, pcm, rate):
        self.played.append((len(pcm), rate))
        return self.ok


class _Cache:
    def __init__(self, clips):
        self.clips = clips

    def get(self, clip):
        return self.clips.get(clip)


@pytest.mark.asyncio
async def test_voice_handler_plays_and_nods():
    av, pl = _Avatar(), _Player()
    v = bc.BackchannelVoice(CFG, cache=_Cache({"mhm": (b"\x10\x00" * 100, 24000)}), player=pl, avatar=av)
    assert await v.handle("mhm") == "voice" and av.cues == ["nod"] and pl.played == [(200, 24000)]
    assert await v.handle("yeah") == "nod"                          # not cached yet: nod only
    assert await v.handle("mhm", voice=False) == "nod"              # her voice is muted in the house
    assert len(pl.played) == 1 and av.cues == ["nod"] * 3


@pytest.mark.asyncio
async def test_voice_handler_busy_off_and_voiceless():
    av, pl = _Avatar(), _Player()
    cache = _Cache({"mhm": (b"\x10\x00" * 100, 24000)})
    busy = bc.BackchannelVoice(CFG, cache=cache, player=pl, avatar=av, busy=lambda: True)
    assert await busy.handle("mhm") == "busy" and av.cues == [] and pl.played == []
    off = bc.BackchannelVoice(bc.BackchannelConfig(enabled=False), cache=cache, player=pl, avatar=av)
    assert await off.handle("mhm") == "off"
    voiceless = bc.BackchannelVoice(CFG, cache=None, player=None, avatar=av)
    assert await voiceless.handle("mhm") == "nod" and av.cues == ["nod"]
    refused = bc.BackchannelVoice(CFG, cache=cache, player=_Player(ok=False), avatar=None)
    assert await refused.handle("mhm") == "nod"


# --- proto + flux hook ------------------------------------------------------------------------------

def test_proto_backchannel_round_trip():
    m = proto.Backchannel(id=3, ts=1.5, clip="mhm", voice=False)
    back = proto.decode(proto.encode(m))
    assert back == m and isinstance(back, proto.Backchannel)
    assert proto.decode(b'{"t": "backchannel", "clip": "yeah"}\n').voice is True
    assert proto.Backchannel in proto.HOUSE_TO_CHARACTER and proto.Backchannel not in proto.CHARACTER_TO_HOUSE


@pytest.mark.asyncio
async def test_flux_turn_info_hook_sees_updates():
    from modules.house_stt import STTConfig, build_stt
    logs, seen = [], []
    stt = build_stt(STTConfig.from_env({}), api_key="dummy", keyterms=[], log=logs.append)
    stt.on_turn_info = lambda ev, text, eot: seen.append((ev, text, eot))
    await stt._handle_turn_info({"type": "TurnInfo", "event": "Update", "transcript": "so I was",
                                 "end_of_turn_confidence": 0.12})
    assert seen == [("Update", "so I was", 0.12)]

    def bad(*a):
        raise RuntimeError("boom")
    stt.on_turn_info = bad
    await stt._handle_turn_info({"type": "TurnInfo", "event": "Update", "transcript": "x"})
    assert any("turn-info hook failed" in m for m in logs)


# --- player + HouseLink ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_player_pushes_a_non_context_tts_burst():
    link = HouseLink(LLMContext(messages=[]), log=lambda m: None)
    player = BackchannelPlayer(link)
    assert await player.play(b"\x01\x00" * 10, 24000) is False      # not started yet
    played = []

    async def drive():
        while not player.ready:
            await asyncio.sleep(0.01)
        played.append(await player.play(b"\x01\x00" * 2400, 24000))

    (down, _), _ = await asyncio.gather(
        run_test(player, frames_to_send=[SleepFrame(0.2)],
                 expected_down_frames=[TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame]),
        drive(),
    )
    assert played == [True]
    assert down[0].append_to_context is False and down[1].sample_rate == 24000
    assert link.backchanneling                                      # its speaking is expected

    link.current_turn = {"id": 4}
    assert await player.play(b"\x01\x00" * 10, 24000) is False      # never over her own turn


@pytest.mark.asyncio
async def test_house_link_swallows_the_clips_speaking_reports():
    speaking = []
    link = HouseLink(LLMContext(messages=[]), on_speaking=lambda tid, on: speaking.append((tid, on)),
                     log=lambda m: None)
    link.last_turn_id = 5
    link.expect_backchannel()
    await run_test(
        link,
        frames_to_send=[BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(),     # the clip
                        BotStartedSpeakingFrame(), BotStoppedSpeakingFrame()],    # a real reply
        frames_to_send_direction=FrameDirection.UPSTREAM,
        expected_up_frames=[BotStartedSpeakingFrame, BotStoppedSpeakingFrame,
                            BotStartedSpeakingFrame, BotStoppedSpeakingFrame],
    )
    assert speaking == [(5, True), (5, False)]
    assert not link.backchanneling


# --- the nod ---------------------------------------------------------------------------------------

def test_nod_dips_and_returns_with_and_without_a_rig():
    from modules.avatar_activity import RIGS
    for rig in (None, RIGS["ariu"]):
        d = ActivityDirector(rig)
        assert d.cue("nod", 10.0)
        assert "ParamAngleY" not in d.step(0.016, 9.9).params
        mid = d.step(0.016, 10.0 + NOD_SECS / 2).params["ParamAngleY"]
        assert mid == pytest.approx(-NOD_DEG)
        assert "ParamAngleY" not in d.step(0.016, 10.0 + NOD_SECS + 0.01).params


@pytest.mark.asyncio
async def test_assistant_aggregator_ignores_a_clip():
    """Downstream of her transport: a clip opens no assistant turn and adds nothing to her context."""
    from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
    ctx = LLMContext(messages=[{"role": "system", "content": "sys"}])
    agg = LLMAssistantAggregator(ctx)
    events = []

    @agg.event_handler("on_assistant_turn_started")
    async def _started(_a):
        events.append("started")

    @agg.event_handler("on_assistant_turn_stopped")
    async def _stopped(_a, _m):
        events.append("stopped")

    await run_test(agg, frames_to_send=[TTSStartedFrame(append_to_context=False), BotStartedSpeakingFrame(),
                                        TTSAudioRawFrame(audio=b"\x01\x00" * 240, sample_rate=24000, num_channels=1),
                                        TTSStoppedFrame(), BotStoppedSpeakingFrame()])
    assert events == [] and ctx.get_messages() == [{"role": "system", "content": "sys"}]
