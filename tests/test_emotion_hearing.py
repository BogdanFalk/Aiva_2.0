"""Emotion hearing: the tolerant parser, merging into the DSP affect, the note, the request shape,
the call with a fake OpenAI client (metrics + limits + timeout + param retry), and the Floor's
late-reading routing. All offline."""
import asyncio
import base64
import io
import json
import sqlite3
import wave
from types import SimpleNamespace as NS

import pytest

from modules import emotion_hearing as eh
from modules import house_proto as p


def test_parse_strict_json():
    out = eh.parse_emotion(json.dumps({
        "emotions": [{"name": "fatigue", "confidence": 0.9}, {"name": "amusement", "confidence": 0.6}],
        "bursts": [{"name": "laugh", "confidence": 0.8}], "valence": 0.2, "arousal": -0.5}))
    assert out["labels"] == [{"name": "fatigue", "confidence": 0.9}, {"name": "amusement", "confidence": 0.6}]
    assert out["bursts"] == [{"name": "laugh", "confidence": 0.8}]
    assert (out["valence"], out["arousal"]) == (0.2, -0.5)


def test_parse_tolerant_fences_synonyms_unknowns_and_clamps():
    raw = ("Sure! ```json\n{\"emotions\": [{\"name\": \"Tired\", \"confidence\": 1.7}, "
           "{\"name\": \"schadenfreude-ish\", \"confidence\": 0.9}, \"amused\", 42], "
           "\"bursts\": [\"laughter\", {\"name\": \"burp\"}], \"valence\": \"-3\", \"arousal\": null}\n```")
    out = eh.parse_emotion(raw)
    names = [l["name"] for l in out["labels"]]
    assert names == ["fatigue", "amusement"]            # unknown dropped, synonyms mapped
    assert out["labels"][0]["confidence"] == 1.0          # clamped
    assert out["bursts"] == [{"name": "laugh", "confidence": 0.8}]
    assert out["valence"] == -1.0 and out["arousal"] == 0.0


def test_parse_dict_form_and_caps_to_four():
    out = eh.parse_emotion({"emotions": {"joy": 0.9, "warmth": 0.8, "calm": 0.7, "relief": 0.6, "pride": 0.5}})
    assert len(out["labels"]) == eh.MAX_LABELS and out["labels"][0]["name"] == "joy"


@pytest.mark.parametrize("raw", [None, "", "no json here", "[1,2]", '{"emotions": [], "bursts": []}',
                                 '{"emotions": [{"name": "zorg"}]}'])
def test_parse_nothing_usable(raw):
    assert eh.parse_emotion(raw) is None


def test_merge_affect_with_and_without_dsp():
    emo = {"labels": [{"name": "fatigue", "confidence": 0.9}], "bursts": [], "valence": -0.5, "arousal": -1.0}
    dsp = {"arousal": 0.2, "valence": 0.1, "salience": 0.08, "label": "steady", "features": {}}
    m = eh.merge_affect(dsp, emo)
    assert m["arousal"] == pytest.approx((0.2 - 0.6) / 2) and m["valence"] == pytest.approx(-0.3)
    assert m["label"] == "tired" and m["emotion"] is emo and m["salience"] > 0.08
    assert "features" in m and dsp["label"] == "steady"        # original untouched
    alone = eh.merge_affect(None, emo)
    assert alone["arousal"] == pytest.approx(-0.6) and alone["label"] == "tired"
    assert eh.merge_affect(dsp, None) is dsp
    assert eh.late_affect(emo) == alone


def test_note_only_when_confident():
    emo = {"labels": [{"name": "fatigue", "confidence": 0.9}, {"name": "amusement", "confidence": 0.75}],
           "bursts": [{"name": "laugh", "confidence": 0.8}]}
    n = eh.emotion_note(emo, 0.7)
    assert n.startswith("[He sounds tired and a little amused — he laughed.") and n.endswith("]")
    assert "A moment ago he sounded tired" in eh.emotion_note(emo, 0.7, late=True)
    low = {"labels": [{"name": "fatigue", "confidence": 0.5}], "bursts": []}
    assert eh.emotion_note(low, 0.7) is None
    plain = {"labels": [{"name": "neutral", "confidence": 0.95}, {"name": "calm", "confidence": 0.9}], "bursts": []}
    assert eh.emotion_note(plain, 0.7) is None
    sigh = {"labels": [], "bursts": [{"name": "sigh", "confidence": 0.9}]}
    assert eh.emotion_note(sigh, 0.7).startswith("[He just sighed.")
    assert eh.emotion_note(None) is None
    for bad in ("*", "#", "\n"):
        assert bad not in n                                  # spoken-prose friendly


def test_wav_keeps_the_last_max_secs():
    pcm = (b"\x01\x00" * 16000) + (b"\x02\x00" * 16000 * 2)     # 3 s
    wav = eh.pcm_to_wav(pcm, max_secs=2.0)
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16000, 32000)
        assert w.readframes(1) == b"\x02\x00"


def test_request_shape():
    msgs = eh.build_messages(b"RIFFxx")
    part = msgs[1]["content"][1]
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "wav"
    assert base64.b64decode(part["input_audio"]["data"]) == b"RIFFxx"
    kw = eh.request_kwargs("gpt-audio-mini", msgs)
    assert kw["modalities"] == ["text"] and kw["max_tokens"] == 220 and "reasoning_effort" not in kw
    assert "reasoning_effort" not in eh.request_kwargs("gpt-5.6-luna", msgs)


class FakeCompletions:
    def __init__(self, content='{"emotions": [{"name": "amusement", "confidence": 0.8}], "bursts": [], '
                               '"valence": 0.4, "arousal": 0.3}', delay=0.0, reject=None):
        self.content, self.delay, self.reject, self.calls = content, delay, reject, []

    async def create(self, **kw):
        self.calls.append(kw)
        if self.reject and self.reject in kw:
            raise RuntimeError(f"Unsupported parameter: '{self.reject}'")
        if self.delay:
            await asyncio.sleep(self.delay)
        usage = NS(prompt_tokens=150, completion_tokens=30, prompt_tokens_details=NS(audio_tokens=100))
        return NS(choices=[NS(message=NS(content=self.content))], usage=usage)


class FakeLimits:
    def __init__(self):
        self.recorded = []

    async def record(self, service, amount=1):
        self.recorded.append((service, amount))


def _client(comp):
    return NS(chat=NS(completions=comp))


PCM_2S = b"\x00\x01" * 32000


@pytest.mark.asyncio
async def test_hear_with_fake_client(monkeypatch):
    rec = []
    from modules import metrics
    monkeypatch.setattr(metrics, "record", lambda *a, **k: rec.append((a, k)))
    comp, lim = FakeCompletions(), FakeLimits()
    h = eh.EmotionHearing(eh.EmotionConfig(), _client(comp), limits=lim, log=lambda *_: None)
    out = await h.hear(PCM_2S)
    assert out["labels"][0]["name"] == "amusement" and out["model"] == "gpt-audio-mini"
    assert comp.calls[0]["model"] == "gpt-audio-mini" and comp.calls[0]["modalities"] == ["text"]
    assert lim.recorded == [("emotion", 1)]
    (args, kw), = rec
    assert args[:2] == ("llm", "emotion") and kw["tokens"] == 180
    assert kw["cost"] == pytest.approx((50 * 0.60 + 100 * 10.0 + 30 * 2.40) / 1e6)


@pytest.mark.asyncio
async def test_hear_skips_short_disabled_and_survives_errors():
    comp = FakeCompletions()
    h = eh.EmotionHearing(eh.EmotionConfig(), _client(comp), log=lambda *_: None)
    assert await h.hear(b"\x00\x00" * 8000) is None and not comp.calls          # 0.5 s < 0.8 s
    assert eh.EmotionHearing(eh.EmotionConfig(enabled=False), _client(comp)).enabled is False
    assert eh.EmotionHearing(eh.EmotionConfig(), None).enabled is False
    slow = eh.EmotionHearing(eh.EmotionConfig(timeout_s=0.01), _client(FakeCompletions(delay=0.5)),
                             log=lambda *_: None)
    assert await slow.hear(PCM_2S) is None
    junk = eh.EmotionHearing(eh.EmotionConfig(), _client(FakeCompletions(content="I can't")), log=lambda *_: None)
    assert await junk.hear(PCM_2S) is None


@pytest.mark.asyncio
async def test_hear_retries_without_a_rejected_param():
    comp = FakeCompletions(reject="temperature")
    h = eh.EmotionHearing(eh.EmotionConfig(), _client(comp), log=lambda *_: None)
    assert (await h.hear(PCM_2S)) is not None
    assert "temperature" in comp.calls[0] and "temperature" not in comp.calls[1]
    await h.hear(PCM_2S)
    assert "temperature" not in comp.calls[2]                                # remembered


# --- the Floor routes a late reading -----------------------------------------------------------

EMO = {"labels": [{"name": "fatigue", "confidence": 0.9}], "bursts": [], "valence": -0.3, "arousal": -0.6}


def _floor():
    from modules.house_floor import ClientInfo, Floor
    from modules.house_memory import HouseMemory
    from modules.sisters import ExchangeConfig

    class C:
        def __init__(self, slug, name):
            self.info, self.sent = ClientInfo(slug, name, "", True), []

        def send(self, m):
            self.sent.append(m)

    a, s = C("aiva", "Aiva"), C("saga", "Saga")
    f = Floor(HouseMemory(sqlite3.connect(":memory:")), private=True, log=lambda *_: None,
              clients={"aiva": a, "saga": s}, cfg=ExchangeConfig())
    return f, a, s


@pytest.mark.asyncio
async def test_late_affect_sent_to_the_holder_and_the_overhearer():
    f, a, s = _floor()
    await f.submit_user_turn("Aiva, I'm so tired today", affect={"arousal": 0.1, "valence": 0.0, "salience": 0.08})
    tid = f.last_user_turn_id
    assert tid and a.sent[-1].__class__ is p.TurnMsg                       # granted already
    where = f.late_affect(tid, EMO)
    assert where == {"aiva": "sent", "saga": "sent"}
    msg = [m for m in a.sent if isinstance(m, p.Affect)][-1]
    assert msg.turn == tid and msg.affect["emotion"] == EMO and msg.affect["label"] == "tired"
    assert f.late_affect(tid, None) == {} and f.late_affect(None, EMO) == {}


@pytest.mark.asyncio
async def test_late_affect_folds_into_a_queued_turn():
    f, a, s = _floor()
    await f.submit_user_turn("Aiva and Saga, how are you both", affect=None)
    tid = f.last_user_turn_id
    queued = [t for t in f._queue if t.kind == "both_second"]
    assert queued, "the second sister's turn waits behind the first"
    where = f.late_affect(tid, EMO)
    assert where[queued[0].mind] == "queued"
    assert queued[0].affect["emotion"] == EMO
    other = "aiva" if queued[0].mind == "saga" else "saga"
    assert where[other] == "sent"


@pytest.mark.asyncio
async def test_unrouted_line_has_no_turn_id():
    f, a, s = _floor()
    f.clients.clear()
    await f.submit_user_turn("hello?")
    assert f.last_user_turn_id is None
