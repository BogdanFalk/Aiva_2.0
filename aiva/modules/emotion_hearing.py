"""Emotion hearing — the model-based second analyzer next to `paralinguistics` (DSP).

Once per user turn the house sends the same 16 kHz PCM the speaker tap captured (as a WAV, the
last `max_secs`) to an OpenAI audio-input chat model (`AIVA_EMOTION_MODEL`, default
gpt-audio-mini) and asks for strict JSON: a few emotion labels from a FIXED vocabulary with a
confidence each, vocal bursts (laugh, sigh, yawn...), and valence/arousal in -1..1.

It must never delay her reply: `house.py` runs `hear()` as a background task; a result that is in
before the turn goes out rides on the turn's `affect` (`merge_affect`), a later one goes to the
Floor (`Floor.late_affect`) which folds it into a still-queued turn or sends the sister a
`proto.Affect` update. On the sister side `emotion_note` turns a CONFIDENT reading into a short
spoken-prose-friendly note ("[He sounds tired and a little amused — he laughed. ...]").

Everything here except `EmotionHearing.hear` is pure and offline-testable.
"""

import asyncio
import base64
import io
import json
import os
import re
import time
import wave
from dataclasses import dataclass

SR = 16000

# the fixed vocabulary — anything else the model says is dropped
VOCAB = (
    "amusement", "joy", "excitement", "warmth", "affection", "gratitude", "pride", "relief",
    "contentment", "calm", "curiosity", "interest", "playfulness", "teasing", "surprise",
    "confusion", "doubt", "boredom", "fatigue", "sleepiness", "sadness", "longing", "loneliness",
    "disappointment", "frustration", "annoyance", "anger", "stress", "anxiety", "embarrassment",
    "determination", "focus", "sarcasm", "neutral",
)
BURSTS = ("laugh", "chuckle", "sigh", "yawn", "groan", "gasp", "sniffle")

# loose words a model may use instead of the vocabulary name
SYNONYMS = {
    "amused": "amusement", "humor": "amusement", "humour": "amusement", "happy": "joy",
    "happiness": "joy", "excited": "excitement", "enthusiasm": "excitement", "warm": "warmth",
    "tenderness": "affection", "love": "affection", "grateful": "gratitude", "thankful": "gratitude",
    "relieved": "relief", "content": "contentment", "satisfaction": "contentment",
    "relaxed": "calm", "calmness": "calm", "curious": "curiosity", "interested": "interest",
    "playful": "playfulness", "surprised": "surprise", "confused": "confusion",
    "uncertainty": "doubt", "unsure": "doubt", "bored": "boredom", "tired": "fatigue",
    "tiredness": "fatigue", "exhaustion": "fatigue", "exhausted": "fatigue", "sleepy": "sleepiness",
    "sad": "sadness", "melancholy": "sadness", "nostalgia": "longing", "lonely": "loneliness",
    "disappointed": "disappointment", "frustrated": "frustration", "irritation": "annoyance",
    "irritated": "annoyance", "annoyed": "annoyance", "angry": "anger", "stressed": "stress",
    "tension": "stress", "tense": "stress", "anxious": "anxiety", "nervous": "anxiety",
    "worry": "anxiety", "worried": "anxiety", "embarrassed": "embarrassment",
    "determined": "determination", "focused": "focus", "concentration": "focus",
    "sarcastic": "sarcasm", "neutrality": "neutral",
    # bursts
    "laughter": "laugh", "laughing": "laugh", "giggle": "chuckle", "chuckling": "chuckle",
    "sighing": "sigh", "yawning": "yawn", "groaning": "groan", "sniff": "sniffle",
}

# how a label reads in a note ("he sounds ...")
ADJECTIVE = {
    "amusement": "amused", "joy": "happy", "excitement": "excited", "warmth": "warm",
    "affection": "affectionate", "gratitude": "grateful", "pride": "proud", "relief": "relieved",
    "contentment": "content", "calm": "calm", "curiosity": "curious", "interest": "interested",
    "playfulness": "playful", "teasing": "teasing", "surprise": "surprised", "confusion": "confused",
    "doubt": "unsure", "boredom": "bored", "fatigue": "tired", "sleepiness": "sleepy",
    "sadness": "sad", "longing": "wistful", "loneliness": "lonely", "disappointment": "disappointed",
    "frustration": "frustrated", "annoyance": "annoyed", "anger": "angry", "stress": "stressed",
    "anxiety": "anxious", "embarrassment": "embarrassed", "determination": "determined",
    "focus": "focused", "sarcasm": "sarcastic", "neutral": "neutral",
}
BURST_VERB = {"laugh": "laughed", "chuckle": "chuckled", "sigh": "sighed", "yawn": "yawned",
              "groan": "groaned", "gasp": "gasped", "sniffle": "sniffled"}
NOTE_SKIP = {"neutral", "calm", "focus", "interest"}     # plain states are not worth a note

MAX_LABELS = 4

PROMPT = (
    "You listen to one short spoken turn from a man talking to his voice companion at home and "
    "judge how he SOUNDS — tone of voice, pacing, energy, laughter, sighs — not what the words mean. "
    "Reply with ONLY a JSON object, no prose, in exactly this shape:\n"
    '{"emotions": [{"name": "<label>", "confidence": 0.0}], '
    '"bursts": [{"name": "<burst>", "confidence": 0.0}], "valence": 0.0, "arousal": 0.0}\n'
    "emotions: at most 3, strongest first, ONLY from this list: " + ", ".join(VOCAB) + ". "
    "bursts: only non-speech sounds you clearly hear, ONLY from: " + ", ".join(BURSTS) + "; "
    "an empty list if none. confidence is 0..1 and should be low when unsure. "
    "valence -1 (negative) .. 1 (positive) and arousal -1 (low energy) .. 1 (high energy) "
    "describe the voice. If the audio is silent, noisy or not a voice, return empty lists and zeros."
)


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _num(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else f          # NaN -> default


# --- config ---------------------------------------------------------------------------------------

@dataclass
class EmotionConfig:
    enabled: bool = True
    model: str = "gpt-audio-mini"
    min_secs: float = 0.8           # shorter turns aren't worth a call
    max_secs: float = 15.0          # only the last N seconds are sent
    timeout_s: float = 10.0
    note_confidence: float = 0.7    # sister side: a note only at/above this confidence
    note_ttl_s: float = 120.0       # sister side: a late reading is noted on the next turn only this fresh

    @classmethod
    def from_env(cls):
        g = os.getenv
        return cls(
            enabled=g("AIVA_EMOTION_HEARING", "1") == "1",
            model=(g("AIVA_EMOTION_MODEL") or "gpt-audio-mini").strip(),
            min_secs=float(g("AIVA_EMOTION_MIN_SECS", "0.8")),
            max_secs=float(g("AIVA_EMOTION_MAX_SECS", "15")),
            timeout_s=float(g("AIVA_EMOTION_TIMEOUT_S", "10")),
            note_confidence=float(g("AIVA_EMOTION_NOTE_CONFIDENCE", "0.7")),
            note_ttl_s=float(g("AIVA_EMOTION_NOTE_TTL_S", "120")),
        )


# --- audio ----------------------------------------------------------------------------------------

def pcm_to_wav(pcm, sr=SR, max_secs=15.0):
    """16-bit mono PCM -> WAV bytes, keeping only the LAST `max_secs` (the end of his turn)."""
    pcm = bytes(pcm or b"")
    if len(pcm) % 2:
        pcm = pcm[:-1]
    cap = int(sr * max_secs) * 2
    if cap > 0 and len(pcm) > cap:
        pcm = pcm[-cap:]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def build_messages(wav_bytes):
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    return [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "How does he sound in this turn? JSON only."},
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        ]},
    ]


def request_kwargs(model, messages, max_tokens=220):
    """Chat Completions kwargs for an audio-input, text-output call. Goes through
    `utilities.llm_kwargs` like every direct OpenAI call; the gpt-audio line is not gpt-5.x, so that
    yields `max_tokens` + temperature and never `reasoning_effort` (which audio models reject)."""
    from modules.utilities import llm_kwargs
    kw = dict(model=model, messages=messages, modalities=["text"],
              **llm_kwargs(model, max_tokens=max_tokens, temperature=0.0))
    kw.pop("reasoning_effort", None)
    return kw


# --- parsing (pure) -------------------------------------------------------------------------------

def _canon(name, allowed):
    if not isinstance(name, str):
        return None
    n = re.sub(r"[^a-z]", "", name.strip().lower())
    n = SYNONYMS.get(n, n)
    return n if n in allowed else None


def _items(raw, allowed, default_conf):
    """[{"name","confidence"}] | {"name": conf} | ["name"] -> {canonical: conf} (max per name)."""
    out = {}
    if isinstance(raw, dict):
        raw = [{"name": k, "confidence": v} for k, v in raw.items()]
    if not isinstance(raw, list):
        return out
    for it in raw:
        if isinstance(it, str):
            name, conf = it, default_conf
        elif isinstance(it, dict):
            name = it.get("name") or it.get("label") or it.get("emotion") or it.get("type")
            conf = _num(it.get("confidence", it.get("score", it.get("p", default_conf))), default_conf)
        else:
            continue
        c = _canon(name, allowed)
        if c is None:
            continue
        conf = round(_clamp(conf, 0.0, 1.0), 3)
        out[c] = max(out.get(c, 0.0), conf)
    return out


def _json_object(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(s[a:b + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def parse_emotion(text):
    """Tolerant parse of the model's reply -> {"labels": [{"name","confidence"}...] (≤4, strongest
    first, vocabulary only), "bursts": [...], "valence", "arousal"} with everything clamped, or None
    when there is nothing usable."""
    obj = _json_object(text)
    if obj is None:
        return None
    raw_labels = obj.get("emotions", obj.get("labels", obj.get("emotion")))
    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]
    labels = _items(raw_labels, set(VOCAB), 0.5)
    bursts = _items(obj.get("bursts", obj.get("vocal_bursts", [])), set(BURSTS), 0.8)
    valence = round(_clamp(_num(obj.get("valence")), -1.0, 1.0), 3)
    arousal = round(_clamp(_num(obj.get("arousal")), -1.0, 1.0), 3)
    if not labels and not bursts and valence == 0.0 and arousal == 0.0:
        return None
    lab = sorted(labels.items(), key=lambda kv: -kv[1])[:MAX_LABELS]
    bur = sorted(bursts.items(), key=lambda kv: -kv[1])
    return {"labels": [{"name": n, "confidence": c} for n, c in lab],
            "bursts": [{"name": n, "confidence": c} for n, c in bur],
            "valence": valence, "arousal": arousal}


# --- merging + notes (pure) -----------------------------------------------------------------------

MODEL_SCALE = 0.6        # the model's -1..1 lands on the same scale the DSP affect uses (±0.6)


def merge_affect(dsp, emotion):
    """Fold a model reading into the turn's DSP affect dict (either may be None). The result keeps
    the DSP keys (arousal/valence/salience/label) the sister already consumes and adds `emotion`.
    Valence comes from the model (the DSP's is a weak lean); arousal averages both."""
    if not emotion:
        return dsp
    out = dict(dsp or {})
    ev = MODEL_SCALE * _num(emotion.get("valence"))
    ea = MODEL_SCALE * _num(emotion.get("arousal"))
    if dsp:
        out["arousal"] = round((_num(dsp.get("arousal")) + ea) / 2, 3)
    else:
        out["arousal"] = round(ea, 3)
    out["valence"] = round(ev, 3)
    labels = emotion.get("labels") or []
    top = labels[0] if labels else None
    conf = _num(top.get("confidence")) if top else 0.0
    out["salience"] = round(max(_num(out.get("salience"), 0.08), _clamp(0.15 + 0.5 * conf, 0.0, 0.8)), 3)
    if top and conf >= 0.5:
        out["label"] = ADJECTIVE.get(top["name"], top["name"])
    else:
        out.setdefault("label", "steady")
    out["emotion"] = emotion
    return out


def late_affect(emotion):
    """The affect dict a late reading carries on its own (no DSP part — that was already emitted
    with the turn): the model's contribution only."""
    return merge_affect(None, emotion)


def describe(emotion, min_conf=0.0):
    """'tired, amused (laughed)' — for the experience-bus summary and logs."""
    if not emotion:
        return ""
    words = [ADJECTIVE.get(l["name"], l["name"]) for l in emotion.get("labels", [])
             if _num(l.get("confidence")) >= min_conf]
    bursts = [BURST_VERB.get(b["name"], b["name"]) for b in emotion.get("bursts", [])
              if _num(b.get("confidence")) >= min_conf]
    s = ", ".join(words)
    if bursts:
        s += (" " if s else "") + "(" + ", ".join(bursts) + ")"
    return s


def emotion_note(emotion, min_conf=0.7, late=False):
    """A short note for her context, ONLY when the reading is confident — else None.
    '[He sounds tired and a little amused — he laughed. Let it colour how you answer; don't name it.]'"""
    if not emotion:
        return None
    labels = [l for l in emotion.get("labels", [])
              if l.get("name") not in NOTE_SKIP and _num(l.get("confidence")) >= min_conf][:2]
    bursts = [b for b in emotion.get("bursts", []) if _num(b.get("confidence")) >= min_conf][:2]
    if not labels and not bursts:
        return None
    parts = []
    for i, l in enumerate(labels):
        adj = ADJECTIVE.get(l["name"], l["name"])
        if i > 0 and _num(l.get("confidence")) < 0.85:
            adj = "a little " + adj
        parts.append(adj)
    verbs = [BURST_VERB.get(b["name"], b["name"]) for b in bursts]
    lead = "A moment ago he sounded" if late else "He sounds"
    if parts:
        s = f"{lead} {' and '.join(parts)}"
        if verbs:
            s += " — he " + " and ".join(verbs)
    else:
        s = ("A moment ago he " if late else "He just ") + " and ".join(verbs)
    return f"[{s}. Let it colour how you answer if it fits; don't name it.]"


# --- the call -------------------------------------------------------------------------------------

AUDIO_TOKENS_PER_SEC = 10        # audio input is billed ~1 token per 100 ms


class EmotionHearing:
    """`await hear(pcm)` -> parsed emotion dict or None. Never raises; bounded by `timeout_s`.
    `client` is an AsyncOpenAI (or a fake with `.chat.completions.create`); `limits` an optional
    ServiceLimits ('emotion' is an alert-only counter)."""

    def __init__(self, cfg: EmotionConfig = None, client=None, *, limits=None, log=print):
        self.cfg = cfg or EmotionConfig.from_env()
        self.client = client
        self.limits = limits
        self.log = log
        self.enabled = bool(self.cfg.enabled and client is not None)
        self._drop = set()       # params the model rejected once (e.g. temperature)

    def worth_hearing(self, pcm):
        return self.enabled and len(pcm or b"") >= int(SR * self.cfg.min_secs) * 2

    async def hear(self, pcm):
        if not self.worth_hearing(pcm):
            return None
        try:
            return await asyncio.wait_for(self._call(pcm), timeout=self.cfg.timeout_s)
        except asyncio.TimeoutError:
            self.log(f"emotion: timed out after {self.cfg.timeout_s:.0f}s")
        except Exception as e:
            self.log(f"emotion: call failed: {e!r}")
        return None

    async def _call(self, pcm):
        from modules import metrics
        from modules.utilities import AUDIO_IN_RATES, LLM_RATES

        wav = pcm_to_wav(pcm, max_secs=self.cfg.max_secs)
        kw = request_kwargs(self.cfg.model, build_messages(wav))
        for k in self._drop:
            kw.pop(k, None)
        t0 = time.perf_counter()
        try:
            resp = await self.client.chat.completions.create(**kw)
        except Exception as e:
            bad = next((k for k in ("temperature", "max_tokens") if k in kw and k in str(e)), None)
            if bad is None:
                raise
            self.log(f"emotion: {self.cfg.model} rejected {bad}; retrying without it")
            self._drop.add(bad)
            kw.pop(bad, None)
            resp = await self.client.chat.completions.create(**kw)
        ms = (time.perf_counter() - t0) * 1000.0
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        at = getattr(details, "audio_tokens", None)
        if at is None:
            at = min(pt, int(len(wav) / (SR * 2) * AUDIO_TOKENS_PER_SEC))
        rin, rout = LLM_RATES.get(self.cfg.model, LLM_RATES.get("gpt-audio-mini", (0.6, 2.4)))
        ain = AUDIO_IN_RATES.get(self.cfg.model, AUDIO_IN_RATES.get("gpt-audio-mini", 10.0))
        cost = (max(pt - at, 0) * rin + at * ain + ct * rout) / 1e6
        metrics.record("llm", "emotion", ms, tokens=pt + ct, cost=cost)
        if self.limits is not None:
            try:
                await self.limits.record("emotion", 1)
            except Exception as e:
                self.log(f"emotion: limits record failed: {e!r}")
        text = resp.choices[0].message.content if resp.choices else None
        out = parse_emotion(text)
        if out is None:
            self.log(f"emotion: unusable reply {str(text)[:120]!r}")
            return None
        out["model"] = self.cfg.model
        out["ms"] = round(ms)
        self.log(f"emotion: {describe(out) or 'nothing'} v={out['valence']} a={out['arousal']} "
                 f"({ms:.0f} ms, ${cost:.5f})")
        return out
