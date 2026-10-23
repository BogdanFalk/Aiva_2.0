"""Backchannels: a soft "mhm" / "yeah" / "right" in her own voice at his clause-complete pauses
while he is still talking, with a small nod. The cheapest strong "she's listening" cue.

The HOUSE hears the pause; the SISTER makes the sound. Neither side runs an LLM, takes the floor or
adds anything to a context.

House side (`Backchanneler`, driven by Deepgram Flux's TurnInfo events — `HouseFluxSTTService`
forwards every StartOfTurn / Update / EndOfTurn with its partial transcript and
`end_of_turn_confidence`):
  - a PAUSE is the partial transcript not growing for `pause_s` while Flux still holds the turn open
    (no second VAD: Flux's ~4 Hz Updates and its own turn state are the signal). Each pause is
    judged once.
  - the pure `decide()` then says yes only when: he has been talking ≥ `min_speech_s` this turn,
    ≥ `cooldown_s` since the last one, Flux is not about to close the turn (eot below a ceiling a
    bit under the house's end-of-turn threshold — he is done, so she will answer instead), the line
    so far is not a finished question, does not end on a dangling word or filler (a word-search
    pause: "and the, um…"), and a `probability` roll passes. The house adds the context gates
    (asleep / unverified wake / mic muted / a sister holds or is about to hold the floor / no target).
  - ONE sister gets it: the sister his partial names, else the sticky addressee, else the last
    sister who spoke to him (`target()`).
  - The message is `house_proto.Backchannel` — not a floor grant, never runs her LLM.

Self-hearing: the clip is NOT gated. He is mid-turn, and the gate zero-fills the mic, which would
cut his words for the clip's length plus the cooldown. The clips are short (~0.3-0.6 s) and played
quieter than her voice; whatever echo Flux still transcribes is removed from the FINAL transcript by
`strip_echo()`: the house remembers how many words the partial had when each clip fired, and drops
the first matching backchannel token run found right at that spot (one word of slack either side).
Nothing else in his line is touched.

Sister side: `ClipCache` synthesises the fixed phrase set ONCE through her ElevenLabs voice
(her voice id + TTS model; files keyed by a hash of both, so a new voice regenerates) into
`aiva/voice_cache/<slug>/`, lazily in the background at boot; `house_link.BackchannelPlayer` plays a
clip into her output, and the avatar gets a `nod` cue. No voice / clip not cached yet → just the nod.

Everything here is stdlib-only and pure except `ClipCache.ensure` (network, injected `post`).
"""

import hashlib
import io
import os
import re
import wave
from dataclasses import dataclass, field
from typing import Optional

# --- the phrase set ---------------------------------------------------------------------------------

# clip id → what her TTS says. Real backchannels are a tiny closed set of sounds; assessments
# ("absolutely", "makes sense") arrive at turn ends, and mid-sentence they sound like answering.
PHRASES = {
    "mhm": "Mhm.",
    "mm_hmm": "Mm-hmm.",
    "uh_huh": "Uh-huh.",
    "yeah": "Yeah.",
    "right": "Right.",
}
CONTINUERS = ("mhm", "mm_hmm", "uh_huh")
AFFIRMS = ("yeah", "right")

SAMPLE_RATE = 24000          # ElevenLabs pcm_24000: raw s16le mono
MAX_CLIP_S = 1.2             # a clip longer than this is cut (with a short fade) — it would talk over him

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.path.join(_DIR, "voice_cache")


def _env_float(env, key, default, lo, hi):
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return v if lo <= v <= hi else default


@dataclass(frozen=True)
class BackchannelConfig:
    enabled: bool = True
    cooldown_s: float = 6.0          # at least this long between two
    probability: float = 0.5         # chance at an otherwise good pause (humans skip some)
    min_speech_s: float = 3.0        # he has been talking this long in this turn
    pause_s: float = 0.6             # the partial transcript has not grown for this long
    volume: float = 0.6              # sister side: linear gain on the clip

    @classmethod
    def from_env(cls, env=None) -> "BackchannelConfig":
        env = os.environ if env is None else env
        return cls(
            enabled=(env.get("AIVA_BACKCHANNEL", "1") or "1").strip().lower() in ("1", "true", "yes", "on"),
            cooldown_s=_env_float(env, "AIVA_BACKCHANNEL_COOLDOWN_S", 6.0, 0.0, 600.0),
            probability=_env_float(env, "AIVA_BACKCHANNEL_PROBABILITY", 0.5, 0.0, 1.0),
            min_speech_s=_env_float(env, "AIVA_BACKCHANNEL_MIN_SPEECH_S", 3.0, 0.0, 60.0),
            pause_s=_env_float(env, "AIVA_BACKCHANNEL_PAUSE_S", 0.6, 0.2, 5.0),
            volume=_env_float(env, "AIVA_BACKCHANNEL_VOLUME", 0.6, 0.0, 1.5),
        )


# --- text heuristics --------------------------------------------------------------------------------

_FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "mm", "like"}
# a clause can't end on these — a pause after one is him searching for the next word
_DANGLING = _FILLERS | {
    "a", "an", "the", "and", "or", "but", "so", "because", "cause", "to", "of", "in", "on", "at",
    "for", "with", "from", "by", "about", "into", "than", "that", "which", "who", "if", "when",
    "while", "my", "your", "his", "her", "our", "their", "its", "this", "these", "those", "is",
    "are", "was", "were", "be", "am", "i", "you", "we", "they", "he", "she", "it", "i'm", "it's",
    "not", "very", "really", "just", "then", "also", "some", "any", "no", "can", "could", "would",
    "should", "will", "do", "does", "did", "have", "has", "had", "gonna", "wanna", "kind", "sort",
    "mean", "know",
}
_QUESTION_STARTS = ("what", "why", "how", "when", "where", "who", "which", "whose", "can you",
                    "could you", "would you", "will you", "do you", "did you", "are you", "is it",
                    "is there", "have you", "should i", "can i", "could i", "shall we", "do we")


def words(text) -> list:
    """Lowercase word tokens; hyphens split ('mm-hmm' → mm, hmm), punctuation dropped."""
    out = []
    for w in re.split(r"[\s\-–—]+", (text or "").lower()):
        w = re.sub(r"[^a-z0-9']", "", w).strip("'")
        if w:
            out.append(w)
    return out


def ends_dangling(text) -> bool:
    ws = words(text)
    return bool(ws) and ws[-1] in _DANGLING


def is_finished_question(text) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if t.rstrip("\"'”’)]* ").endswith("?"):
        return True
    # Flux partials may carry no punctuation: a line that opens like a question is one, here
    return t.startswith(_QUESTION_STARTS)


def is_hesitant(text) -> bool:
    return any(w in _FILLERS for w in words(text)[-4:])


# --- the pure decision ---------------------------------------------------------------------------

def decide(cfg: BackchannelConfig, *, in_turn: bool, speech_s: float, pause_s: float, text: str,
           eot: Optional[float], eot_ceiling: float, since_last_s: float, blocked: Optional[str],
           roll: float):
    """Fire a backchannel at this pause? → (bool, reason). PURE: every input is given.
    `blocked` is the house's context gate (None = clear, else why not). `roll` is uniform 0..1."""
    if not cfg.enabled:
        return False, "disabled"
    if not in_turn:
        return False, "no turn"
    if blocked:
        return False, blocked
    if speech_s < cfg.min_speech_s:
        return False, "too early in his turn"
    if since_last_s < cfg.cooldown_s:
        return False, "cooldown"
    if pause_s < cfg.pause_s:
        return False, "no pause yet"
    if eot is not None and eot >= eot_ceiling:
        return False, "he's finishing"
    if len(words(text)) < 4:
        return False, "too few words"
    if is_finished_question(text):
        return False, "a question"
    if ends_dangling(text):
        return False, "mid-phrase"
    if roll >= cfg.probability:
        return False, "dice"
    return True, "pause"


def choose_clip(text, recent=(), roll=0.0, pick=0.0, available=None) -> str:
    """Which sound. A plain statement sometimes gets "yeah"/"right" (`roll` < 0.35), hesitant speech
    and everything else a continuer. Never the same clip twice in a row. `pick` (0..1) chooses
    within the group. Pure."""
    group = AFFIRMS if (roll < 0.35 and not is_hesitant(text)) else CONTINUERS
    pool = [c for c in group if available is None or c in available] or list(group)
    last = recent[-1] if recent else None
    fresh = [c for c in pool if c != last] or pool
    return fresh[min(len(fresh) - 1, int(max(0.0, min(0.999, pick)) * len(fresh)))]


def target(named, last_addressee, last_addressee_age_s, last_speaker, present, sticky_s):
    """The ONE sister who backchannels: the sister his words so far name (exactly one), else the
    sticky addressee (a single sister, recent), else the last sister who spoke, else the only one
    here. None when nobody fits. Pure."""
    present = list(present or [])
    named = [s for s in (named or ()) if s in present]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        return None                                   # "Aiva, Saga…" — not one listener
    if (last_addressee in present and last_addressee_age_s is not None
            and last_addressee_age_s <= sticky_s):
        return last_addressee
    if last_speaker in present:
        return last_speaker
    if len(present) == 1:
        return present[0]
    return None


# --- echo removal ---------------------------------------------------------------------------------

_HUMS = {"mhm", "mmhm", "mmhmm", "mm", "hmm", "hm", "mmm", "uh", "huh", "uhhuh", "uhuh", "um", "hum"}
_ECHO = {
    "yeah": {"yeah", "yea", "ya", "yep", "yes"},
    "right": {"right", "alright"},
}


def strip_echo(text, fired) -> str:
    """Remove her own backchannels from his final transcript. `fired` = [(word_index, clip_id)] —
    how many words the partial had when each clip fired. For each, the first run of 1-2 matching
    tokens starting within one word of that index is dropped. Only those spots are touched."""
    if not fired or not text:
        return text
    toks = (text or "").split()
    norm = [words(t) for t in toks]                   # one raw token can hold 'mm-hmm' → 2 words
    flat = []                                         # (raw token index, word)
    for i, ws in enumerate(norm):
        for w in ws:
            flat.append((i, w))
    drop = set()
    for idx, clip in sorted(fired):
        want = _ECHO.get(clip, _HUMS)
        idx = max(0, int(idx))
        for start in range(max(0, idx - 1), min(len(flat), idx + 2)):
            if flat[start][1] in want and flat[start][0] not in drop:
                n = 1
                if clip not in _ECHO and start + 1 < len(flat) and flat[start + 1][1] in _HUMS:
                    n = 2
                for k in range(start, start + n):
                    drop.add(flat[k][0])
                break
    if not drop:
        return text
    kept = []
    for i, t in enumerate(toks):
        if i not in drop:
            kept.append(t)
            continue
        end = re.search(r"[.!?]+$", t)          # "home Mhm. And" → "home. And": keep the full stop
        if end and kept and not re.search(r"[.!?,;:]$", kept[-1]):
            kept[-1] += end.group(0)
    out = " ".join(kept).strip()
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    out = re.sub(r"^[,.;:]\s*", "", out)
    return out


# --- the house's per-turn tracker --------------------------------------------------------------

@dataclass
class _Turn:
    started_at: float
    text: str = ""
    n_words: int = 0
    grew_at: float = 0.0
    eot: Optional[float] = None
    judged: bool = False                    # this pause has had its one decision
    fired: list = field(default_factory=list)


class Backchanneler:
    """Stateful wrapper around `decide()`, fed by Flux events and a tick. Clock and dice injected
    (`now` passed to every call, `rand()` → 0..1). Knows nothing about sockets or pipecat."""

    ECHO_TTL_S = 15.0

    def __init__(self, cfg: BackchannelConfig, *, eot_ceiling: float = 0.6, rand=None, log=print):
        import random
        self.cfg = cfg
        self.eot_ceiling = float(eot_ceiling)
        self._rand = rand or random.random
        self._log = log
        self._turn: Optional[_Turn] = None
        self._last_fired_at = -1e9
        self._recent = []
        self._echo = []                     # (expires_at, fired, first words) of ended turns

    @property
    def in_turn(self) -> bool:
        return self._turn is not None

    @property
    def partial(self) -> str:
        return self._turn.text if self._turn else ""

    def observe(self, event, transcript, eot, now):
        """One Flux TurnInfo: StartOfTurn | Update | EndOfTurn (others ignored)."""
        if event == "StartOfTurn":
            self._turn = _Turn(started_at=now, grew_at=now)
            self._grow(transcript, now)
        elif event == "Update":
            if self._turn is None:              # an update with no start seen: the turn is on
                self._turn = _Turn(started_at=now, grew_at=now)
            self._grow(transcript, now)
            if eot is not None:
                try:
                    self._turn.eot = float(eot)
                except (TypeError, ValueError):
                    pass
        elif event == "EndOfTurn":
            self.end_turn(now)

    def _grow(self, transcript, now):
        t = self._turn
        text = (transcript or "").strip()
        if text and text != t.text:             # new words (or a revision): he is still going
            t.grew_at = now
            t.judged = False
            t.text = text
            t.n_words = len(words(text))

    def end_turn(self, now):
        t, self._turn = self._turn, None
        self._echo = [e for e in self._echo if e[0] > now]
        if t is not None and t.fired:
            self._echo.append((now + self.ECHO_TTL_S, list(t.fired), words(t.text)[:3]))

    def tick(self, now, blocked=None):
        """Once per ~100 ms. → (clip, reason, info) when a backchannel should fire now, else None.
        `blocked` is evaluated lazily: a callable → reason|None, consulted only at a pause."""
        t = self._turn
        if t is None or t.judged or not self.cfg.enabled:
            return None
        pause = now - t.grew_at
        if pause < self.cfg.pause_s:
            return None
        t.judged = True
        why_not = blocked() if callable(blocked) else blocked
        ok, reason = decide(self.cfg, in_turn=True, speech_s=now - t.started_at, pause_s=pause,
                            text=t.text, eot=t.eot, eot_ceiling=self.eot_ceiling,
                            since_last_s=now - self._last_fired_at, blocked=why_not,
                            roll=self._rand())
        info = {"pause_s": round(pause, 2), "speech_s": round(now - t.started_at, 1),
                "eot": None if t.eot is None else round(t.eot, 2), "words": t.n_words}
        if not ok:
            return (None, reason, info)
        clip = choose_clip(t.text, self._recent, roll=self._rand(), pick=self._rand())
        self._last_fired_at = now
        self._recent = (self._recent + [clip])[-3:]
        t.fired.append((t.n_words, clip))
        return (clip, reason, info)

    def undo_fire(self):
        """The send failed (she left): don't count it for cooldown or echo."""
        t = self._turn
        if t is not None and t.fired:
            t.fired.pop()
        self._last_fired_at = -1e9

    def strip_final(self, text, now):
        """His final transcript minus her own echo (from turns that fired, freshest first)."""
        self._echo = [e for e in self._echo if e[0] > now]
        head = words(text)[:3]
        for i, (_exp, fired, turn_head) in enumerate(self._echo):
            if turn_head == head:               # the same turn (its final opens like its partial did)
                del self._echo[i]
                return strip_echo(text, fired)
        return text


# --- the sister's clip cache ----------------------------------------------------------------------

def clip_key(clip, voice_id, model) -> str:
    h = hashlib.sha1(f"{voice_id}|{model}|{PHRASES.get(clip, clip)}".encode("utf-8")).hexdigest()[:10]
    return f"{clip}-{h}.wav"


def trim_silence(pcm: bytes, sr=SAMPLE_RATE, threshold=500, pad_s=0.03, max_s=MAX_CLIP_S) -> bytes:
    """Cut leading/trailing near-silence (keeping a short pad) and cap the length with a 40 ms fade."""
    import array
    a = array.array("h")
    a.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not a:
        return b""
    idx = [i for i in range(0, len(a), 48) if abs(a[i]) >= threshold]
    if not idx:
        return b""
    pad = int(pad_s * sr)
    lo, hi = max(0, idx[0] - pad), min(len(a), idx[-1] + 48 + pad)
    a = a[lo:hi]
    cap = int(max_s * sr)
    if len(a) > cap:
        a = a[:cap]
        fade = min(len(a), int(0.04 * sr))
        for k in range(fade):
            j = len(a) - fade + k
            a[j] = int(a[j] * (1 - (k + 1) / fade))
    return a.tobytes()


def scale_pcm(pcm: bytes, volume: float) -> bytes:
    """Mono s16 gain, saturating (an overflow wraps into a click)."""
    if volume == 1.0 or not pcm:
        return pcm
    import array
    a = array.array("h")
    a.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    for i, s in enumerate(a):
        a[i] = max(-32768, min(32767, int(s * volume)))
    return a.tobytes()


def write_wav(path, pcm, sr=SAMPLE_RATE):
    tmp = path + ".tmp"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    os.replace(tmp, path)


def read_wav(path):
    """→ (pcm, sample_rate) of a mono s16 WAV, or None."""
    try:
        with wave.open(path, "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                return None
            return w.readframes(w.getnframes()), w.getframerate()
    except (OSError, wave.Error, EOFError):
        return None


def _requests_post(url, **kw):
    import requests
    return requests.post(url, **kw)


class ClipCache:
    """Her backchannel clips, synthesised once through her ElevenLabs voice and kept as WAVs.
    `ensure()` is blocking (run it under `asyncio.to_thread`); `get()` is cheap after the first load."""

    URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_24000"

    def __init__(self, slug, voice_id, model, api_key, *, root=None, post=None, log=print):
        self.dir = os.path.join(root or CACHE_ROOT, slug)
        self.voice_id = voice_id
        self.model = model
        self.api_key = api_key
        self._post = post or _requests_post
        self._log = log
        self._mem = {}

    @property
    def can_synth(self) -> bool:
        return bool(self.voice_id and self.api_key)

    def path(self, clip) -> str:
        return os.path.join(self.dir, clip_key(clip, self.voice_id, self.model))

    def missing(self) -> list:
        return [c for c in PHRASES if not os.path.isfile(self.path(c))]

    def ensure(self, clips=None) -> list:
        """Synthesise every missing clip (blocking). → the clip ids made (their characters are what
        the caller counts against her TTS budget, on the loop thread). Never raises."""
        if not self.can_synth:
            return []
        made = []
        for clip in (clips or self.missing()):
            if os.path.isfile(self.path(clip)):
                continue
            text = PHRASES.get(clip)
            if not text:
                continue
            try:
                os.makedirs(self.dir, exist_ok=True)
                r = self._post(self.URL.format(voice=self.voice_id),
                               headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
                               json={"text": text, "model_id": self.model}, timeout=20)
                if getattr(r, "status_code", 200) != 200:
                    self._log(f"backchannel: clip {clip!r} synthesis failed (HTTP {r.status_code})")
                    continue
                pcm = trim_silence(r.content or b"")
                if len(pcm) < int(0.08 * SAMPLE_RATE) * 2:
                    self._log(f"backchannel: clip {clip!r} came back silent — skipped")
                    continue
                write_wav(self.path(clip), pcm)
                made.append(clip)
            except Exception as e:
                self._log(f"backchannel: clip {clip!r} synthesis error: {e!r}")
        if made:
            self._log(f"backchannel: {len(made)} clip(s) cached in {self.dir}")
        return made

    def available(self) -> set:
        return {c for c in PHRASES if c in self._mem or os.path.isfile(self.path(c))}

    def get(self, clip):
        """→ (pcm, sample_rate) or None when the clip isn't cached (yet)."""
        if clip in self._mem:
            return self._mem[clip]
        got = read_wav(self.path(clip)) if clip in PHRASES else None
        if got is not None and got[0]:
            self._mem[clip] = got
            return got
        return None


# --- the sister's side of a backchannel -------------------------------------------------------------

class BackchannelVoice:
    """What a sister does with `proto.Backchannel`: nod, and — when she has a voice, it isn't muted
    and the clip is cached — play it quietly through `player` (`house_link.BackchannelPlayer`).
    Dependencies injected; `busy()` says she is in a turn or speaking (then: nothing at all)."""

    def __init__(self, cfg: BackchannelConfig, *, cache=None, player=None, avatar=None,
                 busy=None, log=print):
        self.cfg = cfg
        self.cache = cache
        self.player = player
        self.avatar = avatar
        self._busy = busy or (lambda: False)
        self._log = log

    async def handle(self, clip, voice=True) -> str:
        """→ what she did: off | busy | nod | voice."""
        if not self.cfg.enabled:
            return "off"
        if self._busy():
            return "busy"
        if self.avatar is not None:
            try:
                self.avatar.send_cue("nod")
            except Exception:
                pass
        if not voice or self.player is None or self.cache is None:
            return "nod"
        got = self.cache.get(clip)
        if got is None:
            return "nod"                        # not cached yet (or no voice): the nod is enough
        pcm, rate = got
        try:
            ok = await self.player.play(scale_pcm(pcm, self.cfg.volume), rate)
        except Exception as e:
            self._log(f"backchannel: play failed: {e!r}")
            ok = False
        return "voice" if ok else "nod"
