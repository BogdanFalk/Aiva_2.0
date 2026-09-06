"""Aiva — streaming voice pipeline (pipecat 1.x).

mic -> Deepgram streaming STT -> Silero VAD + local smart-turn (in the user
aggregator) -> LLM (native tool calling, streamed) -> ElevenLabs Flash TTS
-> speaker.

Hotkeys: F9 mute mic, F10 interrupt Aiva, Esc quit.

Optional ambient mode (AIVA_WAKE_WORD=1): the mic sleeps until the wake word
(openWakeWord, local) and dozes off again after AIVA_IDLE_TIMEOUT seconds of
silence. While asleep no audio is sent to Deepgram, so idle time is free.

Run `python main.py --list-devices` to see audio device indices for
AIVA_INPUT_DEVICE_INDEX / AIVA_OUTPUT_DEVICE_INDEX.
"""

import asyncio
import math
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime

from dotenv import load_dotenv

# .env is THE config source (CLAUDE.md): override stale values inherited from the
# launching shell — otherwise a model changed in .env silently stays on the old one
load_dotenv(override=True)

# hotkey/expression names can be CJK; don't let a console print crash a handler
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# pipecat logs "Empty audio frame received for STT service" on EVERY zero/empty
# audio frame — a flood while she's awake. Throttle just that message to once
# per 10 s; everything else logs normally (loguru's default stderr handler,
# same format).
def _install_log_throttle():
    import time as _t
    from loguru import logger as _logger
    _state = {"last": 0.0, "dropped": 0}

    def _filter(record):
        if "Empty audio frame received" in record["message"]:
            now = _t.monotonic()
            if now - _state["last"] < 10:
                _state["dropped"] += 1
                return False
            if _state["dropped"]:
                record["message"] += f"  (+{_state['dropped']} more in the last 10s)"
                _state["dropped"] = 0
            _state["last"] = now
        return True

    try:
        _logger.remove()
    except ValueError:
        pass
    _logger.add(sys.stderr, filter=_filter, level="DEBUG")


_install_log_throttle()

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    InterruptionWorkerFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from modules.avatar import AvatarClient
from modules.cat import CAT_NAME_RECOVERY_NOTE, CAT_NAMING_NOTE, DesktopCat
from modules import capabilities, computer, file_operations
from modules.memory import Memory
from modules.inner import ExperienceBus, InnerBrain, InnerState, SelfEngine
from modules.persona import baseline_traits, build_system_prompt
from modules.limits import ServiceLimits
from modules.notify import Notifier, NotifyConfig
from modules.paralinguistics import AffectConfig, VocalAffect
from modules.preferences import Preferences
from modules.space import SpaceMemory
from modules.speakers import SpeakerConfig, SpeakerMemory, SpeakerRecognizer
from modules.web import WebClient, WebConfig
from modules.task_memory import TaskMemory, embed_text
from modules.tasks import TaskRunner, ToolBox, openai_llm_call
from modules.terminals import TerminalManager
from modules.tools import (TOOL_SCHEMAS, execute_shell, register_becoming_tools,
                           register_preference_tools, register_reading_tools,
                           register_self_tools, register_speaker_tools, register_task_tools,
                           register_tools, register_web_tools)
from modules.reading import BookMemory, ReadingEngine, StudyLimits, embed_long
from modules.becoming import Becoming
from modules.utilities import LLM_RATES as _LLM_RATES


_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def log_event(msg: str):
    """Append a timestamped line to today's event log (aiva/logs/).

    The console has no scrollback once its window closes — this file is what
    'check the logs' reads when something odd happened hours ago: every
    wake/sleep transition (with its reason), wake-word detections and their
    scores, what she heard, and every self-triggered announce."""
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        path = os.path.join(_LOG_DIR, f"aiva-{datetime.now():%Y%m%d}.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%H:%M:%S} {msg}\n")
    except OSError:
        pass  # logging must never take her down


class PatientOpenAILLMService(OpenAILLMService):
    """OpenAILLMService whose client retries 429/5xx with exponential backoff
    (honouring retry-after). pipecat builds its client with the SDK default of
    2 fast retries — on a 30k tokens-per-minute tier that turned a transient
    rate limit into a silent, dead turn. AIVA_LLM_MAX_RETRIES tunes it."""

    def create_client(self, api_key=None, base_url=None, organization=None, project=None,
                      default_headers=None, **kwargs):
        client = super().create_client(api_key=api_key, base_url=base_url, organization=organization,
                                       project=project, default_headers=default_headers, **kwargs)
        return client.with_options(max_retries=int(os.getenv("AIVA_LLM_MAX_RETRIES", "5")))


def trim_context(context, memory=None, budget_chars=None, keep_min=8):
    """Keep the live context under a size budget.

    Every voice turn re-sends the WHOLE context (plus ~5.5k tokens of tool
    schemas), so an unbounded history is what eventually walks into the
    tokens-per-minute limit. Drops the oldest conversation messages — never
    the leading system block, never an in-flight tool exchange, and always
    keeps the last `keep_min` messages. Tells `memory` how many already-saved
    spoken messages went, so transcript persistence stays consistent.
    Returns the number of messages dropped."""
    budget = budget_chars or int(os.getenv("AIVA_CONTEXT_BUDGET_CHARS", "24000"))
    msgs = list(context.get_messages())
    lead = 0
    while lead < len(msgs) and msgs[lead].get("role") == "system":
        lead += 1
    head, body = msgs[:lead], msgs[lead:]

    def size(m):
        c = m.get("content")
        return len(c) if isinstance(c, str) else len(json.dumps(c, ensure_ascii=False, default=str))             if c is not None else 40

    total = sum(size(m) for m in body)
    dropped, spoken = 0, 0
    # trim while over budget, then keep going to the next user message so the
    # surviving history starts at a turn boundary
    while len(body) > keep_min and (total > budget or body[0].get("role") != "user"):
        m = body[0]
        if m.get("role") == "tool" and m.get("content") == "IN_PROGRESS":
            break  # a result is still on its way
        pending_ids = {tc.get("id") for tc in (m.get("tool_calls") or []) if isinstance(tc, dict)}
        if pending_ids and any(x.get("role") == "tool" and x.get("tool_call_id") in pending_ids
                               and x.get("content") == "IN_PROGRESS" for x in body):
            break
        body.pop(0)
        total -= size(m)
        dropped += 1
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            spoken += 1
        # results of a dropped tool call go with it
        while pending_ids and body and body[0].get("role") == "tool"                 and body[0].get("tool_call_id") in pending_ids:
            t = body.pop(0)
            total -= size(t)
            dropped += 1
    if dropped:
        context.set_messages(head + body)
        if memory is not None and spoken:
            memory.forget_saved(spoken)
    return dropped


def drop_intro_note(messages, intro_text):
    """Remove the boot 'introduce yourself' system note. It's a ONE-SHOT: it lives in the leading
    system block, which trim_context never drops, so left in place it keeps telling her to greet
    on every wake. Returns (messages, removed)."""
    kept = [m for m in messages
            if not (m.get("role") == "system" and m.get("content") == intro_text)]
    return kept, len(kept) != len(messages)


class MicController(FrameProcessor):
    """Gates mic audio right after the transport input.

    Three layers, all applied BEFORE audio reaches STT:
    - asleep (ambient mode): audio frames are DROPPED entirely — Deepgram
      receives nothing and bills nothing (its KeepAlive holds the socket)
    - manually muted (F9) or Aiva speaking (+ cooldown for the room's audio
      tail): audio is zero-filled, so her own voice is never transcribed and
      delayed self-transcripts can't come back as user input
    - awake and quiet: audio passes, and billed STT seconds are counted
    """

    def __init__(self, ambient: bool, bot_cooldown_secs: float = 0.5,
                 verify_secs: float = 5.0):
        super().__init__()
        self.muted = False
        self.awake = not ambient
        self.last_activity = time.monotonic()
        self.last_user_final = None   # set on the user's final transcript; for turn-latency metric
        self.stt_seconds = 0.0
        self._ambient = ambient
        self._bot_cooldown_secs = bot_cooldown_secs
        # Verified wake: the wake-word model alone is too trigger-happy (it
        # fires on TV/game speech with the same scores as a real "hey Aiva"),
        # so a wake-word wake is PROVISIONAL until Deepgram's transcript of
        # the pre-roll audio actually contains her name. Until then she is
        # awake only in the sense that audio streams to STT: no [AWAKE] print,
        # eyes stay shut, and nothing reaches the LLM (ActivityWatcher gates
        # transcripts). No name within `verify_secs` -> silently back to sleep.
        self._verify_secs = verify_secs
        self._verifying = False
        self._verify_deadline = 0.0
        self._preroll = b""          # 16 kHz PCM injected ahead of live audio
        self._bot_speaking = False
        self._bot_stopped_at = 0.0
        self._thinking_since = 0.0    # monotonic; 0 = no LLM response in flight
        self._tools_in_flight = {}    # tool_call_id -> monotonic start time
        # The transport must capture at the mic's NATIVE rate (Windows
        # delivers silence/garbage otherwise), but Silero VAD and friends
        # require 16 kHz — so this processor resamples the stream and
        # re-stamps the StartFrame before anything downstream sees it.
        self._resampler = SOXRStreamAudioResampler()
        self._capture_rate = None

    def toggle_mute(self):
        self.muted = not self.muted
        state = "MUTED — deafened, won't wake to your voice" if self.muted else "LIVE"
        print(f"\n[MIC {state}] (F9 to toggle)")
        log_event(f"mic {'muted (deafened)' if self.muted else 'unmuted'} (F9)")

    def wake(self, reason: str = "unspecified", preroll: bytes | None = None,
             verify: bool = False):
        """Wake her. `preroll` (16 kHz int16 PCM captured just before the
        wake word fired) is pushed to STT ahead of live audio so the
        transcript includes the wake phrase itself. With `verify` the wake is
        provisional until confirm_wake()/reject_wake() decide."""
        # F9 deafen: a wake-word wake carries a preroll (her own announces don't) — while muted
        # the wake word can't wake her, so she truly can't hear you; she can still speak up herself.
        if self.muted and (verify or preroll is not None):
            log_event(f"wake ignored — mic muted (F9): {reason}")
            return
        self.last_activity = time.monotonic()
        if preroll:
            self._preroll = preroll
        if not self.awake:
            self.awake = True
            if verify and self._ambient:
                self._verifying = True
                self._verify_deadline = time.monotonic() + self._verify_secs
                print("\n[WAKE?] heard something like the wake word — checking")
                log_event(f"wake candidate — {reason}, verifying transcript")
            else:
                print("\n[AWAKE] Aiva is listening")
                log_event(f"AWAKE — {reason}")
        elif self._verifying and not verify:
            # an authoritative wake (background announce) arrived mid-check
            self.confirm_wake(reason)

    def confirm_wake(self, reason: str = ""):
        if self._verifying:
            self._verifying = False
            self.last_activity = time.monotonic()
            print("\n[AWAKE] Aiva is listening")
            log_event(f"AWAKE — wake word confirmed ({reason})")

    def reject_wake(self, reason: str = ""):
        if self._verifying:
            self._verifying = False
            self._preroll = b""
            self.awake = False
            print("\n[IDLE] false wake ignored — say the wake word to talk to Aiva")
            log_event(f"wake rejected — {reason}")

    @property
    def verifying(self) -> bool:
        return self._verifying

    def verify_expired(self) -> bool:
        return self._verifying and time.monotonic() > self._verify_deadline

    @property
    def visibly_awake(self) -> bool:
        """Awake as far as the user (and her avatar) should be able to tell —
        a provisional wake still being verified doesn't count."""
        return self.awake and not self._verifying

    def poke(self):
        """Conversation activity happened — keep her awake."""
        self.last_activity = time.monotonic()

    def sleep(self):
        if self.awake and self._ambient:
            self.awake = False
            print("\n[IDLE] say the wake word to talk to Aiva")
            log_event("asleep (idle timeout)")

    def set_thinking(self, on: bool):
        self._thinking_since = time.monotonic() if on else 0.0
        self.last_activity = time.monotonic()

    def tool_started(self, call_id):
        self._tools_in_flight[call_id] = time.monotonic()
        self.last_activity = time.monotonic()

    def tool_finished(self, call_id):
        self._tools_in_flight.pop(call_id, None)
        self.last_activity = time.monotonic()

    @property
    def processing(self) -> bool:
        """True while she's still working on a turn — generating a response or
        waiting on tool calls — so the idle timer can't doze her off mid-task.

        Both signals expire (a dropped end/result frame must not pin her awake
        forever: awake means the mic streams to Deepgram and bills).
        """
        now = time.monotonic()
        self._tools_in_flight = {
            k: t for k, t in self._tools_in_flight.items() if now - t < 600
        }
        thinking = self._thinking_since and now - self._thinking_since < 120
        return bool(thinking) or bool(self._tools_in_flight)

    @property
    def bot_speaking(self) -> bool:
        return self._bot_speaking

    @property
    def ambient(self) -> bool:
        return self._ambient

    def idle_for(self) -> float:
        return time.monotonic() - self.last_activity

    def _zeroed(self) -> bool:
        if self.muted or self._bot_speaking:
            return True
        return (time.monotonic() - self._bot_stopped_at) < self._bot_cooldown_secs

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._capture_rate = frame.audio_in_sample_rate
            if self._capture_rate != 16000:
                frame.audio_in_sample_rate = 16000  # what downstream will get

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self.last_activity = time.monotonic()
            if self.last_user_final is not None:   # felt latency: you stop -> she speaks
                from modules import metrics
                metrics.record("turn", "latency",
                               (time.monotonic() - self.last_user_final) * 1000.0)
                self.last_user_final = None
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped_at = time.monotonic()
            self.last_activity = time.monotonic()

        if isinstance(frame, InputAudioRawFrame):
            if not self.awake:
                return  # dropped: nothing streams to Deepgram while asleep
            if self._preroll:
                # the seconds BEFORE the wake fired (the wake phrase itself),
                # already 16 kHz — hand them to STT ahead of the live stream in
                # normal-sized chunks so VAD downstream sees nothing unusual
                preroll, self._preroll = self._preroll, b""
                step = 2 * 16000 * 40 // 1000  # 40 ms of int16 mono
                for i in range(0, len(preroll), step):
                    await self.push_frame(InputAudioRawFrame(
                        audio=preroll[i:i + step], sample_rate=16000,
                        num_channels=1), direction)
                self.stt_seconds += len(preroll) / (2 * 16000)
            audio = frame.audio
            # frame.sample_rate is the transport's TRUE capture rate (the
            # StartFrame only carries the pipeline's configured rate) —
            # resample by what the frame actually is.
            if frame.sample_rate != 16000:
                audio = await self._resampler.resample(audio, frame.sample_rate, 16000)
            if self._zeroed():
                audio = b"\x00" * len(audio)
            if os.getenv("AIVA_DEBUG_MIC") == "1":
                self._debug_meter(frame.audio, audio)
            frame = InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)
            self.stt_seconds += len(audio) / (2 * 16000)

        await self.push_frame(frame, direction)

    _dbg_acc = []

    def _debug_meter(self, raw, processed):
        import numpy as np

        self._dbg_acc.append((raw, processed))
        if sum(len(r) for r, _ in self._dbg_acc) >= self._capture_rate * 2 * 2:  # ~2s
            raw_all = np.frombuffer(b"".join(r for r, _ in self._dbg_acc), dtype=np.int16)
            out_all = np.frombuffer(b"".join(p for _, p in self._dbg_acc), dtype=np.int16)
            print(f"[MIC] in_rate={self._capture_rate} raw_rms={raw_all.std():.0f} "
                  f"resampled_rms={out_all.std():.0f} frames={len(self._dbg_acc)}")
            self._dbg_acc.clear()


# How Deepgram tends to spell her name (keyterm boosting helps, but not
# always). Extend without code changes via AIVA_NAME_VARIANTS (comma list).
_NAME_VARIANTS = ("aiva", "ava", "eva", "iva", "ayva", "eiva", "aeva", "aiwa",
                  "avah", "evah", "aivah", "hey va", "hey iva")


def mentions_name(text: str, variants=_NAME_VARIANTS) -> bool:
    """True if a transcript plausibly contains 'Aiva' — exact variants plus a
    loose vowel(s)+v/w/f+a shape ('eyva', 'ayvah') that no common English
    word has, so TV dialogue doesn't sneak through."""
    words = re.sub(r"[^a-z' ]", " ", text.lower()).split()
    joined = f" {' '.join(words)} "
    for v in variants:
        if (" " in v and f" {v} " in joined) or (" " not in v and v in words):
            return True
    return any(re.fullmatch(r"[aeiy]{1,3}[vwf]ah?", w) for w in words)


class ActivityWatcher(FrameProcessor):
    """Sits after STT: any transcription (interim or final) means the user is
    talking to Aiva, so the idle timer resets — including when they say her
    name mid-conversation.

    Also the second stage of the wake word: while a wake is being verified,
    the first final transcript decides — her name in it confirms the wake and
    passes through; anything else (TV dialogue, a cough) is dropped before the
    LLM sees it and she goes straight back to sleep. Late transcripts arriving
    after she fell asleep are dropped for the same reason."""

    def __init__(self, mic: "MicController", name_variants=_NAME_VARIANTS, on_user_final=None):
        super().__init__()
        self._mic = mic
        self._variants = tuple(name_variants)
        self._on_user_final = on_user_final    # async(text): identify the speaker for this turn

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._mic.verifying:
            if isinstance(frame, TranscriptionFrame):
                if mentions_name(frame.text, self._variants):
                    self._mic.confirm_wake(f"heard {frame.text!r}")
                else:
                    log_event(f"heard during wake check: {frame.text!r}")
                    self._mic.reject_wake(f"no name in {frame.text!r}")
                    return
            elif isinstance(frame, InterimTranscriptionFrame):
                return  # nothing downstream until the wake is confirmed
            elif self._mic.verify_expired():
                self._mic.reject_wake("no speech after the wake word")
        elif self._mic.ambient and not self._mic.awake and \
                isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            if isinstance(frame, TranscriptionFrame):
                log_event(f"heard while asleep (dropped): {frame.text!r}")
            return  # Deepgram flushing audio from before she dozed off
        if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            self._mic.poke()
        if isinstance(frame, TranscriptionFrame):
            log_event(f"heard: {frame.text!r}")
            self._mic.last_user_final = time.monotonic()   # start the turn-latency clock
            if self._on_user_final:                        # attach speaker identity to this turn
                try:
                    await self._on_user_final(frame.text)
                except Exception as e:
                    log_event(f"speaker id error: {e!r}")
        await self.push_frame(frame, direction)


class ProcessingWatcher(FrameProcessor):
    """Sits right after the LLM: marks Aiva as processing while a response is
    being generated or tool calls are still executing (long terminal commands,
    vision calls, ...), so the ambient idle timer waits until she's truly done."""

    def __init__(self, mic: "MicController"):
        super().__init__()
        self._mic = mic

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._mic.set_thinking(True)
        elif isinstance(frame, (LLMFullResponseEndFrame, InterruptionFrame)):
            # an interruption swallows the queued end frame — treat it as one
            self._mic.set_thinking(False)
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._mic.tool_started(frame.tool_call_id)
        elif isinstance(frame, (FunctionCallResultFrame, FunctionCallCancelFrame)):
            self._mic.tool_finished(frame.tool_call_id)
        await self.push_frame(frame, direction)


class UsageTracker(FrameProcessor):
    """Collects LLM token and TTS character usage from metrics frames."""

    def __init__(self):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tts_chars = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, MetricsFrame):
            for data in frame.data:
                if isinstance(data, LLMUsageMetricsData):
                    self.prompt_tokens += data.value.prompt_tokens or 0
                    self.completion_tokens += data.value.completion_tokens or 0
                elif isinstance(data, TTSUsageMetricsData):
                    self.tts_chars += data.value or 0
        await self.push_frame(frame, direction)


class LipSyncTap(FrameProcessor):
    """Sits between TTS and the output transport, feeding the avatar renderer's
    mouth. TTS audio frames pass here FASTER than real-time (the transport does
    the pacing), so each chunk's RMS goes over with its duration and the
    renderer queues them, consuming at wall-clock rate once playback actually
    starts (BotStartedSpeakingFrame — emitted by the transport, so it passes
    this processor travelling upstream). Interruptions flush the queue.

    Everything here is sync fire-and-forget into the avatar socket: a dead or
    missing renderer costs nothing and can never stall the voice pipeline.
    """

    def __init__(self, avatar):
        super().__init__()
        self._avatar = avatar

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and frame.audio:
            import numpy as np

            samples = np.frombuffer(frame.audio, dtype=np.int16)
            if len(samples):
                # One measurement per ~40ms window, not per frame: TTS frames
                # can be hundreds of ms, and a single average erases the
                # syllable pattern (the mouth would hang half-open, motionless).
                ch = max(1, frame.num_channels)
                floats = samples.astype(np.float64)
                if ch > 1:
                    floats = floats.reshape(-1, ch).mean(axis=1)
                sr = max(1, frame.sample_rate)
                win = max(1, int(sr * 0.04))
                chunks = []
                for i in range(0, len(floats), win):
                    seg = floats[i:i + win]
                    rms = float(np.sqrt(np.mean(seg ** 2))) / 32768.0
                    # brightness = spectral centroid mapped log-linearly onto
                    # 0..1: dark/rounded vowels (oo/oh, centroid ~500Hz) -> 0,
                    # bright/wide ones (ee/eh and sibilants, ~2.5kHz+) -> 1.
                    # The renderer turns it into ParamMouthForm (viseme shape).
                    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
                    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
                    band = (freqs >= 200) & (freqs <= 5000)
                    power = spec[band]
                    total = float(power.sum())
                    if total > 0:
                        centroid = float((power * freqs[band]).sum()) / total
                        bright = (math.log(max(centroid, 1.0)) - math.log(500)) \
                            / (math.log(2500) - math.log(500))
                        bright = max(0.0, min(1.0, bright))
                    else:
                        bright = 0.5
                    chunks.append([round(rms, 4), round(len(seg) / sr, 5),
                                   round(bright, 3)])
                self._avatar.send_mouth(chunks)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._avatar.send_speech("start")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._avatar.send_speech("stop")
        elif isinstance(frame, InterruptionFrame):
            self._avatar.send_speech("interrupt")
        await self.push_frame(frame, direction)


class UserAudioTap(FrameProcessor):
    """Buffers the user's 16 kHz PCM for the current turn so speaker recognition can embed it.
    Mirrors LipSyncTap: inspect, buffer, forward untouched. Sits right after MicController, so it
    sees frames with Aiva's own voice already zero-filled and asleep audio already dropped."""

    def __init__(self, mic: "MicController", max_secs: float = 12.0):
        super().__init__()
        self._mic = mic
        self._max_bytes = int(max_secs * 16000) * 2
        self._buf = bytearray()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._buf.clear()                          # new user turn after she speaks
        elif isinstance(frame, InputAudioRawFrame) and frame.audio and not self._mic.bot_speaking:
            import numpy as np
            if np.frombuffer(frame.audio, dtype=np.int16).any():   # skip zero-filled (muted/self)
                self._buf += frame.audio
                if len(self._buf) > self._max_bytes:
                    del self._buf[:-self._max_bytes]
        await self.push_frame(frame, direction)

    def take_segment(self) -> bytes:
        pcm = bytes(self._buf)
        self._buf.clear()
        return pcm


class WakeWordListener:
    """Local wake-word detection (openWakeWord) on its own mic stream.

    Runs in a thread; never sends audio anywhere. Fires on_wake(score,
    preroll) when the wake word scores above threshold for `patience`
    CONSECUTIVE 80ms frames — `preroll` is the last `preroll_secs` of 16 kHz
    PCM (the wake phrase itself), so STT can verify what was actually said. A real spoken "hey Aiva" stays in the model's sliding window for
    several frames, so it keeps scoring high; false triggers from game/TV
    audio are usually a single spiky frame — requiring consecutive frames
    filters them out where a higher threshold can't (false spikes overlap
    the score range of genuine quiet wakes).
    """

    def __init__(self, model, on_wake, device_index=None, threshold=0.5, patience=2,
                 preroll_secs=2.5):
        self._model = model
        self._on_wake = on_wake
        self._device_index = device_index
        self._threshold = threshold
        self._patience = max(1, patience)
        self._preroll_bytes = int(preroll_secs * 16000) * 2
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        from math import gcd

        import numpy as np
        import pyaudio
        from openwakeword.model import Model
        from scipy.signal import resample_poly

        oww = Model(wakeword_models=[self._model], inference_framework="onnx")
        pa = pyaudio.PyAudio()
        # Capture at the device's NATIVE rate and resample to the 16 kHz the
        # model expects — asking Windows for 16 kHz directly yields silence
        # (MME) or corrupted repeating buffers (DirectSound) on this machine.
        if self._device_index is None:
            info = pa.get_default_input_device_info()
        else:
            info = pa.get_device_info_by_index(self._device_index)
        native = int(info["defaultSampleRate"])
        frames = int(native * 0.08)  # 80 ms, the model's chunk size
        g = gcd(16000, native)
        up, down = 16000 // g, native // g

        stream = pa.open(
            rate=native, channels=1, format=pyaudio.paInt16, input=True,
            frames_per_buffer=frames, input_device_index=self._device_index,
        )
        print(f"Wake word armed ({self._model}) on '{info['name']}' @ {native} Hz "
              f"(threshold={self._threshold}, patience={self._patience} frames)")
        try:
            consecutive = 0
            peak = 0.0
            ring = deque()      # recent 16 kHz audio, newest last
            ring_bytes = 0
            while not self._stop.is_set():
                chunk = stream.read(frames, exception_on_overflow=False)
                audio = np.frombuffer(chunk, dtype=np.int16)
                if native != 16000:
                    audio = resample_poly(audio.astype(np.float32), up, down)
                    audio = np.clip(audio, -32768, 32767).astype(np.int16)
                pcm = audio.tobytes()
                ring.append(pcm)
                ring_bytes += len(pcm)
                while ring_bytes > self._preroll_bytes and len(ring) > 1:
                    ring_bytes -= len(ring.popleft())
                scores = oww.predict(audio)
                score = max(scores.values()) if scores else 0.0
                if score >= self._threshold:
                    consecutive += 1
                    peak = max(peak, score)
                    if consecutive >= self._patience:
                        oww.reset()
                        log_event(f"wake word detected (score={peak:.2f}, threshold={self._threshold}, "
                                  f"frames={consecutive})")
                        self._on_wake(peak, b"".join(ring))
                        ring.clear()
                        ring_bytes = 0
                        consecutive = 0
                        peak = 0.0
                        time.sleep(2)  # debounce
                else:
                    if consecutive:
                        # would have woken her before the patience filter — keep a
                        # trace so the threshold/patience knobs can be tuned later
                        log_event(f"wake spike ignored (score={peak:.2f}, "
                                  f"only {consecutive}/{self._patience} frames)")
                    consecutive = 0
                    peak = 0.0
        finally:
            stream.close()
            pa.terminate()


def list_audio_devices():
    import pyaudio

    pa = pyaudio.PyAudio()
    print("index | in/out channels | name")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"{i:5d} | in:{info['maxInputChannels']:2d} out:{info['maxOutputChannels']:2d} | {info['name']}")
    pa.terminate()


def _require_env(*names):
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        print("Missing required .env keys: " + ", ".join(missing))
        if "DEEPGRAM_API_KEY" in missing:
            print("  -> Get a free Deepgram key (comes with $200 credit) at https://console.deepgram.com/signup")
        sys.exit(1)


def _play_ready_chime():
    """Two soft rising blips through the DEFAULT output (the user's speakers,
    not Aiva's voice cable) — audible proof of life once the pipeline is up."""
    import math
    import struct
    import tempfile
    import wave
    import winsound

    sr = 44100

    def tone(freq, dur, vol=0.35):
        n = int(sr * dur)
        return [vol * math.sin(2 * math.pi * freq * t / sr) * (1 - t / n) for t in range(n)]

    samples = tone(660, 0.12) + tone(880, 0.20)
    path = os.path.join(tempfile.gettempdir(), "aiva_ready_chime.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(s * 32767)) for s in samples))
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)


_RESTART_MARKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".restart_marker")


def _consume_restart_marker():
    """True once if the previous process exited via self-restart just now."""
    try:
        fresh = time.time() - os.path.getmtime(_RESTART_MARKER) < 300
        os.remove(_RESTART_MARKER)
        return fresh
    except OSError:
        return False


def _spawn_detached_relauncher():
    """Relaunch fallback for when Aiva was NOT started by start_aiva.bat (whose
    restart loop handles exit code 42 itself): a hidden helper python waits on
    this process's kernel handle, then starts the same command line again in a
    fresh console window."""
    import subprocess

    watcher = (
        "import ctypes, subprocess, time; "
        "k = ctypes.windll.kernel32; "
        f"h = k.OpenProcess(0x00100000, False, {os.getpid()}); "  # SYNCHRONIZE
        "k.WaitForSingleObject(h, 0xFFFFFFFF); "
        "time.sleep(0.3); "
        f"subprocess.Popen({[sys.executable] + sys.argv!r}, cwd={os.getcwd()!r}, "
        "creationflags=subprocess.CREATE_NEW_CONSOLE)"
    )
    subprocess.Popen(
        [sys.executable, "-c", watcher],
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
    )


# _LLM_RATES lives in modules/utilities.py (shared with the task runner's cost budget)


def _print_usage_summary(mic: MicController, usage: UsageTracker, llm_model: str):
    """Session cost estimate (rates: mid-2026, see plan)."""
    stt_min = mic.stt_seconds / 60
    stt_cost = stt_min * 0.0077
    rate_in, rate_out = _LLM_RATES.get(llm_model, _LLM_RATES["gpt-4.1-mini"])
    llm_cost = usage.prompt_tokens * rate_in / 1e6 + usage.completion_tokens * rate_out / 1e6
    tts_credits = usage.tts_chars * 0.5  # Flash v2.5
    print("\n----- session usage -----")
    print(f"STT streamed:  {stt_min:.1f} min       (~${stt_cost:.3f})")
    print(f"LLM tokens:    {usage.prompt_tokens} in / {usage.completion_tokens} out (~${llm_cost:.3f})")
    print(f"TTS:           {usage.tts_chars} chars = {tts_credits:.0f} ElevenLabs credits (free tier: 10k/month)")
    print("-------------------------")


async def main():
    _require_env("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "DEEPGRAM_API_KEY")

    llm_model = os.getenv("AIVA_LLM_MODEL", "gpt-4.1-mini")
    ambient = os.getenv("AIVA_WAKE_WORD", "0") == "1"
    idle_timeout = float(os.getenv("AIVA_IDLE_TIMEOUT", "20"))
    from version import __version__ as _aiva_version
    print(f"Aiva v{_aiva_version}")
    log_event(f"--- boot v{_aiva_version} (pid={os.getpid()}, ambient={ambient}, "
              f"idle_timeout={idle_timeout}s) ---")

    # --- transport & services ----------------------------------------------
    def _device_index(env_index, env_name, output):
        """Resolve an audio device from env: numeric index, or (more robust,
        indices shift when drivers change) a name substring like 'CABLE Input'."""
        value = os.getenv(env_index)
        if value:
            return int(value)
        name = os.getenv(env_name)
        if not name:
            return None
        import pyaudio

        pa = pyaudio.PyAudio()
        try:
            channel_key = "maxOutputChannels" if output else "maxInputChannels"
            candidates = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if name.lower() in info["name"].lower() and info[channel_key] > 0:
                    host = pa.get_host_api_info_by_index(info["hostApi"])["name"]
                    candidates.append((i, info["name"], host))
            if candidates:
                # A device shows up once per host API and they are NOT equal.
                # MME works for both directions as long as capture runs at the
                # device's NATIVE rate (16 kHz capture silently dies) — output
                # MME resamples anything. WASAPI capture is fine but its output
                # rejects non-mix-format rates, and DirectSound capture
                # stutters. So: MME everywhere, native input rate.
                def rank(c):
                    if "MME" in c[2]:
                        return 0
                    if "WASAPI" in c[2]:
                        return 1
                    return 2
                candidates.sort(key=rank)
                i, dev_name, host = candidates[0]
                print(f"Audio {'output' if output else 'input'}: [{i}] {dev_name} ({host})")
                return i
        finally:
            pa.terminate()
        print(f"WARNING: no {'output' if output else 'input'} device matching '{name}'; using default")
        return None

    def _native_rate(device_index):
        """Capture must run at the device's native rate: forcing 16 kHz makes
        Windows deliver silence (MME) or corrupted repeating buffers (DS)."""
        import pyaudio

        pa = pyaudio.PyAudio()
        try:
            if device_index is None:
                return int(pa.get_default_input_device_info()["defaultSampleRate"])
            return int(pa.get_device_info_by_index(device_index)["defaultSampleRate"])
        finally:
            pa.terminate()

    mic_index = _device_index("AIVA_INPUT_DEVICE_INDEX", "AIVA_INPUT_DEVICE_NAME", False)
    mic_rate = _native_rate(mic_index)

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=mic_rate,  # pipecat resamples for STT/VAD
            input_device_index=mic_index,
            output_device_index=_device_index("AIVA_OUTPUT_DEVICE_INDEX", "AIVA_OUTPUT_DEVICE_NAME", True),
        )
    )

    # Keyterm prompting biases nova-3 toward words it otherwise mangles.
    # Add more without code changes via AIVA_STT_KEYTERMS (comma-separated).
    stt_keyterms = ["Aiva", "Claude Code", "PowerShell", "Frank", "Bogdan", "Fianu"]
    stt_keyterms += [k.strip() for k in os.getenv("AIVA_STT_KEYTERMS", "").split(",") if k.strip()]

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-3",
            language="en-US",
            smart_format=True,
            profanity_filter=False,  # default True asterisks words out before the LLM sees them
            keyterm=stt_keyterms,
        ),
    )

    llm = PatientOpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        # NOTE: keep run_in_parallel at its default (True). Sequential mode
        # registers calls lazily, which defeats pipecat's call-grouping and
        # makes her speak once per tool instead of once per turn. Parallel
        # tasks still START in emitted order, and our handlers are nearly all
        # synchronous, so "X then Y" ordering holds in practice.
        # gpt-5.x: max_completion_tokens instead of max_tokens, and function
        # tools on Chat Completions REQUIRE reasoning_effort 'none' (also the
        # low-latency mode we want). gpt-4.x keeps the classic shape.
        settings=OpenAILLMService.Settings(
            model=llm_model,
            temperature=0.7,
            **({"max_completion_tokens": 500, "extra": {"reasoning_effort": "none"}}
               if llm_model.startswith("gpt-5") else {"max_tokens": 500}),  # headroom for multi-tool turns
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID"),
            model=os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
        ),
    )

    # --- avatar + tools ------------------------------------------------------
    mic = MicController(ambient=ambient,
                        verify_secs=float(os.getenv("AIVA_WAKE_VERIFY_SECS", "5")))
    name_variants = _NAME_VARIANTS + tuple(
        v.strip().lower() for v in os.getenv("AIVA_NAME_VARIANTS", "").split(",") if v.strip())
    terminals = TerminalManager()
    memory = Memory()  # created here so the remember tool can write to it
    from modules import metrics as _metrics
    from modules.metrics import Metrics
    _metrics.set_recorder(Metrics(memory.conn))  # operations profiler (dashboard Statistics)
    # one OpenAI client for everything out-of-band: embeddings, task planning
    # (the pipeline's LLM service keeps its own)
    from openai import AsyncOpenAI
    oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                      max_retries=int(os.getenv("AIVA_LLM_MAX_RETRIES", "5")))
    task_memory = TaskMemory(memory.conn, embed=lambda text: embed_text(oai, text),
                             name_variants=name_variants)
    interrupted_tasks = task_memory.reconcile_stale()
    if interrupted_tasks:
        log_event(f"{len(interrupted_tasks)} task(s) were interrupted by the last shutdown")
    cat = DesktopCat()  # her pixel pet — the window runs on its own thread
    cat.start()

    # her body: adopt a renderer that survived a restart, or spawn a fresh one
    # (non-fatal — she goes voice-only if it can't come up)
    avatar = AvatarClient()
    await avatar.ensure_running()

    # --- restart machinery -------------------------------------------------
    # Restarting is the reliable way to pick up code changes (hot-reloading a
    # live pipeline plus threads is not). restart_self lets her do it herself:
    # she says a one-line goodbye, we wait for the TTS to finish, then exit
    # with code 42 — start_aiva.bat's loop relaunches her (or a detached
    # watcher does, if she wasn't started through the bat). The avatar
    # renderer stays up across it, so she's back in seconds.
    stop_event = asyncio.Event()
    restart_state = {"requested": False}

    async def trigger_restart():
        deadline = time.time() + 30
        while not mic.bot_speaking and time.time() < deadline - 20:
            await asyncio.sleep(0.1)  # give her goodbye line time to start
        while mic.bot_speaking and time.time() < deadline:
            await asyncio.sleep(0.1)  # ...and let it finish playing
        await asyncio.sleep(0.3)
        restart_state["requested"] = True
        stop_event.set()

    handlers = register_tools(llm, avatar, mic, terminals, memory, cat, restart=trigger_restart)

    # --- task runner: multi-step jobs in the background ------------------------
    # Reuses the very same tool handlers out-of-band (ToolBox), plans/verifies
    # with its own OpenAI calls, and reports through announce() like Claude
    # Code jobs. Shell steps go through execute_shell only after the user's yes.
    task_model = os.getenv("AIVA_TASK_MODEL", llm_model)
    toolbox = ToolBox(handlers, TOOL_SCHEMAS)
    task_runner = TaskRunner(
        task_memory=task_memory, toolbox=toolbox,
        llm_call=openai_llm_call(oai, task_model, label="task"), terminals=terminals, log=log_event,
        execute_shell=lambda cmd: asyncio.to_thread(execute_shell, cmd),
        describe_screen=computer.describe_screen, list_monitors=computer.list_monitors,
        resolve_path=file_operations.resolve_path, facts=memory.load_facts,
        model_name=task_model,
    )
    task_runner.openai_client = oai        # grades delegated (Claude Code) work
    task_runner.remember = memory.remember  # preference patterns become durable facts
    register_task_tools(llm, task_runner, task_memory, memory)

    # --- reading engine + becoming layer --------------------------------------
    # She reads books from aiva/library in the background, studies them section
    # by section, and is (bounded-ly) changed by them. Study/synthesis run on
    # their own models so they never eat the voice model's tokens-per-minute.
    _LIBRARY_DIR = os.getenv("AIVA_LIBRARY_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "library")
    os.makedirs(_LIBRARY_DIR, exist_ok=True)
    study_model = os.getenv("AIVA_STUDY_MODEL", task_model)
    synth_model = os.getenv("AIVA_STUDY_SYNTH_MODEL", "gpt-5.6-terra")
    book_memory = BookMemory(memory.conn, embed=lambda t: embed_long(oai, t))
    interrupted_reading = book_memory.reconcile_stale()
    reading = ReadingEngine(book_memory, llm_call=openai_llm_call(oai, study_model, label="reading"),
                            synth_call=openai_llm_call(oai, synth_model, label="synth"), client=oai,
                            library_dir=_LIBRARY_DIR, model_name=study_model,
                            synth_model_name=synth_model, log=log_event, limits=StudyLimits.from_env())
    register_reading_tools(llm, reading)

    async def _becoming_llm(messages, max_tokens=1200, temperature=0.2):
        from modules.utilities import llm_kwargs
        async with _metrics.span("llm", "becoming"):
            r = await oai.chat.completions.create(
                model=task_model, messages=messages, response_format={"type": "json_object"},
                **llm_kwargs(task_model, max_tokens, temperature))
        return r.choices[0].message.content

    becoming = Becoming(memory.conn, embed=lambda t: embed_text(oai, t), llm=_becoming_llm,
                        model=task_model, remember=memory.remember, baseline_traits=baseline_traits(),
                        log=log_event, engine=reading)
    register_becoming_tools(llm, becoming)
    # reading's outcomes flow into who she becomes
    reading.on_book_finished = becoming.on_book_finished
    reading.on_session_end = becoming.on_session_end
    reading.on_book_added = becoming.on_book_added
    reading.on_session_start = becoming.note_session_started
    reading.on_reading_state = avatar.send_prop  # her book appears while she reads
    task_runner._reading = becoming.relevant_for  # lessons reach task planning (used if supported)
    becoming.fade()  # let unused traits/lessons decay a little at boot

    # --- her reach to the web + her own space for what she gathers -----------------
    # search/read/look-at-image + a curated "space" (aiva/space) she keeps finds in, mirroring the
    # library. She finds and shows; there is deliberately no buy/sign-in/send tool.
    _SPACE_DIR = os.getenv("AIVA_SPACE_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "space")
    os.makedirs(_SPACE_DIR, exist_ok=True)
    space_mem = SpaceMemory(memory.conn, embed=lambda t: embed_text(oai, t), log=log_event)
    web_client = WebClient(WebConfig.from_env(), log=log_event)
    # per-service spend guardrail: each service capped at its own budget (Tavily monthly is the main
    # one), ntfy alert at ~80%/100%, and the self-started work that hit the cap pauses — voice never.
    notifier = Notifier(NotifyConfig.from_env(), log=log_event)
    service_limits = ServiceLimits(memory.conn, notifier=notifier, log=log_event)
    print(f"ntfy alerts {'ON' if notifier.available() else 'OFF (set NTFY_TOPIC)'} · "
          f"Tavily cap {service_limits.cap('tavily') or 'off'}/mo")
    _web_llm = openai_llm_call(oai, os.getenv("AIVA_WEB_MODEL") or task_model, label="web")

    async def _summarize_page(text, *, focus=None):
        """Summarise a fetched page off the voice model's token bucket (label='web')."""
        prompt = ("Summarize this web page for Aiva in two to four plain spoken sentences — no "
                  "markdown, no lists. Lead with what it actually says; keep any key name or number "
                  "exact.")
        if focus:
            prompt += f" She wants to know: {focus}."
        try:
            reply = await _web_llm([{"role": "system", "content": prompt},
                                    {"role": "user", "content": (text or "")[:8000]}],
                                   max_tokens=300, temperature=0.3)
            return (reply.text or "").strip() or (text or "")[:600]
        except Exception:
            return (text or "")[:600]

    web_handlers = register_web_tools(llm, web_client, space_mem,
                                      describe_image=computer.describe_image,
                                      summarize=_summarize_page, space_dir=_SPACE_DIR,
                                      open_file=file_operations.open_file, limits=service_limits,
                                      log=log_event)
    toolbox.add_handlers(web_handlers)   # so allow-listed web/space verbs are callable by the task runner
    print(f"Web reach {'ON' if web_client.available() else 'OFF (set TAVILY_API_KEY)'} · "
          f"space at {os.path.basename(_SPACE_DIR)}")

    # --- her nervous system: one shared mood + a private diary + her own wishes ----
    # (docs/PLAN-selfhood-and-eval.md) Everything below emits onto ONE experience bus;
    # the self loop integrates it into her mood and, when earned, her diary/desires.
    inner_brain = InnerBrain()                       # mood knobs from AIVA_MOOD_* in .env
    experience = ExperienceBus(memory.conn)
    _inner_state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inner_state.json")
    # her mutable tastes/opinions: seeded once from the frozen baseline, then they evolve from
    # experience (formed in the journal digest, deduped/capped/faded/vetoable like traits/desires)
    preferences = Preferences(memory.conn, embed=lambda t: embed_text(oai, t), log=log_event)
    _seeded = preferences.seed_from_baseline()
    if _seeded:
        print(f"Seeded {_seeded} baseline preferences (now mutable)")
    preferences.fade()                               # let unused likes/opinions decay a touch at boot
    register_preference_tools(llm, preferences)
    self_engine = SelfEngine(
        memory.conn, experience, inner_brain,
        state=InnerState.load(_inner_state_path, inner_brain), state_path=_inner_state_path,
        embed=lambda t: embed_text(oai, t), llm=openai_llm_call(oai, task_model, label="self"),
        model=task_model, log=log_event, preferences=preferences,
        journal_salience=float(os.getenv("AIVA_JOURNAL_SALIENCE", "2.0")),
        journal_min_gap_h=float(os.getenv("AIVA_JOURNAL_MIN_GAP_H", "12")))
    register_self_tools(llm, self_engine)

    # --- speaker recognition: she learns voices and asks when she doesn't know one ---------
    _speaker_model = os.getenv("AIVA_SPEAKER_MODEL") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "campplus_cn_en_common_200k.onnx")
    _speaker_cfg = SpeakerConfig.from_env()
    speaker_mem = SpeakerMemory(memory.conn)
    recognizer = SpeakerRecognizer(
        speaker_mem, _speaker_cfg, model_path=_speaker_model,
        enabled=(os.getenv("AIVA_SPEAKER_ID", "1") == "1" and os.path.isfile(_speaker_model)),
        log=log_event)
    register_speaker_tools(llm, recognizer)
    user_audio_tap = UserAudioTap(mic, max_secs=_speaker_cfg.max_secs)
    if recognizer.enabled:
        print(f"Speaker recognition ON (model {os.path.basename(_speaker_model)})")
        asyncio.create_task(asyncio.to_thread(recognizer.warm))   # load the ONNX off the hot path
    else:
        print("Speaker recognition OFF (set AIVA_SPEAKER_ID=1 and provide the model)")

    # vocal affect (paralinguistics): how he SOUNDS -> nudges her mood via the experience bus.
    # No model, no new deps — runs off the same per-turn audio the speaker tap captures.
    vocal_affect = VocalAffect(AffectConfig.from_env(),
                               enabled=os.getenv("AIVA_VOICE_AFFECT", "1") == "1", log=log_event)
    print(f"Vocal affect {'ON' if vocal_affect.enabled else 'OFF'}")

    # reading feeds her mood: each section read and each finished book emit onto the bus
    _NEG = {"sad", "bleak", "unsettled", "uneasy", "melancholy", "grim", "lonely", "angry",
            "frustrated", "somber", "disturbed"}
    _POS = {"delighted", "warm", "moved", "glad", "hopeful", "amused", "charmed", "content",
            "tender", "curious", "thoughtful"}
    _HI = {"hooked", "tense", "thrilled", "restless", "excited", "gripped", "anxious", "alarmed"}
    _CALM = {"calm", "quiet", "soothing", "gentle", "slow", "peaceful"}

    def _feeling_hint(feeling):
        f = (feeling or "").lower()
        v = -0.3 if any(w in f for w in _NEG) else (0.2 if any(w in f for w in _POS) else 0.05)
        a = 0.25 if any(w in f for w in _HI) else (-0.1 if any(w in f for w in _CALM) else 0.0)
        return v, a

    _prev_on_section = getattr(reading, "on_section", None)

    async def _emit_on_section(book, note):
        try:
            feeling = note.get("feeling") or ""
            v, a = _feeling_hint(feeling)
            experience.emit("chapter_read", "reading",
                            f"read a section of {book.get('title', 'a book')}"
                            + (f" — felt {feeling}" if feeling else ""),
                            valence=v, arousal=a, salience=0.5)
        except Exception as e:
            log_event(f"self emit(section): {e!r}")
        if _prev_on_section:
            await _prev_on_section(book, note)
    reading.on_section = _emit_on_section

    _prev_on_finish = reading.on_book_finished

    async def _emit_on_finish(book, synthesis, notes):
        try:
            experience.emit("book_finished", "reading",
                            f"finished {book.get('title', 'a book')}",
                            valence=0.5, arousal=0.1, salience=0.9)
        except Exception as e:
            log_event(f"self emit(finish): {e!r}")
        if _prev_on_finish:
            await _prev_on_finish(book, synthesis, notes)
    reading.on_book_finished = _emit_on_finish

    # open the vision API connection now so her first "look at the screen"
    # doesn't eat a cold-start handshake on top of the call
    from modules.computer import prewarm as _prewarm_vision
    asyncio.create_task(_prewarm_vision())

    avatar_hotkeys = list(avatar.expressions)
    if avatar_hotkeys:
        print(f"Avatar expressions available: {len(avatar_hotkeys)}")

    # --- memory + context ----------------------------------------------------
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    facts = memory.load_facts()
    if facts:
        print(f"Loaded {len(facts)} remembered facts")

    # episodic continuity: recap whatever session ended last (works even after a
    # hard kill — transcripts flush every 60s), then inject the freshest recaps
    try:
        from openai import AsyncOpenAI

        _recap_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        await asyncio.wait_for(
            memory.recap_last_session(_recap_client, llm_model), timeout=15
        )
    except Exception as e:
        print(f"Session recap skipped: {e}")
    recaps = memory.load_recaps(limit=2)
    if recaps:
        print(f"Loaded {len(recaps)} session recap(s)")

    intro = "Introduce yourself very briefly and greet the user."
    if _consume_restart_marker():
        intro = ("You just restarted yourself (for example to load new code) and are back "
                 "online. Say you're back in one short sentence — no full introduction.")

    try:
        added = await asyncio.wait_for(reading.scan_once(), 10)
        if added.get("added"):
            log_event(f"library: {len(added['added'])} new book(s) on boot")
    except Exception as e:
        log_event(f"library scan skipped: {e}")

    def render_system_prompt():
        blocks = becoming.prompt_blocks(reading.library_status_lines())
        pd = self_engine.prompt_data()
        roster = recognizer.roster() if recognizer.enabled else {"current": None, "known": []}
        return build_system_prompt(
            memory.load_facts(), avatar_hotkeys, cat.persona_note(), memory.load_recaps(limit=2),
            flows=task_memory.load_flows(), lessons=task_memory.load_lessons(),
            capabilities=capabilities.prompt_lines(),
            reading=blocks.get("reading"), traits=blocks.get("traits"),
            voice=blocks.get("voice"), reading_lessons=blocks.get("lessons"),
            mood_line=pd["mood_line"], wants=pd["wants"],
            preferences=preferences.prompt_block(),
            space=space_mem.status_lines(),
            current_speaker=roster["current"], known_speakers=roster["known"])

    messages = [
        {"role": "system", "content": render_system_prompt()},
        {"role": "system", "content": intro},
    ]
    if cat.available and cat.state.name is None and not cat.state.dismissed:
        # unnamed cat: first boot gets the first-meeting note; if that boot
        # somehow didn't save a name, later boots get the quiet recovery note
        # instead — so the adoption can never be announced as news twice
        note = CAT_NAME_RECOVERY_NOTE if cat.state.naming_note_shown else CAT_NAMING_NOTE
        messages.append({"role": "system", "content": note})
        cat.state.naming_note_shown = True
        cat.save_state()
    for t in interrupted_tasks[:1]:
        done = sum(1 for st in t["plan"] if st.get("status") == "done")
        messages.append({"role": "system", "content": (
            f"[A task was interrupted by your restart: {t['goal_text']} (was at step {done} of "
            f"{len(t['plan'])}). Mention it in one sentence and offer to pick it up.]")})
    for r in interrupted_reading[:1]:
        messages.append({"role": "system", "content": (
            f"[Your reading of '{r['title']}' was interrupted by the restart at section "
            f"{r['cursor'] + 1} of {r['sections']}; pick it up when there's a quiet moment.]")})
    context = LLMContext(messages=messages, tools=TOOL_SCHEMAS)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            # second layer against self-hearing (primary gate: MicController)
            user_mute_strategies=[AlwaysUserMuteStrategy()],
            user_turn_strategies=UserTurnStrategies(
                # stop turns on the semantic smart-turn model so stutters
                # and "uhhmm" pauses don't cut the user off
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3()
                    )
                ],
            ),
        ),
    )

    # --- pipeline -------------------------------------------------------------
    usage = UsageTracker()

    async def _on_user_final(text):
        """For the turn that just ended, from the SAME captured PCM: (1) read how he sounded and
        nudge her mood via the experience bus, and (2) identify who spoke and, on a change/unknown
        voice, note it before she answers. Both guarded, both skip silently — never stall the turn."""
        pcm = user_audio_tap.take_segment()
        if len(pcm) < int(16000 * 0.3) * 2:          # < ~0.3s of audio: nothing to read
            return
        # (1) vocal affect -> her mood (no model; the self loop integrates the event)
        if vocal_affect.enabled:
            try:
                affect = await asyncio.to_thread(vocal_affect.analyze, pcm)
                if affect:
                    experience.emit("voice_affect", "conversation", f"he sounded {affect['label']}",
                                    arousal=affect["arousal"], valence=affect["valence"],
                                    social=0.0, salience=affect["salience"])
            except Exception as e:
                log_event(f"vocal affect error: {e!r}")
        # (2) speaker identity -> who is this, ask if unknown
        if recognizer.enabled and len(pcm) >= int(_speaker_cfg.min_secs * 16000) * 2:
            try:
                vec = await asyncio.wait_for(recognizer.embed_pcm(pcm),
                                             timeout=_speaker_cfg.embed_timeout_ms / 1000)
                note = await recognizer.on_turn(vec)
                if note:
                    context.add_message({"role": "system", "content": note})
                    try:
                        refresh_system_prompt()      # roster reflects the current speaker
                    except Exception:
                        pass
            except Exception:
                pass

    pipeline = Pipeline([
        transport.input(),
        mic,
        user_audio_tap,
        stt,
        ActivityWatcher(mic, name_variants, on_user_final=_on_user_final),
        user_aggregator,
        llm,
        ProcessingWatcher(mic),
        tts,
        LipSyncTap(avatar),
        usage,
        transport.output(),
        assistant_aggregator,
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        # Aiva is a long-running companion whose sleep/wake WE manage (ambient
        # mode). Pipecat's built-in idle monitor defaults to cancelling the
        # whole worker after 300s with no BotSpeaking/UserSpeaking frame — but
        # while she's asleep MicController drops all audio, so no such frame
        # ever flows and she'd kill herself ~5 min after the last word. Disable
        # it entirely; she only stops when told to (Ctrl+Alt+X / go_to_sleep).
        idle_timeout_secs=None,
        app_resources={"task_runner": task_runner, "task_memory": task_memory},
    )

    # If the pipeline ever ends for a reason other than us asking it to, say so
    # loudly — a silent clean exit (like the old idle-timeout self-cancel) is
    # otherwise indistinguishable from a normal shutdown.
    @worker.event_handler("on_pipeline_finished")
    async def _on_finished(_worker, frame):
        reason = getattr(frame, "reason", None)
        print(f"\n[PIPELINE ENDED] {type(frame).__name__}"
              + (f" reason={reason}" if reason else ""))

    @worker.event_handler("on_pipeline_error")
    async def _on_pipeline_error(_worker, frame):
        print(f"\n[PIPELINE ERROR] {getattr(frame, 'error', '?')} "
              f"(fatal={getattr(frame, 'fatal', False)})")

    async def announce(text: str):
        """Voice path for background-job news: inject a system note and
        trigger a turn so Aiva proactively speaks up."""
        log_event(f"announce: {text.splitlines()[0][:160]}")
        mic.wake("background news announce")  # eyes open, ears on — she has news
        context.add_message({"role": "system", "content": text})
        await worker.queue_frames([LLMRunFrame()])

    terminals._announce = announce

    async def cat_announce(text: str):
        """Cat events reach her voice only while she's awake. Asleep, the event
        becomes a quiet context note (event line only, minus the 'react out
        loud' instruction) she can bring up naturally next time she's talking."""
        if mic.awake:
            await announce(text)
        else:
            log_event(f"cat event while asleep (noted quietly): {text.splitlines()[0][:160]}")
            context.add_message({
                "role": "system",
                "content": "[While you were asleep] " + text.split("\n")[0],
            })

    async def quiet_task_note(text: str):
        """Task progress while she's asleep: noted, not spoken (cat_announce pattern)."""
        context.add_message({"role": "system", "content": "[While you were asleep] " + text})

    task_runner.attach_voice(announce=announce, quiet_note=quiet_task_note,
                             is_awake=lambda: mic.awake, spawn=llm.create_task)

    def refresh_system_prompt():
        msgs = context.get_messages()
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {"role": "system", "content": render_system_prompt()}
            context.set_messages(msgs)

    _intro_pending = {"v": True}

    def _strip_intro():
        """Once she's spoken her first turn, retire the one-shot intro note so she doesn't
        re-introduce herself on every subsequent wake."""
        if not _intro_pending["v"]:
            return
        msgs = context.get_messages()
        if not any(m.get("role") == "assistant" for m in msgs):
            return  # she hasn't greeted yet — keep the note
        kept, removed = drop_intro_note(msgs, intro)
        if removed:
            context.set_messages(kept)
            log_event("intro note consumed after first greeting")
        _intro_pending["v"] = False

    def _user_present(within_s=180):
        try:
            import ctypes

            class _LII(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
            lii = _LII(); lii.cbSize = ctypes.sizeof(_LII)
            if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
                return None
            idle_ms = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
            return (idle_ms / 1000.0) <= within_s
        except Exception:
            return None

    def _task_running():
        return any(t.get("status") in ("planning", "awaiting_user", "running", "verifying")
                   for t in getattr(task_runner, "tasks", {}).values())

    reading.attach_voice(announce=announce, quiet_note=quiet_task_note, is_awake=lambda: mic.awake,
                         is_idle=lambda: mic.idle_for() > 20, spawn=llm.create_task)
    becoming.attach_voice(announce=announce, quiet_note=quiet_task_note, is_awake=lambda: mic.awake,
                          idle_for=mic.idle_for, user_present=_user_present,
                          task_busy=lambda: mic.bot_speaking or mic.processing or _task_running(),
                          engine=reading, refresh_prompt=refresh_system_prompt)
    # --- safe autonomous action: she may pursue a wish herself, but ONLY with a reversible verb ---
    _ACT_CAPS = {"find_and_keep"}          # what _act_on_wish can actually do in v1
    _self_act_on = os.getenv("AIVA_SELF_ACT", "1") == "1"

    def _self_actionable(desire):
        """True only for a wish she may safely pursue unsupervised: an auto-tier capability this
        dispatcher can perform, whose service isn't paused. Everything else stays an 'ask'."""
        if not _self_act_on:
            return False
        cap = capabilities.match(desire.get("text", ""))
        if not cap or cap.tier != "auto" or cap.name not in _ACT_CAPS:
            return False
        if cap.name == "find_and_keep" and not service_limits.allow("tavily"):
            return False
        return True

    async def _act_on_wish(desire):
        """Find-and-keep: search images for the wish, look at one, keep it in her space, tell him.
        Reversible (downloads to her own space); Tavily-gated; never buys or sends."""
        text = (desire.get("text") or "").strip()
        if not text or not service_limits.allow("tavily"):
            return
        out = await asyncio.to_thread(web_client.search, text, "images")
        await service_limits.record("tavily")
        images = [] if out.get("error") else out.get("images", [])
        for im in images[:3]:
            url = im.get("url")
            if not url:
                continue
            dl = await asyncio.to_thread(web_client.download, url, _SPACE_DIR, text)
            if not dl.get("ok"):
                continue
            try:
                await computer.describe_image(dl["path"], looking_for=text)   # she actually looks
            except Exception:
                pass
            await space_mem.save_find(text[:80], "went looking for this on my own — liked this one",
                                      kind="image", source_url=url, path=dl["path"],
                                      file_name=dl["file_name"], sha256=dl["sha256"])
            announce(f"[On your own, in a quiet moment, you went looking for '{text}' and kept one you "
                     f"liked in your space. In ONE short natural sentence, tell him you found something "
                     f"and offer to show it.]")
            return
        log_event(f"self act: found nothing to keep for {text!r}")

    self_engine.attach(refresh_prompt=refresh_system_prompt,  # mood/wishes update her prompt live
                       actionable=_self_actionable, act=_act_on_wish)
    preferences.attach(refresh_prompt=refresh_system_prompt)  # likes/opinions update her prompt live
    recognizer.attach(refresh_prompt=refresh_system_prompt)   # roster updates when she enrolls/renames
    self_engine.attach_voice(announce=announce, quiet_note=quiet_task_note,
                             is_awake=lambda: mic.awake, idle_for=mic.idle_for,
                             user_present=_user_present,
                             task_busy=lambda: mic.bot_speaking or mic.processing
                             or _task_running() or reading.busy)

    # --- hotkeys ---------------------------------------------------------------
    loop = asyncio.get_running_loop()
    cat.attach_announcer(loop, cat_announce)  # cat events never wake her up
    try:
        import keyboard

        def interrupt_bot():
            print("\n[F10] Interrupting Aiva")
            asyncio.run_coroutine_threadsafe(
                worker.queue_frames([InterruptionWorkerFrame()]), loop
            )

        def hotkey_restart():
            print("\n[Ctrl+Alt+R] Restarting Aiva")
            restart_state["requested"] = True
            loop.call_soon_threadsafe(stop_event.set)

        keyboard.add_hotkey("f9", lambda: loop.call_soon_threadsafe(mic.toggle_mute))
        keyboard.add_hotkey("f10", interrupt_bot)
        # NOT Esc: a global Esc hotkey means any stray Escape press anywhere
        # in Windows silently kills her — it did, repeatedly.
        keyboard.add_hotkey("ctrl+alt+x", lambda: loop.call_soon_threadsafe(stop_event.set))
        keyboard.add_hotkey("ctrl+alt+r", hotkey_restart)
        print("Hotkeys: F9 = mute/unmute mic, F10 = interrupt Aiva, "
              "Ctrl+Alt+R = restart, Ctrl+Alt+X = quit")
    except Exception as e:
        print(f"Global hotkeys unavailable ({e}); use Ctrl+C to quit")

    async def watch_stop():
        await stop_event.wait()
        print("\nShutting down...")
        await worker.cancel()

    watcher = asyncio.create_task(watch_stop())

    # --- ambient mode: wake word + idle timeout --------------------------------
    wake_listener = None
    idle_task = None
    sleep_face_task = None
    if ambient:
        # AIVA_WAKE_VERIFY=0 restores trust-the-model wakes (not recommended:
        # the custom model fires on TV/game speech at the same scores as real
        # wakes — grep aiva/logs for 'wake rejected' to see what it filters)
        verify_wakes = os.getenv("AIVA_WAKE_VERIFY", "1") == "1"
        wake_listener = WakeWordListener(
            model=os.getenv("AIVA_WAKE_MODEL", "hey_jarvis"),
            on_wake=lambda score, preroll: loop.call_soon_threadsafe(
                mic.wake, f"wake word (score={score:.2f})", preroll, verify_wakes
            ),
            device_index=_device_index("AIVA_INPUT_DEVICE_INDEX", "AIVA_INPUT_DEVICE_NAME", False),
            threshold=float(os.getenv("AIVA_WAKE_THRESHOLD", "0.5")),
            patience=int(os.getenv("AIVA_WAKE_PATIENCE", "2")),
        )
        wake_listener.start()

        async def idle_watch():
            while True:
                await asyncio.sleep(2)
                # never doze off mid-sentence, mid-thought, or mid-tool-call
                if (mic.awake and not mic.bot_speaking and not mic.processing
                        and mic.idle_for() > idle_timeout):
                    mic.sleep()

        async def sleep_face():
            """Keep the avatar's sleep state in sync with hers: eyes closed
            while asleep, auto-blink back when awake. Edge-triggered — one IPC
            message per transition (the renderer holds the state). A renderer
            respawn resets `last`, so the fresh body (which boots awake-faced)
            is told the current state on the next tick."""
            last = None
            while True:
                await asyncio.sleep(0.5)
                if not avatar.connected:
                    last = None
                    continue
                asleep = not mic.visibly_awake
                if asleep != last and await avatar.set_sleeping(asleep):
                    last = asleep

        sleep_face_task = asyncio.create_task(sleep_face())
        idle_task = asyncio.create_task(idle_watch())
        print("[IDLE] say the wake word to talk to Aiva")

    # --- periodic transcript persistence ----------------------------------------
    async def flush_transcripts():
        """Persist new transcript lines, then keep the live context inside its
        token budget (save FIRST — trimming drops what was already saved)."""
        while True:
            await asyncio.sleep(15)
            memory.save_new_messages(session_id, context.get_messages())
            _strip_intro()
            dropped = trim_context(context, memory)
            if dropped:
                log_event(f"context trimmed: dropped {dropped} old message(s)")

    flush_task = asyncio.create_task(flush_transcripts())

    library_task = asyncio.create_task(reading.watch_library(60))

    async def curiosity_watch():
        while True:
            await asyncio.sleep(60)
            try:
                await becoming.curiosity_tick()
            except Exception as e:
                log_event(f"curiosity error: {e!r}")

    curiosity_task = asyncio.create_task(curiosity_watch())

    # --- self loop: absorb experience -> mood -> (when earned) diary + desires -----
    def _daypart(h):
        return "night" if h < 6 else "morning" if h < 12 else "afternoon" if h < 18 else "evening"

    async def self_watch():
        interval = int(os.getenv("AIVA_SELF_TICK_S", "60"))
        last_daypart = None
        while True:
            await asyncio.sleep(interval)
            try:
                dp = _daypart(datetime.now().hour)
                if dp != last_daypart:
                    last_daypart = dp
                    experience.emit(
                        "daypart", "time", f"it's {dp} now",
                        arousal=(-0.12 if dp == "night" else 0.08 if dp == "morning" else 0.0),
                        social=(0.1 if dp == "evening" else 0.0), salience=0.2)
                await self_engine.tick()
                await self_engine.initiative_tick()   # she may reach out on her own
            except Exception as e:
                log_event(f"self error: {e!r}")

    self_task = asyncio.create_task(self_watch())

    async def finalize_unsynthesized():
        """Heal books read to the end but never stamped 'finished' (a synthesis that errored):
        run the finalize pass so they complete on their own, no magic phrase required. Reading
        sessions spawn through the pipeline's task manager, so wait for it to come up first —
        retry a few times rather than guess a fixed delay."""
        for _ in range(6):
            await asyncio.sleep(10)
            stuck = book_memory.unfinalized_books()
            if not stuck:
                return  # nothing (or already healed on a previous pass)
            try:
                b = stuck[0]
                log_event(f"reading: auto-finalizing '{b['title']}' (fully read, no synthesis)")
                res = await reading.start_session(b["id"], trigger="resume")
                if res.get("success") and reading._task:
                    await reading._task   # let the synthesis land, then loop for any others
            except Exception as e:
                if "TaskManager is not initialized" in str(e):
                    continue  # pipeline not up yet — wait and retry
                log_event(f"auto-finalize error: {e!r}")
                return

    finalize_task = asyncio.create_task(finalize_unsynthesized())

    # --- avatar watchdog: respawn her body if the renderer dies ----------------
    async def avatar_watchdog():
        spawns = []
        while True:
            await asyncio.sleep(5)
            if avatar.connected:
                continue
            now = time.monotonic()
            spawns = [t for t in spawns if now - t < 600]
            if len(spawns) >= 3:
                continue  # crash-looping (broken GPU driver?) — stop thrashing
            spawns.append(now)
            await avatar.ensure_running()

    avatar_watchdog_task = asyncio.create_task(avatar_watchdog())

    # --- run ---------------------------------------------------------------------
    if not ambient:
        await worker.queue_frames([LLMRunFrame()])

    print("Aiva is ready." + (" (ambient mode)" if ambient else " Just talk."))
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    try:
        _play_ready_chime()
        await runner.run()
    finally:
        for t in (watcher, idle_task, flush_task, sleep_face_task, avatar_watchdog_task,
                  library_task, curiosity_task, self_task, finalize_task):
            if t:
                t.cancel()
        if wake_listener:
            wake_listener.stop()
        await task_runner.shutdown()  # in-flight tasks -> interrupted (persisted)
        await reading.shutdown()      # in-flight reading -> interrupted (persisted)

        # her diary: one private entry digesting how the session felt (never spoken)
        try:
            _recent = "\n".join(
                f"{m.get('role')}: {str(m.get('content'))[:200]}"
                for m in context.get_messages()[-8:]
                if m.get("role") in ("user", "assistant") and m.get("content"))
            await asyncio.wait_for(self_engine.tick(context=_recent, force_journal=True), timeout=25)
        except Exception as e:
            log_event(f"self journal (shutdown) skipped: {e!r}")

        # persist the session and distill long-term facts from it
        messages = context.get_messages()
        memory.save_new_messages(session_id, messages)
        full_transcript = memory.session_messages(session_id) or messages
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            added = await asyncio.wait_for(
                memory.extract_facts(client, llm_model, full_transcript), timeout=30
            )
            if added:
                print(f"Remembered {added} new fact(s) about you")
        except Exception as e:
            print(f"Fact extraction skipped: {e}")
        try:
            noted = await asyncio.wait_for(
                task_memory.extract_feedback(oai, llm_model, task_runner.session_task_ids,
                                             messages, remember=memory.remember), timeout=15)
            if noted:
                print(f"Noted feedback on {noted} task(s)")
        except Exception as e:
            print(f"Task feedback pass skipped: {e}")

        log_event("shutdown" + (" (restarting)" if restart_state["requested"] else ""))
        _print_usage_summary(mic, usage, llm_model)
        terminals.shutdown()
        cat.stop()  # saves its state; her pet leaves with her body
        memory.close()
        if restart_state["requested"]:
            await avatar.close()  # socket only: her body stays up, the next process adopts it
        else:
            await avatar.quit()   # her body leaves with her

    if restart_state["requested"]:
        open(_RESTART_MARKER, "w").close()  # tells the next boot "you're back", not "hello"
        if os.getenv("AIVA_RESTART_LOOP") != "1":
            _spawn_detached_relauncher()  # not under start_aiva.bat's loop — relaunch ourselves
        print("Restarting Aiva...")
        sys.exit(42)  # start_aiva.bat relaunches on this exit code


if __name__ == "__main__":
    if "--list-devices" in sys.argv:
        list_audio_devices()
    else:
        asyncio.run(main())
