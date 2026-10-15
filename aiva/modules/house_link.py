"""Character side of the house: the pipecat processors that let a sister's pipeline run WITHOUT a
microphone, driven by turns the house grants her over the socket.

  HouseLink → her LLM (own tools) → PrivacyGate → her TTS → LipSyncTap(her avatar) → MindWatcher
  → UsageTracker → transport.output() (audio out only) → assistant aggregator

- `HouseLink` is first in line. It forwards every frame untouched; its job is `run_turn()`: put
  the granted turn's lines into HER context and push `LLMContextFrame(ctx)` downstream — the
  OpenAI service runs on any LLMContextFrame it sees (pipecat 1.5 `services/openai/base_llm.py`
  ~:559-573), so no user aggregator, VAD or transcript is needed in her process. It also turns her
  transport's Bot*SpeakingFrames (travelling upstream) into `on_speaking(on)` reports for the house,
  and `interrupt()` injects an `InterruptionFrame` from below where a muted user aggregator could
  swallow it (there is none here anyway).
- `PrivacyGate` sits between the LLM and the TTS. While a granted turn is NOT audible (sister talk
  in private mode) it stamps `skip_tts=True` on `LLMFullResponseStartFrame`, every `TextFrame` and
  `LLMFullResponseEndFrame`; the TTS service forwards such frames untouched (no audio, no
  `TTSUsageMetricsData`, `services/tts_service.py` ~:688-692) while the assistant aggregator still
  commits the text. The latch spans the whole turn including tool re-runs. It fails CLOSED: a
  response that starts with no latched turn is treated as text-only and logged loudly.
- `MindWatcher` reports `response start/end` and `tool ±1` (what ProcessingWatcher used to feed
  the mic with) and keeps a local `processing` predicate for the task_busy closures.
- `UsageTracker` sums LLM tokens and TTS characters from MetricsFrames and hands TTS characters to
  a callback (ServiceLimits.record) so ElevenLabs spend is finally counted per character.
- `LipSyncTap` is unchanged from the single-process days.

Everything reporting to the house is fire-and-forget through callables injected by main.py; a
missing house never stalls the pipeline.
"""

import asyncio
import inspect
import math
import time

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


async def _call(fn, *args, **kwargs):
    """Call a sync-or-async callback; swallow its errors (reporting must never stall speech)."""
    if fn is None:
        return None
    try:
        res = fn(*args, **kwargs)
        if inspect.isawaitable(res):
            res = await res
        return res
    except Exception:
        return None


class HouseLink(FrameProcessor):
    def __init__(self, context, *, on_speaking=None, log=print, name="Aiva"):
        super().__init__()
        self._ctx = context
        self._on_speaking = on_speaking     # (turn_id, on: bool)
        self._log = log
        self._name = name
        self._started = False
        self.current_turn = None            # the granted turn dict while it runs
        # BotStoppedSpeaking can arrive AFTER the assistant aggregator has closed the turn (the
        # transport reports audio end last), so speaking reports fall back to the last granted id
        self.last_turn_id = None
        self._speaking_turn_id = None

    # --- driven by the house -----------------------------------------------------------------

    async def run_turn(self, turn: dict, *, user_text=None, cue=None, notes=()):
        """Put the turn into her context and run the LLM. `notes` are system lines to add first
        (speaker identity etc.), `user_text` the line addressed to her (role user), `cue` an
        ephemeral system stage direction (removed by identity after the reply)."""
        if not self._started:
            self._log("house link: turn arrived before the pipeline started — dropped")
            return False
        self.current_turn = dict(turn)
        self.last_turn_id = self.current_turn.get("id")
        for n in notes:
            if n:
                self._ctx.add_message({"role": "system", "content": n})
        if user_text:
            self._ctx.add_message({"role": "user", "content": user_text})
        if cue:
            cue_msg = {"role": "system", "content": cue}
            self.current_turn["_cue"] = cue_msg
            self._ctx.add_message(cue_msg)
        await self.push_frame(LLMContextFrame(self._ctx), FrameDirection.DOWNSTREAM)
        return True

    def finish_turn(self):
        """Drop the ephemeral cue note (by identity) once the reply is committed."""
        t, self.current_turn = self.current_turn, None
        cue = t.get("_cue") if t else None
        if cue is not None:
            msgs = self._ctx.get_messages()
            kept = [m for m in msgs if m is not cue]
            if len(kept) != len(msgs):
                self._ctx.set_messages(kept)
        return t

    def add_overheard(self, line: str, *, notes=()):
        """A line she overheard (role user, bracketed) — stored, never run."""
        for n in notes:
            if n:
                self._ctx.add_message({"role": "system", "content": n})
        if line:
            self._ctx.add_message({"role": "user", "content": line})

    async def interrupt(self):
        if self._started:
            await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    # --- pipeline ----------------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        from pipecat.frames.frames import StartFrame
        if isinstance(frame, StartFrame):
            self._started = True
        elif isinstance(frame, BotStartedSpeakingFrame):
            turn_id = self.current_turn.get("id") if self.current_turn else self.last_turn_id
            self._speaking_turn_id = turn_id
            await _call(self._on_speaking, turn_id, True)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            turn_id = self._speaking_turn_id
            if turn_id is None:
                turn_id = self.current_turn.get("id") if self.current_turn else self.last_turn_id
            self._speaking_turn_id = None
            await _call(self._on_speaking, turn_id, False)
        await self.push_frame(frame, direction)


class PrivacyGate(FrameProcessor):
    """Also the one place that knows when a granted turn's LLM work is DONE and what she said.
    It sits straight after the LLM, so it sees Start / text / tool call / End in true order. After
    the TTS that order is gone: the ElevenLabs service holds each End until its audio finishes and
    drops the End of a response it already closed, so the assistant aggregator's per-response
    "turn stopped" can land mid-turn or never — deciding the turn's end there lost every reply that
    came after a tool. A response that called a tool is followed by the LLM's re-run inside the same
    turn; the first response that ends without one ends the turn → `on_turn_done(text)` with the
    text of every response in it (pre-tool words included)."""

    def __init__(self, name="Aiva", log=print, on_text_only=None, on_turn_done=None):
        super().__init__()
        self._name = name
        self._log = log
        self._on_text_only = on_text_only   # (addressee, text) when a text-only reply completes
        self._on_turn_done = on_turn_done   # (text) when the latched turn's last response ends
        self._armed = False                 # a turn is latched
        self._audible = False
        self._addressee = "him"
        self._buf = []
        self._said = []                     # every text chunk of the latched turn
        self._tool_called = False           # the current response called a tool → a re-run follows

    def set_audible(self, audible: bool, addressee: str = "him"):
        self._armed = True
        self._audible = bool(audible)
        self._addressee = addressee
        self._buf = []
        self._said = []
        self._tool_called = False

    def release(self):
        self._armed = False
        self._audible = False
        self._buf = []
        self._said = []
        self._tool_called = False

    def take_said(self) -> str:
        """What the latched turn has said so far (for closing a turn whose last response never came)."""
        text = "".join(self._said).strip()
        self._said = []
        return text

    @property
    def audible(self) -> bool:
        return self._armed and self._audible

    def _silent(self, frame) -> bool:
        if self.audible:
            return False
        if not self._armed and isinstance(frame, LLMFullResponseStartFrame):
            self._log(f"{self._name}: response started with no granted turn — kept text-only")
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        turn_done = None
        if direction == FrameDirection.DOWNSTREAM and self._armed:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._tool_called = False
            elif isinstance(frame, (FunctionCallsStartedFrame, FunctionCallInProgressFrame)):
                # FunctionCallsStartedFrame is pushed BEFORE the response's end frame; the
                # in-progress one comes from the call's own task and often lands after it (parallel
                # tools), which used to end the turn early and lose the post-tool reply
                self._tool_called = True
            elif isinstance(frame, TextFrame) and not isinstance(
                    frame, (TranscriptionFrame, InterimTranscriptionFrame)):
                self._said.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame) and not self._tool_called:
                turn_done = "".join(self._said).strip()
        if direction == FrameDirection.DOWNSTREAM and isinstance(
                frame, (LLMFullResponseStartFrame, TextFrame, LLMFullResponseEndFrame)) \
                and not isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if self._silent(frame):
                frame.skip_tts = True
                if isinstance(frame, TextFrame):
                    self._buf.append(frame.text)
                elif isinstance(frame, LLMFullResponseEndFrame):
                    text = "".join(self._buf).strip()
                    self._buf = []
                    if text:
                        self._log(f"{self._name} (text-only) -> {self._addressee}: {text}")
                        print(f"\n[{self._name} -> {self._addressee}, text-only] {text}")
                        await _call(self._on_text_only, self._addressee, text)
        if turn_done is not None:
            await _call(self._on_turn_done, turn_done)     # frames already stamped; may release()
        await self.push_frame(frame, direction)


class MindWatcher(FrameProcessor):
    def __init__(self, on_response=None, on_tool=None, log=print):
        super().__init__()
        self._on_response = on_response     # (phase: "start"|"end", interrupted: bool)
        self._on_tool = on_tool             # (delta: +1|-1, tool_call_id)
        self._log = log
        self._thinking_since = 0.0
        self._tools_in_flight = {}
        self.last_activity = time.monotonic()

    @property
    def processing(self) -> bool:
        now = time.monotonic()
        self._tools_in_flight = {k: t for k, t in self._tools_in_flight.items() if now - t < 600}
        thinking = self._thinking_since and now - self._thinking_since < 120
        return bool(thinking) or bool(self._tools_in_flight)

    def idle_for(self) -> float:
        return time.monotonic() - self.last_activity

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._thinking_since = time.monotonic()
            self.last_activity = time.monotonic()
            await _call(self._on_response, "start", False)
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._thinking_since = 0.0
            self.last_activity = time.monotonic()
            await _call(self._on_response, "end", False)
        elif isinstance(frame, InterruptionFrame):
            # an interruption swallows the queued end frame — treat it as one
            self._thinking_since = 0.0
            self._tools_in_flight.clear()
            self.last_activity = time.monotonic()
            await _call(self._on_response, "end", True)
        elif isinstance(frame, FunctionCallsStartedFrame) and direction == FrameDirection.DOWNSTREAM:
            # reported here, ahead of the response's end, so the house knows a re-run follows
            # (the per-call in-progress frame can arrive after the end — see PrivacyGate)
            for fc in frame.function_calls or ():
                await self._tool_started(fc.tool_call_id)
        elif isinstance(frame, FunctionCallInProgressFrame):
            await self._tool_started(frame.tool_call_id)
        elif isinstance(frame, (FunctionCallResultFrame, FunctionCallCancelFrame)):
            self.last_activity = time.monotonic()
            if self._tools_in_flight.pop(frame.tool_call_id, None) is not None:
                await _call(self._on_tool, -1, frame.tool_call_id)
        await self.push_frame(frame, direction)

    async def _tool_started(self, tool_call_id):
        self.last_activity = time.monotonic()
        if tool_call_id in self._tools_in_flight:
            return                              # already reported (started + in-progress)
        self._tools_in_flight[tool_call_id] = time.monotonic()
        await _call(self._on_tool, +1, tool_call_id)


class UsageTracker(FrameProcessor):
    """Collects LLM token and TTS character usage from metrics frames; `on_tts(chars)` is called
    per TTS utterance (one sqlite write each — metrics frames are per utterance, not per audio
    frame). Errors in the callback never stall speech."""

    def __init__(self, on_tts=None):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tts_chars = 0
        self._on_tts = on_tts

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, MetricsFrame):
            for data in frame.data:
                if isinstance(data, LLMUsageMetricsData):
                    self.prompt_tokens += data.value.prompt_tokens or 0
                    self.completion_tokens += data.value.completion_tokens or 0
                elif isinstance(data, TTSUsageMetricsData):
                    n = data.value or 0
                    self.tts_chars += n
                    if n and self._on_tts:
                        await _call(self._on_tts, n)
        await self.push_frame(frame, direction)


class LipSyncTap(FrameProcessor):
    """Sits between TTS and the output transport, feeding the avatar renderer's mouth. TTS audio
    frames pass here FASTER than real-time (the transport does the pacing), so each chunk's RMS
    goes over with its duration and the renderer queues them, consuming at wall-clock rate once
    playback actually starts (BotStartedSpeakingFrame — emitted by the transport, so it passes
    this processor travelling upstream). Interruptions flush the queue.

    Everything here is sync fire-and-forget into the avatar socket: a dead or missing renderer
    costs nothing and can never stall the voice pipeline."""

    def __init__(self, avatar):
        super().__init__()
        self._avatar = avatar

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and frame.audio:
            import numpy as np

            samples = np.frombuffer(frame.audio, dtype=np.int16)
            if len(samples):
                # One measurement per ~40ms window, not per frame: TTS frames can be hundreds of
                # ms, and a single average erases the syllable pattern (the mouth would hang
                # half-open, motionless).
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
                    # brightness = spectral centroid mapped log-linearly onto 0..1: dark/rounded
                    # vowels (oo/oh, centroid ~500Hz) -> 0, bright/wide ones (ee/eh and
                    # sibilants, ~2.5kHz+) -> 1. The renderer turns it into ParamMouthForm.
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
