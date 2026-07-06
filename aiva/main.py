"""Aiva — streaming voice pipeline (pipecat 1.x).

mic -> Deepgram streaming STT -> Silero VAD + local smart-turn (in the user
aggregator) -> LLM (native tool calling, streamed) -> ElevenLabs Flash TTS
-> speaker.

Hotkeys: F9 mute mic, F10 interrupt Aiva, Esc quit.

Optional ambient mode (AIVA_WAKE_WORD=1): the mic sleeps until the wake word
(openWakeWord, local) and dozes off again after AIVA_IDLE_TIMEOUT seconds of
silence. While asleep no audio is sent to Deepgram, so idle time is free.

Run `python main.py --list-devices` to see audio device indices for
AIVA_INPUT_DEVICE_INDEX / AIVA_OUTPUT_DEVICE_INDEX (VB-Cable routing).
"""

import asyncio
import os
import sys
import threading
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# hotkey/expression names can be CJK; don't let a console print crash a handler
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionWorkerFrame,
    LLMRunFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
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

from modules.memory import Memory
from modules.persona import build_system_prompt
from modules.terminals import TerminalManager
from modules.tools import TOOL_SCHEMAS, register_tools
from modules.vtube_studio import VTubeStudio


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

    def __init__(self, ambient: bool, bot_cooldown_secs: float = 0.5):
        super().__init__()
        self.muted = False
        self.awake = not ambient
        self.last_activity = time.monotonic()
        self.stt_seconds = 0.0
        self._ambient = ambient
        self._bot_cooldown_secs = bot_cooldown_secs
        self._bot_speaking = False
        self._bot_stopped_at = 0.0
        # The transport must capture at the mic's NATIVE rate (Windows
        # delivers silence/garbage otherwise), but Silero VAD and friends
        # require 16 kHz — so this processor resamples the stream and
        # re-stamps the StartFrame before anything downstream sees it.
        self._resampler = SOXRStreamAudioResampler()
        self._capture_rate = None

    def toggle_mute(self):
        self.muted = not self.muted
        print(f"\n[MIC {'MUTED' if self.muted else 'LIVE'}] (F9 to toggle)")

    def wake(self):
        self.last_activity = time.monotonic()
        if not self.awake:
            self.awake = True
            print("\n[AWAKE] Aiva is listening")

    def poke(self):
        """Conversation activity happened — keep her awake."""
        self.last_activity = time.monotonic()

    def sleep(self):
        if self.awake and self._ambient:
            self.awake = False
            print("\n[IDLE] say the wake word to talk to Aiva")

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
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped_at = time.monotonic()
            self.last_activity = time.monotonic()

        if isinstance(frame, InputAudioRawFrame):
            if not self.awake:
                return  # dropped: nothing streams to Deepgram while asleep
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


class ActivityWatcher(FrameProcessor):
    """Sits after STT: any transcription (interim or final) means the user is
    talking to Aiva, so the idle timer resets — including when they say her
    name mid-conversation."""

    def __init__(self, mic: "MicController"):
        super().__init__()
        self._mic = mic

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            self._mic.poke()
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


class WakeWordListener:
    """Local wake-word detection (openWakeWord) on its own mic stream.

    Runs in a thread; never sends audio anywhere. Fires on_wake() when the
    wake word scores above threshold while Aiva is asleep.
    """

    def __init__(self, model, on_wake, device_index=None, threshold=0.5):
        self._model = model
        self._on_wake = on_wake
        self._device_index = device_index
        self._threshold = threshold
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
        print(f"Wake word armed ({self._model}) on '{info['name']}' @ {native} Hz")
        try:
            while not self._stop.is_set():
                chunk = stream.read(frames, exception_on_overflow=False)
                audio = np.frombuffer(chunk, dtype=np.int16)
                if native != 16000:
                    audio = resample_poly(audio.astype(np.float32), up, down)
                    audio = np.clip(audio, -32768, 32767).astype(np.int16)
                scores = oww.predict(audio)
                if scores and max(scores.values()) >= self._threshold:
                    oww.reset()
                    self._on_wake()
                    time.sleep(2)  # debounce
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


def _print_usage_summary(mic: MicController, usage: UsageTracker):
    """Session cost estimate (rates: mid-2026, see plan)."""
    stt_min = mic.stt_seconds / 60
    stt_cost = stt_min * 0.0077
    llm_cost = usage.prompt_tokens * 0.40 / 1e6 + usage.completion_tokens * 1.60 / 1e6
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

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-3",
            language="en-US",
            smart_format=True,
        ),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model=llm_model,
            temperature=0.7,
            max_tokens=300,
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
    mic = MicController(ambient=ambient)
    terminals = TerminalManager()

    vtube = VTubeStudio()
    await vtube.connect()  # non-fatal if VTube Studio isn't running
    register_tools(llm, vtube, mic, terminals)

    avatar_hotkeys = [h["name"] for h in await vtube.get_hotkeys()]
    if avatar_hotkeys:
        print(f"Avatar hotkeys available: {len(avatar_hotkeys)}")

    # --- memory + context ----------------------------------------------------
    memory = Memory()
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    facts = memory.load_facts()
    if facts:
        print(f"Loaded {len(facts)} remembered facts")

    context = LLMContext(
        messages=[
            {"role": "system", "content": build_system_prompt(facts, avatar_hotkeys)},
            {"role": "system", "content": "Introduce yourself very briefly and greet the user."},
        ],
        tools=TOOL_SCHEMAS,
    )
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

    pipeline = Pipeline([
        transport.input(),
        mic,
        stt,
        ActivityWatcher(mic),
        user_aggregator,
        llm,
        tts,
        usage,
        transport.output(),
        assistant_aggregator,
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    async def announce(text: str):
        """Voice path for background-job news: inject a system note and
        trigger a turn so Aiva proactively speaks up."""
        mic.wake()  # eyes open, ears on — she has news
        context.add_message({"role": "system", "content": text})
        await worker.queue_frames([LLMRunFrame()])

    terminals._announce = announce

    # --- hotkeys ---------------------------------------------------------------
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    try:
        import keyboard

        def interrupt_bot():
            print("\n[F10] Interrupting Aiva")
            asyncio.run_coroutine_threadsafe(
                worker.queue_frames([InterruptionWorkerFrame()]), loop
            )

        keyboard.add_hotkey("f9", lambda: loop.call_soon_threadsafe(mic.toggle_mute))
        keyboard.add_hotkey("f10", interrupt_bot)
        keyboard.add_hotkey("esc", lambda: loop.call_soon_threadsafe(stop_event.set))
        print("Hotkeys: F9 = mute/unmute mic, F10 = interrupt Aiva, Esc = quit")
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
        wake_listener = WakeWordListener(
            model=os.getenv("AIVA_WAKE_MODEL", "hey_jarvis"),
            on_wake=lambda: loop.call_soon_threadsafe(mic.wake),
            device_index=_device_index("AIVA_INPUT_DEVICE_INDEX", "AIVA_INPUT_DEVICE_NAME", False),
            threshold=float(os.getenv("AIVA_WAKE_THRESHOLD", "0.5")),
        )
        wake_listener.start()

        async def idle_watch():
            while True:
                await asyncio.sleep(2)
                # never doze off mid-sentence or mid-conversation
                if mic.awake and not mic.bot_speaking and mic.idle_for() > idle_timeout:
                    mic.sleep()

        async def sleep_face():
            """Keep the avatar's eyes closed while she's asleep (injections
            expire in ~1s, so re-send; when awake, auto-blink resumes)."""
            while True:
                await asyncio.sleep(0.6)
                if not mic.awake and vtube.connected:
                    await vtube.set_eyes(0.0)

        sleep_face_task = asyncio.create_task(sleep_face())
        idle_task = asyncio.create_task(idle_watch())
        print("[IDLE] say the wake word to talk to Aiva")

    # --- periodic transcript persistence ----------------------------------------
    async def flush_transcripts():
        while True:
            await asyncio.sleep(60)
            memory.save_new_messages(session_id, context.get_messages())

    flush_task = asyncio.create_task(flush_transcripts())

    # --- run ---------------------------------------------------------------------
    if not ambient:
        await worker.queue_frames([LLMRunFrame()])

    print("Aiva is ready." + (" (ambient mode)" if ambient else " Just talk."))
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        for t in (watcher, idle_task, flush_task, sleep_face_task):
            if t:
                t.cancel()
        if wake_listener:
            wake_listener.stop()

        # persist the session and distill long-term facts from it
        messages = context.get_messages()
        memory.save_new_messages(session_id, messages)
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            added = await asyncio.wait_for(
                memory.extract_facts(client, llm_model, messages), timeout=30
            )
            if added:
                print(f"Remembered {added} new fact(s) about you")
        except Exception as e:
            print(f"Fact extraction skipped: {e}")

        _print_usage_summary(mic, usage)
        terminals.shutdown()
        memory.close()
        await vtube.close()


if __name__ == "__main__":
    if "--list-devices" in sys.argv:
        list_audio_devices()
    else:
        asyncio.run(main())
