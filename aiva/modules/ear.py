"""The house's ear: everything between the microphone and the stage aggregator.

Moved out of main.py when the sisters got their own processes: the ONE mic, the self-hearing
gate, the two-stage wake verification and the per-turn audio tap live here, in the house. The
sisters' processes have no microphone at all.

  transport.input → MicController → UserAudioTap → Deepgram STT → ActivityWatcher → aggregator

- `MicController` gates mic audio right after the transport input. Three layers, all BEFORE STT:
  asleep (ambient) → audio DROPPED (Deepgram bills nothing); muted (F9) or a sister speaking
  (+ cooldown for the room's audio tail) → zero-filled so her voice is never transcribed; awake
  and quiet → passes, and billed STT seconds are counted. The "a sister is speaking" signal comes
  from the Floor (`set_speaking`) the moment an audible turn is granted — strictly before any audio
  can leave her speaker in another process — and from Bot*SpeakingFrames when this processor runs
  inside a single-process pipeline (kept for the tests and for compatibility).
- `ActivityWatcher` sits after STT: any transcription resets the idle timer and, while a wake is
  being verified, the first final transcript decides — a name in it (`name_check`) confirms, anything
  else rejects and is dropped before the LLM.
- `UserAudioTap` buffers the user's 16 kHz PCM for the current turn so the house can embed it once
  (speaker voiceprint, vocal affect) and forward the vectors to each sister.
"""

import os
import re
import time

from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class MicController(FrameProcessor):
    def __init__(self, ambient: bool, bot_cooldown_secs: float = 0.5, verify_secs: float = 5.0,
                 busy=None, log=print, label: str = "Aiva"):
        super().__init__()
        self.muted = False
        self.awake = not ambient
        self.last_activity = time.monotonic()
        self.last_user_final = None   # set on the user's final transcript; for turn-latency metric
        self.stt_seconds = 0.0
        self.wake_hint = None         # whose wake word fired last (her slug, "aiva"/"saga"), if any
        self._ambient = ambient
        self._bot_cooldown_secs = bot_cooldown_secs
        # Verified wake: the wake-word model alone is too trigger-happy (it fires on TV/game
        # speech with the same scores as a real "hey Aiva"), so a wake-word wake is PROVISIONAL
        # until Deepgram's transcript of the pre-roll audio actually contains a name. Until then
        # she is awake only in the sense that audio streams to STT: no [AWAKE] print, eyes stay
        # shut, and nothing reaches the LLM (ActivityWatcher gates transcripts). No name within
        # `verify_secs` -> silently back to sleep.
        self._verify_secs = verify_secs
        self._verifying = False
        self._verify_deadline = 0.0
        self._preroll = b""          # 16 kHz PCM injected ahead of live audio
        self._bot_speaking = False
        self._bot_stopped_at = 0.0
        self._thinking_since = 0.0    # monotonic; 0 = no LLM response in flight
        self._tools_in_flight = {}    # tool_call_id -> monotonic start time
        self._busy = busy             # optional extra "still working" predicate (the Floor)
        self._log = log
        self._label = label
        # The transport must capture at the mic's NATIVE rate (Windows delivers silence/garbage
        # otherwise), but Silero VAD and friends require 16 kHz — so this processor resamples
        # the stream and re-stamps the StartFrame before anything downstream sees it.
        self._resampler = SOXRStreamAudioResampler()
        self._capture_rate = None

    def toggle_mute(self):
        self.muted = not self.muted
        state = "MUTED — deafened, won't wake to your voice" if self.muted else "LIVE"
        print(f"\n[MIC {state}] (F9 to toggle)")
        self._log(f"mic {'muted (deafened)' if self.muted else 'unmuted'} (F9)")

    def wake(self, reason: str = "unspecified", preroll: bytes | None = None,
             verify: bool = False, hint: str | None = None):
        """Wake the house. `preroll` (16 kHz int16 PCM captured just before the wake word fired)
        is pushed to STT ahead of live audio so the transcript includes the wake phrase itself.
        With `verify` the wake is provisional until confirm_wake()/reject_wake() decide."""
        # F9 deafen: a wake-word wake carries a preroll (a sister's own announces don't) — while
        # muted the wake word can't wake her, so she truly can't hear you; the sisters can still
        # speak up themselves.
        if self.muted and (verify or preroll is not None):
            self._log(f"wake ignored — mic muted (F9): {reason}")
            return
        # a sister is speaking (or just stopped): the wake-word listener has its own mic stream and
        # its pre-roll would carry HER voice straight into STT past the self-hearing gate
        if (verify or preroll is not None) and self._zeroed():
            self._log(f"wake ignored — a sister is speaking: {reason}")
            return
        self.last_activity = time.monotonic()
        if hint:
            self.wake_hint = hint
        if preroll:
            self._preroll = preroll
        if not self.awake:
            self.awake = True
            if verify and self._ambient:
                self._verifying = True
                self._verify_deadline = time.monotonic() + self._verify_secs
                print("\n[WAKE?] heard something like the wake word — checking")
                self._log(f"wake candidate — {reason}, verifying transcript")
            else:
                print(f"\n[AWAKE] {self._label} is listening")
                self._log(f"AWAKE — {reason}")
        elif self._verifying and not verify:
            # an authoritative wake (background announce) arrived mid-check
            self.confirm_wake(reason)

    def confirm_wake(self, reason: str = ""):
        if self._verifying:
            self._verifying = False
            self.last_activity = time.monotonic()
            print(f"\n[AWAKE] {self._label} is listening")
            self._log(f"AWAKE — wake word confirmed ({reason})")

    def reject_wake(self, reason: str = ""):
        if self._verifying:
            self._verifying = False
            self._preroll = b""
            self.awake = False
            self.wake_hint = None
            print(f"\n[IDLE] false wake ignored — say the wake word to talk to {self._label}")
            self._log(f"wake rejected — {reason}")

    @property
    def verifying(self) -> bool:
        return self._verifying

    def verify_expired(self) -> bool:
        return self._verifying and time.monotonic() > self._verify_deadline

    @property
    def visibly_awake(self) -> bool:
        """Awake as far as the user (and the avatars) should be able to tell — a provisional wake
        still being verified doesn't count."""
        return self.awake and not self._verifying

    def poke(self):
        """Conversation activity happened — stay awake."""
        self.last_activity = time.monotonic()

    def sleep(self):
        if self.awake and self._ambient:
            self.awake = False
            self.wake_hint = None
            print(f"\n[IDLE] say the wake word to talk to {self._label}")
            self._log("asleep (idle timeout)")

    def set_thinking(self, on: bool):
        self._thinking_since = time.monotonic() if on else 0.0
        self.last_activity = time.monotonic()

    def tool_started(self, call_id):
        self._tools_in_flight[call_id] = time.monotonic()
        self.last_activity = time.monotonic()

    def tool_finished(self, call_id):
        self._tools_in_flight.pop(call_id, None)
        self.last_activity = time.monotonic()

    def set_speaking(self, on: bool):
        """The Floor's signal that a sister holds an audible turn (in another process). Same
        effect as the transport's Bot*SpeakingFrames in a single-process pipeline."""
        if on and not self._bot_speaking:
            self._bot_speaking = True
            self.last_activity = time.monotonic()
        elif not on and self._bot_speaking:
            self._bot_speaking = False
            self._bot_stopped_at = time.monotonic()
            self.last_activity = time.monotonic()

    def note_bot_stopped(self):
        self.set_speaking(False)

    @property
    def processing(self) -> bool:
        """True while a turn is still being worked on — generating a response or waiting on tool
        calls — so the idle timer can't doze the house off mid-task. Both signals expire (a
        dropped end/result frame must not pin her awake forever: awake means the mic streams to
        Deepgram and bills)."""
        now = time.monotonic()
        self._tools_in_flight = {
            k: t for k, t in self._tools_in_flight.items() if now - t < 600
        }
        thinking = self._thinking_since and now - self._thinking_since < 120
        busy = False
        if self._busy is not None:
            try:
                busy = bool(self._busy())
            except Exception:
                busy = False
        return bool(thinking) or bool(self._tools_in_flight) or busy

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
            self.set_speaking(True)
            if self.last_user_final is not None:   # felt latency: you stop -> she speaks
                from modules import metrics
                metrics.record("turn", "latency",
                               (time.monotonic() - self.last_user_final) * 1000.0)
                self.last_user_final = None
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.set_speaking(False)

        if isinstance(frame, InputAudioRawFrame):
            if not self.awake:
                return  # dropped: nothing streams to Deepgram while asleep
            if self._preroll:
                # the seconds BEFORE the wake fired (the wake phrase itself), already 16 kHz —
                # hand them to STT ahead of the live stream in normal-sized chunks so VAD
                # downstream sees nothing unusual
                preroll, self._preroll = self._preroll, b""
                step = 2 * 16000 * 40 // 1000  # 40 ms of int16 mono
                for i in range(0, len(preroll), step):
                    await self.push_frame(InputAudioRawFrame(
                        audio=preroll[i:i + step], sample_rate=16000,
                        num_channels=1), direction)
                self.stt_seconds += len(preroll) / (2 * 16000)
            audio = frame.audio
            # frame.sample_rate is the transport's TRUE capture rate (the StartFrame only
            # carries the pipeline's configured rate) — resample by what the frame actually is.
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


def mentions_any_of(text: str, variants, loose_regex=None) -> bool:
    """True if a transcript plausibly contains one of `variants` (word or phrase match) or a word
    matching the loose shape regex (e.g. Aiva's vowel(s)+v/w/f+a shape 'eyva', 'ayvah' that no
    common English word has, so TV dialogue doesn't sneak through)."""
    words = re.sub(r"[^a-z' ]", " ", text.lower()).split()
    joined = f" {' '.join(words)} "
    for v in variants:
        if (" " in v and f" {v} " in joined) or (" " not in v and v in words):
            return True
    if loose_regex:
        return any(re.fullmatch(loose_regex, w) for w in words)
    return False


class ActivityWatcher(FrameProcessor):
    """Sits after STT: any transcription (interim or final) means the user is talking, so the
    idle timer resets. Also the second stage of the wake word: while a wake is being verified,
    the first final transcript decides — a sister's name in it (`name_check`) confirms the wake
    and passes through; anything else (TV dialogue, a cough) is dropped before any LLM sees it
    and the house goes straight back to sleep. Late transcripts arriving after it fell asleep are
    dropped for the same reason."""

    def __init__(self, mic: MicController, name_check, on_user_final=None, log=print):
        super().__init__()
        self._mic = mic
        self._name_check = name_check           # (text) -> bool
        self._on_user_final = on_user_final     # async(text): per-turn hook before the LLM
        self._log = log

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and not (frame.text or "").strip():
            # Flux closes a turn of pure noise with an EMPTY final transcript: that is not words,
            # so it must neither reject a wake being verified nor start the latency clock
            return
        if self._mic.verifying:
            if isinstance(frame, TranscriptionFrame):
                if self._name_check(frame.text):
                    self._mic.confirm_wake(f"heard {frame.text!r}")
                else:
                    self._log(f"heard during wake check: {frame.text!r}")
                    self._mic.reject_wake(f"no name in {frame.text!r}")
                    return
            elif isinstance(frame, InterimTranscriptionFrame):
                return  # nothing downstream until the wake is confirmed
            elif self._mic.verify_expired():
                self._mic.reject_wake("no speech after the wake word")
        elif self._mic.ambient and not self._mic.awake and \
                isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            if isinstance(frame, TranscriptionFrame):
                self._log(f"heard while asleep (dropped): {frame.text!r}")
            return  # Deepgram flushing audio from before she dozed off
        if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            self._mic.poke()
        if isinstance(frame, TranscriptionFrame):
            self._log(f"heard: {frame.text!r}")
            self._mic.last_user_final = time.monotonic()   # start the turn-latency clock
            if self._on_user_final:
                try:
                    await self._on_user_final(frame.text)
                except Exception as e:
                    self._log(f"user-final hook error: {e!r}")
        await self.push_frame(frame, direction)


class UserAudioTap(FrameProcessor):
    """Buffers the user's 16 kHz PCM for the current turn so it can be embedded once (speaker
    voiceprint + vocal affect). Inspect, buffer, forward untouched. Sits right after
    MicController, so it sees frames with the sisters' own voices already zero-filled and asleep
    audio already dropped. The Floor calls `clear()` when it grants a turn."""

    def __init__(self, mic: MicController, max_secs: float = 12.0):
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

    def clear(self):
        self._buf.clear()

    def take_segment(self) -> bytes:
        pcm = bytes(self._buf)
        self._buf.clear()
        return pcm


# --- audio device helpers -----------------------------------------------------------------

def device_index(env_index, env_name, output):
    """Resolve an audio device from env: numeric index, or (more robust, indices shift when
    drivers change) a name substring like 'CABLE Input'."""
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
            # A device shows up once per host API and they are NOT equal. MME works for both
            # directions as long as capture runs at the device's NATIVE rate (16 kHz capture
            # silently dies) — output MME resamples anything. WASAPI capture is fine but its
            # output rejects non-mix-format rates, and DirectSound capture stutters. So: MME
            # everywhere, native input rate. (MME shared mode also mixes several output
            # streams — each sister's process plays through the same device.)
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


def native_rate(index):
    """Capture must run at the device's native rate: forcing 16 kHz makes Windows deliver
    silence (MME) or corrupted repeating buffers (DS)."""
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        if index is None:
            return int(pa.get_default_input_device_info()["defaultSampleRate"])
        return int(pa.get_device_info_by_index(index)["defaultSampleRate"])
    finally:
        pa.terminate()


def list_audio_devices():
    import pyaudio

    pa = pyaudio.PyAudio()
    print("index | in/out channels | name")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"{i:5d} | in:{info['maxInputChannels']:2d} out:{info['maxOutputChannels']:2d} | {info['name']}")
    pa.terminate()
