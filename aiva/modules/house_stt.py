"""The house's speech-to-text and turn detection, picked by `AIVA_STT`.

  flux (default) — Deepgram Flux (`/v2/listen`, model `flux-general-en`). The STT itself decides
                   when he started and finished a turn (StartOfTurn / EndOfTurn with a confidence
                   threshold), so the stage aggregator runs NO local VAD and NO smart-turn model:
                   `ExternalUserTurnStrategies` resolve Flux's proposed start/stop into the turn.
                   One final `TranscriptionFrame` per turn, at EndOfTurn (since pipecat 1.13 also an
                   interim per `Update`, which `ActivityWatcher` drops during a wake check and while
                   asleep, as it did for nova) — the in-turn `Update` events keep the idle timer awake and, via `on_turn_info`,
                   feed the backchannel pause tracker in `modules/backchannel.py`). With
                   `AIVA_SPECULATE=1` Flux's eager end of turn is on and the stage runs
                   `EagerUserTurnStrategies`: the sister he's talking to prepares (and holds) a reply
                   on Flux's guess, released only if the turn ends on the same words (see
                   `house_stage.StageRouter`, `Floor.speculate`, `HouseLink.prepare`). Off by default:
                   it costs extra model calls when he goes on talking.
  nova           — the previous path: Deepgram nova-3 (`/v1/listen`) streaming interims + finals,
                   Silero VAD + LocalSmartTurnAnalyzerV3 in the stage aggregator.

Ambient sleep DROPS mic audio entirely. Nova survives that on Deepgram's free `KeepAlive` message;
Flux has no keepalive (pipecat's generic one streams silence, which is billed), and Deepgram closes
an idle socket on its own. So `HouseFluxSTTService` closes the socket itself after
`AIVA_FLUX_IDLE_CLOSE_S` seconds without audio and reopens it on the next audio frame — the wake
word's pre-roll waits in the STT's input queue during the ~0.5 s handshake and then streams in
order, so the transcript the wake check needs still starts with the wake phrase.

`STTConfig.from_env()` is pure; the pipecat pieces are built by `build_stt()` / `stage_params()`.
"""

import asyncio
import os
import time
from dataclasses import dataclass

ENGINES = ("flux", "nova")
FLUX_MODEL = "flux-general-en"
SAMPLE_RATE = 16000              # MicController always hands STT 16 kHz mono int16
# Deepgram's accepted ranges for the Flux knobs (outside them the connection is refused silently)
EOT_THRESHOLD_RANGE = (0.5, 0.9)
EOT_TIMEOUT_MS_RANGE = (500, 10000)


def _num(env, key, default, cast, lo=None, hi=None):
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        v = cast(raw)
    except ValueError:
        return default
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        return default
    return v


@dataclass(frozen=True)
class STTConfig:
    engine: str = "flux"
    eot_threshold: float = 0.7          # Flux: end-of-turn confidence needed to close a turn
    eot_timeout_ms: int = 5000          # Flux: close the turn this long after speech regardless
    idle_close_s: float = 3.0           # Flux: close the socket after this long with no audio (0=never)
    speculate: bool = False             # Flux: answer the eager end of turn (held until confirmed)
    note: str = ""                      # why a value fell back (shown at boot)

    @property
    def flux(self) -> bool:
        return self.engine == "flux"

    @classmethod
    def from_env(cls, env=None) -> "STTConfig":
        env = os.environ if env is None else env
        raw = (env.get("AIVA_STT") or "").strip().lower()
        engine, note = (raw or "flux"), ""
        if engine not in ENGINES:
            note = f"AIVA_STT={raw!r} unknown, using flux"
            engine = "flux"
        return cls(
            engine=engine,
            eot_threshold=_num(env, "AIVA_FLUX_EOT_THRESHOLD", 0.7, float, *EOT_THRESHOLD_RANGE),
            eot_timeout_ms=_num(env, "AIVA_FLUX_EOT_TIMEOUT_MS", 5000, int, *EOT_TIMEOUT_MS_RANGE),
            idle_close_s=_num(env, "AIVA_FLUX_IDLE_CLOSE_S", 3.0, float, 0.0, 60.0),
            speculate=engine == "flux" and (env.get("AIVA_SPECULATE") or "").strip().lower() in
            ("1", "true", "yes", "on"),
            note=note,
        )

    def describe(self) -> str:
        if self.flux:
            return (f"Deepgram Flux ({FLUX_MODEL}), turns by Flux: eot_threshold={self.eot_threshold}, "
                    f"eot_timeout={self.eot_timeout_ms} ms"
                    + (", answering sooner on its eager end of turn" if self.speculate else ""))
        return "Deepgram nova-3, turns by Silero VAD + smart-turn v3"


def idle_close_due(*, active: bool, user_speaking: bool, idle_s: float, close_after_s: float) -> bool:
    """PURE: should the Flux socket be closed now? Only an open socket that has had no audio for
    `close_after_s` — and not mid-turn (Flux closes a dangling turn itself on the watchdog's
    silence; give it up to 5x longer before forcing it)."""
    if not active or close_after_s <= 0:
        return False
    if user_speaking:
        return idle_s >= max(close_after_s * 5, 15.0)
    return idle_s >= close_after_s


def _flux_class():
    from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
    from pipecat.services.deepgram.flux.stt_base import FluxConnectionNotConfirmedError
    from pipecat.utils.errors import ErrorCategory

    class HouseFluxSTTService(DeepgramFluxSTTService):
        """Flux with an idle close + lazy reopen, so ambient sleep neither bills nor kills it."""

        REOPEN_BACKOFF_S = 2.0

        def __init__(self, *, idle_close_s: float = 3.0, log=print, **kwargs):
            super().__init__(**kwargs)
            self._idle_close_s = float(idle_close_s)
            self._house_log = log
            self._last_audio_at = time.monotonic()
            self._idle_task = None
            self._conn_lock = asyncio.Lock()
            self._reopen_after = 0.0
            self._idle_closed = False
            # (event, transcript, end_of_turn_confidence) for every TurnInfo — the backchannel
            # tracker's pause signal. Sync, cheap, never allowed to break the receive loop.
            self.on_turn_info = None

        async def _handle_turn_info(self, data):
            await super()._handle_turn_info(data)
            hook = self.on_turn_info
            if hook is not None:
                try:
                    hook(data.get("event"), data.get("transcript") or "", data.get("end_of_turn_confidence"))
                except Exception as e:
                    self._house_log(f"stt: turn-info hook failed: {e!r}")

        def _classify_error(self, exception):
            # Upstream calls an unconfirmed connection INVALID_REQUEST (permanent → the ear goes
            # deaf for good). The house reopens many times a day, so a slow handshake must stay a
            # transient failure; the settings are range-checked in STTConfig instead.
            if isinstance(exception, FluxConnectionNotConfirmedError):
                return ErrorCategory.CONNECTIVITY
            return super()._classify_error(exception)

        async def setup(self, setup):
            await super().setup(setup)
            if self._idle_close_s > 0 and self._idle_task is None:
                self._idle_task = self.create_task(self._idle_handler(), f"{self}::idle_close")

        async def _stop_idle(self):
            if self._idle_task is not None:
                await self.cancel_task(self._idle_task)
                self._idle_task = None

        async def stop(self, frame):
            await self._stop_idle()
            await super().stop(frame)

        async def cancel(self, frame):
            await self._stop_idle()
            await super().cancel(frame)

        async def cleanup(self):
            await self._stop_idle()
            await super().cleanup()

        async def _idle_handler(self):
            while True:
                await asyncio.sleep(0.5)
                try:
                    async with self._conn_lock:
                        if idle_close_due(active=self._transport_is_active(),
                                          user_speaking=self._user_is_speaking,
                                          idle_s=time.monotonic() - self._last_audio_at,
                                          close_after_s=self._idle_close_s):
                            self._house_log("stt: no audio (asleep) — closing the Flux socket")
                            await self._disconnect()
                            self._user_is_speaking = False
                            self._idle_closed = True
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self._house_log(f"stt: idle close failed: {e!r}")

        async def run_stt(self, audio: bytes):
            self._last_audio_at = time.monotonic()
            if not self._transport_is_active():
                now = time.monotonic()
                if now < self._reopen_after:
                    yield None                      # still backing off: this chunk is lost
                    return
                async with self._conn_lock:
                    if not self._transport_is_active():
                        why = "audio again" if self._idle_closed else "socket was down"
                        await self._disconnect()    # clears a dead receive task / closed socket
                        await self._connect()
                        if self._transport_is_active():
                            self._house_log(f"stt: Flux socket reopened ({why})")
                            self._idle_closed = False
                            self._reopen_after = 0.0
                        else:
                            self._house_log("stt: Flux reopen failed — retrying shortly")
                            self._reopen_after = time.monotonic() + self.REOPEN_BACKOFF_S
                            yield None
                            return
            async for f in super().run_stt(audio):
                yield f

    return HouseFluxSTTService


def build_stt(cfg: STTConfig, *, api_key: str, keyterms, log=print):
    """The house's STT service for `cfg.engine`. Constructing it never connects."""
    keyterms = list(keyterms or [])
    if cfg.flux:
        cls = _flux_class()
        return cls(
            api_key=api_key,
            sample_rate=SAMPLE_RATE,                # Flux connects at setup(), before any StartFrame
            should_interrupt=False,                 # nothing in the house to interrupt (F10 is the interrupt)
            enable_eager_end_of_turn=cfg.speculate,  # pipecat applies Flux's default eager threshold
            idle_close_s=cfg.idle_close_s,
            log=log,
            settings=cls.Settings(model=FLUX_MODEL, eot_threshold=cfg.eot_threshold,
                                  eot_timeout_ms=cfg.eot_timeout_ms, keyterm=keyterms),
        )
    from pipecat.services.deepgram.stt import DeepgramSTTService
    return DeepgramSTTService(
        api_key=api_key,
        settings=DeepgramSTTService.Settings(model="nova-3", language="en-US", smart_format=True,
                                             profanity_filter=False, keyterm=keyterms))


def stage_params(cfg: STTConfig, mute_strategies):
    """`LLMUserAggregatorParams` for the stage aggregator — the house's one turn detector.

    flux: no VAD at all. Flux's proposed start/stop (and its one final transcript) drive the turn
    through `ExternalUserTurnStrategies`; the Floor mute still suppresses those frames and the
    transcript while a sister holds an audible turn, exactly as it suppressed VAD frames before.
    nova: Silero VAD starts the turn, smart-turn v3 ends it (unchanged)."""
    from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
    if cfg.flux:
        from pipecat.turns.user_turn_strategies import EagerUserTurnStrategies, ExternalUserTurnStrategies
        strategies = (EagerUserTurnStrategies(enable_interruptions=False) if cfg.speculate
                      else ExternalUserTurnStrategies(enable_interruptions=False))
        return LLMUserAggregatorParams(
            vad_analyzer=None,
            user_mute_strategies=list(mute_strategies),
            user_turn_strategies=strategies,
        )
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
        TurnAnalyzerUserTurnStopStrategy,
    )
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),
        user_mute_strategies=list(mute_strategies),
        user_turn_strategies=UserTurnStrategies(
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]),
    )
