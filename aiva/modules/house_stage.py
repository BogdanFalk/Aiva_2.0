"""The end of the house pipeline: where a finished user turn leaves pipecat and enters the Floor.

  … → ActivityWatcher → LLMUserAggregator(stage_ctx; VAD + smart-turn + FloorUserMuteStrategy)
  → StageRouter

The stage aggregator is the ONE turn detector in the house. When it decides the user's turn is
over it adds the user message to `stage_ctx` and pushes `LLMContextFrame(stage_ctx)` (pipecat
1.5 `llm_response_universal.py` `push_aggregation`). There is no LLM behind it — `StageRouter`
consumes that frame, pops the text and hands it to the Floor, which decides which sister gets it.

`FloorUserMuteStrategy` is the aggregator-level twin of MicController's zero-fill: while the Floor
says a sister holds an audible turn, transcripts/VAD frames are suppressed exactly the way
`AlwaysUserMuteStrategy` suppresses them while a bot speaks — except the signal comes from the
Floor grant (strictly before any audio leaves a sister's speaker in her own process) instead of
from Bot*SpeakingFrames this pipeline never sees.
"""

import inspect

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMContextUserTurnFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy


def _last_user_text(msgs) -> str:
    for m in reversed(msgs or []):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"].strip()
    return ""


class FloorUserMuteStrategy(BaseUserMuteStrategy):
    """Mute the user while `gate()` is True (the Floor's `gate_active`)."""

    def __init__(self, gate):
        super().__init__()
        self._gate = gate

    async def process_frame(self, frame: Frame) -> bool:
        await super().process_frame(frame)
        try:
            return bool(self._gate())
        except Exception:
            return False


class StageRouter(FrameProcessor):
    """Answer sooner (`AIVA_SPECULATE`, Flux eager end of turn + `EagerUserTurnStrategies`): the
    aggregator pushes a provisional `LLMContextFrame(speculation=True)` when Flux GUESSES his turn
    is over → `on_speculate(text)`; an `EagerEndOfTurnCancelFrame` withdraws it → `on_unspeculate()`;
    and when the turn ends on the guessed text the aggregator writes it WITHOUT a new inference
    frame, only an `LLMContextUserTurnFrame` → submitted as the user turn (the floor matches it to
    the held reply). A different final transcript is cancelled first and then arrives as an
    ordinary `LLMContextFrame`."""

    def __init__(self, stage_ctx, *, on_user_turn, on_user_speaking=None, log=print,
                 drop_mic_audio=True, on_speculate=None, on_unspeculate=None):
        super().__init__()
        self._ctx = stage_ctx
        self._on_user_turn = on_user_turn          # async (text) -> None
        self._on_user_speaking = on_user_speaking  # (bool) -> None
        self._on_speculate = on_speculate          # async (text) -> None
        self._on_unspeculate = on_unspeculate      # () -> None
        self._log = log
        self._drop_audio = drop_mic_audio
        self._spec_text = None                     # the line a speculation is answering
        # the cancel is a SYSTEM frame and can overtake the queued speculation it cancels: count
        # such early cancels so the speculation behind them is ignored instead of started
        self._cancels_ahead = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and getattr(frame, "speculation", False):
            if self._cancels_ahead:
                self._cancels_ahead -= 1
                return
            text = _last_user_text(frame.context.get_messages())
            if text and self._on_speculate is not None:
                self._spec_text = text
                self.create_task(self._call(self._on_speculate, text))
            return  # a provisional copy: the stage context is untouched
        if isinstance(frame, EagerEndOfTurnCancelFrame):
            if self._spec_text is not None:
                self._spec_text = None
                await self._call(self._on_unspeculate)
            elif self._on_speculate is not None:
                self._cancels_ahead += 1
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMContextUserTurnFrame) and self._spec_text is not None:
            # the turn ended on the guess: written to the stage context with no inference frame
            self._spec_text = None
            text = self._pop_user_text() or (frame.text or "").strip()
            if text:
                self.create_task(self._submit(text))
            return
        if isinstance(frame, LLMContextFrame) and frame.context is self._ctx:
            self._spec_text = None
            text = self._pop_user_text()
            if text:
                self.create_task(self._submit(text))
            return  # consumed: the stage has no LLM behind it
        if isinstance(frame, UserStartedSpeakingFrame):
            self._note_speaking(True)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._note_speaking(False)
        if self._drop_audio and isinstance(frame, InputAudioRawFrame):
            return  # nothing below the stage consumes mic audio
        await self.push_frame(frame, direction)

    def _pop_user_text(self) -> str:
        msgs = self._ctx.get_messages()
        text = _last_user_text(msgs)
        # the stage context is only a buffer — keep it empty so it can never grow
        self._ctx.set_messages([m for m in msgs if m.get("role") == "system"])
        return text

    async def _submit(self, text):
        try:
            res = self._on_user_turn(text)
            if inspect.isawaitable(res):
                await res
        except Exception as e:
            self._log(f"stage: user turn handler failed: {e!r}")

    async def _call(self, fn, *args):
        if fn is None:
            return
        try:
            res = fn(*args)
            if inspect.isawaitable(res):
                await res
        except Exception as e:
            self._log(f"stage: speculation handler failed: {e!r}")

    def _note_speaking(self, on: bool):
        if self._on_user_speaking is None:
            return
        try:
            self._on_user_speaking(on)
        except Exception:
            pass
