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
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy


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
    def __init__(self, stage_ctx, *, on_user_turn, on_user_speaking=None, log=print,
                 drop_mic_audio=True):
        super().__init__()
        self._ctx = stage_ctx
        self._on_user_turn = on_user_turn          # async (text) -> None
        self._on_user_speaking = on_user_speaking  # (bool) -> None
        self._log = log
        self._drop_audio = drop_mic_audio

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and frame.context is self._ctx:
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
        text = ""
        for m in reversed(msgs):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                text = m["content"].strip()
                break
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

    def _note_speaking(self, on: bool):
        if self._on_user_speaking is None:
            return
        try:
            self._on_user_speaking(on)
        except Exception:
            pass
