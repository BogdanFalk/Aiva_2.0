"""House stage: the router that hands finished user turns to the Floor, and the floor-driven mute."""
import asyncio

import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMContextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import SleepFrame, run_test

from modules.house_stage import FloorUserMuteStrategy, StageRouter


@pytest.mark.asyncio
async def test_router_consumes_stage_turn_and_submits_text():
    ctx = LLMContext(messages=[{"role": "system", "content": "stage"}])
    got = []
    speaking = []

    async def on_turn(text):
        got.append(text)

    router = StageRouter(ctx, on_user_turn=on_turn, on_user_speaking=speaking.append,
                         log=lambda m: None)
    ctx.add_message({"role": "user", "content": "  Saga, are you there?  "})
    other = LLMContext(messages=[])
    down, _ = await run_test(
        router,
        frames_to_send=[UserStartedSpeakingFrame(), UserStoppedSpeakingFrame(),
                        InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1),
                        LLMContextFrame(ctx), LLMContextFrame(other),
                        TranscriptionFrame(text="x", user_id="u", timestamp="t"), SleepFrame(0.1)],
        expected_down_frames=[UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
                              LLMContextFrame, TranscriptionFrame],
    )
    assert got == ["Saga, are you there?"]
    assert speaking == [True, False]
    # the stage buffer keeps only its system head
    assert ctx.get_messages() == [{"role": "system", "content": "stage"}]
    # a context frame for a DIFFERENT context passes through untouched
    assert any(isinstance(f, LLMContextFrame) and f.context is other for f in down)


@pytest.mark.asyncio
async def test_floor_mute_strategy_reads_the_gate():
    state = {"gate": False}
    s = FloorUserMuteStrategy(lambda: state["gate"])
    assert await s.process_frame(TranscriptionFrame(text="a", user_id="u", timestamp="t")) is False
    state["gate"] = True
    assert await s.process_frame(TranscriptionFrame(text="a", user_id="u", timestamp="t")) is True

    def boom():
        raise RuntimeError("x")

    assert await FloorUserMuteStrategy(boom).process_frame(UserStartedSpeakingFrame()) is False
