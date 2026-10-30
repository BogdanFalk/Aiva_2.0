"""Character-side house processors, run through pipecat's own offline harness (no audio)."""
import asyncio

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import TTSUsageMetricsData
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test

from modules.house_link import EnoughSaid, HouseLink, MindWatcher, PrivacyGate, UsageTracker


def _texts(frames):
    return [(type(f).__name__, getattr(f, "skip_tts", None)) for f in frames]


@pytest.mark.asyncio
async def test_privacy_gate_stamps_skip_tts_only_when_not_audible():
    logged = []
    gate = PrivacyGate("Saga", log=logged.append)
    gate.set_audible(False, addressee="aiva")
    down, _ = await run_test(
        gate,
        frames_to_send=[LLMFullResponseStartFrame(), LLMTextFrame("hi "), LLMTextFrame("sis"),
                        LLMFullResponseEndFrame()],
        expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMTextFrame,
                              LLMFullResponseEndFrame],
    )
    assert all(f.skip_tts is True for f in down)
    assert any("text-only" in m and "hi sis" in m for m in logged)

    gate2 = PrivacyGate("Saga")
    gate2.set_audible(True, addressee="him")
    down2, _ = await run_test(
        gate2,
        frames_to_send=[LLMFullResponseStartFrame(), LLMTextFrame("hello"), LLMFullResponseEndFrame()],
        expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame],
    )
    assert all(f.skip_tts is None for f in down2)


@pytest.mark.asyncio
async def test_privacy_gate_fails_closed_without_a_latched_turn():
    logged = []
    gate = PrivacyGate("Saga", log=logged.append)      # never set_audible → not armed
    down, _ = await run_test(
        gate,
        frames_to_send=[LLMFullResponseStartFrame(), LLMTextFrame("oops"), LLMFullResponseEndFrame()],
        expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame],
    )
    assert all(f.skip_tts is True for f in down)
    assert any("no granted turn" in m for m in logged)


@pytest.mark.asyncio
async def test_mind_watcher_reports_response_and_tools():
    events = []
    w = MindWatcher(on_response=lambda ph, intr: events.append(("resp", ph, intr)),
                    on_tool=lambda d, cid, rerun: events.append(("tool", d, cid)))
    await run_test(
        w,
        frames_to_send=[LLMFullResponseStartFrame(),
                        FunctionCallInProgressFrame(function_name="f", tool_call_id="c1",
                                                    arguments={}),
                        FunctionCallResultFrame(function_name="f", tool_call_id="c1",
                                                arguments={}, result={"ok": True}),
                        LLMFullResponseEndFrame()],
        expected_down_frames=[LLMFullResponseStartFrame, FunctionCallInProgressFrame,
                              FunctionCallResultFrame, LLMFullResponseEndFrame],
    )
    assert events == [("resp", "start", False), ("tool", 1, "c1"), ("tool", -1, "c1"),
                      ("resp", "end", False)]
    assert not w.processing


def _tool_call():
    return FunctionCallInProgressFrame(function_name="start_task", tool_call_id="c1", arguments={})


def _calls_started():
    return FunctionCallsStartedFrame(function_calls=[FunctionCallFromLLM(
        function_name="start_task", tool_call_id="c1", arguments={}, context=None)])


@pytest.mark.asyncio
async def test_privacy_gate_waits_for_the_rerun_when_in_progress_lands_after_the_end():
    # 2026-10-13: pipecat runs parallel tools in their own task, so the in-progress frame arrived
    # after the response's end → the turn closed early and the post-tool reply ("say yes and I'll
    # run it") went out text-only on no turn and never reached the transcript
    done = []
    gate = PrivacyGate("Aiva", log=lambda m: None, on_turn_done=done.append)
    gate.set_audible(False, addressee="him")
    await run_test(
        gate,
        frames_to_send=[LLMFullResponseStartFrame(), SleepFrame(0.05), _calls_started(), SleepFrame(0.05),
                        LLMFullResponseEndFrame(),
                        _tool_call(),
                        LLMFullResponseStartFrame(), LLMTextFrame("Say yes and I'll run it."),
                        LLMFullResponseEndFrame()],
        expected_down_frames=[LLMFullResponseStartFrame, FunctionCallsStartedFrame, LLMFullResponseEndFrame,
                              FunctionCallInProgressFrame, LLMFullResponseStartFrame, LLMTextFrame,
                              LLMFullResponseEndFrame],
    )
    assert done == ["Say yes and I'll run it."]


@pytest.mark.asyncio
async def test_mind_watcher_reports_a_tool_before_the_end_and_only_once():
    events = []
    w = MindWatcher(on_response=lambda ph, intr: events.append(("resp", ph)),
                    on_tool=lambda d, cid, rerun: events.append(("tool", d)))
    await run_test(
        w,
        frames_to_send=[LLMFullResponseStartFrame(), SleepFrame(0.05), _calls_started(), SleepFrame(0.05),
                        LLMFullResponseEndFrame(),
                        _tool_call(),
                        FunctionCallResultFrame(function_name="start_task", tool_call_id="c1",
                                                arguments={}, result={"ok": True})],
        expected_down_frames=[LLMFullResponseStartFrame, FunctionCallsStartedFrame, LLMFullResponseEndFrame,
                              FunctionCallInProgressFrame, FunctionCallResultFrame],
    )
    assert events == [("resp", "start"), ("tool", 1), ("resp", "end"), ("tool", -1)]


@pytest.mark.asyncio
@pytest.mark.parametrize("audible", [True, False])
async def test_privacy_gate_ends_the_turn_after_the_tool_rerun_with_all_its_words(audible):
    # 2026-09-26: "Hey Frank, let's finish that folder." → start_task → "On it — ..." lost the
    # second half (the aggregator's stop fired on the first response's delayed end frame)
    done = []
    gate = PrivacyGate("Aiva", log=lambda m: None, on_turn_done=done.append)
    gate.set_audible(audible, addressee="him")
    await run_test(
        gate,
        frames_to_send=[LLMFullResponseStartFrame(), LLMTextFrame("Hey Frank. "), _tool_call(),
                        LLMFullResponseEndFrame(),
                        LLMFullResponseStartFrame(), LLMTextFrame("On it."), LLMFullResponseEndFrame()],
        expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, FunctionCallInProgressFrame,
                              LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame,
                              LLMFullResponseEndFrame],
    )
    assert done == ["Hey Frank. On it."]          # once, at the real end, with both halves


@pytest.mark.asyncio
async def test_privacy_gate_tool_only_turn_waits_and_ungranted_responses_never_end_a_turn():
    done = []
    gate = PrivacyGate("Aiva", log=lambda m: None, on_turn_done=done.append)
    gate.set_audible(True)
    await run_test(gate, frames_to_send=[LLMFullResponseStartFrame(), _tool_call(), LLMFullResponseEndFrame()],
                   expected_down_frames=[LLMFullResponseStartFrame, FunctionCallInProgressFrame,
                                         LLMFullResponseEndFrame])
    assert done == []                             # the re-run is still to come
    gate.release()
    await run_test(gate, frames_to_send=[LLMFullResponseStartFrame(), LLMTextFrame("x"), LLMFullResponseEndFrame()],
                   expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame])
    assert done == []                             # not armed → no turn to end


# --- 2026-10-26: "I'll work through the checklist…" with start_task, then its re-run said "On it —
# --- I'll go item by item…" — the same line twice in one message ---------------------------------

def _result(props=None, call_id="c1"):
    return FunctionCallResultFrame(function_name="start_task", tool_call_id=call_id, arguments={},
                                   result={"success": True, "status": "started"}, properties=props)


@pytest.mark.asyncio
async def test_enough_said_result_ends_the_turn_without_a_rerun():
    done = []
    gate = PrivacyGate("Aiva", log=lambda m: None, on_turn_done=done.append)
    gate.set_audible(True)
    props = EnoughSaid()
    await run_test(
        gate,
        frames_to_send=[LLMFullResponseStartFrame(), LLMTextFrame("I'll work through the checklist."),
                        SleepFrame(0.05), _calls_started(), SleepFrame(0.05), LLMFullResponseEndFrame(),
                        _tool_call(), _result(props)],
        expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, FunctionCallsStartedFrame,
                              LLMFullResponseEndFrame, FunctionCallInProgressFrame, FunctionCallResultFrame],
    )
    assert props.run_llm is False                 # the assistant aggregator won't run her LLM again
    assert done == ["I'll work through the checklist."]


@pytest.mark.asyncio
async def test_enough_said_result_before_the_end_ends_the_turn_at_the_end_once():
    done = []
    gate = PrivacyGate("Aiva", log=lambda m: None, on_turn_done=done.append)
    gate.set_audible(True)
    await run_test(
        gate,
        frames_to_send=[LLMFullResponseStartFrame(), LLMTextFrame("On it."), SleepFrame(0.05), _calls_started(),
                        SleepFrame(0.05), _tool_call(), _result(EnoughSaid()), LLMFullResponseEndFrame()],
        expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, FunctionCallsStartedFrame,
                              FunctionCallInProgressFrame, FunctionCallResultFrame, LLMFullResponseEndFrame],
    )
    assert done == ["On it."]


@pytest.mark.asyncio
async def test_enough_said_still_reruns_when_she_said_nothing_yet():
    done = []
    gate = PrivacyGate("Aiva", log=lambda m: None, on_turn_done=done.append)
    gate.set_audible(True)
    props = EnoughSaid()
    await run_test(
        gate,
        frames_to_send=[LLMFullResponseStartFrame(), SleepFrame(0.05), _calls_started(), SleepFrame(0.05),
                        LLMFullResponseEndFrame(), _tool_call(), _result(props),
                        LLMFullResponseStartFrame(), LLMTextFrame("On it."), LLMFullResponseEndFrame()],
        expected_down_frames=[LLMFullResponseStartFrame, FunctionCallsStartedFrame, LLMFullResponseEndFrame,
                              FunctionCallInProgressFrame, FunctionCallResultFrame,
                              LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame],
    )
    assert props.run_llm is None and done == ["On it."]


@pytest.mark.asyncio
@pytest.mark.parametrize("said, rerun", [("Can I restart the house to check it?", False),
                                         ("Yes, let it run.", True)])
async def test_enough_said_asked_needs_a_question_already_asked(said, rerun):
    # 2026-10-26 13:10: on her news turn she asked him AND called answer_task herself; the refusal
    # says "ask him", so she asked the same question a second time
    done = []
    gate = PrivacyGate("Aiva", log=lambda m: None, on_turn_done=done.append)
    gate.set_audible(True)
    props = EnoughSaid(asked=True)
    await run_test(
        gate,
        frames_to_send=[LLMFullResponseStartFrame(), LLMTextFrame(said), SleepFrame(0.05), _calls_started(),
                        SleepFrame(0.05), LLMFullResponseEndFrame(), _tool_call(), _result(props)],
        expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, FunctionCallsStartedFrame,
                              LLMFullResponseEndFrame, FunctionCallInProgressFrame, FunctionCallResultFrame],
    )
    assert (props.run_llm is None) is rerun
    assert done == ([] if rerun else [said])


@pytest.mark.asyncio
async def test_enough_said_never_skips_the_rerun_a_sibling_call_needs():
    done = []
    gate = PrivacyGate("Aiva", log=lambda m: None, on_turn_done=done.append)
    gate.set_audible(True)
    props = EnoughSaid()
    two = FunctionCallsStartedFrame(function_calls=[
        FunctionCallFromLLM(function_name="start_task", tool_call_id="c1", arguments={}, context=None),
        FunctionCallFromLLM(function_name="web_search", tool_call_id="c2", arguments={}, context=None)])
    await run_test(
        gate,
        frames_to_send=[LLMFullResponseStartFrame(), LLMTextFrame("On it."), SleepFrame(0.05), two,
                        SleepFrame(0.05), LLMFullResponseEndFrame(), _tool_call(), _result(props)],
        expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, FunctionCallsStartedFrame,
                              LLMFullResponseEndFrame, FunctionCallInProgressFrame, FunctionCallResultFrame],
    )
    assert props.run_llm is None and done == []


@pytest.mark.asyncio
async def test_a_reruns_reply_is_joined_as_a_new_sentence():
    done = []
    gate = PrivacyGate("Aiva", log=lambda m: None, on_turn_done=done.append)
    gate.set_audible(True)
    await run_test(
        gate,
        frames_to_send=[LLMFullResponseStartFrame(), LLMTextFrame("as I go."), _tool_call(),
                        LLMFullResponseEndFrame(),
                        LLMFullResponseStartFrame(), LLMTextFrame("The app is open."), LLMFullResponseEndFrame()],
        expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, FunctionCallInProgressFrame,
                              LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame,
                              LLMFullResponseEndFrame],
    )
    assert done == ["as I go. The app is open."]


@pytest.mark.asyncio
@pytest.mark.parametrize("said, reruns", [("On it.", False), ("", True)])
async def test_the_real_assistant_aggregator_obeys_the_gate(said, reruns):
    # her LLM re-runs when the aggregator pushes the context back up after a tool result; the gate
    # must have turned run_llm off on the very frame the aggregator reads
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator

    gate = PrivacyGate("Aiva", log=lambda m: None)
    gate.set_audible(True)
    agg = LLMAssistantAggregator(LLMContext(messages=[{"role": "user", "content": "go through the list"}]))
    text = [LLMTextFrame(said)] if said else []
    _, up = await run_test(
        Pipeline([gate, agg]),
        frames_to_send=[LLMFullResponseStartFrame(), *text, SleepFrame(0.05), _calls_started(),
                        SleepFrame(0.05), LLMFullResponseEndFrame(), _tool_call(), SleepFrame(0.05),
                        _result(EnoughSaid()), SleepFrame(0.2)],
        expected_down_frames=None,
    )
    assert any(isinstance(f, LLMContextFrame) for f in up) is reruns


@pytest.mark.asyncio
@pytest.mark.parametrize("props, rerun", [(None, True), (EnoughSaid(), True),
                                          (EnoughSaid(run_llm=False), False)])
async def test_mind_watcher_tells_the_house_when_no_rerun_follows(props, rerun):
    events = []
    w = MindWatcher(on_tool=lambda d, cid, r: events.append((d, r)))
    await run_test(w, frames_to_send=[_tool_call(), _result(props)],
                   expected_down_frames=[FunctionCallInProgressFrame, FunctionCallResultFrame])
    assert events == [(1, True), (-1, rerun)]


@pytest.mark.asyncio
async def test_usage_tracker_counts_tts_and_calls_back():
    seen = []
    u = UsageTracker(on_tts=lambda n: seen.append(n))
    await run_test(
        u,
        frames_to_send=[MetricsFrame(data=[TTSUsageMetricsData(processor="tts", value=42)])],
        expected_down_frames=[MetricsFrame],
    )
    assert u.tts_chars == 42 and seen == [42]


@pytest.mark.asyncio
async def test_house_link_runs_a_turn_and_reports_speaking():
    ctx = LLMContext(messages=[{"role": "system", "content": "sys"}])
    speaking = []
    link = HouseLink(ctx, on_speaking=lambda tid, on: speaking.append((tid, on)), log=lambda m: None)

    async def drive():
        await asyncio.sleep(0.05)
        ok = await link.run_turn({"id": 7, "audible": True}, user_text="Saga, hi",
                                 cue="[Answer her.]", notes=["[Speaking now: Frank.]"])
        assert ok

    down, up = await asyncio.gather(
        run_test(
            link,
            frames_to_send=[SleepFrame(0.2), BotStartedSpeakingFrame(), BotStoppedSpeakingFrame()],
            frames_to_send_direction=FrameDirection.UPSTREAM,
            expected_up_frames=[BotStartedSpeakingFrame, BotStoppedSpeakingFrame],
        ),
        drive(),
    )
    down_frames, _ = down
    assert any(isinstance(f, LLMContextFrame) and f.context is ctx for f in down_frames)
    roles = [(m["role"], m["content"]) for m in ctx.get_messages()]
    assert roles == [("system", "sys"), ("system", "[Speaking now: Frank.]"),
                     ("user", "Saga, hi"), ("system", "[Answer her.]")]
    assert speaking == [(7, True), (7, False)]
    link.finish_turn()
    assert [m["content"] for m in ctx.get_messages()] == ["sys", "[Speaking now: Frank.]", "Saga, hi"]


@pytest.mark.asyncio
async def test_house_link_overheard_is_stored_not_run_and_interrupt_pushes_frame():
    ctx = LLMContext(messages=[])
    link = HouseLink(ctx, log=lambda m: None)

    async def drive():
        await asyncio.sleep(0.05)
        link.add_overheard('[He said to Aiva: "hello"]')
        await link.interrupt()

    (down_frames, _), _ = await asyncio.gather(
        run_test(link, frames_to_send=[SleepFrame(0.2)],
                 expected_down_frames=[InterruptionFrame]),
        drive(),
    )
    assert not any(isinstance(f, LLMContextFrame) for f in down_frames)
    assert ctx.get_messages() == [{"role": "user", "content": '[He said to Aiva: "hello"]'}]


@pytest.mark.asyncio
async def test_speaking_off_after_the_turn_closed_still_carries_the_turn_id():
    """The assistant aggregator closes the turn (finish_turn) BEFORE the transport reports the
    audio stopped — the stop report must still name the turn or the floor never releases."""
    ctx = LLMContext(messages=[])
    speaking = []
    link = HouseLink(ctx, on_speaking=lambda tid, on: speaking.append((tid, on)), log=lambda m: None)

    async def drive():
        await asyncio.sleep(0.05)
        await link.run_turn({"id": 9, "audible": True}, user_text="hi")
        await asyncio.sleep(0.15)          # BotStarted arrives here (see frames below)
        link.finish_turn()                 # aggregator says the turn is over…

    await asyncio.gather(
        run_test(
            link,
            frames_to_send=[SleepFrame(0.1), BotStartedSpeakingFrame(), SleepFrame(0.2),
                            BotStoppedSpeakingFrame()],   # …but audio stops after that
            frames_to_send_direction=FrameDirection.UPSTREAM,
            expected_up_frames=[BotStartedSpeakingFrame, BotStoppedSpeakingFrame],
        ),
        drive(),
    )
    assert speaking == [(9, True), (9, False)]
