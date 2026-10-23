"""Answer sooner (AIVA_SPECULATE): the floor asks one sister to prepare a held reply on Flux's guess,
the stage turns pipecat's speculation frames into floor calls, HouseLink holds/releases/drops the
reply through pipecat's own SpeculationGate — and the gate behaves the way HouseLink relies on."""
import asyncio

import pytest
from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    LLMContextFrame,
    LLMContextUserTurnFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import run_test
from pipecat.turns.speculation_gate import SpeculationGate

from modules import house_proto as p
from modules.house_link import HouseLink
from modules.house_stage import StageRouter
from modules.house_stt import STTConfig
from modules.sisters import same_line
from test_house_floor import FakeArbiter, Rig

DOWN = FrameDirection.DOWNSTREAM


# --- the pure bits ----------------------------------------------------------------------------

def test_same_line_ignores_formatting_only():
    assert same_line("what's the weather tomorrow", "What's the weather tomorrow?")
    assert not same_line("what's the weather", "what's the weather tomorrow")
    assert not same_line("", "")


def test_speculation_is_off_by_default_and_needs_flux():
    assert not STTConfig.from_env({}).speculate
    assert STTConfig.from_env({"AIVA_SPECULATE": "1"}).speculate
    assert not STTConfig.from_env({"AIVA_SPECULATE": "1", "AIVA_STT": "nova"}).speculate


# --- the floor --------------------------------------------------------------------------------

def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_a_named_line_is_prepared_then_released_on_the_same_words():
    r = Rig()
    assert run(r.floor.speculate("Saga, what are you drawing")) == "saga"
    prep = r.saga.last(p.Prepare)
    assert prep.text == "Saga, what are you drawing" and r.aiva.last(p.Prepare) is None
    assert not r.floor.busy and r.floor.gate_active is False and r.saga.turn is None   # not a grant
    run(r.floor.submit_user_turn("Saga, what are you drawing?"))
    assert r.saga.turn.prepared == prep.spec and r.saga.last(p.Unprepare) is None


def test_different_final_words_drop_the_held_reply():
    r = Rig()
    run(r.floor.speculate("Saga, what are you"))
    spec = r.saga.last(p.Prepare).spec
    run(r.floor.submit_user_turn("Saga, what are you doing tonight?"))
    assert r.saga.last(p.Unprepare).spec == spec and r.saga.turn.prepared is None


def test_he_went_on_and_the_ttl_both_withdraw():
    r = Rig()
    run(r.floor.speculate("Saga, hold on"))
    assert r.floor.unspeculate("he went on") and r.saga.last(p.Unprepare) is not None
    assert not r.floor.unspeculate()                                  # nothing left to drop
    run(r.floor.speculate("Saga, hold on"))
    n = len(r.saga.of(p.Unprepare))
    r.clock.advance(11)
    r.floor.tick()
    assert len(r.saga.of(p.Unprepare)) == n + 1


def test_no_speculation_while_busy_for_both_or_for_an_errand():
    r = Rig()
    run(r.floor.submit_user_turn("Aiva, hello"))                      # aiva holds the floor now
    assert run(r.floor.speculate("Saga, hi")) is None
    r2 = Rig()
    assert run(r2.floor.speculate("Aiva and Saga, good night")) is None
    r3 = Rig()
    assert run(r3.floor.speculate("Aiva, ask Saga what she wants for the cat")) is None
    assert not r2.aiva.of(p.Prepare) and not r3.aiva.of(p.Prepare)


def test_the_arbiter_runs_while_he_finishes_and_is_not_asked_again():
    arb = FakeArbiter('{"to": "saga", "why": "her thread"}')
    r = Rig(arbiter=arb)
    assert run(r.floor.speculate("can you make it gold")) == "saga"
    assert len(arb.calls) == 1
    run(r.floor.submit_user_turn("Can you make it gold?"))
    assert len(arb.calls) == 1 and r.saga.turn.prepared is not None


# --- the stage --------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stage_turns_speculation_frames_into_floor_calls():
    stage = LLMContext(messages=[{"role": "system", "content": "stage"}])
    got = {"spec": [], "unspec": 0, "turns": []}

    async def on_spec(text):
        got["spec"].append(text)

    def on_unspec():
        got["unspec"] += 1

    async def on_turn(text):
        got["turns"].append(text)

    router = StageRouter(stage, on_user_turn=on_turn, on_speculate=on_spec, on_unspeculate=on_unspec,
                         log=lambda m: None)
    provisional = LLMContext(messages=[{"role": "user", "content": "saga what are you drawing"}])
    stage.add_message({"role": "user", "content": "Saga, what are you drawing?"})   # the aggregator's write
    await run_test(router, frames_to_send=[
        LLMContextFrame(context=provisional, speculation=True),
        LLMContextUserTurnFrame(text="Saga, what are you drawing?", timestamp=""),
    ], expected_down_frames=[])
    await asyncio.sleep(0.01)
    assert got["spec"] == ["saga what are you drawing"] and got["turns"] == ["Saga, what are you drawing?"]
    assert stage.get_messages() == [{"role": "system", "content": "stage"}]          # buffer emptied
    # the cancel is a system frame and may overtake the speculation it cancels: in either order,
    # every speculation the floor was asked for is withdrawn and nothing is left pending
    await run_test(router, frames_to_send=[LLMContextFrame(context=provisional, speculation=True),
                                           EagerEndOfTurnCancelFrame()],
                   expected_down_frames=[EagerEndOfTurnCancelFrame])
    await asyncio.sleep(0.01)
    assert len(got["spec"]) == 1 + got["unspec"]
    assert router._spec_text is None and router._cancels_ahead == 0


# --- the sister's link ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prepare_runs_on_a_copy_and_release_writes_his_line_then_stops_the_turn():
    ctx = LLMContext(messages=[{"role": "system", "content": "persona"}])
    link = HouseLink(ctx, log=lambda m: None)
    link._started = True
    pushed = []

    async def capture(frame, direction=DOWN):
        pushed.append(frame)

    link.push_frame = capture
    assert await link.prepare(7, "what are you drawing")
    started, spec = pushed
    assert isinstance(started, UserStartedSpeakingFrame) and spec.speculation is True
    assert spec.context.get_messages()[-1] == {"role": "user", "content": "what are you drawing"}
    assert ctx.get_messages() == [{"role": "system", "content": "persona"}]        # real context untouched
    assert link.prepared_for(7, "What are you drawing?") and not link.prepared_for(8, "what are you drawing")
    assert await link.run_prepared({"id": 3}, user_text="What are you drawing?", notes=["[note]"])
    assert isinstance(pushed[-1], UserStoppedSpeakingFrame) and link.current_turn["id"] == 3
    assert ctx.get_messages()[-2:] == [{"role": "system", "content": "[note]"},
                                       {"role": "user", "content": "What are you drawing?"}]
    assert not link.preparing


@pytest.mark.asyncio
async def test_withdraw_cancels_and_a_tool_wanting_reply_withdraws_itself():
    link = HouseLink(LLMContext(), log=lambda m: None)
    link._started = True
    pushed = []

    async def capture(frame, direction=DOWN):
        pushed.append(frame)

    link.push_frame = capture
    await link.prepare(1, "set a timer")
    assert await link.withdraw("he went on") and isinstance(pushed[-1], EagerEndOfTurnCancelFrame)
    await link.prepare(2, "set a timer")
    link._spec["withdrawn"] = True                    # what the LLM's upstream cancel does
    assert not link.prepared_for(2, "set a timer")   # → the turn runs fresh and the tool runs then
    assert not await link.prepare(3, "") and link.spec_id == 2


# --- the gate HouseLink relies on ---------------------------------------------------------------

def _reply():
    return [(LLMFullResponseStartFrame(), DOWN), (LLMTextFrame("Hi."), DOWN), (LLMFullResponseEndFrame(), DOWN)]


def _feed(gate, frames):
    out = []
    for f, d in frames:
        out += gate.process(f, d)
    return out


def test_gate_holds_until_he_stops_and_drops_on_cancel():
    gate = SpeculationGate()
    _feed(gate, [(UserStartedSpeakingFrame(), DOWN)])
    gate.begin_speculation(True)
    assert _feed(gate, _reply()) == []                                        # held
    released = _feed(gate, [(UserStoppedSpeakingFrame(), DOWN)])
    assert [type(f).__name__ for f, _ in released] == ["UserStoppedSpeakingFrame", "LLMFullResponseStartFrame",
                                                         "LLMTextFrame", "LLMFullResponseEndFrame"]
    _feed(gate, [(UserStartedSpeakingFrame(), DOWN)])                       # the next prepare
    gate.begin_speculation(True)
    assert _feed(gate, _reply()) == []
    dropped = _feed(gate, [(EagerEndOfTurnCancelFrame(), DOWN)])
    assert [type(f).__name__ for f, _ in dropped] == ["EagerEndOfTurnCancelFrame"]


def test_without_a_fresh_start_the_next_speculation_would_be_spoken():
    # why HouseLink.prepare pushes UserStartedSpeakingFrame first: after a release the turn counts
    # as ended, and a speculation for an ended turn is not held at all
    gate = SpeculationGate()
    _feed(gate, [(UserStartedSpeakingFrame(), DOWN)])
    gate.begin_speculation(True)
    _feed(gate, _reply() + [(UserStoppedSpeakingFrame(), DOWN)])
    gate.begin_speculation(True)                                              # no fresh start
    assert len(_feed(gate, _reply())) == 3                                    # straight through


# --- through pipecat's real LLM-service gate into her PrivacyGate ----------------------------------

class _ScriptedLLM:
    """A real pipecat LLMService (so its own SpeculationGate gates everything it pushes) that
    answers every LLMContextFrame with one short reply."""

    @staticmethod
    def build():
        from pipecat.services.llm_service import LLMService

        class ScriptedLLM(LLMService):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, LLMContextFrame):
                    await self.push_frame(LLMFullResponseStartFrame())
                    await self.push_frame(LLMTextFrame("[laughs] Fine, "))
                    await self.push_frame(LLMTextFrame("you win."))
                    await self.push_frame(LLMFullResponseEndFrame())
                else:
                    await self.push_frame(frame, direction)

        return ScriptedLLM()


def _kinds(frames):
    """Frame types that reached the end, minus the service's start-up metadata."""
    return [type(f).__name__ for f in frames if "Metadata" not in type(f).__name__]


@pytest.mark.asyncio
async def test_a_held_reply_reaches_her_voice_and_the_house_only_after_the_release():
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.tests.utils import SleepFrame
    from modules.house_link import PrivacyGate
    said = []
    gate = PrivacyGate("Aiva", log=lambda m: None, on_turn_done=said.append)
    gate.set_audible(True, addressee="him")
    spec = LLMContextFrame(context=LLMContext(messages=[{"role": "user", "content": "you win?"}]),
                           speculation=True)
    down, _ = await run_test(
        Pipeline([_ScriptedLLM.build(), gate]),
        frames_to_send=[UserStartedSpeakingFrame(), spec, SleepFrame(0.2)],
        expected_down_frames=None)
    assert _kinds(down) == ["UserStartedSpeakingFrame"]
    assert said == []                                   # held inside the LLM: nothing reached her gate

    said.clear()
    gate.set_audible(True, addressee="him")
    spec2 = LLMContextFrame(context=LLMContext(messages=[{"role": "user", "content": "you win?"}]),
                            speculation=True)
    down, _ = await run_test(
        Pipeline([_ScriptedLLM.build(), gate]),
        frames_to_send=[UserStartedSpeakingFrame(), spec2, SleepFrame(0.2), UserStoppedSpeakingFrame()],
        expected_down_frames=None)
    assert _kinds(down) == ["UserStartedSpeakingFrame", "UserStoppedSpeakingFrame", "LLMFullResponseStartFrame",
                            "LLMTextFrame", "LLMTextFrame", "LLMFullResponseEndFrame"]
    assert said == ["Fine, you win."]                   # released, once, cues stripped

    said.clear()
    gate.set_audible(True, addressee="him")
    spec3 = LLMContextFrame(context=LLMContext(messages=[{"role": "user", "content": "you win?"}]),
                            speculation=True)
    down, _ = await run_test(
        Pipeline([_ScriptedLLM.build(), gate]),
        frames_to_send=[UserStartedSpeakingFrame(), spec3, SleepFrame(0.2), EagerEndOfTurnCancelFrame()],
        expected_down_frames=None)
    assert _kinds(down) == ["UserStartedSpeakingFrame", "EagerEndOfTurnCancelFrame"]
    assert said == []                                   # withdrawn: never spoken, never reported
