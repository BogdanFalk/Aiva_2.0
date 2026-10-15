"""The Floor with two fake clients (recording every message), a fake clock driving `tick`, a fake
arbiter and an in-memory HouseMemory. No sockets, no sleeps beyond the arbiter-timeout test's
few milliseconds."""
import asyncio
import sqlite3
from datetime import datetime

import pytest

from modules import house_proto as p
from modules.house_floor import ClientInfo, Floor
from modules.house_memory import HouseMemory
from modules.sisters import ExchangeConfig, Turn, absent_cue, both_second_cue, chat_cue, sister_cue

T0 = datetime(2026, 9, 11, 15, 0, 0).timestamp()      # 15:00 local


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s
        return self.t


class FakeClient:
    def __init__(self, slug, name, blurb="", has_voice=True):
        self.info = ClientInfo(slug, name, blurb, has_voice)
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)

    def of(self, cls):
        return [m for m in self.sent if isinstance(m, cls)]

    def last(self, cls):
        ms = self.of(cls)
        return ms[-1] if ms else None

    @property
    def turn(self):
        return self.last(p.TurnMsg)


class FakeArbiter:
    def __init__(self, reply='{"to": "saga", "why": "test"}', delay=0.0, raise_=False):
        self.reply, self.delay, self.raise_, self.calls = reply, delay, raise_, []

    async def __call__(self, messages):
        self.calls.append(messages)
        if self.raise_:
            raise RuntimeError("boom")
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.reply


class Rig:
    def __init__(self, private=True, arbiter=None, cfg=None, conn=None, **kw):
        self.clock = Clock()
        self.mem = HouseMemory(conn or sqlite3.connect(":memory:"), now=self.clock)
        self.aiva = FakeClient("aiva", "Aiva", "cheeky, has tools")
        self.saga = FakeClient("saga", "Saga", "new, still forming")
        self.gates, self.grants, self.logs = [], [], []
        self.floor = Floor(self.mem, private=private, log=self.logs.append, now=self.clock,
                           clients={"aiva": self.aiva, "saga": self.saga}, on_grant=self.grants.append,
                           on_gate_change=self.gates.append, arbiter=arbiter, cfg=cfg or ExchangeConfig(),
                           arbiter_timeout_s=0.05, **kw)

    def finish_audible(self, c, text, addressee="him"):
        f, tid = self.floor, c.turn.turn
        f.on_response(c.info.slug, tid, "start")
        f.on_speaking(c.info.slug, tid, True)
        f.on_response(c.info.slug, tid, "end")
        f.on_speaking(c.info.slug, tid, False)
        f.on_said(c.info.slug, tid, text, addressee, private=False)

    def finish_private(self, c, text, addressee):
        f, tid = self.floor, c.turn.turn
        f.on_response(c.info.slug, tid, "start")
        f.on_response(c.info.slug, tid, "end")
        f.on_said(c.info.slug, tid, text, addressee, private=True)

    def transcript(self):
        return list(reversed(self.mem.recent_transcript()))


# --- routing by name ----------------------------------------------------------------------------

async def test_named_turn_to_her_and_overheard_to_the_other():
    r = Rig()
    d = await r.floor.submit_user_turn("Saga, what's the weather?", speaker_vec=[0.1, 0.2], affect={"arousal": 0.3})
    assert d.addressee == "saga" and d.reason == "named"
    t = r.saga.turn
    assert t is not None and t.kind == "user" and t.text == "Saga, what's the weather?" and t.audible is True
    assert t.addressee == "him" and t.cue is None and t.speaker_vec == [0.1, 0.2] and t.affect == {"arousal": 0.3}
    assert r.aiva.turn is None
    o = r.aiva.last(p.Overheard)
    assert o.line == '[He said to Saga: "Saga, what\'s the weather?"]' and o.turn == t.turn and o.speaker_vec == [0.1, 0.2]
    assert r.floor.holder == "saga" and r.floor.gate_active is True and r.gates == [True] and r.floor.busy
    assert [g.mind for g in r.grants] == ["saga"]
    row = r.transcript()[0]
    assert (row["speaker"], row["addressee"], row["kind"], row["turn_id"]) == ("user", "saga", "named", t.turn)

    r.finish_audible(r.saga, "Sunny all day.")
    assert r.floor.holder is None and r.floor.gate_active is False and r.gates == [True, False] and not r.floor.busy
    assert r.aiva.last(p.Overheard).line == '[Saga said to him: "Sunny all day."]'
    said = r.transcript()[1]
    assert (said["speaker"], said["addressee"], said["audible"], said["kind"]) == ("saga", "him", True, "user")
    st = r.floor.address_state
    assert st.last_addressee == "saga" and st.last_speaker == "saga" and st.last_reply_was_question is False


async def test_wake_hint_routes_when_no_name_spoken():
    r = Rig(arbiter=FakeArbiter())
    await r.floor.submit_user_turn("what time is it", wake_hint="saga")
    assert r.saga.turn.wake_hint == "saga" and r.aiva.turn is None
    assert r.floor.arbiter.calls == []


# --- both -------------------------------------------------------------------------------------------

async def test_both_ordering_with_second_cue():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, Saga, what should we have for dinner?")
    first = r.aiva.turn
    assert first.kind == "user" and first.text.startswith("Aiva, Saga")
    assert r.saga.turn is None
    assert r.saga.last(p.Overheard).line == '[He said to both of you: "Aiva, Saga, what should we have for dinner?"]'
    assert r.transcript()[0]["addressee"] == "both"

    r.finish_audible(r.aiva, "Pizza. Saga, you in?")
    kinds = [type(m).__name__ for m in r.saga.sent]
    assert kinds == ["Overheard", "Overheard", "TurnMsg"]                 # user line, Aiva's reply, then her turn
    assert r.saga.sent[1].line == '[Aiva said to him: "Pizza. Saga, you in?"]'
    second = r.saga.turn
    assert second.kind == "both_second" and second.text is None and second.cue == both_second_cue("Aiva")
    assert second.reply_to == first.turn and second.audible is True and r.floor.gate_active is True
    assert r.floor.queue == []                                            # no extra sister round queued on top

    r.finish_audible(r.saga, "Sure, pizza.")
    assert r.floor.holder is None and r.floor.queue == [] and r.aiva.last(p.Overheard).line == '[Saga said to him: "Sure, pizza."]'


# --- arbiter ----------------------------------------------------------------------------------------

async def test_ambiguous_line_asks_arbiter_once_then_sticks():
    arb = FakeArbiter('{"to": "saga", "why": "her thread"}')
    r = Rig(arbiter=arb)
    d = await r.floor.submit_user_turn("what about tomorrow")
    assert len(arb.calls) == 1 and d.addressee == "saga" and d.reason.startswith("arbiter") and d.confident is False
    assert r.saga.turn.kind == "user" and r.aiva.turn is None
    assert arb.calls[0][1]["content"].endswith('His new line: "what about tomorrow"') and "- aiva: cheeky" in arb.calls[0][1]["content"]
    assert r.transcript()[0]["kind"].startswith("arbiter")
    r.finish_audible(r.saga, "Rain.")
    r.clock.advance(10)
    d2 = await r.floor.submit_user_turn("and the day after")
    assert d2.reason == "sticky" and len(arb.calls) == 1               # no second call inside the sticky window


async def test_arbiter_timeout_falls_back_to_last_speaker():
    arb = FakeArbiter(delay=0.5)
    r = Rig(arbiter=arb)
    await r.floor.submit_user_turn("Aiva, hi")
    r.finish_audible(r.aiva, "Hi.")
    r.clock.advance(600)                                                  # well past sticky
    d = await r.floor.submit_user_turn("so what do you think about it all")
    assert len(arb.calls) == 1 and d.addressee == "aiva" and d.reason == "fallback:last_speaker"
    assert len(r.aiva.of(p.TurnMsg)) == 2 and r.saga.turn is None
    assert any("arbiter timed out" in l for l in r.logs)


async def test_arbiter_unparsable_or_error_falls_back_to_default():
    r = Rig(arbiter=FakeArbiter("no idea"))
    d = await r.floor.submit_user_turn("hmm what now")
    assert d.addressee == "aiva" and d.reason == "fallback:default"
    r2 = Rig(arbiter=FakeArbiter(raise_=True))
    d = await r2.floor.submit_user_turn("hmm what now")
    assert d.addressee == "aiva" and any("arbiter failed" in l for l in r2.logs)


async def test_no_arbiter_call_with_one_sister_connected():
    arb = FakeArbiter()
    r = Rig(arbiter=arb)
    r.floor.on_client_lost("saga")
    await r.floor.submit_user_turn("unaddressed line here")
    assert arb.calls == [] and r.aiva.turn.kind == "user"


# --- said → continuation ----------------------------------------------------------------------------

async def test_said_continues_only_when_she_asks_her_sister():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, what do you make of Saga's idea?")
    r.finish_audible(r.aiva, "I like it. Saga, would you do it differently?")
    assert r.saga.of(p.Overheard)[-1].line == '[Aiva said to him: "I like it. Saga, would you do it differently?"]'
    st = r.saga.turn
    # he asked for this exchange and is in the room: spoken aloud even though the mode is private
    assert st is not None and st.kind == "sister" and st.audible is True and st.addressee == "aiva"
    assert st.cue == sister_cue("Aiva", False) and st.text is None and st.reply_to == r.aiva.turn.turn
    assert r.floor.gate_active is True and r.gates == [True, False, True]
    assert r.floor.exchange_state.rounds == 1

    r.finish_audible(r.saga, "Not really, no.", "aiva")
    assert r.aiva.last(p.Overheard).line == '[Saga said to you: "Not really, no."]'
    assert r.floor.holder is None and r.floor.queue == [] and r.aiva.turn.kind == "user"   # no new turn: no question
    rows = r.transcript()
    assert rows[-1]["speaker"] == "saga" and rows[-1]["audible"] is True and rows[-1]["addressee"] == "aiva" and rows[-1]["kind"] == "sister"

    await r.floor.submit_user_turn("Aiva, and you?")
    r.finish_audible(r.aiva, "I think so too, Saga.")                    # names her but asks nothing
    assert len(r.saga.of(p.TurnMsg)) == 1
    await r.floor.submit_user_turn("Aiva, sure?")
    r.finish_audible(r.aiva, "Are you sure?")                            # asks him, not her
    assert len(r.saga.of(p.TurnMsg)) == 1 and r.gates == [True, False, True, False, True, False, True, False]


async def test_round_cap_sets_backoff():
    r = Rig(cfg=ExchangeConfig(max_rounds=2, backoff_s=120))
    await r.floor.submit_user_turn("Aiva, talk to her for a bit")
    r.finish_audible(r.aiva, "Saga, ready?")                              # round 1 → saga
    assert r.saga.turn.kind == "sister"
    r.finish_audible(r.saga, "Sure, Aiva. You?", "aiva")                  # round 2 → aiva (he started it: aloud)
    assert r.aiva.turn.kind == "sister" and r.floor.exchange_state.rounds == 2
    r.finish_audible(r.aiva, "Yes. Saga, more?", "saga")                  # cap hit → stop, backoff
    assert len(r.saga.of(p.TurnMsg)) == 1 and r.floor.queue == []
    ex = r.floor.exchange_state
    assert ex.rounds == 0 and ex.backoff_until == r.clock.t + 120


# --- private / audible mode -----------------------------------------------------------------------------

async def test_private_default_persisted_and_flip_broadcasts():
    r = Rig()
    assert r.floor.private is True and r.mem.kv_get("sister_talk") == "private" and r.floor.mode == "private"
    assert r.floor.set_private(False) is True
    assert r.aiva.last(p.Mode).sister_talk == "audible" and r.saga.last(p.Mode).sister_talk == "audible"
    assert r.mem.kv_get("sister_talk") == "audible"
    # a new Floor on the same memory picks the persisted mode up (constructor arg loses)
    again = Floor(r.mem, private=True, log=r.logs.append, now=r.clock, clients={})
    assert again.private is False

    await r.floor.submit_user_turn("Aiva, ask her something")
    r.finish_audible(r.aiva, "Saga, what do you think?")
    st = r.saga.turn
    assert st.kind == "sister" and st.audible is True and st.cue == sister_cue("Aiva", False)
    assert r.floor.gate_active is True and r.gates == [True, False, True]


def test_audible_refused_without_a_voice():
    r = Rig()
    r.aiva.info.has_voice = False
    r.saga.info.has_voice = False
    assert r.floor.set_private(False) is False and r.floor.private is True and r.aiva.of(p.Mode) == []


# --- user speech vs queued sister turns ----------------------------------------------------------------------

async def test_user_speaking_drops_queued_sister_and_initiative_turns():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, hello")                        # aiva holds
    r.floor.on_response("aiva", r.aiva.turn.turn, "start")
    t = r.floor.want_floor("saga", "initiative", to="aiva")
    assert t is not None and [q.kind for q in r.floor.queue] == ["initiative"]
    r.clock.advance(1)
    r.floor.note_user_speaking(True)
    assert r.floor.queue == []
    r.clock.advance(1)
    r.floor.note_user_speaking(False)
    r.floor.on_speaking("aiva", r.aiva.turn.turn, True)
    r.floor.on_response("aiva", r.aiva.turn.turn, "end")
    r.floor.on_speaking("aiva", r.aiva.turn.turn, False)
    r.floor.on_said("aiva", r.aiva.turn.turn, "Saga, what do you think?", "him", private=False)
    assert r.saga.turn is None                                            # he spoke since → no continuation


async def test_queued_user_turn_waits_then_grants():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, one")
    await r.floor.submit_user_turn("Saga, two")
    assert r.saga.turn is None and [q.mind for q in r.floor.queue] == ["saga"]
    r.finish_audible(r.aiva, "One.")
    assert r.saga.turn.text == "Saga, two" and r.floor.holder == "saga"


# --- interrupt -----------------------------------------------------------------------------------------------

async def test_interrupt_goes_to_holder_only_and_drops_sister_turns():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, tell me a long story")
    tid = r.aiva.turn.turn
    r.floor.on_response("aiva", tid, "start")
    r.floor.on_speaking("aiva", tid, True)
    r.floor.want_floor("saga", "initiative", to="aiva")
    assert len(r.floor.queue) == 1
    assert r.floor.interrupt() is True
    assert r.aiva.last(p.Interrupt).turn == tid and r.saga.of(p.Interrupt) == [] and r.floor.queue == []
    assert r.floor.holder == "aiva"                                       # until she reports stopping
    r.floor.on_speaking("aiva", tid, False)
    assert r.floor.holder is None and r.floor.gate_active is False
    r.floor.on_said("aiva", tid, "Once upon a time, Saga?", "him", private=False, interrupted=True)
    assert r.transcript()[-1]["interrupted"] is True and r.saga.turn is None
    assert r.floor.interrupt() is False                                   # nothing to interrupt


async def test_interrupt_fallback_release_after_2s():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, go")
    r.floor.on_response("aiva", r.aiva.turn.turn, "start")
    r.floor.interrupt()
    r.floor.tick(r.clock.advance(1.9))
    assert r.floor.holder == "aiva"
    r.floor.tick(r.clock.advance(0.2))
    assert r.floor.holder is None and r.floor.gate_active is False


# --- client death ---------------------------------------------------------------------------------------------

async def test_client_lost_releases_with_gate_tail():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, hi")
    tid = r.aiva.turn.turn
    r.floor.on_response("aiva", tid, "start")
    r.floor.on_speaking("aiva", tid, True)
    r.floor.on_client_lost("aiva")
    assert r.floor.holder is None and r.floor.present == ["saga"] and not r.floor.busy
    assert r.floor.gate_active is True                                    # 3 s tail
    r.floor.tick(r.clock.advance(2.9))
    assert r.floor.gate_active is True
    r.floor.tick(r.clock.advance(0.2))
    assert r.floor.gate_active is False and r.gates == [True, False]


async def test_client_lost_reroutes_queued_user_turn_with_absent_cue():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, first")
    r.floor.on_response("aiva", r.aiva.turn.turn, "start")
    await r.floor.submit_user_turn("Aiva, second question")
    assert [q.mind for q in r.floor.queue] == ["aiva"]
    r.floor.on_client_lost("aiva")
    assert r.saga.last(p.Absent).slug == "aiva"
    t = r.saga.turn
    assert t.kind == "user" and t.text == "Aiva, second question" and t.cue == absent_cue("Aiva") and t.audible is True
    assert r.floor.holder == "saga" and r.floor.queue == []
    assert [e["kind"] for e in r.mem.events_since(0)] == ["absent"]


async def test_client_lost_reroutes_current_turn_she_never_started():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, are you there")
    r.floor.on_client_lost("aiva")                                        # no response start yet
    t = r.saga.turn
    assert t.kind == "user" and t.text == "Aiva, are you there" and t.cue == absent_cue("Aiva")


# --- timers -----------------------------------------------------------------------------------------------------

async def test_grace_release_for_tool_only_reply():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, move the window")
    tid = r.aiva.turn.turn
    r.floor.on_response("aiva", tid, "start")
    r.floor.on_tool("aiva", tid, +1)
    r.floor.on_response("aiva", tid, "end")                               # tools still running → hold
    r.floor.tick(r.clock.advance(1.0))
    assert r.floor.holder == "aiva"
    r.floor.on_tool("aiva", tid, -1)                                      # re-run expected, not a release
    r.floor.tick(r.clock.advance(2.0))
    assert r.floor.holder == "aiva"
    r.floor.on_response("aiva", tid, "start")
    r.floor.on_response("aiva", tid, "end")                               # said nothing → grace
    r.floor.tick(r.clock.advance(1.0))
    assert r.floor.holder == "aiva" and r.floor.gate_active is True
    r.floor.tick(r.clock.advance(0.6))
    assert r.floor.holder is None and r.floor.gate_active is False
    # her (empty) said arrives late: still accepted, closes the turn, no crash
    r.floor.on_said("aiva", tid, "", "him", private=False)
    assert not r.floor.busy


async def test_speech_within_grace_cancels_it():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, hi")
    tid = r.aiva.turn.turn
    r.floor.on_response("aiva", tid, "start")
    r.floor.on_response("aiva", tid, "end")
    r.floor.on_speaking("aiva", tid, True)                                # TTS arrived after the LLM finished
    r.floor.tick(r.clock.advance(5))
    assert r.floor.holder == "aiva" and r.floor.gate_active is True
    r.floor.on_speaking("aiva", tid, False)
    assert r.floor.holder is None


async def test_turn_cap_120s():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, ramble")
    tid = r.aiva.turn.turn
    r.floor.on_response("aiva", tid, "start")
    r.floor.on_speaking("aiva", tid, True)
    r.floor.tick(r.clock.advance(119))
    assert r.floor.holder == "aiva"
    r.floor.tick(r.clock.advance(1))
    assert r.floor.holder is None and r.floor.gate_active is False and r.aiva.last(p.Interrupt).turn == tid


async def test_no_response_start_releases_after_ack_timeout():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, hi")
    await r.floor.submit_user_turn("Saga, hi")
    r.floor.tick(r.clock.advance(10.1))
    assert r.floor.holder == "saga" and r.saga.turn.text == "Saga, hi"   # aiva released, saga granted


async def test_release_waits_briefly_for_said_before_next_grant():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, one")
    await r.floor.submit_user_turn("Saga, two")
    tid = r.aiva.turn.turn
    r.floor.on_response("aiva", tid, "start")
    r.floor.on_speaking("aiva", tid, True)
    r.floor.on_response("aiva", tid, "end")
    r.floor.on_speaking("aiva", tid, False)                               # released, gate down, said pending
    assert r.floor.holder is None and r.floor.gate_active is False and r.saga.turn is None
    r.floor.on_said("aiva", tid, "One.", "him", private=False)
    assert r.saga.turn is not None
    assert [type(m).__name__ for m in r.saga.sent[-2:]] == ["Overheard", "TurnMsg"]   # her line before Saga's run


async def test_said_wait_expires_if_it_never_comes():
    r = Rig()
    await r.floor.submit_user_turn("Aiva, one")
    await r.floor.submit_user_turn("Saga, two")
    tid = r.aiva.turn.turn
    r.floor.on_response("aiva", tid, "start")
    r.floor.on_speaking("aiva", tid, True)
    r.floor.on_response("aiva", tid, "end")
    r.floor.on_speaking("aiva", tid, False)
    r.floor.tick(r.clock.advance(1.1))
    assert r.saga.turn is not None


# --- absence ----------------------------------------------------------------------------------------------------

async def test_absent_addressee_routes_to_present_sister_with_cue():
    r = Rig()
    r.floor.on_client_lost("saga")
    d = await r.floor.submit_user_turn("Saga, are you there?")
    assert d.addressee == "saga"
    assert r.aiva.last(p.Absent) == p.Absent(id=r.aiva.last(p.Absent).id, ts=T0, slug="saga", name="Saga")
    t = r.aiva.turn
    assert t.kind == "user" and t.text == "Saga, are you there?" and t.cue == absent_cue("Saga")
    row = r.transcript()[0]
    assert row["addressee"] == "aiva" and row["kind"] == "named|absent:saga"
    ev = r.mem.events_since(0)
    assert len(ev) == 1 and ev[0]["slug"] == "saga" and ev[0]["kind"] == "absent" and ev[0]["relayed_to"] == "aiva"


async def test_both_with_one_absent_goes_to_the_present_one():
    r = Rig()
    r.floor.on_client_lost("saga")
    await r.floor.submit_user_turn("you two, dinner?")
    t = r.aiva.turn
    assert t.kind == "user" and t.cue == absent_cue("Saga", both=True) and r.floor.queue == []


async def test_nobody_home_logs_only():
    r = Rig()
    r.floor.on_client_lost("saga")
    r.floor.on_client_lost("aiva")
    d = await r.floor.submit_user_turn("Aiva?")
    assert d.addressee == "aiva" and r.floor.holder is None
    row = r.transcript()[0]
    assert row["kind"].endswith("|unrouted") and row["turn_id"] is None and any("nobody home" in l for l in r.logs)


# --- want_floor: announces, initiatives, the daily chat cap --------------------------------------------------------

async def test_announce_waits_for_a_quiet_moment_after_user_speech():
    r = Rig()
    r.floor.note_user_speaking(True)
    r.floor.note_user_speaking(False)
    t = r.floor.want_floor("aiva", "announce")
    assert t is not None and t.kind == "announce" and t.audible is True and r.floor.holder is None
    r.floor.tick(r.clock.advance(10.0))
    assert r.floor.holder is None                       # he spoke 10 s ago: still a conversation
    r.floor.tick(r.clock.advance(40.0))                 # 50 s of quiet → the news may land
    assert r.floor.holder == "aiva" and r.aiva.turn.kind == "announce" and r.aiva.turn.text is None
    assert r.floor.gate_active is True


def test_daily_chat_cap_is_house_wide():
    r = Rig(cfg=ExchangeConfig(chats_per_day=2))
    t1 = r.floor.want_floor("aiva", "initiative", to="saga")
    assert t1 is not None and t1.kind == "initiative" and t1.addressee == "saga" and t1.audible is False
    assert t1.cue == chat_cue("Saga") and r.aiva.turn.turn == t1.id and r.aiva.turn.audible is False
    assert r.floor.gate_active is False
    t2 = r.floor.want_floor("saga", "initiative", to="aiva")               # the other sister's opener counts too
    assert t2 is not None and r.floor.queue == [t2]
    assert r.floor.want_floor("aiva", "initiative", to="saga") is None
    assert r.mem.day_count("sister_chats") == 2 and any("daily cap" in l for l in r.logs)
    # announces and initiatives to him are not capped
    assert r.floor.want_floor("aiva", "announce") is not None
    assert r.floor.want_floor("aiva", "initiative", to="him").addressee == "him"
    r.clock.advance(86400)
    assert r.floor.want_floor("aiva", "initiative", to="saga") is not None
    assert r.mem.day_count("sister_chats") == 1


def test_want_floor_refusals():
    r = Rig()
    r.floor.on_client_lost("saga")
    assert r.floor.want_floor("saga", "announce") is None
    assert r.floor.want_floor("aiva", "initiative", to="saga") is None      # she isn't there to talk to
    assert r.floor.request(Turn(id=0, mind="nobody", kind="announce", audible=True)) is False


async def test_initiative_chat_round_trips_privately():
    r = Rig()
    r.floor.want_floor("aiva", "initiative", to="saga")
    tid = r.aiva.turn.turn
    r.floor.on_response("aiva", tid, "start")
    r.floor.on_response("aiva", tid, "end")
    r.floor.on_said("aiva", tid, "Saga, do you ever get bored?", "saga", private=True)
    assert r.saga.of(p.Overheard)[-1].line == '[Aiva said to you: "Saga, do you ever get bored?"]'
    assert r.saga.turn.kind == "sister" and r.saga.turn.audible is False and r.gates == []
    r.finish_private(r.saga, "Sometimes. You?", "aiva")
    assert r.aiva.turn.kind == "sister" and r.aiva.turn.cue == sister_cue("Saga", True)
    rows = r.transcript()
    assert [(x["speaker"], x["addressee"], x["audible"]) for x in rows] == [("aiva", "saga", False), ("saga", "aiva", False)]


def test_turn_ids_are_monotonic_across_floors():
    conn = sqlite3.connect(":memory:")
    r = Rig(conn=conn)
    t = r.floor.want_floor("aiva", "announce")
    again = Floor(r.mem, log=r.logs.append, now=r.clock, clients={"aiva": r.aiva})
    t2 = again.want_floor("aiva", "announce")
    assert t2.id > t.id


def test_holders_speaking_off_without_a_turn_id_still_releases():
    """A `speaking off` that lost its turn id (or carries a stale one) from the CURRENT holder must
    release the floor and drop the gate instead of pinning the mic until the 120 s cap."""
    import sqlite3
    from modules.house_memory import HouseMemory
    from modules.house_floor import Floor
    from modules.sisters import Turn

    class C:
        def __init__(self, slug, name):
            from modules.house_floor import ClientInfo
            self.info = ClientInfo(slug=slug, name=name)
            self.sent = []

        def send(self, msg):
            self.sent.append(msg)

    clock = {"t": 1000.0}
    mem = HouseMemory(sqlite3.connect(":memory:"))
    aiva = C("aiva", "Aiva")
    floor = Floor(mem, log=lambda m: None, now=lambda: clock["t"], clients={"aiva": aiva})
    assert floor.request(Turn(id=0, mind="aiva", kind="announce", audible=True, addressee="him"))
    tid = floor.current_turn.id
    floor.on_response("aiva", tid, "start")
    floor.on_speaking("aiva", tid, True)
    assert floor.gate_active
    floor.on_response("aiva", tid, "end")
    floor.on_speaking("aiva", None, False)          # id lost on the way
    assert floor.holder is None and not floor.gate_active


def test_queued_announces_from_the_same_sister_collapse_into_one():
    """Reading's session note and becoming's 'what struck you' both announce; her context already
    holds both notes, so only one announce turn should be queued while she waits for the floor."""
    import sqlite3
    from modules.house_memory import HouseMemory
    from modules.house_floor import ClientInfo, Floor
    from modules.sisters import Turn

    class C:
        def __init__(self, slug, name):
            self.info = ClientInfo(slug=slug, name=name)
            self.sent = []

        def send(self, msg):
            self.sent.append(msg)

    clock = {"t": 1000.0}
    mem = HouseMemory(sqlite3.connect(":memory:"))
    aiva, saga = C("aiva", "Aiva"), C("saga", "Saga")
    floor = Floor(mem, log=lambda m: None, now=lambda: clock["t"], clients={"aiva": aiva, "saga": saga})
    # Saga holds the floor; Aiva's two announces arrive meanwhile
    assert floor.request(Turn(id=0, mind="saga", kind="announce", audible=True, addressee="him"))
    assert floor.holder == "saga"
    assert floor.want_floor("aiva", "announce") is not None
    assert floor.want_floor("aiva", "announce") is None          # collapsed, not a second turn
    assert [t.mind for t in floor.queue] == ["aiva"]


# --- regressions from the first live sister exchange (2026-09-11 20:13) ------------------------------

def _rig2():
    import sqlite3
    from modules.house_memory import HouseMemory
    from modules.house_floor import ClientInfo, Floor
    from modules.sisters import ExchangeConfig

    class C:
        def __init__(self, slug, name):
            self.info = ClientInfo(slug=slug, name=name)
            self.sent = []

        def send(self, msg):
            self.sent.append(msg)

    clock = {"t": 1_700_000_000.0 + 12 * 3600}
    mem = HouseMemory(sqlite3.connect(":memory:"))
    aiva, saga = C("aiva", "Aiva"), C("saga", "Saga")
    floor = Floor(mem, private=True, log=lambda m: None, now=lambda: clock["t"],
                  clients={"aiva": aiva, "saga": saga}, cfg=ExchangeConfig(quiet_hours=(0, 0)))
    return floor, mem, aiva, saga, clock


def _speak(floor, slug, turn, text, clock, private=False):
    floor.on_response(slug, turn.id, "start")
    if not private:
        floor.on_speaking(slug, turn.id, True)
    floor.on_said(slug, turn.id, text, addressee=turn.addressee, private=private)
    floor.on_response(slug, turn.id, "end")
    if not private:
        floor.on_speaking(slug, turn.id, False)
    clock["t"] += 0.5
    floor.tick(clock["t"])


@pytest.mark.asyncio
async def test_stale_boot_announce_is_dropped_when_she_speaks_and_does_not_block_her_sister_turn():
    floor, mem, aiva, saga, clock = _rig2()
    # Saga is mid-reply when Aiva connects and queues her boot greeting; he then greets Aiva himself
    await floor.submit_user_turn("Saga, can you bring up your sister?")
    ts = floor.current_turn
    assert floor.want_floor("aiva", "announce") is not None          # queued behind Saga's turn
    _speak(floor, "saga", ts, "Aiva's starting up in her own window.", clock)
    assert any(q.kind == "announce" for q in floor.queue)            # held: someone spoke just now
    await floor.submit_user_turn("Hey, Aiva.")                       # his turn outranks the announce
    t = floor.current_turn
    assert t.mind == "aiva" and t.kind == "user"
    _speak(floor, "aiva", t, "Hey, Frank. I'm Aiva.", clock)
    assert not any(q.kind == "announce" for q in floor.queue)       # her greeting is spent
    await floor.submit_user_turn("Wanna talk to your sister a bit?")
    t2 = floor.current_turn
    _speak(floor, "aiva", t2, "Yeah, Saga's here. Hey, Saga. Fancy joining us?", clock)
    t3 = floor.current_turn
    assert t3 is not None and t3.mind == "saga" and t3.kind == "sister"
    _speak(floor, "saga", t3, "Hey, Aiva. I'm here. What're you in the mood to talk about?", clock)
    t4 = floor.current_turn                                          # round 2 comes back to Aiva
    assert t4 is not None and t4.mind == "aiva" and t4.kind == "sister" and t4.addressee == "saga"


@pytest.mark.asyncio
async def test_announce_queued_mid_turn_survives_her_reply():
    # 2026-09-26: "Yes" → answer_task → the task's next step asks to run a command (announce queued
    # while she's still replying) → her reply was said → the question was dropped as "spoken
    # already" and the task sat waiting for a yes nobody was asked for
    floor, mem, aiva, saga, clock = _rig2()
    await floor.submit_user_turn("Aiva, yes.")
    t = floor.current_turn
    clock["t"] += 1.0
    assert floor.want_floor("aiva", "announce") is not None
    _speak(floor, "aiva", t, "Continuing with it now.", clock)
    assert any(q.kind == "announce" and q.mind == "aiva" for q in floor.queue)


@pytest.mark.asyncio
async def test_a_question_folded_into_an_older_announce_survives_her_reply():
    # 2026-10-14 18:06: a task question queued (#2167), he said "cool go", the task's NEXT question
    # arrived mid-reply and collapsed into #2167, and #2167 was then dropped as "spoken already" —
    # the task waited on him and nobody told him
    floor, mem, aiva, saga, clock = _rig2()
    await floor.submit_user_turn("Saga, hi.")
    ts = floor.current_turn
    assert floor.want_floor("aiva", "announce") is not None          # the first question, queued
    _speak(floor, "saga", ts, "Hi.", clock)
    await floor.submit_user_turn("Aiva, cool go.")                   # his answer outranks it
    t = floor.current_turn
    assert t.mind == "aiva" and t.kind == "user"
    clock["t"] += 1.0
    assert floor.want_floor("aiva", "announce") is None              # the next question: collapsed
    _speak(floor, "aiva", t, "Going ahead.", clock)
    assert any(q.kind == "announce" and q.mind == "aiva" for q in floor.queue)


@pytest.mark.asyncio
async def test_exchange_he_started_is_audible_in_private_mode_but_their_own_chat_is_not():
    floor, mem, aiva, saga, clock = _rig2()
    await floor.submit_user_turn("Aiva, talk to your sister.")
    t = floor.current_turn
    _speak(floor, "aiva", t, "Saga, what do you think about rain?", clock)
    t2 = floor.current_turn
    assert t2.kind == "sister" and t2.audible is True                 # he is in the room and asked
    assert floor.gate_active                                          # so the mic is gated while she speaks
    assert "hear" in (t2.cue or "").lower()
    _speak(floor, "saga", t2, "I like it. Do you, Aiva?", clock)
    t3 = floor.current_turn
    assert t3.mind == "aiva" and t3.audible is True                   # still rooted in his turn
    _speak(floor, "aiva", t3, "Sometimes.", clock)
    assert floor.current_turn is None
    # a chat Saga starts on her own stays private in private mode
    clock["t"] += 200
    floor.tick(clock["t"])
    t4 = floor.want_floor("saga", "initiative", to="aiva")
    assert t4 is not None and floor.current_turn is t4 and t4.audible is False
    assert not floor.gate_active
    _speak(floor, "saga", t4, "Aiva, do you ever get bored at night?", clock, private=True)
    t5 = floor.current_turn
    assert t5.mind == "aiva" and t5.kind == "sister" and t5.audible is False


@pytest.mark.asyncio
async def test_reply_that_opens_with_her_sisters_name_is_addressed_to_her():
    floor, mem, aiva, saga, clock = _rig2()
    await floor.submit_user_turn("Aiva, say hi to Saga.")
    t = floor.current_turn
    _speak(floor, "aiva", t, "Hey, Saga. Fancy joining us?", clock)
    row = [r for r in mem.recent_transcript(limit=5) if r["speaker"] == "aiva"][0]
    assert row["addressee"] == "saga"
    from modules import house_proto as proto
    lines = [m.line for m in saga.sent if isinstance(m, proto.Overheard)]
    assert any(l.startswith("[Aiva said to you:") for l in lines)
    # but a reply that merely mentions her mid-sentence stays addressed to him
    t2 = floor.current_turn
    assert t2.mind == "saga"
    _speak(floor, "saga", t2, "Sure, I'm in.", clock)
    await floor.submit_user_turn("Aiva, how's your day?")
    t3 = floor.current_turn
    _speak(floor, "aiva", t3, "Good — I was reading with Saga earlier.", clock)
    row = [r for r in mem.recent_transcript(limit=1)][0]
    assert row["speaker"] == "aiva" and row["addressee"] == "him"


# --- per-sister voice mute (dashboard control) --------------------------------------------------

async def test_muted_sister_turns_are_forced_silent_and_release_cleanly():
    r = Rig()
    assert r.floor.set_muted("saga", True) is True
    assert r.floor.is_muted("saga") and not r.floor.is_muted("aiva")
    assert r.mem.kv_get("muted_slugs") == "saga"        # persisted for reconnect / house restart

    # a turn addressed to HIM (normally audible) is forced non-audible: no sound, no mic gate
    await r.floor.submit_user_turn("Saga, what's the weather?")
    t = r.saga.turn
    assert t is not None and t.audible is False
    assert r.floor.gate_active is False and r.gates == []    # the mic gate is never raised for her

    # a non-audible turn releases on response end (end:private) — it does NOT hang waiting to "speak"
    r.finish_private(r.saga, "sunny", "him")
    assert r.floor.holder is None and r.floor.gate_active is False and not r.floor.busy

    # unmuting restores her voice on the next turn, and clears the persisted flag
    assert r.floor.set_muted("saga", False) is True
    assert not r.floor.is_muted("saga") and r.mem.kv_get("muted_slugs") == ""
    await r.floor.submit_user_turn("Saga, you back?")
    assert r.saga.turn.audible is True


async def test_mute_persists_across_a_fresh_floor_on_the_same_db():
    conn = sqlite3.connect(":memory:")
    r1 = Rig(conn=conn)
    r1.floor.set_muted("aiva", True)
    r2 = Rig(conn=conn)                                  # a new Floor (house restart) on the same house.db
    assert r2.floor.is_muted("aiva") and not r2.floor.is_muted("saga")


# --- errands: he sends one sister to the other (2026-10-14 21:30, "I had to play middle man") -------

def _errand_rig(*present):
    floor, mem, aiva, saga, clock = _rig2()
    for slug in ("aiva", "saga"):
        if slug not in present:
            floor.clients.pop(slug)
    return floor, mem, aiva, saga, clock


async def test_bring_her_in_and_ask_her_runs_to_the_end_without_him():
    floor, mem, aiva, saga, clock = _errand_rig("aiva")
    await floor.submit_user_turn("Ok bring saga in here and ask her if she wants to change anything about "
                                 "the look and the feel and then let me know")
    t = floor.current_turn
    assert t.mind == "aiva" and "isn't running" in t.cue
    _speak(floor, "aiva", t, "Saga's starting up now; she'll be here in a moment.", clock)
    assert floor.current_turn is None and not floor.queue           # about her, not to her: no round

    floor.add_client("saga", saga)                                   # she booted
    ask = floor.current_turn
    assert ask.mind == "aiva" and ask.kind == "sister" and ask.addressee == "saga" and ask.audible
    assert "is here now" in ask.cue and "look and the feel" in ask.cue
    _speak(floor, "aiva", ask, "Saga, he wants to know if you'd change anything about how your "
                               "constellation looks and feels.", clock)
    answer = floor.current_turn
    assert answer.mind == "saga" and answer.kind == "sister" and answer.addressee == "aiva"
    _speak(floor, "saga", answer, "Tiny stars, two to four pixels, dim white with a blue-violet tint, and "
                                  "dim to thirty percent while I speak.", clock)
    report = floor.current_turn
    assert report.mind == "aiva" and report.kind == "report" and report.addressee == "him"
    assert "Saga has answered" in report.cue and report.audible
    _speak(floor, "aiva", report, "Saga wants smaller, dimmer stars that fade while she talks. Saga, "
                                  "want me to make that change?", clock)
    assert floor.current_turn is None and not floor.queue           # his move now, even with "Saga, …?"
    said = [(r["speaker"], r["addressee"], r["kind"]) for r in reversed(mem.recent_transcript())]
    assert said == [("user", "aiva", "named|absent:saga"), ("aiva", "him", "user"), ("aiva", "saga", "sister"),
                    ("saga", "aiva", "sister"), ("aiva", "him", "report")]


async def test_ask_her_means_ask_her_now_and_tell_him_after():
    floor, mem, aiva, saga, clock = _errand_rig("aiva", "saga")
    await floor.submit_user_turn("Aiva, ask her what she wants changed and let me know")
    t = floor.current_turn
    assert t.mind == "aiva" and t.kind == "user" and t.cue.startswith("[He wants you to put something to Saga")
    _speak(floor, "aiva", t, "Saga, what would you change about the constellation's look?", clock)
    a = floor.current_turn
    assert a.mind == "saga" and a.kind == "sister"
    _speak(floor, "saga", a, "Smaller stars and hairline links.", clock)
    r = floor.current_turn
    assert r.mind == "aiva" and r.kind == "report"
    _speak(floor, "aiva", r, "She wants smaller stars and hairline links.", clock)
    assert floor.current_turn is None and not floor.queue


async def test_a_chat_he_started_without_an_errand_gets_no_report():
    floor, mem, aiva, saga, clock = _errand_rig("aiva", "saga")
    await floor.submit_user_turn("Aiva, what do you think of the rain?")
    t = floor.current_turn
    assert t.cue is None
    _speak(floor, "aiva", t, "I like it. Saga, do you?", clock)
    a = floor.current_turn
    _speak(floor, "saga", a, "Only from indoors.", clock)
    assert floor.current_turn is None and not floor.queue


async def test_a_mid_reply_call_hands_her_the_floor():
    floor, mem, aiva, saga, clock = _errand_rig("aiva", "saga")
    await floor.submit_user_turn("Aiva, too vague, I need the actual look on the screen")
    t = floor.current_turn
    _speak(floor, "aiva", t, "Yeah, that's fair. Saga, describe the actual screen appearance: star size, "
                             "brightness and colour.", clock)
    a = floor.current_turn
    assert a is not None and a.mind == "saga" and a.kind == "sister"


async def test_an_old_errand_is_not_picked_up_and_a_possessive_is_not_a_call():
    floor, mem, aiva, saga, clock = _errand_rig("aiva")
    await floor.submit_user_turn("Saga, what do you think of the sky?")
    _speak(floor, "aiva", floor.current_turn, "She's off right now.", clock)
    clock["t"] += 200                                                # she came much later
    floor.add_client("saga", saga)
    assert floor.current_turn is None and not floor.queue
    await floor.submit_user_turn("Aiva, is she up?")
    t = floor.current_turn
    _speak(floor, "aiva", t, "Saga's already here, and the sky is on.", clock)
    assert floor.current_turn is None and not floor.queue           # about her: no sister round
    row = list(reversed(mem.recent_transcript()))[-1]
    assert (row["speaker"], row["addressee"]) == ("aiva", "him")
