"""House server + client over real loopback sockets: two sisters connect, a user turn is routed,
reports flow back, private sister talk stays text-only, F10 reaches only the holder, events relay,
and a departed sister releases the floor."""
import asyncio
import sqlite3
import time

import pytest

from modules import house_proto as proto
from modules.house_client import HouseClient
from modules.house_floor import Floor
from modules.house_memory import HouseMemory
from modules.house_server import HouseServer
from modules.sisters import ExchangeConfig, overheard_reply_line, overheard_user_line


async def until(pred, timeout=4.0, step=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        await asyncio.sleep(step)
    return pred()


def of(msgs, cls):
    return [m for m in msgs if isinstance(m, cls)]


class Sister:
    def __init__(self, slug, name, state_path, has_voice=True):
        self.msgs = []
        self.welcomes = []
        self.client = HouseClient(
            proto.Hello(slug=slug, name=name, name_variants=[slug], loose_regex="", blurb=f"{name} blurb",
                        pid=1, version="t", has_voice=has_voice),
            on_message=self.msgs.append, log=lambda m: None, state_path=state_path,
            on_connected=self.welcomes.append, backoff=(0.05, 0.2))

    def turns(self):
        return of(self.msgs, proto.TurnMsg)

    def overheard(self):
        return of(self.msgs, proto.Overheard)


@pytest.mark.asyncio
async def test_two_sisters_full_session(tmp_path):
    state_path = str(tmp_path / "house_state.json")
    memory = HouseMemory(sqlite3.connect(":memory:"))
    log = []
    floor = Floor(memory, private=True, log=log.append,
                  cfg=ExchangeConfig(quiet_hours=(0, 0)))       # never quiet in this test
    server = HouseServer(floor, memory, port=0, state_path=state_path, log=log.append,
                         mic_state=lambda: {"awake": True, "mic_muted": False}, tick_s=0.02, ping_s=60)
    await server.start()
    st = proto.read_state(state_path)
    assert st and st["port"] == server.port and st["pid"]

    aiva = Sister("aiva", "Aiva", state_path)
    saga = Sister("saga", "Saga", state_path)
    aiva.client.start()
    saga.client.start()
    try:
        assert await until(lambda: floor.present == ["aiva", "saga"])
        assert await until(lambda: aiva.welcomes and saga.welcomes)
        assert saga.welcomes[0].mode == "private" and "aiva" in saga.welcomes[0].present
        # both were told about presence
        assert await until(lambda: any(s.present == ["aiva", "saga"] for s in of(aiva.msgs, proto.State)))

        # --- a user turn addressed to Saga -----------------------------------------------------
        text = "Saga, are you there?"
        d = await floor.submit_user_turn(text)
        assert d.addressee == "saga"
        assert await until(lambda: saga.turns())
        t1 = saga.turns()[0]
        assert t1.kind == "user" and t1.text == text and t1.audible is True and t1.addressee == "him"
        assert await until(lambda: aiva.overheard())
        assert aiva.overheard()[0].line == overheard_user_line("Saga", text)
        assert not aiva.turns()

        # Saga reports: start, speaks, says, ends, stops speaking → release
        saga.client.send(proto.Response(turn=t1.turn, phase="start"))
        saga.client.send(proto.Speaking(turn=t1.turn, on=True))
        assert await until(lambda: floor.gate_active)
        saga.client.send(proto.Said(turn=t1.turn, text="Yes, right here.", addressee="him", private=False))
        saga.client.send(proto.Response(turn=t1.turn, phase="end"))
        saga.client.send(proto.Speaking(turn=t1.turn, on=False))
        assert await until(lambda: floor.holder is None and not floor.gate_active)
        rows = memory.recent_transcript(limit=10)
        assert [(r["speaker"], r["addressee"], r["text"]) for r in reversed(rows)] == [
            ("user", "saga", text), ("saga", "him", "Yes, right here.")]
        assert await until(lambda: len(aiva.overheard()) >= 2)
        assert aiva.overheard()[1].line == overheard_reply_line("Saga", "him", "Yes, right here.")

        # --- Saga asks Aiva a question → private sister turn for Aiva, gate stays down ------------
        text2 = "Saga, ask your sister something."
        await floor.submit_user_turn(text2)
        assert await until(lambda: len(saga.turns()) >= 2)
        t2 = saga.turns()[1]
        saga.client.send(proto.Response(turn=t2.turn, phase="start"))
        saga.client.send(proto.Speaking(turn=t2.turn, on=True))
        saga.client.send(proto.Said(turn=t2.turn, text="Aiva, do you like rain?", addressee="him", private=False))
        saga.client.send(proto.Response(turn=t2.turn, phase="end"))
        saga.client.send(proto.Speaking(turn=t2.turn, on=False))
        assert await until(lambda: aiva.turns())
        t3 = aiva.turns()[0]
        # he started this exchange and is in the room, so even in private mode it is spoken aloud
        assert t3.kind == "sister" and t3.audible is True and t3.addressee == "saga"
        assert "hear" in (t3.cue or "").lower()
        assert floor.gate_active
        aiva.client.send(proto.Response(turn=t3.turn, phase="start"))
        aiva.client.send(proto.Said(turn=t3.turn, text="I do, actually.", addressee="saga", private=True))
        aiva.client.send(proto.Response(turn=t3.turn, phase="end"))
        assert await until(lambda: floor.holder is None)
        assert not floor.gate_active
        row = memory.recent_transcript(limit=1)[0]
        assert (row["speaker"], row["addressee"], row["audible"]) == ("aiva", "saga", 0)
        assert await until(lambda: any('[Aiva said to you: "I do, actually."]' == o.line
                                       for o in saga.overheard()))

        # --- F10 reaches only the holder -----------------------------------------------------------
        await floor.submit_user_turn("Saga, tell me a long story.")
        assert await until(lambda: len(saga.turns()) >= 3)
        t4 = saga.turns()[2]
        saga.client.send(proto.Response(turn=t4.turn, phase="start"))
        saga.client.send(proto.Speaking(turn=t4.turn, on=True))
        assert await until(lambda: floor.gate_active)
        assert floor.interrupt()
        assert await until(lambda: of(saga.msgs, proto.Interrupt))
        assert of(saga.msgs, proto.Interrupt)[0].turn == t4.turn
        assert not of(aiva.msgs, proto.Interrupt)
        saga.client.send(proto.Speaking(turn=t4.turn, on=False))
        saga.client.send(proto.Response(turn=t4.turn, phase="end"))
        saga.client.send(proto.Said(turn=t4.turn, text="Once upon a", addressee="him", interrupted=True))
        assert await until(lambda: floor.holder is None and not floor.gate_active)

        # --- events relay with actor + id ------------------------------------------------------------
        aiva.client.send(proto.Event(kind="tool_done", summary="looked something up",
                                     capability="look_up_info", payload={"tool": "search_web"}))
        assert await until(lambda: of(saga.msgs, proto.Event))
        ev = of(saga.msgs, proto.Event)[0]
        assert ev.actor == "aiva" and ev.event_id and ev.payload == {"tool": "search_web"}
        assert any(r["kind"] == "tool_done" and r["slug"] == "aiva" for r in memory.events_since(0))
        assert not of(aiva.msgs, proto.Event)

        # --- mode flip broadcast --------------------------------------------------------------------
        assert floor.set_private(False)
        assert await until(lambda: any(m.sister_talk == "audible" for m in of(aiva.msgs, proto.Mode)))

        # --- Saga leaves: floor knows, Aiva is told, her queued turns vanish ---------------------
        await saga.client.stop()
        assert await until(lambda: floor.present == ["aiva"])
        assert await until(lambda: any(s.present == ["aiva"] for s in of(aiva.msgs, proto.State)))
        # addressing the absent sister routes to the present one with the absent cue
        aiva_turns_before = len(aiva.turns())
        d = await floor.submit_user_turn("Saga, you still there?")
        assert await until(lambda: len(aiva.turns()) > aiva_turns_before)
        t5 = aiva.turns()[-1]
        assert "isn't running" in (t5.cue or "")
        assert any(a.slug == "saga" for a in of(aiva.msgs, proto.Absent))
    finally:
        await aiva.client.stop()
        await saga.client.stop()
        await server.stop()
    assert proto.read_state(state_path) is None


@pytest.mark.asyncio
async def test_second_hello_for_same_slug_is_rejected_while_first_is_alive(tmp_path):
    state_path = str(tmp_path / "house_state.json")
    memory = HouseMemory(sqlite3.connect(":memory:"))
    floor = Floor(memory, log=lambda m: None)
    server = HouseServer(floor, memory, port=0, state_path=state_path, log=lambda m: None, tick_s=0.02, ping_s=60)
    await server.start()
    first = Sister("aiva", "Aiva", state_path)
    second_log = []
    second = HouseClient(proto.Hello(slug="aiva", name="Aiva", pid=2), on_message=lambda m: None,
                         log=second_log.append, state_path=state_path, backoff=(0.05, 0.3))
    first.client.start()
    try:
        assert await until(lambda: floor.present == ["aiva"])
        second.start()
        assert await until(lambda: any("rejected" in m for m in second_log), timeout=5.0)
        assert floor.present == ["aiva"] and first.client.connected
    finally:
        await second.stop()
        await first.client.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_client_waits_for_the_house_and_replays_want_floor(tmp_path):
    state_path = str(tmp_path / "house_state.json")
    got = []
    log = []
    client = HouseClient(proto.Hello(slug="saga", name="Saga", pid=3), on_message=got.append,
                         log=log.append, state_path=state_path, backoff=(0.05, 0.1))
    client.start()
    await asyncio.sleep(0.15)
    assert not client.connected and any("waiting for the house" in m for m in log)
    assert client.send(proto.WantFloor(kind="announce")) is False        # queued while offline

    memory = HouseMemory(sqlite3.connect(":memory:"))
    floor = Floor(memory, log=lambda m: None)
    server = HouseServer(floor, memory, port=0, state_path=state_path, log=lambda m: None, tick_s=0.02, ping_s=60)
    await server.start()
    try:
        assert await until(lambda: client.connected, timeout=5.0)
        # the queued announce reached the floor → she was granted an announce turn
        assert await until(lambda: any(isinstance(m, proto.TurnMsg) and m.kind == "announce" for m in got))
    finally:
        await client.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_restart_request_reaches_the_house_callback(tmp_path):
    state_path = str(tmp_path / "house_state.json")
    memory = HouseMemory(sqlite3.connect(":memory:"))
    floor = Floor(memory, log=lambda m: None)
    asked = []
    server = HouseServer(floor, memory, port=0, state_path=state_path, log=lambda m: None, tick_s=0.02,
                         ping_s=60, on_restart=lambda slug, target: asked.append((slug, target)))
    await server.start()
    aiva = Sister("aiva", "Aiva", state_path)
    aiva.client.start()
    try:
        assert await until(lambda: aiva.client.connected)
        aiva.client.send(proto.Restart(target="house"))
        assert await until(lambda: asked == [("aiva", "house")])
        # a house-only restart notice leaves the sister connected (no quit, no self-restart)
        server.broadcast(proto.Shutdown(reason="house_restart"))
        assert await until(lambda: any(isinstance(m, proto.Shutdown) and m.reason == "house_restart"
                                       for m in aiva.msgs))
    finally:
        await aiva.client.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_control_client_is_not_a_sister_and_its_buttons_reach_the_house(tmp_path):
    state_path = str(tmp_path / "house_state.json")
    memory = HouseMemory(sqlite3.connect(":memory:"))
    floor = Floor(memory, log=lambda m: None)
    pressed = []
    server = HouseServer(floor, memory, port=0, state_path=state_path, log=lambda m: None, tick_s=0.02, ping_s=60)
    server.on_control = lambda action, target: pressed.append((action, target))
    await server.start()
    aiva = Sister("aiva", "Aiva", state_path)
    aiva.client.start()
    got = []
    panel = HouseClient(proto.Hello(slug="dashboard", name="Dashboard", role="control", has_voice=False),
                        on_message=got.append, log=lambda m: None, state_path=state_path, backoff=(0.05, 0.2))
    panel.start()
    try:
        assert await until(lambda: aiva.client.connected and panel.connected)
        assert floor.present == ["aiva"]                    # the panel is not a sister
        assert panel.welcome.present == ["aiva"]
        panel.client_send = panel.send
        panel.send(proto.Control(action="mute"))
        panel.send(proto.Control(action="sister_talk", target="audible"))
        panel.send(proto.Control(action="restart_sister", target="aiva"))
        assert await until(lambda: len(pressed) == 3)
        assert pressed == [("mute", None), ("sister_talk", "audible"), ("restart_sister", "aiva")]
        # per-sister delivery helper
        assert server.send_to("aiva", proto.Shutdown(reason="restart")) is True
        assert await until(lambda: any(isinstance(m, proto.Shutdown) and m.reason == "restart" for m in aiva.msgs))
        assert server.send_to("saga", proto.Shutdown(reason="restart")) is False
        # a panel message that isn't a control is ignored, not routed to the floor
        panel.send(proto.Said(turn=1, text="nope"))
        await asyncio.sleep(0.1)
        assert memory.recent_transcript(limit=5) == []
    finally:
        await panel.stop()
        await aiva.client.stop()
        await server.stop()
