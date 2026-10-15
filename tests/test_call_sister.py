"""call_sister: boots the other sister's console when she's off, just points at her when she's on."""
import pytest

from modules import capabilities
from modules.tools import TOOL_SCHEMAS, make_sister_handlers


class Params:
    def __init__(self, **arguments):
        self.arguments = arguments
        self.result = None

    async def result_callback(self, res):
        self.result = res


@pytest.mark.asyncio
async def test_present_sister_is_not_rebooted():
    booted = []
    h = make_sister_handlers(sister_name="Saga", is_present=lambda: True, boot=lambda: booted.append(1))
    p = Params()
    await h["call_sister"](p)
    assert p.result["success"] and p.result["present"] and not booted
    assert "Saga" in p.result["note"]


@pytest.mark.asyncio
async def test_absent_sister_is_booted_once_per_minute():
    booted = []
    clock = {"t": 1000.0}
    h = make_sister_handlers(sister_name="Saga", is_present=lambda: False, boot=lambda: booted.append(1),
                             now=lambda: clock["t"])
    p1 = Params()
    await h["call_sister"](p1)
    assert p1.result["booting"] and booted == [1]
    p2 = Params()
    await h["call_sister"](p2)                      # asked again right away → no second console
    assert p2.result["booting"] and booted == [1] and "already starting" in p2.result["note"]
    clock["t"] += 61
    p3 = Params()
    await h["call_sister"](p3)
    assert booted == [1, 1]


@pytest.mark.asyncio
async def test_boot_failure_is_reported_not_raised():
    def boom():
        raise RuntimeError("no bat")

    h = make_sister_handlers(sister_name="Aiva", is_present=lambda: False, boot=boom)
    p = Params()
    await h["call_sister"](p)
    assert p.result["success"] is False and "Aiva" in p.result["error"]


def test_call_sister_is_a_safe_auto_capability_with_a_schema():
    assert "call_sister" in capabilities.SAFE_TOOLS
    cap = capabilities.match("call your sister over")
    assert cap is not None and cap.name == "call_sister" and cap.tier == "auto"
    assert capabilities.match("get Saga here").name == "call_sister"
    assert any(s.name == "call_sister" for s in TOOL_SCHEMAS.standard_tools)


@pytest.mark.asyncio
async def test_restart_house_sends_to_the_house_and_reports():
    sent = []
    h = make_sister_handlers(sister_name="Saga", is_present=lambda: True, boot=lambda: None,
                             restart_house_send=lambda: sent.append(1) or True)
    p = Params()
    await h["restart_house"](p)
    assert p.result["success"] and sent == [1] and "deaf" in p.result["note"]
    h2 = make_sister_handlers(sister_name="Saga", is_present=lambda: True, boot=lambda: None,
                              restart_house_send=lambda: False)
    p2 = Params()
    await h2["restart_house"](p2)
    assert p2.result["success"] is False


def test_restart_message_round_trips_and_shutdown_reasons_exist():
    from modules import house_proto as proto
    m = proto.decode(proto.encode(proto.Restart(target="house")))
    assert isinstance(m, proto.Restart) and m.target == "house"
    assert proto.decode(proto.encode(proto.Shutdown(reason="house_restart"))).reason == "house_restart"


@pytest.mark.asyncio
async def test_set_sister_talk_reports_and_switches():
    sent = []
    state = {"mode": "private"}
    h = make_sister_handlers(sister_name="Saga", is_present=lambda: True, boot=lambda: None,
                             get_mode=lambda: state["mode"],
                             set_mode=lambda m: sent.append(m) or True)
    p = Params()
    await h["set_sister_talk"](p)                          # no mode → just report
    assert p.result["success"] and p.result["mode"] == "private" and not sent
    p2 = Params(mode="audible")
    await h["set_sister_talk"](p2)
    assert p2.result["success"] and sent == ["audible"] and "hears" in p2.result["note"]
    p3 = Params(mode="private")                            # already private → no message
    await h["set_sister_talk"](p3)
    assert p3.result["mode"] == "private" and sent == ["audible"]
    assert any(s.name == "set_sister_talk" for s in TOOL_SCHEMAS.standard_tools)
    assert "set_sister_talk" in capabilities.SAFE_TOOLS
