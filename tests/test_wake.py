"""Wake-word names: the listener reports the model's basename ("hey_saga"), the floor routes by slug
("saga"). The house translates at `on_wake` with `wake_slugs`; this runs that hint end to end through
the ear and the floor (docs/ISSUE-wake-model-routing.md)."""
from modules.ear import MicController
from modules.wake import model_name, wake_slugs
from test_house_floor import FakeArbiter, Rig


def test_model_name_is_the_basename_without_extension():
    assert model_name("models/hey_saga.onnx") == "hey_saga"
    assert model_name(r"C:\aiva\models\hey_aiva.onnx") == "hey_aiva"
    assert model_name("hey_jarvis") == "hey_jarvis"         # a bare openWakeWord name


def test_wake_slugs_maps_model_names_and_leaves_strangers_out():
    slug_of = wake_slugs({"models/hey_aiva.onnx": "aiva", r"C:\aiva\models\hey_saga.onnx": "saga"})
    assert slug_of == {"hey_aiva": "aiva", "hey_saga": "saga"}
    assert slug_of.get("hey_jarvis") is None                # the fallback model calls nobody
    assert wake_slugs({}) == {}


async def _route(listener_name, line, **mic_kw):
    """What house.py does: on_wake → mic.wake(hint=slug) → confirmed → _on_user_turn → floor."""
    slug_of = wake_slugs({"models/hey_aiva.onnx": "aiva", "models/hey_saga.onnx": "saga"})
    r = Rig(arbiter=FakeArbiter(reply='{"to": "aiva", "why": "test"}'))
    m = MicController(ambient=True, log=lambda s: None, **mic_kw)
    m.wake(f"wake word {listener_name}", preroll=b"\x00\x00", verify=True, hint=slug_of.get(listener_name))
    m.confirm_wake("test")
    hint, m.wake_hint = m.wake_hint, None
    return r, await r.floor.submit_user_turn(line, wake_hint=hint)


async def test_hey_saga_routes_to_saga_when_the_line_names_nobody():
    r, d = await _route("hey_saga", "what time is it")
    assert (d.addressee, d.reason) == ("saga", "wake_word")
    assert r.saga.turn.wake_hint == "saga" and r.aiva.turn is None
    assert r.floor.arbiter.calls == []


async def test_a_spoken_name_still_beats_the_wake_word():
    _, d = await _route("hey_saga", "Aiva, what time is it")
    assert (d.addressee, d.reason) == ("aiva", "named")


async def test_an_unmapped_wake_model_gives_no_hint():
    r, d = await _route("hey_jarvis", "what time is it")
    assert d.reason != "wake_word" and r.floor.arbiter.calls
