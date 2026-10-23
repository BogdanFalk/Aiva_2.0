"""house_proto: every message type round-trips through encode/decode, the envelope shape is
{"t","id","ts",...}, validation rejects unknown types / oversize / bad JSON, speaker vectors are
rounded to 5 dp, unknown fields are ignored, and the state file helpers are atomic."""
import json

import pytest

from modules import house_proto as p


def test_every_type_round_trips():
    samples = [
        p.Hello(slug="saga", name="Saga", name_variants=["saga"], loose_regex="s[ae]+r?g+a+h?r?", blurb="new",
                capabilities=["web"], pid=42, version="0.10.0"),
        p.Response(turn=3, phase="start"), p.Tool(turn=3, delta=1), p.Speaking(turn=3, on=True),
        p.Said(turn=3, text="hi", addressee="him", private=False, interrupted=False),
        p.Event(kind="tool_done", summary="looked up the weather", capability="web"),
        p.WantFloor(kind="initiative", to="aiva"), p.Mode(sister_talk="private"),
        p.Sleep(), p.Wake(), p.Restart(target="house"), p.Control(action="mute"),
        p.Say(text="Aiva, move down"), p.Pong(), p.Bye(),
        p.Welcome(mode="private", state={"awake": True, "mic_muted": False}, present=["aiva"]),
        p.Reject(reason="duplicate"),
        p.TurnMsg(turn=4, kind="user", audible=True, addressee="him", text="hello", cue=None, wake_hint="saga"),
        p.Overheard(turn=4, line='[He said to Saga: "hello"]'),
        p.Affect(turn=4, affect={"arousal": -0.2, "valence": 0.1, "label": "tired",
                                 "emotion": {"labels": [{"name": "fatigue", "confidence": 0.8}], "bursts": []}}),
        p.Backchannel(clip="mhm", voice=False),
        p.Interrupt(turn=4), p.State(awake=False, mic_muted=True, present=[]),
        p.Absent(slug="saga", name="Saga"), p.Ping(), p.Shutdown(reason="quit"),
    ]
    assert {type(s) for s in samples} == set(p.MESSAGE_TYPES.values())
    for msg in samples:
        msg.id, msg.ts = 7, 1700000000.5
        line = p.encode(msg)
        assert line.endswith(b"\n") and line.count(b"\n") == 1
        back = p.decode(line)
        assert back == msg, msg


def test_envelope_shape_and_ordering():
    line = p.encode(p.Said(id=9, ts=1.5, turn=2, text="ok"))
    obj = json.loads(line)
    assert list(obj)[:3] == ["t", "id", "ts"]
    assert obj["t"] == "said" and obj["id"] == 9 and obj["ts"] == 1.5 and obj["turn"] == 2


def test_speaker_vec_rounded_to_5dp():
    line = p.encode(p.TurnMsg(turn=1, kind="user", audible=True, speaker_vec=[0.123456789, -1.000001, 2]))
    assert json.loads(line)["speaker_vec"] == [0.12346, -1.0, 2.0]


def test_decode_validation():
    with pytest.raises(p.ProtoError):
        p.decode(b'{"t": "nope"}\n')
    with pytest.raises(p.ProtoError):
        p.decode(b"not json\n")
    with pytest.raises(p.ProtoError):
        p.decode(b"[1,2]\n")
    with pytest.raises(p.ProtoError):
        p.decode(b'{"t": "said"}\n')                 # missing required `turn`/`text`
    with pytest.raises(p.ProtoError):
        p.decode(b'{"t": "ping", "junk": "' + b"x" * (p.MAX_LINE + 1) + b'"}\n')


def test_decode_ignores_unknown_fields_and_accepts_str():
    msg = p.decode('{"t": "ping", "id": 1, "ts": 0, "future_field": 5}')
    assert isinstance(msg, p.Ping) and msg.id == 1


def test_encode_rejects_non_messages():
    with pytest.raises(p.ProtoError):
        p.encode({"t": "ping"})


def test_direction_tables_cover_the_plan():
    assert {c.T for c in p.CHARACTER_TO_HOUSE} == {"hello", "response", "tool", "speaking", "said", "event",
                                                   "want_floor", "mode", "sleep", "wake", "restart", "control",
                                                   "say", "pong", "bye"}
    assert {c.T for c in p.HOUSE_TO_CHARACTER} == {"welcome", "reject", "turn", "overheard", "affect", "backchannel", "interrupt", "mode",
                                                   "state", "absent", "ping", "shutdown"}
    assert p.PROTOCOL == 1


def test_state_file_helpers(tmp_path):
    path = str(tmp_path / "house_state.json")
    assert p.read_state(path) is None
    p.write_state_atomic({"port": 8761, "pid": 1, "boot_id": "abc", "protocol": p.PROTOCOL}, path)
    assert p.read_state(path)["port"] == 8761
    assert not (tmp_path / "house_state.json.tmp").exists()
    (tmp_path / "house_state.json").write_text("{broken", encoding="utf-8")
    assert p.read_state(path) is None
    p.clear_state(path)
    assert p.read_state(path) is None
    assert p.STATE_FILE.endswith("house_state.json") and p.STATE_FILE.replace("\\", "/").endswith("aiva/house_state.json")
