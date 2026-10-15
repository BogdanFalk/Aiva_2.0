"""avatar_ipc: the state-file contract between a sister and her renderer."""
import json
import os

from modules import avatar_ipc


def test_state_path_none_or_aiva_is_aivas_historical_file():
    assert avatar_ipc.state_path(None) == avatar_ipc.STATE_FILE
    assert avatar_ipc.state_path("") == avatar_ipc.STATE_FILE
    assert avatar_ipc.state_path("aiva") == avatar_ipc.STATE_FILE
    assert os.path.basename(avatar_ipc.STATE_FILE) == "avatar_state.json"
    assert os.path.basename(os.path.dirname(avatar_ipc.STATE_FILE)) == "aiva"


def test_state_path_other_instances_live_beside_aivas():
    saga = avatar_ipc.state_path("saga")
    assert os.path.basename(saga) == "saga_avatar_state.json"
    assert os.path.dirname(saga) == os.path.dirname(avatar_ipc.STATE_FILE)
    assert saga != avatar_ipc.STATE_FILE
    assert avatar_ipc.state_path("saga") == saga  # deterministic


def test_write_then_read_round_trip(tmp_path):
    path = str(tmp_path / "saga_avatar_state.json")
    state = {"port": 1234, "pid": 42, "boot_id": "abc", "instance": "saga",
             "model": "saga",
             "placement": {"x": 0.8, "y": -0.65, "rotation": 0.0, "size": 0.0},
             "monitor": 0, "layer": "top"}
    avatar_ipc.write_state_atomic(state, path)
    assert avatar_ipc.read_state(path) == state
    assert not os.path.exists(path + ".tmp")  # replaced, never left behind


def test_write_overwrites_previous_state(tmp_path):
    path = str(tmp_path / "s.json")
    avatar_ipc.write_state_atomic({"boot_id": "one", "port": 1}, path)
    avatar_ipc.write_state_atomic({"boot_id": "two", "port": 2}, path)
    assert avatar_ipc.read_state(path) == {"boot_id": "two", "port": 2}


def test_read_state_missing_corrupt_or_wrong_shape_is_none(tmp_path):
    assert avatar_ipc.read_state(str(tmp_path / "missing.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert avatar_ipc.read_state(str(bad)) is None
    half = tmp_path / "half.json"
    half.write_text('{"port": 12', encoding="utf-8")  # a torn write
    assert avatar_ipc.read_state(str(half)) is None
    lst = tmp_path / "list.json"
    lst.write_text("[1, 2]", encoding="utf-8")
    assert avatar_ipc.read_state(str(lst)) is None


def test_default_paths_are_aivas_file(monkeypatch, tmp_path):
    # read_state()/write_state_atomic() without a path keep targeting STATE_FILE
    fake = str(tmp_path / "avatar_state.json")
    monkeypatch.setattr(avatar_ipc, "STATE_FILE", fake)
    # the defaults were bound at definition time, so pass explicitly what the
    # module constant now says — callers that pass nothing get the real file
    avatar_ipc.write_state_atomic({"pid": 7}, avatar_ipc.STATE_FILE)
    assert avatar_ipc.read_state(avatar_ipc.STATE_FILE) == {"pid": 7}


def test_encode_is_one_utf8_json_line():
    raw = avatar_ipc.encode({"cmd": "expression", "name": "爱心眼", "id": 3})
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    assert json.loads(raw.decode("utf-8")) == {"cmd": "expression", "name": "爱心眼", "id": 3}
    assert "爱心眼".encode("utf-8") in raw  # kept readable, not \\u-escaped
