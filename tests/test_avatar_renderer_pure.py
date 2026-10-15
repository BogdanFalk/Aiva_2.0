"""avatar_renderer: everything that runs without Qt/GL — placement lerps,
the lip-sync envelope, the IPC dispatcher against a fake window, the
retexture model3 builder, and the per-model fixup table."""
import json
import queue

import pytest

from modules import avatar_ipc
from modules.avatar_renderer import (
    MODEL_FIXUPS, Expressions, MouthEnvelope, Placement, PropArt, cdi3_path_for,
    dispatch, hue_rotate_rgb, make_retexture_model3, parse_rgb, sample_skin_rgb)


# --- Placement -------------------------------------------------------------------

def test_placement_clamps_targets():
    p = Placement()
    p.set_targets(x=5, y=-9, rotation=720, size=-500)
    assert p.target == {"x": 2.0, "y": -2.0, "rotation": 360.0, "size": -100.0}


def test_placement_lerps_over_move_secs_then_snaps():
    p = Placement()
    p.set_targets(x=1.0)
    p.tick(0.1)  # half of the 0.2 s move
    assert p.cur["x"] == pytest.approx(0.5)
    p.tick(0.1)
    assert p.cur["x"] == pytest.approx(1.0)
    p.set_targets(y=-1.0)
    p.tick(0.05)
    p.tick(1.0)  # overshoot the move window: snaps to target exactly
    assert p.cur == p.target
    assert p.cur["y"] == -1.0 and p.cur["x"] == 1.0


def test_placement_partial_targets_keep_the_rest():
    p = Placement({"x": 0.8, "y": -0.65, "rotation": 0, "size": 0})
    assert p.cur == {"x": 0.8, "y": -0.65, "rotation": 0.0, "size": 0.0}
    p.set_targets(size=-20)
    assert p.target["x"] == 0.8 and p.target["y"] == -0.65 and p.target["size"] == -20.0


def test_placement_ignores_garbage_saved_values():
    p = Placement({"x": "nope", "y": None, "size": "12"})
    assert p.cur == {"x": 0.0, "y": 0.0, "rotation": 0.0, "size": 12.0}
    assert Placement("not a dict").cur == {"x": 0.0, "y": 0.0, "rotation": 0.0, "size": 0.0}


# --- MouthEnvelope -----------------------------------------------------------------

def test_envelope_queued_audio_waits_for_speech_start():
    env = MouthEnvelope()
    env.feed([[0.1, 0.05, 0.5]] * 10)
    assert env.step(0.05) == (0.0, 0.0)          # nothing plays before "start"
    assert env._queued_secs == pytest.approx(0.5)  # ...and nothing was consumed
    env.speech("start")
    openness, _ = env.step(0.05)
    assert openness > 0.0


def test_envelope_openness_curve_gates_noise_and_saturates():
    env = MouthEnvelope(gain=2.5)
    assert env._openness(0.0) == 0.0
    assert env._openness(MouthEnvelope.NOISE_FLOOR) == 0.0
    assert 0.0 < env._openness(0.02) < env._openness(0.1) <= 1.0
    assert env._openness(1.0) == 1.0


def test_envelope_consumes_at_wall_clock_rate_then_underruns():
    env = MouthEnvelope()
    env.feed([[0.1, 0.1]])  # 100 ms of sound
    env.speech("start")
    env.step(0.05)
    assert env._queued_secs == pytest.approx(0.05)
    v_playing, _ = env.step(0.05)
    assert env._queued_secs == pytest.approx(0.0) and not env._queue
    v1, _ = env.step(0.05)  # underrun: transport has nothing to play either
    v2, _ = env.step(0.05)
    assert v_playing > v1 > v2 >= 0.0


def test_envelope_stop_clears_queue_and_mouth_relaxes():
    env = MouthEnvelope()
    env.feed([[0.1, 1.0]])
    env.speech("start")
    v, _ = env.step(0.05)
    assert v > 0
    env.speech("stop")
    assert not env._queue and env._queued_secs == 0
    v2, _ = env.step(0.05)
    assert v2 < v


def test_envelope_interrupt_halves_value_instantly():
    env = MouthEnvelope()
    env.feed([[0.15, 1.0]])
    env.speech("start")
    for _ in range(10):
        env.step(0.05)
    before = env._value
    assert before > 0.5
    env.speech("interrupt")
    assert env._value == pytest.approx(before * 0.5)
    assert not env._queue and env._speaking is False


def test_envelope_feed_skips_bad_chunks_and_caps_the_queue():
    env = MouthEnvelope(max_queued_secs=1.0)
    env.feed([["x", 0.1], [0.1], [0.1, 0], [0.1, -1], [0.1, 0.6], [0.1, 0.6], [0.1, 0.3]])
    # the second 0.6 s chunk would push the queue past 1.0 s and is dropped
    assert [c[1] for c in env._queue] == [0.6, 0.3]
    assert env._queued_secs == pytest.approx(0.9)
    # levels/brightness are clamped, brightness defaults to neutral
    env2 = MouthEnvelope()
    env2.feed([[5.0, 0.1, -3.0], [0.2, 0.1]])
    assert env2._queue[0][0] == 1.0 and env2._queue[0][2] == 0.0
    assert env2._queue[1][2] == 0.5


def test_envelope_brightness_becomes_mouth_form():
    bright = MouthEnvelope()
    bright.feed([[0.15, 1.0, 1.0]])
    bright.speech("start")
    dark = MouthEnvelope()
    dark.feed([[0.15, 1.0, 0.0]])
    dark.speech("start")
    for _ in range(10):
        _, f_bright = bright.step(0.05)
        _, f_dark = dark.step(0.05)
    assert f_bright > 0.5 and f_dark < -0.5
    silent = MouthEnvelope(form_scale=0.0)
    silent.feed([[0.15, 1.0, 1.0]])
    silent.speech("start")
    for _ in range(10):
        _, f = silent.step(0.05)
    assert f == pytest.approx(0.0)


# --- dispatch() against a fake window ------------------------------------------------

class FakeEnvelope:
    def __init__(self):
        self.fed = []
        self.states = []

    def feed(self, chunks):
        self.fed.append(chunks)

    def speech(self, state):
        self.states.append(state)


class FakeWindow:
    def __init__(self, exp_dir):
        self.instance = "saga"
        self.model_name = "saga"
        self.envelope = FakeEnvelope()
        self.expressions = Expressions(str(exp_dir))
        self.cmd_q = queue.Queue()
        self.quit_requested = False
        self._snap = {"ready": True, "x": 0.1, "y": -0.2, "rotation": 0.0,
                      "size": 5.0, "monitor": 1, "layer": "top"}

    def snapshot(self):
        return dict(self._snap)

    def request_quit(self):
        self.quit_requested = True


@pytest.fixture
def win(tmp_path):
    (tmp_path / "爱心眼.exp3.json").write_text(
        json.dumps({"Parameters": [{"Id": "aixinyan", "Value": 1.0, "Blend": "Add"}]}),
        encoding="utf-8")
    (tmp_path / "戴帽子.exp3.json").write_text(
        json.dumps({"Parameters": [{"Id": "maozi", "Value": 1.0, "Blend": "Add"}]}),
        encoding="utf-8")
    return FakeWindow(tmp_path)


def _queued(w):
    items = []
    while True:
        try:
            items.append(w.cmd_q.get_nowait())
        except queue.Empty:
            return items


def test_dispatch_hot_path_has_no_reply(win):
    assert dispatch(win, {"cmd": "mouth", "chunks": [[0.1, 0.05]]}) is None
    assert dispatch(win, {"cmd": "mouth"}) is None  # missing chunks -> empty feed
    assert win.envelope.fed == [[[0.1, 0.05]], []]
    assert dispatch(win, {"cmd": "speech", "state": "start"}) is None
    assert dispatch(win, {"cmd": "speech", "state": "interrupt", "id": 9}) is None
    assert win.envelope.states == ["start", "interrupt"]


def test_dispatch_ignores_idless_requests_and_junk(win):
    assert dispatch(win, {"cmd": "hello"}) is None
    assert dispatch(win, {"cmd": "quit"}) is None
    assert win.quit_requested is False
    assert dispatch(win, "not a dict") is None
    assert dispatch(win, [1, 2]) is None
    assert _queued(win) == []


def test_dispatch_hello_describes_the_body(win):
    reply = dispatch(win, {"id": 1, "cmd": "hello"})
    assert reply["id"] == 1 and reply["ok"] is True
    assert reply["version"] == avatar_ipc.PROTOCOL_VERSION
    assert reply["instance"] == "saga"
    assert reply["model"] == "saga"
    assert sorted(reply["expressions"]) == ["戴帽子", "爱心眼"]
    assert reply["ready"] is True


def test_dispatch_hello_instance_defaults_to_aiva_for_old_windows(win):
    del win.instance
    assert dispatch(win, {"id": 1, "cmd": "hello"})["instance"] == "aiva"


def test_dispatch_ping_and_position(win):
    assert dispatch(win, {"id": 2, "cmd": "ping"}) == {"id": 2, "ok": True}
    pos = dispatch(win, {"id": 3, "cmd": "get_position"})
    assert pos == {"id": 3, "ok": True, "x": 0.1, "y": -0.2, "rotation": 0.0,
                   "size": 5.0, "monitor": 1, "layer": "top"}


def test_dispatch_expression_matches_then_queues(win):
    reply = dispatch(win, {"id": 4, "cmd": "expression", "name": "爱心"})  # partial
    assert reply == {"id": 4, "ok": True, "toggled": "爱心眼"}
    bad = dispatch(win, {"id": 5, "cmd": "expression", "name": "nonsense"})
    assert bad["ok"] is False and "nonsense" in bad["error"]
    assert _queued(win) == [("expression", "爱心眼")]


def test_dispatch_move_layer_sleeping_prop_queue_for_the_gl_loop(win):
    assert dispatch(win, {"id": 6, "cmd": "move", "x": 0.3, "size": -20})["ok"] is True
    assert dispatch(win, {"id": 7, "cmd": "layer", "mode": "desktop"})["ok"] is True
    bad = dispatch(win, {"id": 8, "cmd": "layer", "mode": "sideways"})
    assert bad["ok"] is False and "top|desktop" in bad["error"]
    assert dispatch(win, {"id": 9, "cmd": "sleeping", "value": 1})["ok"] is True
    assert _queued(win) == [
        ("move", {"x": 0.3, "y": None, "rotation": None, "size": -20}),
        ("layer", "desktop"),
        ("sleeping", True),
    ]


def test_dispatch_activity_is_fire_and_forget_and_carries_only_given_keys(win):
    assert dispatch(win, {"cmd": "activity", "reading": {"title": "Alice"}, "working": None}) is None
    assert dispatch(win, {"cmd": "activity", "working": {"label": "x"}}) is None
    assert dispatch(win, {"cmd": "activity"}) is None              # nothing to say -> nothing queued
    assert dispatch(win, {"id": 3, "cmd": "activity"}) == {"id": 3, "ok": False}
    assert dispatch(win, {"id": 4, "cmd": "activity", "reading": None}) == {"id": 4, "ok": True}
    assert _queued(win) == [
        ("activity", {"reading": {"title": "Alice"}, "working": None}),
        ("activity", {"working": {"label": "x"}}),
        ("activity", {"reading": None}),
    ]


def test_dispatch_cue_accepts_only_known_beats(win):
    assert dispatch(win, {"cmd": "cue", "kind": "page"}) is None
    assert dispatch(win, {"id": 5, "cmd": "cue", "kind": "done"}) == {"id": 5, "ok": True}
    assert dispatch(win, {"id": 6, "cmd": "cue", "kind": "explode"}) == {"id": 6, "ok": False}
    assert _queued(win) == [("cue", "page"), ("cue", "done")]


def test_dispatch_legacy_prop_means_reading(win):
    """A protocol-v1 sister's `prop` still brings the book out (and puts it away)."""
    assert dispatch(win, {"cmd": "prop", "show": True}) is None
    assert dispatch(win, {"id": 10, "cmd": "prop", "show": False}) == {"id": 10, "ok": True, "prop": False}
    assert _queued(win) == [("activity", {"reading": {"title": ""}}), ("activity", {"reading": None})]


def test_hello_says_whether_the_body_does_activities(win):
    assert dispatch(win, {"id": 1, "cmd": "hello"})["activity"] is False
    win.activity = object()
    assert dispatch(win, {"id": 2, "cmd": "hello"})["activity"] is True


def test_expressions_drives_reports_a_toggled_param(win):
    assert win.expressions.drives("aixinyan") is False
    win.expressions.toggle("爱心眼")
    assert win.expressions.drives("aixinyan") is True
    assert win.expressions.drives("maozi") is False


def test_envelope_speaking_flag_follows_speech_start_and_stop():
    env = MouthEnvelope()
    assert env.speaking is False
    env.speech("start")
    assert env.speaking is True
    env.speech("interrupt")
    assert env.speaking is False


def test_dispatch_quit_and_unknown(win):
    assert dispatch(win, {"id": 11, "cmd": "quit"}) == {"id": 11, "ok": True}
    assert win.quit_requested is True
    reply = dispatch(win, {"id": 12, "cmd": "dance"})
    assert reply["ok"] is False and "dance" in reply["error"] and reply["id"] == 12


# --- make_retexture_model3 -----------------------------------------------------------

ARIU_MODEL3 = {
    "Version": 3,
    "FileReferences": {
        "Moc": "ariu.moc3",
        "Textures": [f"ariu.2048/texture_{i:02d}.png" for i in range(14)],
        "Physics": "ariu.physics3.json",
        "DisplayInfo": "ariu.cdi3.json",
    },
    "Groups": [
        {"Target": "Parameter", "Name": "EyeBlink", "Ids": ["ParamEyeLOpen", "ParamEyeROpen"]},
        {"Target": "Parameter", "Name": "LipSync", "Ids": []},
    ],
}


def test_retexture_rewrites_only_texture_directories():
    src = json.loads(json.dumps(ARIU_MODEL3))
    out = make_retexture_model3(src, "saga.2048")
    refs = out["FileReferences"]
    assert refs["Textures"] == [f"saga.2048/texture_{i:02d}.png" for i in range(14)]
    assert refs["Moc"] == "ariu.moc3"                 # same mesh
    assert refs["Physics"] == "ariu.physics3.json"
    assert refs["DisplayInfo"] == "ariu.cdi3.json"    # same part names for the fixups
    assert out["Groups"] == ARIU_MODEL3["Groups"]
    assert out["Version"] == 3
    assert src == ARIU_MODEL3                         # source untouched
    assert out["Groups"] is not src["Groups"]         # a real copy, not aliased


def test_retexture_handles_bare_names_backslashes_and_trailing_slash():
    src = {"FileReferences": {"Moc": "m.moc3",
                              "Textures": ["texture_00.png", "old\\sub\\texture_01.png"]}}
    out = make_retexture_model3(src, "new.2048/")
    assert out["FileReferences"]["Textures"] == ["new.2048/texture_00.png", "new.2048/texture_01.png"]
    assert make_retexture_model3({"FileReferences": {}}, "x")["FileReferences"]["Textures"] == []


# --- MODEL_FIXUPS -----------------------------------------------------------------------

def test_saga_fixups_share_the_mesh_fixups_but_have_their_own_skin():
    ariu, saga = MODEL_FIXUPS["ariu"], MODEL_FIXUPS["saga"]
    assert saga["camo_drawables"] == ariu["camo_drawables"]
    assert saga["mouth_part"] == ariu["mouth_part"]
    assert saga["mouth_part_floor"] == ariu["mouth_part_floor"]
    assert saga["camo_skin"] != ariu["camo_skin"]
    assert len(saga["camo_skin"]) == 3 and all(0.0 <= c <= 1.0 for c in saga["camo_skin"])
    # her sheets are pinker: red up, blue up relative to Aiva's skin
    assert saga["camo_skin"][0] >= ariu["camo_skin"][0]
    assert saga["camo_skin"][2] > ariu["camo_skin"][2]


def test_parse_rgb_accepts_floats_or_bytes_and_rejects_garbage():
    assert parse_rgb("1,0.929,0.929") == (1.0, 0.929, 0.929)
    assert parse_rgb("255,237,237") == (1.0, 0.929, 0.929)
    assert parse_rgb(" 0.5 ; 0.5 ; 0.5 ") == (0.5, 0.5, 0.5)
    assert parse_rgb(None) is None
    assert parse_rgb("") is None
    assert parse_rgb("1,2") is None
    assert parse_rgb("a,b,c") is None
    assert parse_rgb("-1,0,0") is None
    assert parse_rgb("300,0,0") is None


def test_cdi3_path_prefers_display_info_then_falls_back(tmp_path):
    model = tmp_path / "saga.model3.json"
    (tmp_path / "ariu.cdi3.json").write_text("{}", encoding="utf-8")
    model.write_text(json.dumps({"FileReferences": {"DisplayInfo": "ariu.cdi3.json"}}),
                     encoding="utf-8")
    assert cdi3_path_for(str(model)) == str(tmp_path / "ariu.cdi3.json")
    # DisplayInfo names a missing file -> the historical <stem>.cdi3.json rule
    model.write_text(json.dumps({"FileReferences": {"DisplayInfo": "gone.cdi3.json"}}),
                     encoding="utf-8")
    assert cdi3_path_for(str(model)) == str(tmp_path / "saga.cdi3.json")
    # unreadable model3 -> fallback too, never an exception
    model.write_text("{broken", encoding="utf-8")
    assert cdi3_path_for(str(model)) == str(tmp_path / "saga.cdi3.json")
    assert cdi3_path_for(str(tmp_path / "none.model3.json")) == str(tmp_path / "none.cdi3.json")


def test_sample_skin_rgb_means_only_opaque_pixels(tmp_path):
    QtGui = pytest.importorskip("PyQt5.QtGui")
    img = QtGui.QImage(8, 8, QtGui.QImage.Format_RGBA8888)
    img.fill(QtGui.QColor(255, 237, 237, 255))
    for y in range(8):  # a transparent column must not drag the mean toward black
        img.setPixelColor(0, y, QtGui.QColor(0, 0, 0, 0))
    path = str(tmp_path / "sheet.png")
    assert img.save(path)
    assert sample_skin_rgb(path, (0, 0, 8, 8)) == (1.0, 0.929, 0.929)
    with pytest.raises(ValueError):
        sample_skin_rgb(path, (0, 0, 1, 8))      # only transparent pixels
    with pytest.raises(ValueError):
        sample_skin_rgb(path, (4, 4, 8, 8))      # box outside the image
    with pytest.raises(ValueError):
        sample_skin_rgb(str(tmp_path / "missing.png"), (0, 0, 1, 1))


# --- the held book's art -------------------------------------------------------------------------

def test_hue_rotate_keeps_greys_and_turns_colours():
    import numpy as np
    px = np.array([[255.0, 255.0, 255.0], [128.0, 128.0, 128.0], [150.0, 180.0, 180.0]])
    out = hue_rotate_rgb(px, 170)
    assert np.allclose(out[:2], px[:2], atol=1.0)          # white pages and greys stay put
    assert out[2][0] > out[2][1] + 15                       # Aiva's teal-grey cover turns rosy
    assert np.allclose(hue_rotate_rgb(px, 0), px, atol=1e-6)


def test_prop_art_loads_tints_and_caches_scaled_copies(tmp_path):
    from PyQt5.QtGui import QColor, QImage
    img = QImage(40, 20, QImage.Format_ARGB32)
    img.fill(QColor(150, 180, 180, 255))
    img.setPixelColor(0, 0, QColor(255, 255, 255, 255))
    img.setPixelColor(1, 0, QColor(0, 0, 0, 0))
    path = tmp_path / "book.png"
    assert img.save(str(path))
    art = PropArt(str(path), hue=170)
    cover = art.image.pixelColor(10, 10)
    assert cover.red() > cover.green() and cover.alpha() == 255
    page = art.image.pixelColor(0, 0)
    assert (page.red(), page.green(), page.blue()) == pytest.approx((255, 255, 255), abs=2)
    assert art.image.pixelColor(1, 0).alpha() == 0          # transparency survives the tint
    small = art.at_width(10)
    assert small.width() == 10 and art.at_width(10.2) is small
    plain = PropArt(str(path))
    assert plain.image.pixelColor(10, 10).green() == 180
    assert PropArt(str(tmp_path / "missing.png")).image is None
    assert PropArt(None).at_width(10) is None
