"""Her drawing canvas (modules/drawing.py): opened and drawn on from her own process — no terminal, no
console, no focus taken — plus its tools, allow-lists and capability."""
import json
import os
import sys

import pytest

from modules import capabilities as cap
from modules import drawing
from modules import tasks
from modules import tools
from modules.drawing import DRAWING_APP_DIR, DrawingBoard, _drawapp, is_drawing_app_request
from modules.kin import KIN_LINES, TOOL_TO_CAPABILITY
from modules.tasks import RUNNER_TOOL_ALLOW
from modules.tools import TOOL_SCHEMAS, make_drawing_handlers, make_launch_app_handler

launcher, client = _drawapp()


@pytest.fixture(autouse=True)
def _journal_in_tmp(tmp_path, monkeypatch):
    """Every board journals by default; in tests that never reaches her real aiva/logs."""
    monkeypatch.setattr(drawing, "JOURNAL_DIR", str(tmp_path / "logs"))


class FakeClient:
    def __init__(self, app):
        self.app = app

    def stroke(self, points, author, color=None, width=None):
        self.app.sent.append(("stroke", list(points), author, color, width))
        if self.app.fail:
            raise self.app.fail
        return {"ok": True, "stroke_id": f"{author}-{len(self.app.sent)}", "revision": len(self.app.sent)}

    def clear(self, author):
        self.app.sent.append(("clear", author))
        return {"ok": True, "revision": len(self.app.sent)}

    def undo(self, author):
        self.app.sent.append(("undo", author))
        if self.app.fail:
            raise self.app.fail
        left = self.app.mine
        self.app.mine = max(0, left - 1)
        return {"ok": True, "stroke_id": f"{author}-{left}" if left else None,
                "revision": len(self.app.sent), "remaining": self.app.mine}

    def status(self):
        self.app.sent.append(("status",))
        if self.app.status_fail:
            raise self.app.status_fail
        return {"ok": True, "pid": self.app.pid}

    def save(self, author, name=None):
        self.app.sent.append(("save", author, name))
        if self.app.fail:
            raise self.app.fail
        stem = name or f"{author}-20261020-120000"
        return {"ok": True, "name": stem, "path": f"C:/d/{stem}.aivadraw", "png": f"C:/d/{stem}.png"}

    def open(self, name=None, author="user"):
        self.app.sent.append(("open", name, author))
        if self.app.fail:
            raise self.app.fail
        return {"ok": True, "name": name or "newest", "strokes": 3,
                "saved_previous": self.app.saved_previous}

    def close(self):
        pass


class FakeApp:
    def __init__(self, port):
        self.port, self.pid = port, 1000 + port
        self.alive, self.fail, self.sent = True, None, []
        self.saved_previous, self.restored = None, 0
        self.mine, self.status_fail = 0, None       # her strokes for undo; a pre-0.3 window's status

    def running(self):
        return self.alive

    def client(self):
        return FakeClient(self)

    def close(self):
        self.alive = False


class FakeLauncher:
    def __init__(self, error=None):
        self.apps, self.calls, self.error = [], [], error

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        self.apps.append(FakeApp(len(self.apps) + 1))
        return self.apps[-1]


class Params:
    def __init__(self, **arguments):
        self.arguments = arguments
        self.result = None

    async def result_callback(self, res):
        self.result = res


async def test_first_stroke_opens_the_canvas_once_and_reuses_it():
    launch = FakeLauncher()
    board = DrawingBoard(author="saga", launch=launch)
    first = await board.stroke([[10, 20], [30, 40]], width=6)
    second = await board.stroke([(1, 2)], color="#112233")
    assert first == {"success": True, "stroke_id": "saga-1", "revision": 1, "opened_canvas": True}
    assert second["success"] and second["opened_canvas"] is False
    assert len(launch.calls) == 1
    # the window is the sisters' shared canvas, his mouse draws as himself, her strokes are hers
    assert (launch.calls[0]["canvas"], launch.calls[0]["author"]) == ("shared", "user")
    assert launch.calls[0]["absorb"] == ("aiva", "saga")
    assert launch.apps[0].sent == [("stroke", [(10.0, 20.0), (30.0, 40.0)], "saga", None, 6),
                                   ("stroke", [(1.0, 2.0)], "saga", "#112233", None)]
    assert (await board.open())["opened"] is False


async def test_a_closed_window_is_reopened_on_the_next_call():
    launch = FakeLauncher()
    board = DrawingBoard(launch=launch)
    assert (await board.open())["opened"] is True
    launch.apps[0].alive = False                      # he closed the window
    out = await board.stroke([[5, 5]])
    assert out["success"] and out["opened_canvas"] and len(launch.apps) == 2
    assert launch.apps[1].sent and not launch.apps[0].sent


async def test_a_lost_reply_from_a_live_app_is_never_resent():
    launch = FakeLauncher()
    board = DrawingBoard(launch=launch)
    await board.open()
    launch.apps[0].fail = client.DrawConnectionError("timed out")
    out = await board.stroke([[5, 5], [9, 9]])
    assert out["success"] is False and "couldn't reach" in out["error"]
    assert len(launch.apps[0].sent) == 1 and len(launch.apps) == 1  # it may be drawn: not twice


async def test_an_app_that_died_mid_stroke_is_reopened_and_drawn_on_once():
    launch = FakeLauncher()
    board = DrawingBoard(launch=launch)
    await board.open()
    app = launch.apps[0]

    def die(*_a, **_k):
        app.alive = False
        raise client.DrawConnectionError("closed the connection")

    app.client = lambda: type("C", (), {"stroke": die, "close": lambda self: None})()
    board._client = app.client()
    out = await board.stroke([[5, 5]])
    assert out["success"] and out["opened_canvas"] and len(launch.apps) == 2
    assert len(launch.apps[1].sent) == 1


async def test_refusals_and_failed_launches_are_spoken_errors_not_exceptions():
    launch = FakeLauncher()
    board = DrawingBoard(launch=launch)
    await board.open()
    launch.apps[0].fail = client.CommandRejected(
        {"ok": False, "code": "invalid", "error": "point (2000, 5) is outside the 1024x768 canvas"})
    out = await board.stroke([[2000, 5]])
    assert out == {"success": False, "error": "the canvas refused it (invalid): point (2000, 5) is "
                                              "outside the 1024x768 canvas"}
    broken = DrawingBoard(launch=FakeLauncher(error=launcher.LaunchError("exited with code 1")))
    out = await broken.open()
    assert out["success"] is False and "didn't open" in out["error"]


async def test_malformed_points_never_open_a_window():
    launch = FakeLauncher()
    board = DrawingBoard(launch=launch)
    for bad in (None, [[1, 2, 3]], [["a", "b"]], 7):
        out = await board.stroke(bad)
        assert out["success"] is False and "[x, y]" in out["error"]
    assert not launch.calls


async def test_clear_without_a_canvas_opens_nothing():
    launch = FakeLauncher()
    board = DrawingBoard(launch=launch)
    assert (await board.clear())["success"] and not launch.calls
    await board.stroke([[1, 1]])
    out = await board.clear()
    assert out["success"] and launch.apps[0].sent[-1] == ("clear", "aiva")
    await board.close()
    assert not launch.apps[0].alive and not board.is_open()


async def test_tool_handlers_pass_through_to_the_board():
    launch = FakeLauncher()
    h = make_drawing_handlers(DrawingBoard(launch=launch))
    assert set(h) == {"open_drawing", "draw_stroke", "clear_drawing", "undo_stroke", "save_drawing",
                      "open_saved_drawing"}
    p = Params(points=[[100, 100], [300, 180]], color="", width=5)
    await h["draw_stroke"](p)
    assert p.result["success"] and launch.apps[0].sent == [
        ("stroke", [(100.0, 100.0), (300.0, 180.0)], "aiva", None, 5)]
    p = Params()
    await h["open_drawing"](p)
    assert p.result["success"] and p.result["opened"] is False


def test_schemas_allow_lists_and_capability():
    schemas = {fs.name: fs for fs in TOOL_SCHEMAS.standard_tools}
    assert {"open_drawing", "draw_stroke", "clear_drawing"} <= set(schemas)
    assert schemas["draw_stroke"].required == ["points"]
    assert "terminal" in schemas["open_drawing"].description     # told never to reach for one
    # the task runner draws/verifies through these, not through open_terminal + python -c
    assert {"open_drawing", "draw_stroke", "clear_drawing"} <= RUNNER_TOOL_ALLOW
    # drawing is hers to do; wiping the canvas (no undo) is not in the safe envelope
    assert {"open_drawing", "draw_stroke"} <= cap.SAFE_TOOLS and "clear_drawing" not in cap.SAFE_TOOLS
    m = cap.match("I'd love to sketch something on my canvas")
    assert m is not None and m.name == "draw" and m.tier == "auto"
    assert TOOL_TO_CAPABILITY["draw_stroke"] == "draw" and "draw" in KIN_LINES


# --- launch or reuse, from code; her drawing app is never the generic launcher's to guess -------
# 2026-10-20 a task asked launch_app for "drawing-app" and Windows put Aseprite up instead.

def test_launch_or_reuse_opens_once_then_reuses_and_speaks_failures():
    launch = FakeLauncher()
    board = DrawingBoard(author="saga", launch=launch)
    first, second = board.launch_or_reuse(), board.launch_or_reuse()
    assert first["success"] and first["opened"] and first["port"] == 1
    assert second["success"] and second["opened"] is False and len(launch.calls) == 1
    launch.apps[0].alive = False                      # he closed the window
    assert board.launch_or_reuse()["opened"] and len(launch.apps) == 2
    broken = DrawingBoard(launch=FakeLauncher(error=launcher.LaunchError("exited with code 1")))
    out = broken.launch_or_reuse()
    assert out["success"] is False and "didn't open" in out["error"]


async def test_a_missing_drawing_app_is_a_spoken_failure_not_an_exception(monkeypatch):
    def gone():
        raise ModuleNotFoundError("No module named 'drawapp'")

    monkeypatch.setattr(drawing, "_drawapp", gone)
    launch = FakeLauncher()
    board = DrawingBoard(launch=launch)
    for out in (board.launch_or_reuse(), await board.open(), await board.stroke([[1, 1]]),
                await board.clear()):
        assert out["success"] is False and "couldn't load the drawing app" in out["error"]
    assert not launch.calls


def test_board_for_is_one_board_per_sister_in_the_process(monkeypatch):
    monkeypatch.setattr(drawing, "_BOARDS", {})
    launch = FakeLauncher()
    aiva = drawing.board_for("aiva", launch=launch)
    assert drawing.board_for("aiva") is aiva and drawing.board_for("saga") is not aiva
    assert aiva.canvas == drawing.board_for("saga").canvas == drawing.SHARED_CANVAS  # one window
    assert drawing.launch_or_reuse("aiva")["opened"] is True
    assert drawing.launch_or_reuse("aiva")["opened"] is False and len(launch.calls) == 1


@pytest.mark.parametrize("name", ["drawing-app", "drawapp", "DrawApp.exe", "drawing app",
                                  "your drawing canvas", "Aiva's canvas", "my drawing",
                                  "Saga drawing app", "the drawing application"])
def test_her_own_drawing_app_names_are_recognised(name):
    assert is_drawing_app_request(name)


@pytest.mark.parametrize("name", ["aseprite", "Aseprite.exe", "paint", "mspaint", "krita",
                                  "chrome", "canvas", "drawing", "", None])
def test_other_programs_are_not_her_drawing_app(name):
    assert not is_drawing_app_request(name)


def _no_generic_launch(monkeypatch):
    launched = []
    monkeypatch.setattr(tools, "_launch_app_impl", lambda name: launched.append(name) or "")
    return launched


async def test_launch_app_routes_her_drawing_app_to_the_board_not_the_launcher(monkeypatch):
    launched = _no_generic_launch(monkeypatch)
    launch = FakeLauncher()
    handler = make_launch_app_handler(DrawingBoard(launch=launch))
    p = Params(app_name="drawing-app")
    await handler(p)
    assert p.result["success"] and p.result["opened"] and p.result["via"] == "open_drawing"
    p = Params(app_name="drawapp")
    await handler(p)
    assert p.result["success"] and p.result["opened"] is False      # reused, not a second window
    assert len(launch.calls) == 1 and launched == []                 # no Aseprite, no guessing


async def test_launch_app_without_a_board_refuses_instead_of_guessing(monkeypatch):
    launched = _no_generic_launch(monkeypatch)
    p = Params(app_name="drawing app")
    await tools.launch_app(p)                       # the module-level handler, kept for callers
    assert p.result["success"] is False and "open_drawing" in p.result["detail"]
    assert launched == []


async def test_launch_app_still_opens_other_programs_as_asked(monkeypatch):
    launched = _no_generic_launch(monkeypatch)
    launch = FakeLauncher()
    handler = make_launch_app_handler(DrawingBoard(launch=launch))
    p = Params(app_name="aseprite")                 # he named it: his to open, never substituted
    await handler(p)
    assert p.result == {"success": True, "launched": "aseprite"}
    assert launched == ["aseprite"] and not launch.calls


async def test_built_handlers_route_launch_app_through_her_board(monkeypatch):
    launched = _no_generic_launch(monkeypatch)
    launch = FakeLauncher()
    h = tools.build_handlers(None, drawing=DrawingBoard(launch=launch))
    p = Params(app_name="your drawing app")
    await h["launch_app"](p)
    assert p.result["success"] and p.result["via"] == "open_drawing"
    assert len(launch.calls) == 1 and launched == []


def test_planner_runner_schemas_and_wishes_point_at_open_drawing():
    schemas = {fs.name: fs for fs in TOOL_SCHEMAS.standard_tools}
    assert "open_drawing" in schemas["launch_app"].description
    assert "Aseprite" in schemas["open_drawing"].description
    for prompt in (tasks.PLANNER_PROMPT, tasks.RUNNER_SYSTEM_PROMPT):
        assert "open_drawing" in prompt and "never launch_app" in prompt
    for wish in ("open your drawing app", "open the drawing-app and draw a cat"):
        assert cap.match(wish).name == "draw"


def test_the_child_is_a_windowless_interpreter_that_does_not_take_focus():
    argv = launcher.launch_argv(launcher.gui_python(sys.executable), "ready.json", author="aiva")
    if sys.platform == "win32":
        assert os.path.basename(argv[0]).lower() == "pythonw.exe"  # GUI subsystem: never a console
    assert argv[1:3] == ["-m", "drawapp"] and "--no-activate" in argv
    assert os.path.isdir(os.path.join(DRAWING_APP_DIR, "drawapp"))


# --- the real thing, offscreen: her process launches the app and draws, no terminal anywhere ------

async def test_a_real_canvas_opens_takes_strokes_and_closes(tmp_path):
    board = DrawingBoard(author="aiva", extra_env={"QT_QPA_PLATFORM": "offscreen"},
                         save_dir=str(tmp_path))       # never his real drawings folder
    try:
        opened = await board.open()
        assert opened["success"] and opened["opened"], opened
        app = board._app
        if sys.platform == "win32":
            assert os.path.basename(app.process.args[0]).lower() == "pythonw.exe"
            assert "--no-activate" in app.process.args
        out = await board.stroke([[100, 100], [300, 180], [500, 120]], width=6)
        assert out["success"] and out["stroke_id"].startswith("aiva-"), out
        refused = await board.stroke([[-5, 10], [20, 20]])
        assert refused["success"] is False and "invalid" in refused["error"]
        wrong = await board.stroke([[600, 600], [700, 650]])
        undone = await board.undo()
        assert undone["success"] and undone["undone"] == wrong["stroke_id"], undone
        assert undone["remaining"] == 1 and undone["revision"] == wrong["revision"] + 1
        cleared = await board.clear()
        assert cleared["success"] and cleared["revision"] == undone["revision"] + 1
        nothing = await board.undo()
        assert nothing["success"] and nothing["undone"] is None
    finally:
        await board.close()
    assert not board.is_open()
    with pytest.raises(OSError):
        import socket
        socket.create_connection(("127.0.0.1", app.port), timeout=1).close()


# --- saving, getting a drawing back, and never losing the canvas she already has ----------------

async def test_save_never_opens_a_window_and_names_what_it_saved():
    launch = FakeLauncher()
    board = DrawingBoard(launch=launch)
    out = await board.save("cat")
    assert out["success"] is False and "comes back when you open it" in out["error"]
    assert not launch.calls                                        # no window just to save it
    await board.stroke([[1, 1], [5, 5]])
    out = await board.save("sleepy cat")
    assert out == {"success": True, "saved_as": "sleepy cat", "path": "C:/d/sleepy cat.aivadraw",
                   "picture": "C:/d/sleepy cat.png",
                   "status": "saved it as sleepy cat in your drawings folder"}
    assert (await board.save(""))["saved_as"] == "aiva-20261020-120000"   # blank name: no name
    assert launch.apps[0].sent[-1] == ("save", "aiva", None)


async def test_open_saved_opens_the_canvas_and_says_what_was_kept():
    launch = FakeLauncher()
    board = DrawingBoard(author="saga", launch=launch)
    out = await board.open_saved()
    assert out["success"] and out["opened_canvas"] and out["opened_drawing"] == "newest"
    assert launch.apps[0].sent == [("open", None, "saga")]
    launch.apps[0].saved_previous = "C:/d/saga-20261020-130000.aivadraw"
    out = await board.open_saved("tree")
    assert out["opened_canvas"] is False and out["saved_previous"] == "saga-20261020-130000"
    assert out["status"] == "tree is on your canvas now; what was on it is saved as saga-20261020-130000"


async def test_storage_refusals_are_spoken_in_their_own_words():
    launch = FakeLauncher()
    board = DrawingBoard(launch=launch)
    await board.open()
    launch.apps[0].fail = client.CommandRejected(
        {"ok": False, "code": "storage", "error": "there is no saved drawing called 'x' (saved: cat)"})
    out = await board.open_saved("x")
    assert out == {"success": False, "error": "there is no saved drawing called 'x' (saved: cat)"}
    launch.apps[0].fail = client.CommandRejected(
        {"ok": False, "code": "unsupported", "error": "unsupported cmd 'save'"})
    out = await board.save()
    assert out["success"] is False and "from before saving" in out["error"]
    assert "leave it open" in out["error"] and "close it and" not in out["error"]  # never lose it
    assert launch.apps[0].alive and len(launch.calls) == 1        # nothing closed or reopened


async def test_a_window_still_up_from_before_her_restart_is_adopted_not_doubled():
    launch = FakeLauncher()
    leftover = FakeApp(41)
    looked = []

    def find(author, save_dir):
        looked.append((author, save_dir))
        return leftover if leftover.alive else None

    board = DrawingBoard(author="saga", launch=launch, find=find, save_dir="D:/drawings")
    opened = await board.open()
    assert opened["success"] and opened["opened"] is False and opened["port"] == 41
    assert looked == [("shared", "D:/drawings")] and not launch.calls   # the canvas, not her name
    await board.stroke([[2, 2]])
    assert leftover.sent == [("stroke", [(2.0, 2.0)], "saga", None, None)]
    assert (await board.save())["success"] and (await board.clear())["success"]
    leftover.alive = False                                         # he closed that one
    out = await board.stroke([[3, 3]])
    assert out["opened_canvas"] and launch.calls[0]["save_dir"] == "D:/drawings"


async def test_a_fresh_canvas_that_restored_her_drawing_says_so():
    launch = FakeLauncher()

    def restored(**kwargs):
        app = launch(**kwargs)
        app.restored = 4
        return app

    out = await DrawingBoard(launch=restored).open()
    assert out["status"] == "opened your drawing canvas, with your last drawing back on it"


def test_a_board_with_a_fake_launcher_never_looks_for_his_real_window(monkeypatch):
    def never(*_a, **_k):
        raise AssertionError("looked for a real window")

    monkeypatch.setattr(launcher, "find_running", never)
    board = DrawingBoard(launch=FakeLauncher())
    assert board.launch_or_reuse()["opened"] is True


def test_save_and_open_saved_are_safe_tools_in_the_draw_capability():
    schemas = {fs.name: fs for fs in TOOL_SCHEMAS.standard_tools}
    assert {"save_drawing", "open_saved_drawing"} <= set(schemas)
    assert schemas["save_drawing"].required == [] and schemas["open_saved_drawing"].required == []
    assert "Aiva and Saga Drawings" in schemas["save_drawing"].description
    assert "saved first" in schemas["open_saved_drawing"].description
    assert {"save_drawing", "open_saved_drawing"} <= RUNNER_TOOL_ALLOW
    assert {"save_drawing", "open_saved_drawing"} <= cap.SAFE_TOOLS
    assert {"save_drawing", "open_saved_drawing"} <= set(cap._BY_NAME["draw"].tools)
    assert TOOL_TO_CAPABILITY["save_drawing"] == "draw"


async def test_tool_handlers_save_and_open_pass_through(monkeypatch):
    launch = FakeLauncher()
    h = make_drawing_handlers(DrawingBoard(launch=launch))
    await h["draw_stroke"](Params(points=[[1, 1]]))
    p = Params(name="  ")
    await h["save_drawing"](p)
    assert p.result["success"] and launch.apps[0].sent[-1] == ("save", "aiva", "  ")
    p = Params(name="tree")
    await h["open_saved_drawing"](p)
    assert p.result["opened_drawing"] == "tree"


def test_save_now_and_save_drawing_save_directly_without_opening_or_drawing(monkeypatch):
    monkeypatch.setattr(drawing, "_BOARDS", {})
    launch = FakeLauncher()
    board = drawing.board_for("saga", launch=launch)
    assert drawing.save_drawing("saga", "cat")["success"] is False and not launch.calls
    board.launch_or_reuse()
    sent = list(launch.apps[0].sent)
    out = drawing.save_drawing("saga", "purple cat")
    assert out["success"] and out["saved_as"] == "purple cat" and out["picture"].endswith(".png")
    assert launch.apps[0].sent == sent + [("save", "saga", "purple cat")]  # one save, nothing else
    assert board.save_now()["saved_as"] == "saga-20261020-120000"
    assert len(launch.calls) == 1 and launch.apps[0].alive


@pytest.mark.parametrize("slug", ["aiva", "saga"])
async def test_a_task_saves_through_her_toolbox_in_her_own_name(monkeypatch, slug):
    # 2026-10-20 a task's agent said "no direct tool" for saving: her process was older than the tool.
    # This is the path a task takes once she has it: main.py's handlers, the ToolBox, one save call
    monkeypatch.setattr(drawing, "_BOARDS", {})
    launch = FakeLauncher()
    board = drawing.board_for(slug, launch=launch)
    box = tasks.ToolBox(tools.build_handlers(None, drawing=board), TOOL_SCHEMAS, name=slug.title())
    assert "save_drawing" in box.names
    assert "save_drawing" in [t["function"]["name"] for t in box.openai_tools()]
    board.launch_or_reuse()
    out = await box.call("save_drawing", {"name": "moonlit rooftop"})
    assert out["success"] and out["path"].endswith("moonlit rooftop.aivadraw")
    assert out["picture"].endswith("moonlit rooftop.png")
    assert launch.apps[0].sent[-1] == ("save", slug, "moonlit rooftop")      # her ink, her file
    assert len(launch.calls) == 1                                            # nothing reopened


def test_saving_is_its_own_tool_never_a_keypress():
    schemas = {fs.name: fs for fs in TOOL_SCHEMAS.standard_tools}
    assert "Ctrl+S" in schemas["save_drawing"].description
    for prompt in (tasks.PLANNER_PROMPT, tasks.RUNNER_SYSTEM_PROMPT):
        assert "save_drawing" in prompt and "Ctrl+S" in prompt
    assert cap.match("save my drawing").name == "draw"


# --- the real thing again: her drawing survives the window, and a restart adopts the window ------

async def test_a_real_canvas_saves_comes_back_after_closing_and_is_adopted(tmp_path):
    env = {"QT_QPA_PLATFORM": "offscreen"}
    first = DrawingBoard(author="saga", extra_env=env, save_dir=str(tmp_path))
    second = DrawingBoard(author="saga", extra_env=env, save_dir=str(tmp_path))  # after a restart
    try:
        await first.stroke([[100, 100], [300, 300]], width=8)
        saved = await first.save("ring")
        assert saved["success"] and os.path.isfile(saved["path"]) and os.path.isfile(saved["picture"])
        adopted = await second.open()
        assert adopted["success"] and adopted["opened"] is False      # the same window, not a 2nd
        assert second._app.adopted and second._app.pid == first._app.pid
        await second.clear()                                          # blank, but kept in recovery
        back = await second.open_saved("ring")
        assert back["success"] and back["strokes"] == 1
        await second.stroke([[10, 10], [20, 20]])
    finally:
        await first.close()
    third = DrawingBoard(author="saga", extra_env=env, save_dir=str(tmp_path))
    try:
        reopened = await third.open()
        assert reopened["opened"] and "last drawing back" in reopened["status"], reopened
        assert third._app.restored == 2                               # ring + the stroke after it
    finally:
        await third.close()


# --- one canvas for both sisters: tag-teaming a drawing never opens a second window ---------------
# 2026-10-21 he asked them to tag-team a drawing and saw two windows: each sister launched, and
# looked for, a window under her own name, so Saga never found Aiva's.

class SharedDesk:
    """Both sisters' processes against one screen: whatever window one launches, the other finds."""

    def __init__(self):
        self.launch, self.looked = FakeLauncher(), []

    def find(self, canvas, save_dir):
        self.looked.append(canvas)
        return next((a for a in self.launch.apps if a.alive), None)


async def test_both_sisters_draw_on_one_window_each_in_her_own_name():
    desk = SharedDesk()
    aiva = DrawingBoard(author="aiva", launch=desk.launch, find=desk.find)
    saga = DrawingBoard(author="saga", launch=desk.launch, find=desk.find)
    first = await aiva.stroke([[10, 10], [50, 50]])
    second = await saga.stroke([[60, 60], [90, 90]])
    assert first["opened_canvas"] and second["opened_canvas"] is False     # Saga joined Aiva's
    assert len(desk.launch.calls) == 1 and set(desk.looked) == {"shared"}
    [app] = desk.launch.apps
    assert app.sent == [("stroke", [(10.0, 10.0), (50.0, 50.0)], "aiva", None, None),
                        ("stroke", [(60.0, 60.0), (90.0, 90.0)], "saga", None, None)]
    assert (await saga.open())["status"] == "your canvas is already open"
    await saga.undo()
    await aiva.save("tag team")
    assert app.sent[-2:] == [("undo", "saga"), ("save", "aiva", "tag team")]   # each her own
    app.alive = False                                     # he closed it: whoever draws next opens it
    assert (await saga.stroke([[5, 5]]))["opened_canvas"] and len(desk.launch.apps) == 2
    assert (await aiva.stroke([[6, 6]]))["opened_canvas"] is False
    assert desk.launch.apps[1].sent == [("stroke", [(5.0, 5.0)], "saga", None, None),
                                        ("stroke", [(6.0, 6.0)], "aiva", None, None)]


async def test_a_real_shared_canvas_takes_both_sisters_and_their_old_canvases(tmp_path):
    from drawapp import storage

    recovery = tmp_path / "recovery"                       # what each sister's own window left
    recovery.mkdir()
    for name, sid in (("aiva", "aiva-old1"), ("saga", "saga-old1")):
        (recovery / f"{name}-autosave.aivadraw").write_text(json.dumps({
            "format": "aivadraw", "version": 1, "width": 1024, "height": 768,
            "background": "#ffffff", "strokes": [{"id": sid, "author": name, "color": "#7b4fd6",
                                                  "width": 4, "points": [[1, 1], [5, 5]]}]}))
    env = {"QT_QPA_PLATFORM": "offscreen"}
    aiva = DrawingBoard(author="aiva", extra_env=env, save_dir=str(tmp_path))
    saga = DrawingBoard(author="saga", extra_env=env, save_dir=str(tmp_path))
    try:
        opened = await aiva.open()
        assert opened["opened"] and "last drawing back" in opened["status"], opened
        assert aiva._app.restored == 2                     # both old canvases, on the one
        joined = await saga.open()
        assert joined["opened"] is False and joined["port"] == opened["port"], joined
        assert saga._app.pid == aiva._app.pid             # her sister's window, not a second one
        mine = await aiva.stroke([[100, 100], [300, 180]])
        hers = await saga.stroke([[400, 400], [600, 500]])
        assert mine["stroke_id"].startswith("aiva-") and hers["stroke_id"].startswith("saga-")
        undone = await saga.undo()                         # hers only: Aiva's stroke stays
        assert undone["undone"] == hers["stroke_id"] and undone["remaining"] == 1, undone
        with saga._app.client() as draw:
            status = draw.status()
        assert status["author"] == "shared" and status["strokes"] == 3
    finally:
        await aiva.close()
    kept = sorted(os.listdir(recovery))
    assert not (recovery / "aiva-autosave.aivadraw").exists()      # taken in once, never deleted:
    assert sum(".merged-" in f for f in kept) == 2, kept            # the old copies are kept aside
    snap = storage.read_drawing(str(recovery / "shared-autosave.aivadraw"))
    assert [(s.id, s.author) for s in snap.strokes][:2] == [("aiva-old1", "aiva"), ("saga-old1", "saga")]
    assert [s.author for s in snap.strokes][2:] == ["aiva"]


async def test_two_sisters_opening_the_canvas_at_the_same_moment_get_one_window(tmp_path):
    import asyncio

    env = {"QT_QPA_PLATFORM": "offscreen"}
    aiva = DrawingBoard(author="aiva", extra_env=env, save_dir=str(tmp_path))
    saga = DrawingBoard(author="saga", extra_env=env, save_dir=str(tmp_path))
    try:
        both = await asyncio.gather(aiva.open(), saga.open())      # neither finds one yet
        assert all(o["success"] for o in both) and sorted(o["opened"] for o in both) == [False, True]
        assert aiva._app.pid == saga._app.pid and both[0]["port"] == both[1]["port"], both
    finally:
        await (aiva if not aiva._app.adopted else saga).close()


# --- undo: her own last stroke, nobody else's -------------------------------------------------

async def test_undo_takes_back_her_strokes_one_at_a_time_and_never_opens_a_window():
    launch = FakeLauncher()
    board = DrawingBoard(author="saga", launch=launch)
    out = await board.undo()
    assert out["success"] and out["undone"] is None and "isn't open" in out["status"]
    assert not launch.calls                                        # no window just to undo
    await board.stroke([[1, 1], [5, 5]])
    launch.apps[0].mine = 2
    out = await board.undo()
    assert out["success"] and out["undone"] == "saga-2" and out["remaining"] == 1
    assert out["status"] == "took back your last stroke; 1 of yours are still on it"
    assert (await board.undo())["status"] == "took back your last stroke; none of yours are left on it"
    out = await board.undo()
    assert out["success"] and out["undone"] is None and "nothing to undo" in out["status"]
    assert [c for c in launch.apps[0].sent if c[0] == "undo"] == [("undo", "saga")] * 3


async def test_a_window_older_than_undo_says_how_to_get_it_without_losing_the_drawing():
    launch = FakeLauncher()
    board = DrawingBoard(launch=launch)
    await board.open()
    app = launch.apps[0]
    app.fail = client.CommandRejected({"ok": False, "code": "unsupported", "error": "unsupported cmd"})
    out = await board.undo()                                       # 0.3: autosaves, so reopen
    assert out["success"] is False and "before undo existed" in out["error"]
    assert "autosaved" in out["error"] and app.sent[-1] == ("status",)
    app.status_fail = app.fail                                     # 0.2: no autosave, leave it open
    out = await board.undo()
    assert out["success"] is False and "leave it open" in out["error"]
    assert "before undo" not in out["error"]


def test_undo_stroke_is_a_runner_tool_outside_the_safe_envelope():
    schemas = {fs.name: fs for fs in TOOL_SCHEMAS.standard_tools}
    assert schemas["undo_stroke"].required == []
    assert "sister" in schemas["undo_stroke"].description          # only her own strokes
    assert "undo_stroke" in schemas["clear_drawing"].description   # one stroke isn't a clear
    assert "undo_stroke" in RUNNER_TOOL_ALLOW
    # ask-first as before: on the shared canvas his strokes are his own ("user"), but undo still
    # takes something off a drawing he and her sister are looking at
    assert "undo_stroke" not in cap.SAFE_TOOLS
    assert "undo_stroke" not in cap._BY_NAME["draw"].tools


async def test_the_undo_handler_passes_through_to_the_board():
    launch = FakeLauncher()
    h = make_drawing_handlers(DrawingBoard(launch=launch))
    await h["draw_stroke"](Params(points=[[10, 10]]))
    launch.apps[0].mine = 1
    p = Params()
    await h["undo_stroke"](p)
    assert p.result["success"] and p.result["undone"] == "aiva-1"
    assert launch.apps[0].sent[-1] == ("undo", "aiva")


# --- the journal: every canvas action written down behind the scenes ----------------------------
# 2026-10-21 their heart halves didn't meet, and a screenshot was all anyone had to go on.

async def test_every_canvas_action_is_journaled_with_what_it_takes_to_redraw_it():
    launch, seen = FakeLauncher(), []
    board = DrawingBoard(author="saga", launch=launch, journal=seen.append)
    out = await board.stroke([[512, 390], [600, 340], [512, 530]])
    assert out == {"success": True, "stroke_id": "saga-1", "revision": 1, "opened_canvas": True}
    first = seen[-1]                                    # what she said is unchanged; this is extra
    assert {k: first[k] for k in ("author", "canvas", "action", "outcome", "stroke_id", "revision",
                                  "opened_canvas")} == {
        "author": "saga", "canvas": "shared", "action": "stroke", "outcome": "ok",
        "stroke_id": "saga-1", "revision": 1, "opened_canvas": True}
    assert first["points"] == [[512.0, 390.0], [600.0, 340.0], [512.0, 530.0]]
    assert first["n_points"] == 3 and first["start"] == [512.0, 390.0] and first["end"] == [512.0, 530.0]
    assert first["bbox"] == [512.0, 340.0, 600.0, 530.0]
    # no color asked for: the canvas used her ink, and the record says both
    assert (first["color"], first["color_requested"], first["width"], first["width_requested"]) == \
        ("#d64f8a", None, 4.0, None)
    assert first["window"] == {"pid": 1001, "port": 1, "adopted": None}
    assert "error" not in first and first["seq"] == 1 and first["process"] == os.getpid()

    await board.stroke([[10, 10], [20, 20]], color="#9b6cff", width=8)
    assert (seen[-1]["color"], seen[-1]["color_requested"], seen[-1]["width"]) == ("#9b6cff", "#9b6cff", 8)

    app = launch.apps[0]
    app.fail = client.CommandRejected(
        {"ok": False, "code": "invalid", "error": "point (2000, 5) is outside the 1024x768 canvas"})
    await board.stroke([[2000, 5]])
    refused = seen[-1]
    assert (refused["outcome"], refused["code"], refused["error_type"]) == ("refused", "invalid",
                                                                           "CommandRejected")
    assert refused["points"] == [[2000.0, 5.0]] and refused["stroke_id"] is None
    assert "outside" in refused["error"]
    app.fail = client.DrawConnectionError("timed out")
    await board.stroke([[5, 5]])
    assert seen[-1]["outcome"] == "unknown"             # a live window went silent: maybe drawn
    app.fail = None

    await board.stroke([[1, 2, 3]])
    assert seen[-1]["outcome"] == "refused" and seen[-1]["points"] is None
    assert "[1, 2, 3]" in seen[-1]["points_raw"]

    app.mine = 1
    await board.undo()
    assert (seen[-1]["action"], seen[-1]["stroke_id"], seen[-1]["remaining"]) == ("undo", "saga-1", 0)
    await board.clear()
    assert seen[-1]["action"] == "clear" and seen[-1]["revision"] == len(app.sent)
    await board.save("half heart")
    assert (seen[-1]["action"], seen[-1]["name"], seen[-1]["saved_as"]) == ("save", "half heart",
                                                                            "half heart")
    await board.open_saved("tree")
    assert (seen[-1]["action"], seen[-1]["opened_drawing"]) == ("open_saved", "tree")
    await board.open()
    assert seen[-1]["action"] == "open" and seen[-1]["opened_canvas"] is False
    await board.close()
    assert seen[-1]["action"] == "close" and seen[-1]["window"]["pid"] == 1001
    assert [r["seq"] for r in seen] == list(range(1, len(seen) + 1))


async def test_an_action_with_no_window_is_journaled_as_such():
    seen = []
    board = DrawingBoard(launch=FakeLauncher(error=launcher.LaunchError("exited with code 1")),
                         journal=seen.append)
    await board.clear()                                 # nothing open: nothing cleared, no window
    assert seen[-1]["outcome"] == "ok" and seen[-1]["window"] is None
    await board.stroke([[1, 1]])
    failed = seen[-1]
    assert (failed["outcome"], failed["error_type"], failed["window"]) == ("failed", "LaunchError", None)
    assert failed["points"] == [[1.0, 1.0]] and "didn't open" in failed["error"]


async def test_the_journal_never_costs_her_the_stroke(tmp_path):
    def broken(_record):
        raise RuntimeError("disk on fire")

    launch = FakeLauncher()
    out = await DrawingBoard(launch=launch, journal=broken).stroke([[1, 1]])
    assert out["success"] and len(launch.apps[0].sent) == 1
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("")
    drawing.DrawingJournal("aiva", str(blocked))({"action": "stroke"})   # unwritable: quiet


async def test_both_sisters_journals_read_back_as_one_timeline(tmp_path):
    logs = str(tmp_path / "logs")
    desk = SharedDesk()
    aiva = DrawingBoard(author="aiva", launch=desk.launch, find=desk.find,
                        journal=drawing.DrawingJournal("aiva", logs))
    saga = DrawingBoard(author="saga", launch=desk.launch, find=desk.find,
                        journal=drawing.DrawingJournal("saga", logs))
    await aiva.stroke([[512, 290], [400, 250], [512, 490]])
    await saga.stroke([[512, 390], [620, 350], [512, 530]])
    await aiva.undo()
    with open(drawing.DrawingJournal("saga", logs).path(), "a", encoding="utf-8") as f:
        f.write('{"cut short' + chr(10))                # a half-written line is skipped
    timeline = drawing.read_journal(log_dir=logs)
    assert [(r["author"], r["action"]) for r in timeline] == [
        ("aiva", "stroke"), ("saga", "stroke"), ("aiva", "undo")]
    assert {r["window"]["pid"] for r in timeline} == {1001}     # both on the one window
    assert os.path.basename(aiva._journal.path()).startswith("drawing-aiva-")
    assert drawing.read_journal(log_dir=str(tmp_path / "nowhere")) == []


async def test_a_board_journals_into_her_logs_by_default(tmp_path):
    board = DrawingBoard(author="saga", launch=FakeLauncher())
    await board.stroke([[3, 4]])
    [rec] = drawing.read_journal(authors=("saga",))           # JOURNAL_DIR: tmp in tests
    assert rec["author"] == "saga" and rec["points"] == [[3.0, 4.0]]
    assert board._journal.path().startswith(str(tmp_path / "logs"))


async def test_a_real_shared_canvas_can_be_redrawn_from_both_sisters_journals(tmp_path):
    from drawapp import storage

    env = {"QT_QPA_PLATFORM": "offscreen"}
    seen = []
    aiva = DrawingBoard(author="aiva", extra_env=env, save_dir=str(tmp_path), journal=seen.append)
    saga = DrawingBoard(author="saga", extra_env=env, save_dir=str(tmp_path), journal=seen.append)
    try:
        await aiva.stroke([[512, 300], [430, 230], [370, 300], [512, 500]], width=8)
        await saga.stroke([[512, 300], [594, 230], [654, 300], [512, 500]], color="#2a7de1")
        await saga.stroke([[-1, 0], [5, 5]])                       # refused: nothing on the canvas
        await aiva.stroke([[100, 100], [200, 200]])
        await aiva.undo()                                          # takes the last one back
    finally:
        await aiva.close()
    pids = {r["window"]["pid"] for r in seen if r["window"]}
    assert len(pids) == 1, seen                                    # one window for both
    revisions = [r["revision"] for r in seen if r["outcome"] == "ok" and r["revision"] is not None]
    assert revisions == sorted(revisions)
    undone = {r["stroke_id"] for r in seen if r["action"] == "undo"}
    redrawn = [(r["stroke_id"], r["author"], r["color"], float(r["width"]), r["points"])
               for r in seen if r["action"] == "stroke" and r["outcome"] == "ok"
               and r["stroke_id"] not in undone]
    snap = storage.read_drawing(str(tmp_path / "recovery" / "shared-autosave.aivadraw"))
    assert redrawn == [(s.id, s.author, s.color, s.width, [list(p) for p in s.points])
                       for s in snap.strokes]
