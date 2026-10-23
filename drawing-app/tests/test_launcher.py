"""Launching the window from code: the command line, the ready-file handshake, and a real child process."""
import json
import os
import socket

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from drawapp import app as drawapp_main
from drawapp import launcher
from drawapp.app import MainWindow, parse_args
from drawapp.client import CommandRejected
from drawapp.launcher import LaunchError, gui_python, launch, launch_argv, read_ready, write_ready

OFFSCREEN = {"QT_QPA_PLATFORM": "offscreen"}  # the child draws in memory; nothing reaches the screen


def test_the_child_command_line():
    assert launch_argv("pyw", "r.json") == [
        "pyw", "-m", "drawapp", "--listen", "0", "--ready-file", "r.json", "--author", "user",
        "--no-activate"]
    assert launch_argv("pyw", "r.json", 8770, "saga", 300, 200)[3:] == [
        "--listen", "8770", "--ready-file", "r.json", "--author", "saga", "--no-activate",
        "--width", "300", "--height", "200"]
    assert "--no-activate" not in launch_argv("pyw", "r.json", activate=True)
    parsed = parse_args(launch_argv("pyw", "r.json", author="saga")[3:])
    assert (parsed.listen, parsed.ready_file, parsed.author, parsed.no_activate) == (
        0, "r.json", "saga", True)
    assert parse_args([]).no_activate is False  # by hand, the window takes the focus as usual


def test_a_console_interpreter_is_swapped_for_its_windowed_sibling(tmp_path):
    (tmp_path / "python.exe").write_bytes(b"")
    assert gui_python(str(tmp_path / "python.exe")) == str(tmp_path / "python.exe")  # no pythonw there
    (tmp_path / "pythonw.exe").write_bytes(b"")
    assert gui_python(str(tmp_path / "python.exe")) == str(tmp_path / "pythonw.exe")
    assert gui_python("/usr/bin/python3") == "/usr/bin/python3"


def test_ready_file_round_trip_and_garbage_is_not_ready(tmp_path):
    path = str(tmp_path / "ready.json")
    assert read_ready(path) is None
    write_ready(path, 4321)
    assert read_ready(path) == {"pid": os.getpid(), "port": 4321}
    for junk in ("{", "[]", '{"pid": "1", "port": 2}'):
        (tmp_path / "ready.json").write_text(junk)
        assert read_ready(path) is None


def test_ready_file_needs_listen(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--ready-file", "x.json"])
    assert "--ready-file needs --listen" in capsys.readouterr().err


def test_main_writes_the_ready_file_once_shown_and_removes_it_on_exit(qapp, monkeypatch, tmp_path):
    ready = str(tmp_path / "ready.json")
    seen = {}

    def fake_exec(_app):
        [window] = [w for w in qapp.topLevelWidgets() if isinstance(w, MainWindow) and w.isVisible()]
        seen["ready"] = read_ready(ready)
        seen["title"] = window.windowTitle()
        seen["no_activate"] = window.testAttribute(Qt.WA_ShowWithoutActivating)
        window.close()
        window.deleteLater()
        return 0

    monkeypatch.setattr(QApplication, "exec_", fake_exec)
    assert drawapp_main.main(launch_argv("pyw", ready)[3:]) == 0  # exactly what launch() runs
    assert seen["ready"]["pid"] == os.getpid()
    assert seen["no_activate"]  # shown without taking the focus from whatever had it
    assert seen["title"].endswith(f"127.0.0.1:{seen['ready']['port']}")
    assert not os.path.exists(ready)


# --- a real child process (offscreen): what a sister's process does ---------------------

def test_launch_draws_and_closes_a_real_child(tmp_path):
    app = launch(author="aiva", width=300, height=200, extra_env=OFFSCREEN, run_dir=str(tmp_path))
    try:
        assert app.running() and app.port > 0
        with app.client() as draw:
            stroke = draw.stroke([(10, 10), (290, 190)], author="aiva", width=5)
            assert stroke["ok"] and stroke["stroke_id"].startswith("aiva-")
            with pytest.raises(CommandRejected, match="outside the 300x200 canvas"):
                draw.stroke([(10, 10), (400, 10)])
            assert draw.clear(author="saga")["revision"] == stroke["revision"] + 1
            # Back to back on one connection: with QTcpSocket, the line after a
            # stroke's reply was never read (Qt 5.15 lost the notification).
            for i in range(30):
                draw.stroke([(10, 10 + i), (200, 10 + i)], author="aiva")
                draw.clear()
        with open(app.log_path, encoding="utf-8") as f:
            assert f"listening for drawing commands on 127.0.0.1:{app.port}" in f.read()
    finally:
        app.close()
    assert not app.running()
    assert not os.path.exists(app.ready_file)
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", app.port), timeout=1).close()


def test_launch_on_a_taken_port_fails_fast_with_the_apps_reason(tmp_path):
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen()
        port = squatter.getsockname()[1]
        with pytest.raises(LaunchError, match=f"exited with code 1: .*cannot listen on 127.0.0.1:{port}"):
            launch(port=port, extra_env=OFFSCREEN, timeout=15, run_dir=str(tmp_path))


def test_a_child_that_never_gets_ready_is_killed_at_the_timeout(tmp_path, monkeypatch):
    hang = tmp_path / "hang.py"
    hang.write_text("import time; time.sleep(60)")
    monkeypatch.setattr(launcher, "launch_argv", lambda python, *a, **k: [python, str(hang)])
    with pytest.raises(LaunchError, match="wasn't listening after 0.5s") as e:
        launch(timeout=0.5, run_dir=str(tmp_path))
    assert "[log: " in str(e.value)


# --- the no-terminal check entry point (what a double-click or Aiva's open_file runs) -----------

def test_project_python_is_the_repo_venv():
    python = launcher.project_python()
    if python is not None:  # absent in a checkout without ../.venv
        assert python.lower().endswith(os.path.join(".venv", "scripts", "python.exe"))


def test_verify_drawing_pyw_runs_the_check_on_the_repo_venv(tmp_path):
    import json
    import subprocess
    import sys

    report = tmp_path / "report.json"
    script = os.path.join(launcher.PROJECT_DIR, "verify_drawing.pyw")
    done = subprocess.run([sys.executable, script, "--offscreen", "--report", str(report)],
                          capture_output=True, timeout=60,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    result = json.loads(report.read_text(encoding="utf-8"))
    assert done.returncode == 0 and result["ok"], result
    assert {c["name"] for c in result["checks"]} >= {"launch", "aiva stroke", "saga stroke", "close"}
    if launcher.project_python():  # the app ran on the venv, whatever interpreter ran the script
        assert result["interpreter"] == launcher.gui_python(launcher.project_python())


# --- the existing canvas: found again while it runs, back with its drawing after it ends ---------

def test_the_save_folder_reaches_the_child_command_line():
    argv = launch_argv("pyw", "r.json", author="aiva", save_dir="D:/drawings")
    assert argv[-2:] == ["--save-dir", "D:/drawings"]
    assert parse_args(argv[3:]).save_dir == "D:/drawings"
    assert parse_args([]).save_dir is None                  # the app picks storage.default_save_dir()


def test_pid_alive_never_signals_and_sees_a_finished_process():
    import subprocess
    import sys

    assert launcher.pid_alive(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(10)
    assert not launcher.pid_alive(child.pid)


def test_a_stale_or_foreign_window_file_is_never_adopted(tmp_path):
    from drawapp.storage import window_file

    save_dir = str(tmp_path / "drawings")
    assert launcher.find_running("aiva", save_dir) is None              # no file at all
    path = window_file(save_dir, "aiva")
    os.makedirs(os.path.dirname(path))
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        dead_port = free.getsockname()[1]
    for junk in ("{", "[]", '{"pid": 1, "port": "x"}',
                 json.dumps({"pid": os.getpid(), "port": dead_port, "author": "aiva"})):
        with open(path, "w", encoding="utf-8") as f:
            f.write(junk)
        assert launcher.find_running("aiva", save_dir, timeout=0.5) is None


def test_a_running_window_is_found_again_and_a_killed_one_comes_back_with_its_drawing(tmp_path):
    save_dir = str(tmp_path / "drawings")
    runs = [tmp_path / "run1", tmp_path / "run2"]
    for r in runs:
        r.mkdir()
    app = launch(author="aiva", width=300, height=200, extra_env=OFFSCREEN, run_dir=str(runs[0]),
                 save_dir=save_dir)
    try:
        assert app.restored == 0 and not app.adopted
        with app.client() as draw:
            draw.stroke([(10, 10), (200, 150)], author="aiva", width=5)
        found = launcher.find_running("aiva", save_dir)          # what a restarted sister does
        assert found is not None and found.adopted and found.running()
        assert (found.pid, found.port) == (app.pid, app.port)
        assert launcher.find_running("saga", save_dir) is None   # her sister's canvas is her own
        with found.client() as draw:
            assert draw.status()["strokes"] == 1
    finally:
        app.close()                                               # TerminateProcess: no cleanup ran
    assert launcher.find_running("aiva", save_dir) is None        # the leftover file is stale

    again = launch(author="aiva", width=300, height=200, extra_env=OFFSCREEN, run_dir=str(runs[1]),
                   save_dir=save_dir)
    try:
        assert again.restored == 1                                # the canvas was not lost
        with again.client() as draw:
            status = draw.status()
            assert status["strokes"] == 1 and status["unsaved"] is True
            saved = draw.save(author="aiva", name="kept")
            assert os.path.isfile(saved["path"]) and os.path.isfile(saved["png"])
            assert [d["name"] for d in draw.drawings()["drawings"]] == ["kept"]
        adopted = launcher.find_running("aiva", save_dir)
        assert adopted is not None and adopted.pid == again.pid
        adopted.close()                                           # an adopted window can be ended too
        assert not adopted.running()
    finally:
        again.close()


# --- the sisters' one canvas: named for the canvas, found from either process, opened once ------

def test_a_shared_canvas_reaches_the_child_command_line():
    argv = launch_argv("pyw", "r.json", author="user", canvas="shared", absorb=("aiva", "saga"))
    assert argv[7:] == ["--author", "user", "--canvas", "shared", "--absorb", "aiva",
                        "--absorb", "saga", "--no-activate"]
    parsed = parse_args(argv[3:])
    assert (parsed.author, parsed.canvas, parsed.absorb) == ("user", "shared", ["aiva", "saga"])
    assert "--canvas" not in launch_argv("pyw", "r.json", author="aiva", canvas="aiva")
    assert parse_args([]).canvas is None and parse_args([]).absorb == []


@pytest.mark.parametrize("bad", [["--canvas", "../x"], ["--absorb", "a b"], ["--canvas", "shared.x"]])
def test_canvas_names_are_file_safe(bad, capsys):
    with pytest.raises(SystemExit):
        parse_args(bad)
    assert "lowercase letters, digits or _" in capsys.readouterr().err


def test_a_shared_window_is_found_by_its_canvas_from_any_process(tmp_path):
    save_dir = str(tmp_path / "drawings")
    app = launch(canvas="shared", author="user", width=300, height=200, extra_env=OFFSCREEN,
                 run_dir=str(tmp_path), save_dir=save_dir)
    try:
        found = launcher.find_running("shared", save_dir)        # what either sister does
        assert found is not None and (found.pid, found.port) == (app.pid, app.port)
        assert launcher.find_running("user", save_dir) is None    # the mouse's author isn't the canvas
        assert launcher.find_running("aiva", save_dir) is None
        with found.client() as draw:
            draw.stroke([(10, 10), (100, 100)], author="aiva")
            draw.stroke([(20, 20), (120, 120)], author="saga")
            assert draw.undo(author="aiva")["remaining"] == 0      # hers only; Saga's stays
            assert draw.status()["strokes"] == 1
    finally:
        app.close()


def test_opening_lets_one_process_at_a_time_find_or_launch(tmp_path):
    save_dir = str(tmp_path / "drawings")
    with launcher.opening("shared", save_dir):
        with pytest.raises(LaunchError, match="opening the shared canvas for over 0.2s"):
            with launcher.opening("shared", save_dir, timeout=0.2):
                pass
        with launcher.opening("aiva", save_dir, timeout=0.2):       # another canvas isn't held
            pass
    with launcher.opening("shared", save_dir, timeout=0.2):         # released on the way out
        pass
