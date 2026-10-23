"""The loopback command channel, driven over a real socket into an unshown window (offscreen)."""
import json
import socket
import threading
import time

import pytest
from PyQt5.QtCore import QEvent
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication

from drawapp import app as drawapp_main
from drawapp.app import MainWindow, parse_args, start_listener
from drawapp.commands import CommandRunner
from drawapp.model import Document
from drawapp.server import DEFAULT_PORT, MAX_LINE, CommandServer

WHITE = QColor("#ffffff").rgb()


@pytest.fixture
def served(qapp):
    window = MainWindow(Document(200, 120), author="user")  # never shown
    server = CommandServer(CommandRunner(window.document), parent=window)
    port = server.start(0)
    client = socket.create_connection(("127.0.0.1", port), timeout=5)
    client.settimeout(0.01)
    yield window, server, client
    client.close()
    server.close()
    window.deleteLater()


def _pump_until(qapp, client, lines, timeout=5.0):
    """Run the Qt event loop (the server lives on this thread) until `lines` replies arrive."""
    data = b""
    deadline = time.monotonic() + timeout
    while data.count(b"\n") < lines:
        assert time.monotonic() < deadline, f"timed out; got {data!r}"
        qapp.processEvents()
        try:
            chunk = client.recv(65536)
        except socket.timeout:
            continue
        if not chunk:
            break
        data += chunk
    return [json.loads(line) for line in data.splitlines()]


def test_it_listens_on_loopback_only(served):
    _window, server, _client = served
    assert server.isListening() and server.address() == ("127.0.0.1", server.serverPort())


def test_a_taken_port_is_a_clear_error(served):
    window, server, _client = served
    second = CommandServer(CommandRunner(window.document))
    with pytest.raises(OSError, match=f"127.0.0.1:{server.serverPort()}"):
        second.start(server.serverPort())


def test_a_stroke_command_draws_on_the_window_canvas(qapp, served):
    window, _server, client = served
    client.sendall(b'{"cmd": "stroke", "author": "aiva", "points": [[20, 60], [180, 60]], "width": 8}\n')
    [reply] = _pump_until(qapp, client, 1)

    doc = window.document
    assert reply == {"ok": True, "stroke_id": doc.strokes[0].id, "revision": doc.revision}
    assert doc.strokes[0].author == "aiva" and doc.strokes[0].done
    assert not window.canvas.tool.active  # the pointer's own tool was never touched
    shot = window.canvas.grab().toImage()
    assert shot.pixel(100, 60) != WHITE  # on the line
    assert shot.pixel(100, 10) == WHITE  # well off it


def test_bad_lines_get_error_replies_and_the_connection_stays_usable(qapp, served):
    window, _server, client = served
    client.sendall(b'{oops\n'
                   b'{"cmd": "spiral"}\n'
                   b'\n'  # blank lines are skipped, not answered
                   b'{"cmd": "stroke", "points": [[999, 1]]}\n'
                   b'{"cmd": "stroke", "points": [[1, 1], [9, 9]]}\r\n')
    bad_json, unsupported, off_canvas, ok = _pump_until(qapp, client, 4)

    assert (bad_json["ok"], bad_json["code"]) == (False, "invalid")
    assert (unsupported["ok"], unsupported["code"]) == (False, "unsupported")
    assert (off_canvas["ok"], off_canvas["code"]) == (False, "invalid")
    assert ok["ok"] is True
    assert len(window.document.strokes) == 1


def test_a_command_split_across_packets_is_reassembled(qapp, served):
    window, _server, client = served
    client.sendall(b'{"cmd": "stroke", "points": ')
    _pump_until(qapp, client, 0)
    assert window.document.strokes == []
    client.sendall(b'[[5, 5], [50, 50]]}\n{"cmd": "clear"}\n')
    first, second = _pump_until(qapp, client, 2)
    assert first["ok"] and second["ok"] and window.document.strokes == []


def test_an_unexpected_error_is_a_reply_not_a_crash(qapp, served):
    _window, server, client = served

    def boom(_line):
        raise RuntimeError("kaboom")
    server.runner.handle = boom
    client.sendall(b'{"cmd": "clear"}\n')
    [reply] = _pump_until(qapp, client, 1)
    assert reply == {"ok": False, "code": "internal", "error": "RuntimeError: kaboom"}


def test_an_endless_line_is_refused_and_the_client_dropped(qapp, served):
    window, _server, client = served
    sender = threading.Thread(target=lambda: _send_quietly(client, b"x" * (MAX_LINE + 2)))
    sender.start()
    [reply] = _pump_until(qapp, client, 1)
    sender.join(5)
    assert reply["ok"] is False and "longer than" in reply["error"]
    assert window.document.revision == 0


def test_commands_back_to_back_on_one_connection_all_get_replies(qapp, served):
    # The worker-thread -> GUI-thread handoff, in order, as DrawClient drives it. (The
    # Qt-socket bug this server replaced only showed up out of process; that
    # regression lives in test_launcher.test_launch_draws_and_closes_a_real_child.)
    window, server, _client = served
    replies, errors = [], []

    def chatty():
        try:
            with socket.create_connection(("127.0.0.1", server.serverPort()), timeout=5) as c:
                lines = c.makefile("rb")
                for i in range(40):
                    cmd = {"cmd": "stroke", "points": [[5, 5 + i], [150, 5 + i]]} if i % 2 == 0 else {"cmd": "clear"}
                    c.sendall(json.dumps(cmd).encode() + b"\n")
                    replies.append(json.loads(lines.readline()))
        except Exception as e:
            errors.append(e)

    worker = threading.Thread(target=chatty)
    worker.start()
    deadline = time.monotonic() + 10
    while worker.is_alive():  # the GUI thread (this one) has to run for commands to apply
        assert time.monotonic() < deadline, f"stalled after {len(replies)} replies"
        qapp.processEvents()
    assert not errors and len(replies) == 40 and all(r["ok"] for r in replies)
    assert [r["revision"] for r in replies][-2:] == [window.document.revision - 1, window.document.revision]


def test_closing_the_server_hangs_up_and_stops_listening(qapp, served):
    _window, server, client = served
    port = server.serverPort()
    server.close()
    client.settimeout(5)
    assert client.recv(10) == b""  # the open connection is closed, not left hanging
    assert not server.isListening()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


def test_a_deleted_window_takes_its_listener_with_it(qapp):
    window = MainWindow(Document(50, 50))
    port = start_listener(window, 0).serverPort()
    window.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


def _send_quietly(sock, data):
    # The socket's short timeout is shared with the pumping thread's recv; keep it.
    view = memoryview(data)
    deadline = time.monotonic() + 5
    while view and time.monotonic() < deadline:
        try:
            view = view[sock.send(view):]
        except socket.timeout:
            continue
        except OSError:  # the server may hang up before the last bytes are sent
            return


# --- launch wiring: `python -m drawapp --listen` end to end, minus the event loop ---------

def test_bare_listen_is_the_default_port_and_no_flag_is_no_listener():
    assert parse_args(["--listen"]).listen == DEFAULT_PORT == 8770
    assert parse_args(["--listen", "0"]).listen == 0
    assert parse_args([]).listen is None


def test_a_listening_window_says_so_in_its_title(qapp):
    quiet = MainWindow(Document(50, 50), author="aiva")
    loud = MainWindow(Document(50, 50), author="aiva")
    server = start_listener(loud, 0)
    try:
        assert "commands" not in quiet.windowTitle()
        assert loud.windowTitle().endswith(f"· commands on 127.0.0.1:{server.serverPort()}")
        assert f"127.0.0.1:{server.serverPort()}" in loud.statusBar().currentMessage()
    finally:
        server.close()
        quiet.deleteLater()
        loud.deleteLater()


def test_main_listens_before_showing_and_a_stroke_lands_on_the_shown_window(qapp, monkeypatch, capsys):
    seen = {}

    def fake_exec(_app):  # stands in for the event loop; the window is offscreen
        out = capsys.readouterr().out
        port = int(out.rsplit(":", 1)[1])
        [window] = [w for w in qapp.topLevelWidgets() if isinstance(w, MainWindow) and w.isVisible()]
        assert str(port) in window.windowTitle()
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            client.settimeout(0.01)
            client.sendall(b'{"cmd": "stroke", "points": [[20, 60], [180, 60]], "width": 8}\n')
            [seen["reply"]] = _pump_until(qapp, client, 1)
        seen["strokes"] = list(window.document.strokes)
        seen["pixel"] = window.canvas.grab().toImage().pixel(100, 60)
        window.close()
        window.deleteLater()
        return 0

    monkeypatch.setattr(QApplication, "exec_", fake_exec)
    assert drawapp_main.main(["--listen", "0", "--author", "Saga", "--width", "200", "--height", "120"]) == 0

    [stroke] = seen["strokes"]
    assert seen["reply"] == {"ok": True, "stroke_id": stroke.id, "revision": 3}
    assert stroke.author == "user" and stroke.done  # the command's author, not the window's
    assert seen["pixel"] != WHITE


def test_main_with_a_taken_port_says_which_and_never_shows_a_window(qapp, monkeypatch, capsys):
    monkeypatch.setattr(QApplication, "exec_", lambda _app: pytest.fail("the event loop started"))
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen()
        port = squatter.getsockname()[1]
        assert drawapp_main.main(["--listen", str(port)]) == 1
    assert f"cannot listen on 127.0.0.1:{port}" in capsys.readouterr().err
    assert not [w for w in qapp.topLevelWidgets() if isinstance(w, MainWindow) and w.isVisible()]
