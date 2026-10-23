"""DrawClient against a scripted stdlib server: the wire it writes and how it takes replies. No Qt."""
import json
import socket
import threading

import pytest

from drawapp.client import CommandRejected, DrawClient, DrawConnectionError, DrawError


class FakeApp:
    """Accepts one connection at a time; answers each line with `reply(cmd)` (None = hang up)."""

    def __init__(self, reply):
        self.reply = reply
        self.received = []
        self.sock = socket.create_server(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn, conn.makefile("rb") as lines:
                for line in lines:
                    cmd = json.loads(line)
                    self.received.append(cmd)
                    answer = self.reply(cmd)
                    if answer is None:
                        break
                    conn.sendall(answer if isinstance(answer, bytes) else json.dumps(answer).encode() + b"\n")

    def close(self):
        self.sock.close()


@pytest.fixture
def fake():
    apps = []

    def make(reply):
        apps.append(FakeApp(reply))
        return apps[-1]
    yield make
    for a in apps:
        a.close()


def test_stroke_and_clear_send_the_documented_commands(fake):
    app = fake(lambda cmd: {"ok": True, "revision": len(app.received)})
    with DrawClient(app.port) as draw:
        assert draw.stroke([(1, 2), (3.5, 4)], author="aiva", width=6) == {"ok": True, "revision": 1}
        draw.stroke(iter([(5, 5)]))  # any iterable of pairs
        draw.clear(author="saga")
    assert app.received == [
        {"cmd": "stroke", "author": "aiva", "points": [[1, 2], [3.5, 4]], "width": 6},
        {"cmd": "stroke", "author": "user", "points": [[5, 5]]},  # color/width left to the app
        {"cmd": "clear", "author": "saga"},
    ]


def test_a_refused_command_raises_with_the_apps_code_and_the_connection_stays(fake):
    app = fake(lambda cmd: {"ok": False, "code": "invalid", "error": "outside"}
               if cmd["cmd"] == "stroke" else {"ok": True, "revision": 7})
    with DrawClient(app.port) as draw:
        with pytest.raises(CommandRejected) as e:
            draw.stroke([(-1, 0)])
        assert (e.value.code, e.value.error) == ("invalid", "outside")
        assert draw.send({"cmd": "stroke", "points": [[0, 0]]})["ok"] is False  # send() never raises on ok:false
        assert draw.clear()["revision"] == 7


def test_no_app_listening_is_a_connection_error():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # bound but not listening: refused
        with pytest.raises(DrawConnectionError, match=str(port)):
            DrawClient(port, timeout=1).clear()


def test_a_hang_up_is_a_connection_error_and_the_next_call_reconnects(fake):
    calls = []

    def reply(cmd):
        calls.append(cmd)
        return None if len(calls) == 1 else {"ok": True, "revision": 1}
    app = fake(reply)
    draw = DrawClient(app.port)
    with pytest.raises(DrawConnectionError, match="closed"):
        draw.clear()
    assert draw.clear() == {"ok": True, "revision": 1}  # nothing was retried behind our back
    assert len(calls) == 2
    draw.close()


def test_a_silent_app_times_out_instead_of_hanging(fake):
    gate = threading.Event()
    app = fake(lambda cmd: gate.wait(5) and None)
    with pytest.raises(DrawConnectionError, match="lost"):
        DrawClient(app.port, timeout=0.2).clear()
    gate.set()


def test_a_reply_that_isnt_drawapps_is_an_error(fake):
    app = fake(lambda cmd: b"HTTP/1.1 400 Bad Request\r\n")
    with pytest.raises(DrawError, match="not a drawapp reply"):
        DrawClient(app.port).clear()


def test_undo_sends_the_documented_command(fake):
    app = fake(lambda cmd: {"ok": True, "stroke_id": "saga-1", "revision": 4, "remaining": 0})
    with DrawClient(app.port) as draw:
        assert draw.undo(author="saga")["stroke_id"] == "saga-1"
    assert app.received == [{"cmd": "undo", "author": "saga"}]
