"""Opt-in loopback command channel: JSON lines in, one JSON reply line per command.

    python -m drawapp --listen          # 127.0.0.1:8770
    python -m drawapp --listen 0        # any free port (printed at startup)

Off unless --listen is given, and bound to 127.0.0.1 only. Commands (see
commands.py) are handled on the GUI thread, so they reach the document the same
way the mouse does and the canvas repaints the same way. A bad line gets an
error reply and the connection stays open.

The sockets are plain Python sockets on worker threads, not QTcpSocket: on
Windows, Qt 5.15.2's socket notifiers can lose a connection's read notification
for good, and the client then waits forever for a reply to a line nobody read
(a client that sent its next command right after a stroke's reply hit it every
time). A worker reads a line, hands it to the GUI thread through a queued
signal, waits for the reply and writes it back, so each connection is answered
in order and a command is still applied whole between two GUI events.
"""
from __future__ import annotations

import json
import socket
import threading
import time

from PyQt5.QtCore import QObject, Qt, pyqtSignal

from drawapp.commands import DEFAULT_PORT, CommandRunner  # noqa: F401 (re-exported)

HOST = "127.0.0.1"
MAX_LINE = 1 << 20   # bytes without a newline before a client is cut off


class _Request:
    """One line on its way to the GUI thread and its reply on the way back."""
    __slots__ = ("line", "reply", "done")

    def __init__(self, line: bytes):
        self.line = line
        self.reply: dict | None = None
        self.done = threading.Event()


class _Sockets:
    """The listener and open connections; plain Python, so it can be closed from `destroyed`."""

    def __init__(self):
        self.listener: socket.socket | None = None
        self.clients: set[socket.socket] = set()
        self.lock = threading.Lock()
        self.closed = threading.Event()

    def close(self, *_) -> None:
        self.closed.set()
        with self.lock:
            doomed = [s for s in (self.listener, *self.clients) if s is not None]
            self.listener = None
            self.clients.clear()
        for s in doomed:
            _hang_up(s)


class CommandServer(QObject):
    _request = pyqtSignal(object)

    def __init__(self, runner: CommandRunner, parent: QObject | None = None):
        super().__init__(parent)
        self.runner = runner
        self._sockets = _Sockets()
        self._port = 0
        self._request.connect(self._on_request, Qt.QueuedConnection)
        self.destroyed.connect(self._sockets.close)  # a window that goes away stops listening

    def start(self, port: int = DEFAULT_PORT) -> int:
        """Listen on 127.0.0.1:`port` (0 = any free one); returns the bound port."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows: nobody else may bind it too
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((HOST, port))
            listener.listen()
        except OSError as e:
            listener.close()
            raise OSError(f"cannot listen on {HOST}:{port}: {e.strerror or e}") from None
        self._sockets.listener = listener
        self._port = listener.getsockname()[1]
        threading.Thread(target=self._accept_loop, args=(listener,),
                         name="drawapp-accept", daemon=True).start()
        return self._port

    def isListening(self) -> bool:
        return self._sockets.listener is not None

    def address(self) -> tuple[str, int]:
        return HOST, self._port

    def serverPort(self) -> int:
        return self._port

    def close(self) -> None:
        self._sockets.close()

    # --- worker threads -----------------------------------------------------------

    def _accept_loop(self, listener: socket.socket) -> None:
        socks = self._sockets
        while not socks.closed.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:  # closed
                return
            with socks.lock:
                if socks.closed.is_set():
                    _hang_up(conn)
                    return
                socks.clients.add(conn)
            threading.Thread(target=self._serve, args=(conn,),
                             name="drawapp-client", daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        buf = bytearray()
        try:
            while chunk := conn.recv(65536):
                buf.extend(chunk)
                while (nl := buf.find(b"\n")) >= 0:
                    line = bytes(buf[:nl])
                    del buf[:nl + 1]
                    if line.strip():
                        reply = self._on_gui_thread(line)
                        if reply is None:  # shutting down
                            return
                        conn.sendall(_encode(reply))
                if len(buf) > MAX_LINE:
                    conn.sendall(_encode({"ok": False, "code": "invalid",
                                          "error": f"line longer than {MAX_LINE} bytes"}))
                    _drain_and_close(conn)
                    return
        except (OSError, RuntimeError):  # client gone, or the server object deleted mid-emit
            pass
        finally:
            with self._sockets.lock:
                self._sockets.clients.discard(conn)
            _hang_up(conn)

    def _on_gui_thread(self, line: bytes) -> dict | None:
        req = _Request(line)
        self._request.emit(req)
        while not req.done.wait(0.25):
            if self._sockets.closed.is_set():
                return None
        return req.reply

    # --- GUI thread -----------------------------------------------------------------

    def _on_request(self, req: _Request) -> None:
        try:
            req.reply = self._handle(req.line)
        finally:
            req.done.set()

    def _handle(self, line: bytes) -> dict:
        # An exception escaping a PyQt5 slot aborts the whole app; never let one.
        try:
            return self.runner.handle(line)
        except Exception as e:
            return {"ok": False, "code": "internal", "error": f"{type(e).__name__}: {e}"}


def _encode(reply: dict) -> bytes:
    return json.dumps(reply).encode() + b"\n"


def _hang_up(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)  # also wakes a thread blocked in accept/recv
    except OSError:
        pass
    sock.close()


def _drain_and_close(conn: socket.socket) -> None:
    """Close after a final reply without a reset: closing with unread input sends RST,
    which can destroy the reply before the client reads it."""
    deadline = time.monotonic() + 2.0  # a client that never stops sending gets the reset anyway
    try:
        conn.shutdown(socket.SHUT_WR)
        conn.settimeout(0.5)
        while conn.recv(65536) and time.monotonic() < deadline:
            pass
    except OSError:
        pass
