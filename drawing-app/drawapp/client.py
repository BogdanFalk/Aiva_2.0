"""Drawing commands from code: the JSON-lines wire of server.py as method calls. No Qt.

    from drawapp.client import DrawClient
    with DrawClient(port) as draw:
        draw.stroke([(100, 100), (300, 180)], author="aiva", width=6)
        draw.clear(author="saga")
        draw.undo(author="aiva")                   # takes back aiva's newest stroke, nobody else's
        draw.save(author="aiva", name="waves")     # -> {"path": ".../waves.aivadraw", "png": ...}

Stdlib only, so a process that never loads Qt (a sister's) can import it. The
wire is unchanged: one JSON object per line out, one JSON reply line back
(commands.py has the fields). `send()` returns whatever the app replied;
`stroke()`/`clear()` raise CommandRejected when the reply is `"ok": false`.

A broken or timed-out connection raises DrawConnectionError and is dropped;
the next call connects again. Nothing is retried on its own: a stroke whose
reply was lost may well have been drawn, and sending it twice would draw it twice.
"""
from __future__ import annotations

import json
import socket
from typing import Iterable

from drawapp.commands import DEFAULT_PORT

HOST = "127.0.0.1"  # the app only ever listens on loopback


class DrawError(Exception):
    """Talking to the drawing app failed."""


class DrawConnectionError(DrawError):
    """Couldn't reach the app, or the connection broke / timed out mid-command."""


class CommandRejected(DrawError):
    """The app answered `"ok": false`; nothing was drawn."""

    def __init__(self, reply: dict):
        self.reply = reply
        self.code = reply.get("code", "")
        self.error = reply.get("error", "")
        super().__init__(f"{self.code}: {self.error}")


class DrawClient:
    """One loopback connection to a listening drawapp (`--listen`), opened on first use."""

    def __init__(self, port: int = DEFAULT_PORT, host: str = HOST, timeout: float = 5.0):
        self.port = port
        self.host = host
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._reader = None

    def connect(self) -> "DrawClient":
        if self._sock is None:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            except OSError as e:
                raise DrawConnectionError(
                    f"no drawing app on {self.host}:{self.port} ({e})") from None
            self._sock, self._reader = sock, sock.makefile("rb")
        return self

    def send(self, command: dict) -> dict:
        """Send one command, return its reply (ok or not). DrawConnectionError if unanswered."""
        line = json.dumps(command).encode() + b"\n"  # json.dumps escapes newlines: one line
        self.connect()
        try:
            self._sock.sendall(line)
            raw = self._reader.readline()
        except OSError as e:  # socket.timeout included; the stream is out of step now
            self.close()
            raise DrawConnectionError(f"lost the drawing app on port {self.port} ({e})") from None
        if not raw:
            self.close()
            raise DrawConnectionError(f"the drawing app on port {self.port} closed the connection")
        try:
            reply = json.loads(raw)
        except ValueError:
            self.close()
            raise DrawError(f"not a drawapp reply: {raw[:80]!r}") from None
        if not isinstance(reply, dict):
            self.close()
            raise DrawError(f"not a drawapp reply: {raw[:80]!r}")
        return reply

    def command(self, command: dict) -> dict:
        """send(), but a rejected command raises CommandRejected."""
        reply = self.send(command)
        if reply.get("ok") is not True:
            raise CommandRejected(reply)
        return reply

    def stroke(self, points: Iterable[tuple[float, float]], author: str = "user",
               color: str | None = None, width: float | None = None) -> dict:
        """Draw one stroke through `points` (document pixels); returns {"stroke_id", "revision", ...}."""
        cmd = {"cmd": "stroke", "author": author, "points": [[x, y] for x, y in points]}
        if color is not None:
            cmd["color"] = color
        if width is not None:
            cmd["width"] = width
        return self.command(cmd)

    def clear(self, author: str = "user") -> dict:
        return self.command({"cmd": "clear", "author": author})

    def undo(self, author: str = "user") -> dict:
        """Take back `author`'s newest finished stroke; returns {"stroke_id" (None if they had
        none), "revision", "remaining"}. A window older than undo (drawapp < 0.4) raises
        CommandRejected with code "unsupported"."""
        return self.command({"cmd": "undo", "author": author})

    def save(self, author: str = "user", name: str | None = None) -> dict:
        """Save the canvas (to its own file, or a new one named `name`); returns {"path", "png", "name", ...}."""
        cmd = {"cmd": "save", "author": author}
        if name is not None:
            cmd["name"] = name
        return self.command(cmd)

    def open(self, name: str | None = None, author: str = "user") -> dict:
        """Put a saved drawing on the canvas (default: the newest); unsaved work is saved first."""
        cmd = {"cmd": "open", "author": author}
        if name is not None:
            cmd["name"] = name
        return self.command(cmd)

    def drawings(self) -> dict:
        """{"save_dir", "drawings": [{"name", "path", "modified"}], "recovery": [...]}, newest first."""
        return self.command({"cmd": "drawings"})

    def status(self) -> dict:
        """{"pid", "author", "revision", "strokes", "save_dir", "current", "unsaved"} of the window."""
        return self.command({"cmd": "status"})

    def close(self) -> None:
        sock, reader = self._sock, self._reader
        self._sock = self._reader = None
        for f in (reader, sock):
            if f is not None:
                try:
                    f.close()
                except OSError:
                    pass

    def __enter__(self) -> "DrawClient":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()
