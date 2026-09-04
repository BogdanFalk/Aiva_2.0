"""Aiva's handle on her own body — the client for the avatar renderer process.

Plays the role modules/vtube_studio.py used to: constructed in main(), passed
to register_tools, and every failure degrades to voice-only instead of taking
her down. The renderer (modules/avatar_renderer.py) is a separate process so
a GL crash can't kill her voice, and so her body survives her self-restarts:
ensure_running() first tries to ADOPT a live renderer via aiva/avatar_state.json
(pid + hello handshake) and only spawns a new one if there isn't one.

Two traffic classes on one socket:
- request/reply (expressions, moves, position reads): async, id-matched.
- the lip-sync hot path (send_mouth/send_speech): sync fire-and-forget —
  never awaits, never reconnects, swallows every error. The voice pipeline
  must not hiccup because her body blinked.

Standalone probe (never starts Aiva):  cd aiva && python -m modules.avatar --probe
Runs a scripted command sequence against a running renderer (adopting it, or
spawning one if none is alive).
"""

import asyncio
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from modules import avatar_ipc

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


class AvatarClient:
    def __init__(self):
        self.connected = False
        self.expressions = []       # names from hello, for the system prompt
        self._reader = None
        self._writer = None
        self._reader_task = None
        self._pending = {}          # id -> Future
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock() # serializes connect/spawn attempts

    # -- lifecycle -------------------------------------------------------------

    async def ensure_running(self) -> bool:
        """Adopt a live renderer, or spawn one. Non-fatal on failure."""
        async with self._lock:
            if self.connected:
                return True
            state = avatar_ipc.read_state()
            if state and state.get("pid") and avatar_ipc.pid_alive(state["pid"]):
                if await self._connect(state):
                    print(f"Avatar: adopted running renderer (pid {state['pid']})")
                    return True
            return await self._spawn(state)

    async def _spawn(self, old_state) -> bool:
        old_boot = (old_state or {}).get("boot_id")
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
            log_path = os.path.join(_LOG_DIR, f"avatar-{datetime.now():%Y%m%d}.log")
            log = open(log_path, "a", encoding="utf-8")
            aiva_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            subprocess.Popen(
                [sys.executable, "-m", "modules.avatar_renderer"],
                cwd=aiva_dir, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW, close_fds=False,
            )
        except OSError as e:
            print(f"Avatar: couldn't spawn renderer ({e}) — going voice-only")
            return False
        # wait for the NEW instance's state file (boot_id changes), then connect;
        # first boot loads 14 textures, so allow a generous window
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            await asyncio.sleep(0.25)
            state = avatar_ipc.read_state()
            if state and state.get("boot_id") and state["boot_id"] != old_boot:
                if await self._connect(state):
                    print(f"Avatar: renderer up (pid {state.get('pid')})")
                    return True
        print("Avatar: renderer didn't come up in 20s — going voice-only")
        return False

    async def _connect(self, state) -> bool:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", state["port"]), timeout=2)
        except (OSError, asyncio.TimeoutError, KeyError, TypeError):
            return False
        self._reader, self._writer = reader, writer
        self.connected = True
        if self._reader_task:
            self._reader_task.cancel()
        self._reader_task = asyncio.create_task(self._read_loop())
        hello = await self._request({"cmd": "hello"})
        if not hello or not hello.get("ok"):
            self._drop()
            return False
        self.expressions = hello.get("expressions", [])
        return True

    def _drop(self):
        self.connected = False
        if self._writer:
            try:
                self._writer.close()
            except OSError:
                pass
        self._reader = self._writer = None
        for fut in self._pending.values():
            if not fut.done():
                fut.set_result(None)
        self._pending.clear()

    async def _read_loop(self):
        try:
            while self._reader:
                line = await self._reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fut = self._pending.pop(msg.get("id"), None)
                if fut and not fut.done():
                    fut.set_result(msg)
        except (OSError, asyncio.CancelledError):
            pass
        finally:
            self._drop()

    # -- request/reply ----------------------------------------------------------

    async def _request(self, msg, timeout=2.0):
        """One id-matched round-trip; None on any failure."""
        if not self.connected:
            return None
        msg = dict(msg, id=next(self._ids))
        fut = asyncio.get_running_loop().create_future()
        self._pending[msg["id"]] = fut
        try:
            self._writer.write(avatar_ipc.encode(msg))
            await self._writer.drain()
            return await asyncio.wait_for(fut, timeout)
        except (OSError, asyncio.TimeoutError):
            self._pending.pop(msg["id"], None)
            self._drop()
            return None

    async def _command(self, msg):
        """Request with one reconnect attempt, mirroring the old
        VTubeStudio.ensure_connected behavior."""
        reply = await self._request(msg)
        if reply is None and await self.ensure_running():
            reply = await self._request(msg)
        return reply

    # -- the tool-facing API ------------------------------------------------------

    async def toggle_expression(self, name):
        """(ok, detail): detail is the matched name or an error string."""
        reply = await self._command({"cmd": "expression", "name": name})
        if reply is None:
            return False, "the avatar isn't running right now"
        if not reply.get("ok"):
            return False, reply.get("error", "unknown expression")
        return True, reply.get("toggled", name)

    async def move(self, x=None, y=None, rotation=None, size=None):
        reply = await self._command(
            {"cmd": "move", "x": x, "y": y, "rotation": rotation, "size": size})
        return bool(reply and reply.get("ok"))

    async def get_position(self):
        reply = await self._command({"cmd": "get_position"})
        return reply if reply and reply.get("ok") else None

    async def set_monitor(self, index):
        reply = await self._command({"cmd": "monitor", "index": index})
        if reply is None:
            return False, "the avatar isn't running right now"
        return reply.get("ok", False), reply.get("error", "")

    async def set_layer(self, mode):
        reply = await self._command({"cmd": "layer", "mode": mode})
        return bool(reply and reply.get("ok"))

    async def set_sleeping(self, value):
        reply = await self._request({"cmd": "sleeping", "value": bool(value)})
        return bool(reply and reply.get("ok"))

    # -- lip-sync hot path (sync, fire-and-forget, never stalls the pipeline) -----

    def send_mouth(self, chunks):
        """chunks: [[raw_rms, duration_secs], ...] for one TTS frame."""
        if not self._writer:
            return
        try:
            self._writer.write(avatar_ipc.encode({"cmd": "mouth", "chunks": chunks}))
        except (OSError, RuntimeError):
            pass  # never reconnect here; the watchdog handles her body

    def send_speech(self, state):
        if not self._writer:
            return
        try:
            self._writer.write(avatar_ipc.encode({"cmd": "speech", "state": state}))
        except (OSError, RuntimeError):
            pass

    def send_prop(self, show, path=None, x=None, y=None, scale=None):
        """Show/hide a held prop (her book while reading). Fire-and-forget:
        degrades silently to no prop if the renderer is down. x/y/scale are
        fractions of the monitor for live position tuning."""
        if not self._writer:
            return
        try:
            msg = {"cmd": "prop", "show": bool(show)}
            for k, v in (("path", path), ("x", x), ("y", y), ("scale", scale)):
                if v is not None:
                    msg[k] = v
            self._writer.write(avatar_ipc.encode(msg))
        except (OSError, RuntimeError):
            pass

    # -- shutdown ---------------------------------------------------------------

    async def close(self):
        """Restart path: drop the socket only — her body stays up and the
        next Aiva process adopts it."""
        if self._reader_task:
            self._reader_task.cancel()
        self._drop()

    async def quit(self):
        """Quit path: ask the renderer to exit; taskkill fallback."""
        state = avatar_ipc.read_state()
        reply = await self._request({"cmd": "quit"}, timeout=3.0)
        await self.close()
        if reply is None and state and state.get("pid") and \
                avatar_ipc.pid_alive(state["pid"]):
            subprocess.run(f'taskkill /PID {state["pid"]} /F', shell=True,
                           capture_output=True)


# --- standalone probe -------------------------------------------------------------

async def _probe():
    client = AvatarClient()
    print("probe: ensure_running (adopt or spawn)...")
    ok = await client.ensure_running()
    print(f"probe: connected={client.connected}, expressions={client.expressions}")
    if not ok:
        return 1

    pos = await client.get_position()
    print(f"probe: position -> {pos}")

    ok, detail = await client.toggle_expression("爱心眼")
    print(f"probe: toggle 爱心眼 on -> {ok} ({detail})")
    await asyncio.sleep(1.0)
    ok, detail = await client.toggle_expression("爱心眼")
    print(f"probe: toggle 爱心眼 off -> {ok} ({detail})")

    ok, detail = await client.toggle_expression("nonsense-name")
    print(f"probe: bogus expression rejected -> ok={ok} ({detail})")

    print("probe: move to x=0.3, size=-20 ...")
    print(f"probe: move -> {await client.move(x=0.3, size=-20)}")
    await asyncio.sleep(0.5)
    print(f"probe: position now -> {await client.get_position()}")
    print(f"probe: move back -> {await client.move(x=0.0, size=0.0)}")

    print(f"probe: layer desktop -> {await client.set_layer('desktop')}")
    await asyncio.sleep(1.0)
    print(f"probe: layer top -> {await client.set_layer('top')}")

    print(f"probe: sleeping on -> {await client.set_sleeping(True)}")
    await asyncio.sleep(1.5)
    print(f"probe: sleeping off -> {await client.set_sleeping(False)}")

    client.send_speech("start")
    client.send_mouth(
        [[0.075 + 0.075 * __import__("math").sin(i / 3), 0.05] for i in range(40)])
    await asyncio.sleep(2.5)
    client.send_speech("stop")
    print("probe: mouth burst sent (watch her talk)")

    await client.close()
    print("probe: done (renderer left running)")
    return 0


if __name__ == "__main__":
    if "--probe" in sys.argv:
        sys.exit(asyncio.run(_probe()))
    print(__doc__)
