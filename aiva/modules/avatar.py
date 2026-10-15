"""A sister's handle on her own body — the client for the avatar renderer process.

Plays the role modules/vtube_studio.py used to: constructed in main(), passed
to register_tools, and every failure degrades to voice-only instead of taking
her down. The renderer (modules/avatar_renderer.py) is a separate process so
a GL crash can't kill her voice, and so her body survives her self-restarts:
ensure_running() first tries to ADOPT a live renderer via her state file
(pid + hello handshake) and only spawns a new one if there isn't one.

One client = one instance = one state file. `AvatarClient()` is Aiva exactly
as before (aiva/avatar_state.json, ariu). `AvatarClient(instance="saga",
model=".../saga.model3.json")` owns aiva/saga_avatar_state.json, spawns the
renderer with `--instance saga --state <file> --model <model>`, and refuses
to adopt a renderer whose `hello` reports a different model — a sister never
borrows the wrong body.

Two traffic classes on one socket:
- request/reply (expressions, moves, position reads): async, id-matched.
- the lip-sync hot path (send_mouth/send_speech/send_witness): sync fire-and-forget —
  never awaits, never reconnects, swallows every error. The voice pipeline
  must not hiccup because her body blinked.

Standalone probe (never starts Aiva):
  cd aiva && python -m modules.avatar --probe [--instance saga --model PATH]
Runs a scripted command sequence against a running renderer (adopting it, or
spawning one if none is alive).
"""

import argparse
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


def model_stem(model_path):
    """'.../saga.model3.json' -> 'saga' — the renderer's `model_name` rule."""
    return os.path.basename(model_path).split(".")[0] if model_path else None


class AvatarClient:
    def __init__(self, instance="aiva", model=None, monitor=None, state_file=None):
        self.instance = instance or "aiva"
        self.model = model                    # model3 path to spawn with (None = renderer default)
        self.monitor = monitor                # monitor index for the very first spawn only
        self._state_file = state_file or avatar_ipc.state_path(self.instance)
        self._want_model = model_stem(model)  # adopt only a body running THIS model
        self.renderer_model = None            # what the adopted renderer reported
        self.connected = False
        self.expressions = []       # names from hello, for the system prompt
        self.witness = False        # her renderer draws a constellation witness (hello says so)
        self.renderer_version = None  # the adopted renderer's protocol version (hello)
        self._reader = None
        self._writer = None
        self._reader_task = None
        self._pending = {}          # id -> Future
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock() # serializes connect/spawn attempts

    @property
    def state_file(self):
        return self._state_file

    def _say(self, msg):
        print(f"Avatar[{self.instance}]: {msg}")

    # -- lifecycle -------------------------------------------------------------

    async def ensure_running(self) -> bool:
        """Adopt a live renderer, or spawn one. Non-fatal on failure."""
        async with self._lock:
            if self.connected:
                return True
            state = avatar_ipc.read_state(self._state_file)
            if state and state.get("pid") and avatar_ipc.pid_alive(state["pid"]):
                if await self._connect(state):
                    if (self.renderer_version or 1) >= avatar_ipc.PROTOCOL_VERSION:
                        self._say(f"adopted running renderer (pid {state['pid']})")
                        return True
                    # her body outlives her restarts, so an update to the renderer only reaches
                    # it if the old one is retired; its placement is in the state file it leaves
                    self._say(f"renderer pid {state['pid']} speaks protocol v{self.renderer_version or 1}, "
                              f"this is v{avatar_ipc.PROTOCOL_VERSION} — replacing it")
                    state = await self._retire(state)
            return await self._spawn(state)

    async def _retire(self, state):
        """Ask an outdated renderer to quit (taskkill if it won't), wait for it to go, and
        return the state file it left behind (placement/monitor/layer carry over)."""
        pid = state.get("pid")
        reply = await self._request({"cmd": "quit"}, timeout=3.0)
        self._drop()
        deadline = time.monotonic() + 5
        while pid and avatar_ipc.pid_alive(pid) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if pid and avatar_ipc.pid_alive(pid):
            subprocess.run(f"taskkill /PID {int(pid)} /F", shell=True, capture_output=True)
            if reply is not None:
                self._say(f"old renderer pid {pid} ignored quit — killed it")
        return avatar_ipc.read_state(self._state_file) or state

    def _spawn_argv(self, old_state):
        """The renderer command line. Pure so the probe/tests can inspect it."""
        argv = [sys.executable, "-m", "modules.avatar_renderer",
                "--instance", self.instance, "--state", self._state_file]
        if self.model:
            argv += ["--model", self.model]
        # placement/monitor live in the state file once a body has run; only a
        # first-ever spawn (no prior state at all) takes the configured monitor
        if self.monitor is not None and old_state is None:
            argv += ["--monitor", str(int(self.monitor))]
        return argv

    def _log_path(self):
        day = f"{datetime.now():%Y%m%d}"
        name = f"avatar-{day}.log" if self.instance == "aiva" else f"avatar-{self.instance}-{day}.log"
        return os.path.join(_LOG_DIR, name)

    async def _spawn(self, old_state) -> bool:
        old_boot = (old_state or {}).get("boot_id")
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
            log = open(self._log_path(), "a", encoding="utf-8")
            aiva_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            subprocess.Popen(
                self._spawn_argv(old_state),
                cwd=aiva_dir, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW, close_fds=False,
            )
        except OSError as e:
            self._say(f"couldn't spawn renderer ({e}) — going voice-only")
            return False
        # wait for the NEW instance's state file (boot_id changes), then connect;
        # first boot loads 14 textures, so allow a generous window
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            await asyncio.sleep(0.25)
            state = avatar_ipc.read_state(self._state_file)
            if state and state.get("boot_id") and state["boot_id"] != old_boot:
                if await self._connect(state):
                    self._say(f"renderer up (pid {state.get('pid')})")
                    return True
        self._say("renderer didn't come up in 20s — going voice-only")
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
        if self._want_model and hello.get("model") != self._want_model:
            # never adopt the wrong sister: a body advertising in our state
            # file but rendering another model is a misconfiguration, not ours
            self._say(f"renderer on port {state.get('port')} runs model "
                      f"{hello.get('model')!r}, not {self._want_model!r} — not adopting it")
            self._drop()
            return False
        self.renderer_model = hello.get("model")
        self.renderer_version = hello.get("version")
        self.expressions = hello.get("expressions", [])
        self.witness = bool(hello.get("witness"))  # an older renderer simply doesn't say
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

    def send_activity(self, reading=None, working=None):
        """What her body is doing — the WHOLE state each time (idempotent, so a resend after a
        renderer restart heals it): reading {title} while she reads (she pulls up a book),
        working {label, detail, step, total, state} while she's on a task (her work panel).
        None = not doing it. Fire-and-forget; degrades silently to no props."""
        if not self._writer:
            return
        try:
            self._writer.write(avatar_ipc.encode({"cmd": "activity", "reading": reading,
                                                  "working": working}))
        except (OSError, RuntimeError):
            pass

    def send_cue(self, kind):
        """A one-shot beat for her body: 'page' (a section read — she turns the page),
        'done' / 'failed' (a task ended — her panel shows a check or a cross). Fire-and-forget."""
        if not self._writer:
            return
        try:
            self._writer.write(avatar_ipc.encode({"cmd": "cue", "kind": kind}))
        except (OSError, RuntimeError):
            pass

    def send_witness(self, kind):
        """One lived moment for her constellation witness (said|heard|sister|read).
        Carries only the KIND, never text. Fire-and-forget; a no-op unless her
        renderer said in `hello` that it has a witness (Aiva's never does)."""
        if not self._writer or not self.witness:
            return
        try:
            self._writer.write(avatar_ipc.encode({"cmd": "witness", "kind": kind}))
        except (OSError, RuntimeError):
            pass

    # -- shutdown ---------------------------------------------------------------

    async def close(self):
        """Restart path: drop the socket only — her body stays up and the
        next process for this instance adopts it."""
        if self._reader_task:
            self._reader_task.cancel()
        self._drop()

    async def quit(self):
        """Quit path: ask the renderer to exit; taskkill fallback."""
        state = avatar_ipc.read_state(self._state_file)
        reply = await self._request({"cmd": "quit"}, timeout=3.0)
        await self.close()
        if reply is None and state and state.get("pid") and \
                avatar_ipc.pid_alive(state["pid"]):
            subprocess.run(f'taskkill /PID {state["pid"]} /F', shell=True,
                           capture_output=True)


# --- standalone probe -------------------------------------------------------------

async def _probe(instance="aiva", model=None):
    client = AvatarClient(instance=instance, model=model)
    print(f"probe[{instance}]: state file {client.state_file}")
    print("probe: ensure_running (adopt or spawn)...")
    ok = await client.ensure_running()
    print(f"probe: connected={client.connected}, model={client.renderer_model}, "
          f"expressions={client.expressions}")
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
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--instance", default="aiva")
    ap.add_argument("--model", default=None)
    opts, _rest = ap.parse_known_args()
    if opts.probe:
        sys.exit(asyncio.run(_probe(opts.instance, opts.model)))
    print(__doc__)
