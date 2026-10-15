"""The house's socket server: one `ClientSession` per connected sister, dispatching her reports into
the `Floor` and carrying the Floor's messages back. Loopback TCP, newline JSON (`house_proto`).

The Floor knows nothing about sockets — it calls `session.send(msg)` synchronously, so `send` only
writes to the StreamWriter and schedules the drain. Everything the server originates itself
(`welcome`, `state`, `ping`, `shutdown`, relayed `event`s) is stamped here.

Presence rules: a second `hello` for a slug that is already connected pings the existing session
and, if it answers within a second, rejects the newcomer (two Aivas must never run); otherwise the
dead one is replaced. A session that stops answering pings is dropped and the Floor is told
(`on_client_lost`) so a held turn is released. `house_state.json` (`{port, pid, boot_id,
protocol}`) is written when the server is up and removed when it stops.
"""

import asyncio
import inspect
import os
import time
import uuid

from modules import house_proto as proto
from modules.house_floor import ClientInfo
from modules.sisters import NameSet


class ClientSession:
    def __init__(self, server, reader, writer):
        self.server = server
        self.reader = reader
        self.writer = writer
        self.slug = None
        self.info = None
        self.pid = 0
        self.version = ""
        self.last_pong = time.time()
        self.connected_at = time.time()
        self._closed = False
        self._drain_task = None

    @property
    def closed(self) -> bool:
        return self._closed

    def send(self, msg) -> None:
        """Synchronous write (the Floor calls this). The drain runs in the background; a failed
        socket closes the session on its next drain or read."""
        if self._closed:
            raise ConnectionError(f"session {self.slug or '?'} is closed")
        if not msg.id:
            self.server._stamp(msg)
        self.writer.write(proto.encode(msg))
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.get_event_loop().create_task(self._drain())

    async def _drain(self):
        try:
            await asyncio.wait_for(self.writer.drain(), 2.0)
        except Exception as e:
            self.server.log(f"house: drain to {self.slug or '?'} failed ({e!r}); closing")
            await self.close()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.writer.close()
        except Exception:
            pass

    async def run(self):
        try:
            try:
                line = await asyncio.wait_for(self.reader.readline(), 5.0)
                if not line.strip():
                    return          # a liveness probe (the launcher) connected and left — not a sister
                hello = proto.decode(line)
            except (asyncio.TimeoutError, proto.ProtoError) as e:
                self.server.log(f"house: bad handshake: {e!r}")
                return
            if not isinstance(hello, proto.Hello):
                self.server.log(f"house: expected hello, got {hello.T}")
                return
            if not await self.server._register(self, hello):
                return
            while not self._closed:
                line = await self.reader.readline()
                if not line:
                    break
                try:
                    msg = proto.decode(line)
                except proto.ProtoError as e:
                    self.server.log(f"house: bad line from {self.slug}: {e}")
                    continue
                await self.server._dispatch(self, msg)
        except (ConnectionError, asyncio.IncompleteReadError, OSError) as e:
            self.server.log(f"house: connection to {self.slug or '?'} dropped ({e!r})")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.server.log(f"house: session {self.slug or '?'} error: {e!r}")
        finally:
            await self.close()
            await self.server._unregister(self)


class HouseServer:
    def __init__(self, floor, memory, *, host="127.0.0.1", port=0, log=print, state_path=None,
                 mic_state=None, on_sleep=None, on_wake=None, on_announce=None,
                 on_presence_change=None, on_restart=None, tick_s=0.1, ping_s=15.0,
                 pong_timeout_s=10.0, version=""):
        self.floor = floor
        self.memory = memory
        self.host = host
        self.port = port
        self.log = log
        self.state_path = state_path
        self.mic_state = mic_state or (lambda: {"awake": True, "mic_muted": False})
        self.on_sleep = on_sleep
        self.on_wake = on_wake
        self.on_announce = on_announce
        self.on_presence_change = on_presence_change
        self.on_restart = on_restart          # (slug, target) — a sister asked the house to restart
        self.on_control = None                # (action, target) — a dashboard button (set by the house)
        self.on_say = None                    # (text) — a line typed in the dashboard's chat box
        self.controls = set()                 # role=control sessions: buttons, never voices
        self.tick_s = tick_s
        self.ping_s = ping_s
        self.pong_timeout_s = pong_timeout_s
        self.version = version
        self.boot_id = uuid.uuid4().hex
        self.sessions = {}          # slug -> ClientSession
        self._pending = set()       # sessions before hello
        self._server = None
        self._tasks = []
        self._msg_seq = 0

    # --- lifecycle ------------------------------------------------------------------------------

    async def start(self):
        self._server = await asyncio.start_server(self._on_connection, self.host, self.port)
        self.port = self._server.sockets[0].getsockname()[1]
        proto.write_state_atomic({"port": self.port, "pid": os.getpid(), "boot_id": self.boot_id,
                                  "protocol": proto.PROTOCOL, "ts": time.time()}, self.state_path)
        self._tasks = [asyncio.create_task(self._tick_loop()), asyncio.create_task(self._ping_loop())]
        self.log(f"house: listening on {self.host}:{self.port}")
        return self.port

    async def stop(self, reason=None):
        for t in self._tasks:
            t.cancel()
        self._tasks = []
        for s in list(self.sessions.values()) + list(self._pending):
            try:
                s.send(proto.Shutdown(reason=reason))
            except Exception:
                pass
        await asyncio.sleep(0.05)
        for s in list(self.sessions.values()) + list(self._pending):
            await s.close()
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), 2.0)
            except Exception:
                pass
        proto.clear_state(self.state_path)

    async def _on_connection(self, reader, writer):
        session = ClientSession(self, reader, writer)
        self._pending.add(session)
        try:
            await session.run()
        finally:
            self._pending.discard(session)

    # --- plumbing -----------------------------------------------------------------------------

    def _stamp(self, msg):
        self._msg_seq += 1
        msg.id = self._msg_seq
        msg.ts = time.time()
        return msg

    @property
    def present(self) -> list:
        return sorted(self.sessions.keys())

    def state_msg(self) -> proto.State:
        st = {}
        try:
            st = dict(self.mic_state() or {})
        except Exception:
            pass
        return proto.State(awake=bool(st.get("awake", True)), mic_muted=bool(st.get("mic_muted", False)),
                           present=self.present)

    def broadcast(self, msg, exclude=None):
        for slug, s in list(self.sessions.items()):
            if slug == exclude:
                continue
            try:
                s.send(msg)
            except Exception as e:
                self.log(f"house: broadcast to {slug} failed: {e!r}")

    def broadcast_state(self):
        for s in list(self.sessions.values()) + list(self.controls):
            try:
                s.send(self.state_msg())
            except Exception:
                pass

    def send_to(self, slug, msg) -> bool:
        """Send to one connected sister; False when she isn't connected."""
        s = self.sessions.get(slug)
        if s is None or s.closed:
            return False
        try:
            s.send(msg)
            return True
        except Exception as e:
            self.log(f"house: send to {slug} failed: {e!r}")
            return False

    async def _call(self, fn, *args):
        if fn is None:
            return
        try:
            res = fn(*args)
            if inspect.isawaitable(res):
                await res
        except Exception as e:
            self.log(f"house: callback {getattr(fn, '__name__', fn)!r} failed: {e!r}")

    # --- presence -----------------------------------------------------------------------------

    async def _register(self, session, hello: proto.Hello) -> bool:
        slug = (hello.slug or "").strip().lower()
        if not slug:
            session.send(proto.Reject(reason="no slug"))
            return False
        if hello.protocol != proto.PROTOCOL:
            session.send(proto.Reject(reason=f"protocol {hello.protocol} != {proto.PROTOCOL}"))
            self.log(f"house: rejected {slug}: protocol {hello.protocol}")
            return False
        if (hello.role or "sister") == "control":
            # a control client (the dashboard's buttons): not a sister — no presence, no floor
            session.slug = f"control:{slug}"
            session.info = ClientInfo(slug=session.slug, name=hello.name or "control", has_voice=False)
            session.last_pong = time.time()
            self.controls.add(session)
            st = self.state_msg()
            session.send(proto.Welcome(mode=self.floor.mode, state={"awake": st.awake, "mic_muted": st.mic_muted},
                                       present=self.present))
            return True
        old = self.sessions.get(slug)
        if old is not None and not old.closed:
            # is the existing one alive? ping it and give it a second
            t0 = time.time()
            try:
                old.send(proto.Ping())
            except Exception:
                pass
            deadline = t0 + 1.0
            while time.time() < deadline and old.last_pong < t0 and not old.closed:
                await asyncio.sleep(0.05)
            if old.last_pong >= t0 and not old.closed:
                session.send(proto.Reject(reason=f"{slug} is already connected"))
                self.log(f"house: rejected a second {slug} (the first is alive)")
                return False
            self.log(f"house: replacing a dead {slug} session")
            await old.close()
            self.sessions.pop(slug, None)
        nameset = None
        if hello.name_variants:
            nameset = NameSet(slug, hello.name, tuple(v.lower() for v in hello.name_variants),
                              hello.loose_regex or "")
        session.slug = slug
        session.info = ClientInfo(slug=slug, name=hello.name or slug.capitalize(), blurb=hello.blurb or "",
                                  has_voice=bool(hello.has_voice), nameset=nameset)
        session.pid = hello.pid
        session.version = hello.version
        session.last_pong = time.time()
        self.sessions[slug] = session
        self.floor.add_client(slug, session)
        st = self.state_msg()
        session.send(proto.Welcome(mode=self.floor.mode, state={"awake": st.awake, "mic_muted": st.mic_muted},
                                   present=self.present))
        self.log(f"house: {session.info.name} joined (pid={hello.pid}, v{hello.version}, "
                 f"voice={'yes' if hello.has_voice else 'no'})")
        self.memory.add_event(slug, "joined", f"{session.info.name} came online")
        self.broadcast_state()
        await self._call(self.on_presence_change, self.present)
        return True

    async def _unregister(self, session):
        if session in self.controls:
            self.controls.discard(session)
            return
        slug = session.slug
        if slug is None or self.sessions.get(slug) is not session:
            return
        self.sessions.pop(slug, None)
        self.log(f"house: {session.info.name if session.info else slug} left")
        try:
            self.floor.on_client_lost(slug)
        except Exception as e:
            self.log(f"house: floor.on_client_lost failed: {e!r}")
        self.memory.add_event(slug, "left", f"{session.info.name if session.info else slug} went offline")
        self.broadcast_state()
        await self._call(self.on_presence_change, self.present)

    # --- dispatch -----------------------------------------------------------------------------

    async def _dispatch(self, session, msg):
        slug = session.slug
        f = self.floor
        if session in self.controls:
            if isinstance(msg, proto.Control):
                self.log(f"house: control '{msg.action}'" + (f" -> {msg.target}" if msg.target else "")
                         + f" from {session.info.name}")
                await self._call(self.on_control, msg.action, msg.target)
            elif isinstance(msg, proto.Say):
                text = (msg.text or "").strip()
                self.log(f"house: typed {text[:80]!r} from {session.info.name}")
                if text:
                    await self._call(self.on_say, text)
            elif isinstance(msg, proto.Bye):
                await session.close()
            elif not isinstance(msg, proto.Pong):
                self.log(f"house: ignoring {msg.T} from control client {session.info.name}")
            return
        if isinstance(msg, proto.Pong):
            session.last_pong = time.time()
        elif isinstance(msg, proto.Response):
            f.on_response(slug, msg.turn, msg.phase)
        elif isinstance(msg, proto.Tool):
            f.on_tool(slug, msg.turn, msg.delta)
        elif isinstance(msg, proto.Speaking):
            f.on_speaking(slug, msg.turn, msg.on)
        elif isinstance(msg, proto.Said):
            f.on_said(slug, msg.turn, msg.text, msg.addressee, msg.private, msg.interrupted)
        elif isinstance(msg, proto.Event):
            others = [s for s in self.sessions if s != slug]
            eid = self.memory.add_event(slug, msg.kind, msg.summary, msg.capability,
                                        relayed_to=",".join(others) or None)
            relay = proto.Event(kind=msg.kind, summary=msg.summary, capability=msg.capability,
                                payload=msg.payload, actor=slug, event_id=eid)
            self.broadcast(relay, exclude=slug)
        elif isinstance(msg, proto.WantFloor):
            if msg.kind == "announce":
                await self._call(self.on_announce, slug)
            f.want_floor(slug, msg.kind, msg.to)
        elif isinstance(msg, proto.Mode):
            f.set_private(msg.sister_talk == "private")
        elif isinstance(msg, proto.Sleep):
            await self._call(self.on_sleep, slug)
        elif isinstance(msg, proto.Wake):
            await self._call(self.on_wake, slug)
        elif isinstance(msg, proto.Restart):
            self.log(f"house: {slug} asked for a restart of the {msg.target}")
            await self._call(self.on_restart, slug, msg.target)
        elif isinstance(msg, proto.Bye):
            await session.close()
        elif isinstance(msg, proto.Hello):
            self.log(f"house: duplicate hello from {slug} ignored")
        else:
            self.log(f"house: unexpected {msg.T} from {slug}")

    # --- loops ---------------------------------------------------------------------------------

    async def _tick_loop(self):
        while True:
            await asyncio.sleep(self.tick_s)
            try:
                self.floor.tick()
            except Exception as e:
                self.log(f"house: floor.tick failed: {e!r}")

    async def _ping_loop(self):
        while True:
            await asyncio.sleep(self.ping_s)
            now = time.time()
            for slug, s in list(self.sessions.items()):
                if now - s.last_pong > self.ping_s + self.pong_timeout_s:
                    self.log(f"house: {slug} stopped answering pings; dropping")
                    await s.close()
                    continue
                try:
                    s.send(proto.Ping())
                except Exception:
                    await s.close()
