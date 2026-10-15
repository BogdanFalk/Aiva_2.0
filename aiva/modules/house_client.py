"""A sister's connection to the house.

Discovers the house through `aiva/house_state.json`, connects, says `hello`, and then runs a reader
loop forever: every decoded message goes to `on_message(msg)` (sync or async); `ping` is answered
here. When the house is down or drops her she logs "waiting for the house" (once a minute), keeps
retrying with 0.5 → 5 s backoff, and queues up to five `want_floor`s to replay after the next
`welcome` — everything else sent while offline is dropped (the house never replays either).

`send(msg)` is synchronous and fire-and-forget: it stamps the envelope, writes to the socket and
schedules the drain; a failed write closes the connection so the reader loop reconnects.
"""

import asyncio
import inspect
import time

from modules import house_proto as proto


class HouseClient:
    def __init__(self, hello: proto.Hello, *, on_message, log=print, state_path=None,
                 on_connected=None, on_disconnected=None, backoff=(0.5, 5.0), queue_limit=5,
                 handshake_timeout_s=5.0):
        self.hello = hello
        self._on_message = on_message
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self.log = log
        self.state_path = state_path
        self._backoff = backoff
        self._queue_limit = queue_limit
        self._handshake_timeout_s = handshake_timeout_s
        self._reader = None
        self._writer = None
        self._task = None
        self._drain_task = None
        self._connected = False
        self._stopping = False
        self._pending = []          # want_floor messages queued while offline
        self._msg_seq = 0
        self._last_wait_log = 0.0
        self.welcome = None
        self.house_state = None

    @property
    def connected(self) -> bool:
        return self._connected

    # --- lifecycle ------------------------------------------------------------------------------

    def start(self):
        if self._task is None:
            self._task = asyncio.get_event_loop().create_task(self._run())
        return self._task

    async def stop(self):
        self._stopping = True
        if self._connected:
            try:
                self.send(proto.Bye())
                await asyncio.sleep(0.05)
            except Exception:
                pass
        await self._close()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # --- sending --------------------------------------------------------------------------------

    def send(self, msg) -> bool:
        """Fire-and-forget. False when not connected (want_floor is queued for later)."""
        if not self._connected or self._writer is None:
            if isinstance(msg, proto.WantFloor):
                self._pending.append(msg)
                del self._pending[:-self._queue_limit]
            return False
        try:
            self._stamp(msg)
            self._writer.write(proto.encode(msg))
            if self._drain_task is None or self._drain_task.done():
                self._drain_task = asyncio.get_event_loop().create_task(self._drain())
            return True
        except Exception as e:
            self.log(f"house link: send failed ({e!r})")
            asyncio.get_event_loop().create_task(self._close())
            return False

    def _stamp(self, msg):
        self._msg_seq += 1
        msg.id = self._msg_seq
        msg.ts = time.time()
        return msg

    async def _drain(self):
        try:
            if self._writer is not None:
                await asyncio.wait_for(self._writer.drain(), 2.0)
        except Exception as e:
            self.log(f"house link: drain failed ({e!r})")
            await self._close()

    # --- the loop -------------------------------------------------------------------------------

    async def _run(self):
        delay = self._backoff[0]
        while not self._stopping:
            st = proto.read_state(self.state_path)
            if not st or not st.get("port"):
                self._note_waiting("no house_state.json")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._backoff[1])
                continue
            try:
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", int(st["port"])), 3.0)
                self._writer.write(proto.encode(self._stamp(self.hello)))
                await self._writer.drain()
                line = await asyncio.wait_for(self._reader.readline(), self._handshake_timeout_s)
                reply = proto.decode(line)
            except Exception as e:
                self._note_waiting(f"{e!r}")
                await self._close()
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._backoff[1])
                continue
            if isinstance(reply, proto.Reject):
                self.log(f"house link: rejected by the house: {reply.reason}")
                await self._close()
                await asyncio.sleep(self._backoff[1])
                continue
            if not isinstance(reply, proto.Welcome):
                self.log(f"house link: expected welcome, got {reply.T}")
                await self._close()
                await asyncio.sleep(delay)
                continue
            self.welcome = reply
            self.house_state = st
            self._connected = True
            delay = self._backoff[0]
            self.log(f"house link: connected (house pid={st.get('pid')}, port={st.get('port')}, "
                     f"mode={reply.mode}, present={reply.present})")
            await self._call(self._on_connected, reply)
            for m in self._pending:
                self.send(m)
            self._pending = []
            await self._read_loop()
            self._connected = False
            await self._close()
            await self._call(self._on_disconnected)
            if not self._stopping:
                self.log("house link: disconnected — reconnecting")
                await asyncio.sleep(delay)

    async def _read_loop(self):
        try:
            while not self._stopping:
                line = await self._reader.readline()
                if not line:
                    return
                try:
                    msg = proto.decode(line)
                except proto.ProtoError as e:
                    self.log(f"house link: bad line from the house: {e}")
                    continue
                if isinstance(msg, proto.Ping):
                    self.send(proto.Pong())
                    continue
                await self._call(self._on_message, msg)
        except (ConnectionError, asyncio.IncompleteReadError, OSError) as e:
            self.log(f"house link: connection lost ({e!r})")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.log(f"house link: reader error: {e!r}")

    async def _close(self):
        self._connected = False
        w, self._writer, self._reader = self._writer, None, None
        if w is not None:
            try:
                w.close()
            except Exception:
                pass

    async def _call(self, fn, *args):
        if fn is None:
            return
        try:
            res = fn(*args)
            if inspect.isawaitable(res):
                await res
        except Exception as e:
            self.log(f"house link: handler failed: {e!r}")

    def _note_waiting(self, why):
        now = time.time()
        if now - self._last_wait_log >= 60:
            self._last_wait_log = now
            self.log(f"waiting for the house ({why})")
            print("[house] waiting for the house to come up...")
