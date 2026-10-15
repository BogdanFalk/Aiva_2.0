"""AvatarClient: per-instance state file, spawn argv, and the never-adopt-the-
wrong-sister handshake (against an in-process fake renderer, no Qt)."""
import asyncio
import json
import os
import sys

import pytest

from modules import avatar as avatar_mod
from modules import avatar_ipc
from modules.avatar import AvatarClient, model_stem


def test_default_client_is_aiva_exactly_as_before():
    c = AvatarClient()
    assert c.instance == "aiva"
    assert c.state_file == avatar_ipc.STATE_FILE
    argv = c._spawn_argv(None)
    assert argv[:3] == [sys.executable, "-m", "modules.avatar_renderer"]
    assert argv[3:] == ["--instance", "aiva", "--state", avatar_ipc.STATE_FILE]
    assert "--model" not in argv and "--monitor" not in argv
    assert os.path.basename(c._log_path()).startswith("avatar-2")  # avatar-YYYYMMDD.log
    assert "-aiva-" not in os.path.basename(c._log_path())


def test_saga_client_owns_her_own_state_file_model_and_log():
    c = AvatarClient(instance="saga", model="assets/avatar/ariu/saga.model3.json", monitor=1)
    assert c.state_file == avatar_ipc.state_path("saga")
    assert c.state_file.endswith("saga_avatar_state.json")
    first = c._spawn_argv(None)  # first-ever spawn: no prior state -> monitor passed
    assert first[3:] == ["--instance", "saga", "--state", c.state_file,
                         "--model", "assets/avatar/ariu/saga.model3.json", "--monitor", "1"]
    again = c._spawn_argv({"boot_id": "old", "pid": 1})  # a body has run before
    assert "--monitor" not in again and "--model" in again
    assert os.path.basename(c._log_path()).startswith("avatar-saga-")


def test_explicit_state_file_wins():
    c = AvatarClient(instance="saga", state_file=r"C:\x\custom.json")
    assert c.state_file == r"C:\x\custom.json"
    assert c._spawn_argv(None)[3:] == ["--instance", "saga", "--state", r"C:\x\custom.json"]


def test_model_stem():
    assert model_stem("assets/avatar/ariu/saga.model3.json") == "saga"
    assert model_stem(r"C:\a\b\ariu.model3.json") == "ariu"
    assert model_stem(None) is None


async def _fake_renderer(model, version=1, seen=None, on_quit=None):
    async def handle(reader, writer):
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                msg = json.loads(line)
                if seen is not None:
                    seen.append(msg)
                if msg.get("cmd") == "hello":
                    writer.write(avatar_ipc.encode({
                        "id": msg["id"], "ok": True, "version": version, "instance": "whoever",
                        "model": model, "expressions": ["爱心眼"], "ready": True}))
                    await writer.drain()
                elif msg.get("cmd") == "quit":
                    writer.write(avatar_ipc.encode({"id": msg["id"], "ok": True}))
                    await writer.drain()
                    if on_quit:
                        on_quit()
                    break
        finally:
            writer.close()
    return await asyncio.start_server(handle, "127.0.0.1", 0)


def test_connect_refuses_a_renderer_running_another_model():
    async def run():
        srv = await _fake_renderer("ariu")
        port = srv.sockets[0].getsockname()[1]
        try:
            wrong = AvatarClient(instance="saga", model="x/saga.model3.json")
            assert await wrong._connect({"port": port}) is False
            assert wrong.connected is False and wrong.expressions == []

            right = AvatarClient(instance="saga", model="x/ariu.model3.json")
            assert await right._connect({"port": port}) is True
            assert right.connected and right.renderer_model == "ariu"
            assert right.expressions == ["爱心眼"]
            await right.close()

            anybody = AvatarClient()  # no model asked for -> adopts what's there (today's behaviour)
            assert await anybody._connect({"port": port}) is True
            assert anybody.renderer_model == "ariu"
            await anybody.close()
        finally:
            srv.close()
            await srv.wait_closed()
    asyncio.run(run())


def _adopt(tmp_path, monkeypatch, version):
    """ensure_running() against a live fake renderer speaking `version`; never really spawns,
    never really kills anything. Returns (adopted?, quits, spawned states, messages seen)."""
    async def run():
        quits, seen, spawned = [], [], []
        alive = {"v": True}

        def quit_():
            quits.append(1)
            alive["v"] = False

        srv = await _fake_renderer("ariu", version=version, seen=seen, on_quit=quit_)
        port = srv.sockets[0].getsockname()[1]
        state_file = tmp_path / "avatar_state.json"
        avatar_ipc.write_state_atomic({"port": port, "pid": 4242, "boot_id": "old",
                                       "placement": {"x": 0.8}}, str(state_file))
        monkeypatch.setattr(avatar_ipc, "pid_alive", lambda pid: alive["v"])
        monkeypatch.setattr(avatar_mod.subprocess, "run",
                            lambda *a, **k: pytest.fail("taskkill must not be needed"))

        async def fake_spawn(self, old_state):
            spawned.append(old_state)
            return True
        monkeypatch.setattr(AvatarClient, "_spawn", fake_spawn)
        c = AvatarClient(state_file=str(state_file))
        try:
            ok = await c.ensure_running()
            await c.close()
        finally:
            srv.close()
            await srv.wait_closed()
        return ok, quits, spawned, seen
    return asyncio.run(run())


def test_a_current_renderer_is_adopted_as_is(tmp_path, monkeypatch):
    ok, quits, spawned, _ = _adopt(tmp_path, monkeypatch, avatar_ipc.PROTOCOL_VERSION)
    assert ok and not quits and not spawned


def test_an_outdated_renderer_is_retired_and_respawned_with_its_placement(tmp_path, monkeypatch):
    """Her body outlives her restarts: without this an update to the renderer never reaches it."""
    ok, quits, spawned, seen = _adopt(tmp_path, monkeypatch, avatar_ipc.PROTOCOL_VERSION - 1)
    assert ok and quits == [1]
    assert [m["cmd"] for m in seen] == ["hello", "quit"]
    assert spawned and spawned[0]["boot_id"] == "old"          # _spawn waits for a NEW boot id
    assert spawned[0]["placement"] == {"x": 0.8}                # where she stood carries over


def test_send_activity_and_cue_are_fire_and_forget():
    class W:
        def __init__(self):
            self.out = []

        def write(self, b):
            self.out.append(json.loads(b))

    c = AvatarClient()
    c.send_activity({"title": "Alice"}, None)                     # not connected: silently nothing
    c.send_cue("page")
    c._writer = W()
    c.send_activity({"title": "Alice"}, {"label": "x", "state": "running"})
    c.send_activity()
    c.send_cue("done")
    assert c._writer.out == [
        {"cmd": "activity", "reading": {"title": "Alice"}, "working": {"label": "x", "state": "running"}},
        {"cmd": "activity", "reading": None, "working": None},
        {"cmd": "cue", "kind": "done"},
    ]
