"""The house — the one ear and the floor the sisters share.

    python house.py [--list-devices]

Owns: the microphone (LocalAudioTransport input only), MicController (self-hearing gate, F9 deafen,
ambient sleep), Deepgram STT, the two-stage wake word (openWakeWord models for every sister → a
verified transcript must contain a name), the ONE turn detector in the stage aggregator (AIVA_STT:
`flux` = Deepgram Flux's own end-of-turn, no local VAD; `nova` = nova-3 + Silero VAD + smart-turn), the Floor (who speaks, who overhears, private sister talk, caps), the socket
server the sisters connect to, global hotkeys, the conversation log in aiva/house.db, and the
read-only dashboard. It computes the speaker voiceprint and vocal affect ONCE per user turn and
forwards them so each sister matches against her own roster.

It has no LLM of its own except the arbiter side-call (who did he mean, when no name was said)
and never speaks. Sisters come and go; the house stays up. Exit code 42 = restart (start_house.bat).

Hotkeys: F9 mute/deafen, F10 interrupt whoever is speaking, AIVA_HOTKEY_SISTER_TALK (default
Ctrl+Alt+P) flip sister talk private/audible, Ctrl+Alt+R restart the sisters, Ctrl+Alt+X quit all.
"""

import asyncio
import json
import os
import sqlite3
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv(override=True)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _install_log_throttle():
    """pipecat logs "Empty audio frame received for STT service" on EVERY zero/empty audio frame —
    a flood while awake. Throttle just that message to once per 10 s."""
    import time as _t
    from loguru import logger as _logger
    _state = {"last": 0.0, "dropped": 0}

    def _filter(record):
        if "Empty audio frame received" in record["message"]:
            now = _t.monotonic()
            if now - _state["last"] < 10:
                _state["dropped"] += 1
                return False
            if _state["dropped"]:
                record["message"] += f"  (+{_state['dropped']} more in the last 10s)"
                _state["dropped"] = 0
            _state["last"] = now
        return True

    try:
        _logger.remove()
    except ValueError:
        pass
    _logger.add(sys.stderr, filter=_filter, level="DEBUG")


_install_log_throttle()

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

from modules import character, house_proto as proto, metrics as _metrics
from modules.ear import ActivityWatcher, MicController, UserAudioTap, device_index, list_audio_devices, native_rate
from modules.eventlog import EventLog
from modules.house_floor import Floor
from modules.house_memory import HouseMemory
from modules.house_server import HouseServer
from modules.house_stage import FloorUserMuteStrategy, StageRouter
from modules.house_stt import STTConfig, build_stt, stage_params
from modules.housedb import house_db_path, migrate_limits_from, open_house_db
from modules.limits import ServiceLimits
from modules.metrics import Metrics
from modules.notify import Notifier, NotifyConfig
from modules.paralinguistics import AffectConfig, VocalAffect
from modules.sisters import ExchangeConfig, namesets_from_env, which_names
from modules.speakers import SpeakerConfig, SpeakerMemory, SpeakerRecognizer
from modules.tasks import openai_llm_call
from modules.wake import WakeWordListener, model_name

log_event = EventLog("house")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _require_env(*names):
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        print("Missing required .env keys: " + ", ".join(missing))
        if "DEEPGRAM_API_KEY" in missing:
            print("  -> Get a free Deepgram key (comes with $200 credit) at https://console.deepgram.com/signup")
        sys.exit(1)


def _launch_dashboard():
    """Open the read-only dashboard in the background (separate process). Skips if one is already
    listening on the port — so a restart doesn't spawn a duplicate or pop a new browser tab.
    Toggles: AIVA_DASHBOARD_AUTOSTART=0 to disable, AIVA_DASHBOARD_OPEN=0 for no browser tab."""
    if os.getenv("AIVA_DASHBOARD_AUTOSTART", "1") != "1":
        return
    import socket
    import subprocess

    port = int(os.getenv("AIVA_DASHBOARD_PORT", "8760"))
    try:
        with socket.socket() as s:
            s.settimeout(0.3)
            running = s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        running = False
    if running:
        # a dashboard is up — but is it running the code on disk? It is detached from us, so an old
        # one would otherwise serve stale pages forever. Ask it, and replace it if it's behind.
        stale_pid = None
        try:
            import json as _json
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1.0) as r:
                info = _json.loads(r.read().decode("utf-8"))
            on_disk = round(os.path.getmtime(os.path.join(_HERE, "dashboard.py")), 3)
            if abs(float(info.get("code_mtime", 0)) - on_disk) > 0.01:
                stale_pid = int(info.get("pid") or 0) or None
        except Exception:
            stale_pid = -1        # too old to even answer /api/health
        if stale_pid is None:
            print(f"Dashboard already running at http://127.0.0.1:{port}")
            return
        print("Dashboard is running old code — replacing it")
        log_event(f"dashboard stale (pid={stale_pid}); replacing")
        try:
            if stale_pid and stale_pid > 0:
                import subprocess as _sp
                _sp.run(["taskkill", "/PID", str(stale_pid), "/F"], capture_output=True, timeout=5)
            else:
                # no pid from the old one: find the listener on the port
                import subprocess as _sp
                out = _sp.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True, timeout=5).stdout
                for line in out.splitlines():
                    if f":{port}" in line and "LISTENING" in line:
                        _sp.run(["taskkill", "/PID", line.split()[-1], "/F"], capture_output=True, timeout=5)
            for _ in range(30):
                with socket.socket() as s:
                    s.settimeout(0.2)
                    if s.connect_ex(("127.0.0.1", port)) != 0:
                        break
                time.sleep(0.1)
        except Exception as e:
            print(f"Dashboard: couldn't replace the old one ({e}); it stays as is")
            return
    args = [sys.executable, "dashboard.py"]
    if os.getenv("AIVA_DASHBOARD_OPEN", "1") == "1":
        args.append("--open")
    try:
        os.makedirs(os.path.join(_HERE, "logs"), exist_ok=True)
        log = open(os.path.join(_HERE, "logs", f"dashboard-{datetime.now():%Y%m%d}.log"), "a",
                   encoding="utf-8")
        subprocess.Popen(
            args, cwd=_HERE, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=False)
        print(f"Dashboard launched at http://127.0.0.1:{port}")
    except OSError as e:
        print(f"Dashboard: couldn't launch ({e})")


def _wake_models():
    """Every sister's wake model that exists on disk (Aiva's is required to arm ambient mode)."""
    models = []
    for slug in character.KNOWN:
        try:
            ch = character.Character.from_env(slug)
        except Exception:
            continue
        m = ch.wake_model
        if not m:
            continue
        if os.path.isfile(m) or not m.endswith(".onnx"):    # a bare openWakeWord name is fine too
            if m not in models:
                models.append(m)
    return models


async def main():
    _require_env("DEEPGRAM_API_KEY")
    from version import __version__ as _version

    ambient = os.getenv("AIVA_WAKE_WORD", "0") == "1"
    idle_timeout = float(os.getenv("AIVA_IDLE_TIMEOUT", "20"))
    print(f"The house v{_version}")
    log_event(f"--- boot v{_version} (pid={os.getpid()}, ambient={ambient}, idle_timeout={idle_timeout}s) ---")

    # --- house memory, caps, metrics --------------------------------------------------------------
    house_conn = open_house_db(house_db_path())
    memory = HouseMemory(house_conn)
    notifier = Notifier(NotifyConfig.from_env(), log=log_event)
    service_limits = ServiceLimits(house_conn, notifier=notifier, log=log_event)
    # one-shot: carry Aiva's mid-month spend counters over from her old db
    try:
        aiva_db = character.Character.from_env("aiva").db_path
        if os.path.isfile(aiva_db):
            src = sqlite3.connect(aiva_db)
            try:
                moved = migrate_limits_from(house_conn, src)
                if moved:
                    log_event(f"limits: copied {moved} counter row(s) from {os.path.basename(aiva_db)}")
            finally:
                src.close()
    except Exception as e:
        log_event(f"limits migration skipped: {e!r}")
    _metrics.set_recorder(Metrics(house_conn))
    memory.prune()
    _launch_dashboard()

    # --- the arbiter (only when nobody was named and both sisters are present) -------------------
    from openai import AsyncOpenAI
    arbiter = None
    if os.getenv("OPENAI_API_KEY"):
        oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                          max_retries=int(os.getenv("AIVA_LLM_MAX_RETRIES", "5")))
        task_model = os.getenv("AIVA_TASK_MODEL", os.getenv("AIVA_LLM_MODEL", "gpt-4.1-mini"))
        _arbiter_call = openai_llm_call(oai, task_model, label="arbiter")

        async def arbiter(messages):
            return await _arbiter_call(messages, json_mode=True, max_tokens=60, temperature=0.0)

    # --- the ear ----------------------------------------------------------------------------------
    mic_index = device_index("AIVA_INPUT_DEVICE_INDEX", "AIVA_INPUT_DEVICE_NAME", False)
    mic_rate = native_rate(mic_index)
    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=True, audio_out_enabled=False,
        audio_in_sample_rate=mic_rate, input_device_index=mic_index))

    namesets = namesets_from_env()
    stt_keyterms = list(character.SHARED_STT_KEYTERMS) if hasattr(character, "SHARED_STT_KEYTERMS") else []
    for ns in namesets:
        if ns.name not in stt_keyterms:
            stt_keyterms.append(ns.name)
    stt_keyterms += [k.strip() for k in os.getenv("AIVA_STT_KEYTERMS", "").split(",") if k.strip()]
    stt_cfg = STTConfig.from_env()
    if stt_cfg.note:
        print(f"WARNING: {stt_cfg.note}")
    print(f"STT: {stt_cfg.describe()}")
    log_event(f"stt: {stt_cfg.describe()}")
    stt = build_stt(stt_cfg, api_key=os.getenv("DEEPGRAM_API_KEY"), keyterms=stt_keyterms, log=log_event)

    # --- the floor ---------------------------------------------------------------------------------
    floor = Floor(memory, private=os.getenv("AIVA_SISTER_TALK", "private").strip().lower() != "audible",
                  log=log_event, arbiter=arbiter, cfg=ExchangeConfig.from_env(), namesets=namesets)
    mic = MicController(ambient=ambient, verify_secs=float(os.getenv("AIVA_WAKE_VERIFY_SECS", "5")),
                        busy=lambda: floor.busy, log=log_event, label="the house")
    floor.on_gate_change = mic.set_speaking
    speaker_cfg = SpeakerConfig.from_env()
    tap = UserAudioTap(mic, max_secs=speaker_cfg.max_secs)

    def _on_grant(turn):
        tap.clear()
        if turn.audible and not mic.awake:
            # she is about to speak aloud: be awake for it (and for what he says back), the way an
            # announce used to wake her — the quiet rule may have delayed this past the doze
            mic.wake(f"{turn.mind} has the floor ({turn.kind})")
    floor.on_grant = _on_grant

    _orig_on_speaking = floor.on_speaking

    def _on_speaking(slug, turn_id, on):
        if on and mic.last_user_final is not None:      # felt latency: he stops -> a sister speaks
            _metrics.record("turn", "latency", (time.monotonic() - mic.last_user_final) * 1000.0)
            mic.last_user_final = None
        _orig_on_speaking(slug, turn_id, on)
    floor.on_speaking = _on_speaking

    # speaker voiceprint + vocal affect, computed ONCE per turn and forwarded to every sister
    speaker_model = os.getenv("AIVA_SPEAKER_MODEL") or os.path.join(_HERE, "models", "campplus_cn_en_common_200k.onnx")
    embedder = SpeakerRecognizer(SpeakerMemory(sqlite3.connect(":memory:")), speaker_cfg,
                                 model_path=speaker_model,
                                 enabled=(os.getenv("AIVA_SPEAKER_ID", "1") == "1" and os.path.isfile(speaker_model)),
                                 log=log_event)
    if embedder.enabled:
        print(f"Speaker embedding ON (model {os.path.basename(speaker_model)})")
        asyncio.create_task(asyncio.to_thread(embedder.warm))
    else:
        print("Speaker embedding OFF (set AIVA_SPEAKER_ID=1 and provide the model)")
    vocal_affect = VocalAffect(AffectConfig.from_env(),
                               enabled=os.getenv("AIVA_VOICE_AFFECT", "1") == "1", log=log_event)
    pending = {"vec": None, "affect": None}

    async def _on_user_final(text):
        pcm = tap.take_segment()
        pending["vec"], pending["affect"] = None, None
        if len(pcm) < int(16000 * 0.3) * 2:
            return
        if vocal_affect.enabled:
            try:
                pending["affect"] = await asyncio.to_thread(vocal_affect.analyze, pcm)
            except Exception as e:
                log_event(f"vocal affect error: {e!r}")
        if embedder.enabled and len(pcm) >= int(speaker_cfg.min_secs * 16000) * 2:
            try:
                vec = await asyncio.wait_for(embedder.embed_pcm(pcm), timeout=speaker_cfg.embed_timeout_ms / 1000)
                if vec is not None:
                    pending["vec"] = [float(x) for x in vec]
            except Exception:
                pass

    async def _on_user_turn(text):
        hint = mic.wake_hint
        mic.wake_hint = None
        vec, affect = pending["vec"], pending["affect"]
        pending["vec"], pending["affect"] = None, None
        await floor.submit_user_turn(text, speaker_vec=vec, affect=affect, wake_hint=hint)

    def _name_check(text):
        return bool(which_names(text, namesets))

    if stt_cfg.flux:
        # Flux sends no interim transcripts (ActivityWatcher pokes the idle timer on those with
        # nova) — its in-turn events keep the house awake while he talks instead
        @stt.event_handler("on_start_of_turn")
        async def _flux_turn_start(_svc, _transcript):
            mic.poke()

        @stt.event_handler("on_update")
        async def _flux_update(_svc, _transcript):
            mic.poke()

    stage_ctx = LLMContext(messages=[{"role": "system", "content": "stage"}])
    user_aggregator = LLMUserAggregator(
        stage_ctx,
        params=stage_params(stt_cfg, [FloorUserMuteStrategy(lambda: floor.gate_active)]),
    )
    router = StageRouter(stage_ctx, on_user_turn=_on_user_turn, on_user_speaking=floor.note_user_speaking,
                         log=log_event)

    pipeline = Pipeline([
        transport.input(),
        mic,
        tap,
        stt,
        ActivityWatcher(mic, _name_check, on_user_final=_on_user_final, log=log_event),
        user_aggregator,
        router,
    ])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        # while asleep MicController drops all audio, so pipecat's idle monitor would cancel us
        idle_timeout_secs=None,
    )

    @worker.event_handler("on_pipeline_finished")
    async def _on_finished(_worker, frame):
        reason = getattr(frame, "reason", None)
        print(f"\n[PIPELINE ENDED] {type(frame).__name__}" + (f" reason={reason}" if reason else ""))

    @worker.event_handler("on_pipeline_error")
    async def _on_pipeline_error(_worker, frame):
        print(f"\n[PIPELINE ERROR] {getattr(frame, 'error', '?')} (fatal={getattr(frame, 'fatal', False)})")

    # --- the server -------------------------------------------------------------------------------
    def _mic_state():
        return {"awake": mic.visibly_awake, "mic_muted": mic.muted}

    def _on_presence(present):
        memory.kv_set("present", json.dumps(present))
        names = ", ".join(s.capitalize() for s in present) or "nobody"
        print(f"[HOUSE] connected: {names}")

    stop_event = asyncio.Event()
    exit_code = {"code": 0}

    def _restart_house(slug=None, target="house"):
        """Exit 42 so start_house.bat relaunches us; the sisters are told it's only the house and
        keep running — they reconnect on their own when we're back (a few seconds of deafness)."""
        who = f"{slug} asked" if slug else "hotkey"
        print(f"\n[HOUSE] restarting ({who})")
        log_event(f"house restart requested ({who})")
        exit_code["code"] = 42
        stop_event.set()

    server = HouseServer(
        floor, memory, host=os.getenv("AIVA_HOUSE_HOST", "127.0.0.1"),
        port=int(os.getenv("AIVA_HOUSE_PORT", "8761")), log=log_event, mic_state=_mic_state,
        on_sleep=lambda slug: mic.sleep(),
        on_wake=lambda slug: mic.wake(f"{slug} asked to wake"),
        on_announce=lambda slug: mic.wake(f"{slug} has news"),
        on_presence_change=_on_presence, on_restart=_restart_house, version=_version)
    # --- dashboard buttons: the hotkeys, over the socket ----------------------------------------
    def _start_sister(slug):
        import subprocess
        bat = os.path.join(os.path.dirname(_HERE), f"start_{slug}.bat")
        if slug in server.sessions:
            log_event(f"control: {slug} is already connected")
            return
        if not os.path.isfile(bat):
            log_event(f"control: no {os.path.basename(bat)} to start {slug}")
            return
        log_event(f"control: starting {slug} via {os.path.basename(bat)}")
        subprocess.Popen(["cmd", "/c", bat], cwd=os.path.dirname(bat),
                         creationflags=subprocess.CREATE_NEW_CONSOLE)

    def _on_control(action, target=None):
        target = (target or "").strip().lower() or None
        if action == "mute":
            mic.toggle_mute()
        elif action == "interrupt":
            _interrupt()
        elif action == "sister_talk":
            if target in ("audible", "private"):
                floor.set_private(target == "private")
            else:
                _flip()
        elif action == "sleep":
            mic.sleep()
        elif action == "wake":
            mic.wake("dashboard")
        elif action == "restart_sisters":
            _restart_sisters()
        elif action == "restart_sister" and target:
            log_event(f"control: restart {target}")
            server.send_to(target, proto.Shutdown(reason="restart"))
        elif action == "quit_sister" and target:
            log_event(f"control: quit {target}")
            server.send_to(target, proto.Shutdown(reason="quit"))
        elif action == "mute_sister" and target:
            log_event(f"control: mute {target}")
            floor.set_muted(target, True)
        elif action == "unmute_sister" and target:
            log_event(f"control: unmute {target}")
            floor.set_muted(target, False)
        elif action == "start_sister" and target:
            _start_sister(target)
        elif action == "restart_house":
            _restart_house(None)
        elif action == "quit":
            _quit()
        else:
            log_event(f"control: unknown action {action!r} target={target!r}")

    async def _on_say(text):
        """A line typed into the dashboard's chat box. It goes through the Floor exactly like a
        final transcript: same addressee decision, same queue, same spoken reply — no voiceprint
        and no vocal affect, because nobody spoke. Typing is presence, so it wakes the house (even
        while the mic is deafened: it isn't audio, and he is plainly here)."""
        text = (text or "").strip()
        if not text:
            return
        mic.poke()
        if not mic.awake:
            mic.wake("typed in the dashboard")
        log_event(f"typed: {text!r}")
        print(f"\n[typed] {text}")
        await floor.submit_user_turn(text)

    server.on_control = _on_control
    server.on_say = _on_say
    await server.start()
    memory.kv_set("present", json.dumps([]))
    memory.kv_set("house_pid", os.getpid())
    print(f"House listening on 127.0.0.1:{server.port} · sister talk {floor.mode}")
    if notifier.available() and os.getenv("AIVA_NTFY_ON_START", "1") == "1":
        asyncio.create_task(notifier.notify_async(f"The house is up (v{_version}).", title="Aiva", tags=["house"]))

    # --- hotkeys -----------------------------------------------------------------------------------
    loop = asyncio.get_running_loop()
    flip_key = os.getenv("AIVA_HOTKEY_SISTER_TALK", "ctrl+alt+p")

    def _flip():
        ok = floor.set_private(not floor.private)
        mode = floor.mode.upper()
        print(f"\n[{flip_key}] sister talk: {mode}" + ("" if ok else " (unchanged — no connected sister has a voice)"))

    def _interrupt():
        print("\n[F10] Interrupting" + (f" {floor.holder}" if floor.holder else " (nobody speaking)"))
        floor.interrupt()

    def _restart_sisters():
        print("\n[Ctrl+Alt+R] Restarting the sisters")
        log_event("restart requested for the sisters (Ctrl+Alt+R)")
        server.broadcast(proto.Shutdown(reason="restart"))

    def _quit():
        print("\n[Ctrl+Alt+X] Quitting the house and the sisters")
        server.broadcast(proto.Shutdown(reason="quit"))
        stop_event.set()

    try:
        import keyboard

        keyboard.add_hotkey("f9", lambda: loop.call_soon_threadsafe(mic.toggle_mute))
        keyboard.add_hotkey("f10", lambda: loop.call_soon_threadsafe(_interrupt))
        keyboard.add_hotkey(flip_key, lambda: loop.call_soon_threadsafe(_flip))
        keyboard.add_hotkey("ctrl+alt+r", lambda: loop.call_soon_threadsafe(_restart_sisters))
        keyboard.add_hotkey("ctrl+alt+h", lambda: loop.call_soon_threadsafe(_restart_house))
        keyboard.add_hotkey("ctrl+alt+x", lambda: loop.call_soon_threadsafe(_quit))
        print(f"Hotkeys: F9 = mute/unmute mic, F10 = interrupt, {flip_key} = sister talk private/audible, "
              "Ctrl+Alt+R = restart sisters, Ctrl+Alt+H = restart the house, Ctrl+Alt+X = quit all")
    except Exception as e:
        print(f"Global hotkeys unavailable ({e}); use Ctrl+C to quit")

    async def watch_stop():
        await stop_event.wait()
        print("\nShutting down the house...")
        await worker.cancel()

    watcher = asyncio.create_task(watch_stop())

    # --- ambient mode: wake words + idle timeout -------------------------------------------------
    wake_listener = None
    idle_task = None
    if ambient:
        verify_wakes = os.getenv("AIVA_WAKE_VERIFY", "1") == "1"
        models = _wake_models() or [os.getenv("AIVA_WAKE_MODEL", "hey_jarvis")]
        wake_listener = WakeWordListener(
            models=models,
            on_wake=lambda score, preroll, name: loop.call_soon_threadsafe(
                mic.wake, f"wake word {name} (score={score:.2f})", preroll, verify_wakes, name),
            device_index=mic_index,
            threshold=float(os.getenv("AIVA_WAKE_THRESHOLD", "0.5")),
            patience=int(os.getenv("AIVA_WAKE_PATIENCE", "2")),
            log=log_event,
        )
        wake_listener.start()

        async def idle_watch():
            while True:
                await asyncio.sleep(2)
                if (mic.awake and not mic.bot_speaking and not mic.processing
                        and mic.idle_for() > idle_timeout):
                    mic.sleep()

        idle_task = asyncio.create_task(idle_watch())
        print("[IDLE] say a wake word to talk to the sisters (" + ", ".join(model_name(m) for m in models) + ")")

    async def state_watch():
        """Tell the sisters when the house sleeps/wakes or the mic is (un)muted."""
        last = None
        while True:
            await asyncio.sleep(0.3)
            cur = (mic.visibly_awake, mic.muted)
            if cur != last:
                last = cur
                floor.set_awake(mic.awake)
                server.broadcast_state()

    state_task = asyncio.create_task(state_watch())

    async def housekeeping():
        while True:
            await asyncio.sleep(3600)
            try:
                memory.prune()
            except Exception as e:
                log_event(f"house prune failed: {e!r}")

    keep_task = asyncio.create_task(housekeeping())

    # --- run ---------------------------------------------------------------------------------------
    print("The house is ready." + (" (ambient mode)" if ambient else " Just talk."))
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        for t in (watcher, idle_task, state_task, keep_task):
            if t:
                t.cancel()
        if wake_listener:
            wake_listener.stop()
        try:
            stt_min = mic.stt_seconds / 60
            await service_limits.record("stt", stt_min)
            print(f"\nSTT streamed: {stt_min:.1f} min (~${stt_min * 0.0077:.3f})")
        except Exception:
            pass
        await server.stop("house_restart" if exit_code["code"] == 42 else "quit")
        log_event("house shutdown" + (" (restarting)" if exit_code["code"] == 42 else ""))
        try:
            house_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        house_conn.close()
    return exit_code["code"]


if __name__ == "__main__":
    if "--list-devices" in sys.argv:
        list_audio_devices()
    else:
        code = asyncio.run(main())
        if code:
            sys.exit(code)
