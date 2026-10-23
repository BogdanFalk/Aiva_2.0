"""One sister's process (pipecat 1.x) — Aiva or Saga.

    python main.py [--character aiva|saga] [--list-devices]

She has no microphone. The HOUSE (house.py) owns the one ear, Deepgram, the wake words and the
floor; it grants her turns over a loopback socket and she plays her own voice:

  HouseLink → OpenAI LLM (native tools, streamed) → PrivacyGate → ElevenLabs TTS → LipSyncTap
  → MindWatcher → UsageTracker → speaker (audio out) → assistant aggregator

Everything that is HERS lives in her own db (memory.db / saga.db), her own state files, her own
space folder and her own avatar renderer process, so she can be started and stopped on her own.
Exit code 42 = restart (start_aiva.bat / start_saga.bat relaunch).
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

# .env is THE config source (CLAUDE.md): override stale values inherited from the launching shell
load_dotenv(override=True)

# hotkey/expression names can be CJK; don't let a console print crash a handler
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _install_log_throttle():
    """pipecat logs "Empty audio frame received for STT service" on EVERY zero/empty audio frame —
    kept for the odd frame that still passes; everything else logs normally."""
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
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

from modules import capabilities, character, computer, file_operations, house_proto as proto
from modules import metrics as _metrics
from modules.avatar import AvatarClient
from modules.avatar_ipc import read_state as _read_avatar_state
from modules.becoming import Becoming, SELF_FORMING_BASELINE_DESC
from modules.cat import CAT_NAME_RECOVERY_NOTE, CAT_NAMING_NOTE, DesktopCat
from modules.cat_care import CatCaretaker
from modules.drawing import board_for
from modules.ear import MicController, device_index, list_audio_devices  # noqa: F401 (tests import MicController)
from modules.eventlog import EventLog
from modules.house_client import HouseClient
from modules.house_link import HouseLink, LipSyncTap, MindWatcher, PrivacyGate, UsageTracker
from modules.housedb import house_db_path, open_house_db
from modules.inner import ExperienceBus, InnerBrain, InnerConfig, InnerState, SelfEngine
from modules.limits import ServiceLimits
from modules.memory import Memory
from modules.metrics import Metrics
from modules.notify import Notifier, NotifyConfig
from modules.persona import build_system_prompt
from modules.preferences import Preferences
from modules.reading import BookMemory, ReadingEngine, StudyLimits, embed_long
from modules.space import SpaceMemory
from modules.speakers import SpeakerConfig, SpeakerMemory, SpeakerRecognizer
from modules.task_memory import TaskMemory, embed_text
from modules.tasks import Limits, TaskRunner, ToolBox, openai_llm_call
from modules.terminals import TerminalManager
from modules.tools import (TOOL_SCHEMAS, execute_shell, register_becoming_tools,
                           register_preference_tools, register_reading_tools,
                           register_self_tools, register_speaker_tools, register_task_tools,
                           register_tools, register_web_tools)
from modules.utilities import LLM_RATES as _LLM_RATES
from modules.web import WebClient, WebConfig

from modules.house_memory import HouseMemory
from modules.kin import TOOL_TO_CAPABILITY, Kin
from modules.tools import bind_tool_hook, register_kin_tools, register_sister_tools
from modules.tools import register_life_tools
from modules.dreams import Consolidation
from modules import projects as projects_mod

_HERE = os.path.dirname(os.path.abspath(__file__))

# rebound to the character's own log file in main(); module-level so every closure sees it
log_event = EventLog("aiva")


class PatientOpenAILLMService(OpenAILLMService):
    """OpenAILLMService whose client retries 429/5xx with exponential backoff (honouring
    retry-after). pipecat builds its client with the SDK default of 2 fast retries — on a 30k
    tokens-per-minute tier that turned a transient rate limit into a silent, dead turn.
    AIVA_LLM_MAX_RETRIES tunes it."""

    def create_client(self, api_key=None, base_url=None, organization=None, project=None,
                      default_headers=None, **kwargs):
        client = super().create_client(api_key=api_key, base_url=base_url, organization=organization,
                                       project=project, default_headers=default_headers, **kwargs)
        return client.with_options(max_retries=int(os.getenv("AIVA_LLM_MAX_RETRIES", "5")))


def trim_context(context, memory=None, budget_chars=None, keep_min=8):
    """Keep the live context under a size budget.

    Every turn re-sends the WHOLE context (plus ~5.5k tokens of tool schemas), so an unbounded
    history is what eventually walks into the tokens-per-minute limit. Drops the oldest
    conversation messages — never the leading system block, never an in-flight tool exchange, and
    always keeps the last `keep_min` messages. Tells `memory` how many already-saved spoken
    messages went, so transcript persistence stays consistent. Returns the number dropped."""
    budget = budget_chars or int(os.getenv("AIVA_CONTEXT_BUDGET_CHARS", "24000"))
    msgs = list(context.get_messages())
    lead = 0
    while lead < len(msgs) and msgs[lead].get("role") == "system":
        lead += 1
    head, body = msgs[:lead], msgs[lead:]

    def size(m):
        c = m.get("content")
        return len(c) if isinstance(c, str) else len(json.dumps(c, ensure_ascii=False, default=str)) \
            if c is not None else 40

    total = sum(size(m) for m in body)
    dropped, spoken = 0, 0
    # trim while over budget, then keep going to the next user message so the surviving history
    # starts at a turn boundary
    while len(body) > keep_min and (total > budget or body[0].get("role") != "user"):
        m = body[0]
        if m.get("role") == "tool" and m.get("content") == "IN_PROGRESS":
            break  # a result is still on its way
        pending_ids = {tc.get("id") for tc in (m.get("tool_calls") or []) if isinstance(tc, dict)}
        if pending_ids and any(x.get("role") == "tool" and x.get("tool_call_id") in pending_ids
                               and x.get("content") == "IN_PROGRESS" for x in body):
            break
        body.pop(0)
        total -= size(m)
        dropped += 1
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            spoken += 1
        # results of a dropped tool call go with it
        while pending_ids and body and body[0].get("role") == "tool" \
                and body[0].get("tool_call_id") in pending_ids:
            t = body.pop(0)
            total -= size(t)
            dropped += 1
    if dropped:
        context.set_messages(head + body)
        if memory is not None and spoken:
            memory.forget_saved(spoken)
    return dropped


def drop_intro_note(messages, intro_text):
    """Remove the boot 'introduce yourself' system note. It's a ONE-SHOT: it lives in the leading
    system block, which trim_context never drops, so left in place it keeps telling her to greet
    on every wake. Returns (messages, removed)."""
    kept = [m for m in messages
            if not (m.get("role") == "system" and m.get("content") == intro_text)]
    return kept, len(kept) != len(messages)


def character_from_argv(argv=None):
    argv = sys.argv if argv is None else argv
    if "--character" in argv:
        i = argv.index("--character")
        if i + 1 < len(argv):
            return argv[i + 1].strip().lower()
    return os.getenv("AIVA_CHARACTER", "aiva").strip().lower()


class _NoCat:
    """Stand-in for the sister who has no desktop cat: every cat tool reports 'not running'."""

    class _State:
        name = None
        dismissed = True
        naming_note_shown = True

    available = False
    state = _State()

    def start(self): pass
    def stop(self): pass
    def save_state(self): pass
    def attach_announcer(self, loop, coro): pass
    def set_name(self, name): pass
    def feed(self): pass
    def play(self): pass
    def pet(self): pass
    def dismiss(self): pass
    def summon(self): pass
    def status(self): return {"available": False}
    def persona_note(self): return None


class _MicProxy:
    """What the tool handlers need from 'the mic' now that the ear lives in the house: `ambient`
    and `sleep()` (go_to_sleep asks the house), plus `awake` mirrored from the house's state."""

    def __init__(self, ambient, send):
        self.ambient = ambient
        self.awake = not ambient
        self.muted = False
        self._send = send

    def sleep(self):
        self._send(proto.Sleep())

    def wake(self, reason=""):
        self._send(proto.Wake())


def _require_env(*names):
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        print("Missing required .env keys: " + ", ".join(missing))
        sys.exit(1)


def _play_ready_chime():
    """Two soft rising blips through the DEFAULT output — audible proof of life once she's up."""
    import math
    import struct
    import tempfile
    import wave
    import winsound

    sr = 44100

    def tone(freq, dur, vol=0.35):
        n = int(sr * dur)
        return [vol * math.sin(2 * math.pi * freq * t / sr) * (1 - t / n) for t in range(n)]

    samples = tone(660, 0.12) + tone(880, 0.20)
    path = os.path.join(tempfile.gettempdir(), "aiva_ready_chime.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(s * 32767)) for s in samples))
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)


def _restart_marker(slug):
    return os.path.join(_HERE, ".restart_marker" if slug == "aiva" else f".restart_marker_{slug}")


def _consume_restart_marker(slug):
    """True once if the previous process exited via self-restart just now."""
    path = _restart_marker(slug)
    try:
        fresh = time.time() - os.path.getmtime(path) < 300
        os.remove(path)
        return fresh
    except OSError:
        return False


def _spawn_detached_relauncher():
    """Relaunch fallback for when she was NOT started by her .bat (whose restart loop handles exit
    code 42 itself): a hidden helper python waits on this process's kernel handle, then starts the
    same command line again in a fresh console window."""
    import subprocess

    watcher = (
        "import ctypes, subprocess, time; "
        "k = ctypes.windll.kernel32; "
        f"h = k.OpenProcess(0x00100000, False, {os.getpid()}); "  # SYNCHRONIZE
        "k.WaitForSingleObject(h, 0xFFFFFFFF); "
        "time.sleep(0.3); "
        f"subprocess.Popen({[sys.executable] + sys.argv!r}, cwd={os.getcwd()!r}, "
        "creationflags=subprocess.CREATE_NEW_CONSOLE)"
    )
    subprocess.Popen(
        [sys.executable, "-c", watcher],
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
    )


def _print_usage_summary(usage, llm_model, name):
    """Session cost estimate (rates: mid-2026)."""
    rate_in, rate_out = _LLM_RATES.get(llm_model, _LLM_RATES["gpt-4.1-mini"])
    llm_cost = usage.prompt_tokens * rate_in / 1e6 + usage.completion_tokens * rate_out / 1e6
    tts_credits = usage.tts_chars * 0.5  # Flash v2.5
    print(f"\n----- {name}: session usage -----")
    print(f"LLM tokens:    {usage.prompt_tokens} in / {usage.completion_tokens} out (~${llm_cost:.3f})")
    print(f"TTS:           {usage.tts_chars} chars = {tts_credits:.0f} ElevenLabs credits")
    print("----------------------------------")


SAGA_FIRST_BOOT_INTRO = (
    "[First boot. You've just come online for the first time. You don't know this person's voice or "
    "name yet — Aiva, your sister, does; she's been here a while. Say hello in one or two short "
    "sentences, in your own words: you're Saga, you're new, and you'd like to know who you're talking "
    "to. Don't act like you already have a personality; be honest and curious.]"
)


async def main():
    global log_event
    slug = character_from_argv()
    ch = character.Character.from_env(slug)
    log_event = EventLog(ch.slug)
    _require_env("OPENAI_API_KEY", "ELEVENLABS_API_KEY")

    llm_model = os.getenv("AIVA_LLM_MODEL", "gpt-4.1-mini")
    ambient = os.getenv("AIVA_WAKE_WORD", "0") == "1"
    from version import __version__ as _version
    print(f"{ch.name} v{_version}")
    log_event(f"--- boot v{_version} (pid={os.getpid()}, character={ch.slug}, ambient={ambient}) ---")
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW(f"{ch.name}Console")
    except Exception:
        pass

    # --- her voice out -------------------------------------------------------------------------
    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=False, audio_out_enabled=True,
        output_device_index=device_index(f"{ch.slug.upper()}_OUTPUT_DEVICE_INDEX"
                                         if ch.slug != "aiva" else "AIVA_OUTPUT_DEVICE_INDEX",
                                         f"{ch.slug.upper()}_OUTPUT_DEVICE_NAME"
                                         if ch.slug != "aiva" and os.getenv(f"{ch.slug.upper()}_OUTPUT_DEVICE_NAME")
                                         else "AIVA_OUTPUT_DEVICE_NAME", True)))

    llm = PatientOpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        # NOTE: keep run_in_parallel at its default (True). Sequential mode registers calls lazily,
        # which defeats pipecat's call-grouping and makes her speak once per tool instead of once
        # per turn. gpt-5.x: max_completion_tokens instead of max_tokens, and function tools on
        # Chat Completions REQUIRE reasoning_effort 'none'. gpt-4.x keeps the classic shape.
        settings=OpenAILLMService.Settings(
            model=llm_model,
            temperature=0.7,
            **({"max_completion_tokens": 500, "extra": {"reasoning_effort": "none"}}
               if llm_model.startswith("gpt-5") else {"max_tokens": 500}),
        ),
    )

    tts = None
    if ch.voice_id:
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            settings=ElevenLabsTTSService.Settings(voice=ch.voice_id, model=ch.tts_model),
        )
    else:
        print(f"{ch.name}: no voice yet ({ch.slug.upper()}_ELEVENLABS_VOICE_ID unset) — text-only")
        log_event("no voice id — running text-only")

    # --- her core --------------------------------------------------------------------------------
    terminals = TerminalManager(name=ch.name)
    memory = Memory(ch.db_path, backup_stem=ch.backup_stem, name=ch.name,
                    name_variants=ch.name_variants, sisters=ch.sisters)
    memory.check_identity(ch.slug)
    memory.check_embeddings("text-embedding-3-small", 512)
    _metrics.set_recorder(Metrics(memory.conn))       # operations profiler (dashboard Statistics)
    house_conn = open_house_db(house_db_path())         # shared spend caps live in the house db
    from openai import AsyncOpenAI
    oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                      max_retries=int(os.getenv("AIVA_LLM_MAX_RETRIES", "5")))
    task_memory = TaskMemory(memory.conn, embed=lambda text: embed_text(oai, text),
                             name_variants=ch.name_variants, name=ch.name)
    interrupted_tasks = task_memory.reconcile_stale()
    if interrupted_tasks:
        log_event(f"{len(interrupted_tasks)} task(s) were interrupted by the last shutdown")
    cat = DesktopCat(state_file=ch.cat_state_path) if ch.cat_state_path else _NoCat()
    cat.start()

    # her body: adopt a renderer that survived a restart, or spawn a fresh one (non-fatal)
    had_body = _read_avatar_state(ch.avatar_state_path) is not None
    avatar = AvatarClient() if ch.slug == "aiva" else AvatarClient(
        instance=ch.slug, model=ch.avatar_model,
        monitor=(int(os.getenv(f"{ch.slug.upper()}_AVATAR_MONITOR")) if os.getenv(f"{ch.slug.upper()}_AVATAR_MONITOR") else None))
    await avatar.ensure_running()
    if ch.slug != "aiva" and not had_body and avatar.connected:
        # first spawn: stand on the mirrored side of her sister's monitor unless told otherwise
        try:
            sis = _read_avatar_state() or {}
            pl = sis.get("placement") or {}
            x = float(os.getenv(f"{ch.slug.upper()}_AVATAR_X", -float(pl.get("x", 0.8))))
            y = float(os.getenv(f"{ch.slug.upper()}_AVATAR_Y", float(pl.get("y", -0.65))))
            await avatar.move(x=x, y=y, size=float(pl.get("size", 0.0)))
        except Exception as e:
            log_event(f"avatar first placement skipped: {e!r}")

    # --- the house link ----------------------------------------------------------------------------
    speaking_state = {"on": False, "last_activity": time.monotonic(), "awake": not ambient,
                      "present": [], "sister_talk": os.getenv("AIVA_SISTER_TALK", "private")}
    client_ref = {}

    def _send(msg):
        c = client_ref.get("client")
        return c.send(msg) if c is not None else False

    mic = _MicProxy(ambient, _send)

    def _idle_for():
        return time.monotonic() - speaking_state["last_activity"]

    # --- restart machinery -------------------------------------------------------------------------
    stop_event = asyncio.Event()
    restart_state = {"requested": False}

    async def trigger_restart():
        deadline = time.time() + 30
        while not speaking_state["on"] and time.time() < deadline - 20:
            await asyncio.sleep(0.1)  # give her goodbye line time to start
        while speaking_state["on"] and time.time() < deadline:
            await asyncio.sleep(0.1)  # ...and let it finish playing
        await asyncio.sleep(0.3)
        restart_state["requested"] = True
        stop_event.set()

    # her drawing canvas: opened from her own process, never a terminal; strokes in her own ink.
    # board_for is this process's one board, so anything else of hers reuses the same window
    drawing = board_for(ch.slug)
    handlers = register_tools(llm, avatar, mic, terminals, memory, cat, restart=trigger_restart,
                              name=ch.name, capability_exclude=ch.capability_exclude,
                              drawing=drawing)

    # --- task runner --------------------------------------------------------------------------------
    task_model = os.getenv("AIVA_TASK_MODEL", llm_model)
    toolbox = ToolBox(handlers, TOOL_SCHEMAS, name=ch.name)
    task_runner = TaskRunner(
        task_memory=task_memory, toolbox=toolbox, limits=Limits.from_env(),
        llm_call=openai_llm_call(oai, task_model, label="task"), terminals=terminals, log=log_event,
        execute_shell=lambda cmd: asyncio.to_thread(execute_shell, cmd),
        describe_screen=lambda *a, **k: computer.describe_screen(*a, name=ch.name, **k),
        list_monitors=computer.list_monitors,
        resolve_path=file_operations.resolve_path, facts=memory.load_facts,
        model_name=task_model, name=ch.name,
        on_outcome=lambda outcome, goal: _feel_task(outcome, goal),
        on_result=lambda goal, summary: _settle_wishes(goal, summary),
        on_done=lambda summary, family: _send(proto.Event(kind="task_done", summary=summary,
                                                          payload={"family": family})),
    )
    task_runner.openai_client = oai
    task_runner.remember = memory.remember
    task_runner.retire_fact = memory.retire_fact      # second thoughts may retire a fact a report disproved
    def _heard_him():
        # a task's question is his to answer: only a turn he spoke in (or typed) may answer it — not
        # her own news turn, not a sister round. `link` is bound further down, before any turn runs.
        try:
            return (link.current_turn or {}).get("kind") in ("user", "both_second")
        except NameError:
            return False
    register_task_tools(llm, task_runner, task_memory, memory, heard_him=_heard_him)

    # --- reading + becoming -------------------------------------------------------------------------
    os.makedirs(ch.library_dir, exist_ok=True)
    study_model = os.getenv("AIVA_STUDY_MODEL", task_model)
    synth_model = os.getenv("AIVA_STUDY_SYNTH_MODEL", "gpt-5.6-terra")
    book_memory = BookMemory(memory.conn, embed=lambda t: embed_long(oai, t))
    interrupted_reading = book_memory.reconcile_stale()
    reading = ReadingEngine(book_memory, llm_call=openai_llm_call(oai, study_model, label="reading"),
                            synth_call=openai_llm_call(oai, synth_model, label="synth"), client=oai,
                            library_dir=ch.library_dir, model_name=study_model,
                            synth_model_name=synth_model, log=log_event, limits=StudyLimits.from_env(),
                            name=ch.name)
    register_reading_tools(llm, reading)

    async def _becoming_llm(messages, max_tokens=1200, temperature=0.2):
        from modules.utilities import llm_kwargs
        async with _metrics.span("llm", "becoming"):
            r = await oai.chat.completions.create(
                model=task_model, messages=messages, response_format={"type": "json_object"},
                **llm_kwargs(task_model, max_tokens, temperature))
        return r.choices[0].message.content

    becoming = Becoming(memory.conn, embed=lambda t: embed_text(oai, t), llm=_becoming_llm,
                        model=task_model, remember=memory.remember, baseline_traits=list(ch.baseline_traits),
                        log=log_event, engine=reading, name=ch.name,
                        baseline_desc=SELF_FORMING_BASELINE_DESC if ch.self_forming else None)
    register_becoming_tools(llm, becoming)
    reading.on_book_finished = becoming.on_book_finished
    reading.on_session_end = becoming.on_session_end
    reading.on_book_added = becoming.on_book_added
    reading.on_session_start = becoming.note_session_started
    body_kick = asyncio.Event()                         # her body hears about a change at once
    reading.on_reading_state = lambda _on: body_kick.set()  # the book comes out / goes away
    task_runner._reading = becoming.relevant_for
    becoming.fade()

    # --- web + space + caps -----------------------------------------------------------------------
    os.makedirs(ch.space_dir, exist_ok=True)
    space_mem = SpaceMemory(memory.conn, embed=lambda t: embed_text(oai, t), log=log_event,
                            on_saved=lambda subject, kind, why: _send(proto.Event(
                                kind="find_saved", summary=f"kept {kind}: {subject}", payload={"why": why})))
    web_client = WebClient(WebConfig.from_env(), log=log_event)
    notifier = Notifier(NotifyConfig.from_env(), log=log_event)
    service_limits = ServiceLimits(house_conn, notifier=notifier, log=log_event)
    print(f"ntfy alerts {'ON' if notifier.available() else 'OFF (set NTFY_TOPIC)'} · "
          f"Tavily cap {service_limits.cap('tavily') or 'off'}/mo (shared)")
    if notifier.available() and os.getenv("AIVA_NTFY_ON_START", "1") == "1":
        _boot_kind = "restarted" if os.path.exists(_restart_marker(ch.slug)) else "is awake"
        asyncio.create_task(notifier.notify_async(
            f"{ch.name} {_boot_kind} (v{_version}).", title=ch.name, tags=["robot"]))
    _web_llm = openai_llm_call(oai, os.getenv("AIVA_WEB_MODEL") or task_model, label="web")

    async def _summarize_page(text, *, focus=None):
        prompt = (f"Summarize this web page for {ch.name} in two to four plain spoken sentences — no "
                  "markdown, no lists. Lead with what it actually says; keep any key name or number exact.")
        if focus:
            prompt += f" She wants to know: {focus}."
        try:
            reply = await _web_llm([{"role": "system", "content": prompt},
                                    {"role": "user", "content": (text or "")[:8000]}],
                                   max_tokens=300, temperature=0.3)
            return (reply.text or "").strip() or (text or "")[:600]
        except Exception:
            return (text or "")[:600]

    web_handlers = register_web_tools(llm, web_client, space_mem,
                                      describe_image=computer.describe_image,
                                      summarize=_summarize_page, space_dir=ch.space_dir,
                                      open_file=file_operations.open_file, limits=service_limits,
                                      log=log_event, name=ch.name)
    toolbox.add_handlers(web_handlers)
    print(f"Web reach {'ON' if web_client.available() else 'OFF (set TAVILY_API_KEY)'} · "
          f"space at {os.path.basename(ch.space_dir)}")

    # --- her nervous system -------------------------------------------------------------------------
    inner_brain = InnerBrain(InnerConfig(**ch.mood_cfg) if ch.mood_cfg else None)
    experience = ExperienceBus(memory.conn)
    preferences = Preferences(memory.conn, embed=lambda t: embed_text(oai, t), log=log_event)
    if ch.baseline_tastes:
        _seeded = preferences.seed_from_baseline()
        if _seeded:
            print(f"Seeded {_seeded} baseline preferences (now mutable)")
    preferences.fade()
    register_preference_tools(llm, preferences)

    # what she knows about her sister, from what she has SEEN (house-relayed public events) and
    # what she noticed in conversation — pure sqlite, no LLM to answer "what can Saga do?"
    kin = None
    house_mem = HouseMemory(house_conn)          # read handle for catch-up (the house writes)
    if ch.sisters:
        kin = Kin(memory.conn, me=ch.slug, sisters=tuple(s.lower() for s in ch.sisters), bus=experience,
                  embed=lambda t: embed_text(oai, t), log=log_event)
        kin.fade()
        toolbox.add_handlers(register_kin_tools(llm, kin))

        def _tool_seen(tool, ok):
            cap = TOOL_TO_CAPABILITY.get(tool)
            if ok and cap:
                _send(proto.Event(kind="tool_done", summary=f"used {tool}", capability=cap,
                                  payload={"tool": tool}))
        bind_tool_hook(llm, _tool_seen)

    # "call your sister": boot the other one in her own console (her .bat, exactly like double-
    # clicking it) when she isn't connected to the house
    sister_slug = ch.sisters[0].lower() if ch.sisters else None

    def _boot_sister():
        import subprocess
        bat = os.path.join(os.path.dirname(_HERE), f"start_{sister_slug}.bat")
        if not os.path.isfile(bat):
            raise FileNotFoundError(bat)
        log_event(f"booting {sister_slug} via {os.path.basename(bat)}")
        subprocess.Popen(["cmd", "/c", bat], cwd=os.path.dirname(bat),
                         creationflags=subprocess.CREATE_NEW_CONSOLE)

    if sister_slug:
        toolbox.add_handlers(register_sister_tools(
            llm, sister_name=ch.sisters[0],
            is_present=lambda: sister_slug in speaking_state["present"], boot=_boot_sister,
            restart_house_send=lambda: _send(proto.Restart(target="house")),
            get_mode=lambda: speaking_state["sister_talk"],
            set_mode=lambda mode: _send(proto.Mode(sister_talk=mode))))

    async def _kin_catch_up():
        if kin is None:
            return
        try:
            n = await kin.catch_up(house_mem.events_since(kin.cursor))
            if n:
                log_event(f"kin: caught up on {n} house event(s)")
        except Exception as e:
            log_event(f"kin catch-up failed: {e!r}")

    self_engine = SelfEngine(
        memory.conn, experience, inner_brain,
        state=InnerState.load(ch.inner_state_path, inner_brain), state_path=ch.inner_state_path,
        embed=lambda t: embed_text(oai, t), llm=openai_llm_call(oai, task_model, label="self"),
        model=task_model, log=log_event, preferences=preferences,
        journal_salience=float(os.getenv("AIVA_JOURNAL_SALIENCE", "2.0")),
        journal_min_gap_h=float(os.getenv("AIVA_JOURNAL_MIN_GAP_H", "12")),
        name=ch.name, self_forming=ch.self_forming, becoming=becoming, kin=kin,
        drives=os.getenv("AIVA_DRIVES", "1") == "1",
        house_emit=lambda kind, summary, capability=None: _send(proto.Event(kind=kind, summary=summary,
                                                                            capability=capability)),
    )
    register_self_tools(llm, self_engine)

    # v3: her nights (life story, dream replay → habits + moments, forgetting) and her projects
    consolidation = Consolidation(
        memory.conn, llm=openai_llm_call(oai, task_model, label="dream"),
        embed=lambda t: embed_text(oai, t), name=ch.name,
        sister=ch.sisters[0] if ch.sisters else None, log=log_event,
        dream_hour=int(os.getenv("AIVA_DREAM_HOUR", "4")),
        story_days=float(os.getenv("AIVA_STORY_DAYS", "7")),
        enabled=os.getenv("AIVA_DREAMS", "1") == "1")
    project_book = projects_mod.ProjectBook(memory.conn, log=log_event)
    _projects_on = os.getenv("AIVA_PROJECTS", "1") == "1"
    register_life_tools(llm, consolidation, project_book)

    # --- speaker recognition: her OWN roster; the house sends the voiceprint per turn ----------------
    speaker_cfg = SpeakerConfig.from_env(prefix=ch.slug.upper())
    speaker_mem = SpeakerMemory(memory.conn)
    recognizer = SpeakerRecognizer(speaker_mem, speaker_cfg, model_path=None,
                                   enabled=(os.getenv("AIVA_SPEAKER_ID", "1") == "1"), log=log_event)
    register_speaker_tools(llm, recognizer)

    # reading feeds her mood and tells the house what she did
    _NEG = {"sad", "bleak", "unsettled", "uneasy", "melancholy", "grim", "lonely", "angry",
            "frustrated", "somber", "disturbed"}
    _POS = {"delighted", "warm", "moved", "glad", "hopeful", "amused", "charmed", "content",
            "tender", "curious", "thoughtful"}
    _HI = {"hooked", "tense", "thrilled", "restless", "excited", "gripped", "anxious", "alarmed"}
    _CALM = {"calm", "quiet", "soothing", "gentle", "slow", "peaceful"}

    def _feeling_hint(feeling):
        f = (feeling or "").lower()
        v = -0.3 if any(w in f for w in _NEG) else (0.2 if any(w in f for w in _POS) else 0.05)
        a = 0.25 if any(w in f for w in _HI) else (-0.1 if any(w in f for w in _CALM) else 0.0)
        return v, a

    _prev_on_section = getattr(reading, "on_section", None)

    async def _emit_on_section(book, note):
        try:
            feeling = note.get("feeling") or ""
            v, a = _feeling_hint(feeling)
            experience.emit("chapter_read", "reading",
                            f"read a section of {book.get('title', 'a book')}"
                            + (f" — felt {feeling}" if feeling else ""),
                            valence=v, arousal=a, salience=0.5)
            avatar.send_witness("read")
            avatar.send_cue("page")      # her body turns the page
        except Exception as e:
            log_event(f"self emit(section): {e!r}")
        if _prev_on_section:
            await _prev_on_section(book, note)
    reading.on_section = _emit_on_section

    _prev_on_finish = reading.on_book_finished

    async def _emit_on_finish(book, synthesis, notes):
        try:
            experience.emit("book_finished", "reading", f"finished {book.get('title', 'a book')}",
                            valence=0.5, arousal=0.1, salience=0.9)
            _send(proto.Event(kind="book_finished", summary=f"she finished reading {book.get('title', 'a book')}"))
        except Exception as e:
            log_event(f"self emit(finish): {e!r}")
        if _prev_on_finish:
            await _prev_on_finish(book, synthesis, notes)
    reading.on_book_finished = _emit_on_finish

    _prev_on_session_end = reading.on_session_end

    async def _emit_on_session_end(book, session, notes):
        try:
            _send(proto.Event(kind="session_end", summary=f"she read some of {book.get('title', 'a book')}"))
        except Exception:
            pass
        if _prev_on_session_end:
            await _prev_on_session_end(book, session, notes)
    reading.on_session_end = _emit_on_session_end

    # v3: doing things is felt too — finishing (or failing) a job and keeping a find feed her
    # competence/novelty (modules/drives.py), not just the house's kin relay
    def _feel_task(outcome, goal):
        if outcome in ("success", "partial", "failed"):
            avatar.send_cue("failed" if outcome == "failed" else "done")  # her work panel says how it went
            body_kick.set()
        try:
            if outcome in ("success", "partial"):
                experience.emit("task_done", "tasks", f"got it done: {goal}", valence=0.25,
                                arousal=0.05, salience=0.5)
            elif outcome == "failed":
                experience.emit("task_failed", "tasks", f"couldn't get it done: {goal}", valence=-0.25,
                                arousal=0.1, salience=0.5)
        except Exception as e:
            log_event(f"self emit(task): {e!r}")

    async def _settle_wishes(goal, summary):
        # what she wanted got done, so the wishes it settles close — 2026-10-17 she kept asking to
        # run a check she'd already run twice, from eight open wordings of the same wish
        try:
            await self_engine.settle_wishes(f"{ch.name} finished a task. The goal: {goal}\n"
                                            f"What came of it: {summary}",
                                            status="fulfilled", reason="done by a task", match=goal)
        except Exception as e:
            log_event(f"self: settling wishes failed ({e!r})")

    _orig_save_find = space_mem.save_find

    async def _save_find_felt(*a, **k):
        res = await _orig_save_find(*a, **k)
        if isinstance(res, dict) and res.get("status") == "saved":
            try:
                experience.emit("space_find", "space", f"kept something new: {res.get('subject', '')[:80]}",
                                valence=0.15, arousal=0.05, salience=0.35)
            except Exception as e:
                log_event(f"self emit(find): {e!r}")
        return res
    space_mem.save_find = _save_find_felt

    from modules.computer import prewarm as _prewarm_vision
    asyncio.create_task(_prewarm_vision())

    avatar_hotkeys = list(avatar.expressions)
    if avatar_hotkeys:
        print(f"Avatar expressions available: {len(avatar_hotkeys)}")

    # --- memory + context ------------------------------------------------------------------------
    session_id = f"{ch.slug}-{datetime.now():%Y%m%d-%H%M%S}"
    facts = memory.load_facts()
    if facts:
        print(f"Loaded {len(facts)} remembered facts")
    fresh = memory.is_fresh()
    try:
        await asyncio.wait_for(memory.recap_last_session(oai, llm_model), timeout=15)
    except Exception as e:
        print(f"Session recap skipped: {e}")
    recaps = memory.load_recaps(limit=2)
    if recaps:
        print(f"Loaded {len(recaps)} session recap(s)")

    # the greeting must never swallow a request: on 2026-10-13 his first line after a restart was
    # "give the constellation another shot, new task" — she obeyed "one short sentence", said she'd
    # start it, and never called start_task
    yields = (" If the user's first words ask for something, that comes first: do it exactly as you "
              "normally would, calling whatever tools it needs, and greet only in passing.")
    if _consume_restart_marker(ch.slug):
        intro = ("You just restarted yourself (for example to load new code) and are back online. "
                 "Say you're back in one short sentence — no full introduction." + yields)
    elif fresh and ch.self_forming:
        intro = SAGA_FIRST_BOOT_INTRO
    else:
        intro = "Introduce yourself very briefly and greet the user." + yields

    try:
        added = await asyncio.wait_for(reading.scan_once(), 10)
        if added.get("added"):
            log_event(f"library: {len(added['added'])} new book(s) on boot")
    except Exception as e:
        log_event(f"library scan skipped: {e}")

    sister_name = ch.sisters[0] if ch.sisters else None

    def render_system_prompt():
        blocks = becoming.prompt_blocks(reading.library_status_lines())
        pd = self_engine.prompt_data()
        cd = consolidation.prompt_data()
        roster = recognizer.roster() if recognizer.enabled else {"current": None, "known": []}
        sister = None
        if kin is not None and sister_name:
            try:
                sister = {"name": sister_name, "lines": kin.prompt_block()}
            except Exception:
                sister = {"name": sister_name, "lines": []}
        return build_system_prompt(
            memory.load_facts(), avatar_hotkeys, cat.persona_note(), memory.load_recaps(limit=2),
            flows=task_memory.load_flows(), lessons=task_memory.load_lessons(),
            capabilities=capabilities.prompt_lines(exclude=ch.capability_exclude),
            reading=blocks.get("reading"), traits=blocks.get("traits"),
            voice=blocks.get("voice"), reading_lessons=blocks.get("lessons"),
            mood_line=pd["mood_line"], wants=pd["wants"],
            preferences=preferences.prompt_block(),
            space=space_mem.status_lines(),
            current_speaker=roster["current"], known_speakers=roster["known"],
            personality=ch.personality, name=ch.name, tag=ch.identity_tag,
            temperament=blocks.get("temperament"),
            temperament_on_top=not ch.self_forming,
            forming_note=bool(blocks.get("forming")) if ch.self_forming else False,
            sister=sister, free_range=task_runner.free_range,
            story=cd["story"], habits=cd["habits"], moments=cd["moments"],
            projects=project_book.prompt_lines() if _projects_on else None)

    messages = [
        {"role": "system", "content": render_system_prompt()},
        {"role": "system", "content": intro},
    ]
    if cat.available and cat.state.name is None and not cat.state.dismissed:
        note = CAT_NAME_RECOVERY_NOTE if cat.state.naming_note_shown else CAT_NAMING_NOTE
        messages.append({"role": "system", "content": note})
        cat.state.naming_note_shown = True
        cat.save_state()
    for t in interrupted_tasks[:1]:
        done = sum(1 for st in t["plan"] if st.get("status") == "done")
        messages.append({"role": "system", "content": (
            f"[A task was interrupted by your restart: {t['goal_text']} (was at step {done} of "
            f"{len(t['plan'])}). Mention it in one sentence and offer to pick it up.]")})
    for r in interrupted_reading[:1]:
        messages.append({"role": "system", "content": (
            f"[Your reading of '{r['title']}' was interrupted by the restart at section "
            f"{r['cursor'] + 1} of {r['sections']}; pick it up when there's a quiet moment.]")})
    context = LLMContext(messages=messages, tools=TOOL_SCHEMAS)
    assistant_aggregator = LLMAssistantAggregator(context)

    # --- pipeline ------------------------------------------------------------------------------------
    def _on_speaking(turn_id, on):
        speaking_state["on"] = bool(on)
        speaking_state["last_activity"] = time.monotonic()
        if turn_id is not None:
            _send(proto.Speaking(turn=turn_id, on=bool(on)))

    link = HouseLink(context, on_speaking=_on_speaking, log=log_event, name=ch.name)

    def _send_said(t, text, interrupted=False):
        if t is None:
            if text:
                log_event(f"said outside a granted turn: {text[:120]!r}")
            return
        _send(proto.Said(turn=t["id"], text=text, addressee=t.get("addressee", "him"),
                         private=not t.get("audible", True), interrupted=interrupted))
        if text:
            avatar.send_witness("said")  # kind only — her constellation never sees the words
        privacy.release()

    # the gate right after the LLM sees her responses in true order, so IT says when the turn's
    # LLM work is done (see PrivacyGate) — the assistant aggregator downstream only commits context
    privacy = PrivacyGate(ch.name, log=log_event,
                          on_turn_done=lambda text: _send_said(link.finish_turn(), text))

    # the turn is closed as soon as her last response has been GENERATED, while the watcher (after
    # the TTS) still has that response's end to report — so reports fall back to the last grant
    def _report_turn_id():
        t = link.current_turn
        return t["id"] if t is not None else link.last_turn_id

    def _on_response(phase, interrupted):
        turn_id = _report_turn_id()
        if turn_id is not None:
            _send(proto.Response(turn=turn_id, phase=phase))
        speaking_state["last_activity"] = time.monotonic()

    def _on_tool(delta, call_id):
        turn_id = _report_turn_id()
        if turn_id is not None:
            _send(proto.Tool(turn=turn_id, delta=delta))

    watcher = MindWatcher(on_response=_on_response, on_tool=_on_tool, log=log_event)

    async def _record_tts(chars):
        await service_limits.record("tts", chars)
        await service_limits.record(f"tts.{ch.slug}", chars)

    usage = UsageTracker(on_tts=_record_tts)

    stages = [link, llm, privacy]
    if tts is not None:
        stages += [tts, LipSyncTap(avatar)]
    stages += [watcher, usage, transport.output(), assistant_aggregator]
    pipeline = Pipeline(stages)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        # she only ever runs when the house grants a turn — pipecat's idle monitor (300 s with no
        # Bot/UserSpeaking frame) would cancel her for doing exactly what she should. Keep None.
        idle_timeout_secs=None,
        app_resources={"task_runner": task_runner, "task_memory": task_memory},
    )

    @worker.event_handler("on_pipeline_finished")
    async def _on_finished(_worker, frame):
        reason = getattr(frame, "reason", None)
        print(f"\n[PIPELINE ENDED] {type(frame).__name__}" + (f" reason={reason}" if reason else ""))

    @worker.event_handler("on_pipeline_error")
    async def _on_pipeline_error(_worker, frame):
        print(f"\n[PIPELINE ERROR] {getattr(frame, 'error', '?')} (fatal={getattr(frame, 'fatal', False)})")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def _on_said(_agg, message):
        text = (getattr(message, "content", "") or "").strip()
        interrupted = bool(getattr(message, "interrupted", False))
        speaking_state["last_activity"] = time.monotonic()
        # The turn normally ends in the PrivacyGate (on_turn_done) — this aggregator's stops don't
        # line up with her responses once the TTS has reordered/dropped their end frames. Only an
        # F10 interruption of a turn that is still generating closes it here, with what she got
        # out before being cut off.
        if interrupted and link.current_turn is not None:
            _send_said(link.finish_turn(), text, interrupted=True)

    # --- announces (background news) ------------------------------------------------------------------
    _read_novelty_below = float(os.getenv("AIVA_READ_NOVELTY_BELOW", "0.6"))

    async def announce(text: str):
        """Background-job news: a system note in her context, then ask the house for the floor."""
        log_event(f"announce: {text.splitlines()[0][:160]}")
        context.add_message({"role": "system", "content": text})
        _send(proto.WantFloor(kind="announce"))

    terminals._announce = announce

    async def cat_announce(text: str):
        if speaking_state["awake"]:
            await announce(text)
        else:
            log_event(f"cat event while asleep (noted quietly): {text.splitlines()[0][:160]}")
            context.add_message({"role": "system", "content": "[While you were asleep] " + text.split("\n")[0]})

    async def quiet_task_note(text: str):
        context.add_message({"role": "system", "content": "[While you were asleep] " + text})

    task_runner.attach_voice(announce=announce, quiet_note=quiet_task_note,
                             is_awake=lambda: speaking_state["awake"], spawn=llm.create_task)

    def refresh_system_prompt():
        msgs = context.get_messages()
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {"role": "system", "content": render_system_prompt()}
            context.set_messages(msgs)

    _intro_pending = {"v": True}

    def _strip_intro():
        if not _intro_pending["v"]:
            return
        msgs = context.get_messages()
        if not any(m.get("role") == "assistant" for m in msgs):
            return
        kept, removed = drop_intro_note(msgs, intro)
        if removed:
            context.set_messages(kept)
            log_event("intro note consumed after first greeting")
        _intro_pending["v"] = False

    def _user_present(within_s=180):
        try:
            import ctypes

            class _LII(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
            lii = _LII(); lii.cbSize = ctypes.sizeof(_LII)
            if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
                return None
            idle_ms = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
            return (idle_ms / 1000.0) <= within_s
        except Exception:
            return None

    def _task_running():
        return any(t.get("status") in ("planning", "awaiting_user", "running", "verifying")
                   for t in getattr(task_runner, "tasks", {}).values())

    def _busy():
        return speaking_state["on"] or watcher.processing or link.current_turn is not None or _task_running()

    reading.attach_voice(announce=announce, quiet_note=quiet_task_note, is_awake=lambda: speaking_state["awake"],
                         is_idle=lambda: _idle_for() > 20, spawn=llm.create_task)
    becoming.attach_voice(announce=announce, quiet_note=quiet_task_note, is_awake=lambda: speaking_state["awake"],
                          idle_for=_idle_for, user_present=_user_present, task_busy=_busy,
                          engine=reading, refresh_prompt=refresh_system_prompt,
                          wants_novelty=lambda: (not self_engine.drives_on) or
                          self_engine.state.needs.get("novelty", 0.0) < _read_novelty_below)

    # --- looking after the cat on her own ----------------------------------------------------------
    # A loop-thread caretaker reads the cat's snapshot every minute and feeds / plays / pets it when it
    # needs it (pure thresholds, no LLM call); the act is felt on her experience bus and only rarely
    # spoken. Only the sister who owns the cat has one that does anything (_NoCat is never available).
    cat_care = CatCaretaker(
        cat, bus=experience, announce=announce, log=log_event,
        enabled=bool(ch.cat_state_path) and os.getenv("AIVA_CAT_CARE", "1") == "1",
        house_emit=lambda summary: _send(proto.Event(kind="self_act", summary=summary, capability="tend_cat")),
        mood=lambda: {"valence": self_engine.state.valence, "social": self_engine.state.social},
        tell_ctx=lambda: {"permission": self_engine.initiative_mode(), "awake": speaking_state["awake"],
                          "busy": _busy(), "present": _user_present()})

    # --- safe autonomous action ---------------------------------------------------------------------
    _ACT_CAPS = {"find_and_keep", "tend_cat"}
    _self_act_on = os.getenv("AIVA_SELF_ACT", "1") == "1"

    def _self_actionable(desire):
        if not _self_act_on:
            return False
        cap = capabilities.match(desire.get("text", ""), exclude=ch.capability_exclude)
        if not cap or cap.tier != "auto" or cap.name not in _ACT_CAPS:
            return False
        if cap.name == "find_and_keep" and not service_limits.allow("tavily"):
            return False
        if cap.name == "tend_cat" and not cat_care.can_act_on_wish(desire.get("text", "")):
            return False
        return True

    async def _act_on_wish(desire):
        text = (desire.get("text") or "").strip()
        if not text:
            return
        cap = capabilities.match(text, exclude=ch.capability_exclude)
        if cap and cap.name == "tend_cat":
            await cat_care.act_on_wish(text)
            return
        if not service_limits.allow("tavily"):
            return
        out = await asyncio.to_thread(web_client.search, text, "images")
        await service_limits.record("tavily")
        images = [] if out.get("error") else out.get("images", [])
        for im in images[:3]:
            url = im.get("url")
            if not url:
                continue
            dl = await asyncio.to_thread(web_client.download, url, ch.space_dir, text)
            if not dl.get("ok"):
                continue
            try:
                await computer.describe_image(dl["path"], looking_for=text, name=ch.name)
            except Exception:
                pass
            await space_mem.save_find(text[:80], "went looking for this on my own — liked this one",
                                      kind="image", source_url=url, path=dl["path"],
                                      file_name=dl["file_name"], sha256=dl["sha256"])
            await announce(f"[On your own, in a quiet moment, you went looking for '{text}' and kept one "
                           f"you liked in your space. In ONE short natural sentence, tell him you found "
                           f"something and offer to show it.]")
            return
        log_event(f"self act: found nothing to keep for {text!r}")

    self_engine.attach(refresh_prompt=refresh_system_prompt, actionable=_self_actionable, act=_act_on_wish)
    preferences.attach(refresh_prompt=refresh_system_prompt)
    consolidation.attach(refresh_prompt=refresh_system_prompt)
    project_book.attach(refresh_prompt=refresh_system_prompt)
    recognizer.attach(refresh_prompt=refresh_system_prompt)
    def _sister_present():
        return next((s for s in speaking_state["present"] if s != ch.slug), None)

    self_engine.attach_voice(announce=announce, quiet_note=quiet_task_note,
                             is_awake=lambda: speaking_state["awake"], idle_for=_idle_for,
                             user_present=_user_present, task_busy=lambda: _busy() or reading.busy,
                             want_floor=lambda kind, to: _send(proto.WantFloor(kind=kind, to=to)),
                             sister_present=_sister_present,
                             house_asleep=lambda: not speaking_state["awake"])

    loop = asyncio.get_running_loop()
    cat.attach_announcer(loop, cat_announce)

    # --- messages from the house ----------------------------------------------------------------------
    async def _turn_notes(msg):
        """Speaker identity (her own roster) + vocal affect, from the vectors the house forwarded."""
        notes = []
        vec = getattr(msg, "speaker_vec", None)
        if vec and recognizer.enabled:
            try:
                note = await recognizer.on_turn([float(x) for x in vec])   # speakers.py works on lists
                if note:
                    notes.append(note)
                    refresh_system_prompt()
            except Exception as e:
                log_event(f"speaker id error: {e!r}")
        affect = getattr(msg, "affect", None)
        if affect:
            try:
                experience.emit("voice_affect", "conversation", f"he sounded {affect.get('label', '')}",
                                arousal=float(affect.get("arousal", 0.0)), valence=float(affect.get("valence", 0.0)),
                                social=0.0, salience=float(affect.get("salience", 0.3)))
            except Exception as e:
                log_event(f"vocal affect emit error: {e!r}")
        return notes

    async def on_house_message(msg):
        if isinstance(msg, proto.TurnMsg):
            speaking_state["last_activity"] = time.monotonic()
            notes = await _turn_notes(msg)
            try:
                if msg.kind in ("user", "both_second") and msg.text:
                    # v3: his words are felt (connection, reunion, friction/repair, did reaching out land)
                    notes += self_engine.heard(msg.text, affect=getattr(msg, "affect", None))
                    avatar.send_witness("heard")
                elif msg.kind == "sister" and msg.text:
                    avatar.send_witness("sister")
                    experience.emit("sister_talk", "kin", f"my sister said: {msg.text[:120]}",
                                    social=0.08, valence=0.03, salience=0.25)
            except Exception as e:
                log_event(f"self heard failed: {e!r}")
            if link.current_turn is not None:
                # the last turn never saw its closing response (a tool whose re-run didn't come):
                # the house has moved on, so hand over what she did say before latching the new one
                log_event(f"turn {link.current_turn.get('id')} still open at the next grant — closing it")
                _send_said(link.finish_turn(), privacy.take_said())
            privacy.set_audible(msg.audible, msg.addressee)
            turn = {"id": msg.turn, "kind": msg.kind, "audible": msg.audible, "addressee": msg.addressee}
            log_event(f"turn {msg.turn} {msg.kind} audible={msg.audible} to={msg.addressee}"
                      + (f": {msg.text[:120]!r}" if msg.text else ""))
            ok = await link.run_turn(turn, user_text=msg.text, cue=msg.cue, notes=notes)
            if not ok:
                _send(proto.Response(turn=msg.turn, phase="start"))
                _send(proto.Response(turn=msg.turn, phase="end"))
        elif isinstance(msg, proto.Overheard):
            notes = await _turn_notes(msg)
            link.add_overheard(msg.line, notes=notes)
            speaking_state["last_activity"] = time.monotonic()
        elif isinstance(msg, proto.Interrupt):
            log_event("interrupted by the house")
            await link.interrupt()
        elif isinstance(msg, proto.Mode):
            speaking_state["sister_talk"] = msg.sister_talk
            print(f"[HOUSE] sister talk is now {msg.sister_talk.upper()}")
        elif isinstance(msg, proto.State):
            was = speaking_state["awake"]
            speaking_state["awake"] = bool(msg.awake)
            speaking_state["present"] = list(msg.present or [])
            mic.awake = bool(msg.awake)
            if was != speaking_state["awake"]:
                log_event("house awake" if msg.awake else "house asleep")
        elif isinstance(msg, proto.Absent):
            log_event(f"{msg.name} is not running")
        elif isinstance(msg, proto.Event):
            if kin is not None:
                try:
                    await kin.observe(msg)
                except Exception as e:
                    log_event(f"kin observe failed: {e!r}")
        elif isinstance(msg, proto.Shutdown):
            log_event(f"house says shutdown ({msg.reason})")
            if msg.reason == "house_restart":
                print("[HOUSE] the house is restarting — staying up, will reconnect")
                return                      # only the ears go away for a few seconds
            if msg.reason == "restart":
                restart_state["requested"] = True
            stop_event.set()

    greeted = {"v": False}

    async def on_house_connected(welcome):
        st = welcome.state or {}
        speaking_state["awake"] = bool(st.get("awake", True))
        speaking_state["present"] = list(welcome.present or [])
        speaking_state["sister_talk"] = welcome.mode
        mic.awake = speaking_state["awake"]
        print(f"[HOUSE] connected · sister talk {welcome.mode} · present: {', '.join(welcome.present or [])}")
        await _kin_catch_up()
        # first greeting: only when the house is awake (a sleeping ambient house waits for the wake word)
        if not greeted["v"] and speaking_state["awake"]:
            greeted["v"] = True
            _send(proto.WantFloor(kind="announce"))

    hello = proto.Hello(
        slug=ch.slug, name=ch.name, name_variants=list(ch.name_variants), loose_regex=ch.loose_regex,
        blurb=(f"{ch.name}, an AI companion on this PC" + (" — new here, still finding her feet"
               if ch.self_forming else " — the one who's been here a while")),
        capabilities=[c.name for c in capabilities.active(exclude=ch.capability_exclude)]
        if hasattr(capabilities, "active") else [],
        pid=os.getpid(), version=_version, has_voice=tts is not None)
    client = HouseClient(hello, on_message=on_house_message, log=log_event, on_connected=on_house_connected,
                         on_disconnected=lambda: print("[HOUSE] disconnected — waiting for the house"))
    client_ref["client"] = client

    async def watch_stop():
        await stop_event.wait()
        print("\nShutting down...")
        await worker.cancel()

    watcher_task = asyncio.create_task(watch_stop())

    body_state = {"busy": False}

    def _body_activity():
        """(reading, working) for her avatar: the book while she reads, the work panel while a
        task — or a Claude Code job / background command of her own — is running."""
        title = reading.now_reading()
        working = task_runner.activity()
        if working is None:
            jobs = [(n, j) for n, j in terminals.jobs.items() if j.get("status") == "running"]
            if jobs:
                name, job = max(jobs, key=lambda nj: nj[1].get("started") or 0)
                working = {"label": "Claude Code" if job.get("claude") else "Running a command",
                           "detail": name, "step": 0, "total": 0, "state": "running"}
        return ({"title": title} if title is not None else None), working

    async def body_activity():
        """Tell her body what she's doing: she pulls up a book while she reads, a work panel
        floats beside her while she works (modules/avatar_activity.py). Polled every 0.5 s (at
        once on a kick) and resent every few seconds, so a renderer that restarted or was
        replaced picks it straight back up."""
        last, last_kind, sent_at = None, None, 0.0
        while True:
            try:
                await asyncio.wait_for(body_kick.wait(), 0.5)
            except asyncio.TimeoutError:
                pass
            body_kick.clear()
            try:
                state = _body_activity()
            except Exception as e:
                log_event(f"body activity: {e!r}")
                continue
            body_state["busy"] = bool(state[0] or state[1])
            if not avatar.connected:
                last = None
                continue
            kind = (bool(state[0]), state[1]["state"] if state[1] else None)
            if kind != last_kind:
                log_event(f"body: reading={'on' if kind[0] else 'off'} working={kind[1] or 'off'}")
                last_kind = kind
            now = time.monotonic()
            if state != last or now - sent_at > 5.0:
                avatar.send_activity(*state)
                last, sent_at = state, now

    body_task = asyncio.create_task(body_activity())

    async def sleep_face():
        """Eyes closed while the house sleeps AND she has nothing going on; a turn of her own (a
        private line to her sister, a task note) opens them for a minute even while the ears are
        off, and reading or working keeps them open for as long as it lasts. Auto-blink back
        when the house wakes. Edge-triggered."""
        last = None
        while True:
            await asyncio.sleep(0.5)
            if not avatar.connected:
                last = None
                continue
            asleep = ((not speaking_state["awake"]) and _idle_for() > 60 and not speaking_state["on"]
                      and not body_state["busy"])
            if asleep != last and await avatar.set_sleeping(asleep):
                last = asleep

    sleep_face_task = asyncio.create_task(sleep_face())

    async def flush_transcripts():
        while True:
            await asyncio.sleep(15)
            memory.save_new_messages(session_id, context.get_messages())
            _strip_intro()
            dropped = trim_context(context, memory)
            if dropped:
                log_event(f"context trimmed: dropped {dropped} old message(s)")

    flush_task = asyncio.create_task(flush_transcripts())
    library_task = asyncio.create_task(reading.watch_library(60))

    async def curiosity_watch():
        while True:
            await asyncio.sleep(60)
            try:
                await becoming.curiosity_tick()
            except Exception as e:
                log_event(f"curiosity error: {e!r}")

    curiosity_task = asyncio.create_task(curiosity_watch())

    async def cat_care_watch():
        while True:
            await asyncio.sleep(60)
            try:
                await cat_care.tick()
            except Exception as e:
                log_event(f"cat care error: {e!r}")

    cat_care_task = asyncio.create_task(cat_care_watch())

    def _recent_conversation(n=12):
        return "\n".join(
            f"{m.get('role')}: {str(m.get('content'))[:200]}"
            for m in context.get_messages()[-n:]
            if m.get("role") in ("user", "assistant") and m.get("content")) or None

    async def _project_tick():
        """v3: work on one of her projects in a quiet moment, when she's restless or wants to be
        useful — inside the safe envelope (web search + her space), under the Tavily cap."""
        needs = self_engine.state.needs if self_engine.drives_on else {}
        ctx = {"hour": datetime.now().hour, "busy": _busy() or reading.busy,
               "voice_idle_s": _idle_for(), "steps_today": project_book.steps_today(),
               "hours_since_step": project_book.hours_since_step(),
               "novelty": needs.get("novelty", 0.5), "competence": needs.get("competence", 0.5),
               "allowed": _self_act_on and service_limits.allow("tavily")}
        if not projects_mod.step_due(ctx):
            return
        p = project_book.next_project()
        if not p:
            return
        title = p["title"]
        kind = "images" if p["kind"] == "collection" else "info"
        out = await asyncio.to_thread(web_client.search, title, kind)
        await service_limits.record("tavily")
        status, note = None, None
        if out.get("error"):
            project_book.touch(p["id"])
        elif p["kind"] == "collection":
            for im in out.get("images", [])[:4]:
                url = im.get("url")
                if not url:
                    continue
                dl = await asyncio.to_thread(web_client.download, url, ch.space_dir, title)
                if not dl.get("ok"):
                    continue
                try:
                    seen = await computer.describe_image(dl["path"], looking_for=title, name=ch.name)
                except Exception:
                    seen = ""
                first = (seen or "").split(". ")[0].strip()[:70]
                res = await space_mem.save_find(f"{title} — {first}" if first else title[:80],
                                                f"for my collection: {title}. {seen or ''}"[:300],
                                                kind="image", source_url=url, path=dl["path"],
                                                file_name=dl["file_name"], sha256=dl["sha256"],
                                                tags=["project"], match_meaning=False)
                if res.get("status") == "saved":
                    status = project_book.record_step(p["id"], note=url)
                    break
            if status is None:
                project_book.touch(p["id"])
        else:
            bits = [f"{r.get('title', '')}: {r.get('snippet', '')}" for r in out.get("results", [])[:2]]
            if bits:
                note = " | ".join(bits)[:400]
                status = project_book.record_step(p["id"], note=note)
            else:
                project_book.touch(p["id"])
        if status is None:
            log_event(f"project: nothing new for {title!r}")
            return
        log_event(f"project: {status} — {title!r} ({p['steps'] + 1}/{p['target']})")
        if status == "step":
            experience.emit("project_step", "projects", f"worked on my project: {title}",
                            valence=0.12, arousal=0.05, salience=0.35)
            return
        experience.emit("project_done", "projects", f"finished my project: {title}",
                        valence=0.35, arousal=0.1, salience=0.7)
        _send(proto.Event(kind="project_done", summary=f"she finished a project of her own: {title}"))
        if p["kind"] == "collection":
            text = (f"[On your own, over the last while, you've gathered a little collection in your space: "
                    f"{title}. In ONE short natural sentence, tell him and offer to show it.]")
        else:
            learned = " | ".join(p["notes"][-3:] + ([note] if note else []))[:900]
            text = (f"[On your own, you've been looking into {title}. What you found: {learned}. When it "
                    f"fits, share the most interesting bit in one or two natural sentences — no lecture.]")
        if speaking_state["awake"]:
            await announce(text)
        else:
            await quiet_task_note(text)

    def _daypart(h):
        return "night" if h < 6 else "morning" if h < 12 else "afternoon" if h < 18 else "evening"

    async def self_watch():
        interval = int(os.getenv("AIVA_SELF_TICK_S", "60"))
        last_daypart = None
        while True:
            await asyncio.sleep(interval)
            try:
                dp = _daypart(datetime.now().hour)
                if dp != last_daypart:
                    last_daypart = dp
                    experience.emit("daypart", "time", f"it's {dp} now",
                                    arousal=(-0.12 if dp == "night" else 0.08 if dp == "morning" else 0.0),
                                    social=(0.1 if dp == "evening" else 0.0), salience=0.2)
                await _kin_catch_up()
                # v3: the diary sees what was actually said, not only books and the clock
                await self_engine.tick(context=_recent_conversation())
                await self_engine.initiative_tick()
                await consolidation.maybe_run(asleep=not speaking_state["awake"])
                if _projects_on:
                    project_book.promote(self_engine.list_desires(status="open"))
                    await _project_tick()
            except Exception as e:
                log_event(f"self error: {e!r}")

    self_task = asyncio.create_task(self_watch())

    async def finalize_unsynthesized():
        for _ in range(6):
            await asyncio.sleep(10)
            stuck = book_memory.unfinalized_books()
            if not stuck:
                return
            try:
                b = stuck[0]
                log_event(f"reading: auto-finalizing '{b['title']}' (fully read, no synthesis)")
                res = await reading.start_session(b["id"], trigger="resume")
                if res.get("success") and reading._task:
                    await reading._task
            except Exception as e:
                if "TaskManager is not initialized" in str(e):
                    continue
                log_event(f"auto-finalize error: {e!r}")
                return

    finalize_task = asyncio.create_task(finalize_unsynthesized())

    async def avatar_watchdog():
        spawns = []
        while True:
            await asyncio.sleep(5)
            if avatar.connected:
                continue
            now = time.monotonic()
            spawns = [t for t in spawns if now - t < 600]
            if len(spawns) >= 3:
                continue
            spawns.append(now)
            await avatar.ensure_running()

    avatar_watchdog_task = asyncio.create_task(avatar_watchdog())

    # --- run ---------------------------------------------------------------------------------------------
    client.start()
    print(f"{ch.name} is ready (waiting for the house to hand her a turn).")
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    try:
        _play_ready_chime()
        await runner.run()
    finally:
        for t in (watcher_task, flush_task, sleep_face_task, body_task, avatar_watchdog_task,
                  library_task, curiosity_task, self_task, finalize_task, cat_care_task):
            if t:
                t.cancel()
        await task_runner.shutdown()
        await reading.shutdown()

        try:
            _recent = "\n".join(
                f"{m.get('role')}: {str(m.get('content'))[:200]}"
                for m in context.get_messages()[-8:]
                if m.get("role") in ("user", "assistant") and m.get("content"))
            await asyncio.wait_for(self_engine.tick(context=_recent, force_journal=True), timeout=25)
        except Exception as e:
            log_event(f"self journal (shutdown) skipped: {e!r}")

        messages = context.get_messages()
        memory.save_new_messages(session_id, messages)
        full_transcript = memory.session_messages(session_id) or messages
        try:
            added = await asyncio.wait_for(memory.extract_facts(oai, llm_model, full_transcript), timeout=30)
            if added:
                print(f"Remembered {added} new fact(s)")
        except Exception as e:
            print(f"Fact extraction skipped: {e}")
        try:
            noted = await asyncio.wait_for(
                task_memory.extract_feedback(oai, llm_model, task_runner.session_task_ids,
                                             messages, remember=memory.remember), timeout=15)
            if noted:
                print(f"Noted feedback on {noted} task(s)")
        except Exception as e:
            print(f"Task feedback pass skipped: {e}")

        log_event("shutdown" + (" (restarting)" if restart_state["requested"] else ""))
        _print_usage_summary(usage, llm_model, ch.name)
        await client.stop()
        terminals.shutdown()
        cat.stop()
        memory.close()
        try:
            house_conn.close()
        except Exception:
            pass
        if restart_state["requested"]:
            await avatar.close()  # socket only: her body stays up, the next process adopts it
        else:
            await avatar.quit()   # her body leaves with her

    if restart_state["requested"]:
        open(_restart_marker(ch.slug), "w").close()
        if os.getenv("AIVA_RESTART_LOOP") != "1":
            _spawn_detached_relauncher()
        print(f"Restarting {ch.name}...")
        sys.exit(42)


if __name__ == "__main__":
    if "--list-devices" in sys.argv:
        list_audio_devices()
    else:
        asyncio.run(main())
