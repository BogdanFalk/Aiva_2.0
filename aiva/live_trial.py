"""GPT-Live trial: talk to one sister in true full duplex — she hears you while she talks and you can
cut in any time — to feel what OpenAI's gpt-live-1 is like with HER persona and memories, side by
side with the house rather than instead of it.

STOP THE HOUSE AND BOTH SISTERS FIRST: the trial owns the microphone and refuses to start while the
house runs. USE HEADPHONES: there is no echo cancellation here, and through speakers she would hear
herself and cut herself off (the house avoids that by muting the mic while a sister speaks, which a
full-duplex model can't do).

It is a sandbox. Her persona and her long-term facts are read from her own db (`mode=ro`) into the
session's instructions; nothing is written back — no transcript, no new facts, no mood. Work she
delegates goes to an OpenAI-hosted backend (`ResponsesDelegation`, `AIVA_LIVE_BACKEND_MODEL`,
default gpt-6-luna) with hosted web search, so she can look things up while she keeps talking. Her
house tools, tasks and her sister aren't here, and she's told so.

Voice: a preset (`--voice marin`, `cedar`, …) or one designed from a description
(`--design "a warm, playful adult woman's voice, ..."`): free to create, kept by OpenAI (up to 100
per organization), usable only with GPT-Live; its id is printed so `--voice <id>` reuses it.
Cost: $0.05 a minute of session time, billed per second, plus backend tokens; printed at the end.
`--minutes` caps the session (default 10).

  cd aiva && ..\\.venv\\Scripts\\python live_trial.py --who aiva --voice marin
  cd aiva && ..\\.venv\\Scripts\\python live_trial.py --who saga --design "..."
  cd aiva && ..\\.venv\\Scripts\\python live_trial.py --check      # config + house check, no session
"""

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modules.persona import build_system_prompt, named  # noqa: E402

PRICE_PER_MINUTE = 0.05
VOICES_URL = "https://api.openai.com/v1/audio/voices"
PRESET_VOICES = ("marin", "cedar")

TRIAL_NOTE_T = (
    "THIS IS A TRIAL, NOT YOUR USUAL HOUSE — he is trying a new full-duplex voice model with you: "
    "you can hear him while you speak, and he can talk over you any time; when he does, let him and "
    "pick up naturally, like in a real conversation. Short spoken sentences. Your usual tools, tasks, "
    "your sister and your memory-saving aren't here; you can still look things up, and that happens "
    "while you keep talking. If he asks for something you can't do here, say it's just the trial and "
    "you'll do it in the house. Nothing from this conversation is saved, so don't promise to remember "
    "it. You're still {name}.")


def house_running(state_path):
    """The house's pid when its state file names a live process, else None."""
    try:
        with open(state_path, encoding="utf-8") as f:
            pid = int(json.load(f).get("pid") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    if pid <= 0:
        return None
    from modules.avatar_ipc import pid_alive
    return pid if pid_alive(pid) else None


def load_facts(db_path, limit=200):
    """Her long-term facts about him (not about her sister), read-only."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [r[0] for r in conn.execute(
            "SELECT fact FROM facts WHERE COALESCE(category, '') != 'sister' ORDER BY created_ts LIMIT ?",
            (limit,))]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def build_instructions(name, personality, tag, facts, voice_description=None):
    text = build_system_prompt(facts or None, personality=personality, name=name, tag=tag)
    text += "\n\n" + named(TRIAL_NOTE_T, name)
    if voice_description:
        # the Live API asks for the designed voice's description in the instructions too
        text += f"\n\nYOUR VOICE in this trial: {voice_description.strip()}"
    return text


def session_cost(seconds):
    return round(max(0.0, seconds) / 60.0 * PRICE_PER_MINUTE, 4)


def design_voice(api_key, name, description, post=None):
    """Create a GPT-Live voice from a text description → its id."""
    import requests
    post = post or requests.post
    r = post(VOICES_URL, headers={"Authorization": f"Bearer {api_key}"},
             json={"type": "prompt", "name": name, "prompt": description[:2000]}, timeout=60)
    if getattr(r, "status_code", 200) >= 400:
        raise RuntimeError(f"voice design failed: HTTP {r.status_code}: {str(getattr(r, 'text', ''))[:300]}")
    data = r.json()
    vid = data.get("id") or (data.get("voice") or {}).get("id")
    if not vid:
        raise RuntimeError(f"voice design returned no id: {str(data)[:300]}")
    return vid


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Full-duplex GPT-Live trial with one sister (stop the house first; headphones).")
    ap.add_argument("--who", default="aiva", choices=("aiva", "saga"))
    ap.add_argument("--voice", default=None, help="a preset (marin, cedar) or a designed voice id")
    ap.add_argument("--design", default=None, help="describe a voice to create for this trial")
    ap.add_argument("--minutes", type=float, default=10.0, help="end the session after this long")
    ap.add_argument("--check", action="store_true", help="check config and the house, open no session")
    return ap.parse_args(argv)


def build_service(api_key, instructions, voice, backend_model):
    """The Live service: her instructions, the voice, an OpenAI-hosted backend with web search."""
    from pipecat.services.openai.live.llm import OpenAILiveLLMService
    from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
    return OpenAILiveLLMService(
        api_key=api_key,
        settings=OpenAILiveLLMService.Settings(system_instruction=instructions, voice=voice or None),
        delegation=OpenAILiveLLMService.ResponsesDelegation(
            settings=OpenAIResponsesLLMService.Settings(
                model=backend_model, extra={"tools": [{"type": "web_search"}]})),
    )


async def run(args):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(_HERE), ".env"), override=True)
    from modules.character import Character
    from modules.ear import device_index, native_rate

    ch = Character.from_env(args.who)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set in .env")
        return 2
    pid = house_running(os.path.join(_HERE, "house_state.json"))
    if pid:
        print(f"The house is running (pid {pid}). Stop the house and both sisters first: the trial needs the mic.")
        return 3
    facts = load_facts(ch.db_path)
    voice, described = args.voice, None
    if args.design:
        described = args.design
        if not args.check:
            voice = design_voice(api_key, f"{ch.name} trial {time.strftime('%Y%m%d-%H%M')}", described)
            print(f"Designed a voice for this trial: {voice}  (reuse it with --voice {voice})")
    backend_model = os.getenv("AIVA_LIVE_BACKEND_MODEL") or "gpt-6-luna"
    instructions = build_instructions(ch.name, ch.personality, ch.identity_tag, facts, described)
    mic_index = device_index("AIVA_INPUT_DEVICE_INDEX", "AIVA_INPUT_DEVICE_NAME", False)
    out_index = device_index("AIVA_OUTPUT_DEVICE_INDEX", "AIVA_OUTPUT_DEVICE_NAME", True)
    print(f"{ch.name}: {len(facts)} facts, voice={voice or 'API default'}, backend={backend_model}, "
          f"mic={mic_index} @ {native_rate(mic_index)} Hz, speakers={out_index}, cap {args.minutes:g} min")
    if args.check:
        build_service(api_key, instructions, voice, backend_model)
        print("Check OK: config loads, the house is off. Run without --check to talk (headphones!).")
        return 0

    from pipecat.frames.frames import EndFrame, LLMRunFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMAssistantAggregator, LLMUserAggregator, LLMUserAggregatorParams)
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
    from pipecat.workers.runner import WorkerRunner

    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=True, audio_out_enabled=True, audio_in_sample_rate=native_rate(mic_index),
        input_device_index=mic_index, output_device_index=out_index))
    llm = build_service(api_key, instructions, voice, backend_model)
    context = LLMContext()                             # instructions ride on the service settings
    user = LLMUserAggregator(context, params=LLMUserAggregatorParams(
        vad_analyzer=None, user_turn_strategies=ExternalUserTurnStrategies(enable_interruptions=False)))
    assistant = LLMAssistantAggregator(context)
    worker = PipelineWorker(Pipeline([transport.input(), user, llm, transport.output(), assistant]),
                            params=PipelineParams(enable_metrics=True))
    started = time.monotonic()

    async def cap():
        await asyncio.sleep(args.minutes * 60.0)
        print(f"\n{args.minutes:g} minutes are up, ending the trial.")
        await worker.queue_frame(EndFrame())

    await worker.queue_frame(LLMRunFrame())           # the session opens on the first context frame
    cap_task = asyncio.create_task(cap())
    print(f"Talking to {ch.name} in full duplex. Ctrl+C ends it.")
    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        cap_task.cancel()
        secs = time.monotonic() - started
        print(f"\nTrial over: {secs / 60:.1f} min of session ≈ ${session_cost(secs):.2f} "
              "(+ backend tokens). Nothing was saved to her memory.")
    return 0


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
