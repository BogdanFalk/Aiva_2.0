"""Aiva — streaming voice pipeline (pipecat 1.x).

mic -> Deepgram streaming STT -> Silero VAD + local smart-turn (in the user
aggregator) -> LLM (native tool calling, streamed) -> ElevenLabs Flash TTS
-> speaker.

Hotkeys: F9 toggles the mic mute, Esc quits.
Run `python main.py --list-devices` to see audio device indices for
AIVA_INPUT_DEVICE_INDEX / AIVA_OUTPUT_DEVICE_INDEX (VB-Cable routing).
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, InputAudioRawFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from modules.persona import build_system_prompt
from modules.tools import TOOL_SCHEMAS, register_tools
from modules.vtube_studio import VTubeStudio


class MicGate(FrameProcessor):
    """Mute gate right after the transport input.

    While muted, incoming audio is replaced with silence before it reaches
    STT and the VAD, so nothing is transcribed and nothing interrupts.
    """

    def __init__(self):
        super().__init__()
        self.muted = False

    def toggle(self):
        self.muted = not self.muted
        print(f"\n[MIC {'MUTED' if self.muted else 'LIVE'}] (F9 to toggle)")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self.muted and isinstance(frame, InputAudioRawFrame):
            frame = InputAudioRawFrame(
                audio=b"\x00" * len(frame.audio),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
        await self.push_frame(frame, direction)


def list_audio_devices():
    import pyaudio

    pa = pyaudio.PyAudio()
    print("index | in/out channels | name")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"{i:5d} | in:{info['maxInputChannels']:2d} out:{info['maxOutputChannels']:2d} | {info['name']}")
    pa.terminate()


def _require_env(*names):
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        print("Missing required .env keys: " + ", ".join(missing))
        if "DEEPGRAM_API_KEY" in missing:
            print("  -> Get a free Deepgram key (comes with $200 credit) at https://console.deepgram.com/signup")
        sys.exit(1)


async def main():
    _require_env("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "DEEPGRAM_API_KEY")

    # --- transport & services ----------------------------------------------
    def _device_index(env_name):
        value = os.getenv(env_name)
        return int(value) if value else None

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=_device_index("AIVA_INPUT_DEVICE_INDEX"),
            output_device_index=_device_index("AIVA_OUTPUT_DEVICE_INDEX"),
        )
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-3",
            language="en-US",
            smart_format=True,
        ),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("AIVA_LLM_MODEL", "gpt-4.1-mini"),
        params=OpenAILLMService.InputParams(
            temperature=0.7,
            max_completion_tokens=300,
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        model=os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
    )

    # --- avatar + tools ------------------------------------------------------
    vtube = VTubeStudio()
    await vtube.connect()  # non-fatal if VTube Studio isn't running
    register_tools(llm, vtube)

    # --- context: system prompt + startup greeting ---------------------------
    context = LLMContext(
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "system", "content": "Introduce yourself very briefly and greet the user."},
        ],
        tools=TOOL_SCHEMAS,
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                # keep default VAD start strategy (with interruptions);
                # stop turns on the semantic smart-turn model so stutters
                # and "uhhmm" pauses don't cut the user off
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3()
                    )
                ],
            ),
        ),
    )

    # --- pipeline -------------------------------------------------------------
    mic_gate = MicGate()

    pipeline = Pipeline([
        transport.input(),
        mic_gate,
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(pipeline, params=PipelineParams())

    # --- hotkeys ---------------------------------------------------------------
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    try:
        import keyboard

        keyboard.add_hotkey("f9", lambda: loop.call_soon_threadsafe(mic_gate.toggle))
        keyboard.add_hotkey("esc", lambda: loop.call_soon_threadsafe(stop_event.set))
        print("Hotkeys: F9 = mute/unmute mic, Esc = quit")
    except Exception as e:
        print(f"Global hotkeys unavailable ({e}); use Ctrl+C to quit")

    async def watch_stop():
        await stop_event.wait()
        print("\nShutting down...")
        await task.cancel()

    watcher = asyncio.create_task(watch_stop())

    # --- run --------------------------------------------------------------------
    await task.queue_frames([LLMRunFrame()])

    print("Aiva is listening. Just talk.")
    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    finally:
        watcher.cancel()
        await vtube.close()


if __name__ == "__main__":
    if "--list-devices" in sys.argv:
        list_audio_devices()
    else:
        asyncio.run(main())
