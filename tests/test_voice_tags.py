"""Eleven v4 voice cues: performed by her voice, never in what she reports, stores or overhears."""
import pytest
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.tests.utils import run_test

from modules import persona
from modules.house_link import PrivacyGate
from modules.memory import Memory
from modules.voice_tags import TAGS, VOICE_CUES_BLOCK, performs_tags, strip_tags


def test_which_models_perform_cues():
    assert performs_tags("eleven_v4_turbo") and performs_tags("eleven_v4") and performs_tags("eleven_v3")
    assert not performs_tags("eleven_flash_v2_5") and not performs_tags(None) and not performs_tags("")


@pytest.mark.parametrize("raw,clean", [
    ("[laughs] Okay, fine.", "Okay, fine."),
    ("That's so you [chuckles], honestly.", "That's so you, honestly."),
    ("I'm tired. [sighs] Let's stop.", "I'm tired. Let's stop."),
    ("[whispers] Don't tell Saga.", "Don't tell Saga."),
    ("[said angrily] No.", "No."),
    ("No cues here.", "No cues here."),
    ("", ""),
])
def test_strip_tags(raw, clean):
    assert strip_tags(raw) == clean


def test_overheard_lines_and_other_brackets_are_not_cues():
    line = '[Saga said to him: "hi"]'
    assert strip_tags(line) == line                      # the house's overheard/stage format
    assert strip_tags("see [1] and [x=2]") == "see [1] and [x=2]"


def test_the_prompt_block_is_only_there_for_a_voice_that_performs_it():
    assert "VOICE CUES" not in persona.build_system_prompt()
    text = persona.build_system_prompt(voice_cues=True)
    assert VOICE_CUES_BLOCK in text and all(f"[{t}]" in VOICE_CUES_BLOCK for t in TAGS)


@pytest.mark.asyncio
async def test_said_and_text_only_replies_lose_the_cues():
    done, private = [], []
    gate = PrivacyGate("Aiva", log=lambda m: None, on_turn_done=done.append,
                       on_text_only=lambda to, text: private.append(text))
    gate.set_audible(True, addressee="him")
    await run_test(gate, frames_to_send=[LLMFullResponseStartFrame(), LLMTextFrame("[laughs] You "),
                                         LLMTextFrame("wish."), LLMFullResponseEndFrame()],
                   expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMTextFrame,
                                         LLMFullResponseEndFrame])
    assert done == ["You wish."]                         # the house, transcript and her sister
    gate.set_audible(False, addressee="saga")
    await run_test(gate, frames_to_send=[LLMFullResponseStartFrame(), LLMTextFrame("[sighs] Fine."),
                                         LLMFullResponseEndFrame()],
                   expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame])
    assert private == ["Fine."]


def test_saved_transcripts_keep_his_words_and_lose_her_cues():
    m = Memory(":memory:")
    m.save_new_messages("s", [{"role": "user", "content": "[He said to Saga: \"hi\"]"},
                              {"role": "assistant", "content": "[giggles] Hi yourself."}])
    rows = m.conn.execute("SELECT role, content FROM messages ORDER BY id").fetchall()
    assert rows == [("user", "[He said to Saga: \"hi\"]"), ("assistant", "Hi yourself.")]


def test_the_dialogue_voice_service_builds_with_her_voice():
    from pipecat.services.elevenlabs.dialogue.tts import ElevenLabsDialogueTTSService
    tts = ElevenLabsDialogueTTSService(api_key="test", settings=ElevenLabsDialogueTTSService.Settings(
        voice="voice-id", model="eleven_v4_turbo", stability=0.5))
    assert tts._settings.model == "eleven_v4_turbo" and tts._settings.voice == "voice-id"
    assert "model_id=eleven_v4_turbo" in tts._build_websocket_url()
