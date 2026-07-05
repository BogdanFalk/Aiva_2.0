# Aiva 2.0

A voice AI companion for Windows with a VTube Studio avatar. Talk to her, and she talks back —
while launching apps, managing files, checking the weather, and emoting through her avatar.

Rebuilt in 2026 as a fully streaming pipeline:

```
mic ─► Silero VAD + smart-turn v3 (local; tolerates stutters and "uhhmm")
    ─► Deepgram Nova-3 streaming STT
    ─► LLM with native tool calling (gpt-4.1-mini, streamed)
    ─► ElevenLabs Flash v2.5 streaming TTS (Aiva's voice)
    ─► speakers  (+ VB-Audio Cable ─► VTube Studio lip-sync)
```

Voice-to-voice latency is ~1 s (the old blocking loop was 8–15 s).

## Setup

1. **Python 3.11** (`pyenv-win` manages `.python-version`), then:
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
2. **API keys** in `.env` at the repo root:
   ```
   OPENAI_API_KEY=...
   ELEVENLABS_API_KEY=...
   ELEVENLABS_VOICE_ID=...       # Aiva's voice
   DEEPGRAM_API_KEY=...          # free at console.deepgram.com ($200 credit)
   OPENWEATHER_API_KEY=...       # optional, for weather
   DEFAULT_CITY=...              # optional
   ```
3. Optional overrides in `.env`: `AIVA_LLM_MODEL` (default `gpt-4.1-mini`),
   `ELEVENLABS_MODEL` (default `eleven_flash_v2_5`),
   `AIVA_INPUT_DEVICE_INDEX` / `AIVA_OUTPUT_DEVICE_INDEX` (see below).

## Run

```
cd aiva
..\.venv\Scripts\python main.py
```

- **F9** — mute/unmute the mic
- **F10** — interrupt Aiva mid-sentence
- **Esc** — quit (prints a session cost summary, saves the conversation, and
  distills new long-term facts about you)
- Speakers are fine: while Aiva talks, her ears are off (audio is gated before
  STT), so she can't hear or interrupt herself. Voice barge-in is therefore
  disabled — use F10 to cut her off.

`python launcher.py` additionally boots VTube Studio (and OBS unless `--no-obs`).

## Memory

Aiva keeps `aiva/memory.db`: full transcripts per session, plus a `facts` table
of durable things she's learned about you (extracted by a cheap LLM pass at
shutdown). Facts are injected into her system prompt at startup, so she
remembers you across restarts.

## Ambient mode (wake word)

Set `AIVA_WAKE_WORD=1` in `.env` and she starts asleep: nothing streams to
Deepgram (idle time costs nothing) until the wake word — currently the
pretrained **"Hey Jarvis"** model from openWakeWord as a placeholder. After
`AIVA_IDLE_TIMEOUT` seconds (default 60) without conversation she dozes off
again.

To make her answer to "Aiva" instead: train a custom openWakeWord model (their
[automatic training notebook](https://github.com/dscripka/openWakeWord#training-new-models)
takes ~an hour on Colab), drop the `.onnx` file in the repo, and set
`AIVA_WAKE_MODEL=path\to\aiva.onnx`.

## Audio routing for avatar lip-sync (VTube Studio)

1. Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (free).
2. `python main.py --list-devices` and set `AIVA_OUTPUT_DEVICE_INDEX` to the
   **CABLE Input** device index in `.env`.
3. In VTube Studio: microphone = **CABLE Output**, enable voice-based lip-sync.
4. To hear Aiva yourself: Windows Sound → Recording → CABLE Output → Properties →
   Listen → "Listen to this device" → your headphones (or monitor via OBS).

## Shell command safety

Aiva can stage shell commands but **cannot execute them without your verbal
confirmation** — she'll tell you what the command does and ask first. Every staged and
executed command is printed to the console.

## Project layout

```
aiva/main.py                 pipeline assembly + hotkeys
aiva/launcher.py             boots VTube Studio (+ OBS), then Aiva
aiva/modules/persona.py      Aiva's personality / system prompt
aiva/modules/tools.py        native LLM tool schemas + handlers
aiva/modules/app_launcher.py launch Windows apps (app_paths.json + registry)
aiva/modules/file_operations.py  create/open/list files (fenced to your user profile)
aiva/modules/utilities.py    time/date/weather
aiva/modules/vtube_studio.py pyvts wrapper (expressions, model movement)
aiva/modules/memory.py       session transcripts + long-term facts (SQLite)
```

## Roadmap

- [x] Phase 0 — repo hygiene
- [x] Phase 1 — streaming pipeline core, native tool calling, hotkeys
- [x] Phase 2 — self-hearing gate for speaker use, F10 interrupt, pipecat 1.x API migration
  - [ ] VB-Cable lip-sync routing into VTube Studio (needs VB-Cable installed)
- [x] Phase 3 — long-term memory (facts table, injected into the prompt)
- [x] Phase 4 — wake word + ambient mode (placeholder "Hey Jarvis"; custom "Aiva" model pending)
- [x] Phase 5 — session cost telemetry
  - [ ] packaging / autostart
