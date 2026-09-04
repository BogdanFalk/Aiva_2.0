# Aiva 2.0

A voice AI companion for Windows with her own Live2D desktop avatar. Talk to her, and she talks
back — while launching apps, managing files, checking the weather, and emoting through her avatar.

Rebuilt in 2026 as a fully streaming pipeline:

```
mic ─► Silero VAD + smart-turn v3 (local; tolerates stutters and "uhhmm")
    ─► Deepgram Nova-3 streaming STT
    ─► LLM with native tool calling (gpt-4.1-mini, streamed)
    ─► ElevenLabs Flash v2.5 streaming TTS (Aiva's voice)
    ─► speakers  (+ RMS envelope ─► her Live2D avatar's mouth, in-process)
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
- **Ctrl+Alt+R** — restart her (her body stays up); **Ctrl+Alt+X** — quit
  (prints a session cost summary, saves the conversation, and distills new
  long-term facts about you)
- Speakers are fine: while Aiva talks, her ears are off (audio is gated before
  STT), so she can't hear or interrupt herself. Voice barge-in is therefore
  disabled — use F10 to cut her off.

`python launcher.py` additionally boots OBS (unless `--no-obs`). Aiva starts —
or adopts — her own avatar renderer herself.

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

**Verified wakes.** A wake-word hit alone is only a *candidate*: the listener
keeps the last 2.5 s of audio (the wake phrase itself), hands it to Deepgram
ahead of the live mic, and she only truly wakes if that first transcript
contains her name — otherwise she drops it and goes straight back to sleep
without reacting. This is what stops TV/game dialogue and other people's
speech waking her (the custom model fires on those with the same scores as a
real "hey Aiva"). Knobs: `AIVA_WAKE_VERIFY=0` disables the check,
`AIVA_WAKE_VERIFY_SECS` (default 5) is how long she waits for speech, and
`AIVA_NAME_VARIANTS=eyva,ayvah` adds spellings Deepgram uses for her name.
The log shows `wake candidate`, `AWAKE — wake word confirmed`, and
`wake rejected — ...` lines so you can see what got filtered.

## Reading books (a library, and being changed by it)

Drop `.epub`, `.txt`, `.md`, or `.pdf` files into `aiva/library/` and Aiva
notices them, tells you once (with a glad word if it's one she asked for), and
can read them. Reading happens in the background, a few chapters per session:
she studies each section — what happens, how it's written, what's left unsaid —
keeps notes with short exact quotes, and when she finishes a book she forms an
opinion. Say "read a bit of X", "read it all now", or "continue the book"; ask
"what are you reading" or "what did you think of chapter two" and she answers
from her notes (`recall_reading`), quoting at most one line, word for word.

**It changes her, within limits.** A finished book can leave her a lesson she
applies, a small shift in what she values, or a turn of phrase — always
*added* to who she is, never contradicting her baseline personality, capped and
fading if unused. She mentions a new trait once, when she finishes the book,
and never again as news. If you dislike something she picked up, say "drop
that" and it's gone for good (`drop_trait`). Say "you can read whenever you
like" and she'll read quietly when you're away (`reading_permission`);
otherwise she asks first, and only when you're at the PC but it's gone quiet.

Study and synthesis run on their own models (`AIVA_STUDY_MODEL`,
`AIVA_STUDY_SYNTH_MODEL` — luna and terra by default) so they never eat the
voice model's rate limit; a whole novel costs roughly ten to twenty-five cents,
capped at `AIVA_BOOK_MAX_COST_USD`. `python -m modules.becoming --show` prints
the traits, lessons, and turns of phrase she's picked up. Offline tests:
`.venv\Scripts\python -m pytest tests -q`; a semi-live read:
`cd aiva && ..\.venv\Scripts\python ..\tests\e2e_reading.py <book> --all`.

## Multi-step tasks (the task runner)

Anything with ordered steps — "set up my stream", "clean up the downloads
folder", "pull the repo and run the tests" — she hands to her **task runner**
(`aiva/modules/tasks.py`), which plans it, works through the steps in the
background with the same tools she uses by voice, checks each step, repairs
what didn't take, and reports back in a sentence. You can chat with her while
it runs, ask "how's it going", or say "stop".

**How autonomous she is depends on experience** (`aiva/modules/task_memory.py`):
if she has done that *kind* of task before and you never told her she got it
wrong, she just does it — one short line naming what she's about to touch, then
go. If it's new to her, or you corrected her on it, or it deletes, moves, sends
or pays for anything, she states the plan in one sentence and waits for your
yes. Shell commands and handing code to Claude Code always wait for a yes.
When a new task has two genuinely different approaches (UI vs terminal vs
Claude Code, quick vs thorough) she offers both and asks which you prefer —
and after three consistent picks she remembers your preference and stops
asking. She grades Claude Code's work (diff, tests) and tells you honestly
whether it did well, got partway, or botched it.

Tasks that go well twice become **flows** she offers to keep ("want me to
remember that as a flow called streaming setup?"); "run my streaming setup"
then replays it. Say "no, that's wrong" or "next time do X first" right after
a task and she records it against that task — the next time it comes up she
confirms first and says what she'll do differently.

Knobs: `AIVA_TASK_MODEL` (planner/verifier model; `.env` sets gpt-4.1-mini so task
calls don't eat the voice model's tokens-per-minute allowance — see below).
Per-task safety limits live in `tasks.Limits` (12 steps, 40 tool calls, 60 LLM
calls, 20 minutes, $1.50). Look for `task t-…` lines in `aiva/logs` to see
plans, familiarity scores and decisions. Offline tests:
`.venv\Scripts\python -m pytest tests -q`.

## Models

`AIVA_LLM_MODEL` (her voice), `AIVA_TASK_MODEL` (task planner/verifier) and
`AIVA_VISION_MODEL` (screen vision) all default to **gpt-5.6-luna** as of
August 2026: OpenAI's fast/cheap tier — sub-second first token on tool-calling
turns, $0.20 / $1.20 per million tokens, and 500k tokens-per-minute even on the
entry usage tier (gpt-4.1 gets 30k). `gpt-5.6-terra` is the step up if you want
more brains at the same limits; `gpt-4.1` still works if you ever want to
compare. The gpt-5 line needs `max_completion_tokens` and `reasoning_effort:
none` on Chat Completions — the code handles that automatically per model
(`modules/utilities.py: llm_kwargs`), so switching is just the env var.

## Rate limits (OpenAI tokens per minute)

Every voice turn re-sends her whole context plus all tool schemas (roughly
9–10k tokens before any conversation), so on OpenAI's entry tier (30k TPM for
gpt-4.1) a few quick turns plus a task in the background can hit a 429. What's
in place: the OpenAI clients retry rate limits with backoff
(`AIVA_LLM_MAX_RETRIES`, default 5) instead of dropping the turn; task planning,
verification and screen vision run on `gpt-4.1-mini` (`AIVA_TASK_MODEL`,
`AIVA_VISION_MODEL`), which has its own, far larger allowance; and the live
context is trimmed to `AIVA_CONTEXT_BUDGET_CHARS` (24000) so long sessions
don't grow without bound. The lasting fix is a higher usage tier on the
OpenAI account (tier 2 lifts gpt-4.1 to 450k TPM) — then point the side-call
models back at gpt-4.1 if you want.

## Desktop avatar (self-owned Live2D renderer)

Aiva renders her own body: `modules/avatar_renderer.py` is a separate process
(live2d-py + PyQt5) drawing the Live2D model in a transparent, click-through,
always-on-top window. No VTube Studio, no Spout overlay, no VB-Cable — her TTS
audio goes straight to your speakers, and a pipeline tap streams its loudness
to the renderer for lip-sync.

- Model assets live in `aiva/assets/avatar/ariu/` (licensed content, not committed).
- The renderer outlives Aiva's self-restarts; a rebooted Aiva adopts it via
  `aiva/avatar_state.json`. It dies with her on a real quit.
- Demo it standalone (never starts Aiva): `cd aiva && python -m modules.avatar_renderer --demo`
- Lip-sync is two-channel: loudness drives how open the mouth is, spectral
  brightness drives its shape (`ParamMouthForm`) — dark vowels (oo/oh) purse it,
  bright ones (ee/eh) widen it, so she appears to mouth phoneme shapes.
- Env knobs (all optional): `AIVA_AVATAR_MODEL`, `AIVA_LIPSYNC_GAIN` (default 2.5),
  `AIVA_LIPSYNC_FORM` (viseme shape strength, default 1.0; 0 disables),
  `AIVA_AVATAR_FPS` (32), `AIVA_AVATAR_SWAY` (1).

## Shell command safety

Aiva can stage shell commands but **cannot execute them without your verbal
confirmation** — she'll tell you what the command does and ask first. Every staged and
executed command is printed to the console.

## Project layout

```
aiva/main.py                 pipeline assembly + hotkeys
aiva/launcher.py             boots OBS (optional), then Aiva
aiva/modules/persona.py      Aiva's personality / system prompt
aiva/modules/tools.py        native LLM tool schemas + handlers
aiva/modules/avatar_renderer.py  her body: Live2D renderer process (live2d-py + PyQt5)
aiva/modules/avatar.py       Aiva-side client (spawn/adopt renderer, lip-sync feed)
aiva/modules/avatar_ipc.py   shared IPC protocol + state file helpers
aiva/modules/app_launcher.py launch Windows apps (app_paths.json + registry)
aiva/modules/file_operations.py  create/open/list files (fenced to your user profile)
aiva/modules/utilities.py    time/date/weather
aiva/modules/memory.py       session transcripts + long-term facts (SQLite)
```

## Roadmap

- [x] Phase 0 — repo hygiene
- [x] Phase 1 — streaming pipeline core, native tool calling, hotkeys
- [x] Phase 2 — self-hearing gate for speaker use, F10 interrupt, pipecat 1.x API migration
  - [x] VB-Cable lip-sync routing into VTube Studio + expression hotkeys (queried from the live model at startup)
- [x] Phase 3 — long-term memory (facts table, injected into the prompt)
- [x] Phase 4 — wake word + ambient mode (placeholder "Hey Jarvis"; custom "Aiva" model pending)
- [x] Phase 5 — session cost telemetry
- [x] Phase 6 — self-owned Live2D avatar renderer (dropped VTube Studio, Spout overlay, VB-Cable)
  - [ ] packaging / autostart
