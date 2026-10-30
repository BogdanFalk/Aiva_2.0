# Aiva 2.0

A voice AI companion for Windows — two of them: **Aiva** and her sister **Saga**, each with her own
Live2D desktop avatar. Talk to them, and they talk back — while launching apps, managing files,
checking the weather, and emoting through their avatars.

Rebuilt in 2026 as a fully streaming pipeline, split across three processes: the **house** (the one
microphone and ear) and one process per sister:

```
house:   mic ─► Silero VAD + smart-turn v3 (local; tolerates stutters and "uhhmm")
             ─► Deepgram Nova-3 streaming STT ─► the floor (who answers) ─► loopback TCP
sister:  ─► LLM with native tool calling (streamed; AIVA_LLM_MODEL)
         ─► ElevenLabs Flash v2.5 streaming TTS (her own voice)
         ─► speakers  (+ RMS envelope ─► her avatar renderer process, for lip-sync)
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
   SAGA_ELEVENLABS_VOICE_ID=...  # optional, Saga's voice (text-only without one)
   TAVILY_API_KEY=...            # optional, web search for both sisters
   NTFY_TOPIC=...                # optional, phone alerts when a spend cap is near
   ```
3. Optional overrides in `.env`: `AIVA_LLM_MODEL` (default `gpt-5.6-luna`; the sisters run
   `gpt-6-luna`), `ELEVENLABS_MODEL` (default `eleven_flash_v2_5`; `eleven_v4_turbo` gives them
   the voice that can laugh, sigh and whisper),
   `AIVA_INPUT_DEVICE_INDEX` / `AIVA_OUTPUT_DEVICE_INDEX` (see below). Every knob, with its
   default, is also listed and editable on the dashboard's **Settings** tab (config is read at
   boot, so it tells you what to restart).

## Run

Three consoles: the **house** (one microphone, wake words, hotkeys, the floor) and one per sister.

```
start_house.bat      the ear + the floor (start this first; the others start it if it's down)
start_aiva.bat       Aiva, in her own process
start_saga.bat       Saga, in her own process
```

By hand: `cd aiva && ..\.venv\Scripts\python house.py`, then `..\.venv\Scripts\python main.py
--character aiva` (or `saga`). A sister started without a house waits and reconnects.

Hotkeys (they belong to the house):

- **F9** — mute/deafen the mic
- **F10** — interrupt whoever is speaking
- **Ctrl+Alt+P** — flip sister talk between **private** (text only, default) and **audible**
- **Ctrl+Alt+R** — restart the sisters (their bodies stay up); **Ctrl+Alt+H** — restart the house
  only (the sisters stay up and reconnect); **Ctrl+Alt+X** — quit everything (each sister prints
  her session cost summary, saves her conversation, and distills new long-term facts)
- By voice: "restart yourself" restarts that sister (new code, same body, she reconnects and says
  she's back); "restart the house" restarts only the shared ears (`restart_house`), which go deaf
  for ten to twenty seconds and then need the wake word again; "let me hear you two" / "keep it
  between you" flips sister talk (`set_sister_talk`, no argument = tell me the current mode).
- By mouse: the dashboard (http://127.0.0.1:8760) has a **Controls** strip with the same actions as
  buttons — mute, interrupt, sister talk, sleep/wake, Start/Restart/Quit per sister, Restart house,
  Quit everything. It talks to the house over its local socket and is loopback-only. Without a house the sisters keep
  working on their own things but nobody can hear you — starting any sister, or asking one to call
  the other, brings the house back up.
- Speakers are fine: while a sister talks, the ears are off (audio is gated before STT from the
  moment she is granted the floor), so nobody hears herself. Voice barge-in is therefore disabled —
  use F10 to cut her off.

`python launcher.py --character aiva` additionally boots OBS (unless `--no-obs`). Each sister
starts — or adopts — her own avatar renderer herself.

## Two sisters: Aiva and Saga

Aiva has a sister, **Saga**. They are two people, not one voice:

- **Own cores.** Each has her own database (`aiva/memory.db`, `aiva/saga.db`), diary, mood, wishes,
  tastes, traits, space folder, task memory, voice roster and avatar — and her own console, so you
  can stop one and keep the other. Nothing about you is shared: Saga learns your voice herself and
  asks who you are the first time.
- **Talking to them.** Say a name and that sister answers; say "you two" and both do, in turn; say
  nothing and the house's arbiter picks who you meant (topic and who spoke last). The other sister
  always overhears what was said, so both know who you're talking to.
- **Talking to each other.** If one asks the other a question they continue on their own, capped
  (`AIVA_SISTER_MAX_ROUNDS`, per hour, per day). An exchange that grows out of something *you* said
  ("Aiva, ask your sister…") is spoken aloud — you're in the room and asked for it. Exchanges they
  start themselves are **private** by default: text only, no ElevenLabs spend, visible on the
  dashboard's **Conversations** tab. Press Ctrl+Alt+P or ask either of them ("let me hear you two" /
  `set_sister_talk audible`) to hear those too. When you're away they may start a short private chat
  of their own a few times a day (`AIVA_SISTER_CHATS_PER_DAY`).
- **Saga has no written personality.** Her floor is only the shared rules (how to talk, understanding
  you, tools, safety) plus "you're new, nobody wrote who you are". Her temperament forms from her own
  diary over days and weeks, her tastes from experience. The dashboard's Saga tab shows "who she is
  so far".
- **They know each other from experience.** Each keeps a kin store of what she has actually seen the
  other do, make, read or say, and answers "what can Saga do?" from it without asking an LLM —
  honestly saying "I haven't seen her do that" when she hasn't.
- **Background news waits its turn.** A sister's own news (a reading session she finished, a task
  that completed, a thought of her own) never lands mid-conversation: the house holds it until
  nobody — you or a sister — has spoken for `AIVA_ANNOUNCE_QUIET_S` seconds (default 45), and a
  running sister exchange keeps going in the meantime. Replies to you always come first.
- **Calling each other.** Only Aiva starts by default. Say "Aiva, call your sister" (or "get Saga")
  and she boots Saga in her own console window — the same as running `start_saga.bat` — and Saga
  says hello when she's in, about half a minute later. Works the other way round too. Stopping a
  sister is still yours: close her window or Ctrl+Alt+X.

Saga's look is a retexture of the same Live2D mesh: put her textures in
`aiva/assets/avatar/ariu/saga.2048/` (same file names as `ariu.2048/`) and run
`python -m modules.avatar_renderer --write-model saga` to generate `saga.model3.json`. Her voice is
an ElevenLabs voice id in `.env` (`SAGA_ELEVENLABS_VOICE_ID`; `python tools\pick_voice.py --verify
<id>` checks it, `--set <id>` writes it). Her wake word is `aiva/models/hey_saga.onnx`, trained with
the recipe in `training/` (`hey_saga.yaml`).

All the knobs live in the `SAGA_*` and `AIVA_SISTER_*` / `AIVA_HOUSE_*` blocks of `.env`.

## Dashboard

A live web view of both sisters at http://127.0.0.1:8760 — the house opens it at boot; by hand:
`cd aiva && ..\.venv\Scripts\python dashboard.py` (`--open` opens the browser). Tabs **Aiva | Saga |
Conversations | Tasks | Settings**: mood, diary, wishes, traits, books, tasks start to finish, the
sisters' conversations, spend caps. It is read-only and loopback-only, except the Controls strip,
the chat box (types a line to them as if it had been heard) and Settings' Save.

## Memory

Each sister keeps her own SQLite file — `aiva/memory.db` for Aiva, `aiva/saga.db` for Saga: full
transcripts per session, plus a `facts` table of durable things she's learned about you (extracted
by a cheap LLM pass at shutdown). Facts are injected into her system prompt at startup, so she
remembers you across restarts. The same file holds her diary, wishes, tastes and task history, so
it is snapshotted into `aiva/backups/` at boot; her current mood is kept apart, in
`aiva/inner_state.json` (`aiva/saga_inner_state.json` for Saga). The house keeps the shared
conversation log in `aiva/house.db`.

## Ambient mode (wake word)

Set `AIVA_WAKE_WORD=1` (the `start_*.bat` launchers set it) and the house starts
asleep: nothing streams to Deepgram (idle time costs nothing) until a wake word.
After `AIVA_IDLE_TIMEOUT` seconds (default 20) without conversation it dozes off
again.

The repo ships custom openWakeWord models: `aiva/models/hey_aiva.onnx` (set
`AIVA_WAKE_MODEL=models/hey_aiva.onnx`; without it the house falls back to the
pretrained **"Hey Jarvis"**) and `aiva/models/hey_saga.onnx`, which the house
picks up for Saga on its own (`SAGA_WAKE_MODEL` overrides it). The training
recipes live in `training/`.

**Verified wakes.** A wake-word hit alone is only a *candidate*: the listener
keeps the last 2.5 s of audio (the wake phrase itself), hands it to Deepgram
ahead of the live mic, and the house only truly wakes if that first transcript
contains a sister's name — otherwise it drops it and goes straight back to sleep
without reacting. This is what stops TV/game dialogue and other people's
speech waking them (the custom model fires on those with the same scores as a
real "hey Aiva"). Knobs: `AIVA_WAKE_VERIFY=0` disables the check,
`AIVA_WAKE_VERIFY_SECS` (default 5) is how long it waits for speech, and
`AIVA_NAME_VARIANTS=eyva,ayvah` / `SAGA_NAME_VARIANTS` add spellings Deepgram
uses for their names.
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

Study and synthesis run on their own models (`AIVA_STUDY_MODEL`, which falls
back to the task model, and `AIVA_STUDY_SYNTH_MODEL`, default `gpt-5.6-terra`)
so they can stay off the voice model's rate limit; a whole novel costs roughly
ten to twenty-five cents, capped at `AIVA_BOOK_MAX_COST_USD`.
`python -m modules.becoming --show` prints the traits, lessons, and turns of
phrase she's picked up. Offline tests:
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
yes. Shell commands and handing code to Claude Code wait for a yes.
When a new task has two genuinely different approaches (UI vs terminal vs
Claude Code, quick vs thorough) she offers both and asks which you prefer —
and after three consistent picks she remembers your preference and stops
asking. She grades Claude Code's work (diff, tests) and tells you honestly
whether it did well, got partway, or botched it. `AIVA_TASK_FREE_RANGE=1` (off by
default) lets coding tasks run without the plan and hand-off approvals; shell
commands and anything that sends or pays on your behalf still wait for a yes.

Tasks that go well twice become **flows** she offers to keep ("want me to
remember that as a flow called streaming setup?"); "run my streaming setup"
then replays it. Say "no, that's wrong" or "next time do X first" right after
a task and she records it against that task — the next time it comes up she
confirms first and says what she'll do differently.

Knobs: `AIVA_TASK_MODEL` (planner/verifier model; defaults to the voice model —
point it elsewhere if task calls compete for the voice model's tokens-per-minute
allowance, see below). Per-task safety limits live in `tasks.Limits` (12 steps,
80 tool calls, 90 LLM calls, 20 minutes, $1.50). Look for `task t-…` lines in
`aiva/logs` to see plans, familiarity scores and decisions. Offline tests:
`.venv\Scripts\python -m pytest tests -q`.

## Models

`AIVA_LLM_MODEL` (their voice), `AIVA_TASK_MODEL` (task planner/verifier) and
`AIVA_VISION_MODEL` (screen vision) fall back to `gpt-5.6-luna` in code (the
task and vision models follow `AIVA_LLM_MODEL` when unset): OpenAI's fast/cheap
tier — sub-second first token on tool-calling turns, $0.20 / $1.20 per million
tokens, and 500k tokens-per-minute even on the entry usage tier.
`gpt-5.6-terra` is the step up if you want more brains at the same limits.
Since 0.11.0 both sisters' voice runs on **gpt-6-luna** (`AIVA_LLM_MODEL=gpt-6-luna`,
about half the price), with the task and vision models left on gpt-5.6-luna;
`SAGA_LLM_MODEL` gives Saga her own model if you ever want to compare them again.
gpt-4.1 / gpt-4.1-mini shut down on 2027-01-14. The gpt-5 line and gpt-6-luna
need `max_completion_tokens` and `reasoning_effort: none` on Chat Completions —
the code handles that automatically per model (`modules/utilities.py: llm_kwargs`),
so switching is just the env var.

## Rate limits (OpenAI tokens per minute)

Every voice turn re-sends her whole context plus all tool schemas (roughly
9–10k tokens before any conversation), so on OpenAI's entry tier (30k TPM for
gpt-4.1) a few quick turns plus a task in the background can hit a 429. What's
in place: the OpenAI clients retry rate limits with backoff
(`AIVA_LLM_MAX_RETRIES`, default 5) instead of dropping the turn; task planning,
verification and screen vision can run on a different model with its own
allowance (`AIVA_TASK_MODEL`, `AIVA_VISION_MODEL`); and the live context is
trimmed to `AIVA_CONTEXT_BUDGET_CHARS` (24000) so long sessions don't grow
without bound. On gpt-5.6-luna (500k TPM) this is rarely an issue; on gpt-4.1
the lasting fix is a higher usage tier (tier 2 lifts it to 450k TPM).

## Desktop avatar (self-owned Live2D renderer)

Each sister renders her own body: `modules/avatar_renderer.py` is a separate process
(live2d-py + PyQt5) drawing the Live2D model in a transparent, click-through,
always-on-top window. No VTube Studio, no Spout overlay, no VB-Cable — her TTS
audio goes straight to your speakers, and a pipeline tap streams its loudness
to the renderer for lip-sync.

- Model assets live in `aiva/assets/avatar/ariu/` (licensed content, not committed).
- The renderer outlives a sister's self-restarts; a rebooted sister adopts it via
  her state file (`aiva/avatar_state.json`, `aiva/saga_avatar_state.json`). It dies
  with her on a real quit. If the renderer she finds speaks an older protocol (an
  update changed it), she retires it and starts a fresh one — her placement carries over.
- Saga's body also draws a quiet **constellation** to her upper left: one faint star per
  moment she lives through (`SAGA_CONSTELLATION=0` turns it off).
- Demo it standalone (never starts Aiva): `cd aiva && python -m modules.avatar_renderer --demo`
- **What she's doing shows.** While she reads, she pulls up a book: her hands come
  up, the book rises into them, her head tips down and her eyes run along the lines;
  a finished section turns the page, and she looks up from it while she talks. While
  she's on a task (or a Claude Code job of her own runs), a small work panel floats up
  beside her head — the task, the step it's on, a progress bar; amber and pulsing
  while it waits on you, a check or a cross when it ends — and she glances at it.
  Either way her eyes stay open even while the house sleeps. Preview it without a
  sister: `cd aiva && python -m modules.avatar_renderer --instance demo --demo-activity`
  (add `--windowed` for a small movable window).
- **She has her own drawing canvas.** `open_drawing` / `draw_stroke` / `clear_drawing`
  open the standalone `drawing-app/` window straight from her process (no terminal, no
  console, your keyboard focus left alone) and draw on it in her own ink; one window per
  sister, reused until you close it. "Open your drawing app" means this canvas, never
  Aseprite or Paint. Check it with no terminal by opening `drawing-app/verify_drawing.pyw`
  (details in `drawing-app/README.md`).
- **Her drawings are kept.** Saved drawings go to
  `%USERPROFILE%\Pictures\Aiva and Saga Drawings\` as `.aivadraw` plus a `.png`. Use
  **Save** (`Ctrl+S`), **Save As…** or **Open…** (`Ctrl+O`) in the canvas toolbar, or
  ask her: `save_drawing` ("save that as sleepy cat") and `open_saved_drawing` ("bring
  back the tree"). The status bar shows the folder and whether the drawing is saved.
  You don't need to save for nothing to be lost: every finished stroke updates a
  recovery copy, and closing the window, a crash or her restart never loses the canvas.
  The next time it opens, her last drawing is back. After a restart she reuses the
  window that's still up rather than opening a second, blank one. Clear keeps a
  `recovery\<name>-before-clear` copy, and opening a drawing over unsaved work saves
  that work first. A damaged recovery copy is set aside, never overwritten. Details:
  *Saving and recovery* in `drawing-app/README.md`.
- **A stroke that came out wrong can be taken back.** The canvas toolbar has **Undo**
  (`Ctrl+Z`), and she has `undo_stroke` ("undo that"). Each one takes back only that
  author's newest finished stroke, never her sister's, and repeating it walks back
  further. In her window, strokes you draw by hand count as hers. Clear still can't
  be undone; its before-clear copy is how you get a cleared drawing back.
- Lip-sync is two-channel: loudness drives how open the mouth is, spectral
  brightness drives its shape (`ParamMouthForm`) — dark vowels (oo/oh) purse it,
  bright ones (ee/eh) widen it, so she appears to mouth phoneme shapes.
- Env knobs (all optional): `AIVA_AVATAR_MODEL`, `AIVA_LIPSYNC_GAIN` (default 2.5),
  `AIVA_LIPSYNC_FORM` (viseme shape strength, default 1.0; 0 disables),
  `AIVA_AVATAR_FPS` (32), `AIVA_AVATAR_SWAY` (1), `AIVA_READING_PROP` (the book
  image), `AIVA_AVATAR_TASK_LABEL` (1; 0 hides the task's words on the work panel,
  e.g. while streaming).

## Shell command safety

Either sister can stage shell commands but **cannot execute them without your verbal
confirmation** — she'll tell you what the command does and ask first. Every staged and
executed command is printed to the console.

## Project layout

```
aiva/house.py                the house: mic, STT, wake words, hotkeys, the floor, house.db
aiva/main.py                 one sister's process (--character aiva|saga): her LLM, voice, core
aiva/launcher.py             boots OBS (optional), ensures the house, then one sister
aiva/dashboard.py            live web view of both sisters (http://127.0.0.1:8760)
aiva/version.py              her version (changes are logged in CHANGELOG.md)
aiva/modules/house_*.py      protocol, server, client, floor, memory, stage (house side)
aiva/modules/house_link.py   sister-side processors (HouseLink, PrivacyGate, MindWatcher, ...)
aiva/modules/sisters.py      pure turn-taking decisions (addressee, sister rounds, floor)
aiva/modules/character.py    Character.from_env(slug): what makes a sister herself
aiva/modules/kin.py          what she knows about her sister, from experience
aiva/modules/ear.py          MicController / ActivityWatcher / UserAudioTap (house)
aiva/modules/wake.py         openWakeWord listener, N models
aiva/modules/persona.py      personality floors / system prompt
aiva/modules/tools.py        native LLM tool schemas + handlers
aiva/modules/avatar_renderer.py  her body: Live2D renderer process (live2d-py + PyQt5)
aiva/modules/avatar.py       Aiva-side client (spawn/adopt renderer, lip-sync feed)
aiva/modules/avatar_ipc.py   shared IPC protocol + state file helpers
aiva/modules/app_launcher.py launch Windows apps (app_paths.json + registry)
aiva/modules/file_operations.py  create/open/list files (fenced to your user profile)
aiva/modules/utilities.py    time/date/weather
aiva/modules/memory.py       session transcripts + long-term facts (SQLite)
aiva/modules/tasks.py        task runner (+ task_memory.py: experience-based autonomy, flows)
aiva/modules/reading.py      reading books (+ library.py parsers, becoming.py how books change her)
aiva/modules/inner.py        mood, diary, wishes (+ drives.py, dreams.py, projects.py, preferences.py)
aiva/modules/web.py          web search/fetch (+ space.py, her kept finds; limits.py spend caps)
aiva/modules/constellation.py  Saga's constellation witness
aiva/modules/drawing.py      her drawing canvas: launch or reuse drawing-app/, draw on it
drawing-app/                 the standalone drawing app (own README and tests; imports nothing from aiva/)
```

## Roadmap

- [x] Phase 0 — repo hygiene
- [x] Phase 1 — streaming pipeline core, native tool calling, hotkeys
- [x] Phase 2 — self-hearing gate for speaker use, F10 interrupt, pipecat 1.x API migration
  - [x] VB-Cable lip-sync routing into VTube Studio + expression hotkeys (queried from the live model at startup)
- [x] Phase 3 — long-term memory (facts table, injected into the prompt)
- [x] Phase 4 — wake word + ambient mode (custom "Hey Aiva" / "Hey Saga" models, transcript-verified)
- [x] Phase 5 — session cost telemetry
- [x] Phase 6 — self-owned Live2D avatar renderer (dropped VTube Studio, Spout overlay, VB-Cable)
  - [ ] packaging / autostart
