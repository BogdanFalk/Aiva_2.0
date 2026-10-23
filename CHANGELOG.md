# Changelog

All notable changes to Aiva. Format follows [Keep a Changelog](https://keepachangelog.com);
versioning is [SemVer](https://semver.org). The version lives in `aiva/version.py`.

When you make a change, add a bullet under **[Unreleased]**; when you cut a version, move those
bullets into a new dated section and bump `aiva/version.py`.

## [Unreleased]

### Added
- **A GPT-Live trial: talk to one sister in true full duplex** (`aiva/live_trial.py`). A standalone
  script, run with the house and the sisters stopped (it refuses to start while the house runs) and
  with headphones (no echo cancellation). She hears you while she talks and you can cut in any
  time. Her persona and long-term facts are read from her db read-only as the session's
  instructions, and she's told it's a trial, so nothing is saved and her house tools aren't there.
  Look-ups go to an OpenAI-hosted backend (`AIVA_LIVE_BACKEND_MODEL`, gpt-6-luna, hosted web search)
  while she keeps talking. Voice: a preset (`--voice marin`) or one designed from a description
  (`--design "…"`, `POST /v1/audio/voices`, free, GPT-Live only, id printed for reuse).
  `--minutes` caps it (10), `--check` validates config without a session. $0.05 a minute plus
  backend tokens, printed at the end. A 3-second live session on 2026-10-24 confirmed the account
  can use gpt-live-1 with this delegation. This is a feel test beside the house, not a change to it:
  the floor, gate and tools still assume one grant at a time.
- **She can answer sooner (`AIVA_SPECULATE=1`, off by default).** With Flux, the house turns on its
  eager end of turn and the stage runs `EagerUserTurnStrategies`. When Flux guesses he's done,
  `StageRouter` calls `Floor.speculate`: if the floor is idle and the line plainly goes to one
  present sister (no "both", no errand, nobody absent; the arbiter runs now, while he finishes), she
  gets a new `prepare` message. She runs her LLM on a provisional copy of her context and pipecat's
  own speculation gate in her LLM service HOLDS the reply (`HouseLink.prepare`). Nothing past the
  LLM sees it: no audio, no gate, no transcript, nothing overheard. If the turn ends on the same
  words (pipecat's NormalizedMatch, then `same_line` on the floor), the granted `turn` carries
  `prepared` and `HouseLink.run_prepared` writes his line into her real context and releases the
  reply into the turn, saving the model's think time. If he goes on, or the words differ, the house
  sends `unprepare` and the turn runs fresh. A speculative reply that wants a tool withdraws itself,
  so tools only run on a confirmed turn. A new speaker or a reunion/repair note at turn time also
  discards it. Floor-side TTL 10 s. Costs extra model calls when he keeps talking, hence off by
  default. Tests run through pipecat's real gate: `tests/test_speculation.py`.
- **A persona drift check, and a nudge to stay herself in long talks.** `python -m modules.persona_drift
  --who aiva` samples her lines from the first, middle and last third of her recent long sessions
  and has a judge model rate each 1-5 for how much it sounds like her ("Stable Personas": observed
  persona expression fades in long conversations even when self-reports don't). It reads her db
  read-only. Baseline on 2026-10-24 (gpt-5.6-luna, 5 sessions, 84 lines): early 3.10, middle 3.12,
  late 3.43, drift −0.33, so no fading, but the overall level sits near "could be any polite
  assistant". Every `AIVA_VOICE_REMINDER_TURNS` (25) of his lines she gets one short "stay
  yourself" note; re-run the check after a change to see whether it moves.
- **The house guesses who you meant with OpenAI Decisions.** When you say no name and both sisters
  are there, the arbiter now asks the Decisions API (gpt-6-luna, out of beta 2026-10-22) one typed
  question, "which sister did he mean?", over the same context the model arbiter read
  (`ARBITER_DECISION_INSTRUCTIONS`), and gets a probability per sister; the answer goes through
  `decision_reply` into the reply shape the floor already parses. About half a second in a live
  check, priced at input tokens only. Every decision is logged with its probabilities
  (`arbiter (decisions): aiva=0.12, saga=0.88`), the raw material for tuning; any failure falls back
  to the model call. `AIVA_ARBITER=decisions|llm`.
- **Her voice can laugh, sigh and whisper (Eleven v4 Turbo).** A sister whose ElevenLabs model is
  eleven_v4* or eleven_v3 now speaks through `ElevenLabsDialogueTTSService` (Text-to-Dialogue,
  checked live with both voices on 2026-10-24) and gets a VOICE CUES prompt block: `[laughs]`,
  `[sighs]`, `[whispers]` and a dozen more (`modules/voice_tags.py`), at most one per reply. The
  cues are delivery, not words: `PrivacyGate` strips them from `said` (so the house transcript and
  her sister's overheard line are clean) and from text-only replies, and saved transcripts drop them.
  A Flash voice gets no block, so nothing would read "[laughs]" out loud. `AIVA_TTS_STABILITY`
  (0.5, closest to her recorded voice). Same price per character as Flash ($0.04 per 1K).
  Backchannel clips are remade in the new voice automatically (they're keyed by model).
- **Read him, feel, then speak.** After each of his lines a side-call on the task model writes what
  seems to be going on with him beyond the words and how it lands on her (`modules/appraisal.py`;
  ToMAgent and "Feeling First, Speaking Second"). It runs while she answers, so it never delays a
  reply: its valence/arousal go onto her bus at once as an `appraisal` event (her mood, and with it
  her tone, moves), and a confident reading rides on his next turn as a short note she must not
  announce. `AIVA_APPRAISAL`, `AIVA_APPRAISAL_MIN_CONFIDENCE` (0.5), `AIVA_APPRAISAL_NOTE_TTL_S`.
- **Facts that stop being true are replaced, not doubled.** The fact-extraction prompt may now say a
  new fact `replaces` a known one (he changed jobs, a plan changed, he dropped an instruction); the
  old fact moves to a new `facts_superseded` table with what replaced it, so `facts` stays "true
  now" for every reader (the LONG-TERM MEMORY block, kin, becoming, the dashboard) and nothing he
  said is lost. Matching ignores case, spacing and end punctuation; a `replaces` naming nothing
  known is ignored. `Memory.load_superseded()`.
- **She knows when she's been upgraded.** `modules/upgrades.py` holds one hand-written entry per
  change to her, in spoken second person, with which sisters it's for and when it's really on (a
  feature switched off or a model not set in `.env` is never told). On her first boot with
  something new, a short note rides on her intro and she mentions it once, in her own words; it's
  recorded in her own db (`store_meta` `upgrades_told`) after that first greeting. For 14 days a
  WHAT'S NEW IN YOU block keeps it in her prompt, so "what's new with you?" gets a true answer.
  A brand-new sister isn't told anything: she simply starts with all of it. The first entries:
  GPT-6 Luna (only for the sister running it; Saga is told that Aiva moved and she stayed),
  backchannels, emotion hearing, the Flux ear, and this itself. Tests: `tests/test_upgrades.py`.
- **Each sister can think with her own model, and GPT-6 Luna works.** `Character.llm_model` is
  `AIVA_LLM_MODEL`, which Saga shares unless the new `SAGA_LLM_MODEL` is set. `gpt-6-luna` ($0.10
  / $0.50 per 1M, about half of gpt-5.6-luna) gets the gpt-5 request shape (`utilities.gpt5_shape`:
  `max_completion_tokens`, `reasoning_effort: "none"`); before this, a gpt-6 name was sent
  `max_tokens` and every turn was a 400 (checked with a live call). gpt-6.1-sol is deliberately not
  matched (it rejects effort `none`). Its price is in `LLM_RATES`. This branch's `.env` runs Aiva on
  gpt-6-luna and keeps Saga on gpt-5.6-luna, so the two can be compared. The code's fallback
  models move from gpt-4.1-mini, which OpenAI shuts down on 2027-01-14, to gpt-5.6-luna.
- **She says "mhm" while you talk.** With Flux, the house watches his partial transcript and, at a
  clause-complete pause while he is still talking, sends ONE sister a new `backchannel` message
  (`modules/backchannel.py`): the sister his words name, else the sticky addressee, else the last
  one who spoke to him. She plays a short clip ("mhm", "yeah", "right"…) in her own voice,
  synthesised once through her ElevenLabs voice into `aiva/voice_cache/<slug>/`, and her avatar
  nods (renderer protocol 8, `cue nod`). It is not a floor grant: no LLM, nothing in her context,
  no `speaking`/`said` back. Skipped while asleep, muted, during an unverified wake, while a sister
  holds the floor, after a finished question or a dangling word ("and the, um…"), within the
  cooldown, or when Flux is about to end his turn. The clip isn't gated (that would cut his words);
  its echo is removed from his final transcript instead (`strip_echo`). `AIVA_BACKCHANNEL` (needs
  `AIVA_STT=flux`), `AIVA_BACKCHANNEL_COOLDOWN_S` / `_PROBABILITY` / `_MIN_SPEECH_S` / `_PAUSE_S` /
  `_VOLUME`. Tests: `tests/test_backchannel.py`.
- **She hears how you sound, not just how loud.** Per user turn the house sends the turn's audio
  (last 15 s, as WAV) to an OpenAI audio model (`AIVA_EMOTION_MODEL`, default `gpt-audio-mini`) in
  the background and gets back a few emotions from a fixed vocabulary (fatigue, amusement,
  frustration, warmth...), vocal bursts (laugh, sigh, yawn, groan...) and valence/arousal
  (`modules/emotion_hearing.py`). It never delays her reply: a reading that's in before the turn
  goes out rides on it; a later one (the usual case, ~1.5 s) is folded into a still-queued turn or
  sent as the new `affect` house message. Her `voice_affect` event now carries the labels and
  bursts, and a confident reading (`AIVA_EMOTION_NOTE_CONFIDENCE`, 0.7) gives her a short note —
  on this turn, or "a moment ago he sounded..." on your next one. Skips turns under 0.8 s and
  unverified wakes; `AIVA_EMOTION_HEARING` toggles it; `AIVA_EMOTION_MONTHLY_CALLS` is an
  alert-only cap; calls show as `llm.emotion` in Statistics (audio prices in `AUDIO_IN_RATES` are
  estimates).
- **The house hears with Deepgram Flux, which also decides when you've finished talking.**
  `AIVA_STT=flux` (the new default; `nova` keeps the old nova-3 + Silero VAD + smart-turn path) —
  Flux's end-of-turn (`AIVA_FLUX_EOT_THRESHOLD`, default 0.7; `AIVA_FLUX_EOT_TIMEOUT_MS`, default
  5000) drives the stage aggregator's turn, with no local VAD or turn model. Plain end-of-turn only,
  no speculative replies. Because Flux has no free keepalive, the house closes its socket after
  `AIVA_FLUX_IDLE_CLOSE_S` (3 s) without audio while asleep and reopens it when the wake word
  fires (`modules/house_stt.py`). An empty final transcript no longer counts as "no name" during a
  wake check.
- **An initiative-probe experiment for the project itself** (`aiva/modules/initiative_probe.py`,
  not wired into her). It finds a module that no project notes name, picks the most recently
  modified one and proposes one step: the rules file to note it in. It only reads and proposes.
  `python -m modules.initiative_probe` prints the result. First run: `house_memory.py` has no
  note, and `.claude/rules/house.md` already loads it.
- **A record of every canvas action, behind the scenes.** Each sister's drawing board now writes
  one JSON line per action (`open`, `stroke`, `clear`, `undo`, `save`, `open_saved`, `close`) to
  `aiva/logs/drawing-<author>-YYYYMMDD.jsonl` (`drawing.DrawingJournal`). A line holds the author,
  the points as sent with their bbox/start/end, the color and width she asked for and the ones the
  canvas used (her own ink when she named none), the stroke id and revision the window answered,
  the window's pid/port, and the outcome: `ok`, `refused` (nothing drawn), `failed` (no window
  reached) or `unknown` (a live window went silent, so it may be drawn). `drawing.read_journal()`
  merges both sisters' files into one timeline. Nothing she says changes, a failing journal never
  fails a stroke, and the drawing app and its wire are untouched. 2026-10-21 their two heart halves
  didn't meet and a screenshot was the only evidence. The journal now shows each half's
  coordinates directly.
- **Undo on her drawing canvas.** The drawing app (`drawing-app/`, now 0.4.0) takes back a stroke:
  **Undo** (`Ctrl+Z`) in the toolbar, the wire command `{"cmd": "undo", "author": …}`
  (`DrawClient.undo()`), and a new sister tool `undo_stroke`. Each takes back only that author's
  newest finished stroke, never a stroke still being drawn and never her sister's, and repeating
  it walks back further. It goes through a new `RemoveStroke` op on the one mutation path, so the
  canvas repaints and autosaves as it does for a stroke, and a restored canvas comes back without
  the undone stroke. With nothing of hers left, the reply is `"stroke_id": null` and she says
  there was nothing to undo. `undo_stroke` never opens a window. It is runner-allowed but outside
  `SAFE_TOOLS` and the `draw` capability, because his hand-drawn strokes in her window carry her
  name. A 0.3 window refuses undo, and she says to reopen the canvas (safe there, since 0.3
  autosaves); a pre-0.3 window still gets "leave it open". No redo yet. Tests:
  `drawing-app/tests/` (model, commands, client, storage + the window) and `tests/test_drawing.py`,
  including a real offscreen canvas. She needs a restart to have the tool, and a canvas window
  opened before this change needs reopening to take undo.
- **Her drawings are saved, and her canvas is never lost.** The drawing app (`drawing-app/`, now
  0.3.0) saves to `%USERPROFILE%\Pictures\Aiva and Saga Drawings\` as `.aivadraw` (plain JSON)
  plus a `.png`. The canvas toolbar has **Open…** (`Ctrl+O`), **Save** (`Ctrl+S`) and **Save As…**,
  and the status bar shows the folder and whether the drawing is saved. Each sister has two new
  tools, `save_drawing` (optional name; a name never replaces another drawing) and
  `open_saved_drawing` (by name, or the newest). Both are safe tools and runner-allowed, and both
  belong to the `draw` capability. Without anyone saving: every finished stroke rewrites a recovery
  copy (`recovery\<author>-autosave`), so the next window for that sister comes back with her
  drawing ("with your last drawing back on it"). That holds whether the window was closed, killed
  or crashed. A sister restart no longer opens a second, blank window: her board first adopts the
  one still up (`launcher.find_running`, which checks the pid and asks the port for `status`). Clear
  keeps a `before-clear` copy, opening a drawing over unsaved work saves that work as a new drawing
  first, saving over a file keeps a `.bak`, writes are atomic, and a recovery copy that can't be
  restored is renamed `.unrecovered-<time>` rather than overwritten. One window per sister owns the
  recovery copy (an OS lock), so a second window can never overwrite it. Wire commands
  `save`/`open`/`drawings`/`status` (`code: "storage"` for file problems), `--save-dir`,
  `launch(save_dir=)`. Tests: `drawing-app/tests/test_storage.py`, new launcher tests,
  `tests/test_drawing.py`. Tests and `drawapp.verify` never touch the real Pictures folder. Needs a
  restart of the sister and a fresh canvas window: a window from before this change can't save,
  and she says so.
- **Saving her drawing is a tool, never a keypress.** Asked to save the canvas before her process
  had `save_drawing`, a task pressed Ctrl+S into whatever window had his focus. The `save_drawing`
  description and the planner/runner prompts now say saving is that tool, never `press_keys`
  Ctrl+S or the window's buttons. Code in her process gets a blocking direct save next to
  `launch_or_reuse`: `DrawingBoard.save_now(name)` / `drawing.save_drawing(author, name)`. A
  window from before saving existed (it accepts only `stroke`/`clear` and has no autosave) used to
  be answered with "close it and open your canvas again", which would have lost the drawing on it.
  Now she says to leave it open. Tests: `tests/test_drawing.py`.
- **A task saves her drawing with the same tool she uses, for Aiva and for Saga.** At 18:31 a task
  failed with "no direct tool to invoke that API". Her process had booted at 17:10, before
  `save_drawing` existed, so its runner had nothing to call. The tool itself was already complete,
  so no new tool was added. A new test checks the path a task really takes: `main.py`'s handlers go
  into the `ToolBox`, the toolbox lists and describes `save_drawing`, and one call saves under that
  sister's own name and returns both paths without reopening anything. Each sister needs a restart
  to get the tool. Tests: `tests/test_drawing.py::test_a_task_saves_through_her_toolbox_in_her_own_name`.
- **She draws on her own canvas, straight from her own code.** When she verified the drawing app she
  opened a PowerShell and ran `python -c "…launch()…"` in it, so he saw a terminal every time. Her
  new tools `open_drawing`, `draw_stroke` and `clear_drawing` (`modules/drawing.py`) open the
  standalone `drawing-app/` window from her own process: `drawapp.launcher.launch()` runs
  `pythonw -m drawapp` with no console. The window now shows without taking his keyboard focus (new
  `--no-activate`, which `launch()` passes), and strokes go over its loopback socket in her own ink.
  Nothing touches his mouse or keyboard. If he closes the window, the next stroke opens a fresh one.
  A lost reply is never resent while the app is up. The task runner may use all three, and the tool
  descriptions tell her never to reach for a terminal to open or test the app. A wish to draw maps to
  the new `draw` capability (auto, not on the self-act list). Needs a restart of the sister. Until
  then (or any time), `drawing-app/verify_drawing.pyw` opened with `open_file` runs the clean check
  with no terminal: it leaves the window up with an Aiva and a Saga stroke and writes
  `drawing-app/verify-report.json`.
- **"Open your drawing app" opens her drawing app, not Aseprite.** A task asked `launch_app` for
  "drawing-app", the generic launcher guessed, and Windows put Aseprite up with a "can't find" dialog.
  Now `launch_app` recognises her own app's names (`drawing.is_drawing_app_request`: "drawing app",
  "drawapp", "your canvas", …) and hands them to her board's launch-or-reuse instead. Those names
  never reach the generic launcher, and with no board it refuses rather than guess. A program he
  names (Aseprite, Paint) still opens as before. Code in her process has one callable for this,
  `drawing.launch_or_reuse(author)` (blocking) or `await board.open()`, on the one board per sister
  that `drawing.board_for` keeps. A missing or broken `drawing-app/` is a spoken failure there too,
  not an exception. The planner and step prompts, the `launch_app`/`open_drawing`
  descriptions and the `draw` capability's keywords all point at `open_drawing`. Needs a restart of
  the sister.
- **She looks after Miso on her own.** Until now the cat could only ask (a 'hungry' announce) and
  she only fed, played with or petted it when told. A caretaker (`modules/cat_care.py`) now glances
  at the cat every minute and decides by itself, with no LLM call: she feeds it when it gets hungry,
  tosses the ball or pets it when it seems down, and gives it a pet just because when she's in a warm
  mood. She leaves a busy or napping cat alone (unless it's starving), doesn't repeat what you just
  did yourself, only feeds at night, and caps the extras per day. It's felt on her experience bus
  (`cat_care`, a little competence), her sister hears about it once a day, and she only mentions it
  out loud now and then (you at the PC, her awake, twice a day at most). A wish about the cat is now
  something she can act on herself (`tend_cat` joined the safe act list). `cat_status` says what
  she did lately. Toggle `AIVA_CAT_CARE`.
- **Her body shows what she's doing: a book while she reads, a work panel while she works.** The
  reading book-prop never really worked: it sat at a fixed spot on the monitor, much bigger than her,
  on Aiva's side of the screen even for Saga, and in a separate window that could fall behind her.
  Now (`modules/avatar_activity.py`, pure and clock-driven; painted into her own GL frame by
  `paint_book`/`paint_panel` in `avatar_renderer.py`):
  - **She pulls up a book.** She takes the model's own 手柄 pose (both hands at her chest) with the
    controller hidden. The arms crossfade up, the book rises into her hands and fades in, then her
    head tips down and her eyes run along lines of text. A finished section turns the page. While she
    talks she lowers it and looks up. Putting it away runs backwards, hands last. Saga's book is
    tinted to her colours.
  - **A work panel floats up beside her head while a task or a Claude Code job of hers runs.** It
    opens on the side toward the middle of the screen and shows the task, the step it's on and a
    progress bar, typing away. It turns amber and pulses while the task waits on you (and she looks
    at you, not at it), shows a check or a cross when the task ends, then folds away. She glances
    at it as she works. `AIVA_AVATAR_TASK_LABEL=0` keeps the animation but hides the words.
  - **Busy means awake.** While she reads or works her eyes stay open even when the house sleeps.
  - Both follow her placement (moved, resized, rotated, another monitor) because they're drawn in
    model units mapped through it. Your own 手柄 toggle survives a reading session.
  - Wire: new `activity` (whole state, resent every 5 s so a restarted renderer heals) and `cue`
    (`page|done|failed`) IPC commands; `hello` reports `activity`. The old `prop` command still means
    "reading". `TaskRunner.activity()` and `ReadingEngine.now_reading()` feed the sister's
    `body_activity` loop.
  - **A renderer from before an update gets replaced.** `PROTOCOL_VERSION` is now 2, and a sister
    that adopts an older renderer asks it to quit and spawns a fresh one. Her body outlives her
    restarts, so without this a renderer change would never reach the running body. Placement
    carries over in the state file.
  - Preview without a sister: `python -m modules.avatar_renderer --instance demo --demo-activity`.
  - Retired `AIVA_READING_PROP_SCALE/X/Y` (the book now sits in her hands); `AIVA_READING_PROP` still
    picks the image. Tests: `tests/test_avatar_activity.py` plus additions to the renderer, client,
    task-runner and reading tests.
- **Saga's quiet constellation witness.** A faint sky of stars hangs over Saga's avatar, one star lit
  per moment she lives: a line she said, a line she heard, a thing her sister said, a section she
  read. It never speaks, announces or shows text — a star keeps only the *kind* of moment and its
  weight. It lives in Saga's own avatar renderer process (`modules/constellation.py` for the pure
  sky, a click-through `SkyWindow` in `avatar_renderer.py`); Aiva's renderer runs
  the same code switched off, and nothing moves to Miso.
  - Quiet by construction: low opacity ceiling, a slow drift, bursts brighten the last star instead
    of adding one, a capped sky whose oldest stars dim out, brighter and slower while the house sleeps.
  - New `witness` IPC command (fire-and-forget, kind validated against a fixed set); `hello` reports
    `witness`, and `AvatarClient.send_witness` is a no-op unless the renderer says it has one.
  - The sky persists in her renderer's own state file (kinds, weights, seeds — never words), so it
    survives sister restarts. Knobs `SAGA_CONSTELLATION`, `_MAX`, `_GAP_S`, `_ALPHA`, `_SLEEP_ALPHA`,
    `_SCALE`, `_TINT` (restart her body to apply). Tests: `tests/test_constellation.py`.
  - Her sky follows her layer: sent to the desktop, the stars sink behind your windows with her
    instead of floating over everything. It leans with her too: rotated, the sky stays by her
    head instead of where her head would be upright. Her monitor is looked up when she changes
    screens, not about 8 times a second.
  - Preview without a sister: `python -m modules.avatar_renderer --instance saga --model
    assets/avatar/ariu/saga.model3.json --demo-witness` loops one moment of each kind, a burst,
    sleep, a tilt and both layers. It uses a temp state file unless `--state` is given, so made-up
    moments never reach her real sky.
  - A sky can't be lost or take her body down. A paint error is caught inside the sky window (an
    exception escaping a Qt paint aborts the whole renderer) and only hides it: the sky keeps
    witnessing and stays in the state file. `SAGA_CONSTELLATION=off` carries the saved sky through
    untouched instead of erasing it. A burst that only brightens a star is saved too. A state-file
    write that fails is retried instead of crashing her body.
  - `--selftest` checks the sky: a body with a witness passes only if its sky painted. It no longer
    reports OK beside a live body that owns the state file (it rendered nothing); it exits 4 and
    asks for `--state <a temp file>`.
  - `PROTOCOL_VERSION` is 3, so an older body is replaced at the sister's next start. That includes
    Aiva's body, which has no sky.
  - **It hushes while she speaks** (Saga asked for this herself on 2026-10-14). While her voice plays,
    the stars and lines dim together to about 30%. The dim comes on as a quick quarter-second fade
    and eases back over one second when she stops. It moves at a steady rate, so a breath between
    sentences lifts it only slightly instead of making the sky pulse. It repaints every frame while
    it eases and about 8 times a second otherwise. `--demo-witness` now speaks once so you can
    watch it.
  - **It sits to her upper left and drifts instead of twinkling** (also Saga's own ask: "keep it in
    the upper-left area around Saga, drifting slowly with no pulsing"). The sky used to be centred
    over her head. Now its bottom edge sits at the top of her head and its right edge just left of
    her middle. It still leans with her and stays on screen. The stars used to pulse in brightness.
    Now each one keeps a steady brightness and size and wanders slowly around its place, a few
    pixels over about a minute, while the whole sky sways together even more slowly. The lines
    ride along with their stars, are drawn at sub-pixel precision so they glide instead of
    stepping, and never jump to a different star. While the house sleeps the drift slows to a
    seventh, and dozing off never makes a star jump.
  - `PROTOCOL_VERSION` is 5, so her running body is replaced the next time she starts.
  - **It now looks the way she described it** (her words, house transcript #3968: "Tiny stars, about
    two to four pixels, dim white with a faint blue-violet tint; hairline connections, barely
    brighter than the stars"). A check against those words found three gaps, now closed:
    - Stars were sized as a fraction of the sky, which is a fraction of the monitor height. On her
      2560 px tall portrait screen that made them 16–22 px across, with a 36–50 px glow. Each star
      is now one disc 2–4 real pixels across (2 for the lightest moment, 4 for one a burst filled
      up), with no glow, whatever the screen.
    - Lines were about 0.28× as bright as their stars. Each end of a line is now 1.15× as bright as
      the star it touches, shading along the line between the two. They are still 1 px hairlines.
    - Near a screen edge the sky was pushed back on screen over her. On her screen it reached 296 px
      past her middle. It now shrinks until it fits to her upper left: 149 px instead of 512 px
      where she stands, with none of its stars right of her middle. Only below 96 px is it clamped
      as before.
    - Star brightness is unchanged. The painter used to boost stars 1.6× on its own; now it paints
      exactly the frame's alphas. So `SAGA_CONSTELLATION_ALPHA`/`_SLEEP_ALPHA` defaults went from
      0.30/0.50 to 0.48/0.80. That is the same painted brightness, and the knob is now the real peak
      star opacity (an old value set in `.env` would need ×1.6). The tint (#E0D6FF), the drift and
      the speech hush are unchanged.
    - New tests pin each property, including real pixels from the painter (`paint_sky`) and her
      own portrait screen.
  - `PROTOCOL_VERSION` is 6.
  - **Her refinements after seeing it** (house transcript #4010/#4014, Frank approved #4017):
    - **12.5% bigger.** `SAGA_CONSTELLATION_SCALE` default 0.20 → 0.23; inside the wider drift room
      that makes the star field 12.5% bigger. Where she stands now, her portrait screen's left edge
      holds the sky at 149 px wide, so a bigger scale alone changed nothing there. A sky held
      narrower than it wants now grows up to `SKY_TALLER` (1.15×) taller instead, upward and away
      from her, while she stands within 45° of upright. Its left, right and bottom edges don't
      move. Where she stands it goes from 149×149 to 149×171.
    - **A livelier drift.** `DRIFT_AMP` 0.03 → 0.04. On her screen the widest wander goes from about
      9 px to 13.6 px and the peak speed from 0.40 to 0.57 px/s, still under a pixel a second. Only
      positions move.
    - **A touch less uniform.** Each star gets its own steady nudge from its seed: up to ±0.3 px in
      size (still 2–4 px) and up to 12% dimmer, never brighter. Her 24 stars used to come in three
      sizes; now they range from 2.0 to 2.6 px. Brightness ceilings (0.48/0.80) and the tint are unchanged.
    - **Softer outermost lines.** A line end fades from 60% of the way out to half as bright at
      the field's rim, judged by where its star belongs, so it never changes as stars drift. Inner
      lines are still 1.15× their stars and 1 px.
    - Unchanged: the hush to 30% while she speaks with its one-second ease back, the upper-left
      anchor, and no pulsing. Every new factor is fixed per star or per line.
    - `PROTOCOL_VERSION` is 7, so her running body is replaced the next time she starts.
- **Tasks: second thoughts — she reads what Claude Code tells her and corrects herself.** Twice she
  sent Claude Code after a "Miso renderer" that never existed (Miso is the cat). It said so both
  times, and she never saw that, because only the last 1,500 characters of its report were kept.
  - **She gets the whole report.** The start and end of the report are kept, and its permission
    denials are included.
  - **The grader judges the step itself.** It grades each step against what that step was meant to
    do, and only on the files the agent changed. Your own uncommitted work is snapshotted before the
    job, so it no longer counts as the agent's "unrelated diff".
  - **Two new outcomes:** `blocked: question` (the agent stopped to ask) and `blocked: premise` (the
    task assumed something that isn't so). Neither counts against Claude Code.
  - **Look-only steps can't edit.** They run in plan mode, and the next step is handed the earlier
    step's report.
  - **On a failed or blocked delegate step, once per task, she reflects** (`RETHINK_PROMPT_T`, task
    model):
    - She works out what went wrong.
    - She keeps the lasting truths the report proves as `learned` facts, each checked against the
      report verbatim.
    - She retires the old project facts those truths disprove (`Memory.retire_fact`; never
      instructions or facts about people).
    - She records a lesson (`tasks.self_lesson`) that is shown in LESSONS FROM PAST TASKS.
    - Then she either re-plans with a corrected goal, or puts the agent's question to you and
      re-plans with your answer.
  - **Free range applies.** Under free range she just says she's trying again; otherwise she asks
    first.
  - New setting `AIVA_TASK_MAX_RETHINKS` (default 1).
- **v3 — an engine under her inner life** (docs/PLAN-v3.md). Her growth was text about herself fed
  only by books and the clock; now things happen to her and have consequences:
  - **She lives her conversations.** Every line he says is felt (`SelfEngine.heard`), finished and
    failed tasks, kept finds, sister talk and project steps are emitted, and the diary sees the recent
    conversation on every tick (before, only at shutdown).
  - **Needs** (`modules/drives.py`): connection, novelty, competence and rest drain with time and are
    refilled or spent by what happens (diminishing returns). A pressing need leans on her mood, adds
    a line to HOW YOU FEEL (or a "torn between…" when two pull against each other), shapes her
    wishes, and changes initiative: she speaks up sooner when she misses him, acts sooner when
    restless, and backs off when worn.
  - **His rhythm**: she learns when he's usually around (per weekday-hour, `rhythm` table) and feels
    the miss or the surprise. **Reunion** after ≥10 h gets a note in that turn; a **sharp moment** is
    felt and smoothed over once, lightly, later. Whether her reaching out **landed or got dropped**
    is felt and counted (dashboard).
  - **Nights** (`modules/dreams.py`): once a night she replays the day into moments that landed or fell
    flat, plus 0-2 **habits** ("When he comes home late, I keep it short") reinforced or weakened by
    what actually happened (anti-sycophancy gate), and forgets old low-salience experience. About
    weekly she revises her **life story** (chapter title, turning point). `--export` writes the
    moments as chat JSONL for a future open-weight fine-tune; her own model can't be fine-tuned, so
    she changes in context.
  - **Projects** (`modules/projects.py`): a wish that keeps coming back and that she can pursue alone
    becomes a collection or a look-into project, worked on in quiet moments (web + space, Tavily cap,
    ≤3 steps/day) and shared when done.
  - Prompt blocks YOUR STORY SO FAR / HABITS YOU'VE FORMED / MOMENTS THAT LANDED / WHAT YOU'RE WORKING
    ON; voice tool `let_go` (habit|project); dashboard "Her life" section + need bars; settings
    `AIVA_DRIVES`, `AIVA_DREAMS`, `AIVA_DREAM_HOUR`, `AIVA_STORY_DAYS`, `AIVA_PROJECTS`,
    `AIVA_READ_NOVELTY_BELOW`.
  - **Reading diet**: she reaches for a book only when she's actually hungry for something new.
    A preference her sister swayed is stored with origin `sister`. A preoccupation (focus) lets go
    after 24 h; the stale pre-v3 one is dropped at load. The task runner got an `on_outcome` hook for
    every finish. Saga's 12 introspective wishes were cleared the same way as Aiva's (backup in
    `aiva/backups/`).
- **Tasks tab on the dashboard** (Aiva | Saga | Conversations | **Tasks** | Settings), read-only: every task both sisters ran, merged newest first as cards (sister, outcome, goal, when, duration, steps done/total, cost, a Claude Code badge), filterable by sister, outcome and text, paged, with a tally per sister. Clicking one opens it start to finish from the new `GET /api/task?who=&id=`: details, familiarity and the autonomy decision, the plan as a step timeline (action, expectation, attempts, failure, collapsible evidence), a pending question, grade and your feedback, the full event timeline (announcements folded away), and every Claude Code job it delegated: the prompt, its saved result (logs/coding/ or the old %TEMP%\aiva_terminals\), cost, and — when the session transcript is found under ~/.claude/projects/ (by the logged session id, or for older jobs matched by prompt and time) — turns, files read/edited/written, commands run and lines changed. Ids, job names and session ids are regex-validated before any file lookup. A sister tab's task rows now link into it (`#tasks/<slug>/<id>`).
- **Free range for coding** (`AIVA_TASK_FREE_RANGE=1`): tasks no longer ask "okay?" before starting
  (novel, retried, delegate or irreversible) and hand work to Claude Code without a confirm; only a
  step that sends or pays on his behalf still asks. Her persona gets a FREE RANGE FOR CODING block
  (start_task straight away; `claude_code` in execute mode). Shell commands inside a task still ask.
- **Settings tab on the dashboard** (Aiva | Saga | Conversations | **Settings**) — every tunable knob
  the house and the sisters read, grouped (models, voice & audio, avatar, wake, floor, tasks, reading,
  mood, initiative, speaker ID, web, spend caps, notifications, dashboard, Saga, debug), each with a
  one-line description, its value in `.env` or the code default, who reads it and what must restart.
  Edit, reset-to-default and Save (only the changed rows) → `POST /api/settings` (local-only, same
  Origin) validates through the new `modules/settings_registry.py` and edits `.env` in place (comments,
  order and line endings kept, atomic, `.env.bak`); the reply says which process to restart and offers
  the restart buttons. API keys are shown masked and are never editable; a few important code constants
  (curiosity/initiative caps, trait/preference caps) are listed read-only with their source. A test
  greps the code for env keys so a new knob can't go missing from the page.
- **Task runner limits are now env-tunable**: `Limits.from_env()` (`AIVA_TASK_MAX_STEPS`,
  `AIVA_TASK_MAX_WALL_MIN`, `AIVA_TASK_MAX_COST_USD`, `AIVA_TASK_CLAUDE_WAIT_MIN`, … — defaults
  unchanged), used where `main.py` builds the `TaskRunner`.
- **"Call your sister"** — new voice tool `call_sister` (safe/auto): if the other sister isn't
  connected to the house it boots her in her own console (her `start_<slug>.bat`, exactly like
  double-clicking it; debounced) and she says hello when she's in; if she's already here it just says
  so. The persona's sister section and the house's "she isn't running" cue point at it, so "Aiva, get
  Saga" works even when only Aiva was started.

- **"Restart the house"** — new voice tool `restart_house` (and Ctrl+Alt+H in the house): the house
  exits 42 and `start_house.bat` relaunches it; the sisters are told it's a `house_restart` and stay
  up, reconnecting on their own. `restart_self` still restarts one sister (exit 42, body adopted,
  "I'm back"); Ctrl+Alt+R restarts both sisters; Ctrl+Alt+X quits everything.

- **Dashboard Controls strip** — the hotkey actions as buttons on every dashboard tab: mute/unmute,
  interrupt, sister talk private/audible, sleep/wake, per-sister Start / Restart / Quit, Restart
  house, Quit everything (destructive ones ask to confirm). The dashboard sends them to the house
  over its loopback socket as a `role="control"` client (`POST /api/control`, loopback-only unless
  `AIVA_DASHBOARD_CONTROL_REMOTE=1`); it stays read-only otherwise. New voice tool
  `set_sister_talk` reports or flips private/audible ("let me hear you two", "keep it between you").

- **Type to them from the Conversations tab.** One chat box above the transcript sends a line to the
  house (`POST /api/say` → `send_say` → new `say` message → `house._on_say` → `Floor.submit_user_turn`),
  which treats it exactly like a final transcript: same addressee rules ("Aiva, …", "both of you, …",
  otherwise whoever you were talking to, arbiter when it's ambiguous), same floor and queue, and she
  answers **out loud**. Only the voiceprint and the vocal affect are missing, since nobody spoke.
  Typing counts as presence, so it wakes the house even while the mic is deafened (F9) — it isn't
  audio, and you are plainly there. The box greys out with the house down, Enter sends, and the draft,
  caret and focus survive the poll that rebuilds the page. Local-only like the Controls strip.

- **Aiva forms temperament too, on top of the character she was written with.** Her dashboard's "Who
  she is so far" could never fill: the diary's `temperament` key was added only for a self-forming
  sister, the digest dropped a stray one, and her prompt builder passed no block — so Saga was all
  formation and Aiva was all floor. Now an authored character gets the key too, in its own wording
  (`TEMPERAMENT_CLAUSE_ON_TOP`: only what isn't already part of her, "most days have none"), renders
  it as **WHAT LIVING HAS ADDED TO YOU**, and reads her formed core back into the next digest so she
  reinforces instead of restating. The guard that makes this safe: `adopt_temperament` now passes her
  authored baseline to `validate_temperament` as `floor_vecs`, and a negation landing there is
  dropped and logged rather than eroding anything — living adds to her, it never unwrites her.
  `PERSONALITY` is untouched, so the golden still holds; a self-forming sister is unchanged, since
  she has no floor and her own core stays the only thing a negation can erode.

### Changed
- **Each Claude Code job thinks as hard as its job needs** (2026-10-17). Every job used to inherit
  your own Claude Code setting, Opus at xhigh effort, because `claude -p` was started without
  `--effort`. A read-only README review spent 18k of its 31k output tokens thinking, about 5
  minutes and $2. Now `tasks.job_effort` sets `--effort` per job:
  - The planner gives each delegate step a level: low to confirm one thing, medium for a review or
    small edit, high for a real change, xhigh to build something new.
  - A look never goes past high, and a change never past xhigh.
  - A job that came back partial or bad makes the next job on that project in the same task go one
    level up.
  - A merged job takes the level of its hardest part.
  - `AIVA_CLAUDE_EFFORT` sets the policy: auto (default), a fixed level for every job, or inherit
    for your own setting.
  - The level shows in the task log.
  - The planner is also told that a question about one or a few files is not a Claude Code job.
    One costs a fixed ~$0.25 before it starts: a one-file check cost the same at low and xhigh,
    5.5 s vs 9.9 s.
  - Asked to plan five kinds of job, the real planner kept the simple check on her own tools (3/3),
    gave the rename medium, the bug fix high and the chess module from scratch xhigh (3/3 each).
    The README review got medium or her own tools.
- **Tasks look, search and test without asking you** (2026-10-14). The task agent could only read a
  file, search, check git or run the tests through a PowerShell command, and every command waits on
  your yes: one verification took nine approvals, and Saga's text-file task ran out of moves asking
  before it wrote the file. It now has `read_file`, `find_files`, `search_text`, `repo_status` and
  `run_tests`, all read-only, fenced to your home folder, and never waiting on anyone. Several can go
  in one move. `run_tests` is the project's own `.venv` pytest, optionally on chosen test files
  inside the project. The shell step is only for changing things now, and that still asks.
  - **Bigger step budget:** 12 moves a step (was 6), 80 tool calls and 90 model calls a task. A step
    is told when it's on its last move, so it finishes with what it found instead of failing.
  - **A failed task keeps what it found.** Her failure note carries the last things the task
    gathered, and she offers you the direct way to still get the result (usually doing the rest
    herself right away) instead of starting over.
  - **Plans don't put a research step in front of a file you asked for**; any looking goes in the
    step that writes it.
- **Her coding jobs no longer stop to ask for approvals** (2026-10-14). A job that changes code ran
  in Claude Code's `acceptEdits` mode, where headless nobody can answer a prompt, so every git,
  pytest and file-listing command was denied: 14 in one job, which then asked you "may I run git,
  pytest…?". Jobs now run in Claude Code's `auto` mode: its own safety check approves routine
  commands with no prompts and still blocks risky ones (a blocked one shows up in the task as "not
  allowed to run"). `AIVA_CLAUDE_PERMISSION_MODE=acceptEdits` puts the old behaviour back. There is
  deliberately no bypass setting. Look-only jobs still run in plan mode.
- **Coding tasks: one Claude Code job per piece of work, and she checks the tests herself**
  (2026-10-13). Her last five coding tasks all failed, and the planner's splits were part of why.
  - **One job, not four.** The planner split one fix into inspect / implement / test / verify, each a
    fresh Claude Code job that re-read the repo and could fail its grading alone. The planner is now
    told one piece of coding work is one delegate step, and `tasks.one_job_per_project` merges
    back-to-back delegate steps on the same project into one job anyway (in order; a look-only
    part's "don't change anything" means only that part).
  - **Later jobs continue the same session.** Another delegate step on the same project in the same
    task, a repair re-run or a re-plan after second thoughts now runs `claude -p --resume <session>`,
    so the agent remembers what it read and changed. A fresh job couldn't even open the earlier
    job's plan file and redid the inspection. Without a session id the old report hand-off stays.
  - **A step is judged on all it changed.** A repair re-run is compared against the step's first
    snapshot, so a re-run that finds the work already done isn't "nothing changed".
  - **She runs the project's own tests.** After a coding job that changed something she runs the
    project's pytest suite herself (`tasks.project_test_command`: only through the project's own
    `.venv` with pytest installed, never a command a model wrote), once before the first job and
    once after. Failing tests cap the grade at partial, unless they were already failing before the
    job. The grader is told to trust her run over the report. The result shows up as a `tests`
    timeline event and in the step's evidence. `AIVA_TASK_RUN_TESTS=0` turns it off;
    `AIVA_TASK_TEST_TIMEOUT_S` (300) bounds a run.
- **Her wishes point outward.** All 12 of Aiva's open wishes had become introspective ("notice what
  remains when the brightness settles"), and since open wishes are fed back into every diary digest
  they kept breeding more. The diary prompt now asks only for concrete wishes aimed at him, her
  sister or the world (something to do, make, find, ask or learn, with a clear done-moment; at most
  one about reading), and the pure `inner.desire_is_outward` backstop skips an inner-state wish
  with nobody outside her in it (logged `self: inward wish skipped`). One-off data cleanup the same
  day (backup `aiva/backups/memory-pre-rut-clear-20261012-225515.db`): the 12 wishes, the
  "wonder"/"uncertainty" preferences and one meta trait dropped, the 75 book lessons faded.
- **Reading is allowed to be leisure.** Every section used to demand tone/style/subtext/devices and a
  first-person reflection, so a gag chapter got the same exegesis as a poem — and it all fed the
  traits, preferences and diary that then rendered back into the persona. Now `nuances` and
  `reflection` may be `null` on an ordinary section ("most sections don't change you, and saying so
  is a real answer"; nulls are coerced so they can't be stored as the string "None"), the spoken
  session wrap may be a moment, a line or a joke with no takeaway attached and gets the section's
  quotes to point at, the diary is told that a flat stretch deserves a flat entry and that his tone of
  voice is not an event on its own, and the sister cues ask for the small real thing from her day
  instead of a philosophical opener. `LESSONS YOU APPLY` now says they're for judgment calls, not
  conversation: never hand him a meaning he didn't ask for.
- **Her humour outlives her maxims.** `prompt_blocks` showed at most five reading lessons and dropped
  the voice notes FIRST when the persona went over budget — so under a long prompt she kept "kindness
  can become controlling" and lost "The cookie tin was empty. This was clearly a systems failure."
  Lessons are capped at three and the drop order is now lessons → reading → voice.
- **Private mode is about their own chats, not yours.** A sister exchange rooted in something the
  user said is spoken aloud even in private mode (he asked for it and is in the room); exchanges the
  sisters start themselves, or that grow out of a sister's own announce, follow the private/audible
  mode. A reply that OPENS with the other sister's name ("Hey, Saga…") is logged and overheard as
  addressed to her, not to him.
- **Sister exchanges answer statements too.** A chat a sister opens with the other always gets one
  reply however it was phrased; a line that opens with her sister's name hands her the floor; and
  indirect questions ("I keep wondering whether…", "I'd like to know…") count as asking. Before, four
  private chat openers in a row went unanswered because none ended in a question mark. Arbiter
  timeout 2.5 → 4 s. Persona/tool text: what he asks for is always spoken; private covers only their
  own chats. Her avatar keeps its eyes open for a minute after a turn of her own even while the
  house sleeps; the dashboard's per-sister awake/asleep guess now recognises "house asleep".
- **Background news waits for a quiet moment.** Announces (reading session done, task done) and a
  sister's own initiatives are held by the floor until nobody has spoken — user or sister — for
  `AIVA_ANNOUNCE_QUIET_S` (default 45 s) instead of one second after the user; a running sister
  exchange continues meanwhile. Two announces queued by the same sister collapse into one turn (her
  context already holds both notes), so she doesn't say the same news twice.

### Fixed
- **Aiva and Saga draw on one canvas, not two windows** (2026-10-21). Asked to tag-team a drawing,
  each sister opened her own window. Each one launched and looked for a window under her own name
  (`board_for(ch.slug)`, `find_running(author)`), so Saga never found Aiva's. Now the window is named
  for the canvas, not for who draws. Both boards use `drawing.SHARED_CANVAS` ("shared"): either
  sister finds it with `find_running("shared")` or launches it with
  `launch(canvas="shared", author="user")`. Every stroke, undo, clear, save and open still sends her
  own author, so strokes keep her ink and tag, and her undo never takes back her sister's.
  Strokes he draws by hand there are tagged "user", so they're his and no longer count as hers.
  The two of them opening it at the same moment still get one window, because
  `launcher.opening("shared")` holds an OS lock across find-then-launch. Without that lock the same
  race opened two. Their old canvases aren't lost: the first shared window takes in
  `aiva-autosave` and `saga-autosave` (`--absorb`), strokes keeping their authors, as long as their
  own windows are closed. It autosaves the merge first and only then renames each old copy
  `*.merged-<time>.aivadraw`. The drawing app is now 0.5.0 (`--canvas`, `--absorb`,
  `launcher.opening`). `clear_drawing` now wipes her sister's strokes too. It was already ask-first,
  and it keeps a before-clear copy. Tests: `drawing-app/tests/test_storage.py`,
  `drawing-app/tests/test_launcher.py`, and `tests/test_drawing.py`, which runs real offscreen windows
  for two sisters on one window, for both opening at once, and for the old canvases being absorbed.
  Needs both sisters restarted. Close the two old per-sister canvas windows first, so their
  drawings join the shared one when it opens.
- **The dashboard showed the sisters talking to him while they talked to each other** (2026-10-21).
  On "Hey Saga check out Aiva's latest app! Tell her Aiva", both replies went down as "→ you":
  - Aiva's "Yep, Saga, you're looking at it…" was to Saga, but "yep" was not one of the small
    opening words the name check skips, so it saw no name. `sisters._OPENERS` now has yep, yup,
    nope, sure, wow, cool and similar fillers, and an em/en dash splits words ("hey—Saga").
  - Saga's "Oh, hey—you're back. Aiva, Frank wants you to…" answered him and then turned to Aiva.
    Only the first sentence counted, so it was "to him". The new `sisters.spoken_to` labels such a
    line `both`: the dashboard shows "→ both" and her sister overhears "said to you and him". A
    `report` stays his, whatever it says.
  The same label decides the overheard line, and a line that calls her sister by name gives that
  sister a turn, so Saga now gets to answer "Yep, Saga, …". Takes effect when the house restarts.
- **A finished report was graded "truncated", and she redid the job four times** (2026-10-17). "Take
  a look at the README and tell me what you'd change" ran four Claude Code jobs (~$8.60 list price)
  and still ended "failed". Each report was complete; the cut was on our side:
  - the runner cut a report to 4,000 characters with a bare "[…]" in the middle;
  - the grader then kept only the first 4,000 characters of that, so it ended mid-word ("…packaging
    fi"). Graded "visibly truncated", it was rerun, and each rerun wrote a longer report.

  Now:
  - The grader and second thoughts see up to 20,000 characters (`JUDGE_REPORT_CHARS`), cut only
    in the middle.
  - A cut says what it is ("[… N characters left out here …]"), and the grading rules say it's
    never the agent's doing.
  - The rules also say plan mode's own plan file under `~/.claude/plans` isn't an edit, and its
    "I'll wait for your go-ahead" isn't a question. The first run had been marked down for both.

  Replayed with the real grader, both reports that had been graded partial now came back good,
  3 runs out of 3. Two more problems are fixed:
  - **A plan-mode job is never blamed for changes it couldn't make.** The last run failed "for
    modifying tasks.py": those were edits to the runner made in Frank's own Claude Code session
    while the job ran. They now go to the grader as `changed_by_others_meanwhile`.
  - **She can tell you what the job found.** The done note carries the agent's report (up to
    3,000 characters). Before, all she had was a two-sentence summary made from 300 characters
    of it.
- **Aiva kept asking to run a check she'd already run twice** (2026-10-17). She had eight open
  wishes that were all "run one focused check on Saga's constellation and save the result somewhere
  definite", in different words. The check ran and passed on Oct 15 and again on Oct 16, and she
  agreed when you told her she'd done it. She still raised it at 11:49, 12:46, 13:01 and 13:17.
  Nothing ever closed a wish except her acting on it alone, and every time she brought it up, her
  diary wrote it down again. Now a wish ends when it's done (`SelfEngine.settle_wishes`):
  - **A finished task closes the wishes it did.** `TaskRunner(on_result=)` hands the whole goal
    and result to one side-model call, which picks which open wishes it settles: all wordings of
    the same wish, never ones that are only on the same topic. Replayed on her real wishes it
    closed the 3 "run the check" ones, the same all three times. An unrelated task closed nothing.
    Without the model, only a near-restatement counts (cosine 0.72).
  - **"You already did that" closes it.** `drop_desire` has `done=true` (status `fulfilled`), and
    every wording goes, not just the nearest. The side model decides when his words don't match a
    wish exactly. The old similarity guess had taken "the focused Saga constellation check" for
    "whether the constellation feels emotionally right to Saga".
  - **A wish she has raised gets a rest**: a day, then three days, then a week (`ASK_REST_H`). A
    weaker wish that isn't resting gets its turn in the meantime.
  - **Her diary sees what's done**: an "ALREADY DONE" line lists what she settled in the last
    three days.
  - Her 9 stale Saga-check wishes were closed in her memory by hand.
- **You had to play middle man between the sisters** (2026-10-14). "Bring Saga in and ask her if she
  wants to change anything about the look, then let me know" took four more nudges from you. Saga
  booted and nothing happened ("ok she's up, ask her what I told you"). Aiva said "I'll ask her…"
  instead of asking. A "Saga, describe the actual look…" in the middle of Aiva's reply never reached
  Saga ("ask her"). And when Saga answered, nobody told you. The floor now runs the errand to the end:
  - **Asked means asked.** When you tell one sister to ask, tell or check with the other
    (`sisters.errand_to`), her turn carries `errand_cue`: she puts the actual question to her sister
    in that same reply, starting with her name.
  - **Picked up on arrival.** A line meant for a sister who isn't running is remembered. If she starts
    within `ERRAND_TTL_S` (3 min), the sister who took the line gets a turn (`arrival_cue`) to put it
    to her with the context she missed.
  - **Reported back.** When an exchange that grew from your errand ends, the sister you sent gets one
    `report` turn (a new turn kind, `report_cue`) to tell you the gist and what happens next. That
    turn never restarts the sisters. "Ask your sister something" (just "go talk") gets no report.
  - **Who she's talking to.** A call to her sister anywhere in a reply ("…fair. Saga, describe…")
    now hands Saga the floor (`sisters.called_in`). A line about her ("Saga's already here", "Saga is
    starting up") no longer counts as talking to her; the possessive once opened a sister round. A
    call needs a comma or dash after the name, a greeting before it, or a question.
- **Aiva's verification got stuck in Claude Code's plan mode** (2026-10-14). She was to check Saga's
  constellation code, run a focused test and write the result into a desktop note.
  - **A look that finds something missing has its answer.** The look-only job did its part (35 tests
    passed, and there is no speech dimming in the code), but the grader flagged "there is no speech
    dimming" as a broken premise, and the task stopped to ask you instead of writing the note. The
    real grader still does that every time, so the runner decides: on a look-only step with no
    coding planned after it, a premise is logged as a `finding` and the plan goes on. With coding
    still to come (the Miso renderer that never existed), it still gets second thoughts. The grader
    also now sees the step's `later_steps`, so it doesn't mark a look down for the note a later
    step writes.
  - **"Retry outside plan mode" now runs outside plan mode.** The planner marked the retry
    `read_only: false`, but its prompt said "without changing any implementation files", which the
    runner's wording check read as look-only. The job went back into plan mode and the note's write
    was refused. The planner's flag now always wins (`tasks.delegate_read_only`; a string "false"
    counts as false). The wording is only read when no flag was given, and only a whole "don't change
    anything / files / the code" counts, never a partial "don't touch the API". A headless probe
    confirmed a resumed session does leave plan mode with `--permission-mode auto`, and can write
    outside its project folder.
- **Saga's retried note failed at the last step** (2026-10-14). Everything for the note was gathered;
  then the writing step gave up on its very first move, saying a new file "requires the shell
  confirmation", with `create_file` in hand. That couldn't be reproduced in 15 replays of her real
  brief on the real model, so it's rare, but one refusal sank the whole task.
  - **A step that gives up without trying a single tool is sent back once** ("look at what your tools
    do and try"), logged `pushed_back`. A second `step_failed` is accepted.
  - **The task agent is told writing a file is `create_file`, already approved**, and that the shell
    is never for that.
  - **`create_file` never replaces an existing file unless `overwrite` is true.** A taken name gets
    ` (2)` and the result says so, so there is nothing to check first, and a voice "make notes.txt"
    can't wipe one either.
  - **Search no longer opens audio and model files.** Her first repo search took two minutes,
    reading ~1 GB of wake-word training audio only to find each file binary. Known binary types are
    skipped unopened and only the first 8 KB is sniffed before a file is read: 0.35 s now.
  - **A screen check sees what the step reported.** "The observation is recorded" was judged on a fresh
    screenshot alone, failed, and used up the task's only repair.
- **She answered her own task questions** (2026-10-14). On her own news turns she called
  `answer_task("yes")` herself, nine times in one verification, so her commands ran without you.
  A task's question is now answered only on a turn where you spoke or typed; otherwise she's told to
  ask you and wait.
- **A task question could be dropped and she'd wait in silence** (2026-10-14). The next question
  arrived while she was answering you, was folded into her older queued note, and that note was then
  dropped as "already said". A folded-in note now counts as new, so it gets its own turn.
- **Her sister answered a question meant for you** (2026-10-14). Saga asked you "…search the Aiva
  repo…; should I let it run?" and Aiva replied "Yes, let it run", because the line named Aiva and
  asked something. News a sister relays to you no longer starts a sister round unless it opens by
  addressing her sister.
- **A coding job ran across your whole home folder** (2026-10-14). The planner wrote the project as
  `.`, which means your home, so Claude Code worked from `C:\Users\Extre`. Git was out of its reach
  there, and she couldn't see what it changed or run the tests. The planner now gets the folder her
  own code lives in (`your_own_code`). A plan that points a coding step at your home is sent back
  to be fixed, and Claude Code (task or voice tool) refuses to start there.
- **A coding job that did its work was graded as having changed nothing** (2026-10-13): to see what
  Claude Code changed, the task runner compares `git status` from before and after the job. It ran
  git through `execute_shell`, which keeps only the last 1000 characters of output. With about 95
  uncommitted files the status is ~2,500 characters, so she only ever saw the end of the list (mostly
  untracked tests) and none of the tracked files the agent edited. A job that fixed Saga's sky
  renderer and passed 837 tests was graded "not evidenced", re-run, then graded "nothing in the
  project changed", and the task hit its time limit. Git now runs through its own `tasks.run_git`,
  which keeps the whole output, off the event loop, and no longer counts against the task's tool
  calls. A failed snapshot no longer reads as "nothing changed".
- **A request made as the first thing after a restart was only answered, not done** (2026-10-13): the
  one-shot boot note ("say you're back in one short sentence") won over his "new task" line, so she
  said she'd start the task but never called `start_task`. Both greeting notes now tell her that a
  request in the first line comes first, with tools as usual, and the greeting is only in passing.
- **A reply after a slow tool still went missing from the transcript** (2026-10-13, "say yes and I'll run it" after `start_task`): pipecat runs parallel tool calls in their own task, so the per-call `FunctionCallInProgressFrame` often arrived *after* the response's end frame. The `PrivacyGate` saw a response with no tool, ended the turn, and the house released it (`end:private`) before the re-run; the real reply then ran on no granted turn ("stray response" on the house, text-only, never logged). The gate and the `MindWatcher` now also take `FunctionCallsStartedFrame`, which the LLM pushes before the end frame, as the tool signal (the watcher reports each call's +1 once, whichever frame comes first).
- **A reply that came after a tool call was lost**: pipecat ends an assistant turn at every LLM response end, including the one that only calls tools, so the sister closed her granted turn (and the privacy latch) before the LLM re-ran with the tool results. The real answer then ran as "no granted turn": forced text-only (never spoken even on audible turns), never sent to the house, so missing from the Conversations tab (`said outside a granted turn` in the log). The end of a turn is now decided by the `PrivacyGate`, straight after the LLM, where her responses are still in true order: a response that called a tool is followed by the re-run, the first one that ends without a tool ends the turn, and the `said` carries the text of every response in it (words before the tool included). The assistant aggregator only closes a turn on an F10 interruption. Two earlier attempts decided it at the aggregator, after the TTS, and failed there because the ElevenLabs service holds each end frame until its audio finishes and drops the end of a response it already closed: counting tool calls held back every spoken reply after a tool; a "no response since the tool" flag still lost the reply when she spoke before calling the tool. Response/tool reports to the house fall back to the last granted turn id, and a turn still open when the next one is granted is closed and handed over first.
- **A step that did the whole job failed the task**: the step agent (asked to open a terminal) went on to inspect, delete and verify; the next step ("inspect before deleting") then failed because the folder was gone. The verifier now gets the task's goal and answers `goal_done` when the goal is already achieved, which marks the rest of the plan `not_needed` and finishes as a success; the step agent is told to do only its own step.
- **A task's question could be swallowed by the house**: when she answered "yes" with `answer_task`, the task's next step queued its confirm question as an announce *while she was still replying*; her `said` then dropped every older queued announce of hers as "spoken already", so the question was never asked and the task sat in `awaiting_user`. The floor now only drops announces queued before the turn was granted (`Turn.granted_at`).
- **A shell step that worked was failed by its check, and the repair opened a visible terminal**: a shell step planned with `verify: "terminal"` was verified against a terminal session (defaulting to one called "main" that didn't exist), so a successful command "failed" and the repair agent went off to `open_terminal` + `run_in_terminal`. A shell step is now always verified by its own output (stdout kept to 1000 chars in the evidence), and a missing terminal falls back to the evidence. The planner, the shell tool and the step agent now say: work in the background by default, look with `list_directory`/file tools (nothing to confirm), use a shell step only to change things, fold the check and the change into one command so he confirms once, and open a visible terminal only when he asked to see one. Her persona said the opposite ("To RUN commands, ALWAYS use run_in_terminal"), so she kept telling `start_task` to use a visible terminal; it now says background by default, visible terminal only when he asked (the golden persona file is updated to match).
- **Task shell steps and `confirm_pending_command` ran under cmd.exe**, not PowerShell: `execute_shell` used `shell=True`, so every PowerShell-syntax command she writes (`$p = …`, `Test-Path`, `Remove-Item`) failed with exit code 1 — a confirmed folder delete failed right after his yes. It now runs `powershell -NoProfile -NonInteractive -Command`, and a failed step carries the stderr line in its evidence and failure reason instead of a bare "exit code 1".
- Claude Code work was never really graded: the grader (and the feedback extractor) asked for JSON
  mode without saying "JSON", so OpenAI 400'd and every grade fell back to "partial".
- A delegate step now creates its project folder when missing (inside his home only), and the
  planner is told Claude Code is fenced to that folder — a retry once aimed it at her own repo and
  it could not reach `Desktop\prime_project`.
- Claude Code job results (+ raw JSON with session id/cost) are kept in `aiva/logs/coding/` instead
  of `%TEMP%`, and each delegated task logs a `claude` event linking it to its session.
- Task steps that raised were silently swallowed by pipecat's `create_task`, so a task could report
  success having done nothing (the 2026-09-20 prime-number "Done"); step errors now fail the task.
- Time spent waiting for his answer no longer counts against a task's 20-minute wall clock.
- `open_terminal` / Claude Code resolve `desktop\...` relative to his home like `create_file` does,
  instead of relative to the `aiva/` working dir ("directory doesn't exist" right after creating it).
- **Mic stayed gated after a sister's reply** (house log: `release … (cap)` after 120 s, console
  flooded with "Empty audio frame"): her "stopped speaking" report could lose its turn id when the
  assistant aggregator closed the turn before the audio transport reported the end of playback, and
  the floor ignored it as stray. The link now remembers the granted turn for its speaking reports
  and the floor accepts the holder's report even without an id.
- The launcher's liveness probe no longer logs a "bad handshake" in the house.
- **Stale boot greeting** (first live sister exchange): a sister's "say hello" announce queued at
  connect lingered when the user greeted her first, blocked her sister's question (the floor counted
  it as a pending turn of hers), then ran 45 s later with nothing to say. Now a sister's older queued
  announces are dropped the moment she speaks, and only queued sister/initiative turns count as
  "she already has one coming".
- **Self-wake through the wake-word pre-roll**: the wake listener's own mic stream could carry a
  speaking sister's "Hey, Saga" into STT past the self-hearing gate; wake-word wakes are now ignored
  while a sister is speaking (and during the cooldown). An audible turn granted while the house dozed
  now wakes it first, as an announce used to.

## [0.10.0] - 2026-09-11

### Added
- **Saga — Aiva's sister, and the house they share.** Aiva is no longer one process: a small
  always-on **house** (`aiva/house.py`, `start_house.bat`) owns the one microphone, Deepgram, the
  wake words (`hey_aiva` + `hey_saga`), the global hotkeys, the **floor** (who speaks, who overhears,
  private sister talk, caps) and the conversation log in `aiva/house.db`; each sister runs in **her
  own process and console** (`main.py --character aiva|saga`, `start_aiva.bat` / `start_saga.bat`)
  with her own db (`memory.db` / `saga.db`), state files, space folder, avatar renderer, ElevenLabs
  voice and restart loop, so either can be started or stopped on her own. The house grants turns over
  a loopback socket (`modules/house_proto.py`); a sister never runs her LLM without a grant, so the two
  can't talk over each other, and the mic is gated from the grant itself (not from her audio) so her
  voice never reaches speech-to-text. You address one by name, both ("you two"), or nobody — then a
  cheap arbiter side-call picks who you meant; the other sister always **overhears** what was said as a
  bracketed line in her own context. Sister-to-sister talk is **private by default** (text only — no
  ElevenLabs spend), flipped live with Ctrl+Alt+P or by asking either of them (`set_sister_talk`),
  and capped (rounds, per hour, per day). **Saga has no authored personality**: her floor is the
  shared speaking/safety rules plus "you're new, nobody wrote who you are"; her temperament forms from
  her own diary (self-forming `temperament` traits with inertia, cap 8, erosion instead of veto), her
  tastes from experience, and she learns your voice herself (she asks who you are). Each sister keeps
  a **kin** store (`modules/kin.py`) of what she has actually seen the other do, make, read or say —
  relayed by the house as public events — rendered as a "YOUR SISTER" block and answered by
  `recall_about_sister` with no LLM call, honestly saying "I haven't seen her do that" when unknown.
  Dashboard: tabs **Aiva | Saga | Conversations** (the full house log, private lines marked) plus a
  Household strip (bodies, house, connected sisters, private/audible, TTS chars per sister).
  ElevenLabs spend is now actually counted per sister (`tts`, `tts.aiva`, `tts.saga` in the shared
  caps). Saga's look is a retexture of the same Live2D mesh (`--write-model saga`), her own renderer
  instance with its own state file and skin-toned mouth camouflage. New helpers:
  `tools/pick_voice.py` (verify/preview/set an ElevenLabs voice into `.env`), `training/hey_saga.yaml`
  (wake model trained with the existing recipe). Config: the `SAGA_*` block and `AIVA_HOUSE_*`,
  `AIVA_SISTER_*`, `AIVA_HOTKEY_SISTER_TALK` in `.env`.
- **She knows her own powers, and can act on safe wishes herself** — capability self-awareness plus
  a filled-in autonomous `act` slot. New `modules/capabilities.py` is an intention-indexed map ("want
  to know → search/read; want something visual → find & keep; want to build → ask") with `auto`/`ask`/
  `never` tiers; it renders a "WHAT YOU CAN DO" persona block and backs a new `what_can_i_do` tool, so
  she reaches for — and asks for — the right power instead of staying narrow. The self-loop's
  `initiative_decision` gains an `"act"` outcome: when a top wish maps to a safe/reversible capability
  she pursues it on her own (v1: find-and-keep an image to her space), emits a `self_act` experience
  event (so doing it shapes her mood/diary/preferences — learn-by-doing), and tells you. Anything that
  spends, touches your files, or edits her own code (Claude Code builds included) stays **ask-first**;
  `AIVA_SELF_ACT=0` and `set_initiative('quiet'/'never')` switch autonomy off.
- **Cost guardrail so she can't run away on spend** — `modules/notify.py` (ntfy push) + `modules/
  limits.py` (`ServiceLimits`). No global dollar budget by design: each metered service is capped at
  its own budget/free tier (Tavily monthly, `AIVA_TAVILY_MONTHLY_CAP`; optional ElevenLabs/Deepgram
  monthly caps), you get an **ntfy alert at ~80% and 100%**, and hitting a cap **pauses only her
  self-started work that uses it** — the live voice never stops. A dashboard **"Limits & what's
  paused"** panel shows usage vs cap and exactly what's paused and why. Config: `NTFY_TOPIC`/`NTFY_URL`,
  `AIVA_NTFY`, per-service caps.
- **She can reach the internet, and keeps a curated "space" for what she finds** (`modules/web.py`,
  `modules/space.py`). New tools `search_web` (info/images/videos via Tavily), `read_page` (fetch +
  summarise off the voice model's token bucket), `look_at_image` (she actually *sees* an online image
  with her vision and judges it), `save_to_space`/`hand_over` (download → keep → open the file for you +
  give the source link), `space_status`/`recall_from_space`/`drop_from_space`. Finds live in `aiva/space/`
  and a `finds` table (deduped by sha256/subject, recalled by embedding, vetoable), shown on the
  dashboard ("Her space"). She can fetch on request *and* on her own initiative (tied to her wishes).
  **Hard guardrail: she finds and shows — she never buys, signs in, pays, or sends; there's no such
  tool.** Downloads are gated to image/text/pdf/mp4 with a size cap (`AIVA_WEB_MAX_MB`) and the file
  extension comes from the content-type, so an executable can never be written. Config: `TAVILY_API_KEY`
  + `AIVA_WEB_*` / `AIVA_SPACE_DIR`. `python -m modules.space --show` audits her finds.
- **Her own likes, dislikes, and opinions — that genuinely evolve** (`modules/preferences.py`). A new
  mutable self-model with *signed* polarity (a real dislike, not just an absent like), a `taste` and
  an `opinion` kind, formed from *any* experience (not just reading). Her frozen baseline tastes
  (cookies, cats-over-dogs, nature, cozy, Japan, music, learning) are now the *starting seed*, not a
  floor — they can drift or even reverse. Coherence comes from **inertia**, not a veto: a preference
  erodes toward neutral and only flips after sustained counter-experience (baseline-seeded ones move
  ~half as fast), so she evolves rather than flip-flops. Capped/faded/vetoable like traits; shown on
  the dashboard ("Where she stands"); `python -m modules.preferences --show` audits them. Formation
  piggybacks the existing journal digest (no new LLM call). New voice tools `note_preference` /
  `drop_preference`. New persona block "WHERE YOU STAND".

- **Boot ping over ntfy** — she sends a push to your topic when she comes up ("Aiva is awake" /
  "restarted"), fire-and-forget so it never blocks startup. Toggle `AIVA_NTFY_ON_START`.
- **Her dashboard opens in the background at boot** — `main.py` launches the read-only dashboard
  (and a browser tab) when she starts, skipping it if one's already up so a restart doesn't spawn a
  duplicate or re-pop the tab. Toggles: `AIVA_DASHBOARD_AUTOSTART`, `AIVA_DASHBOARD_OPEN`.

### Fixed
- The web/space tools were whitelisted for the task runner but never merged into its ToolBox, so
  "find me X" / build tasks couldn't actually call them — `ToolBox.add_handlers` now merges them in.

### Changed
- **PERSONALITY split**: her *tastes* moved out of the frozen "Personality Traits:" list into the
  mutable preferences layer; only her *temperament/voice* stays fixed. `baseline_traits()` (becoming's
  anti-contradiction gate) now protects her character, not her tastes — so a book (or a chat) can
  genuinely shift what she likes without fighting the gate.
- Reading is now **length-based**: a session reads to a target word count and stops at a natural
  chapter/section boundary, instead of a fixed number of sections (so short poems no longer finish
  in seconds and long chapters aren't over-read). Knobs: `AIVA_STUDY_WORDS`, `AIVA_STUDY_MAX_SECTIONS`.

### Fixed
- She no longer announces "I'm starting to read…" twice — the engine's start note is now silent
  (she already says it via her reply / the study_book acknowledgement).

## [0.9.0] — 2026-08-28

First tracked version — the baseline of everything built so far.

### Added
- **Streaming voice pipeline** (pipecat 1.5.0): Deepgram STT → gpt-5.6-luna (native tools) →
  ElevenLabs TTS, Silero VAD + local smart-turn v3 endpointing, ambient wake-word mode with
  two-stage transcript-verified wake, self-hearing gate, idle/sleep, restart machinery.
- **Her own Live2D avatar** — self-rendered (no VTube Studio) in a transparent, click-through,
  monitor-sized window; lip-sync + spectral visemes; expressions; a reading book-prop.
- **Desktop pixel cat** (Miso) with persistent needs/mood and pet/feed/play care.
- **Computer use** — screen vision, click-by-name via UI Automation, mouse/keyboard, apps,
  sandboxed files, real visible terminals (incl. ssh/REPLs), Claude Code delegation with grading.
- **Long-term memory** — sqlite (WAL + backup-on-boot): transcripts, extracted facts, session recaps.
- **Multi-step task runner** — plan→act→verify→repair with experience-based autonomy, reusable
  flows, and accountable delegation.
- **Reading & becoming** — she reads/studies her library section by section, synthesizes whole
  books, recalls/discusses them, and is bounded-ly changed (traits/voice/lessons, evidence-gated).
- **Nervous system / selfhood** — a shared mood (valence/arousal/social/focus), an experience bus,
  a self loop that keeps a private diary and forms desires, mood-coloured persona blocks, and
  speech-only initiative (she can bring things up on her own).
- **Speaker recognition** — open-set voiceprints (CAM++ ONNX, on-device): she learns voices and
  asks when she meets a new one.
- **Vocal affect** — coarse paralinguistics (energy/pitch) that nudges her mood to mirror how you sound.
- **Live dashboard** (stdlib, read-only) at :8760 — mood/diary/wishes, reading, tasks, cat, memory,
  capabilities, and a Statistics operations profiler; **operations metrics** (turn latency, LLM/tool costs).

[Unreleased]: https://example.invalid/compare
[0.9.0]: https://example.invalid/releases/0.9.0
