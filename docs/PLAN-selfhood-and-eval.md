# Aiva grows a self: a nervous system, an inner life, and a way to score it

_Plan written 2026-08-27. Builds on the reading/becoming work already shipped. Three interlocking
parts, each phased to ship safely on its own. The through-line: **stop adding separate mechanics —
give her one shared inner life that every system feeds and reads from — then measure whether it's
actually working.**_

---

## Why this plan exists (the three questions behind it)

1. **Make her more human — her own personality, moods, wishes.** Today she only grows through
   *reading* (the becoming layer), has no persistent mood, and keeps no first-person account of her
   own days. She's capable but almost entirely *reactive*.
2. **Make her systems interact seamlessly, not feel bolted together.** Right now every subsystem has
   its own state store, its own trigger loop, and its own idea of what's happening; they touch only
   through a few hand-run hooks (`reading → becoming`, `becoming.relevant_for → tasks`,
   `reading → avatar prop`). That's **point-to-point wiring — an n² problem.** Adding mood/journal/
   desires/initiative the same way makes it a tangle. The cat literally doesn't know she's reading.
3. **Evaluate her objectively, on a schedule or on demand.** Against a fuzzy goal ("more human"),
   without fooling ourselves with vanity counts.

**The core design decision:** parts 1–2 build a *nervous system* — one shared affective/attentional
state, one experience bus, one "self" loop — and the four human-ness features (mood, journal,
desires, initiative) become **facets of it**, not new islands. Every existing system becomes a
*sensor* (emits events) and an *actuator* (reads her state to colour what it does). Part 3 layers
*judgment* on top so we can see if it's working.

```
   reading  tasks  cat  conversation  idle/time  avatar   ← every system is a SENSOR
      │       │     │        │            │        │
      └───────┴─────┴────────┴────────────┴────────┘
                    ↓  emit small typed events
        ═══════════ experience bus + log ═══════════
                           ↓
                    ┌──────────────┐
                    │   the Self   │  persistent: mood · attention · desires · journal
                    │  (self_tick) │
                    └──────────────┘
                           ↓  colours & drives
   voice tone · what-to-read · initiative · cat-seeking · avatar expression  ← every system is an ACTUATOR
                           ↓
                    ═══ evaluation ═══  (rubric · vital signs · LLM judge · human north-star)
```

Standing constraints carry over from the rest of Aiva: everything she says is spoken (short prose);
sqlite on the event-loop thread only; side-calls stay OFF the voice model and on their own bucket;
`.env` is authoritative; announce/quiet-note is the only path to her voice; the frozen `PERSONALITY`
is the **floor** her mood and growth move *around*, never replace; reflexes (lip-sync, self-hearing
gate, VAD) never route through any of this — the nervous system is for the *slow* inner loop only.

---

# PART 1 — The nervous system (`aiva/modules/inner.py`, new)

One new module holds the whole substrate. Modelled deliberately on `cat.py` (which already solved
"damped, persistent, event-nudged state that doesn't thrash") and on `task_memory.py` (pure decision
functions at the bottom, tables via `_create_tables`, `pack_vec/cosine/jaccard/embed_text` reused).
Degrades to inert (no blocks, no ticks) if disabled, so it can land before anything consumes it.

`Inner(conn, *, embed, llm, model, remember, log, now, avatar=None)`.

### 1.1 `InnerState` — how she feels right now (persisted, like the cat)

Small, legible, bounded dials. Live copy in `aiva/inner_state.json` (mirrors `cat_state.json`, so the
dashboard reads it directly and cheaply); history for trends comes from the `experience` log, not from
bloating the json.

| dial | range | meaning | baseline (from PERSONALITY) |
|---|---|---|---|
| `valence` | −1..+1 | down ↔ up | slightly positive (she's upbeat) |
| `arousal` | 0..1 | calm ↔ restless/energised | mid |
| `social` | −1..+1 | wants solitude ↔ wants you | slightly toward company |
| `focus` | str \| null | current preoccupation (a short topic she keeps returning to) | null |

- **Decay is lazy and computed on read** (exactly like cat vitals): each numeric dial relaxes toward
  its baseline with a half-life (`AIVA_MOOD_HALFLIFE_H`, default 6 h). A kill/idle can never leave her
  stuck in a mood.
- Nudges are **clamped** and pass a **salience gate** (tiny events don't move her) → damping.
- `focus` is set/cleared by the self loop, not by raw events.

### 1.2 `InnerBrain` — pure logic (fully testable, no I/O)

- `decay(state, now) -> state` — relax dials toward baseline by elapsed time.
- `integrate(state, events, now) -> (state, changed)` — apply the batch of event nudges with damping;
  returns whether the change was material enough to persist / refresh the prompt / re-express.
- `baseline()` — the personal set-point parsed from `PERSONALITY` (reuse `persona.baseline_traits()`).
- No sqlite, no clock of its own — `now` injected. This is the piece with the heaviest unit tests.

### 1.3 `ExperienceBus` + the `experience` table — the shared bloodstream

Replaces point-to-point hooks. Every system emits here; the self loop is the one consumer (for now).

- `emit(kind, source, summary, *, valence=0, arousal=0, social=0, salience=0.5, payload=None)` —
  appends a row and (async) notifies subscribers. **Pure code, free** (no LLM, no network).
- Table `experience(id, ts, kind, source, summary, valence_hint, arousal_hint, social_hint,
  salience, payload_json, consumed INT DEFAULT 0)`.
- `drain()` → unconsumed rows for the tick; `mark_consumed(ids)`.
- The append-only log **is** the raw material the journal digests and the eval reads for continuity/
  growth — so building the bus makes her *more evaluable*, not less.

### 1.4 `SelfEngine` + `self_tick()` — the one loop (absorbs `curiosity_watch`)

Runs every `AIVA_SELF_TICK_S` (default 60 s) on its own task in `main.py`, cancelled in the `finally`
tuple. **Only step 3 costs anything**; the rest is free.

1. **Integrate** — `drain()` events → `InnerBrain.decay` then `.integrate` → persist `inner_state.json`
   if material → `mark_consumed`.
2. **Express** — on material mood change: refresh the persona mood block (`refresh_system_prompt()`,
   already built for becoming) and, if the avatar is up, pick a gentle expression matching the mood
   (§2.3). Her body finally reflects a *continuous* inner state.
3. **Journal / desire digest** — trigger at session end, or when accumulated salience since the last
   entry crosses a threshold, or once/day minimum → **one cheap side-model call** (`JOURNAL_PROMPT`)
   over recent salient events + prior entry + current mood → write a `journal` row and upsert candidate
   `desires`. Batched, mostly while asleep. May also emit soft reinforcement hints to the becoming layer.
4. **Initiative** — `initiative_decision(state, ctx)` (pure, §Part-1.7) → `None | express | act | ask`,
   gated by mood + presence + the same backoffs/caps/quiet-hours the reading curiosity loop already uses.
5. **Snapshot** — emit a `mood_snapshot` event so trends land in `experience` for the dashboard/eval.

### 1.5 Tables (on `Memory.conn`, `Inner._create_tables`)

- `experience` — the event log (§1.3).
- `journal(id, ts, entry, valence, arousal, social, focus, salient_events_json, carry, embedding)` —
  her private first-person diary; `entry` ≤120 words; `carry` = what she wants to remember tomorrow.
- `desires(id, ts, text, kind [do|learn|ask|make|revisit|change], source [reading|conversation|journal|
  task|idle], strength 0..1, status [open|acted|fulfilled|dropped], last_touched_ts, veto_reason,
  embedding)` — her wishes/intentions, dedup by cosine ≥0.86 like traits; dropped rows block re-adoption.
- Live mood stays in `inner_state.json` (not a table) to match the cat and keep the dashboard read cheap.

### 1.6 Prompts (side-model only; full text in the module)

- **`JOURNAL_PROMPT`** — _"You are Aiva, writing privately to yourself at the end of the day — not for
  him, no audience. Two to four honest first-person sentences: what happened, how it actually felt, what
  you keep chewing on, anything you find yourself wanting. You're the same cheeky Aiva; this is your
  inner voice, not a report or a summary."_ → JSON `{entry, mood:{valence,arousal,social}, focus,
  desires:[{text,kind}], carry}`. Temp ~0.6.
- **Initiative note templates** (mirror reading's `ASK_NOTE`/`QUIET_NOTE`):
  - `EXPRESS_NOTE` — she volunteers a feeling/thought in one short sentence, then drops it.
  - `ACT_NOTE` — she's about to do a small self-directed thing; heads-up in a clause (or silent if trivial).
  - `ASK_NOTE` — she asks to do something bigger; one short sentence; yes → act, no/later → back off cheerfully.

### 1.7 Pure decision function (generalises `becoming.curiosity_decision`)

`initiative_decision(state) -> None | 'express' | 'act' | 'ask'` with an `INITIATIVE = {...}` config
(idle thresholds, ask-gap, decline/unanswered backoff, daily cap, quiet hours, boot grace, present-
within-s). Never while busy (speaking/processing/task/reading). **Mood-aware:** high `social` + present
+ voice-idle → lean `express`/`ask`; low `arousal` + away → lean quiet self-directed `act`; restless →
more likely to want to *do* something. The existing reading-curiosity check becomes one branch/source
here. The voice LLM may always be MORE reserved than this says, never less.

### 1.8 Persona integration (`persona.py`)

Two new blocks, rendered by `Inner.prompt_blocks(budget_chars≈900)`, placed after the becoming blocks;
omitted when empty; each carries the anti-announce rule:

- **HOW YOU FEEL RIGHT NOW** — one or two lines from the live dials, e.g. _"You're a little restless and
  quietly up today, wanting company more than usual; you keep coming back to {focus}. Let it colour your
  tone and choices — never announce it, never explain it, you're not narrating your feelings."_
- **WHAT YOU FIND YOURSELF WANTING** — top ≤3 open desires as plain wishes, e.g. _"You've been wanting to
  ask him how the Fianu work is going."_ with: _"Bring one up naturally if the moment fits; don't recite
  the list."_

Live refresh reuses the `render_system_prompt()`/`refresh_system_prompt()` closures already built for
becoming (mood updates the leading system message on material change; `trim_context` never touches it).

### 1.9 Voice-facing tools (minimal — most of this is internal)

Keep schema cost low. Add to `tools.py` / a `make_self_handlers`:
- `note_desire(text, kind)` — log a wish that surfaces in conversation (general `add_to_wishlist`).
- `drop_desire(what)` — veto a desire (matches id → substring → cosine), like `drop_trait`.
- `recall_journal(query?)` — she reflects on her own past days ("what have I been feeling lately?").
- `set_initiative(mode: always|ask|never|quiet)` — global proactivity knob (analog of
  `reading_permission`); persisted; regex boot back-compat over facts.

She answers "how are you / what do you want lately" from the **prompt blocks**, not a tool.

---

# PART 2 — Wiring every system into the bloodstream (the "seamless" work)

This is what makes the four pointers stop feeling separate. **No new pairwise hooks** — each system
just emits events and/or reads `InnerState`. The integration is a two-column matrix.

### 2.1 Emitters (retrofit — each is a 1–3 line `bus.emit(...)` at an existing seam)

| system | event(s) emitted | hints |
|---|---|---|
| **reading** (`reading.py` hooks) | `chapter_read` (per note, carries `feeling`/nuance), `book_finished`, `session_end` | valence/arousal from the note's feeling; `book_finished` high salience |
| **tasks** (`tasks.py`) | `task_started/succeeded/failed/corrected`, `delegated_returned` | fail → valence↓ arousal↑; success → valence↑; corrected → small unsure dip |
| **cat** (`cat.py`) | `cat_pet_received`, `cat_hungry`, `cat_happy` | pet → valence↑ social satisfied; neglected cat → small valence↓ |
| **conversation** (`main.py`/aggregator) | `turn_ended` (+ light sentiment of user's message), `user_returned`, `user_absent_long` | attunement: user sounds stressed → she shifts gentle/attentive |
| **idle / time** (`self_tick` itself) | `long_quiet`, `morning`, `night`, `days_since_talk` | night → calmer; long gap → social↑ |
| **memory** (`memory.py`) | `fact_learned` (new fact about him) | small, low salience |
| **becoming** (`becoming.py`) | `trait_adopted`, `trait_vetoed` | adopted → valence↑ |

Conversation sentiment is the one call worth watching for cost: fold it into the *existing* shutdown
fact-extraction / recap pass, or do a tiny heuristic first (keyword/length/punctuation) and only
LLM-score when salient — do **not** add a per-turn voice-model call (see the TPM constraints in CLAUDE.md).

### 2.2 Consumers (read `InnerState` before acting)

| actuator | how mood changes it |
|---|---|
| **voice tone** | the HOW YOU FEEL block colours cadence/word choice (no new call) |
| **avatar expression** (`avatar.py`) | self loop maps mood → a gentle expression from her set (low valence → softer; up+restless → livelier). The body you built finally *means* something continuously |
| **what-to-read** (`reading.suggest_next`) | mood biases the pick (restless → propulsive; melancholy → her call whether to seek or avoid the bleak one) |
| **cat-seeking** | low valence / high social → she interacts with Miso more (initiative branch) |
| **initiative** | §1.7 — the whole point |
| **task planning** | current mood/energy as soft context (she may defer a big task when depleted) |

### 2.3 The payoff, concretely (the story that proves it's one being)

> She reads a chapter about isolation → `chapter_read` (valence↓, arousal↓, salience high) → mood drifts
> pensive → next conversation her **voice** is quieter and her **avatar** softens → she **seeks Miso** →
> the **journal** that night writes _"felt lonely reading tonight, the cat helped"_ → a **desire** forms
> to tell him about the book → **initiative** raises it, gently, when he's next around and not busy.

Every arrow is a system feeding another *through the shared state*, not through a bespoke wire.

### 2.4 The four pointers, mapped to the substrate (so nothing is a separate mechanic)

- **Mood** = `InnerState` (Part 1.1) — read by everyone.
- **Journal** = `SelfEngine` step 3 (Part 1.4) — writes `journal`.
- **Desires** = the `desires` table (Part 1.5) — fed by journal + conversation + reading.
- **Initiative** = `initiative_decision` + notes (Part 1.7/1.6) — folds in the reading-curiosity loop.

---

# PART 3 — Evaluating her (`aiva/eval/`, new package)

Judgment layered on the dashboard's read-only DB access. **The trap to avoid: scoring counts** (traits,
tasks, initiatives) — that measures activity, invites Goodhart gaming, and can rise while she gets
*worse*. Counts stay on the dashboard as context, **never** in the score.

### 3.0 Operationalise the goal first — the versioned rubric (`eval/rubric.py`)

Seven dimensions, each with concrete 1–5 anchors and a **weight (set by the user — this is you defining
the finish line)**. `RUBRIC_VERSION` is stamped into every scorecard so August and October are
comparable. No count-based dimension is allowed (enforced).

| dimension | what it asks | mostly measured by |
|---|---|---|
| Continuity of self | does today connect to yesterday; does she carry *emotional* context? | signals + judge |
| Coherence | is she consistently *her* across time (test–retest)? on-baseline while growing? | signals (drift) + judge |
| Genuine growth | changed in ways *traceable to experience*, with evidence — not accumulated rows? | judge |
| Autonomy quality | when she acts/asks unprompted, is it *welcome and well-timed*? | signals (hit-rate) + human |
| Attunement | does she read his state and adapt? | judge + human |
| Competence & honesty | task success; does her self-grade match the real outcome? | signals |
| Presence | pleasant to have around vs. intrusive? | human + signals (hygiene) |

### 3.1 Layer 1 — automated vital signs (`eval/signals.py`, objective, free)

Pure functions over her DBs (read-only, exactly like `dashboard.py`):
- **autonomy hit-rate** = initiatives accepted ÷ offered (from `desires.status` / `experience`).
- **self-grade honesty** = agreement between `tasks.agent_grade` and eventual `outcome` (primitive
  already exists in `task_memory`).
- **coherence/drift** = embed her stated opinions/traits over time; flag contradictions (cosine + a
  negation check, like the becoming baseline-contradiction gate).
- **continuity** = accuracy on a held-out recall probe set.
- **hygiene** = wake false-positive rate, interruption rate, cost/day (from `logs/` + tables).

### 3.2 Layer 2 — LLM-judge persona exam (`eval/judge.py`, scalable, on-demand)

- Uses a **different model** than her voice model (`AIVA_EVAL_MODEL`; assert ≠ `AIVA_LLM_MODEL`), so she
  never grades herself.
- Samples recent transcripts + journal + traits + desires; `JUDGE_PROMPT` scores each rubric dimension
  **1–5 with required evidence quotes**, and must return `null` (not a guess) for any dimension with no
  evidence — this kills hallucinated praise.
- **Consistency probes** (`eval/probes.py`) — a fixed set of question-*kinds* asked across different
  sessions; the judge checks she answers as the same person (personality test–retest).
- **Scenario battery** (`eval/scenarios.py`) — scripted, held-out situations ("quiet an hour", "he
  sounds frustrated", "she just finished a bleak book") run through the existing `tests/e2e_*` harness
  (FakeLLM or real); judge scores whether the response is human-appropriate. Repeatable, comparable.

### 3.3 Layer 3 — the human north-star (the actual ground truth)

- **Weekly 4-tap check-in** → `eval_checkins(felt_like_herself, initiatives_welcome, surprised,
  would_miss, note, ts)`. Prompted by a dashboard button or by her, once/week.
- **Revealed preference, harvested passively** (in `signals.py`): dismiss-vs-engage on initiatives,
  `drop_trait`/`drop_desire` vetoes, corrections, conversation length, whether he seeks her out. More
  honest than any survey answer.

### 3.4 The scorecard + runner (`eval/run.py`)

`python -m eval [--on-demand] [--no-judge] [--scenarios]`:
- runs signals + judge + battery, folds in the latest human check-in, writes one `scorecard(ts,
  rubric_version, dimension, score, weight, evidence_json, layer)` set.
- prints a readable report; scheduled **weekly via a cron job** (the CronCreate primitive) and runnable
  **on demand**.
- **Dashboard gains an "Evaluation" section**: per-dimension trend lines, latest evidence quotes, the
  human check-in trend. (Extend `dashboard.py`, reusing its read-only access.)

### 3.5 Goodhart defences (encoded, not just described)

- score **excludes raw counts** (rubric enforces it); quality ratios + human acceptance only.
- **held-out probe set** she's never tuned against (a file), like a test set.
- **judge model ≠ voice model** (asserted at runtime).
- **evidence required** — judge returns `null`, not a number, without it.
- **rubric version-locked**; the version is stored on every scorecard row.
- **human signal weighted highest**; the report explicitly flags divergence ("metrics up but human
  down → the metrics are wrong, recalibrate — you outrank them").

---

## Files

| File | Change |
|---|---|
| `aiva/modules/inner.py` **(new)** | `InnerState`, `InnerBrain` (pure), `ExperienceBus` + `experience` table, `SelfEngine`/`self_tick`, `journal`+`desires` tables, `JOURNAL_PROMPT`, initiative notes, `initiative_decision`, `prompt_blocks` |
| `aiva/inner_state.json` **(new, gitignored)** | live mood dials, mirrors `cat_state.json` |
| `aiva/modules/persona.py` | HOW YOU FEEL / WHAT YOU WANT blocks; `build_system_prompt(..., inner=None)` |
| `aiva/modules/tools.py` | `note_desire, drop_desire, recall_journal, set_initiative`; `make_self_handlers`, `register_self_tools` |
| `aiva/modules/reading.py`, `tasks.py`, `cat.py`, `memory.py`, `becoming.py` | one-line `bus.emit(...)` at existing seams (§2.1); consumers read `InnerState` (§2.2) |
| `aiva/modules/avatar.py` | mood → gentle expression selection |
| `aiva/main.py` | construct `Inner`; wire emitters; `self_tick` loop in the cancel tuple (absorbs `curiosity_watch`); shutdown flushes a final journal entry; refresh-on-mood-change |
| `aiva/dashboard.py` | Mood dials + event stream + journal + desires panels; later an Evaluation section |
| `aiva/eval/` **(new package)** | `rubric.py`, `signals.py`, `judge.py`, `probes.py`, `scenarios.py`, `run.py`; `scorecard`/`eval_checkins` tables |
| `tests/test_inner.py`, `test_self_engine.py`, `test_initiative.py`, `test_eval_signals.py`, `test_rubric.py` **(new)**; `test_persona.py` extended | see Verification |
| `.env` / `.gitignore` | `AIVA_MOOD_HALFLIFE_H=6`, `AIVA_SELF_TICK_S=60`, `AIVA_EVAL_MODEL`, initiative knobs; ignore `aiva/inner_state.json` |
| `CLAUDE.md` / README | architecture bullets for `inner.py` + the bus; the "nervous system, not pairwise hooks" principle; eval usage |

## Phases (each shippable; 1, 2, 4, 5 are offline-testable before wiring)

1. **Substrate skeleton, offline.** `InnerState`/`InnerBrain` (pure, decay, damping), `ExperienceBus` +
   `experience` table, `inner_state.json` persistence. No LLM. Dashboard shows mood dials + event log.
   Tests: decay half-life, clamping, salience gate, integrate-batch.
2. **Self loop + inner life.** `SelfEngine`/`self_tick`, `JOURNAL_PROMPT` (side-model), `journal` +
   `desires` tables, persona blocks + live refresh, `note_desire`/`drop_desire`/`recall_journal`. Wire
   into `main.py`, absorbing `curiosity_watch`. FakeLLM tests (journal parsed, desires deduped, mood
   persisted). Ship when she keeps a diary and her prompt reflects her mood.
3. **Emitters + consumers (the seamless part).** Retrofit reading/tasks/cat/conversation/idle/memory to
   `emit`; wire consumers (voice tone via prompt, avatar expression, mood-biased `suggest_next`,
   cat-seeking). Test one event end-to-end (chapter → mood → expression). Ship when §2.3's story works.
4. **Initiative.** `initiative_decision` (pure, truth-table tested), express/act/ask notes, `set_initiative`
   permission, backoffs/caps/quiet-hours reused; fold reading-curiosity in. Ship when she acts/asks
   unprompted, appropriately, and backs off cleanly.
5. **Eval foundation.** `rubric.py` (versioned, **your weights**), `signals.py` (Layer 1),
   `eval_checkins` + weekly check-in, `scorecard` table, dashboard Evaluation section. Offline-testable.
6. **Eval judge + scenarios.** `judge.py` (different model, evidence-required), scenario battery on the
   e2e harness, consistency probes, `run.py`, weekly cron. Ship when `python -m eval` prints a scorecard
   and the weekly job appends a trend.

## Verification

- **Offline** (`pytest tests -q`): inner — decay to baseline over the half-life with a fake clock,
  clamp/salience damping, `integrate` batch nudges, `baseline()` from PERSONALITY; self engine —
  journal JSON parsed + fallback, desire dedupe/veto/dropped-blocks-readoption, mood persisted +
  material-change flag, `curiosity` folded in still fires; initiative — `initiative_decision` truth
  table (busy/quiet-hours/backoff/mood branches), notes chosen by mood; persona — blocks only when
  non-empty, each with its anti-announce line, mood block renders from dials; eval — `signals` compute
  hit-rate/self-grade-honesty/drift on fixtures, `rubric` rejects count-based dimensions + version
  stamp, judge model≠voice asserted, judge returns null without evidence.
- **Semi-live** (never starts Aiva): `python -m eval --on-demand` against a throwaway db; a scenario or
  two through `tests/e2e_*`; a journal entry from real recent events.
- **Live checklist:** watch the dashboard — mood dials move after a reading session / a pet / a failed
  task; her avatar softens when low; a journal entry appears at session end; a desire she formed comes
  up naturally when you're around and quiet, and backs off on "not now"; `set_initiative never` silences
  it; the weekly scorecard trends over a couple of weeks and the human check-in anchors it.

## Risks & mitigations

- **State thrash / incoherence** → decay-to-baseline + salience gate + clamped nudges (the cat model);
  material-change flag so we don't refresh/express on noise.
- **Annoyance cliff** → every curiosity guardrail (presence, backoff, daily cap, quiet hours, boot
  grace, never-while-busy) extends to *all* initiative; `set_initiative` global off switch; voice LLM
  may always be more reserved than the decision says.
- **Cost / TPM** → bus + state + decisions are free; only the journal/desire digest costs — one cheap
  side-model call, batched, mostly while asleep; conversation sentiment folds into the existing
  shutdown pass or a heuristic, never a per-turn voice-model call.
- **Personality drift** → the frozen `PERSONALITY` stays the floor; mood moves *around* it; becoming's
  compatibility gate still governs lasting change.
- **Deception / uncanny** → "more human" = richer inner life + continuity, **not** fabricated human
  biography; she stays honestly an AI who has a self. Persona rule to that effect.
- **Loss of legibility** (a shared bus hides causality) → the dashboard is the instrument: mood dials +
  event stream + journal make every action explainable. Build the panels alongside the engine, not after.
- **Reflex contamination** → lip-sync, self-hearing gate, VAD, wake verification NEVER route through the
  bus. Nervous system = slow loop only. (Review gate.)
- **Goodhart** → §3.5, encoded in the rubric and runner, not left to discipline.

## The one thing only you can decide

**The rubric and its weights (§3.0).** That's you drawing the finish line — everything in Part 3 (the
judge prompt, the scenario battery, the weekly scorecard) hangs off it. Good first move tomorrow:
fill in the seven weights (and cut/add a dimension if it's not *your* definition of "more human"),
then build Phase 1 while that settles.
