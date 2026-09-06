# Changelog

All notable changes to Aiva. Format follows [Keep a Changelog](https://keepachangelog.com);
versioning is [SemVer](https://semver.org). The version lives in `aiva/version.py`.

When you make a change, add a bullet under **[Unreleased]**; when you cut a version, move those
bullets into a new dated section and bump `aiva/version.py`.

## [Unreleased]

### Added
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
