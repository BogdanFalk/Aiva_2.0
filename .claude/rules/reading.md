---
paths:
  - "aiva/modules/library.py"
  - "aiva/modules/reading.py"
  - "aiva/modules/becoming.py"
  - "tests/test_library.py"
  - "tests/test_reading.py"
  - "tests/test_reading_becoming.py"
  - "tests/test_becoming.py"
  - "tests/e2e_reading.py"
---

# Reading books and being changed by them

- `aiva/modules/library.py` — pure, sqlite-free, thread-safe book readers: EPUB (stdlib zip+XML+html.parser), TXT/MD (decode chain + Project Gutenberg strip), PDF (pypdf; scan detection, running-header removal, de-hyphenation, outline→chapters). `parse_file` → ordered `Section`s; `segment`/`normalize_sections`/`chunk`. Runs under `asyncio.to_thread`; `clean_text` normalises whitespace ONLY so copied quotes verify verbatim.
- `aiva/modules/reading.py` — `BookMemory` (7 tables on `Memory.conn`) + `ReadingEngine`. One background reading *session* at a time (`llm.create_task`): per section a close-reading note (summary/key ideas/exact ≤25-word quotes VERIFIED against the text/nuances/reflection/running summary+state), `commit_section` moves the cursor atomically with the note (kill mid-section → redo that section, never a dup), session wrap → "what struck me", whole-book synthesis on finish (own model), embedding `recall`. Announces via injected callbacks (quiet while asleep). Hooks `on_section/on_session_end/on_book_finished/on_book_added` feed becoming.
- `aiva/modules/becoming.py` — how reading changes her: `reading_lessons`, `traits`, `voice_notes`, `becoming_state` on `Memory.conn`. On book finish, one JSON call extracts candidates; `validate_candidates` (pure) enforces first-person, evidence-in-notes, bounded content, and baseline-contradiction rejection; `adopt` caps trait strength 0.7 / 12 active, dedupes+reinforces, `fade`s unused ones; tell-once rides the finish note; `drop_trait` vetoes permanently; `reading_permission` + the pure `curiosity_decision` govern when she asks to read. `prompt_blocks` renders the four persona blocks. `python -m modules.becoming --show` audits her.
