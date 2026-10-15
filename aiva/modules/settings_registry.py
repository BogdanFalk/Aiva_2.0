"""Every tunable knob the household reads, in one authored registry — the dashboard's Settings tab.

Pure and stdlib-only (the dashboard must stay light): a `Setting` per env key the code reads —
its group, type, the default the CODE actually uses, one plain sentence of what it does, who reads
it (`applies_to`) and so which process must restart for a change to land (config is read at boot;
there is no hot reload). Secrets (API keys) are listed masked and never editable. A handful of
important hard-coded constants are listed `readonly` with their source location, for reference.

    current_values(env_path)        -> {key: {value, source ".env"|"default"|"code", default, ...}}
    validate(key, raw)              -> (True, coerced | None for "reset") | (False, error)
    write_env(env_path, updates)    -> edit .env IN PLACE (comments, blank lines, order and unknown
                                       lines kept; `None` removes a key), atomic, keeps .env.bak
    apply_updates(env_path, body)   -> validate everything first, then one write; the restart list
    grouped(env_path)               -> the Settings tab payload

`tests/test_settings_registry.py` greps the source for env keys and fails when one isn't here, so
a new knob can't silently go missing from the page.
"""

import os
import re
import shutil
import tempfile
from dataclasses import dataclass

DASHBOARD_MARKER = "# --- set from the dashboard ---"
TYPES = ("bool", "int", "float", "str", "choice")
MAX_VALUE_CHARS = 400

# who reads a key → the processes that must restart for it to take effect
PROCESS_LABELS = {"aiva": "Aiva", "saga": "Saga", "house": "the house", "dashboard": "the dashboard",
                  "body": "the avatar bodies (a full quit, not a restart — the body survives restarts)"}
_APPLIES_RESTART = {"aiva": ("aiva",), "saga": ("saga",), "sisters": ("aiva", "saga"),
                    "house": ("house",), "all": ("house", "aiva", "saga"), "dashboard": ("dashboard",)}

GROUPS = (
    "API keys & services", "Models", "Voice & audio", "Avatar & body", "Wake & sleep",
    "Floor & sisters", "Tasks", "Reading & books", "Mood & self loop", "Initiative & autonomy",
    "Speaker ID", "Web & space", "Limits & spend caps", "Notifications", "Dashboard", "Saga",
    "Debug & internal",
)


@dataclass(frozen=True)
class Setting:
    key: str
    group: str
    type: str                     # bool | int | float | str | choice
    default: object               # what the code uses when unset (None = unset / computed)
    description: str              # one plain sentence
    applies_to: str = "sisters"   # aiva | saga | sisters | house | all | dashboard
    choices: tuple = None
    secret: bool = False
    restart: tuple = None         # processes to restart; None → derived from applies_to
    min: float = None
    max: float = None
    default_from: str = None      # unset → this other key's value (e.g. task model = voice model)
    default_note: str = None      # human text for a computed default ("aiva/space")
    readonly: bool = False        # shown, never written (code constants, process-internal keys)
    source: str = None            # where a readonly code constant lives

    @property
    def restart_targets(self):
        return tuple(self.restart) if self.restart is not None else _APPLIES_RESTART.get(self.applies_to, ())

    @property
    def editable(self):
        return not (self.secret or self.readonly)


def S(key, group, type_, default, description, applies_to="sisters", **kw):
    return Setting(key, group, type_, default, description, applies_to, **kw)


def C(key, group, value, description, source):
    """A hard-coded constant, listed for reference only."""
    t = "bool" if isinstance(value, bool) else "int" if isinstance(value, int) else \
        "float" if isinstance(value, float) else "str"
    return Setting(key, group, t, value, description, "sisters", readonly=True, source=source)


_BODY = ("body",)

SETTINGS = (
    # --- API keys & services -------------------------------------------------------------------
    S("OPENAI_API_KEY", "API keys & services", "str", None, "OpenAI key for her voice model, side-calls, embeddings and the house arbiter.", "all", secret=True),
    S("ELEVENLABS_API_KEY", "API keys & services", "str", None, "ElevenLabs key for both sisters' voices.", secret=True),
    S("DEEPGRAM_API_KEY", "API keys & services", "str", None, "Deepgram key for the house's speech-to-text.", "house", secret=True),
    S("TAVILY_API_KEY", "API keys & services", "str", None, "Tavily key for web search and page extraction.", secret=True),
    S("SEARCH_API_KEY", "API keys & services", "str", None, "Fallback web search key, used only when TAVILY_API_KEY is empty.", secret=True),
    S("OPENWEATHER_API_KEY", "API keys & services", "str", None, "OpenWeather key for the weather tool.", secret=True),
    S("DEFAULT_CITY", "API keys & services", "str", "London", "City the weather tool uses when you don't name one."),

    # --- Models ----------------------------------------------------------------------------------
    S("AIVA_LLM_MODEL", "Models", "str", "gpt-4.1-mini", "The voice model both sisters think and talk with.", "all"),
    S("AIVA_TASK_MODEL", "Models", "str", None, "Model for background tasks and the house arbiter, on its own token bucket.", "all", default_from="AIVA_LLM_MODEL"),
    S("AIVA_VISION_MODEL", "Models", "str", None, "Model that looks at screenshots and images.", default_from="AIVA_LLM_MODEL"),
    S("AIVA_STUDY_MODEL", "Models", "str", None, "Model that close-reads each section of a book.", default_from="AIVA_TASK_MODEL"),
    S("AIVA_STUDY_SYNTH_MODEL", "Models", "str", "gpt-5.6-terra", "Model that writes the whole-book synthesis when she finishes one."),
    S("AIVA_WEB_MODEL", "Models", "str", None, "Model that summarises web pages she reads.", default_from="AIVA_TASK_MODEL"),
    S("AIVA_LLM_MAX_RETRIES", "Models", "int", 5, "How many times an OpenAI call retries a rate limit or server error with backoff.", "all", min=0, max=20),
    S("AIVA_CONTEXT_BUDGET_CHARS", "Models", "int", 24000, "How many characters of conversation history she keeps in context before trimming the oldest.", min=2000, max=400000),
    S("AIVA_VISION_MAX_EDGE", "Models", "int", 1024, "Longest edge in pixels a screenshot is shrunk to before she looks at it.", min=256, max=4096),

    # --- Voice & audio ---------------------------------------------------------------------------
    S("ELEVENLABS_VOICE_ID", "Voice & audio", "str", None, "Aiva's ElevenLabs voice; without one she runs text-only.", "aiva", default_note="none (text-only)"),
    S("ELEVENLABS_MODEL", "Voice & audio", "str", "eleven_flash_v2_5", "ElevenLabs model both voices use unless Saga has her own."),
    S("AIVA_INPUT_DEVICE_INDEX", "Voice & audio", "int", None, "Microphone by device index; wins over the name when set.", "house", min=0, default_note="use the name"),
    S("AIVA_INPUT_DEVICE_NAME", "Voice & audio", "str", None, "Microphone by name substring, which survives driver re-numbering.", "house", default_note="system default"),
    S("AIVA_OUTPUT_DEVICE_INDEX", "Voice & audio", "int", None, "Aiva's speaker by device index; wins over the name when set.", "aiva", min=0, default_note="use the name"),
    S("AIVA_OUTPUT_DEVICE_NAME", "Voice & audio", "str", None, "Speaker by name substring; Saga uses it too unless she has her own.", default_note="system default"),
    S("AIVA_STT", "Voice & audio", "choice", "flux", "Which Deepgram ear the house uses: flux decides when you've finished talking by itself; nova is the older model with a local voice detector and turn model.", "house", choices=("flux", "nova")),
    S("AIVA_FLUX_EOT_THRESHOLD", "Voice & audio", "float", 0.7, "With flux, how sure it must be that you've finished before your turn ends; lower answers sooner but cuts in more.", "house", min=0.5, max=0.9),
    S("AIVA_FLUX_EOT_TIMEOUT_MS", "Voice & audio", "int", 5000, "With flux, end your turn this many milliseconds after you stop talking, however unsure it is.", "house", min=500, max=10000),
    S("AIVA_FLUX_IDLE_CLOSE_S", "Voice & audio", "float", 3.0, "With flux, close the connection after this many seconds without audio (while asleep) and reopen it on wake; 0 keeps it open.", "house", min=0.0, max=60.0),
    S("AIVA_STT_KEYTERMS","Voice & audio", "str", None, "Extra words Deepgram should listen for, comma separated.", "house", default_note="none"),
    S("AIVA_NAME_VARIANTS", "Voice & audio", "str", None, "Extra ways the transcript may spell Aiva, comma separated.", "all", default_note="none"),
    S("AIVA_VOICE_AFFECT", "Voice & audio", "bool", True, "Let her mood mirror how you sound, from the energy and pitch of your voice.", "house"),
    S("AIVA_AFFECT_EMA", "Voice & audio", "float", 0.15, "How quickly your running voice baseline adapts, per turn.", "house", min=0.0, max=1.0),
    S("AIVA_AFFECT_GAIN", "Voice & audio", "float", 0.6, "How strongly a lively or tired voice moves her mood.", "house", min=0.0, max=3.0),
    S("AIVA_AFFECT_SALIENCE_BASE", "Voice & audio", "float", 0.2, "Smallest weight a vocal-affect reading carries.", "house", min=0.0, max=1.0),
    S("AIVA_AFFECT_SALIENCE_MAX", "Voice & audio", "float", 0.7, "Largest weight a vocal-affect reading carries.", "house", min=0.0, max=1.0),
    S("AIVA_AFFECT_NOTABLE", "Voice & audio", "float", 0.12, "How far off your baseline your voice must be before it counts.", "house", min=0.0, max=1.0),
    S("AIVA_EMOTION_HEARING", "Voice & audio", "bool", True, "Send each of your turns' audio to an OpenAI audio model to hear how you sound (tired, amused, a laugh, a sigh); it never delays her reply.", "house"),
    S("AIVA_EMOTION_MODEL", "Voice & audio", "str", "gpt-audio-mini", "The OpenAI audio model that listens for how you sound (gpt-audio-mini, gpt-audio, gpt-audio-1.5).", "house"),
    S("AIVA_EMOTION_MIN_SECS", "Voice & audio", "float", 0.8, "Turns shorter than this many seconds aren't sent to the emotion model.", "house", min=0.0, max=30.0),
    S("AIVA_EMOTION_MAX_SECS", "Voice & audio", "float", 15.0, "Only the last this-many seconds of a turn are sent to the emotion model.", "house", min=1.0, max=60.0),
    S("AIVA_EMOTION_TIMEOUT_S", "Voice & audio", "float", 10.0, "Give up on an emotion reading after this many seconds.", "house", min=1.0, max=120.0),
    S("AIVA_EMOTION_NOTE_CONFIDENCE", "Voice & audio", "float", 0.7, "How sure the emotion model must be before she gets a note about how you sound.", "sisters", min=0.0, max=1.0),
    S("AIVA_EMOTION_NOTE_TTL_S", "Voice & audio", "float", 120.0, "A reading that arrives after her reply is noted on your next turn only if it's this fresh, in seconds.", "sisters", min=0.0, max=3600.0),

    # --- Avatar & body ---------------------------------------------------------------------------
    S("AIVA_AVATAR_MODEL", "Avatar & body", "str", None, "Aiva's Live2D model file.", "aiva", restart=("aiva",) + _BODY, default_note="assets/avatar/ariu/ariu.model3.json"),
    S("AIVA_AVATAR_FPS", "Avatar & body", "int", 32, "Frames per second the avatar renders at.", restart=_BODY, min=10, max=144),
    S("AIVA_AVATAR_SWAY", "Avatar & body", "bool", True, "Let the avatar gently sway while idle.", restart=_BODY),
    S("AIVA_LIPSYNC_GAIN", "Avatar & body", "float", 2.5, "How wide her mouth opens for a given loudness.", restart=_BODY, min=0.0, max=10.0),
    S("AIVA_LIPSYNC_FORM", "Avatar & body", "float", 1.0, "How much her mouth shape changes while talking.", restart=_BODY, min=0.0, max=3.0),
    S("AIVA_MOUTH_PART_FLOOR", "Avatar & body", "float", None, "Mouth-open level below which the inner mouth part is hidden.", restart=_BODY, min=0.0, max=1.0, default_note="the model's own value"),
    S("AIVA_CAMO_SKIN", "Avatar & body", "str", None, "Skin colour as R,G,B used to hide stray model parts.", restart=_BODY, default_note="the model's own colour"),
    S("AIVA_READING_PROP", "Avatar & body", "str", None, "Image of the book she holds in her hands while reading.", restart=_BODY, default_note="assets/reading_prop.png"),
    S("AIVA_AVATAR_TASK_LABEL", "Avatar & body", "bool", True, "Show the task's goal and current step on the work panel beside her while she works; off shows only the animation (e.g. when streaming).", restart=_BODY),

    # --- Wake & sleep ----------------------------------------------------------------------------
    S("AIVA_WAKE_WORD", "Wake & sleep", "bool", False, "Ambient mode: the house sleeps until it hears a wake word, and Deepgram bills nothing meanwhile.", "all"),
    S("AIVA_WAKE_MODEL", "Wake & sleep", "str", "hey_jarvis", "Aiva's wake-word model file or built-in model name.", "house"),
    S("AIVA_WAKE_THRESHOLD", "Wake & sleep", "float", 0.5, "Wake-word score needed to count as a candidate wake.", "house", min=0.0, max=1.0),
    S("AIVA_WAKE_PATIENCE", "Wake & sleep", "int", 2, "Consecutive frames above threshold needed for a wake.", "house", min=1, max=20),
    S("AIVA_WAKE_VERIFY", "Wake & sleep", "bool", True, "Confirm a wake only when the transcript actually contains a sister's name.", "house"),
    S("AIVA_WAKE_VERIFY_SECS", "Wake & sleep", "float", 5.0, "How long a provisional wake waits for her name before going back to sleep.", "house", min=1.0, max=30.0),
    S("AIVA_IDLE_TIMEOUT", "Wake & sleep", "float", 20.0, "Seconds of silence before the house goes back to sleep in ambient mode.", "house", min=5.0, max=3600.0),

    # --- Floor & sisters -------------------------------------------------------------------------
    S("AIVA_SISTER_TALK", "Floor & sisters", "choice", "private", "Whether the sisters' lines to each other are spoken aloud or text-only.", "all", choices=("private", "audible")),
    S("AIVA_HOTKEY_SISTER_TALK", "Floor & sisters", "str", "ctrl+alt+p", "Global hotkey that flips sister talk between private and audible.", "house"),
    S("AIVA_SISTER_MAX_ROUNDS", "Floor & sisters", "int", 3, "Sister replies in one exchange before it has to stop.", "house", min=0, max=20),
    S("AIVA_SISTER_BACKOFF_S", "Floor & sisters", "float", 120.0, "Quiet seconds after an exchange hits the round cap.", "house", min=0.0, max=86400.0),
    S("AIVA_SISTER_MAX_PER_HOUR", "Floor & sisters", "int", 8, "Sister rounds allowed per rolling hour, house-wide.", "house", min=0, max=200),
    S("AIVA_SISTER_CHATS_PER_DAY", "Floor & sisters", "int", 4, "Sister-initiated chats allowed per day, house-wide.", "house", min=0, max=100),
    S("AIVA_ADDRESS_STICKY_S", "Floor & sisters", "float", 90.0, "How long an unaddressed line still goes to whoever you last spoke to.", "house", min=0.0, max=3600.0),
    S("AIVA_SPEECH_GRACE_S", "Floor & sisters", "float", 1.5, "Seconds the floor waits for speech after a reply ends before releasing it.", "house", min=0.0, max=30.0),
    S("AIVA_ANNOUNCE_QUIET_S", "Floor & sisters", "float", 45.0, "Background news waits this long after anyone last spoke.", "house", min=0.0, max=3600.0),
    S("AIVA_HOUSE_PORT", "Floor & sisters", "int", 8761, "Loopback port the house listens on for the sisters.", "house", min=1024, max=65535),
    S("AIVA_HOUSE_HOST", "Floor & sisters", "str", "127.0.0.1", "Address the house binds to; keep it on loopback.", "house"),
    S("AIVA_HOUSE_DB", "Floor & sisters", "str", None, "Path of the shared house database.", "all", restart=("house", "aiva", "saga", "dashboard"), default_note="aiva/house.db"),

    # --- Tasks (modules/tasks.py Limits) ----------------------------------------------------------
    S("AIVA_TASK_MAX_STEPS", "Tasks", "int", 12, "Most plan steps a background task may have.", min=1, max=100),
    S("AIVA_TASK_MAX_TOOL_CALLS", "Tasks", "int", 80, "Most tool calls one task may make in total.", min=1, max=1000),
    S("AIVA_TASK_MAX_LLM_CALLS", "Tasks", "int", 90, "Most model calls one task may make in total.", min=1, max=1000),
    S("AIVA_TASK_MAX_WALL_MIN", "Tasks", "float", 20.0, "Minutes a task may run before it is stopped.", min=1.0, max=720.0),
    S("AIVA_TASK_STEP_TOOL_CALLS", "Tasks", "int", 12, "Most moves within a single step (one move can read or search several things at once).", min=1, max=100),
    S("AIVA_TASK_TOOL_TIMEOUT_S", "Tasks", "float", 60.0, "Seconds one tool call may take inside a task.", min=5.0, max=3600.0),
    S("AIVA_TASK_CLAUDE_WAIT_MIN", "Tasks", "float", 30.0, "Minutes a task waits for a delegated Claude Code job.", min=1.0, max=720.0),
    S("AIVA_TASK_ANSWER_REMINDER_S", "Tasks", "float", 180.0, "Seconds a waiting task lets pass before reminding you it needs an answer.", min=10.0, max=86400.0),
    S("AIVA_TASK_MAX_COST_USD", "Tasks", "float", 1.50, "Most dollars of model spend one task may use.", min=0.0, max=100.0),
    S("AIVA_TASK_MILESTONE_GAP_S", "Tasks", "float", 30.0, "Fewest seconds between two spoken progress updates.", min=0.0, max=3600.0),
    S("AIVA_TASK_PLAN_WAIT_S", "Tasks", "float", 12.0, "Seconds she waits for a plan before answering you in the same turn.", min=1.0, max=120.0),
    S("AIVA_TASK_MAX_REPAIRS_PER_STEP", "Tasks", "int", 1, "Repair attempts a failed step gets.", min=0, max=10),
    S("AIVA_TASK_MAX_REPAIRS_PER_TASK", "Tasks", "int", 2, "Repair attempts a whole task gets.", min=0, max=20),
    S("AIVA_TASK_MAX_RETHINKS", "Tasks", "int", 1, "Times a task may stop, work out from a coding agent's report what went wrong, and re-plan (or ask you).", min=0, max=5),
    S("AIVA_TASK_FREE_RANGE", "Tasks", "bool", False, "Let coding tasks run without plan and hand-off approvals; shell commands still ask first."),
    S("AIVA_CLAUDE_PERMISSION_MODE", "Tasks", "choice", "auto", "How a Claude Code job that changes code gets its commands approved: auto = Claude Code's safety check approves routine commands with no prompts; acceptEdits = only file edits, every command denied.", choices=("auto", "acceptEdits")),
    S("AIVA_TASK_RUN_TESTS", "Tasks", "bool", True, "After a coding job, run the project's own pytest suite herself (its .venv only) and grade the job on the result."),
    S("AIVA_TASK_TEST_TIMEOUT_S", "Tasks", "float", 300.0, "Seconds she lets a project's test suite run when checking a coding job.", min=10.0, max=3600.0),

    # --- Reading & books -------------------------------------------------------------------------
    S("AIVA_STUDY_WORDS", "Reading & books", "int", 3500, "Target words per reading session.", min=200, max=100000),
    S("AIVA_STUDY_MAX_SECTIONS", "Reading & books", "int", 20, "Most sections read in one session.", min=1, max=500),
    S("AIVA_STUDY_SECTIONS", "Reading & books", "int", 4, "Sections per session when a book has no word counts to pace by.", min=1, max=100),
    S("AIVA_BOOK_MAX_COST_USD", "Reading & books", "float", 3.0, "Most dollars one book may cost to study.", min=0.0, max=100.0),
    S("AIVA_LIBRARY_DIR", "Reading & books", "str", None, "Folder the shared book library lives in.", default_note="aiva/library"),
    C("becoming.MAX_ACTIVE_TRAITS", "Reading & books", 12, "Most reading-shaped traits active at once.", "aiva/modules/becoming.py MAX_ACTIVE_TRAITS"),
    C("becoming.TRAIT_ADOPT_CAP", "Reading & books", 0.7, "Strongest a trait can start at, so one book can't make a core value.", "aiva/modules/becoming.py TRAIT_ADOPT_CAP"),
    C("becoming.TEMPERAMENT_CAP", "Reading & books", 8, "Most temperament traits she can hold.", "aiva/modules/becoming.py TEMPERAMENT_CAP"),
    C("becoming.CURIOSITY.max_asks_per_day", "Reading & books", 2, "Times a day she may ask to read.", "aiva/modules/becoming.py CURIOSITY"),
    C("becoming.CURIOSITY.ask_gap_s", "Reading & books", 4 * 3600, "Seconds between two asks to read.", "aiva/modules/becoming.py CURIOSITY"),
    C("becoming.CURIOSITY.decline_backoff_s", "Reading & books", 6 * 3600, "Seconds she waits after you say no to reading.", "aiva/modules/becoming.py CURIOSITY"),

    # --- Mood & self loop ------------------------------------------------------------------------
    S("AIVA_MOOD_BASE_VALENCE", "Mood & self loop", "float", 0.15, "Aiva's resting mood from gloomy to cheerful.", "aiva", min=-1.0, max=1.0),
    S("AIVA_MOOD_BASE_AROUSAL", "Mood & self loop", "float", 0.50, "Aiva's resting energy.", "aiva", min=0.0, max=1.0),
    S("AIVA_MOOD_BASE_SOCIAL", "Mood & self loop", "float", 0.10, "Aiva's resting wish for company.", "aiva", min=-1.0, max=1.0),
    S("AIVA_MOOD_HALFLIFE_H", "Mood & self loop", "float", 6.0, "Hours for a mood swing of Aiva's to fade halfway back to rest.", "aiva", min=0.1, max=240.0),
    S("AIVA_MOOD_SALIENCE_FLOOR", "Mood & self loop", "float", 0.15, "Experiences weaker than this don't move her mood.", "aiva", min=0.0, max=1.0),
    S("AIVA_MOOD_GAIN", "Mood & self loop", "float", 0.60, "How strongly an experience moves her mood.", "aiva", min=0.0, max=3.0),
    S("AIVA_MOOD_MATERIAL_EPS", "Mood & self loop", "float", 0.05, "Mood change big enough to refresh how her prompt describes her feelings.", "aiva", min=0.0, max=1.0),
    S("AIVA_SELF_TICK_S", "Mood & self loop", "int", 60, "Seconds between turns of her self loop (mood, diary, initiative).", min=5, max=3600),
    S("AIVA_JOURNAL_SALIENCE", "Mood & self loop", "float", 2.0, "How much felt experience piles up before she writes a diary entry mid-session.", min=0.1, max=50.0),
    S("AIVA_JOURNAL_MIN_GAP_H", "Mood & self loop", "float", 12.0, "Fewest hours between two mid-session diary entries.", min=0.0, max=240.0),
    S("AIVA_DRIVES", "Mood & self loop", "bool", True, "Her needs (connection, novelty, competence, rest), learning his rhythm, reunions and smoothing over tense moments."),
    S("AIVA_READ_NOVELTY_BELOW", "Mood & self loop", "float", 0.6, "She only reaches for a book when her need for something new is below this (0 starved, 1 met).", min=0.0, max=1.0),
    S("AIVA_DREAMS", "Mood & self loop", "bool", True, "Her nights: replaying the day into habits and moments that landed, forgetting old noise, and revising her life story."),
    S("AIVA_DREAM_HOUR", "Mood & self loop", "int", 4, "Earliest hour of the night she dreams (once a day, while the house sleeps).", min=0, max=23),
    S("AIVA_STORY_DAYS", "Mood & self loop", "float", 7.0, "About how many days between revisions of her life story.", min=1.0, max=90.0),
    S("AIVA_PROJECTS", "Initiative & autonomy", "bool", True, "Wishes that keep coming back become projects she works on in quiet moments (collections, looking things up)."),
    S("AIVA_INNER_STATE", "Mood & self loop", "str", None, "File Aiva's mood dials are saved to.", "aiva", restart=("aiva", "dashboard"), default_note="aiva/inner_state.json"),
    C("preferences.MAX_ACTIVE", "Mood & self loop", 24, "Most likes, dislikes and opinions active at once.", "aiva/modules/preferences.py MAX_ACTIVE"),
    C("preferences.ADOPT_CAP", "Mood & self loop", 0.6, "Strongest a new preference can start at.", "aiva/modules/preferences.py ADOPT_CAP"),

    # --- Initiative & autonomy -------------------------------------------------------------------
    S("AIVA_SELF_ACT", "Initiative & autonomy", "bool", True, "Let her pursue a safe wish on her own, like finding and keeping an image."),
    C("inner.INITIATIVE.max_per_day", "Initiative & autonomy", 4, "Times a day she may speak up on her own.", "aiva/modules/inner.py INITIATIVE"),
    C("inner.INITIATIVE.max_acts_per_day", "Initiative & autonomy", 3, "Times a day she may act on a wish herself.", "aiva/modules/inner.py INITIATIVE"),
    C("inner.INITIATIVE.min_gap_s", "Initiative & autonomy", 25 * 60, "Fewest seconds between two initiatives.", "aiva/modules/inner.py INITIATIVE"),
    C("inner.INITIATIVE.idle_express_s", "Initiative & autonomy", 8 * 60, "Quiet seconds before she'd volunteer a thought.", "aiva/modules/inner.py INITIATIVE"),
    C("inner.INITIATIVE.idle_ask_s", "Initiative & autonomy", 12 * 60, "Quiet seconds before she'd raise a wish.", "aiva/modules/inner.py INITIATIVE"),
    C("inner.INITIATIVE.max_chats_per_day", "Initiative & autonomy", 3, "Times a day she may start a chat with her sister.", "aiva/modules/inner.py INITIATIVE"),

    # --- Speaker ID ------------------------------------------------------------------------------
    S("AIVA_SPEAKER_ID", "Speaker ID", "bool", True, "Recognise who is talking by voice and ask when it's someone new.", "all"),
    S("AIVA_SPEAKER_MODEL", "Speaker ID", "str", None, "Voiceprint model file.", "house", default_note="aiva/models/campplus_cn_en_common_200k.onnx"),
    S("AIVA_SPEAKER_PRIMARY_NAME", "Speaker ID", "str", "Frank", "Name the first voice Aiva ever hears is enrolled under.", "all"),
    S("AIVA_SPEAKER_SEED_PRIMARY", "Speaker ID", "bool", True, "Silently enrol the first voice Aiva hears instead of asking who it is.", "all"),
    S("AIVA_SPEAKER_T_HIGH", "Speaker ID", "float", 0.62, "Voice match score that counts as a confident match.", "all", min=0.0, max=1.0),
    S("AIVA_SPEAKER_T_LOW", "Speaker ID", "float", 0.45, "Voice match score below which the speaker is unknown.", "all", min=0.0, max=1.0),
    S("AIVA_SPEAKER_MARGIN", "Speaker ID", "float", 0.06, "How far the best match must beat the runner-up.", "all", min=0.0, max=1.0),
    S("AIVA_SPEAKER_STICKY_KEEP", "Speaker ID", "float", 0.50, "Score that keeps the current speaker across a borderline turn.", "all", min=0.0, max=1.0),
    S("AIVA_SPEAKER_STICKY_SECS", "Speaker ID", "float", 45.0, "Seconds the current speaker stays sticky.", "all", min=0.0, max=3600.0),
    S("AIVA_SPEAKER_MIN_SECS", "Speaker ID", "float", 1.5, "Shortest speech in seconds worth a voiceprint.", "all", min=0.2, max=30.0),
    S("AIVA_SPEAKER_MAX_SECS", "Speaker ID", "float", 12.0, "Longest stretch of speech used for a voiceprint.", "all", min=1.0, max=120.0),
    S("AIVA_SPEAKER_EMBED_TIMEOUT_MS", "Speaker ID", "int", 800, "Milliseconds a voiceprint may take before the turn goes on without it.", "all", min=50, max=10000),

    # --- Web & space -----------------------------------------------------------------------------
    S("AIVA_WEB_SEARCH", "Web & space", "bool", True, "Let her search and read the web."),
    S("AIVA_WEB_PROVIDER", "Web & space", "choice", "tavily", "Web search provider.", choices=("tavily",)),
    S("AIVA_WEB_MAX_MB", "Web & space", "float", 25.0, "Largest file in megabytes she may download into her space.", min=0.1, max=500.0),
    S("AIVA_WEB_TIMEOUT", "Web & space", "float", 15.0, "Seconds a web request may take.", min=1.0, max=300.0),
    S("AIVA_WEB_MAX_RESULTS", "Web & space", "int", 6, "Results she asks for per search.", min=1, max=20),
    S("AIVA_SPACE_DIR", "Web & space", "str", None, "Folder Aiva keeps her finds in.", "aiva", default_note="aiva/space"),

    # --- Limits & spend caps ---------------------------------------------------------------------
    S("AIVA_TAVILY_MONTHLY_CAP", "Limits & spend caps", "int", 1000, "Web searches per month before her self-started searching pauses.", restart=("aiva", "saga", "dashboard"), min=0, max=1000000),
    S("AIVA_TTS_MONTHLY_CHARS", "Limits & spend caps", "int", 0, "Your ElevenLabs monthly characters, for alerts only; 0 turns alerts off.", restart=("aiva", "saga", "dashboard"), min=0, max=100000000),
    S("AIVA_STT_MONTHLY_MINUTES", "Limits & spend caps", "int", 0, "Your Deepgram monthly minutes, for alerts only; 0 turns alerts off.", "all", restart=("house", "aiva", "saga", "dashboard"), min=0, max=10000000),
    S("AIVA_EMOTION_MONTHLY_CALLS", "Limits & spend caps", "int", 0, "Monthly emotion-hearing calls before you get an alert (never paused); 0 turns alerts off.", "all", restart=("house", "aiva", "saga", "dashboard"), min=0, max=10000000),

    # --- Notifications ---------------------------------------------------------------------------
    S("AIVA_NTFY", "Notifications", "bool", True, "Send phone alerts through ntfy.", "all"),
    S("NTFY_TOPIC", "Notifications", "str", None, "Your ntfy topic; alerts are off while it's empty.", "all", default_note="none (alerts off)"),
    S("NTFY_URL", "Notifications", "str", "https://ntfy.sh", "Your ntfy server.", "all"),
    S("AIVA_NTFY_ON_START", "Notifications", "bool", True, "Ping the topic when the house or a sister boots.", "all"),

    # --- Dashboard -------------------------------------------------------------------------------
    S("AIVA_DASHBOARD_PORT", "Dashboard", "int", 8760, "Port this dashboard is served on.", "house", restart=("house", "dashboard"), min=1024, max=65535),
    S("AIVA_DASHBOARD_HOST", "Dashboard", "str", "127.0.0.1", "Address this dashboard binds to; keep it on this machine.", "dashboard"),
    S("AIVA_DASHBOARD_POLL_SECS", "Dashboard", "int", 3, "Seconds between page refreshes.", "dashboard", min=1, max=300),
    S("AIVA_DASHBOARD_AUTOSTART", "Dashboard", "bool", True, "The house starts this dashboard when it boots.", "house"),
    S("AIVA_DASHBOARD_OPEN", "Dashboard", "bool", True, "Open a browser tab when the house starts the dashboard.", "house"),
    S("AIVA_DASHBOARD_CONTROL_REMOTE", "Dashboard", "bool", False, "Accept control, chat and settings posts from other machines too.", "dashboard"),
    S("AIVA_DASHBOARD_NO_LAUNCH", "Dashboard", "bool", False, "Stop the Start buttons from launching console windows.", "dashboard"),

    # --- Saga ------------------------------------------------------------------------------------
    S("SAGA_ELEVENLABS_VOICE_ID", "Saga", "str", None, "Saga's ElevenLabs voice; without one she runs text-only.", "saga", default_note="none (text-only)"),
    S("SAGA_ELEVENLABS_MODEL", "Saga", "str", None, "Saga's ElevenLabs model.", "saga", default_from="ELEVENLABS_MODEL"),
    S("SAGA_OUTPUT_DEVICE_INDEX", "Saga", "int", None, "Saga's speaker by device index.", "saga", min=0, default_note="use the name"),
    S("SAGA_OUTPUT_DEVICE_NAME", "Saga", "str", None, "Saga's speaker by name substring.", "saga", default_from="AIVA_OUTPUT_DEVICE_NAME"),
    S("SAGA_DB", "Saga", "str", None, "Saga's own database file.", "saga", restart=("saga", "dashboard"), default_note="aiva/saga.db"),
    S("SAGA_INNER_STATE", "Saga", "str", None, "File Saga's mood dials are saved to.", "saga", restart=("saga", "dashboard"), default_note="aiva/saga_inner_state.json"),
    S("SAGA_SPACE_DIR", "Saga", "str", None, "Folder Saga keeps her finds in.", "saga", default_note="aiva/space_saga"),
    S("SAGA_AVATAR_MODEL", "Saga", "str", None, "Saga's Live2D model file.", "saga", restart=("saga", "body"), default_note="assets/avatar/ariu/saga.model3.json"),
    S("SAGA_AVATAR_MONITOR", "Saga", "int", None, "Monitor Saga's body stands on.", "saga", restart=("saga", "body"), min=0, default_note="Aiva's monitor"),
    S("SAGA_AVATAR_X", "Saga", "float", None, "Where Saga first stands across the monitor, from -1 to 1.", "saga", min=-1.0, max=1.0, default_note="mirror of Aiva"),
    S("SAGA_AVATAR_Y", "Saga", "float", None, "Where Saga first stands up the monitor, from -1 to 1.", "saga", min=-1.0, max=1.0, default_note="same as Aiva"),
    S("SAGA_CONSTELLATION", "Saga", "bool", True, "Saga's quiet constellation witness: a faint sky of tiny stars to her upper left, one lit per moment she lives (never any text).", "saga", restart=("saga", "body")),
    S("SAGA_CONSTELLATION_MAX", "Saga", "int", 24, "How many stars her constellation holds before the oldest dim out.", "saga", restart=("saga", "body"), min=3, max=64),
    S("SAGA_CONSTELLATION_GAP_S", "Saga", "float", 20.0, "Moments closer together than this brighten the latest star instead of adding one.", "saga", restart=("saga", "body"), min=0.0, max=3600.0),
    S("SAGA_CONSTELLATION_ALPHA", "Saga", "float", 0.48, "Peak opacity of her stars while she is awake (it is meant to be quiet; the lines are drawn barely brighter).", "saga", restart=("saga", "body"), min=0.05, max=1.0),
    S("SAGA_CONSTELLATION_SLEEP_ALPHA", "Saga", "float", 0.80, "Peak opacity of her stars while the house sleeps.", "saga", restart=("saga", "body"), min=0.05, max=1.0),
    S("SAGA_CONSTELLATION_SCALE", "Saga", "float", 0.23, "Width of her constellation as a fraction of the monitor height (smaller where that wouldn't fit to her upper left, and then a little taller).", "saga", restart=("saga", "body"), min=0.05, max=0.6),
    S("SAGA_CONSTELLATION_TINT", "Saga", "str", None, "Colour of her stars as r,g,b (0-255 or 0-1).", "saga", restart=("saga", "body"), default_note="white with a faint blue-violet tint (224,214,255)"),
    S("SAGA_NAME_VARIANTS", "Saga", "str", None, "Extra ways the transcript may spell Saga, comma separated.", "all", default_note="none"),
    S("SAGA_STT_KEYTERMS", "Saga", "str", None, "Extra words for Saga's own speech hints, comma separated.", "saga", default_note="none"),
    S("SAGA_WAKE_MODEL", "Saga", "str", None, "Saga's wake-word model file.", "house", default_note="aiva/models/hey_saga.onnx"),
    S("SAGA_SPEAKER_SEED_PRIMARY", "Saga", "bool", False, "Silently enrol the first voice Saga hears instead of asking who it is.", "saga"),
    S("SAGA_SPEAKER_PRIMARY_NAME", "Saga", "str", "Frank", "Name Saga's first voice is enrolled under if she seeds one.", "saga"),
    S("SAGA_MOOD_BASE_VALENCE", "Saga", "float", 0.0, "Saga's resting mood from gloomy to cheerful.", "saga", min=-1.0, max=1.0),
    S("SAGA_MOOD_BASE_AROUSAL", "Saga", "float", 0.5, "Saga's resting energy.", "saga", min=0.0, max=1.0),
    S("SAGA_MOOD_BASE_SOCIAL", "Saga", "float", 0.0, "Saga's resting wish for company.", "saga", min=-1.0, max=1.0),
    S("SAGA_MOOD_HALFLIFE_H", "Saga", "float", 6.0, "Hours for Saga's mood swings to fade halfway back to rest.", "saga", min=0.1, max=240.0),
    S("SAGA_MOOD_SALIENCE_FLOOR", "Saga", "float", 0.15, "Experiences weaker than this don't move Saga's mood.", "saga", min=0.0, max=1.0),
    S("SAGA_MOOD_GAIN", "Saga", "float", 0.60, "How strongly an experience moves Saga's mood.", "saga", min=0.0, max=3.0),
    S("SAGA_MOOD_MATERIAL_EPS", "Saga", "float", 0.05, "Mood change big enough to refresh how Saga's prompt describes her feelings.", "saga", min=0.0, max=1.0),

    # --- Debug & internal ------------------------------------------------------------------------
    S("AIVA_DEBUG_MIC", "Debug & internal", "bool", False, "Print microphone level debugging in the house console.", "house"),
    S("AIVA_CAT_DEBUG", "Debug & internal", "bool", False, "Show the desktop cat's debug overlay.", "aiva"),
    S("AIVA_CHARACTER", "Debug & internal", "str", "aiva", "Which sister a process is; set by the launchers and --character.", readonly=True),
    S("AIVA_RESTART_LOOP", "Debug & internal", "bool", False, "Set by the start scripts so a restart relaunches in the same console.", "all", readonly=True),
    S("AIVA_AVATAR_INSTANCE", "Debug & internal", "str", "aiva", "Which sister an avatar body belongs to; set when a body is spawned.", restart=_BODY, readonly=True),
)

BY_KEY = {s.key: s for s in SETTINGS}
assert len(BY_KEY) == len(SETTINGS), "duplicate setting key"
assert all(s.group in GROUPS for s in SETTINGS), "setting in an unknown group"
assert all(s.type in TYPES for s in SETTINGS), "setting with an unknown type"


def env_keys():
    """Every env-backed key (readonly code constants aren't env keys)."""
    return {s.key for s in SETTINGS if not s.source}


# --- .env parsing ---------------------------------------------------------------------------------

_ASSIGN = re.compile(r"^(\s*)(export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _split_value(rest):
    """`rest` = everything after '=' on a line → (value, quote char or '', trailing comment incl. its
    leading whitespace). Unquoted: a '#' after whitespace starts a comment (python-dotenv's rule)."""
    s = rest.lstrip()
    if s[:1] in ("'", '"'):
        q = s[0]
        i = 1
        while i < len(s):
            if s[i] == "\\" and q == '"' and i + 1 < len(s):
                i += 2
                continue
            if s[i] == q:
                break
            i += 1
        inner = s[1:i]
        if q == '"':
            inner = inner.replace('\\"', '"').replace("\\\\", "\\")
        tail = s[i + 1:]
        m = re.search(r"\s+#.*$", tail)
        return inner, q, (m.group(0) if m else "")
    m = re.search(r"\s+#.*$", s)
    if m:
        return s[:m.start()].rstrip(), "", m.group(0)
    if s.startswith("#"):
        return "", "", s
    return s.rstrip(), "", ""


def parse_env_text(text):
    """{KEY: value} from .env text, last assignment winning (as load_dotenv does)."""
    out = {}
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _ASSIGN.match(line)
        if m:
            out[m.group(3)] = _split_value(m.group(4))[0]
    return out


def _read_text(env_path):
    """(text, had_bom) — missing file → ('', False)."""
    try:
        with open(env_path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return "", False
    bom = raw.startswith(b"\xef\xbb\xbf")
    return raw.decode("utf-8-sig", errors="replace"), bom


def read_env(env_path):
    return parse_env_text(_read_text(env_path)[0])


# --- values -----------------------------------------------------------------------------------------

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _typed(setting, raw):
    """.env string → typed value, or raise ValueError."""
    t = setting.type
    if t == "bool":
        v = str(raw).strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
        raise ValueError(f"{raw!r} is not on/off")
    if t == "int":
        f = float(str(raw).strip())
        if f != int(f):
            raise ValueError(f"{raw!r} is not a whole number")
        return int(f)
    if t == "float":
        return float(str(raw).strip())
    return str(raw)


def mask(value):
    """'sk-abc…wxyz' → '••••wxyz'; short or empty values reveal nothing."""
    v = "" if value is None else str(value)
    if not v:
        return ""
    return "••••" + (v[-4:] if len(v) > 8 else "")


def _effective_default(setting, file_vals, seen=None):
    if setting.default_from:
        seen = seen or set()
        other = BY_KEY.get(setting.default_from)
        if other and other.key not in seen:
            seen.add(setting.key)
            if other.key in file_vals and file_vals[other.key] != "":
                try:
                    return _typed(other, file_vals[other.key])
                except ValueError:
                    return file_vals[other.key]
            return _effective_default(other, file_vals, seen)
    return setting.default


def describe_default(setting):
    if setting.default_from:
        return f"same as {setting.default_from}"
    if setting.default is None:
        return setting.default_note or "unset"
    if setting.type == "bool":
        return "on" if setting.default else "off"
    if isinstance(setting.default, tuple):
        return "–".join(str(x) for x in setting.default)
    return str(setting.default)


def current_values(env_path):
    """{key: {value, source, default, default_text, modified, invalid, raw?}} for every setting.
    `source` is '.env' when the key has a non-empty line there, 'code' for readonly constants,
    else 'default'. Secrets never leave here unmasked."""
    file_vals = read_env(env_path)
    out = {}
    for s in SETTINGS:
        eff_default = _effective_default(s, file_vals)
        rec = {"default": eff_default, "default_text": describe_default(s), "invalid": False}
        if s.source:
            rec.update(value=s.default, source="code", modified=False)
        elif s.key in file_vals and file_vals[s.key] != "":
            raw = file_vals[s.key]
            if s.secret:
                rec.update(value=mask(raw), source=".env", modified=True, is_set=True)
            else:
                try:
                    val = _typed(s, raw)
                except ValueError:
                    val, rec["invalid"] = raw, True
                rec.update(value=val, source=".env", modified=(val != eff_default))
        else:
            rec.update(value=(None if s.secret else eff_default), source="default", modified=False)
            if s.secret:
                rec["is_set"] = False
        out[s.key] = rec
    return out


def validate(key, raw):
    """(True, typed value) | (True, None) meaning "reset to default" | (False, error text)."""
    s = BY_KEY.get(key)
    if s is None:
        return False, f"unknown setting {key!r}"
    if s.secret:
        return False, f"{key} is a secret; edit it in .env by hand"
    if s.readonly:
        return False, f"{key} is read-only here"
    if raw is None or (isinstance(raw, str) and raw.strip() == "" and s.type != "str"):
        return True, None
    if s.type == "bool":
        if isinstance(raw, bool):
            return True, raw
        if isinstance(raw, (int, float)) and raw in (0, 1):
            return True, bool(raw)
    if isinstance(raw, bool) and s.type != "bool":
        return False, f"{key} takes a {s.type}, not on/off"
    if isinstance(raw, (dict, list, tuple)):
        return False, f"{key} takes a single {s.type}"
    text = str(raw)
    if "\n" in text or "\r" in text or "\x00" in text:
        return False, f"{key} must be one line"
    if len(text) > MAX_VALUE_CHARS:
        return False, f"{key} is too long (max {MAX_VALUE_CHARS} characters)"
    if s.type == "str":
        text = text.strip()
        return True, (text or None)
    if s.type == "choice":
        v = text.strip().lower()
        if v not in s.choices:
            return False, f"{key} must be one of {', '.join(s.choices)}"
        return True, v
    try:
        val = _typed(s, text)
    except ValueError as e:
        return False, f"{key}: {e}" if s.type == "bool" else f"{key} needs a {'whole number' if s.type == 'int' else 'number'}"
    if s.type == "float" and val != val:  # NaN
        return False, f"{key} needs a number"
    if s.min is not None and val < s.min:
        return False, f"{key} must be at least {s.min:g}"
    if s.max is not None and val > s.max:
        return False, f"{key} must be at most {s.max:g}"
    return True, val


def format_value(setting, value):
    """Typed value → the .env string."""
    if setting.type == "bool":
        return "1" if value else "0"
    if setting.type == "float":
        return repr(float(value)) if float(value) != int(float(value)) else str(int(float(value)))
    return str(value)


# --- writing .env ------------------------------------------------------------------------------------

def _quote_needed(v):
    return (v != v.strip() or " #" in v or "\t#" in v or v.startswith("#") or v[:1] in ("'", '"')
)


def _render(key, value, quote="", comment="", indent="", export=""):
    if quote or _quote_needed(value):
        q = quote or '"'
        if q == '"':
            value = value.replace("\\", "\\\\").replace('"', '\\"')
        elif "'" in value:
            q = '"'
            value = value.replace("\\", "\\\\").replace('"', '\\"')
        value = f"{q}{value}{q}"
    return f"{indent}{export}{key}={value}{comment}"


def write_env(env_path, updates):
    """Apply `updates` {KEY: str | None} to the .env at `env_path` IN PLACE: an existing `KEY=` line
    is rewritten where it stands (its inline comment kept), `None` deletes the key's lines, a new
    key is appended under DASHBOARD_MARKER. Everything else — comments, blank lines, ordering,
    unknown keys, line endings, a BOM — is left byte-for-byte. Atomic (temp file + os.replace) and
    the previous file is kept as `.env.bak`. Returns {"changed", "added", "removed"} key lists."""
    text, bom = _read_text(env_path)
    # keep every line's OWN ending (a hand-edited .env can mix CRLF and LF); new lines use the
    # file's first ending, or the platform's newline for a new file
    lines = re.findall(r"[^\n]*\n|[^\n]+$", text)
    first = next((ln for ln in lines if ln.endswith(("\n", "\r"))), None)
    newline = (first[len(first.rstrip("\r\n")):] if first else os.linesep) or "\n"
    pending = dict(updates)
    changed, removed = [], []
    out = []          # [(content, ending)]
    for raw in lines:
        line = raw.rstrip("\r\n")
        ending = raw[len(line):]
        m = _ASSIGN.match(line) if line.strip() and not line.lstrip().startswith("#") else None
        if m and m.group(3) in pending:
            key = m.group(3)
            new = pending[key]
            if new is None:
                if key not in removed:
                    removed.append(key)
                continue
            _old, quote, comment = _split_value(m.group(4))
            out.append((_render(key, str(new), quote, comment, m.group(1), m.group(2) or ""), ending))
            if key not in changed:
                changed.append(key)
            continue
        out.append((line, ending))
    added = [k for k, v in pending.items() if v is not None and k not in changed]
    if added:
        if out and not out[-1][1]:
            out[-1] = (out[-1][0], newline)          # the old last line gets its line break
        if DASHBOARD_MARKER not in (c.strip() for c, _e in out):
            if out and out[-1][0].strip():
                out.append(("", newline))
            out.append((DASHBOARD_MARKER, newline))
        for k in added:
            out.append((_render(k, str(pending[k])), newline))
    body = "".join(c + e for c, e in out)
    d = os.path.dirname(os.path.abspath(env_path))
    os.makedirs(d, exist_ok=True)
    if os.path.exists(env_path):
        shutil.copy2(env_path, env_path + ".bak")
    fd, tmp = tempfile.mkstemp(prefix=".env.", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
            f.write(body)
        os.replace(tmp, env_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return {"changed": changed, "added": added, "removed": [k for k in removed if k not in changed]}


def restart_needed(keys):
    """Ordered process slugs that must restart for `keys` to take effect."""
    order = ("house", "aiva", "saga", "body", "dashboard")
    need = set()
    for k in keys:
        s = BY_KEY.get(k)
        if s:
            need.update(s.restart_targets)
    return [p for p in order if p in need]


def restart_text(procs):
    names = [PROCESS_LABELS.get(p, p) for p in procs]
    if not names:
        return "saved"
    if len(names) == 1:
        return f"saved — restart {names[0]} to apply"
    return f"saved — restart {', '.join(names[:-1])} and {names[-1]} to apply"


def apply_updates(env_path, updates):
    """Validate EVERY update first (nothing is written if one is bad), then one write_env.
    Returns (ok, payload): payload has `errors` {key: text} on failure, else `written`, `restart`
    (process slugs), `restart_labels` and a one-line `message`."""
    if not isinstance(updates, dict) or not updates:
        return False, {"errors": {"_": "send {\"updates\": {KEY: value}} with at least one key"}}
    errors, clean = {}, {}
    for key, raw in updates.items():
        ok, val = validate(str(key), raw)
        if not ok:
            errors[str(key)] = val
            continue
        s = BY_KEY[key]
        clean[key] = None if val is None else format_value(s, val)
    if errors:
        return False, {"errors": errors}
    result = write_env(env_path, clean)
    procs = restart_needed(clean)
    return True, {"written": result, "restart": procs,
                  "restart_labels": [PROCESS_LABELS.get(p, p) for p in procs],
                  "message": restart_text(procs)}


def grouped(env_path):
    """The Settings tab payload: [{group, settings: [...]}] in GROUPS order."""
    vals = current_values(env_path)
    by_group = {g: [] for g in GROUPS}
    for s in SETTINGS:
        v = vals[s.key]
        by_group[s.group].append({
            "key": s.key, "type": s.type, "description": s.description, "applies_to": s.applies_to,
            "choices": list(s.choices) if s.choices else None, "secret": s.secret,
            "readonly": s.readonly, "editable": s.editable, "source_file": s.source,
            "min": s.min, "max": s.max, "default_from": s.default_from,
            "restart": list(s.restart_targets),
            "value": list(v["value"]) if isinstance(v["value"], tuple) else v["value"],
            "default": list(v["default"]) if isinstance(v["default"], tuple) else v["default"],
            "default_text": v["default_text"], "source": v["source"], "modified": v["modified"],
            "invalid": v["invalid"], "is_set": v.get("is_set"),
        })
    return [{"group": g, "settings": by_group[g]} for g in GROUPS if by_group[g]]
