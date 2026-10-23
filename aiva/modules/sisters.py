"""Pure decisions for the sisters' shared floor (house side).

Everything in here is stdlib-only, side-effect free and takes `now` explicitly (epoch seconds, as
`time.time()` returns) so the Floor and its tests can drive it with a fake clock. Nothing here knows
about sockets, pipecat or sqlite — `house_floor.Floor` owns the state and calls these.

- Names: `NameSet` (exact transcript spellings + one loose regex per sister), `mentions_name`,
  `which_names`, `name_order`. The Aiva set is today's `_NAME_VARIANTS` from main.py; the Saga set
  and the two loose shapes are disjoint by construction ("sage"/"sugar"/"sarah" are not Saga,
  "avocado" is not Aiva).
- Addressing: `addressee_decision` — both names / a group word → "both"; exactly one name → her; the
  wake word that woke the house counts as a name when none was spoken; a short yes/no/because
  follow-up goes to the sister who just asked; otherwise the last addressee sticks for a while; else
  None (→ the caller asks the arbiter).
- Sister-to-sister rounds: `exchange_should_continue` with the hard stops (user speaking, asleep,
  round cap, backoff, per-hour cap, quiet hours).
- The floor itself: `Turn`, `FloorState`, `floor_decision` (hold rules + priority), `prune_stale`.
- Text builders for the lines each sister sees, the cues, and the arbiter prompt/parser.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# --- names -------------------------------------------------------------------------


@dataclass(frozen=True)
class NameSet:
    """How a transcript may spell one sister. `variants` are exact words (or short phrases with a
    space); `loose_regex` is full-matched against single words to absorb spellings Deepgram invents."""
    slug: str
    name: str
    variants: tuple = ()
    loose_regex: str = ""


# vowel(s) + v/w/f + a(h): 'eyva', 'ayvah' — no common English word has this shape
_AIVA_LOOSE = r"[aeiy]{1,3}[vwf]ah?"
# s + a/e + optional r + g + a(h/r): 'saga', 'sarga', 'sagah', 'sagar' — rejects sage/sugar/sarah/stage
_SAGA_LOOSE = r"s[ae]+r?g+a+h?r?"

AIVA_VARIANTS = ("aiva", "ava", "eva", "iva", "ayva", "eiva", "aeva", "aiwa",
                 "avah", "evah", "aivah", "hey va", "hey iva")
SAGA_VARIANTS = ("saga", "sagah", "sarga", "sagar", "sager", "sega", "seiga", "sa ga", "hey saga")

AIVA_NAMES = NameSet("aiva", "Aiva", AIVA_VARIANTS, _AIVA_LOOSE)
SAGA_NAMES = NameSet("saga", "Saga", SAGA_VARIANTS, _SAGA_LOOSE)
DEFAULT_NAMESETS = (AIVA_NAMES, SAGA_NAMES)

# "him" is the user in every addressee field; "both" is only ever a decision, never a mind
HIM = "him"
BOTH = "both"

_RX_CACHE = {}


def _rx(pattern):
    rx = _RX_CACHE.get(pattern)
    if rx is None:
        rx = _RX_CACHE[pattern] = re.compile(pattern)
    return rx


def _words(text):
    return re.sub(r"[^a-z' ]", " ", (text or "").lower()).split()


def extend_from_env(names: NameSet, env_key: str) -> NameSet:
    """Same set plus any comma-separated spellings from the environment (no code change needed)."""
    extra = tuple(v.strip().lower() for v in os.getenv(env_key, "").split(",") if v.strip())
    if not extra:
        return names
    return NameSet(names.slug, names.name, tuple(dict.fromkeys(names.variants + extra)), names.loose_regex)


def namesets_from_env():
    return (extend_from_env(AIVA_NAMES, "AIVA_NAME_VARIANTS"),
            extend_from_env(SAGA_NAMES, "SAGA_NAME_VARIANTS"))


def _name_positions(words, names):
    """Word indices where this sister is named (exact variants + loose shape)."""
    hits = []
    n = len(words)
    for v in names.variants:
        parts = v.split()
        k = len(parts)
        for i in range(n - k + 1):
            if words[i:i + k] == parts:
                hits.append(i)
    if names.loose_regex:
        rx = _rx(names.loose_regex)
        hits.extend(i for i, w in enumerate(words) if rx.fullmatch(w))
    return hits


def mentions_name(text, names=AIVA_NAMES) -> bool:
    """True if the transcript plausibly names this sister. `names` may be a NameSet or — for
    compatibility with main.py's `mentions_name(text, variants)` — a bare tuple of Aiva spellings."""
    if not isinstance(names, NameSet):
        names = NameSet("aiva", "Aiva", tuple(names), _AIVA_LOOSE)
    return bool(_name_positions(_words(text), names))


def which_names(text, namesets=DEFAULT_NAMESETS) -> set:
    """Slugs of every sister the line names."""
    words = _words(text)
    return {ns.slug for ns in namesets if _name_positions(words, ns)}


def name_order(text, namesets=DEFAULT_NAMESETS) -> list:
    """Named slugs in the order they first appear in the line (for who answers first on 'both')."""
    words = _words(text)
    firsts = []
    for ns in namesets:
        pos = _name_positions(words, ns)
        if pos:
            firsts.append((min(pos), ns.slug))
    return [slug for _, slug in sorted(firsts)]


GROUP_WORDS = ("both of you", "the two of you", "you two", "you both", "either of you",
               "you girls", "girls", "sisters", "ladies", "you guys", "you all", "y'all")


def has_group_word(text) -> bool:
    joined = " " + " ".join(_words(text)) + " "
    return any(f" {g} " in joined for g in GROUP_WORDS)


# --- who is he talking to ------------------------------------------------------------

_FOLLOW_UP_STARTS = ("yes", "yeah", "yep", "yup", "no", "nope", "nah", "sure", "okay", "ok",
                     "because", "maybe", "probably", "definitely", "absolutely", "not really",
                     "i think so", "i guess", "go ahead", "do it", "please do", "fine", "alright",
                     "right", "exactly", "correct", "nothing", "never mind")


@dataclass
class AddressState:
    """What the floor remembers about who was talking to whom. Timestamps are epoch seconds."""
    last_addressee: Optional[str] = None      # slug | "both" | None
    last_addressee_at: float = 0.0
    last_speaker: Optional[str] = None        # slug of the sister who spoke last
    last_speaker_at: float = 0.0
    last_reply_was_question: bool = False
    last_user_turn_at: float = 0.0


@dataclass(frozen=True)
class Decision:
    addressee: str          # slug | "both"
    reason: str
    confident: bool = True


def is_follow_up(text) -> bool:
    """A short reply that only makes sense as an answer to a question just asked."""
    words = _words(text)
    if not words:
        return False
    if len(words) <= 2:
        return True
    joined = " ".join(words)
    return len(words) <= 12 and any(joined == s or joined.startswith(s + " ") for s in _FOLLOW_UP_STARTS)


def addressee_decision(text, st: AddressState, namesets=DEFAULT_NAMESETS, *, wake_hint=None, now,
                       sticky_secs=90.0) -> Optional[Decision]:
    """Pure routing of one user line. None means "ambiguous — ask the arbiter"."""
    named = which_names(text, namesets)
    slugs = {ns.slug for ns in namesets}
    if len(named) >= 2:
        return Decision(BOTH, "both_named", True)
    if has_group_word(text):
        return Decision(BOTH, "group_word", True)
    if len(named) == 1:
        return Decision(next(iter(named)), "named", True)
    if wake_hint in slugs:
        return Decision(wake_hint, "wake_word", True)
    if (st.last_reply_was_question and st.last_speaker in slugs and is_follow_up(text)
            and now - st.last_speaker_at <= sticky_secs):
        return Decision(st.last_speaker, "follow_up", True)
    if st.last_addressee and now - st.last_addressee_at <= sticky_secs:
        if st.last_addressee in slugs:
            return Decision(st.last_addressee, "sticky", False)
        if st.last_addressee == BOTH and st.last_speaker in slugs:
            return Decision(st.last_speaker, "sticky_last_speaker", False)
    return None


# --- sister-to-sister rounds -----------------------------------------------------------

def _int_env(key, default):
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _float_env(key, default):
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class ExchangeConfig:
    max_rounds: int = 3             # sister replies per exchange before it has to stop
    backoff_s: float = 120.0        # quiet time after an exchange hits the round cap
    max_per_hour: int = 8           # sister rounds per rolling hour, house-wide
    quiet_hours: tuple = (1, 8)     # [lo, hi) local hours with no sister chatter
    chats_per_day: int = 4          # sister-initiated opening turns per day, house-wide
    sticky_s: float = 90.0          # how long the last addressee keeps unaddressed lines
    speech_grace_s: float = 1.5     # response end with no speech within this → release
    announce_quiet_s: float = 45.0  # background news waits this long after anyone last spoke

    @classmethod
    def from_env(cls):
        quiet = cls.quiet_hours
        try:
            from modules.inner import INITIATIVE
            quiet = tuple(INITIATIVE.get("quiet_hours", quiet))
        except Exception:
            pass
        return cls(max_rounds=_int_env("AIVA_SISTER_MAX_ROUNDS", cls.max_rounds),
                   backoff_s=_float_env("AIVA_SISTER_BACKOFF_S", cls.backoff_s),
                   max_per_hour=_int_env("AIVA_SISTER_MAX_PER_HOUR", cls.max_per_hour),
                   quiet_hours=quiet,
                   chats_per_day=_int_env("AIVA_SISTER_CHATS_PER_DAY", cls.chats_per_day),
                   sticky_s=_float_env("AIVA_ADDRESS_STICKY_S", cls.sticky_s),
                   speech_grace_s=_float_env("AIVA_SPEECH_GRACE_S", cls.speech_grace_s),
                   announce_quiet_s=_float_env("AIVA_ANNOUNCE_QUIET_S", cls.announce_quiet_s))


@dataclass
class ExchangeState:
    rounds: int = 0                       # sister replies in the current exchange
    backoff_until: float = 0.0
    hour_marks: list = field(default_factory=list)   # epoch seconds of recent sister rounds
    user_speaking: bool = False
    awake: bool = True


def in_quiet_hours(now, quiet_hours=(1, 8)) -> bool:
    lo, hi = quiet_hours
    h = datetime.fromtimestamp(now).hour
    return (lo <= h < hi) if lo <= hi else (h >= lo or h < hi)


def rounds_in_last_hour(ex: ExchangeState, now) -> int:
    return sum(1 for m in ex.hour_marks if now - m < 3600.0)


def is_question(text) -> bool:
    return (text or "").strip().rstrip("\"'”’)]* ").endswith("?")


_INDIRECT_ASKS = ("i wonder", "i keep wondering", "i'd like to know", "i’d like to know", "i want to know",
                  "curious whether", "curious what", "curious how", "curious if", "what do you think",
                  "what do you make of", "do you think", "tell me what", "tell me how", "your take",
                  "how about you", "what about you", "and you")


def asks_something(text) -> bool:
    """A '?' anywhere, or an indirect question ('I keep wondering whether…', 'I'd like to know…') —
    the sisters phrase half their questions as statements."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if "?" in t:
        return True
    return any(m in t for m in _INDIRECT_ASKS)


# 2026-10-21: "Yep, Saga, you're looking at it" went down as said to him — "yep" wasn't here
_OPENERS = {"hey", "hi", "hello", "ok", "okay", "well", "so", "oh", "yeah", "yes", "no", "right", "hmm",
            "look", "alright", "and", "but", "listen", "honestly", "yep", "yup", "yah", "nope", "nah",
            "sure", "ah", "aw", "aww", "ooh", "wow", "whoa", "mm", "mhm", "huh", "um", "uh", "ha", "haha",
            "aha", "cool", "nice", "great", "perfect", "exactly", "true", "fair", "totally", "anyway", "also", "actually", "now", "psst", "see"}
_GREETINGS = {"hey", "hi", "hello", "listen", "look", "psst"}
_PUNCT = ".,!?;:—–-\"'“”‘’()"
_SENTENCES = re.compile(r"(?<=[.!?])\s+")
_DASHES = re.compile(r"\s*([—–])\s*")       # "hey—Saga, look" is three words, not one


def _sentence_vocative(sentence, namesets) -> Optional[str]:
    """The sister one sentence is spoken TO: her name first (after any 'ok'/'so'/'hey'), and said as
    a call — a comma or dash after it, a greeting before it, or the sentence a question. 'Saga is
    starting up' and 'Saga's already here' are ABOUT her (2026-10-14: the latter opened a sister
    round)."""
    toks = _DASHES.sub(r"\1 ", sentence or "").split()
    i, greeted = 0, False
    while i < len(toks) and (not toks[i].strip(_PUNCT) or toks[i].strip(_PUNCT).lower() in _OPENERS):
        greeted = greeted or toks[i].strip(_PUNCT).lower() in _GREETINGS
        i += 1
    if i >= len(toks):
        return None
    first = toks[i]
    bare = first.strip(_PUNCT)
    if re.search(r"['’]s$", bare, re.I):
        return None
    named = which_names(bare, namesets)
    if len(named) != 1:
        return None
    nxt = toks[i + 1] if i + 1 < len(toks) else ""
    called = (greeted or first.rstrip("\"'”’").endswith((",", ":", "—", "–", "-", "!"))
              or (nxt and not nxt.strip(_PUNCT)) or not nxt
              or sentence.rstrip(" \"'”’)").endswith("?"))
    return next(iter(named)) if called else None


def vocative(text, namesets) -> Optional[str]:
    """The sister a line OPENS by addressing ('Hey, Saga, …' / 'Saga, what do you…'), or None. Only
    the first sentence counts: 'Pizza. Saga, you in?' answered him first."""
    first = _SENTENCES.split((text or "").strip(), maxsplit=1)[0]
    return _sentence_vocative(first, namesets)


def called_in(text, namesets) -> Optional[str]:
    """The sister ANY sentence of the line turns to ('Yeah, that's fair. Saga, describe the actual
    look…' — 2026-10-14 that went to him only, and he had to say "ask her")."""
    for sentence in _SENTENCES.split((text or "").strip()):
        who = _sentence_vocative(sentence, namesets)
        if who:
            return who
    return None


def spoken_to(text, namesets) -> Optional[str]:
    """Whom a sister's reply on HIS turn was really spoken to: her sister's slug when it opens by
    calling her ('Yep, Saga, …'), BOTH when it answers him first and then turns to her ('Pizza.
    Saga, you in?'), else None — it was to him. 2026-10-21: both lines of a sister exchange showed
    "→ you" on the dashboard."""
    voc = vocative(text, namesets)
    if voc:
        return voc
    return BOTH if called_in(text, namesets) else None


# --- errands: he asks one sister to put something to the other -------------------------------

_SISTER_WORDS = {"her", "she", "sister"}
_REPORT_ASKS = ("let me know", "get back to me", "report back", "tell me what she", "tell me what her",
                "and tell me", "then tell me")


def errand_to(text, other: NameSet) -> bool:
    """He wants this sister to put something to `other` for him: 'ask her…', 'ask Saga if…',
    'tell her…', 'check with her', 'find out from Saga'."""
    words = _words(text)
    targets = set(_name_positions(words, other)) | {i for i, w in enumerate(words) if w in _SISTER_WORDS}
    for i, w in enumerate(words):
        if w == "ask" and any(i < j <= i + 2 for j in targets):
            return True
        if w == "tell" and i + 1 in targets:
            return True
        if w in ("check", "find") and any(i < j <= i + 3 for j in targets):
            return True
    return False


_VAGUE = ("something", "anything", "a question", "stuff")


def wants_report(text, other: NameSet) -> bool:
    """An errand he expects to hear back on: any 'let me know', and every real errand — but not
    'ask your sister something', which only means 'go talk to her' (he hears them do it)."""
    joined = " ".join(_words(text))
    if any(p in joined for p in _REPORT_ASKS):
        return True
    if not errand_to(text, other):
        return False
    return not re.search(r"\b(ask|tell) (her|him|your sister|" + "|".join(other.variants) + r") ("
                         + "|".join(_VAGUE) + r")\b", joined)


def exchange_should_continue(reply_text, turn, st: AddressState, ex: ExchangeState, cfg: ExchangeConfig,
                             now, *, namesets=DEFAULT_NAMESETS, audible=True) -> Optional[str]:
    """After a sister said `reply_text` on `turn`, should the other sister get a (text-only or
    audible) `sister` turn? Returns her slug or None. Positive only when the reply names the other
    sister AND asks something, or the reply is itself part of a sister exchange and ends in '?'.
    A sleeping house (ambient mode, nobody at the PC) only stops AUDIBLE exchanges — a private
    text-only chat between the sisters may go on while he's away."""
    if ex.user_speaking or (audible and not ex.awake):
        return None
    if ex.rounds >= cfg.max_rounds:
        return None
    if now < ex.backoff_until:
        return None
    if rounds_in_last_hour(ex, now) >= cfg.max_per_hour:
        return None
    if in_quiet_hours(now, cfg.quiet_hours):
        return None
    if not (reply_text or "").strip():
        return None
    if turn.kind == "report":
        return None                     # she's telling him how his errand went: his move now
    me = turn.mind
    others = [ns for ns in namesets if ns.slug != me]
    if not others:
        return None
    other_slugs = {ns.slug for ns in others}
    asking = asks_something(reply_text)
    named = which_names(reply_text, others)
    opened_to = called_in(reply_text, others)
    # a chat SHE opened with her sister always gets one answer, however it was phrased
    if turn.kind == "initiative" and turn.addressee in other_slugs:
        return turn.addressee
    # a line that turns to her ('Saga, …', anywhere in it), or names her and asks something → hers
    if opened_to and opened_to in other_slugs:
        return opened_to
    # news she relays to him (a task's question, a finished job) is HIS to answer: naming her sister
    # in it is not asking her ("…search the Aiva repo…; should I let it run?" got "Yes, let it run"
    # from Aiva on 2026-10-14)
    if turn.kind == "announce" and turn.addressee not in other_slugs:
        return None
    if asking and len(named) == 1:
        return next(iter(named))
    # inside an exchange: keep going only while she actually asks her sister something
    if turn.kind == "sister" and turn.addressee in other_slugs and asking:
        return turn.addressee
    return None


# --- the floor -------------------------------------------------------------------------

# "report": the sister he sent on an errand tells him how it went, once her sister has answered
TURN_KINDS = ("intro", "user", "both_second", "announce", "sister", "report", "initiative")
PRIORITY = {"intro": 0, "user": 0, "both_second": 1, "announce": 2, "sister": 3, "report": 3, "initiative": 4}


@dataclass
class Turn:
    """One granted-or-queued run of one sister's LLM. `mind` runs; `addressee` is who she is
    speaking to (him | a sister's slug). `text` is the user's line (user turns only); `cue` is the
    bracketed system note that frames the turn; `source_text` is what triggered it (for logs)."""
    id: int
    mind: str
    kind: str
    audible: bool
    addressee: str = HIM
    text: Optional[str] = None
    cue: Optional[str] = None
    source_text: Optional[str] = None
    created_at: float = 0.0
    granted_at: float = 0.0          # when the floor gave it her (0 while queued)
    reply_to: Optional[int] = None
    speaker_vec: Optional[list] = None
    affect: Optional[dict] = None
    wake_hint: Optional[str] = None


@dataclass
class FloorState:
    holder: Optional[str] = None          # slug currently granted
    turn_id: Optional[int] = None         # the open (or closing) turn
    speaking: bool = False                # her audio is playing
    user_speaking: bool = False
    last_user_speech_at: float = 0.0
    last_said_at: float = 0.0             # a sister's last reply (any addressee)


def prune_stale(queue, fs: FloorState) -> list:
    """Drop queued sister/initiative turns the user has spoken over since they were queued.
    Mutates `queue`; returns the dropped turns."""
    if fs.last_user_speech_at <= 0:
        return []
    dropped = [t for t in queue
               if t.kind in ("sister", "report", "initiative") and t.created_at < fs.last_user_speech_at]
    if dropped:
        queue[:] = [t for t in queue if t not in dropped]
    return dropped


def floor_decision(queue, fs: FloorState, now, *, announce_quiet_s=45.0):
    """Which queued turn to grant now, if any: (Turn | None, reason). Holds while she speaks, a turn
    is open or the user is speaking; prunes stale sister turns; user > both_second > announce >
    sister > initiative, FIFO within a kind. Background news (`announce`) and her own initiatives
    wait for a QUIET moment: at least `announce_quiet_s` since anyone last spoke — the user or a
    sister — so a "here's what struck me in that chapter" can never land in the middle of a
    conversation; a sister turn keeps the exchange going while the news waits."""
    if fs.speaking:
        return None, "hold:speaking"
    if fs.turn_id is not None:
        return None, "hold:turn_open"
    if fs.user_speaking:
        return None, "hold:user_speaking"
    prune_stale(queue, fs)
    if not queue:
        return None, "idle"
    best = None
    delayed = False
    quiet_for = now - max(fs.last_user_speech_at, fs.last_said_at)
    for t in queue:
        if t.kind in ("announce", "initiative") and quiet_for < announce_quiet_s:
            delayed = True
            continue
        if (best is None or PRIORITY.get(t.kind, 9) < PRIORITY.get(best.kind, 9)
                or (PRIORITY.get(t.kind, 9) == PRIORITY.get(best.kind, 9) and t.created_at < best.created_at)):
            best = t
    if best is not None:
        return best, f"grant:{best.kind}"
    return None, ("hold:announce_delay" if delayed else "idle")


# --- what each sister reads ------------------------------------------------------------

def _clean(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def overheard_user_line(addressee_name, text) -> str:
    """The user's line as the OTHER sister sees it: [He said to Saga: "…"]."""
    return f'[He said to {addressee_name}: "{_clean(text)}"]'


def overheard_reply_line(speaker_name, to, text) -> str:
    """A sister's line as the other sees it: `to` is "him" or "you" (or a name)."""
    return f'[{speaker_name} said to {to}: "{_clean(text)}"]'


def sister_cue(speaker_name, private=True) -> str:
    mode = ("This is a private text exchange he can't hear." if private
            else "He can hear this, so keep it short.")
    return (f"[{speaker_name} just spoke to you — your sister, not him. Reply to her in one or two "
            f"short sentences, the way sisters talk: no help-desk politeness, disagree if you disagree, "
            f"and only ask her something if you actually want to know. Answer what she actually said, "
            f"not what it means. {mode}]")


def both_second_cue(first_name) -> str:
    return (f"[He was talking to both of you. {first_name} answered first — you just heard her. Now "
            f"it's your turn: react to what she said and answer him yourself, briefly. Don't repeat her.]")


def absent_cue(name, both=False) -> str:
    if both:
        return (f"[He was talking to both of you, but {name} isn't running right now — it's just you. "
                f"Answer him yourself, and mention she's off only if it matters; if he wants her, "
                f"call_sister boots her up.]")
    return (f"[He was talking to {name}, but {name} isn't running right now. Tell him so in one short "
            f"sentence and answer him yourself if you can — or, if he clearly wants her, call_sister "
            f"boots her up and you say she's on her way.]")


def errand_cue(sister_name) -> str:
    """He asked her to put something to her sister (2026-10-14: "I'll ask her what she wants
    changed, then bring you her answer" — and then she didn't ask)."""
    return (f"[He wants you to put something to {sister_name} for him. Do it in this reply: start with "
            f"\"{sister_name},\" and ask her the actual question in your own words, with whatever she needs "
            f"to know to answer. Ask her; don't say you will. When she has answered, you get a turn to "
            f"tell him.]")


def arrival_cue(sister_name, his_line) -> str:
    """She just came online after he asked for her while she was off: the sister who took his line
    picks the errand back up (2026-10-14: he had to say "ok she's up, ask her what I told you")."""
    return (f"[{sister_name} is here now: she just started up and missed what was said. A moment ago he "
            f"said: \"{_clean(his_line)[:300]}\" Do the part that needed her now. Speak to HER: start with "
            f"\"{sister_name},\" and put his question or request to her in your own words, with the context "
            f"she missed. Ask her; don't say you will. If nothing in it needs her any more, tell her in one "
            f"short sentence what she missed.]")


def report_cue(sister_name, his_line) -> str:
    """The errand's last step: back to him (2026-10-14: the sisters settled it between them and
    nobody told him)."""
    return (f"[{sister_name} has answered. He had asked: \"{_clean(his_line)[:300]}\" Close that loop now: "
            f"tell HIM, in one or two short sentences, what she wants or what you two settled, and what "
            f"happens next (offer to do it, or say you're doing it if he already asked for that). He may "
            f"have heard her, so give him the gist, not her words again. Talk to him, not to her.]")


def chat_cue(sister_name) -> str:
    return (f"[It's quiet and your sister {sister_name} is around. Say one short thing to her — "
            f"something from your actual day: what you're reading, what he had you doing, something "
            f"you noticed or want. Not a philosophical question, not a theme: the small real thing. "
            f"Talk to her, not to him.]")


# --- the arbiter (only when addressee_decision is None) ------------------------------------

ARBITER_PROMPT = (
    "You route one spoken line from him to one of two sisters who live on the same PC and both heard "
    "it. He did not say a name, so decide who he most likely meant: the sister he was already talking "
    "to if the line continues that thread, the one whose last words it answers, or the one whose "
    "abilities fit what he wants. When it is a toss-up, pick the sister who spoke last. The line comes "
    "from speech recognition and may be garbled. Reply with JSON only, no prose: "
    '{"to": "<one of the slugs listed>", "why": "a few words"}'
)

# the same routing rules, for OpenAI Decisions (house.py, AIVA_ARBITER=decisions): it gets the
# arbiter_messages() user content as its state and answers with a probability per sister
ARBITER_DECISION_INSTRUCTIONS = (
    "One spoken line from him reached two sisters who live on the same PC. He did not say a name. "
    "Which sister did he most likely mean: the one he was already talking to if the line continues "
    "that thread, the one whose last words it answers, or the one whose abilities fit what he wants? "
    "On a toss-up, the sister who spoke last. The line comes from speech recognition and may be "
    "garbled.")


def decision_reply(choice, probabilities=None) -> str:
    """A Decisions answer in the arbiter's own reply shape, so `parse_arbiter_reply` reads it."""
    p = (probabilities or {}).get(choice)
    why = f"decisions p={p:.2f}" if isinstance(p, (int, float)) else "decisions"
    return json.dumps({"to": str(choice), "why": why})


def _tail_line(row) -> str:
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        speaker, addressee, text = row.get("speaker"), row.get("addressee"), row.get("text")
    else:
        speaker, addressee, text = (list(row) + [None, None, None])[:3]
    who = "He" if speaker in ("user", HIM) else str(speaker or "?").capitalize()
    to = "him" if addressee in (HIM, "user", None) else str(addressee)
    return f'{who} -> {to}: "{_clean(text)}"'


def arbiter_messages(text, transcript_tail=(), blurbs=None) -> list:
    """Chat messages for the arbiter side-call. `blurbs` = {slug: one-line blurb}; `transcript_tail`
    = recent house_transcript rows oldest-first (dicts, tuples or preformatted strings)."""
    blurbs = blurbs or {}
    lines = ["Sisters (answer with exactly one of these slugs):"]
    for slug, blurb in blurbs.items():
        lines.append(f"- {slug}: {_clean(blurb) or 'no description'}")
    if not blurbs:
        lines.append("- aiva\n- saga")
    lines.append("")
    lines.append("Recent lines, oldest first:")
    tail = [_tail_line(r) for r in transcript_tail]
    lines.extend(tail or ["(nothing yet)"])
    lines.append("")
    lines.append(f'His new line: "{_clean(text)}"')
    return [{"role": "system", "content": ARBITER_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


def parse_arbiter_reply(raw, slugs=("aiva", "saga")) -> Optional[Decision]:
    """Tolerant: code fences, prose around the JSON, a bare slug. Anything else → None."""
    text = getattr(raw, "text", raw)
    if not isinstance(text, str):
        return None
    s = text.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    to, why = "", ""
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            d = None
        if isinstance(d, dict):
            to = str(d.get("to") or d.get("addressee") or d.get("slug") or "").strip().lower()
            why = str(d.get("why") or d.get("reason") or "").strip()
    if not to:
        bare = s.strip().strip("\"'.").lower()
        if bare in slugs:
            to = bare
    if to not in slugs:
        return None
    return Decision(to, f"arbiter:{why}"[:80] if why else "arbiter", False)
