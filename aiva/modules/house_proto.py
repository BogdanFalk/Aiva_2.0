"""The wire protocol between the house (one ear, the floor) and each sister's process.

Newline-delimited JSON over loopback TCP: one object per line, `{"t": <type>, "id": <msg id>,
"ts": <epoch seconds>, ...fields}`. Only `hello → welcome|reject` is request/response; everything
else is fire-and-forget, correlated by the house-issued monotonic turn id carried in `turn`.

Both ends import this, so it stays stdlib-only (no pipecat, no sqlite) and imports instantly. The
message types are plain dataclasses; `encode`/`decode` are the only functions that touch JSON.
`decode` validates: unknown type or a line over `MAX_LINE` bytes raises `ProtoError`; unknown
fields are ignored (forward compatibility), missing required fields are an error.

The house also owns `aiva/house_state.json` (`{port, pid, boot_id, protocol}`), written atomically
so a sister can never read a half-written file — same convention as `avatar_ipc`.
"""

import json
import os
from dataclasses import dataclass, fields, asdict
from typing import ClassVar, Optional

PROTOCOL = 1
MAX_LINE = 256 * 1024
VEC_DECIMALS = 5

STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "house_state.json")


class ProtoError(ValueError):
    """A line that is not a valid house message."""


# --- message dataclasses -----------------------------------------------------------------
# kw_only so subclasses can add required fields after the envelope defaults.

@dataclass(kw_only=True)
class Msg:
    T: ClassVar[str] = ""
    id: int = 0
    ts: float = 0.0


# character → house

@dataclass(kw_only=True)
class Hello(Msg):
    T: ClassVar[str] = "hello"
    slug: str
    name: str
    name_variants: list = None
    loose_regex: str = ""
    blurb: str = ""
    capabilities: list = None
    pid: int = 0
    version: str = ""
    protocol: int = PROTOCOL
    has_voice: bool = True
    role: str = "sister"            # sister | control (the dashboard's buttons; never a voice)


@dataclass(kw_only=True)
class Response(Msg):
    T: ClassVar[str] = "response"
    turn: int
    phase: str                      # start | end


@dataclass(kw_only=True)
class Tool(Msg):
    T: ClassVar[str] = "tool"
    turn: int
    delta: int                      # +1 when a tool call starts, -1 when it completes


@dataclass(kw_only=True)
class Speaking(Msg):
    T: ClassVar[str] = "speaking"
    turn: int
    on: bool


@dataclass(kw_only=True)
class Said(Msg):
    T: ClassVar[str] = "said"
    turn: int
    text: str
    addressee: str = "him"
    private: bool = False
    interrupted: bool = False


@dataclass(kw_only=True)
class Event(Msg):
    """A public act. Character → house without `actor`; the house relays it to the other sisters
    with `actor` (who did it) and `event_id` (its `house_events` row, for their kin cursors)."""
    T: ClassVar[str] = "event"
    kind: str
    summary: str
    capability: Optional[str] = None
    payload: Optional[dict] = None
    actor: Optional[str] = None
    event_id: Optional[int] = None


@dataclass(kw_only=True)
class WantFloor(Msg):
    T: ClassVar[str] = "want_floor"
    kind: str                       # announce | initiative
    to: Optional[str] = None        # him | <sister slug>


@dataclass(kw_only=True)
class Mode(Msg):                    # both directions
    T: ClassVar[str] = "mode"
    sister_talk: str                # private | audible


@dataclass(kw_only=True)
class Sleep(Msg):
    T: ClassVar[str] = "sleep"


@dataclass(kw_only=True)
class Wake(Msg):
    T: ClassVar[str] = "wake"


@dataclass(kw_only=True)
class Restart(Msg):
    """A sister asks the house to restart itself (exit 42 → start_house.bat relaunches it). The
    sisters stay up and reconnect; the house announces the restart with `shutdown(house_restart)`."""
    T: ClassVar[str] = "restart"
    target: str = "house"           # house (only)


@dataclass(kw_only=True)
class Control(Msg):
    """A button press from a `role=control` client (the dashboard): the same actions as the house
    hotkeys. action: mute | interrupt | sister_talk (target audible|private|None=toggle) | sleep |
    wake | restart_sisters | restart_sister (target slug) | quit_sister (target slug) |
    start_sister (target slug) | restart_house | quit."""
    T: ClassVar[str] = "control"
    action: str
    target: Optional[str] = None


@dataclass(kw_only=True)
class Say(Msg):
    """A typed line from a `role=control` client (the dashboard's chat box). The house treats it
    exactly like a final transcript from the mic: same addressee decision, same floor, spoken
    reply — only the voiceprint and the vocal affect are missing, because nobody spoke."""
    T: ClassVar[str] = "say"
    text: str


@dataclass(kw_only=True)
class Pong(Msg):
    T: ClassVar[str] = "pong"


@dataclass(kw_only=True)
class Bye(Msg):
    T: ClassVar[str] = "bye"


# house → character

@dataclass(kw_only=True)
class Welcome(Msg):
    T: ClassVar[str] = "welcome"
    mode: str
    state: dict = None              # {awake, mic_muted}
    present: list = None            # connected slugs
    protocol: int = PROTOCOL


@dataclass(kw_only=True)
class Reject(Msg):
    T: ClassVar[str] = "reject"
    reason: str


@dataclass(kw_only=True)
class TurnMsg(Msg):
    """The only message that runs her LLM; at most one outstanding house-wide."""
    T: ClassVar[str] = "turn"
    turn: int
    kind: str                       # intro | user | both_second | sister | announce | initiative
    audible: bool
    addressee: str = "him"
    text: Optional[str] = None
    cue: Optional[str] = None
    speaker_vec: Optional[list] = None
    affect: Optional[dict] = None
    wake_hint: Optional[str] = None
    reply_to: Optional[int] = None


@dataclass(kw_only=True)
class Overheard(Msg):
    T: ClassVar[str] = "overheard"
    turn: int
    line: str                       # already bracketed, e.g. [He said to Saga: "…"]
    speaker_vec: Optional[list] = None
    affect: Optional[dict] = None


@dataclass(kw_only=True)
class Affect(Msg):
    """A late reading of how he sounded on user turn `turn` (the emotion-hearing side-call
    finished after that turn had already gone out). `affect` is the model's contribution only
    (`emotion_hearing.late_affect`): arousal/valence/salience/label + `emotion`. Never runs her LLM."""
    T: ClassVar[str] = "affect"
    turn: int
    affect: dict


@dataclass(kw_only=True)
class Backchannel(Msg):
    """A soft "mhm" at a pause while he is still talking (`modules/backchannel.py`). Sent to ONE
    sister; NOT a floor grant — it never runs her LLM, adds nothing to her context and gets no
    `speaking`/`said` back. `clip` is a `backchannel.PHRASES` id; `voice` False → nod only (her
    voice is muted in the house)."""
    T: ClassVar[str] = "backchannel"
    clip: str
    voice: bool = True


@dataclass(kw_only=True)
class Interrupt(Msg):
    T: ClassVar[str] = "interrupt"
    turn: int


@dataclass(kw_only=True)
class State(Msg):
    T: ClassVar[str] = "state"
    awake: bool = True
    mic_muted: bool = False
    present: list = None


@dataclass(kw_only=True)
class Absent(Msg):
    T: ClassVar[str] = "absent"
    slug: str
    name: str


@dataclass(kw_only=True)
class Ping(Msg):
    T: ClassVar[str] = "ping"


@dataclass(kw_only=True)
class Shutdown(Msg):
    """reason: 'quit' → the sisters quit too; 'restart' → the sisters restart themselves (exit 42);
    'house_restart' → only the house is restarting, the sisters stay up and reconnect."""
    T: ClassVar[str] = "shutdown"
    reason: Optional[str] = None


CHARACTER_TO_HOUSE = (Hello, Response, Tool, Speaking, Said, Event, WantFloor, Mode, Sleep, Wake, Restart, Control,
                      Say, Pong, Bye)
HOUSE_TO_CHARACTER = (Welcome, Reject, TurnMsg, Overheard, Affect, Backchannel, Interrupt, Mode, State, Absent, Ping, Shutdown)
MESSAGE_TYPES = {cls.T: cls for cls in CHARACTER_TO_HOUSE + HOUSE_TO_CHARACTER}


# --- codec -----------------------------------------------------------------------------------

def _round_vec(v):
    if isinstance(v, (list, tuple)):
        return [round(float(x), VEC_DECIMALS) for x in v]
    return v


def encode(msg: Msg) -> bytes:
    """One line: `{"t", "id", "ts", ...}` as UTF-8 JSON + newline. `speaker_vec` is rounded to 5 dp."""
    if not isinstance(msg, Msg) or not msg.T:
        raise ProtoError(f"not a house message: {msg!r}")
    body = asdict(msg)
    if "speaker_vec" in body:
        body["speaker_vec"] = _round_vec(body["speaker_vec"])
    out = {"t": msg.T, "id": body.pop("id", 0), "ts": body.pop("ts", 0.0)}
    out.update(body)
    return (json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8")


def decode(line) -> Msg:
    """bytes or str → message. Raises ProtoError on oversize, bad JSON, unknown type or missing fields."""
    if isinstance(line, str):
        raw = line.encode("utf-8")
    else:
        raw = bytes(line)
    if len(raw) > MAX_LINE:
        raise ProtoError(f"line too long ({len(raw)} bytes > {MAX_LINE})")
    try:
        data = json.loads(raw.decode("utf-8").strip())
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtoError(f"bad json: {e}") from e
    if not isinstance(data, dict):
        raise ProtoError("message is not an object")
    t = data.get("t")
    cls = MESSAGE_TYPES.get(t) if isinstance(t, str) else None
    if cls is None:
        raise ProtoError(f"unknown message type {t!r}")
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in known}
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ProtoError(f"{t}: {e}") from e


# --- state file (house is the sole writer) ------------------------------------------------------

def read_state(path=None):
    """The house's state file as a dict, or None if missing/corrupt."""
    try:
        with open(path or STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_state_atomic(state: dict, path=None):
    """Atomic write so a reader can never see a half-written file."""
    dest = path or STATE_FILE
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, dest)


def clear_state(path=None):
    try:
        os.remove(path or STATE_FILE)
    except OSError:
        pass
