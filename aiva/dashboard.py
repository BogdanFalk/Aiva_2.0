"""The household's dashboard — a live window on each sister (everything she is and everything
she's becoming) plus the house's conversation log, and a Controls strip with the house hotkeys.

Run it any time (nobody needs to be awake):

    cd aiva && ..\\.venv\\Scripts\\python dashboard.py
    # then open http://127.0.0.1:8760   (tabs: Aiva | Saga | Conversations | Tasks | Settings)
    ..\\.venv\\Scripts\\python dashboard.py --once --who saga            # one JSON snapshot
    ..\\.venv\\Scripts\\python dashboard.py --once --who conversations

It opens the databases READ-ONLY (a fresh connection per request, `mode=ro` with a busy
timeout) so it can never lock anyone up or change a thing, reads the state files and today's
event logs, and serves one self-contained page that polls itself every few seconds. Nothing
leaves the machine — host defaults to 127.0.0.1.

The dashboard is read-only EXCEPT the **Controls** strip (under the Household strip on every
tab). Its buttons do exactly what the house hotkeys do — mute/unmute the mic, interrupt, flip
sister talk, sleep/wake, start/restart/quit a sister, restart the house, quit everything — by
sending one `control` message to the house over its local socket (`modules/house_proto`, bound
to 127.0.0.1): `send_control()` connects as a `role="control"` client (`hello` → `welcome`,
`control`, `bye`), never as a sister — no presence, no floor, no voice; the house executes it in
`house._on_control`. `POST /api/control {"action", "target"}` is the route; it is validated here
and only accepted from this machine. With the house down there is nobody to send to, so the
strip greys out and says how to bring it up. The same handshake (`hello` → `welcome` → `bye`)
is how the Household strip learns the live mic/awake state for the button labels.

The **Conversations** tab adds the one other write: a chat box above the transcript that types a
line to them instead of saying it (`send_say` → `say` → `house._on_say` → `Floor.submit_user_turn`).
It is the same path as a final transcript — same addressee rules ("Aiva, …", "both of you, …",
otherwise whoever you were talking to), same floor, and she answers out loud — minus the
voiceprint and the vocal affect, since nobody spoke. Typing wakes the house even while the mic is
deafened. `POST /api/say {"text"}`, same local-only rule.

The **Tasks** tab is read-only: every task each sister ran (her `tasks` table), merged newest
first with filters, and one task in full from `GET /api/task?who=<slug>&id=<task id>` — plan steps
with their evidence, the event timeline, and every Claude Code job it delegated (its saved result
from logs/coding/ or the old %TEMP%\\aiva_terminals\\, and — when the session transcript is found
under ~/.claude/projects/ — what it read, edited and ran).

The **Settings** tab lists every knob from `modules/settings_registry.py` (grouped; value in .env
or the code default; who reads it; secrets masked and never editable; a few code constants shown
read-only) and is the third write: `POST /api/settings {"updates": {KEY: value | null}}` validates
every value through the registry, then edits the repo-root `.env` in place (comments and order
kept, `.env.bak` alongside; `null` resets a key to its default). Same local-only rule, plus the
browser's Origin must be this page. Config is read at boot, so the reply names what to restart.

Where things live: each character's core is her own db (`Character.from_env(slug).db_path`:
memory.db / saga.db) plus her inner/cat/avatar state files; the shared `house.db` holds the
conversation transcript, the house event stream, the sister-talk mode and the spend caps
(service_usage — incl. per-character `tts.<slug>` counters). A character whose db does not
exist yet shows as "not started yet" rather than an error.

Knobs: AIVA_DASHBOARD_PORT (8760), AIVA_DASHBOARD_HOST (127.0.0.1),
AIVA_DASHBOARD_POLL_SECS (page refresh cadence), AIVA_HOUSE_DB (house.db path),
AIVA_DASHBOARD_CONTROL_REMOTE=1 (accept control POSTs from other machines too — off by default,
so exposing the page on the LAN never exposes the buttons).

This is otherwise a pure observer. It imports no pipecat, no live2d, nothing heavy — just the
standard library and the stdlib-only character/house helpers — so it starts instantly and can't
drag anyone down.
"""

import ctypes
import glob
import json
import os
import re
import socket
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

# --- paths (resolve relative to THIS file, so it runs from aiva/ or the repo root) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR = os.path.join(_HERE, "logs")
_MODULES_DIR = os.path.join(_HERE, "modules")

PORT = int(os.getenv("AIVA_DASHBOARD_PORT", "8760"))
HOST = os.getenv("AIVA_DASHBOARD_HOST", "127.0.0.1")
POLL_SECS = int(os.getenv("AIVA_DASHBOARD_POLL_SECS", "3"))

# Reuse the real mood logic (decay-on-read + label) if inner.py is importable — it's
# stdlib-only, so this is cheap and keeps one source of truth. The dashboard still works
# without it (older checkout), falling back to the raw persisted dials.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from modules.inner import InnerBrain, InnerState
    _INNER_OK = True
except Exception:
    _INNER_OK = False
try:
    from version import __version__ as AIVA_VERSION
except Exception:
    AIVA_VERSION = "?"

try:
    from modules import settings_registry
except Exception:  # an older checkout: no Settings tab content, the rest still works
    settings_registry = None

CONVERSATIONS = "conversations"
TASKS_TAB = "tasks"
SETTINGS = "settings"
ENV_PATH = os.path.join(os.path.dirname(_HERE), ".env")   # the repo-root .env every process loads
MAX_SETTINGS_BODY = 65536


# --- the characters: one core each ----------------------------------------------------

def _legacy_characters():
    """Aiva-only map with today's hardcoded paths — used when modules.character is missing
    (an older checkout) so an old single-db install still shows something."""
    return {"aiva": {"slug": "aiva", "name": "Aiva", "self_forming": False,
                     "db": os.path.join(_HERE, "memory.db"),
                     "inner": os.getenv("AIVA_INNER_STATE") or os.path.join(_HERE, "inner_state.json"),
                     "cat": os.path.join(_HERE, "cat_state.json"),
                     "avatar": os.path.join(_HERE, "avatar_state.json")}}


def _build_characters():
    """{slug: {name, db, inner, cat, avatar, self_forming}} from Character.from_env for every
    KNOWN slug. modules.character is stdlib + persona only, so this costs milliseconds."""
    try:
        from modules.character import Character, KNOWN
    except Exception:
        return _legacy_characters()
    chars = {}
    for slug in KNOWN:
        try:
            ch = Character.from_env(slug)
        except Exception:
            continue
        chars[ch.slug] = {"slug": ch.slug, "name": ch.name, "self_forming": bool(ch.self_forming),
                          "db": ch.db_path, "inner": ch.inner_state_path,
                          "cat": ch.cat_state_path, "avatar": ch.avatar_state_path}
    return chars or _legacy_characters()


def _house_db_default():
    try:
        from modules.housedb import house_db_path
        return house_db_path()
    except Exception:
        return os.getenv("AIVA_HOUSE_DB") or os.path.join(_HERE, "house.db")


def _house_state_default():
    try:
        from modules.house_proto import STATE_FILE
        return STATE_FILE
    except Exception:
        return os.path.join(_HERE, "house_state.json")


CHARACTERS = _build_characters()
DEFAULT_WHO = "aiva" if "aiva" in CHARACTERS else next(iter(CHARACTERS))
HOUSE_DB = _house_db_default()          # shared: transcript, events, kv, spend caps
HOUSE_STATE = _house_state_default()    # {port, pid, boot_id, protocol, ts}; the house is its sole writer

# a self-formed temperament trait counts as "core" from this strength; fewer than FORMED core
# traits → she is still forming. Mirrors becoming.TEMPERAMENT_CORE and the persona forming
# note (3 core traits) — duplicated on purpose: importing becoming drags in task_memory →
# utilities → the OpenAI client, and this process must stay stdlib-light.
TEMPERAMENT_CORE = 0.5
TEMPERAMENT_FORMED = 3


def tabs():
    """The header tab strip, in order: one per character, then Conversations, Tasks, Settings."""
    return ([{"who": slug, "label": ch["name"]} for slug, ch in CHARACTERS.items()]
            + [{"who": CONVERSATIONS, "label": "Conversations"}, {"who": TASKS_TAB, "label": "Tasks"},
               {"who": SETTINGS, "label": "Settings"}])


# --- the capability map (her "parts"), grouped for the page -------------------------
TOOL_GROUPS = [
    ("Presence & voice", ["get_datetime", "get_weather", "go_to_sleep", "remember"]),
    ("Body — Live2D avatar", ["avatar_expression", "avatar_move", "avatar_position",
                              "move_avatar_to_monitor", "set_avatar_layer"]),
    ("Desktop cat", ["cat_status", "feed_cat", "pet_cat", "play_with_cat", "summon_cat",
                     "dismiss_cat", "name_cat"]),
    ("Apps & files", ["launch_app", "open_file", "create_file", "list_directory"]),
    ("Terminals", ["open_terminal", "close_terminal", "list_terminals", "show_terminal",
                   "run_in_terminal", "type_in_terminal", "read_terminal"]),
    ("Eyes & hands", ["look_at_screen", "click_at", "click_ui_element", "press_keys",
                      "type_text", "scroll", "focus_window"]),
    ("Shell (gated)", ["run_command", "confirm_pending_command"]),
    ("Delegation & lifecycle", ["claude_code", "restart_self"]),
    ("Tasks & flows", ["start_task", "answer_task", "task_status", "cancel_task",
                       "run_flow", "task_feedback", "save_flow"]),
    ("Reading", ["study_book", "stop_reading", "reading_status", "recall_reading",
                 "add_to_wishlist"]),
    ("Becoming", ["drop_trait", "reading_permission"]),
    # sister tools may not exist in tools.py yet — the parser only lists what it finds
    ("Sisters & the house", ["set_sister_talk", "recall_about_sister", "forget_about_sister"]),
]

SUBSYSTEMS = [
    ("Voice pipeline & memory", ["main.py", "memory.py", "persona.py", "utilities.py"]),
    ("Body — Live2D avatar", ["avatar.py", "avatar_renderer.py", "avatar_ipc.py"]),
    ("Desktop cat", ["cat.py", "cat_sprites.py"]),
    ("Computer use", ["computer.py", "app_launcher.py", "file_operations.py", "terminals.py"]),
    ("Tool surface", ["tools.py"]),
    ("Tasks & competence", ["tasks.py", "task_memory.py"]),
    ("Reading & becoming", ["library.py", "reading.py", "becoming.py"]),
    ("Sisters & the house", ["character.py", "housedb.py", "house_proto.py", "house_server.py",
                             "house_floor.py", "house_stage.py", "house_memory.py",
                             "house_client.py", "house_link.py", "sisters.py", "wake.py",
                             "kin.py"]),
]
RETIRED_MODULES = {"vtube_studio.py": "retired — she renders her own avatar now"}


# --- small helpers ------------------------------------------------------------------

# The tab icon: the house roof over two dots — the sisters (Aiva's teal, Saga's rose), on the
# dashboard's own dark purple. Inline SVG, so no asset file and it reads fine at 16 px.
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<rect width="64" height="64" rx="14" fill="#161222"/>'
    '<path d="M10 32 L32 12 L54 32" fill="none" stroke="#c9c2e6" stroke-width="6" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="22" cy="44" r="8" fill="#38d2c8"/>'
    '<circle cx="42" cy="44" r="8" fill="#f28ab0"/>'
    '</svg>'
)


def _code_mtime():
    """Modification time of this file — the house uses it to spot a dashboard running stale code."""
    try:
        return round(os.path.getmtime(os.path.abspath(__file__)), 3)
    except OSError:
        return 0.0


def _pid_alive(pid):
    """True if a Windows process with this pid exists. ctypes only; no psutil."""
    if not pid:
        return False
    try:
        PROCESS_QUERY_LIMITED = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED, False, int(pid))
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    except Exception:
        return False


def _load_json(path):
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _ago(ts):
    """Humanize an ISO string or epoch float into '3m ago' / '2h ago' / '5d ago'."""
    if ts is None or ts == "":
        return None
    try:
        if isinstance(ts, (int, float)):
            then = datetime.fromtimestamp(ts)
        else:
            then = datetime.fromisoformat(str(ts))
    except (ValueError, OSError):
        return None
    secs = (datetime.now() - then).total_seconds()
    return _dur(secs) + " ago" if secs >= 0 else "just now"


def _dur(secs):
    secs = max(0, int(secs))
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    if secs < 86400:
        return f"{secs // 3600}h"
    return f"{secs // 86400}d"


def _jloads(s, default):
    try:
        v = json.loads(s) if s else default
        return v if v is not None else default
    except (ValueError, TypeError):
        return default


def _db_uri(path):
    """file:///C:/... form (spaces and odd characters percent-encoded) for a read-only open."""
    try:
        return Path(path).resolve().as_uri()
    except (ValueError, OSError):
        return "file:" + str(path).replace("\\", "/")


def _connect(path):
    """Fresh read-only connection; waits briefly if she's mid-write. Never blocks long."""
    con = sqlite3.connect(f"{_db_uri(path)}?mode=ro", uri=True, timeout=3)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=3000")
    return con


def _connect_optional(path):
    """Read-only connection, or None when the file is missing/unreadable (never raises)."""
    if not path or not os.path.exists(path):
        return None
    try:
        return _connect(path)
    except sqlite3.Error:
        return None


def _q(con, sql, params=()):
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    except sqlite3.Error:
        return []


def _one(con, sql, params=()):
    try:
        r = con.execute(sql, params).fetchone()
        return r[0] if r else None
    except sqlite3.Error:
        return None


def _has_table(con, name):
    if con is None:
        return False
    return bool(_one(con, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)))


def _has_column(con, table, column):
    try:
        return any(r["name"] == column for r in _q(con, f"PRAGMA table_info({table})"))
    except sqlite3.Error:
        return False


_WHO_LABELS = {"user": "You", "him": "you", "both": "both"}


def _who_label(x):
    """Display name for a transcript speaker/addressee: a character's name, You/you, both."""
    if x in CHARACTERS:
        return CHARACTERS[x]["name"]
    return _WHO_LABELS.get(x, x or "?")


# --- the actual data collection -----------------------------------------------------

def collect(who=None, **opts):
    """One snapshot for a tab: a character slug ('aiva'|'saga') or 'conversations'.
    Unknown slug → KeyError (the route turns that into a 404)."""
    who = (who or DEFAULT_WHO).strip().lower()
    if who == CONVERSATIONS:
        return collect_conversations(**opts)
    if who == TASKS_TAB:
        return collect_tasks(**opts)
    if who == SETTINGS:
        return collect_settings(**opts)
    ch = CHARACTERS.get(who)
    if ch is None:
        raise KeyError(who)
    return collect_character(ch)


def _stamp(who, name):
    return {"generated_ts": datetime.now().isoformat(), "version": AIVA_VERSION,
            "who": who, "name": name, "tabs": tabs()}


def collect_character(ch):
    slug, name, db = ch["slug"], ch["name"], ch["db"]
    data = _stamp(slug, name)
    data.update({"self_forming": bool(ch.get("self_forming")), "db_path": db,
                 "db_present": os.path.exists(db), "started": os.path.exists(db),
                 "has_cat": bool(ch.get("cat"))})
    house = _connect_optional(HOUSE_DB)
    try:
        data["household"] = _collect_household(house)
        data["presence"] = _collect_presence(ch)
        data["cat"] = _collect_cat(ch.get("cat")) if ch.get("cat") else None
        data["capabilities"] = _collect_capabilities()
        data["inner"] = _collect_inner(None, ch.get("inner"))  # mood-only until a db read replaces it
        if not data["db_present"]:
            data["headline"] = {}
            data["note"] = (f"{name} hasn't started yet — no {os.path.basename(db)} on disk. "
                            f"Her core forms the first time she boots.")
            data["limits"] = _collect_limits(None, house)  # caps + shared usage still mean something
            return data

        try:
            con = _connect(db)
        except sqlite3.Error as e:
            data["error"] = f"database busy or unreadable: {e}"
            data["headline"] = {}
            return data
        try:
            data["reading"] = _collect_reading(con)
            data["inner"] = _collect_inner(con, ch.get("inner"))
            data["preferences"] = _collect_preferences(con)
            data["space"] = _collect_space(con)
            data["limits"] = _collect_limits(con, house)
            data["stats"] = _collect_stats(con)
            data["becoming"] = _collect_becoming(con)
            data["temperament"] = _collect_temperament(con)
            data["kin"] = _collect_kin(con)
            data["tasks"] = _collect_tasks(con)
            data["memory"] = _collect_memory(con)
            data["headline"] = _collect_headline(con, data)
        finally:
            con.close()
    finally:
        if house is not None:
            house.close()
    return data


def collect_conversations(day=None, before=None, limit=200):
    """The house transcript newest-first (optional `day`='YYYY-MM-DD' filter, `before`=<id>
    paging) plus the last 30 house events. Degrades to 'not available' without a house.db."""
    try:
        limit = max(1, min(int(limit or 200), 500))
    except (TypeError, ValueError):
        limit = 200
    day = day if (day and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(day))) else None
    try:
        before = int(before) if before not in (None, "") else None
    except (TypeError, ValueError):
        before = None

    data = _stamp(CONVERSATIONS, "Conversations")
    data["house_db_present"] = os.path.exists(HOUSE_DB)
    conv = {"available": False, "lines": [], "day": day, "before": before, "limit": limit,
            "has_more": False, "next_before": None, "total": 0, "days": []}
    events = []
    house = _connect_optional(HOUSE_DB)
    try:
        data["household"] = _collect_household(house)
        if _has_table(house, "house_transcript"):
            conv["available"] = True
            where, args = [], []
            if before is not None:
                where.append("id < ?")
                args.append(before)
            if day:
                where.append("ts LIKE ?")
                args.append(f"{day}%")
            sql = ("SELECT id, ts, turn_id, speaker, addressee, text, audible, kind, interrupted "
                   "FROM house_transcript")
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY id DESC LIMIT ?"
            rows = _q(house, sql, (*args, limit + 1))
            conv["has_more"] = len(rows) > limit
            rows = rows[:limit]
            conv["lines"] = [_transcript_line(r) for r in rows]
            if rows and conv["has_more"]:
                conv["next_before"] = rows[-1]["id"]
            if day:
                conv["total"] = _one(house, "SELECT COUNT(*) FROM house_transcript WHERE ts LIKE ?",
                                     (f"{day}%",)) or 0
            else:
                conv["total"] = _one(house, "SELECT COUNT(*) FROM house_transcript") or 0
            conv["days"] = [r["d"] for r in _q(house, "SELECT DISTINCT substr(ts, 1, 10) AS d "
                                                     "FROM house_transcript ORDER BY d DESC LIMIT 30")]
        if _has_table(house, "house_events"):
            events = _q(house, "SELECT id, ts, slug, kind, summary, capability, relayed_to "
                               "FROM house_events ORDER BY id DESC LIMIT 30")
            for e in events:
                e["name"] = _who_label(e.get("slug"))
                e["time"] = (e.get("ts") or "")[11:19]
                e["when"] = _ago(e.get("ts"))
    finally:
        if house is not None:
            house.close()
    data["conversations"] = conv
    data["house_events"] = events
    return data


def collect_settings(env_path=None):
    """The Settings tab: every knob from `modules.settings_registry`, grouped, with its value in
    .env (or the code default), description, who reads it and what must restart. Secrets arrive
    masked. Reads .env only; the one write is `POST /api/settings`."""
    env_path = env_path or ENV_PATH
    data = _stamp(SETTINGS, "Settings")
    house = _connect_optional(HOUSE_DB)
    try:
        data["household"] = _collect_household(house)
    finally:
        if house is not None:
            house.close()
    data["env_path"] = env_path
    data["env_present"] = os.path.isfile(env_path)
    if settings_registry is None:
        data["error"] = "modules/settings_registry.py is missing — no settings to show"
        data["settings"] = []
        return data
    groups = settings_registry.grouped(env_path)
    rows = [s for g in groups for s in g["settings"]]
    data["settings"] = groups
    data["processes"] = settings_registry.PROCESS_LABELS
    data["settings_summary"] = {
        "total": len(rows), "editable": sum(1 for s in rows if s["editable"]),
        "in_env": sum(1 for s in rows if s["source"] == ".env"),
        "modified": sum(1 for s in rows if s["modified"] and not s["secret"]),
        "secrets_set": sum(1 for s in rows if s["secret"] and s["is_set"]),
        "secrets": sum(1 for s in rows if s["secret"]),
        "invalid": [s["key"] for s in rows if s["invalid"]]}
    return data


# --- the Tasks tab: every task / coding session each sister ran, start to finish ----------
# The db is the source of truth (each sister's `tasks` table: plan_json + log_json). A Claude Code
# job's saved result is found by its NAME only (`<task-id>-s<n>`, validated) in logs/coding/ or the
# old %TEMP%\aiva_terminals\; its session transcript by a validated UUID under ~/.claude/projects/.
# No path is ever built from anything that didn't pass one of those regexes.

TASK_ID_RE = re.compile(r"^t-\d{8}-\d{6}-[0-9a-f]{4}$")
JOB_RE = re.compile(r"^t-\d{8}-\d{6}-[0-9a-f]{4}-s\d+$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_CURSOR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+\|t-\d{8}-\d{6}-[0-9a-f]{4}$")
LEGACY_JOBS_DIR = os.path.join(tempfile.gettempdir(), "aiva_terminals")   # before logs/coding/ existed
CLAUDE_PROJECTS_DIR = os.path.join(os.path.expanduser("~"), ".claude", "projects")
TASK_BUCKETS = ("success", "failed", "cancelled", "interrupted", "active")
MAX_RESULT_CHARS = 6000
MAX_ARG_CHARS = 400
MAX_TRANSCRIPT_BYTES = 64 * 1024 * 1024
MAX_TASK_ROWS = 2000            # per sister, per list request — far more than she has ever run
_TASK_WANTED_COLS = ("id", "created_ts", "updated_ts", "finished_ts", "goal_text", "details", "family",
                     "approach", "status", "outcome", "plan_json", "log_json", "pending_json",
                     "result_summary", "duration_s", "delegated_to", "agent_grade", "agent_grade_reason",
                     "user_feedback", "corrected", "flow_id", "llm_calls", "tool_calls", "cost_usd")
_READ_TOOLS = {"Read", "NotebookRead"}
_EDIT_TOOLS = {"Edit", "MultiEdit", "NotebookEdit"}
_WRITE_TOOLS = {"Write"}
_SHELL_TOOLS = {"Bash", "PowerShell"}
_SEARCH_TOOLS = {"Grep", "Glob"}
_DELEGATE_JOB_RE = re.compile(r"^(t-\d{8}-\d{6}-[0-9a-f]{4}-s\d+)\b")


def _clip(s, n):
    """A string cut to n chars with a marker saying how much was left out."""
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + f"… (+{len(s) - n} chars)"


def _clip_obj(v, n=MAX_ARG_CHARS):
    """Tool args for display: long strings clipped, nested structures as clipped JSON."""
    if isinstance(v, str):
        return _clip(v, n)
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    try:
        return _clip(json.dumps(v, ensure_ascii=False), n)
    except (TypeError, ValueError):
        return _clip(repr(v), n)


def _dur_long(secs):
    """'45s' / '3m 12s' / '1h 2m' / '2d 3h' — a task's length, finer than _dur."""
    if secs is None:
        return None
    try:
        secs = max(0, int(float(secs)))
    except (TypeError, ValueError):
        return None
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60:02d}s"
    if secs < 86400:
        return f"{secs // 3600}h {(secs % 3600) // 60:02d}m"
    return f"{secs // 86400}d {(secs % 86400) // 3600}h"


def _parse_ts(ts):
    """ISO (naive local, or 'Z'/offset → converted to local naive) → datetime, or None."""
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is not None:
        d = d.astimezone().replace(tzinfo=None)
    return d


def _took(started, finished):
    a, b = _parse_ts(started), _parse_ts(finished)
    return _dur_long((b - a).total_seconds()) if a and b else None


def task_bucket(status, outcome):
    """One of TASK_BUCKETS for the outcome pill and the filter. The recorded outcome wins; a task
    the process died under (status interrupted, no outcome) is 'interrupted'; the rest is live."""
    if outcome in ("success", "failed", "cancelled"):
        return outcome
    return {"done": "success", "failed": "failed", "cancelled": "cancelled",
            "interrupted": "interrupted"}.get(status or "", "active")


def _action_view(kind, action):
    """What a step does, for display: the shell command, tool + clipped args, the delegate prompt."""
    a = action if isinstance(action, dict) else {}
    if kind == "shell":
        return {"command": _clip(a.get("command"), 2000), "purpose": _clip(a.get("purpose"), 400)}
    if kind == "delegate":
        return {"project_dir": _clip(a.get("project_dir"), 300), "prompt": _clip(a.get("prompt"), 4000)}
    if kind == "ask":
        return {"question": _clip(a.get("question"), 1000)}
    if kind == "tool":
        args = a.get("args") if isinstance(a.get("args"), dict) else {}
        return {"tool": _clip(a.get("tool"), 80), "args": {str(k): _clip_obj(v) for k, v in args.items()}}
    return {str(k): _clip_obj(v) for k, v in a.items()}


def task_steps(plan):
    """plan_json (list of step dicts) → display rows. Pure; tolerant of missing/odd fields."""
    out = []
    for i, s in enumerate(plan if isinstance(plan, list) else []):
        if not isinstance(s, dict):
            continue
        kind = str(s.get("kind") or "tool")
        ev = s.get("evidence") if isinstance(s.get("evidence"), list) else []
        try:
            attempts = int(s.get("attempts") or 0)
        except (TypeError, ValueError):
            attempts = 0
        out.append({"n": s.get("n", i + 1), "title": _clip(s.get("title") or f"step {i + 1}", 200),
                    "kind": kind, "status": str(s.get("status") or "pending"), "attempts": attempts,
                    "risk": s.get("risk"), "expect": _clip(s.get("expect"), 600),
                    "verify": s.get("verify"), "action": _action_view(kind, s.get("action")),
                    "evidence": [_clip(x, 2000) for x in ev],
                    "failure": _clip(s.get("failure"), 1000) if s.get("failure") else None,
                    "started": s.get("started"), "finished": s.get("finished"),
                    "took": _took(s.get("started"), s.get("finished"))})
    return out


def parse_claude_event(detail):
    """The `claude` log event's detail (JSON {job, session, cost, status, log}) → dict with a
    validated job/session, or None. Tolerant: the runner clips details at 300 chars, so a cut-off
    JSON still yields whatever fields a regex can recover."""
    d = None
    try:
        d = json.loads(detail) if isinstance(detail, str) else detail
    except ValueError:
        d = None
    if not isinstance(d, dict):
        s = str(detail or "")
        d = {}
        for key in ("job", "session", "status", "log"):
            m = re.search(r'"%s"\s*:\s*"([^"]*)"' % key, s)
            if m:
                d[key] = m.group(1)
        m = re.search(r'"cost"\s*:\s*([0-9.eE+-]+)', s)
        if m:
            d["cost"] = m.group(1)
    job = str(d.get("job") or "")
    if not JOB_RE.match(job):
        return None
    session = str(d.get("session") or "").lower()
    try:
        cost = float(d["cost"]) if d.get("cost") not in (None, "") else None
    except (TypeError, ValueError):
        cost = None
    return {"job": job, "session": session if UUID_RE.match(session) else None, "cost": cost,
            "status": _clip(d.get("status"), 60) if d.get("status") else None,
            "log": _clip(d.get("log"), 300) if d.get("log") else None}


def task_timeline(log):
    """log_json → display events. Announcements are marked `secondary` (the UI folds them away);
    `claude` events carry their parsed record; each gets a tone for its dot. Pure."""
    out = []
    for e in log if isinstance(log, list) else []:
        if not isinstance(e, dict):
            continue
        ev = str(e.get("event") or "?")
        detail = "" if e.get("detail") is None else str(e.get("detail"))
        ts = str(e.get("ts") or "")
        tone = "info"
        if ev in ("error", "verify_failed", "plan_invalid"):
            tone = "bad"
        elif ev == "finish":
            tone = "good" if detail.startswith("success") else "bad" if detail.startswith("failed") else "warn"
        elif ev == "grade":
            tone = "good" if detail.startswith("good") else "bad" if detail.startswith("bad") else "warn"
        elif ev in ("decision", "answer", "plan", "delegate", "claude"):
            tone = "accent"
        elif ev == "pushed_back":
            tone = "warn"
        item = {"ts": ts, "day": ts[:10], "time": ts[11:19], "event": ev, "detail": detail,
                "secondary": ev.startswith("announce"), "tone": tone}
        if ev == "claude":
            item["claude"] = parse_claude_event(detail)
        out.append(item)
    return out


def claude_jobs(task_id, plan, log):
    """Every Claude Code job a task ran: from `delegate` events (one per run start) and `claude`
    events (one per run end, newer tasks), else inferred from delegate steps + the job naming
    `<task-id>-s<n>`. → [{job, n, prompt, project_dir, runs: [{started, ended, session, cost,
    status}]}] in step order. Pure; only JOB_RE-valid names survive."""
    steps = {}
    for s in plan if isinstance(plan, list) else []:
        if isinstance(s, dict) and s.get("kind") == "delegate":
            steps[f"{task_id}-s{s.get('n')}"] = s
    jobs = {}

    def job_entry(name):
        if name not in jobs:
            st = steps.get(name) or {}
            a = st.get("action") if isinstance(st.get("action"), dict) else {}
            try:
                n = int(name.rsplit("-s", 1)[1])
            except (IndexError, ValueError):
                n = None
            jobs[name] = {"job": name, "n": n, "title": st.get("title"), "status": st.get("status"),
                          "prompt": _clip(a.get("prompt"), 4000) if a.get("prompt") else None,
                          "project_dir": a.get("project_dir"), "runs": []}
        return jobs[name]

    for e in log if isinstance(log, list) else []:
        if not isinstance(e, dict):
            continue
        ev, detail, ts = e.get("event"), str(e.get("detail") or ""), e.get("ts")
        if ev == "delegate":
            m = _DELEGATE_JOB_RE.match(detail)
            if m and m.group(1).startswith(task_id + "-s"):
                job_entry(m.group(1))["runs"].append({"started": ts, "ended": None, "session": None,
                                                      "cost": None, "status": None})
        elif ev == "claude":
            c = parse_claude_event(detail)
            if not c or not c["job"].startswith(task_id + "-s"):
                continue
            j = job_entry(c["job"])
            run = next((r for r in reversed(j["runs"]) if r["ended"] is None), None)
            if run is None:
                run = {"started": None}
                j["runs"].append(run)
            run.update({"ended": ts, "session": c["session"], "cost": c["cost"], "status": c["status"]})
    for name in steps:              # a delegate step whose run left no event (older / never started)
        if JOB_RE.match(name):
            job_entry(name)
    return sorted(jobs.values(), key=lambda j: (j["n"] is None, j["n"] or 0))


def summarize_claude_transcript(lines, cmd_cap=60, search_cap=30):
    """A Claude Code session transcript (JSONL lines) → a compact activity summary: turns, tool
    counts, files read / edited / written (with counts, relative to the session's cwd), shell
    commands and searches (capped), lines added/removed, cost, duration and the final answer
    (the `result` event, else the last assistant text). Pure; bad lines are skipped."""
    msg_ids, tools = set(), {}
    read, edited, written = {}, {}, {}
    commands, searches = [], []
    n_commands = n_searches = errors = 0
    final = result_ev = cost_state = None
    cwd = model = prompt = first_ts = last_ts = None
    for raw in lines:
        raw = raw.strip() if isinstance(raw, str) else ""
        if not raw:
            continue
        try:
            e = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(e, dict):
            continue
        if e.get("timestamp"):
            first_ts = first_ts or e["timestamp"]
            last_ts = e["timestamp"]
        cwd = cwd or e.get("cwd")
        t = e.get("type")
        if t == "queue-operation" and e.get("operation") == "enqueue" and prompt is None:
            prompt = e.get("content")
        elif t == "user":
            content = (e.get("message") or {}).get("content")
            if prompt is None and isinstance(content, str):
                prompt = content
            if isinstance(content, list):
                errors += sum(1 for b in content if isinstance(b, dict)
                              and b.get("type") == "tool_result" and b.get("is_error"))
        elif t == "assistant" and not e.get("isSidechain"):
            m = e.get("message") or {}
            msg_ids.add(m.get("id") or e.get("uuid"))
            model = model or m.get("model")
            for b in m.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and str(b.get("text") or "").strip():
                    final = b["text"]
                elif b.get("type") == "tool_use":
                    name = str(b.get("name") or "?")
                    inp = b.get("input") if isinstance(b.get("input"), dict) else {}
                    tools[name] = tools.get(name, 0) + 1
                    path = inp.get("file_path") or inp.get("notebook_path")
                    if name in _READ_TOOLS and path:
                        read[path] = read.get(path, 0) + 1
                    elif name in _EDIT_TOOLS and path:
                        edited[path] = edited.get(path, 0) + 1
                    elif name in _WRITE_TOOLS and path:
                        written[path] = written.get(path, 0) + 1
                    elif name in _SHELL_TOOLS and inp.get("command"):
                        n_commands += 1
                        if len(commands) < cmd_cap:
                            commands.append(_clip(inp["command"], 300))
                    elif name in _SEARCH_TOOLS and inp.get("pattern"):
                        n_searches += 1
                        if len(searches) < search_cap:
                            where = inp.get("path") or inp.get("glob")
                            searches.append(_clip(f"{name} {inp['pattern']}" + (f"  in {where}" if where else ""), 200))
        elif t == "result":
            result_ev = e
        elif t == "cost-state":
            cost_state = e

    def rel(p):
        p = str(p)
        if cwd:
            base = str(cwd).rstrip("\\/")
            if p.lower().startswith(base.lower()) and p[len(base):len(base) + 1] in ("\\", "/"):
                p = p[len(base) + 1:]
        return p.replace("\\", "/")

    def ranked(d):
        return [{"path": rel(k), "count": v} for k, v in sorted(d.items(), key=lambda kv: (-kv[1], rel(kv[0])))]

    r = result_ev or {}
    cs = cost_state or {}
    text = r.get("result") if isinstance(r.get("result"), str) and r.get("result").strip() else final
    return {"turns": r.get("num_turns") or len(msg_ids), "model": model, "cwd": cwd,
            "prompt": _clip(prompt, 600) if prompt else None,
            "first_ts": first_ts, "last_ts": last_ts,
            "tools": dict(sorted(tools.items(), key=lambda kv: -kv[1])),
            "files_read": ranked(read), "files_edited": ranked(edited), "files_written": ranked(written),
            "commands": commands, "commands_total": n_commands,
            "searches": searches, "searches_total": n_searches, "tool_errors": errors,
            "cost_usd": r.get("total_cost_usd") if r.get("total_cost_usd") is not None else cs.get("totalCostUSD"),
            "duration_ms": r.get("duration_ms") or cs.get("totalDuration"),
            "lines_added": cs.get("totalLinesAdded"), "lines_removed": cs.get("totalLinesRemoved"),
            "final_text": _clip(text, MAX_RESULT_CHARS) if text else None}


def _job_dirs():
    return [os.path.join(_LOG_DIR, "coding"), LEGACY_JOBS_DIR]


def _read_text(path, limit):
    try:
        with open(path, "rb") as f:
            raw = f.read(limit + 1)
    except OSError:
        return None
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw[:limit].decode("utf-16", errors="replace")
    return raw[:limit].decode("utf-8-sig", errors="replace")


def job_result(job):
    """A Claude Code job's saved output, by NAME: logs/coding/<job>.log (+ <job>.json, the raw
    `claude -p` payload) or, for old jobs, %TEMP%\\aiva_terminals\\<job>.log. None if not found."""
    if not isinstance(job, str) or not JOB_RE.match(job):
        return None
    for d in _job_dirs():
        log_path = os.path.join(d, job + ".log")
        if not os.path.isfile(log_path):
            continue
        text = _read_text(log_path, 256 * 1024) or ""
        out = {"path": log_path, "legacy": d == LEGACY_JOBS_DIR, "chars": len(text),
               "text": _clip(text.strip(), MAX_RESULT_CHARS), "payload": None}
        pj = _load_json(os.path.join(d, job + ".json"))
        if isinstance(pj, dict):
            sid = str(pj.get("session_id") or "").lower()
            out["payload"] = {"session_id": sid if UUID_RE.match(sid) else None,
                              "total_cost_usd": pj.get("total_cost_usd"), "num_turns": pj.get("num_turns"),
                              "duration_ms": pj.get("duration_ms"), "is_error": pj.get("is_error"),
                              "subtype": _clip(pj.get("subtype"), 40) if pj.get("subtype") else None}
        return out
    return None


def find_transcript(session):
    """~/.claude/projects/*/<session>.jsonl for a UUID session id, else None."""
    session = str(session or "").lower()
    if not UUID_RE.match(session):
        return None
    hits = glob.glob(os.path.join(glob.escape(CLAUDE_PROJECTS_DIR), "*", session + ".jsonl"))
    hits = [h for h in hits if os.path.isfile(h)]
    return max(hits, key=os.path.getmtime) if hits else None


def read_transcript(path):
    """Summary of one transcript file (size-capped), or None."""
    try:
        if os.path.getsize(path) > MAX_TRANSCRIPT_BYTES:
            return {"too_big": True}
        with open(path, encoding="utf-8", errors="replace") as f:
            return summarize_claude_transcript(f)
    except OSError:
        return None


def _norm_words(s):
    return " ".join(str(s or "").split())


def infer_sessions(project_dir, prompt, since, until, max_files=400):
    """For a job logged before `claude` events existed: the sessions in that project's Claude Code
    folder whose first prompt matches the delegate prompt, written inside the task's time window.
    → [(started datetime, session uuid)]. The folder name is Claude Code's own mangling of the
    project path (every non-alphanumeric → '-'), so it can't escape CLAUDE_PROJECTS_DIR."""
    if not project_dir or not prompt:
        return []
    folder = os.path.join(CLAUDE_PROJECTS_DIR, re.sub(r"[^A-Za-z0-9]", "-", str(project_dir)))
    if not os.path.isdir(folder):
        return []
    want = _norm_words(prompt)[:120]
    lo = (since.timestamp() - 300) if since else 0
    hi = (until.timestamp() + 3600) if until else time.time() + 3600
    found = []
    try:
        names = [n for n in os.listdir(folder) if n.endswith(".jsonl") and UUID_RE.match(n[:-6])]
    except OSError:
        return []
    cands = []
    for n in names:
        p = os.path.join(folder, n)
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        if lo <= mt <= hi:
            cands.append((mt, p, n[:-6]))
    for _mt, p, sid in sorted(cands, reverse=True)[:max_files]:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                head = f.read(32768)
        except OSError:
            continue
        first_prompt = started = None
        for line in head.splitlines()[:40]:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not isinstance(e, dict):
                continue
            started = started or _parse_ts(e.get("timestamp"))
            if e.get("type") == "queue-operation" and e.get("operation") == "enqueue":
                first_prompt = e.get("content")
            elif e.get("type") == "user" and isinstance((e.get("message") or {}).get("content"), str):
                first_prompt = e["message"]["content"]
            if first_prompt:
                break
        if first_prompt and _norm_words(first_prompt)[:120] == want:
            found.append((started, sid))
    return sorted(found, key=lambda x: x[0] or datetime.min)


def _attach_inferred(job, created, until):
    """Give runs without a session the inferred ones, nearest start first (within 10 min)."""
    missing = [r for r in job["runs"] if not r.get("session")]
    if not missing:
        return
    taken = {r.get("session") for r in job["runs"] if r.get("session")}
    pool = [(t, s) for t, s in infer_sessions(job.get("project_dir"), job.get("prompt"), created, until)
            if s not in taken]
    if not job["runs"] and pool:           # a delegate step with no run events at all
        job["runs"] = [{"started": None, "ended": None, "session": None, "cost": None, "status": None}]
        missing = job["runs"]
    for run in missing:
        st = _parse_ts(run.get("started"))
        best = None
        for t, s in pool:
            gap = abs((t - st).total_seconds()) if (t and st) else None
            if st is None or (gap is not None and gap <= 600):
                if best is None or (gap is not None and gap < best[0]):
                    best = (gap if gap is not None else 0, t, s)
        if best:
            run["session"], run["inferred"] = best[2], True
            pool = [(t, s) for t, s in pool if s != best[2]]


def _task_select(con):
    have = {r["name"] for r in _q(con, "PRAGMA table_info(tasks)")}
    return ", ".join(c for c in _TASK_WANTED_COLS if c in have) or "id"


def _task_list_row(slug, r):
    plan = _jloads(r.get("plan_json"), [])
    plan = plan if isinstance(plan, list) else []
    log = _jloads(r.get("log_json"), [])
    log = log if isinstance(log, list) else []
    claude_cost = 0.0
    for e in log:
        if isinstance(e, dict) and e.get("event") == "claude":
            c = parse_claude_event(e.get("detail"))
            if c and c["cost"]:
                claude_cost += c["cost"]
    st = [s for s in plan if isinstance(s, dict)]
    summary = " ".join(str(r.get("result_summary") or "").split())
    delegated = r.get("delegated_to") or "none"
    created = r.get("created_ts")
    return {"sister": slug, "sister_name": _who_label(slug), "id": r.get("id"),
            "goal": _clip(r.get("goal_text"), 400), "status": r.get("status"), "outcome": r.get("outcome"),
            "bucket": task_bucket(r.get("status"), r.get("outcome")),
            "created": created, "created_ago": _ago(created), "updated": r.get("updated_ts"),
            "finished": r.get("finished_ts"), "duration_s": r.get("duration_s"),
            "duration": _dur_long(r.get("duration_s")),
            "steps_total": len(st), "steps_done": sum(1 for s in st if s.get("status") == "done"),
            "steps_failed": sum(1 for s in st if s.get("status") == "failed"),
            "delegated_to": delegated,
            "claude": delegated == "claude_code" or any(s.get("kind") == "delegate" for s in st),
            "agent_grade": r.get("agent_grade"), "corrected": bool(r.get("corrected")),
            "cost_usd": round(r.get("cost_usd") or 0, 4), "claude_cost": round(claude_cost, 4) or None,
            "llm_calls": r.get("llm_calls") or 0, "tool_calls": r.get("tool_calls") or 0,
            "summary": _clip(summary, 220) if summary else None,
            "awaiting": r.get("status") == "awaiting_user"}


def _read_tasks(ch):
    """All of one sister's task rows (list form), newest first; [] without a db/table."""
    con = _connect_optional(ch["db"])
    if con is None:
        return None
    try:
        if not _has_table(con, "tasks"):
            return []
        rows = _q(con, f"SELECT {_task_select(con)} FROM tasks ORDER BY created_ts DESC LIMIT ?",
                  (MAX_TASK_ROWS,))
    finally:
        con.close()
    return [_task_list_row(ch["slug"], r) for r in rows]


def _tally(rows):
    t = {"total": len(rows), "delegated": sum(1 for r in rows if r["claude"]),
         "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
         "claude_cost": round(sum(r["claude_cost"] or 0 for r in rows), 4)}
    for b in TASK_BUCKETS:
        t[b] = sum(1 for r in rows if r["bucket"] == b)
    return t


def collect_tasks(sister=None, outcome=None, q=None, before=None, limit=50):
    """The Tasks tab list: every sister's tasks merged newest first, filtered (sister slug, outcome
    bucket, text in goal/summary/id), paged by a `<created_ts>|<id>` cursor, plus a tally per
    sister over ALL her tasks."""
    try:
        limit = max(1, min(int(limit or 50), 200))
    except (TypeError, ValueError):
        limit = 50
    sister = sister if sister in CHARACTERS else None
    outcome = outcome if outcome in TASK_BUCKETS else None
    q = " ".join(str(q or "").split())[:120].lower() or None
    before = before if (before and _CURSOR_RE.match(str(before))) else None
    data = _stamp(TASKS_TAB, "Tasks")
    house = _connect_optional(HOUSE_DB)
    try:
        data["household"] = _collect_household(house)
    finally:
        if house is not None:
            house.close()
    everything, tallies, sisters = [], {}, []
    for slug, ch in CHARACTERS.items():
        rows = _read_tasks(ch)
        sisters.append({"slug": slug, "name": ch["name"], "started": rows is not None})
        rows = rows or []
        tallies[slug] = _tally(rows)
        everything.extend(rows)
    rows = everything
    if sister:
        rows = [r for r in rows if r["sister"] == sister]
    if outcome:
        rows = [r for r in rows if r["bucket"] == outcome]
    if q:
        rows = [r for r in rows if q in " ".join(str(r.get(k) or "") for k in ("goal", "summary", "id")).lower()]
    rows.sort(key=lambda r: (str(r["created"] or ""), str(r["id"] or "")), reverse=True)
    matched = len(rows)
    if before:
        bts, bid = before.split("|", 1)
        rows = [r for r in rows if (str(r["created"] or ""), str(r["id"] or "")) < (bts, bid)]
    page_rows = rows[:limit]
    has_more = len(rows) > limit
    data["tasks"] = {"rows": page_rows, "has_more": has_more,
                     "next_before": (f"{page_rows[-1]['created']}|{page_rows[-1]['id']}"
                                     if has_more and page_rows else None),
                     "before": before, "limit": limit, "matched": matched,
                     "filters": {"sister": sister, "outcome": outcome, "q": q},
                     "tally": tallies, "sisters": sisters, "buckets": list(TASK_BUCKETS)}
    return data


def collect_task(who, task_id):
    """One task in full for `/api/task`. Raises KeyError (unknown who / no such task) or
    ValueError (malformed id) — the route maps them to 404 / 400."""
    ch = CHARACTERS.get(who)
    if ch is None:
        raise KeyError(f"unknown who {who!r}")
    if not isinstance(task_id, str) or not TASK_ID_RE.match(task_id):
        raise ValueError("malformed task id")
    con = _connect_optional(ch["db"])
    if con is None:
        raise KeyError(f"{ch['name']} has no database yet")
    try:
        if not _has_table(con, "tasks"):
            raise KeyError("no tasks table")
        rows = _q(con, f"SELECT {_task_select(con)} FROM tasks WHERE id = ?", (task_id,))
    finally:
        con.close()
    if not rows:
        raise KeyError(f"no task {task_id} for {ch['name']}")
    r = rows[0]
    out = _task_list_row(ch["slug"], r)
    plan = _jloads(r.get("plan_json"), [])
    log = _jloads(r.get("log_json"), [])
    pending = _jloads(r.get("pending_json"), None)
    timeline = task_timeline(log)
    decision = next((e["detail"] for e in reversed(timeline) if e["event"] == "decision"), None)
    familiarity = next((e["detail"] for e in timeline if e["event"] == "familiarity"), None)
    out.update({"goal": r.get("goal_text"), "details": r.get("details") or "",
                "family": r.get("family"), "approach": r.get("approach"),
                "result_summary": r.get("result_summary"), "agent_grade_reason": r.get("agent_grade_reason"),
                "user_feedback": r.get("user_feedback"), "flow_id": r.get("flow_id"),
                "finished_ago": _ago(r.get("finished_ts")), "decision": decision, "familiarity": familiarity,
                "steps": task_steps(plan), "timeline": timeline,
                "announcements": sum(1 for e in timeline if e["secondary"]),
                "pending": ({"kind": pending.get("kind"), "text": _clip(pending.get("text"), 1000),
                             "payload": {str(k): _clip_obj(v, 1000) for k, v in (pending.get("payload") or {}).items()}
                             if isinstance(pending.get("payload"), dict) else None}
                            if isinstance(pending, dict) else None)})
    created = _parse_ts(r.get("created_ts"))
    until = _parse_ts(r.get("finished_ts")) or _parse_ts(r.get("updated_ts"))
    jobs = claude_jobs(task_id, plan, log)
    for j in jobs:
        res = job_result(j["job"])
        j["result"] = res
        sid = ((res or {}).get("payload") or {}).get("session_id")
        if sid and not any(rn.get("session") == sid for rn in j["runs"]):
            last = next((rn for rn in reversed(j["runs"]) if not rn.get("session")), None)
            if last is not None:
                last["session"] = sid
            else:
                j["runs"].append({"started": None, "ended": None, "session": sid, "cost": None, "status": None})
        _attach_inferred(j, created, until)
        for rn in j["runs"]:
            rn["started_ago"] = _ago(rn.get("started"))
            path = find_transcript(rn.get("session")) if rn.get("session") else None
            rn["transcript_path"] = path
            act = read_transcript(path) if path else None
            rn["took"] = _took(rn.get("started") or (act or {}).get("first_ts"),
                               rn.get("ended") or (act or {}).get("last_ts"))
            if act and res and act.get("final_text") and _norm_words(act["final_text"])[:400] == _norm_words(res.get("text"))[:400]:
                act["final_same_as_result"] = True
                act["final_text"] = None
            rn["activity"] = act
            if rn.get("cost") is None and act and act.get("cost_usd") is not None:
                rn["cost"] = act["cost_usd"]
        j["cost"] = round(sum(rn["cost"] or 0 for rn in j["runs"]), 4) or None
    out["claude_jobs"] = jobs
    out["claude_cost"] = round(sum(j["cost"] or 0 for j in jobs), 4) or out.get("claude_cost")
    out["generated_ts"] = datetime.now().isoformat()
    return out


def task_response(query=""):
    """(status, payload) for `GET /api/task?who=<slug>&id=<task id>` — 400 on a malformed id,
    404 for an unknown sister or a task that isn't there."""
    qs = parse_qs(query or "", keep_blank_values=False)
    who = (qs.get("who", [""])[0] or "").strip().lower()
    task_id = (qs.get("id", [""])[0] or "").strip()
    if who not in CHARACTERS:
        return 404, {"error": f"unknown who {who!r}", "known": list(CHARACTERS)}
    if not TASK_ID_RE.match(task_id):
        return 400, {"error": "malformed task id"}
    try:
        return 200, collect_task(who, task_id)
    except KeyError as e:
        return 404, {"error": str(e).strip("'\"")}
    except ValueError as e:
        return 400, {"error": str(e)}
    except Exception as e:  # never let one odd task kill the page
        return 500, {"error": f"collect failed: {e}"}


def _transcript_line(r):
    ts = r.get("ts") or ""
    audible = bool(r.get("audible"))
    return {"id": r.get("id"), "ts": ts, "day": ts[:10], "time": ts[11:19], "when": _ago(ts),
            "turn_id": r.get("turn_id"),
            "speaker": r.get("speaker"), "speaker_name": _who_label(r.get("speaker")),
            "addressee": r.get("addressee"), "addressee_name": _who_label(r.get("addressee")),
            "text": r.get("text") or "", "audible": audible,
            "badge": "audible" if audible else "private",
            "interrupted": bool(r.get("interrupted")), "kind": r.get("kind") or ""}


def _collect_household(house=None):
    """The strip on every tab: each body up/down (avatar state file + pid), the house process
    (state file pid alive + port), who is connected to it (house_kv.present), the sister-talk
    mode (house_kv.sister_talk) and this month's TTS characters per character + total
    (service_usage tts / tts.<slug>). `house` is an open read-only house.db connection or None."""
    period = datetime.now().strftime("%Y-%m")
    bodies = []
    for slug, ch in CHARACTERS.items():
        av = _load_json(ch.get("avatar")) or {}
        pid = av.get("pid")
        bodies.append({"slug": slug, "name": ch["name"], "up": _pid_alive(pid), "pid": pid,
                       "monitor": av.get("monitor"), "layer": av.get("layer"), "muted": False})
    st = _load_json(HOUSE_STATE) or {}
    hpid = st.get("pid")
    house_info = {"up": bool(_pid_alive(hpid) and st.get("port")), "pid": hpid,
                  "port": st.get("port"), "boot_id": st.get("boot_id"),
                  "protocol": st.get("protocol"), "since": _ago(st.get("ts")),
                  "state_present": bool(st), "state": None}
    if house_info["up"]:
        # the live {awake, mic_muted} for the Controls labels — a hello→welcome→bye handshake as a
        # control client; None when the house doesn't answer in time (labels fall back to neutral)
        house_info["state"] = _peek_house(st.get("port"))
    present, sister_talk = [], None
    try:
        cap = int(os.getenv("AIVA_TTS_MONTHLY_CHARS", "0") or 0)
    except ValueError:
        cap = 0
    tts = {"period": period, "total": 0, "per": {slug: 0 for slug in CHARACTERS}, "cap": cap}
    if house is not None:
        kv = {r["key"]: r["value"] for r in _q(house, "SELECT key, value FROM house_kv")}
        sister_talk = kv.get("sister_talk")
        muted = {s for s in (kv.get("muted_slugs", "") or "").split(",") if s}
        for body in bodies:
            body["muted"] = body["slug"] in muted
        raw = _jloads(kv.get("present"), [])
        if isinstance(raw, list):
            present = [s for s in raw if isinstance(s, str)]
        for r in _q(house, "SELECT service, used FROM service_usage WHERE period=? "
                           "AND (service='tts' OR service LIKE 'tts.%')", (period,)):
            used = int(r.get("used") or 0)
            if r["service"] == "tts":
                tts["total"] = used
            else:
                tts["per"][r["service"][4:]] = used
    if not house_info["up"]:
        present = []  # a dead house has nobody connected, whatever it last wrote
    return {"house": house_info, "bodies": bodies, "present": present,
            "present_names": [_who_label(s) for s in present], "sister_talk": sister_talk,
            "tts": tts, "house_db_present": os.path.exists(HOUSE_DB)}


def _collect_presence(ch):
    av = _load_json(ch.get("avatar")) or {}
    pid = av.get("pid")
    running = _pid_alive(pid)

    # newest log file → recent activity feed + coarse awake/asleep guess
    recent, last_mtime, state = [], None, "unknown"
    logs = sorted(glob.glob(os.path.join(_LOG_DIR, f"{ch['slug']}-*.log")), reverse=True)
    if logs:
        try:
            last_mtime = os.path.getmtime(logs[0])
            with open(logs[0], encoding="utf-8", errors="replace") as f:
                lines = [ln.rstrip("\n") for ln in f if ln.strip()]
            recent = lines[-14:]
            # last real sleep/wake transition wins (ignore rejected/candidate noise)
            for ln in reversed(lines):
                low = ln.lower()
                if "asleep (" in low or "going to sleep" in low or low.rstrip().endswith("house asleep"):
                    state = "asleep"
                    break
                if "wake confirmed" in low or "i'm back" in low or "intro" in low \
                        or low.rstrip().endswith("awake"):
                    state = "awake"
                    break
        except OSError:
            pass
    if not running:
        state = "offline"
    return {
        "avatar_running": running, "avatar_pid": pid,
        "monitor": av.get("monitor"), "layer": av.get("layer"),
        "placement": av.get("placement"), "boot_id": av.get("boot_id"),
        "state": state,
        "last_activity": _ago(last_mtime),
        "last_activity_secs": (time.time() - last_mtime) if last_mtime else None,
        "recent_log": recent,
    }


def _collect_cat(cat_state_path):
    c = _load_json(cat_state_path)
    if not c:
        return None
    return {
        "name": c.get("name"),
        "hunger": round(c.get("hunger", 0), 1),
        "mood": round(c.get("mood", 0), 1),
        "energy": round(c.get("energy", 0), 1),
        "total_pets": c.get("total_pets", 0),
        "dismissed": bool(c.get("dismissed")),
        "last_fed": _ago(c.get("last_fed")),
        "last_seen": _ago(c.get("last_seen")),
    }


def _read_mood(inner_state_path):
    """Current mood dials from her inner_state.json, decayed forward to now (so the dashboard
    shows what she feels *right now*, not the last-saved value). None until it exists."""
    data = _load_json(inner_state_path)
    if not data:
        return None
    if _INNER_OK:
        try:
            brain = InnerBrain()
            st = InnerState()
            for k, v in data.items():
                if hasattr(st, k):
                    setattr(st, k, v)
            brain.decay(st, datetime.now().isoformat())
            return st.to_public()
        except Exception:
            pass
    return {"valence": round(data.get("valence", 0), 3), "arousal": round(data.get("arousal", 0), 3),
            "social": round(data.get("social", 0), 3), "focus": data.get("focus"),
            "mood_word": None, "last_update": data.get("last_update")}


def _collect_inner(con, inner_state_path=None):
    """Her nervous system: current mood, the recent experience stream, her private diary,
    and her open wishes. All degrade to empty — these tables/files only appear once the
    self loop is wired live."""
    events, total, journal, desires, dtotal = [], 0, [], [], 0
    if con is not None:
        events = _q(con, "SELECT ts, kind, source, summary, valence_hint, arousal_hint, "
                         "social_hint, salience FROM experience ORDER BY id DESC LIMIT 20")
        total = _one(con, "SELECT COUNT(*) FROM experience") or 0
        journal = _q(con, "SELECT ts, entry, focus, valence, arousal, social, carry FROM journal "
                          "ORDER BY id DESC LIMIT 8")
        for j in journal:
            j["when"] = _ago(j.get("ts"))
        desires = _q(con, "SELECT text, kind, strength, source, status FROM desires "
                          "WHERE status = 'open' ORDER BY strength DESC, id DESC LIMIT 12")
        for d in desires:
            d["strength"] = round(d.get("strength") or 0, 2)
        dtotal = _one(con, "SELECT COUNT(*) FROM desires WHERE status = 'open'") or 0
    initiative = None
    if con is not None:
        kv = {r["key"]: r["value"] for r in _q(con, "SELECT key, value FROM self_kv")}
        if kv:
            day_raw = kv.get("initiative_day", "") or ""
            today = datetime.now().strftime("%Y-%m-%d")
            today_count = 0
            if day_raw.startswith(today + ":"):
                try:
                    today_count = int(day_raw.split(":", 1)[1])
                except ValueError:
                    today_count = 0
            try:
                feedback = json.loads(kv.get("initiative_feedback") or "{}")
            except ValueError:
                feedback = {}
            initiative = {"permission": kv.get("initiative", "ask"), "today": today_count,
                          "last": _ago(kv.get("last_initiative_ts")),
                          "welcomed": int(feedback.get("welcomed", 0)),
                          "ignored": int(feedback.get("ignored", 0)),
                          "last_talk": _ago(kv.get("last_talk_ts"))}
    return {"mood": _read_mood(inner_state_path), "events": events, "events_total": total,
            "journal": journal, "desires": desires, "desires_total": dtotal,
            "initiative": initiative, "life": _collect_life(con)}


def _collect_life(con):
    """v3: her story, the habits her nights formed, moments that landed (or fell flat), her
    projects, and what she's learned of his rhythm. All empty until the tables appear."""
    if con is None:
        return {}
    story = _q(con, "SELECT ts, chapter, text, turning_point FROM life_story ORDER BY id DESC LIMIT 1")
    habits = _q(con, "SELECT text, strength, evidence, last_reinforced_ts FROM habits "
                     "WHERE status = 'active' ORDER BY strength DESC LIMIT 8")
    for h in habits:
        h["strength"] = round(h.get("strength") or 0, 2)
        h["when"] = _ago(h.get("last_reinforced_ts"))
    moments = _q(con, "SELECT ts, his, hers, label, why, better FROM moments ORDER BY id DESC LIMIT 8")
    for m in moments:
        m["when"] = _ago(m.get("ts"))
    projects = _q(con, "SELECT title, kind, status, steps, target, last_step_ts FROM projects "
                       "ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, id DESC LIMIT 8")
    for pr in projects:
        pr["when"] = _ago(pr.get("last_step_ts"))
    dreams = _q(con, "SELECT ts, summary_json FROM dream_log ORDER BY id DESC LIMIT 1")
    last_dream = None
    if dreams:
        try:
            last_dream = {"when": _ago(dreams[0]["ts"]), **json.loads(dreams[0]["summary_json"] or "{}")}
        except ValueError:
            last_dream = {"when": _ago(dreams[0]["ts"])}
    slots = _q(con, "SELECT slot, p, n FROM rhythm WHERE n >= 3 ORDER BY p DESC")
    usual = [r["slot"] for r in slots if (r.get("p") or 0) >= 0.7]
    return {"story": story[0] if story else None, "habits": habits, "moments": moments,
            "projects": projects, "last_dream": last_dream,
            "rhythm": {"learned": len(slots), "usual": _usual_hours(usual)}}


_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _usual_hours(slots):
    """['0-19', '0-20', '2-19'] → ['Mon 19-21h', 'Wed 19-20h'] (contiguous hours merged)."""
    by_day = {}
    for sl in slots:
        try:
            d, h = (int(x) for x in sl.split("-"))
        except ValueError:
            continue
        by_day.setdefault(d, []).append(h)
    out = []
    for d in sorted(by_day):
        hours = sorted(by_day[d])
        start = prev = hours[0]
        for h in hours[1:] + [None]:
            if h is not None and h == prev + 1:
                prev = h
                continue
            out.append(f"{_WEEKDAYS[d]} {start}-{prev + 1}h")
            if h is not None:
                start = prev = h
    return out


def _collect_preferences(con):
    """Her mutable likes / dislikes / opinions (preferences table). Signed valence = strength·sign;
    grouped for display. Degrades to empty until the table exists (post-restart with the seed)."""
    if con is None:
        return None
    try:
        rows = _q(con, "SELECT subject, stance, kind, valence, evidence_count, origin, status "
                       "FROM preferences WHERE status = 'active' "
                       "ORDER BY ABS(valence) DESC, last_reinforced_ts DESC LIMIT 40")
    except sqlite3.Error:
        return None
    likes, dislikes, opinions = [], [], []
    for r in rows:
        item = {"subject": r["subject"], "stance": r["stance"], "origin": r["origin"],
                "evidence": r["evidence_count"], "valence": round(r["valence"] or 0, 2)}
        if r["kind"] == "opinion":
            opinions.append(item)
        elif (r["valence"] or 0) >= 0:
            likes.append(item)
        else:
            dislikes.append(item)
    active_total = _one(con, "SELECT COUNT(*) FROM preferences WHERE status = 'active'") or 0
    dropped = _one(con, "SELECT COUNT(*) FROM preferences WHERE status = 'dropped'") or 0
    return {"likes": likes, "dislikes": dislikes, "opinions": opinions,
            "active_total": active_total, "dropped": dropped}


def _collect_space(con):
    """Things she's gathered from the web and kept (finds table). Degrades to empty until the
    table exists (created the first time she saves a find, post-restart)."""
    if con is None:
        return None
    try:
        rows = _q(con, "SELECT kind, subject, why, source_url, path, tags, added_ts FROM finds "
                       "WHERE status = 'kept' ORDER BY updated_ts DESC, id DESC LIMIT 30")
    except sqlite3.Error:
        return None
    items = []
    for r in rows:
        items.append({"kind": r["kind"], "subject": r["subject"], "why": r["why"],
                      "source_url": r["source_url"], "has_file": bool(r["path"]),
                      "tags": [t for t in (r["tags"] or "").split(",") if t],
                      "when": _ago(r["added_ts"])})
    kept = _one(con, "SELECT COUNT(*) FROM finds WHERE status = 'kept'") or 0
    dropped = _one(con, "SELECT COUNT(*) FROM finds WHERE status = 'dropped'") or 0
    return {"items": items, "kept": kept, "dropped": dropped}


_SVC_LABEL = {"tavily": "Web search & finds", "tts": "Voice (ElevenLabs)",
              "stt": "Listening (Deepgram)", "tts.aiva": "Voice — Aiva", "tts.saga": "Voice — Saga"}


def _collect_limits(con, house=None):
    """Per-service usage vs each service's own budget + what's currently PAUSED and why (so runaway
    spend can't happen silently). Caps come from env; usage/pause from the SHARED house.db (every
    process records there) — falling back to the character's own db when there is no house.db yet
    (an old single-db install). Per-item book/task costs stay per character."""
    src = house if house is not None else con
    if src is None:
        return None
    caps = {"tavily": int(os.getenv("AIVA_TAVILY_MONTHLY_CAP", "1000")),
            "tts": int(os.getenv("AIVA_TTS_MONTHLY_CHARS", "0")),
            "stt": int(os.getenv("AIVA_STT_MONTHLY_MINUTES", "0"))}
    period = datetime.now().strftime("%Y-%m")
    services = []
    for svc, cap in caps.items():
        if not cap or cap <= 0:
            continue
        used = _one(src, "SELECT used FROM service_usage WHERE service=? AND period=?",
                    (svc, period)) or 0
        paused = bool(_one(src, "SELECT 1 FROM service_pause WHERE service=? AND period=?",
                           (svc, period)))
        services.append({"service": svc, "label": _SVC_LABEL.get(svc, svc), "used": int(used),
                         "cap": int(cap), "pct": min(100, int(100 * used / cap)), "paused": paused})
    paused = _q(src, "SELECT service, reason, resets_ts FROM service_pause WHERE period=?", (period,))
    for p in paused:
        p["label"] = _SVC_LABEL.get(p.get("service"), p.get("service"))
        p["resets"] = (p.get("resets_ts") or "")[:10]
    book_cost = (_one(con, "SELECT COALESCE(SUM(cost_usd),0) FROM books") or 0) if con is not None else 0
    task_cost = (_one(con, "SELECT COALESCE(SUM(cost_usd),0) FROM tasks") or 0) if con is not None else 0
    return {"services": services, "paused": paused,
            "source": "house" if house is not None else "character",
            "book_cost": round(book_cost, 2), "task_cost": round(task_cost, 2)}


def _collect_stats(con, days=7):
    """Operations profiler: roll up the metrics table by system.op over the last `days`, so the
    biggest time-sinks are obvious. Degrades to empty until the metrics table exists (post-restart)."""
    if con is None:
        return None
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = _q(con, "SELECT system, op, COUNT(*) n, AVG(ms) mean, MAX(ms) mx, SUM(ms) total, "
                   "SUM(cost_usd) cost, SUM(tokens) toks, SUM(ok) oks FROM metrics WHERE ts >= ? "
                   "GROUP BY system, op ORDER BY total DESC", (cutoff,))
    total_ms = sum((r["total"] or 0) for r in rows) or 1
    ops = []
    for r in rows:
        ops.append({"label": f"{r['system']}.{r['op']}", "n": r["n"],
                    "mean_ms": round(r["mean"] or 0), "max_ms": round(r["mx"] or 0),
                    "total_ms": round(r["total"] or 0),
                    "share": round(100 * (r["total"] or 0) / total_ms),
                    "cost": round(r["cost"] or 0, 4), "fails": (r["n"] or 0) - (r["oks"] or 0)})
    turn = next((o for o in ops if o["label"] == "turn.latency"), None)
    side = _one(con, "SELECT COALESCE(SUM(cost_usd),0) FROM metrics WHERE ts>=? AND system='llm'",
                (cutoff,)) or 0
    reading_cost = _one(con, "SELECT COALESCE(SUM(cost_usd),0) FROM books") or 0
    task_cost = _one(con, "SELECT COALESCE(SUM(cost_usd),0) FROM tasks") or 0
    return {"days": days, "ops": ops[:22], "turn": turn,
            "cost": {"side": round(side, 4), "reading": round(reading_cost, 4),
                     "task": round(task_cost, 4)}}


def _collect_reading(con):
    books = _q(con, "SELECT * FROM books ORDER BY "
                    "CASE status WHEN 'reading' THEN 0 WHEN 'new' THEN 1 "
                    "WHEN 'finished' THEN 2 ELSE 3 END, last_read_ts DESC")
    out_books, current = [], None
    for b in books:
        notes = _one(con, "SELECT COUNT(*) FROM book_notes WHERE book_id=?", (b["id"],)) or 0
        sess = _one(con, "SELECT COUNT(*) FROM book_sessions WHERE book_id=?", (b["id"],)) or 0
        sections = b.get("sections") or 0
        cur = b.get("cursor_section") or 0
        pct = int(100 * cur / sections) if sections else 0
        row = {
            "id": b["id"], "title": b["title"], "author": b.get("author") or "unknown",
            "kind": b.get("kind"), "format": b.get("format"), "status": b["status"],
            "status_reason": b.get("status_reason"),
            "cursor": cur, "sections": sections, "pct": pct, "notes": notes, "sessions": sess,
            "words": b.get("words") or 0, "cost_usd": round(b.get("cost_usd") or 0, 4),
            "last_read": _ago(b.get("last_read_ts")), "added": _ago(b.get("added_ts")),
        }
        out_books.append(row)
        if b["status"] == "reading" and current is None:
            sec = _q(con, "SELECT title FROM book_sections WHERE book_id=? AND idx=?",
                     (b["id"], min(cur, sections - 1) if sections else 0))
            last_feel = _q(con, "SELECT feeling, summary FROM book_notes WHERE book_id=? "
                                "ORDER BY section_idx DESC LIMIT 1", (b["id"],))
            current = dict(row)
            current["section_title"] = sec[0]["title"] if sec else None
            current["rolling"] = (b.get("rolling_summary") or "")[:400]
            current["last_feeling"] = last_feel[0]["feeling"] if last_feel else None

    wishlist = _q(con, "SELECT title, author, why, fulfilled_book_id, added_ts FROM wishlist "
                       "ORDER BY added_ts DESC")
    for w in wishlist:
        w["added"] = _ago(w.pop("added_ts", None))
        w["fulfilled"] = bool(w.pop("fulfilled_book_id", None))

    perm = _one(con, "SELECT value FROM becoming_state WHERE key='reading_permission'") or "ask"
    total_cost = _one(con, "SELECT COALESCE(SUM(cost_usd),0) FROM books") or 0
    return {"current": current, "books": out_books, "wishlist": wishlist,
            "permission": perm, "total_cost": round(total_cost, 4)}


def _collect_becoming(con):
    # temperament rows have their own panel ("Who she is so far"); this one is what reading gave her
    traits = _q(con, "SELECT text, kind, strength, status, source_book_id, adopted_ts, "
                     "announced, veto_reason FROM traits WHERE kind != 'temperament' "
                     "ORDER BY status='active' DESC, strength DESC")
    for t in traits:
        t["strength"] = round(t.get("strength") or 0, 2)
        t["adopted"] = _ago(t.pop("adopted_ts", None))
        if t.get("source_book_id"):
            t["book"] = _one(con, "SELECT title FROM books WHERE id=?", (t["source_book_id"],))
        t.pop("source_book_id", None)
    voice = _q(con, "SELECT kind, note, example, strength, status FROM voice_notes "
                    "ORDER BY status='active' DESC, strength DESC")
    for v in voice:
        v["strength"] = round(v.get("strength") or 0, 2)
    lessons = _q(con, "SELECT lesson, kind, domains, strength, applied_count, status FROM reading_lessons "
                      "ORDER BY status='active' DESC, strength DESC")
    for le in lessons:
        le["strength"] = round(le.get("strength") or 0, 2)
        le["domains"] = _jloads(le.get("domains"), [])
    state = {r["key"]: r["value"] for r in _q(con, "SELECT key, value FROM becoming_state")}
    return {"traits": traits, "voice": voice, "lessons": lessons,
            "permission": state.get("reading_permission", "ask"),
            "last_fade": _ago(state.get("last_fade_ts"))}


def _collect_temperament(con):
    """"Who she is so far": self-formed temperament (traits WHERE kind='temperament'), strongest
    first, each marked core at >= TEMPERAMENT_CORE; `forming` while fewer than TEMPERAMENT_FORMED
    core traits. Tolerates a missing traits table and a missing `source` column (older schema)."""
    if not _has_table(con, "traits"):
        return {"available": False, "traits": [], "core_count": 0, "forming": True,
                "core_mark": TEMPERAMENT_CORE, "formed_at": TEMPERAMENT_FORMED}
    has_source = _has_column(con, "traits", "source")
    cols = "text, strength, status, adopted_ts, last_reinforced_ts" + (", source" if has_source else "")
    rows = _q(con, f"SELECT {cols} FROM traits WHERE kind='temperament' AND status='active' "
                   "ORDER BY strength DESC, last_reinforced_ts DESC")
    traits = []
    for r in rows:
        s = round(r.get("strength") or 0, 2)
        traits.append({"text": r.get("text"), "strength": s, "core": s >= TEMPERAMENT_CORE,
                       "source": r.get("source") if has_source else None,
                       "adopted": _ago(r.get("adopted_ts")),
                       "reinforced": _ago(r.get("last_reinforced_ts"))})
    core = sum(1 for t in traits if t["core"])
    dropped = _one(con, "SELECT COUNT(*) FROM traits WHERE kind='temperament' AND status!='active'") or 0
    return {"available": True, "traits": traits, "core_count": core,
            "forming": core < TEMPERAMENT_FORMED, "dropped": dropped,
            "core_mark": TEMPERAMENT_CORE, "formed_at": TEMPERAMENT_FORMED}


def _collect_kin(con, limit=8):
    """"Her sister": what she has noticed her sister do/make/say/read — kin_observations, the
    strongest `limit` active rows. Tolerates the table not existing yet (kin lands separately)."""
    if not _has_table(con, "kin_observations"):
        return {"available": False, "items": [], "total": 0, "dropped": 0, "by_kind": {}}
    rows = _q(con, "SELECT about, kind, capability, summary, last_example, count, strength, "
                   "first_ts, last_ts FROM kin_observations WHERE status='active' "
                   "ORDER BY strength DESC, count DESC, last_ts DESC LIMIT ?", (int(limit),))
    items = []
    for r in rows:
        items.append({"about": r.get("about"), "about_name": _who_label(r.get("about")),
                      "kind": r.get("kind"), "capability": r.get("capability"),
                      "summary": r.get("summary"), "last_example": r.get("last_example"),
                      "count": int(r.get("count") or 0), "strength": round(r.get("strength") or 0, 2),
                      "first": _ago(r.get("first_ts")), "last": _ago(r.get("last_ts")),
                      "last_ts": r.get("last_ts")})
    total = _one(con, "SELECT COUNT(*) FROM kin_observations WHERE status='active'") or 0
    dropped = _one(con, "SELECT COUNT(*) FROM kin_observations WHERE status!='active'") or 0
    by_kind = {r["kind"]: r["c"] for r in _q(con, "SELECT kind, COUNT(*) c FROM kin_observations "
                                                  "WHERE status='active' GROUP BY kind")}
    return {"available": True, "items": items, "total": total, "dropped": dropped, "by_kind": by_kind}


def _collect_tasks(con):
    tasks = _q(con, "SELECT id, goal_text, status, outcome, agent_grade, corrected, delegated_to, "
                    "family, approach, cost_usd, duration_s, created_ts, finished_ts "
                    "FROM tasks ORDER BY created_ts DESC LIMIT 12")
    for t in tasks:
        t["created"] = _ago(t.pop("created_ts", None))
        t["finished"] = _ago(t.pop("finished_ts", None))
        t["cost_usd"] = round(t.get("cost_usd") or 0, 4)
        t["corrected"] = bool(t.get("corrected"))
        t["full_id"] = t.get("id") or ""          # the Tasks tab link (#tasks/<slug>/<id>)
        t["id"] = (t.get("id") or "")[:8]
    flows = _q(con, "SELECT name, family, trigger_phrases, run_count, success_count, "
                    "confirmed_by_user, replay, last_run_ts FROM flows ORDER BY run_count DESC")
    for fl in flows:
        fl["triggers"] = _jloads(fl.pop("trigger_phrases", None), [])
        fl["confirmed_by_user"] = bool(fl.get("confirmed_by_user"))
        fl["last_run"] = _ago(fl.pop("last_run_ts", None))
    prefs = _q(con, "SELECT family, chosen_approach, user_reason, ts FROM approach_preferences "
                    "ORDER BY ts DESC LIMIT 8")
    for p in prefs:
        p["when"] = _ago(p.pop("ts", None))
    # outcome tally
    total = _one(con, "SELECT COUNT(*) FROM tasks") or 0
    done = _one(con, "SELECT COUNT(*) FROM tasks WHERE outcome='success'") or 0
    corrected = _one(con, "SELECT COUNT(*) FROM tasks WHERE corrected=1") or 0
    delegated = _one(con, "SELECT COUNT(*) FROM tasks WHERE delegated_to!='none'") or 0
    return {"recent": tasks, "flows": flows, "prefs": prefs,
            "tally": {"total": total, "success": done, "corrected": corrected,
                      "delegated": delegated}}


def _collect_memory(con):
    by_cat = {r["category"]: r["c"] for r in
              _q(con, "SELECT category, COUNT(*) c FROM facts GROUP BY category ORDER BY c DESC")}
    recent_facts = _q(con, "SELECT fact, category, last_confirmed_ts FROM facts "
                           "ORDER BY last_confirmed_ts DESC LIMIT 8")
    for f in recent_facts:
        f["when"] = _ago(f.pop("last_confirmed_ts", None))
    messages = _one(con, "SELECT COUNT(*) FROM messages") or 0
    sessions = _one(con, "SELECT COUNT(DISTINCT session_id) FROM messages") or 0
    recaps = _q(con, "SELECT recap, created_ts FROM session_recaps ORDER BY created_ts DESC LIMIT 3")
    for r in recaps:
        r["when"] = _ago(r.pop("created_ts", None))
        r["recap"] = (r.get("recap") or "")[:280]
    recap_count = _one(con, "SELECT COUNT(*) FROM session_recaps") or 0
    return {"facts_total": sum(by_cat.values()), "facts_by_cat": by_cat,
            "facts_recent": recent_facts, "messages": messages, "sessions": sessions,
            "recaps_total": recap_count, "recaps_recent": recaps}


def _collect_headline(con, data):
    tp = data.get("temperament") or {}
    return {
        "traits": _one(con, "SELECT COUNT(*) FROM traits WHERE status='active' "
                            "AND kind != 'temperament'") or 0,
        "temperament_core": tp.get("core_count", 0),
        "voice_notes": _one(con, "SELECT COUNT(*) FROM voice_notes WHERE status='active'") or 0,
        "lessons": _one(con, "SELECT COUNT(*) FROM reading_lessons WHERE status='active'") or 0,
        "books_total": _one(con, "SELECT COUNT(*) FROM books") or 0,
        "books_finished": _one(con, "SELECT COUNT(*) FROM books WHERE status='finished'") or 0,
        "books_reading": _one(con, "SELECT COUNT(*) FROM books WHERE status='reading'") or 0,
        "sections_read": _one(con, "SELECT COUNT(*) FROM book_notes") or 0,
        "tasks": _one(con, "SELECT COUNT(*) FROM tasks") or 0,
        "flows": _one(con, "SELECT COUNT(*) FROM flows") or 0,
        "facts": _one(con, "SELECT COUNT(*) FROM facts") or 0,
        "sessions": _one(con, "SELECT COUNT(DISTINCT session_id) FROM messages") or 0,
        "messages": _one(con, "SELECT COUNT(*) FROM messages") or 0,
    }


def _collect_capabilities():
    # tool names parsed live from tools.py (no heavy import), grouped
    names = set()
    try:
        with open(os.path.join(_MODULES_DIR, "tools.py"), encoding="utf-8") as f:
            names = set(re.findall(r'name="([a-z_]+)"', f.read()))
    except OSError:
        pass
    grouped, seen = [], set()
    for label, tools in TOOL_GROUPS:
        present = [t for t in tools if t in names]
        seen.update(tools)
        if present:
            grouped.append({"label": label, "tools": present})
    other = sorted(n for n in names if n not in seen)
    if other:
        grouped.append({"label": "Other", "tools": other})

    # modules present on disk, grouped into subsystems, with rough line counts
    present_files = {os.path.basename(p) for p in glob.glob(os.path.join(_MODULES_DIR, "*.py"))}

    def _lines(fn):
        try:
            with open(os.path.join(_MODULES_DIR, fn), encoding="utf-8", errors="replace") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    subsystems, grouped_files = [], set()
    for label, mods in SUBSYSTEMS:
        mm = [{"file": m, "lines": _lines(m)} for m in mods if m in present_files]
        grouped_files.update(mods)
        if mm:
            subsystems.append({"label": label, "modules": mm,
                               "lines": sum(x["lines"] for x in mm)})
    retired = [{"file": m, "note": note} for m, note in RETIRED_MODULES.items()
               if m in present_files]

    return {"tool_count": len(names), "tool_groups": grouped,
            "subsystems": subsystems, "retired": retired,
            "module_count": len([f for f in present_files if f != "__init__.py"])}


# --- controls: the house hotkeys as buttons ----------------------------------------------
# The ONE thing the dashboard ever sends anywhere: a `control` message to the house over its
# loopback socket — exactly what the hotkeys do. Validated in `control_response`, carried by
# `send_control`, executed by `house._on_control`. A control client is a button, never a sister.

CONTROL_ACTIONS = ("mute", "interrupt", "sister_talk", "sleep", "wake", "restart_sisters",
                   "restart_sister", "quit_sister", "start_sister", "mute_sister", "unmute_sister",
                   "start_house", "restart_house", "quit")
# these don't need the house at all — the dashboard runs on this PC and launches the same .bat a
# double-click would (a sister's .bat brings the house up with her)
_LOCAL_LAUNCH_ACTIONS = ("start_sister", "start_house")
_REPO_ROOT = os.path.dirname(_HERE) if "_HERE" in globals() else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def launch_bat(name):
    """Start `start_<name>.bat` (aiva | saga | house) in its own console window, exactly like
    double-clicking it. Returns {"ok", "launched"|"error"}."""
    import subprocess
    name = (name or "").strip().lower()
    if name not in ("aiva", "saga", "house"):
        return {"ok": False, "error": f"nothing to start for {name!r}"}
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("AIVA_DASHBOARD_NO_LAUNCH") == "1":
        # a test (or an explicit opt-out) must never start a real console on this PC
        return {"ok": False, "error": "launching is disabled here"}
    bat = os.path.join(_REPO_ROOT, f"start_{name}.bat")
    if not os.path.isfile(bat):
        return {"ok": False, "error": f"{os.path.basename(bat)} is missing"}
    try:
        subprocess.Popen(["cmd", "/c", bat], cwd=_REPO_ROOT,
                         creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
    except OSError as e:
        return {"ok": False, "error": f"couldn't start {os.path.basename(bat)}: {e}"}
    return {"ok": True, "launched": os.path.basename(bat)}
CONTROL_TARGETS = ("aiva", "saga", "audible", "private")
_SISTER_ACTIONS = ("restart_sister", "quit_sister", "start_sister",
                   "mute_sister", "unmute_sister")                    # need a sister slug
_MODE_TARGETS = ("audible", "private")                                  # sister_talk only
CONTROL_TIMEOUT = 2.0          # a button press
CONTROL_PEEK_TIMEOUT = 0.5     # the per-poll state peek — never let it slow the page
HOUSE_DOWN = "the house isn't running"
MAX_CONTROL_BODY = 4096
MAX_SAY_CHARS = 600            # one typed line — she is answering out loud, not reading an essay


def _proto():
    """modules.house_proto (stdlib-only), or None on a checkout without a house."""
    try:
        from modules import house_proto
        return house_proto
    except Exception:
        return None


def _house_port(state_path=None):
    """The house's listening port from its state file (HOUSE_STATE by default) — None when the
    file is missing/corrupt, has no port, or its pid is dead (the same "up" rule the Household
    strip uses, so a stale file left by a crash never costs a socket timeout)."""
    st = _load_json(state_path or HOUSE_STATE) or {}
    try:
        port = int(st.get("port") or 0)
    except (TypeError, ValueError):
        port = 0
    return port if (port and _pid_alive(st.get("pid"))) else None


def _house_connect(port, timeout):
    """A socket to the house on 127.0.0.1:`port`, or OSError when nobody is accepting there
    (refused — or, behind a firewall that drops loopback SYNs, timed out)."""
    return socket.create_connection(("127.0.0.1", int(port)), timeout=timeout)


def _house_hello(sock, timeout, proto):
    """Say hello as the dashboard's control client and read the one reply. Returns the `welcome`;
    raises PermissionError (a `reject`), ValueError (garbage / not a welcome — ProtoError is a
    ValueError) or OSError (timeout / reset / closed without a word)."""
    sock.settimeout(timeout)
    hello = proto.Hello(slug="dashboard", name="Dashboard", role="control", has_voice=False,
                        pid=os.getpid(), version=str(AIVA_VERSION))
    sock.sendall(proto.encode(hello))
    rfile = sock.makefile("rb")
    try:
        line = rfile.readline(proto.MAX_LINE + 1)
    finally:
        rfile.close()          # the socket itself stays open until the caller closes it
    if not line:
        raise ConnectionError("closed the connection without a welcome")
    reply = proto.decode(line)
    if isinstance(reply, proto.Reject):
        raise PermissionError(f"the house refused the dashboard: {reply.reason}")
    if not isinstance(reply, proto.Welcome):
        raise ValueError(f"expected welcome, got {reply.T}")
    return reply


def send_control(action, target=None, timeout=CONTROL_TIMEOUT, *, state_path=None, port=None):
    """Press one house hotkey from here: hello as a control client, one `control`, bye.
    Returns {"ok": True, "action", "target"} or {"ok": False, "error"}; never raises. Finds the
    house through its state file (`state_path` overrides HOUSE_STATE; `port` skips the file) —
    no file, a stale one (dead pid) or nobody accepting on the port all mean "the house isn't
    running"; a house that answers with anything but a `welcome` is described."""
    proto = _proto()
    if proto is None:
        return {"ok": False, "error": "modules.house_proto is missing — this checkout has no house"}
    port = port or _house_port(state_path)
    if not port:
        return {"ok": False, "error": HOUSE_DOWN}
    try:
        sock = _house_connect(port, timeout)
    except OSError:
        return {"ok": False, "error": HOUSE_DOWN}
    try:
        _house_hello(sock, timeout, proto)
        sock.sendall(proto.encode(proto.Control(action=action, target=target)))
        try:
            sock.sendall(proto.encode(proto.Bye()))
        except OSError:
            pass        # `quit` / `restart_house` may tear the socket down before the bye lands
    except (PermissionError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    except OSError as e:
        return {"ok": False, "error": f"no welcome from the house on :{port} ({e})"}
    finally:
        sock.close()
    return {"ok": True, "action": action, "target": target}


def send_say(text, timeout=CONTROL_TIMEOUT, *, state_path=None, port=None):
    """Type one line to them: hello as a control client, one `say`, bye. The house runs it through
    the Floor exactly like something heard, so whoever it was meant for answers out loud. Returns
    {"ok": True, "text"} or {"ok": False, "error"}; never raises."""
    proto = _proto()
    if proto is None:
        return {"ok": False, "error": "modules.house_proto is missing — this checkout has no house"}
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "nothing to say"}
    port = port or _house_port(state_path)
    if not port:
        return {"ok": False, "error": HOUSE_DOWN}
    try:
        sock = _house_connect(port, timeout)
    except OSError:
        return {"ok": False, "error": HOUSE_DOWN}
    try:
        _house_hello(sock, timeout, proto)
        sock.sendall(proto.encode(proto.Say(text=text)))
        try:
            sock.sendall(proto.encode(proto.Bye()))
        except OSError:
            pass
    except (PermissionError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    except OSError as e:
        return {"ok": False, "error": f"no welcome from the house on :{port} ({e})"}
    finally:
        sock.close()
    return {"ok": True, "text": text}


def _peek_house(port, timeout=CONTROL_PEEK_TIMEOUT):
    """The house's live {awake, mic_muted, mode, present} from a hello→welcome→bye as a control
    client (nothing else is sent). None when it isn't answering; never raises."""
    proto = _proto()
    if proto is None or not port:
        return None
    try:
        sock = _house_connect(port, timeout)
    except OSError:
        return None
    try:
        w = _house_hello(sock, timeout, proto)
        try:
            sock.sendall(proto.encode(proto.Bye()))
        except OSError:
            pass
    except (OSError, ValueError):
        return None
    finally:
        sock.close()
    st = dict(w.state or {})
    return {"awake": st.get("awake"), "mic_muted": st.get("mic_muted"), "mode": w.mode,
            "present": [s for s in (w.present or []) if isinstance(s, str)]}


# --- the page (self-contained; polls /api/data?who=<tab>) ---------------------------

PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aiva Dashboard</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<style>
:root{
  --bg:#0e0b16; --bg2:#161222; --panel:#1b1630; --panel2:#221b3d; --line:#2e2748;
  --ink:#ece8fb; --muted:#a79fce; --dim:#786f9e;
  --accent:#c084fc; --accent2:#f0abfc; --good:#4ade80; --warn:#fbbf24; --bad:#fb7185;
  --saga:#7dd3fc;
  --shadow:0 8px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:radial-gradient(1200px 700px at 80% -10%,#241a3d 0%,var(--bg) 55%);
  color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,-apple-system,sans-serif;min-height:100vh}
a{color:var(--accent2)}
.wrap{max-width:1180px;margin:0 auto;padding:26px 20px 60px}
header{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:8px}
.logo{width:46px;height:46px;border-radius:14px;flex:0 0 auto;
  background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:var(--shadow);
  display:grid;place-items:center;font-size:24px}
h1{font-size:24px;margin:0;letter-spacing:.2px}
.sub{color:var(--muted);font-size:13px;margin-top:2px}
.spacer{flex:1}
.status{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted);
  background:var(--panel);border:1px solid var(--line);padding:7px 12px;border-radius:999px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--dim);box-shadow:0 0 0 0 rgba(0,0,0,0);display:inline-block;flex:0 0 auto}
.dot.live{background:var(--good);animation:pulse 2s infinite}
.dot.asleep{background:var(--warn)}
.dot.offline{background:var(--bad)}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(74,222,128,.5)}70%{box-shadow:0 0 0 7px rgba(74,222,128,0)}100%{box-shadow:0 0 0 0 rgba(74,222,128,0)}}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0 0}
.tab{font:inherit;font-size:13px;padding:7px 16px;border-radius:999px;border:1px solid var(--line);
  background:var(--panel);color:var(--muted);cursor:pointer;letter-spacing:.2px}
.tab:hover{color:var(--ink);border-color:var(--accent)}
.tab.on{color:#0e0b16;background:linear-gradient(90deg,var(--accent),var(--accent2));border-color:transparent;font-weight:600}
.house{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 0}
.chip{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--muted);
  background:var(--panel);border:1px solid var(--line);padding:5px 11px;border-radius:999px}
.chip b{color:var(--ink);font-weight:600}
.badge{font-size:10px;padding:2px 7px;border-radius:6px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;
  background:rgba(148,163,184,.18);color:var(--muted)}
.badge.audible{background:rgba(74,222,128,.16);color:var(--good)}
.badge.private{background:rgba(192,132,252,.18);color:var(--accent2)}
.badge.cut{background:rgba(251,113,133,.16);color:var(--bad)}
.grid{display:grid;gap:16px}
.stats{grid-template-columns:repeat(auto-fit,minmax(120px,1fr));margin:18px 0 8px}
.stat{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);
  border-radius:16px;padding:15px 16px;box-shadow:var(--shadow)}
.stat .n{font-size:30px;font-weight:700;line-height:1}
.stat .l{color:var(--muted);font-size:12px;margin-top:6px;text-transform:uppercase;letter-spacing:.6px}
.stat .s{color:var(--dim);font-size:12px;margin-top:2px}
section{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:18px 20px;
  margin-top:16px;box-shadow:var(--shadow)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.cols{grid-template-columns:1fr}}
h2{font-size:15px;margin:0 0 14px;letter-spacing:.4px;color:var(--accent2);
  text-transform:uppercase;display:flex;align-items:center;gap:9px}
h2 .cnt{color:var(--dim);font-weight:400;font-size:12px;text-transform:none;letter-spacing:0}
.row{display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-top:1px solid var(--line)}
.row:first-of-type{border-top:0}
.row .k{color:var(--ink)}
.row .v{color:var(--muted);text-align:right;font-variant-numeric:tabular-nums}
.bar{height:9px;border-radius:6px;background:var(--bg2);overflow:hidden;margin-top:8px;border:1px solid var(--line)}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2))}
.tag{display:inline-block;font-size:11px;padding:2px 9px;border-radius:999px;border:1px solid var(--line);
  color:var(--muted);background:var(--bg2);margin:2px 4px 2px 0}
.tag.on{color:#0e0b16;background:linear-gradient(90deg,var(--accent),var(--accent2));border-color:transparent;font-weight:600}
.pill{font-size:11px;padding:2px 8px;border-radius:6px;font-weight:600}
.pill.reading{background:rgba(192,132,252,.18);color:var(--accent2)}
.pill.finished{background:rgba(74,222,128,.16);color:var(--good)}
.pill.new{background:rgba(148,163,184,.16);color:var(--muted)}
.pill.abandoned{background:rgba(251,113,133,.16);color:var(--bad)}
.pill.unreadable{background:rgba(251,113,133,.16);color:var(--bad)}
.muted{color:var(--muted)} .dim{color:var(--dim)}
.empty{color:var(--dim);font-style:italic;padding:6px 0}
.item{padding:11px 0;border-top:1px solid var(--line)}
.item:first-child{border-top:0}
.item .t{font-weight:600}
.item .d{color:var(--muted);font-size:13px;margin-top:3px}
.book .head{display:flex;justify-content:space-between;gap:10px;align-items:baseline}
.chips{margin-top:8px}
.log{font:12px/1.7 "Cascadia Code",Consolas,monospace;background:var(--bg2);border:1px solid var(--line);
  border-radius:12px;padding:12px 14px;max-height:260px;overflow:auto;white-space:pre-wrap;color:var(--muted)}
.strength{display:inline-block;width:64px;height:6px;border-radius:4px;background:var(--bg2);vertical-align:middle;
  border:1px solid var(--line);overflow:hidden}
.strength>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2))}
.mods{display:flex;flex-wrap:wrap;gap:6px}
.mod{font:12px/1 "Cascadia Code",Consolas,monospace;background:var(--bg2);border:1px solid var(--line);
  border-radius:8px;padding:6px 9px;color:var(--muted)}
.mod b{color:var(--ink);font-weight:600} .mod .ln{color:var(--dim)}
.stat-table{width:100%;border-collapse:collapse;font-size:13px;min-width:520px}
.stat-table th{text-align:right;color:var(--dim);font-weight:500;padding:6px 10px;border-bottom:1px solid var(--line);font-size:11px;text-transform:uppercase;letter-spacing:.4px}
.stat-table th:first-child,.stat-table td:first-child{text-align:left}
.stat-table td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}
.stat-table tbody tr:last-child td{border-bottom:0}
.stat-table tr:hover td{background:var(--bg2)}
.stat-table td:first-child{font-family:"Cascadia Code",Consolas,monospace;color:var(--ink)}
.sharebar{display:inline-block;width:54px;height:6px;border-radius:4px;background:var(--bg2);border:1px solid var(--line);overflow:hidden;vertical-align:middle;margin-right:6px}
.sharebar>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2))}
footer{color:var(--dim);font-size:12px;margin-top:24px;text-align:center}
.vitals{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:6px}
.vital .lab{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.vital .bar{margin-top:5px}
.banner{background:rgba(251,113,133,.12);border:1px solid rgba(251,113,133,.4);color:var(--bad);
  padding:10px 14px;border-radius:12px;margin-top:12px;font-size:14px}
.banner.soft{background:rgba(192,132,252,.10);border-color:rgba(192,132,252,.35);color:var(--accent2)}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px;font-size:13px;color:var(--muted)}
.toolbar select,.toolbar button{font:inherit;font-size:13px;background:var(--bg2);color:var(--ink);border:1px solid var(--line);
  border-radius:8px;padding:5px 10px;cursor:pointer}
.toolbar button:disabled{opacity:.4;cursor:default}
.line{display:grid;grid-template-columns:66px 64px 96px 1fr auto;gap:10px;padding:8px 0;border-top:1px solid var(--line);align-items:baseline}
.line:first-child{border-top:0}
.line .time{color:var(--dim);font:12px/1.6 "Cascadia Code",Consolas,monospace}
.line .spk{font-weight:600;color:var(--ink)}
.line .spk.aiva{color:var(--accent2)} .line .spk.saga{color:var(--saga)}
.line .arrow{color:var(--dim);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.line .txt{color:var(--ink);word-break:break-word}
.line.priv .txt{color:var(--muted)}
.line .marks{display:flex;gap:6px;align-items:center;white-space:nowrap;font-size:11px}
@media(max-width:820px){.line{grid-template-columns:1fr 1fr;gap:4px}.line .txt,.line .marks{grid-column:1/-1}}
.ev{display:grid;grid-template-columns:66px 64px 1fr;gap:10px;padding:6px 0;border-top:1px solid var(--line);font-size:13px;align-items:baseline}
.ev:first-child{border-top:0}
.ev .time{color:var(--dim);font:12px/1.6 "Cascadia Code",Consolas,monospace}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0 0;padding:10px 12px;
  background:var(--panel);border:1px solid var(--line);border-radius:14px}
.controls:empty{display:none}
.controls .clabel{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.6px;margin-right:4px}
.controls .csep{width:1px;height:18px;background:var(--line);margin:0 2px}
.cbtn{font:inherit;font-size:12px;padding:5px 12px;border-radius:999px;border:1px solid var(--line);
  background:var(--bg2);color:var(--ink);cursor:pointer;letter-spacing:.2px;white-space:nowrap}
.cbtn:hover{border-color:var(--accent);color:var(--accent2)}
.cbtn.danger{color:var(--bad);border-color:rgba(251,113,133,.35)}
.cbtn.danger:hover{border-color:var(--bad);background:rgba(251,113,133,.10);color:var(--bad)}
.cbtn:disabled,.cbtn:disabled:hover{opacity:.4;cursor:default;border-color:var(--line);background:var(--bg2)}
.controls.down{opacity:.75}
.controls .cnote{font-size:12px;color:var(--warn);flex-basis:100%}
.say{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 12px}
.say input{flex:1;font:inherit;font-size:14px;padding:10px 14px;border-radius:999px;border:1px solid var(--line);
  background:var(--bg2);color:var(--ink);outline:none}
.say input:focus{border-color:var(--accent)}
.say input:disabled{opacity:.5}
.say button{font:inherit;font-size:13px;padding:9px 18px;border-radius:999px;border:1px solid var(--line);
  background:var(--bg2);color:var(--ink);cursor:pointer;white-space:nowrap}
.say button:hover:not(:disabled){border-color:var(--accent);color:var(--accent2)}
.say button:disabled{opacity:.4;cursor:default}
.say .hint{font-size:12px;color:var(--warn);flex-basis:100%}
.toast{position:fixed;right:18px;bottom:18px;z-index:9;font-size:13px;padding:9px 14px;border-radius:12px;
  background:var(--panel2);border:1px solid var(--line);color:var(--ink);box-shadow:var(--shadow);max-width:70vw;
  opacity:0;transform:translateY(8px);transition:opacity .2s,transform .2s;pointer-events:none}
.toast.show{opacity:1;transform:none}
.toast.ok{border-color:rgba(74,222,128,.45)} .toast.bad{border-color:rgba(251,113,133,.5);color:var(--bad)}
.setbar{position:sticky;top:0;z-index:5;display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:16px;
  padding:10px 14px;background:var(--panel2);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);font-size:13px;color:var(--muted)}
.setbar input{flex:1;min-width:160px;font:inherit;font-size:13px;padding:7px 12px;border-radius:999px;border:1px solid var(--line);
  background:var(--bg2);color:var(--ink);outline:none}
.setbar input:focus{border-color:var(--accent)}
.setbar label{display:inline-flex;gap:6px;align-items:center;cursor:pointer;white-space:nowrap}
.setbar .cbtn.primary{color:#0e0b16;background:linear-gradient(90deg,var(--accent),var(--accent2));border-color:transparent;font-weight:600}
.setbar .cbtn.primary:disabled{background:var(--bg2);color:var(--ink)}
.setsaved{flex-basis:100%;display:flex;gap:8px;align-items:center;flex-wrap:wrap;color:var(--good)}
.setsaved:empty{display:none}
.srow{display:grid;grid-template-columns:minmax(0,1fr) minmax(220px,300px);gap:6px 16px;padding:11px 0;border-top:1px solid var(--line);align-items:start}
.srow:first-of-type{border-top:0}
.srow .sk{font:12.5px/1.5 "Cascadia Code",Consolas,monospace;color:var(--ink);word-break:break-all}
.srow .sd{color:var(--muted);font-size:13px;margin-top:2px}
.srow .sm{display:flex;gap:6px;flex-wrap:wrap;margin-top:5px}
.srow .sctl{display:flex;flex-direction:column;gap:4px}
.srow .sline{display:flex;gap:6px;align-items:center}
.srow .sline input[type=text],.srow .sline input[type=number],.srow .sline select{flex:1;min-width:0;font:inherit;font-size:13px;
  padding:6px 10px;border-radius:8px;border:1px solid var(--line);background:var(--bg2);color:var(--ink);outline:none}
.srow .sline input:focus,.srow .sline select:focus{border-color:var(--accent)}
.srow .sline input[type=checkbox]{width:18px;height:18px;accent-color:var(--accent);cursor:pointer}
.srow .shint{font-size:11.5px;color:var(--dim)}
.srow .serr{font-size:12px;color:var(--bad)}
.srow .sreset{font:inherit;font-size:11px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);background:transparent;
  color:var(--muted);cursor:pointer;white-space:nowrap}
.srow .sreset:hover{border-color:var(--accent);color:var(--accent2)}
.srow .sval{font:12.5px/1.5 "Cascadia Code",Consolas,monospace;color:var(--muted);word-break:break-all}
.srow.mod .sk::after{content:" ●";color:var(--accent2)}
.srow.dirty{background:linear-gradient(90deg,rgba(192,132,252,.10),transparent 70%)}
.srow.dirty .sk::after{content:" ● unsaved";color:var(--warn);font-size:11px}
.srow.bad .sline input,.srow.bad .sline select{border-color:var(--bad)}
.badge.ro{background:rgba(148,163,184,.14);color:var(--dim)}
.badge.sec{background:rgba(251,191,36,.14);color:var(--warn)}
.badge.env{background:rgba(192,132,252,.18);color:var(--accent2)}
@media(max-width:820px){.srow{grid-template-columns:1fr}}
.pill.success{background:rgba(74,222,128,.16);color:var(--good)}
.pill.failed{background:rgba(251,113,133,.16);color:var(--bad)}
.pill.cancelled{background:rgba(148,163,184,.16);color:var(--muted)}
.pill.interrupted{background:rgba(251,191,36,.14);color:var(--warn)}
.pill.active{background:rgba(192,132,252,.18);color:var(--accent2)}
.sb{font-size:11px;font-weight:700;letter-spacing:.3px;padding:2px 8px;border-radius:6px;background:rgba(192,132,252,.14);color:var(--accent2)}
.sb.saga{background:rgba(125,211,252,.14);color:var(--saga)}
.badge.cc{background:rgba(240,171,252,.14);color:var(--accent2)}
.tkgrid{display:grid;grid-template-columns:minmax(280px,380px) minmax(0,1fr);gap:16px;align-items:start}
@media(max-width:980px){.tkgrid{grid-template-columns:minmax(0,1fr)}}
.tklist{display:flex;flex-direction:column;gap:8px}
.tcard{background:var(--bg2);border:1px solid var(--line);border-radius:14px;padding:11px 13px;cursor:pointer}
.tcard:hover{border-color:var(--accent)}
.tcard.on{border-color:var(--accent2);background:linear-gradient(180deg,var(--panel2),var(--bg2))}
.tcard .th{display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:12px}
.tcard .th .when{margin-left:auto;color:var(--dim)}
.tcard .tg{font-weight:600;margin-top:6px;word-break:break-word}
.tcard .d{color:var(--muted);font-size:12.5px;margin-top:4px}
.tkpane{position:sticky;top:10px;max-height:calc(100vh - 20px);overflow:auto;background:var(--bg2);border:1px solid var(--line);
  border-radius:14px;padding:16px 18px}
@media(max-width:980px){.tkpane{position:static;max-height:none}}
.tkpane .row .v{text-align:left;max-width:78%}
.tkpane h3{font-size:13px;margin:18px 0 8px;color:var(--accent2);text-transform:uppercase;letter-spacing:.4px}
.tkpane .goal{font-size:17px;font-weight:700;line-height:1.35;margin:8px 0 4px;word-break:break-word}
.mono{font:12px/1.6 "Cascadia Code",Consolas,monospace}
pre.box{font:12px/1.6 "Cascadia Code",Consolas,monospace;background:var(--bg);border:1px solid var(--line);border-radius:10px;
  padding:9px 11px;margin:6px 0 0;white-space:pre-wrap;word-break:break-word;color:var(--muted);max-height:360px;overflow:auto}
.steps{position:relative;margin-left:4px}
.step{position:relative;padding:0 0 14px 30px}
.step::before{content:"";position:absolute;left:10px;top:22px;bottom:0;width:2px;background:var(--line)}
.step:last-child::before{display:none}
.step .ic{position:absolute;left:0;top:1px;width:22px;height:22px;border-radius:50%;display:grid;place-items:center;
  font-size:12px;font-weight:700;background:var(--panel);border:1px solid var(--line);color:var(--dim)}
.step.done .ic{color:var(--good);border-color:rgba(74,222,128,.5)}
.step.failed .ic{color:var(--bad);border-color:rgba(251,113,133,.5)}
.step.running .ic{color:var(--accent2);border-color:var(--accent)}
.step .st{font-weight:600}
.step .d{color:var(--muted);font-size:12.5px;margin-top:2px}
.step .fail{color:var(--bad);font-size:13px;margin-top:4px}
details.more{margin-top:6px}
details.more>summary{cursor:pointer;color:var(--muted);font-size:12.5px}
details.more>summary:hover{color:var(--accent2)}
.tev{display:grid;grid-template-columns:14px 64px 118px minmax(0,1fr);gap:8px;padding:5px 0;border-top:1px solid var(--line);font-size:13px;align-items:baseline}
.tev:first-child{border-top:0}
.tev .time{color:var(--dim);font:12px/1.6 "Cascadia Code",Consolas,monospace}
.tev .evn{color:var(--ink);font-weight:600;font-size:12px;border:0;padding:0;display:block}
.tev .dt{color:var(--muted);word-break:break-word}
.tev.sec{opacity:.6}
.tdot{width:8px;height:8px;border-radius:50%;display:inline-block;background:var(--dim)}
.tdot.good{background:var(--good)} .tdot.bad{background:var(--bad)} .tdot.warn{background:var(--warn)} .tdot.accent{background:var(--accent)}
.ccjob{border:1px solid var(--line);border-radius:12px;padding:11px 13px;margin-top:10px;background:var(--panel)}
.ccrun{border-top:1px dashed var(--line);margin-top:10px;padding-top:10px}
.flist{margin:4px 0 0;padding:0;list-style:none;font:12px/1.7 "Cascadia Code",Consolas,monospace;color:var(--muted)}
.flist li{word-break:break-all}
.flist b{color:var(--ink);font-weight:600}
.toolbar input[type=search]{font:inherit;font-size:13px;background:var(--bg2);color:var(--ink);border:1px solid var(--line);
  border-radius:8px;padding:5px 10px;min-width:180px;flex:1;outline:none}
.toolbar input[type=search]:focus{border-color:var(--accent)}
a.tlink{color:inherit;text-decoration:none} a.tlink:hover .t{color:var(--accent2)}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">✦</div>
    <div>
      <h1 id="title">Aiva</h1>
      <div class="sub" id="subtitle">loading…</div>
    </div>
    <div class="spacer"></div>
    <div class="status"><span class="dot" id="dot"></span><span id="statusText">connecting…</span></div>
  </header>
  <div class="tabs" id="tabs"></div>
  <div class="house" id="household"></div>
  <div class="controls" id="controls"></div>
  <div id="banner"></div>
  <div class="grid stats" id="stats"></div>
  <div id="sayhost"></div>
  <div id="content"></div>
  <footer>Reads the databases <b>read-only</b> — safe to keep open; the Controls strip only sends the
    house hotkeys over its local socket, and Settings' Save only edits .env. Polls every __POLL__s · <span id="stamp"></span></footer>
</div>
<div class="toast" id="toast"></div>
<script>
const POLL = __POLL__ * 1000;
const TABS = __TABS__;
const TEMP_CORE = __TEMP_CORE__;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s==null?"":s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const bar = (pct) => `<div class="bar"><i style="width:${Math.max(0,Math.min(100,pct))}%"></i></div>`;
const strength = (s) => `<span class="strength"><i style="width:${Math.round((s||0)*100)}%"></i></span>`;
const nameOf = (who) => { const t = TABS.find(t=>t.who===who); return t ? t.label : who; };
const fmtK = (n) => { n = n||0; return n>=1000000 ? (n/1000000).toFixed(1)+'M' : n>=1000 ? (n/1000).toFixed(n>=10000?0:1)+'k' : String(n); };

// --- tabs: which core we are looking at lives in location.hash ---------------------
let WHO = pickWho();
let CONV = {day:null, before:null};      // conversations tab: day filter + paging cursor
// the chat box lives in #sayhost, built once and never replaced by a poll, so its focus/caret/draft
// are kept natively by the browser; SAY only tracks the in-flight send.
let SAY = {sending:false};
// the Settings tab: `dirty` = {KEY: new value | null (reset to default)} not yet saved. The cards are
// rebuilt only when the data changes AND nothing is unsaved, so a poll never eats an edit or focus.
let SET = {sig:null, dirty:{}, errors:{}, saving:false, filter:'', onlyMod:false, rows:{}, saved:null};

function pickWho(){
  const h = (location.hash||'').replace(/^#/,'').split('/')[0];   // '#tasks/<slug>/<id>' → tasks
  return TABS.some(t=>t.who===h) ? h : TABS[0].who;
}
function renderTabs(){
  $('tabs').innerHTML = TABS.map(t=>`<button class="tab${t.who===WHO?' on':''}" data-who="${esc(t.who)}">${esc(t.label)}</button>`).join('');
  $('tabs').querySelectorAll('.tab').forEach(b => b.onclick = () => selectTab(b.dataset.who));
}
function selectTab(who){
  if(who===WHO) return;
  if(WHO==='settings' && Object.keys(SET.dirty).length && !confirm('Leave Settings? Your unsaved changes will be lost.')) return;
  SET.dirty = {}; SET.errors = {}; SET.sig = null;
  WHO = who; CONV = {day:null, before:null};
  if(location.hash !== '#'+who) location.hash = who;
  renderTabs(); tick();
}
window.addEventListener('hashchange', () => {
  const w = pickWho();
  const picked = readTaskHash();
  if(w!==WHO){ WHO = w; CONV = {day:null, before:null}; SET.dirty = {}; SET.errors = {}; SET.sig = null; renderTabs(); tick(); }
  else if(w==='tasks' && picked && $('tkList')){ openTask(TK.sel.who, TK.sel.id, true); }
});
window.addEventListener('beforeunload', (e) => { if(Object.keys(SET.dirty).length){ e.preventDefault(); e.returnValue = ''; } });

function statCard(n, label, sub){
  return `<div class="stat"><div class="n">${esc(n)}</div><div class="l">${esc(label)}</div>${sub?`<div class="s">${esc(sub)}</div>`:''}</div>`;
}
function pill(status){ return `<span class="pill ${esc(status)}">${esc(status)}</span>`; }

// --- the household strip (every tab) ------------------------------------------------
function renderHousehold(hh){
  const h = hh.house||{};
  const chips = [];
  (hh.bodies||[]).forEach(b => chips.push(`<span class="chip"><span class="dot ${b.up?'live':'offline'}"></span><b>${esc(b.name)}</b> body ${b.up?'up':'down'}${b.up&&b.pid?` <span class="dim">pid ${esc(b.pid)}</span>`:''}</span>`));
  chips.push(`<span class="chip"><span class="dot ${h.up?'live':'offline'}"></span><b>house</b> ${h.up?'up':'down'}${h.up&&h.port?` <span class="dim">:${esc(h.port)}</span>`:''}${h.up&&h.since?` <span class="dim">· ${esc(h.since)}</span>`:''}</span>`);
  const names = hh.present_names||[];
  chips.push(`<span class="chip">connected: ${names.length?`<b>${names.map(esc).join(', ')}</b>`:'<span class="dim">nobody</span>'}</span>`);
  const mode = hh.sister_talk;
  chips.push(`<span class="chip">sister talk <span class="badge ${mode==='audible'?'audible':mode==='private'?'private':''}">${esc(mode?mode:'—')}</span></span>`);
  const t = hh.tts||{};
  const per = Object.entries(t.per||{}).map(([s,n]) => `${esc(nameOf(s))} <b>${fmtK(n)}</b>`).join(' · ');
  chips.push(`<span class="chip">TTS ${esc(t.period||'')}: ${per}${per?' · ':''}total <b>${fmtK(t.total||0)}</b>${t.cap?` / ${fmtK(t.cap)}`:''} chars</span>`);
  if(hh.house_db_present===false) chips.push(`<span class="chip dim">no house.db yet</span>`);
  $('household').innerHTML = chips.join('');
}

// --- the controls strip (every tab): the house hotkeys as buttons ---------------------
// One request in flight at a time; every button disables meanwhile, the result shows as a toast
// and the page re-polls at once. With the house down there is nobody to send to → greyed out.
let BUSY = false;
function renderControls(hh){
  const h = hh.house||{}, up = !!h.up, st = h.state||null;
  const present = hh.present||[], mode = hh.sister_talk;
  // `local` buttons launch a .bat on this PC and work with the house down (a sister's .bat brings
  // the house up with her); everything else needs the house to be listening
  const btn = (action, label, o={}) =>
    `<button class="cbtn${o.danger?' danger':''}" data-action="${esc(action)}"${o.target?` data-target="${esc(o.target)}"`:''}${o.confirm?` data-confirm="${esc(o.confirm)}"`:''}${((!up&&!o.local)||BUSY)?' disabled':''}>${esc(label)}</button>`;
  const b = [];
  if(!up) b.push(btn('start_house', 'Start house', {local:true}));
  b.push(btn('mute', st && typeof st.mic_muted==='boolean' ? (st.mic_muted?'Unmute mic':'Mute mic') : 'Mute/unmute mic'));
  b.push(btn('interrupt', 'Interrupt'));
  if(mode==='private') b.push(btn('sister_talk', 'Sister talk: PRIVATE → make audible', {target:'audible'}));
  else if(mode==='audible') b.push(btn('sister_talk', 'Sister talk: AUDIBLE → make private', {target:'private'}));
  else b.push(btn('sister_talk', 'Sister talk: toggle'));
  b.push(btn('sleep', 'Sleep')); b.push(btn('wake', 'Wake'));
  b.push('<span class="csep"></span>');
  (hh.bodies||[]).forEach(s => {
    if(present.includes(s.slug)){
      b.push(btn(s.muted?'unmute_sister':'mute_sister',
                 (s.muted?'Unmute ':'Mute ')+s.name+' voice', {target:s.slug}));
      b.push(btn('restart_sister', 'Restart '+s.name, {target:s.slug}));
      b.push(btn('quit_sister', 'Quit '+s.name, {target:s.slug, danger:true, confirm:`Quit ${s.name}? Her process exits; the house stays up.`}));
    } else {
      b.push(btn('start_sister', 'Start '+s.name, {target:s.slug, local:true}));
    }
  });
  b.push('<span class="csep"></span>');
  b.push(btn('restart_house', 'Restart house', {confirm:'Restart the house? The sisters stay up and reconnect.'}));
  b.push(btn('quit', 'Quit everything', {danger:true, confirm:'Quit the house AND every sister?'}));
  let html = `<span class="clabel">Controls</span>` + b.join('');
  if(!up) html += `<span class="cnote">House is down — Start house, or Start a sister (she brings the house up with her); the other buttons need it listening</span>`;
  const el = $('controls');
  el.className = 'controls' + (up?'':' down');
  el.innerHTML = html;
  el.querySelectorAll('.cbtn').forEach(x => x.onclick = () => pressControl(x.dataset.action, x.dataset.target||null, x.dataset.confirm||null, x.textContent));
}
async function pressControl(action, target, ask, label){
  if(BUSY) return;
  if(ask && !confirm(ask)) return;
  BUSY = true;
  document.querySelectorAll('#controls .cbtn').forEach(x => x.disabled = true);
  let res;
  try{
    const r = await fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action, target})});
    res = await r.json();
  }catch(e){ res = {ok:false, error:'dashboard server not reachable'}; }
  BUSY = false;
  toast(res && res.ok ? `${label}: ok` : `${label}: ${(res&&res.error)||'failed'}`, !!(res && res.ok));
  tick();
}
let TOAST_T = null;
function toast(text, ok){
  const t = $('toast'); t.textContent = text; t.className = 'toast show ' + (ok?'ok':'bad');
  clearTimeout(TOAST_T); TOAST_T = setTimeout(() => { t.className = 'toast'; }, 2800);
}

// --- dispatch -----------------------------------------------------------------------
function render(d){
  renderHousehold(d.household||{});
  renderControls(d.household||{});
  if(d.who==='conversations'){ SET.sig = null; renderConversations(d); }
  else if(d.who==='tasks'){ SET.sig = null; clearSayBox(); renderTasks(d); }
  else if(d.who==='settings'){ clearSayBox(); renderSettings(d); }
  else { SET.sig = null; clearSayBox(); renderCharacter(d); }
  $('stamp').textContent = 'updated ' + new Date().toLocaleTimeString();
}

// --- a character's core --------------------------------------------------------------
function renderCharacter(d){
  const name = d.name || nameOf(d.who) || 'Aiva';
  $('title').textContent = name;
  // status line
  const p = d.presence || {};
  const dot = $('dot'), st = $('statusText');
  dot.className = 'dot ' + (p.state==='awake'?'live':p.state==='asleep'?'asleep':p.state==='offline'?'offline':'');
  const bits = [];
  if(p.state) bits.push(p.state);
  if(p.avatar_running) bits.push('body up (pid '+esc(p.avatar_pid)+')'); else bits.push('body down');
  if(p.last_activity) bits.push('active '+esc(p.last_activity));
  st.textContent = bits.join(' · ');
  $('subtitle').textContent = 'v' + (d.version||'?') + ' · ' + name + "'s core" + (d.self_forming?' · self-forming':'') + ' · ' + (d.capabilities?`${d.capabilities.tool_count} tools · ${d.capabilities.module_count} modules`:'');

  $('banner').innerHTML = d.error ? `<div class="banner">${esc(d.error)}</div>`
    : (d.started===false ? `<div class="banner soft">${esc(d.note||(name+" hasn't started yet."))}</div>` : '');

  // headline stats
  const h = d.headline||{}, r = d.reading||{};
  const mood = (d.inner&&d.inner.mood)||null;
  $('stats').innerHTML = [
    `<div class="stat"><div class="n" style="font-size:19px;line-height:1.25">${esc(mood?(mood.mood_word||'—'):'—')}</div><div class="l">Mood</div><div class="s">${esc(mood?(mood.focus?('on: '+mood.focus):'resting'):'not live yet')}</div></div>`,
    statCard(h.traits??0,'Traits', (h.temperament_core?`${h.temperament_core} core · `:'')+(h.voice_notes?`+${h.voice_notes} voice`:'')+(h.lessons?` · ${h.lessons} lessons`:'')),
    statCard(`${h.books_finished??0}/${h.books_total??0}`,'Books', `${h.sections_read??0} sections studied`),
    statCard(h.tasks??0,'Tasks', h.flows?`${h.flows} flows`:'no flows yet'),
    statCard(h.facts??0,'Facts', 'remembered'),
    statCard(h.sessions??0,'Sessions', `${h.messages??0} messages`),
    statCard('$'+(r.total_cost??0).toFixed(2),'Reading spend', 'lifetime'),
  ].join('');

  let html = '';

  // presence + cat
  html += d.has_cat ? `<div class="cols">` : '';
  html += `<section><h2>Presence</h2>`;
  html += rowKV('State', p.state||'unknown');
  html += rowKV('Avatar body', p.avatar_running?`running (pid ${esc(p.avatar_pid)})`:'down');
  html += rowKV('Monitor / layer', `${p.monitor??'?'} · ${esc(p.layer||'?')}`);
  html += rowKV('Last activity', p.last_activity||'—');
  html += rowKV('Reading permission', (d.becoming&&d.becoming.permission)||(r.permission)||'—');
  html += rowKV('Core', d.db_path ? d.db_path.split(/[\\/]/).pop() : '—');
  html += `</section>`;
  if(d.has_cat){
    if(d.cat){
      const c = d.cat;
      html += `<section><h2>Cat <span class="cnt">${esc(c.name||'')}${c.dismissed?' · dismissed':''}</span></h2>`;
      html += `<div class="vitals">
        ${vital('Hunger',c.hunger)}${vital('Mood',c.mood)}${vital('Energy',c.energy)}</div>`;
      html += `<div style="margin-top:12px">`+rowKV('Total pets', c.total_pets)+rowKV('Last fed', c.last_fed||'—')+rowKV('Last seen', c.last_seen||'—')+`</div>`;
      html += `</section>`;
    } else {
      html += `<section><h2>Cat</h2><div class="empty">No cat state yet.</div></section>`;
    }
    html += `</div>`;
  }

  // statistics — operations profiler (where her time goes)
  const sx = d.stats;
  html += `<section><h2>Statistics <span class="cnt">operations profile · last ${sx?sx.days:7}d</span></h2>`;
  if(sx && sx.ops && sx.ops.length){
    if(sx.turn){
      html += `<div class="d" style="margin-bottom:12px">Response latency (you stop → she speaks): <b style="color:var(--accent2)">${(sx.turn.mean_ms/1000).toFixed(2)}s</b> mean · ${(sx.turn.max_ms/1000).toFixed(2)}s max · ${sx.turn.n} turns</div>`;
    }
    html += `<div style="overflow-x:auto"><table class="stat-table"><thead><tr><th>operation</th><th>n</th><th>mean</th><th>max</th><th>total</th><th>share</th><th>cost</th></tr></thead><tbody>`;
    sx.ops.forEach(o=>{
      html += `<tr><td>${esc(o.label)}${o.fails?` <span style="color:var(--bad)">·${o.fails}✗</span>`:''}</td><td>${o.n}</td><td>${fmtms(o.mean_ms)}</td><td>${fmtms(o.max_ms)}</td><td>${fmtms(o.total_ms)}</td><td><span class="sharebar"><i style="width:${o.share}%"></i></span>${o.share}%</td><td>${o.cost?('$'+o.cost.toFixed(3)):'—'}</td></tr>`;
    });
    html += `</tbody></table></div>`;
    if(sx.cost){
      html += `<div class="dim" style="font-size:12px;margin-top:10px">Cost last ${sx.days}d — side-calls $${sx.cost.side.toFixed(3)} · reading $${sx.cost.reading.toFixed(3)} (lifetime) · tasks $${sx.cost.task.toFixed(3)} (lifetime). Voice-model cost is in her session usage summary.</div>`;
    }
  } else {
    html += `<div class="empty">No operations recorded yet — this fills as she thinks, uses tools and reads (after her next restart).</div>`;
  }
  html += `</section>`;

  // nervous system — inner state + experience stream (Phase 1 substrate)
  const inn = d.inner||{};
  html += `<section><h2>Nervous system <span class="cnt">mood &amp; experience</span></h2>`;
  if(inn.mood){
    const mo = inn.mood;
    html += `<div class="d" style="margin-bottom:12px">Right now she feels <b style="color:var(--accent2)">${esc(mo.mood_word||'—')}</b>${mo.focus?` · preoccupied with “${esc(mo.focus)}”`:''}${mo.last_update?` · <span class="dim">settled ${esc((mo.last_update||'').slice(11,19))}</span>`:''}</div>`;
    html += `<div class="vitals">${dial('Valence',mo.valence,true)}${dial('Arousal',mo.arousal,false)}${dial('Social',mo.social,true)}</div>`;
    if(mo.needs){
      const nd = mo.needs;
      html += `<div class="d" style="margin:14px 0 6px">What she needs <span class="dim">(0 starved · 1 met)</span>${mo.tension?` · <b style="color:var(--warn)">${esc(mo.tension)}</b>`:(mo.pressing&&mo.pressing.length?` · <b style="color:var(--warn)">pressing: ${esc(mo.pressing.join(', '))}</b>`:'')}</div>`;
      html += `<div class="vitals">${dial('Connection',nd.connection,false)}${dial('Novelty',nd.novelty,false)}${dial('Competence',nd.competence,false)}${dial('Rest',nd.rest,false)}</div>`;
    }
    if(inn.initiative){
      const ini = inn.initiative;
      html += `<div class="dim" style="font-size:12px;margin-top:10px">Initiative: <b style="color:var(--muted)">${esc(ini.permission)}</b> · ${ini.today} today${ini.last?` · last ${esc(ini.last)}`:''} · landed ${ini.welcomed||0} · let drop ${ini.ignored||0}${ini.last_talk?` · last talked with him ${esc(ini.last_talk)}`:''}</div>`;
    }
  } else {
    html += `<div class="empty">No inner state yet — the dials light up once she has run and settled a mood.</div>`;
  }
  if(inn.events && inn.events.length){
    html += `<h2 style="margin-top:18px">Experience <span class="cnt">${inn.events_total} events</span></h2><div>`;
    inn.events.forEach(e=>{
      const col = e.valence_hint>0.05?'var(--good)':e.valence_hint<-0.05?'var(--bad)':'var(--muted)';
      const sg = (n)=>((n>=0?'+':'')+(n||0).toFixed(2));
      html += `<div class="item"><div class="t" style="font-weight:500"><span style="color:${col}">●</span> ${esc(e.summary||e.kind)}</div><div class="d">${esc(e.source)}/${esc(e.kind)} · v${sg(e.valence_hint)} a${sg(e.arousal_hint)} s${sg(e.social_hint)} · sal ${(e.salience||0).toFixed(2)} · ${esc((e.ts||'').slice(11,19))}</div></div>`;
    });
    html += `</div>`;
  } else {
    html += `<div class="empty" style="margin-top:14px">No experience logged yet — reading, conversation and time-of-day feed this stream; it fills as she lives.</div>`;
  }
  html += `<h2 style="margin-top:18px">Diary <span class="cnt">private to her</span></h2>`;
  if(inn.journal && inn.journal.length){
    inn.journal.forEach(j=>{
      html += `<div class="item"><div class="d" style="color:var(--ink)">“${esc(j.entry)}”</div><div class="dim" style="font-size:12px;margin-top:4px">${esc(j.when||'')}${j.focus?` · on “${esc(j.focus)}”`:''}${j.carry?` · to carry: ${esc(j.carry)}`:''}</div></div>`;
    });
  } else {
    html += `<div class="empty">No diary yet — she writes to herself once the self loop runs.</div>`;
  }
  html += `<h2 style="margin-top:18px">Wishes <span class="cnt">${inn.desires_total||0} open</span></h2>`;
  if(inn.desires && inn.desires.length){
    inn.desires.forEach(dd=>{
      html += `<div class="item"><div class="t" style="font-weight:500">${esc(dd.text)}</div><div class="d">${esc(dd.kind)} · ${strength(dd.strength)} ${dd.strength}${dd.source?` · from ${esc(dd.source)}`:''}</div></div>`;
    });
  } else {
    html += `<div class="empty">No wishes yet — they form from her diary and conversation.</div>`;
  }
  html += `</section>`;

  // v3 — her life: story, habits from her nights, moments, projects, his rhythm
  const lf = inn.life||{};
  html += `<section><h2>Her life <span class="cnt">story · habits · projects</span></h2>`;
  if(lf.story){
    html += `<div class="item"><div class="t" style="font-weight:500">${lf.story.chapter?`“${esc(lf.story.chapter)}”`:'Her story so far'}</div><div class="d" style="color:var(--ink)">${esc(lf.story.text)}</div>${lf.story.turning_point?`<div class="dim" style="font-size:12px;margin-top:4px">turning point: ${esc(lf.story.turning_point)}</div>`:''}</div>`;
  } else {
    html += `<div class="empty">No story yet — she writes it once she has a few diary entries, then revises it about weekly.</div>`;
  }
  html += `<h2 style="margin-top:18px">Habits <span class="cnt">formed in her sleep</span></h2>`;
  if(lf.habits && lf.habits.length){
    lf.habits.forEach(h=>{
      html += `<div class="item"><div class="t" style="font-weight:500">${esc(h.text)}</div><div class="d">${strength(h.strength)} ${h.strength}${h.evidence?` · from “${esc(h.evidence)}”`:''}${h.when?` · reinforced ${esc(h.when)}`:''}</div></div>`;
    });
  } else {
    html += `<div class="empty">No habits yet — her nights turn what landed with him into habits.</div>`;
  }
  if(lf.last_dream){
    const ld = lf.last_dream;
    html += `<div class="dim" style="font-size:12px;margin-top:6px">Last dream ${esc(ld.when||'')}: ${ld.moments||0} moment(s), ${ld.habits||0} habit change(s), forgot ${ld.forgotten||0} old event(s)</div>`;
  }
  if(lf.moments && lf.moments.length){
    html += `<h2 style="margin-top:18px">Moments <span class="cnt">replayed at night</span></h2>`;
    lf.moments.forEach(m=>{
      const col = m.label==='landed'?'var(--good)':'var(--bad)';
      html += `<div class="item"><div class="d"><span style="color:${col}">●</span> him: “${esc(m.his)}”<br>her: “${esc(m.hers)}”${m.better?`<br><span class="dim">now she'd say: “${esc(m.better)}”</span>`:''}</div><div class="dim" style="font-size:12px;margin-top:4px">${esc(m.label)}${m.why?` · ${esc(m.why)}`:''} · ${esc(m.when||'')}</div></div>`;
    });
  }
  html += `<h2 style="margin-top:18px">Projects</h2>`;
  if(lf.projects && lf.projects.length){
    lf.projects.forEach(pr=>{
      html += `<div class="item"><div class="t" style="font-weight:500">${esc(pr.title)}</div><div class="d">${esc(pr.kind)} · ${esc(pr.status)} · ${pr.steps}/${pr.target}${pr.when?` · last step ${esc(pr.when)}`:''}</div></div>`;
    });
  } else {
    html += `<div class="empty">No projects yet — a wish that keeps coming back and that she can pursue alone becomes one.</div>`;
  }
  if(lf.rhythm){
    html += `<div class="dim" style="font-size:12px;margin-top:10px">His rhythm: ${lf.rhythm.learned} hour-slots learned${lf.rhythm.usual&&lf.rhythm.usual.length?` · usually around ${esc(lf.rhythm.usual.slice(0,8).join(', '))}`:' · nothing she can count on yet (needs ~3 weeks)'}</div>`;
  }
  html += `</section>`;

  // who she is so far (self-formed temperament) + her sister (kin observations)
  const tp = d.temperament||{};
  const kin = d.kin||{};
  html += `<div class="cols">`;
  const tpTraits = tp.traits||[];
  const tpCnt = tp.available===false ? 'not tracked yet' : `${tp.core_count||0} core · ${Math.max(0, tpTraits.length-(tp.core_count||0))} forming`;
  html += `<section><h2>Who she is so far <span class="cnt">${tpCnt}</span></h2>`;
  if(tp.forming){
    html += d.self_forming
      ? `<div class="d" style="margin-bottom:10px;color:var(--accent2)">She's new — still finding out who she is, and that's fine. Nobody wrote her temperament; it forms from what she lives through, and a trait counts as core once it reaches ${TEMP_CORE}.</div>`
      : `<div class="d" style="margin-bottom:10px">Her temperament is written into her persona floor; anything listed here is what experience has added on top.</div>`;
  }
  if(tpTraits.length){
    tpTraits.forEach(t => {
      html += `<div class="item"><div class="t">${t.core?'<span class="tag on">core</span> ':''}${esc(t.text)}</div><div class="d">${strength(t.strength)} ${t.strength}${t.source?` · from ${esc(t.source)}`:''}${t.adopted?` · since ${esc(t.adopted)}`:''}${t.reinforced?` · last ${esc(t.reinforced)}`:''}</div></div>`;
    });
    if(tp.dropped) html += `<div class="dim" style="font-size:12px;margin-top:8px">${tp.dropped} let go</div>`;
  } else {
    html += `<div class="empty">${d.self_forming?'Nothing has taken shape yet — it comes from her diary, a little at a time.':'Nothing formed from experience yet.'}</div>`;
  }
  html += `</section>`;
  const kinCnt = kin.available===false ? 'nothing observed yet' : `${kin.total||0} observations${kin.dropped?` · ${kin.dropped} dropped`:''}`;
  html += `<section><h2>Her sister <span class="cnt">${kinCnt}</span></h2>`;
  if((kin.items||[]).length){
    kin.items.forEach(k => {
      html += `<div class="item"><div class="t">${esc(k.summary)}</div><div class="d">${esc(k.about_name)} · ${esc(k.kind)}${k.capability?` · <span class="tag">${esc(k.capability)}</span>`:''} · seen ${k.count}× · ${strength(k.strength)} ${k.strength}${k.last?` · last ${esc(k.last)}`:''}</div>${k.last_example?`<div class="d dim">e.g. “${esc(k.last_example)}”</div>`:''}</div>`;
    });
  } else {
    html += `<div class="empty">She hasn't seen her sister do anything yet — this fills from what she observes and hears, never from guesswork.</div>`;
  }
  html += `</section>`;
  html += `</div>`;

  // preferences — her mutable likes / dislikes / opinions
  const pf = d.preferences;
  const pfCnt = pf ? `${pf.active_total||0} active${pf.dropped?` · ${pf.dropped} dropped`:''}`
                   : 'likes · dislikes · opinions';
  html += `<section><h2>Where she stands <span class="cnt">${pfCnt}</span></h2>`;
  if(pf && ((pf.likes||[]).length || (pf.dislikes||[]).length || (pf.opinions||[]).length)){
    const prow = (p)=>{
      const col = p.valence>0?'var(--good)':p.valence<0?'var(--bad)':'var(--muted)';
      const w = Math.round(Math.abs(p.valence)*100);
      return `<div class="item"><div class="t">${esc(p.stance)}</div>`
        + `<div style="height:5px;border-radius:3px;background:var(--panel2);margin:5px 0;overflow:hidden"><span style="display:block;height:100%;width:${w}%;background:${col}"></span></div>`
        + `<div class="d">${(p.valence>=0?'+':'')+p.valence} · ${p.evidence}× · ${esc(p.origin)}</div></div>`;
    };
    const grp = (label,arr)=>{ if(arr && arr.length){ html += `<div class="dim" style="font-size:12px;margin-top:12px;text-transform:uppercase;letter-spacing:.5px">${label}</div>`; arr.forEach(p=>html+=prow(p)); } };
    grp('Likes', pf.likes); grp('Dislikes', pf.dislikes); grp('Opinions', pf.opinions);
  } else {
    html += `<div class="empty">No preferences yet — ${d.self_forming?'she forms hers from what she reads, does and talks about.':'her baseline tastes seed the first time you fully restart her (Ctrl+Alt+X), then they genuinely evolve from what she reads, does and talks about.'}</div>`;
  }
  html += `</section>`;

  // her space — things she gathered from the web
  const sp = d.space;
  const spCnt = sp ? `${sp.kept||0} kept${sp.dropped?` · ${sp.dropped} dropped`:''}` : 'web finds';
  html += `<section><h2>Her space <span class="cnt">${spCnt}</span></h2>`;
  if(sp && (sp.items||[]).length){
    sp.items.forEach(it=>{
      const src = it.source_url ? `<a href="${esc(it.source_url)}" target="_blank" rel="noreferrer" style="color:var(--muted)">source</a>` : '';
      const file = it.has_file ? 'saved file' : 'link';
      const tags = (it.tags||[]).length ? ' · '+it.tags.map(t=>`<span class="tag">${esc(t)}</span>`).join('') : '';
      html += `<div class="item"><div class="t">${pill(it.kind)} ${esc(it.subject)}</div>`
        + (it.why?`<div class="d" style="color:var(--ink)">“${esc(it.why)}”</div>`:'')
        + `<div class="d">${file} · ${esc(it.when||'')}${src?' · '+src:''}${tags}</div></div>`;
    });
  } else {
    html += `<div class="empty">Nothing gathered yet — she keeps things here when she finds something online she likes. Set TAVILY_API_KEY and restart to switch her web reach on.</div>`;
  }
  html += `</section>`;

  // limits & what's paused — per-service budgets + why anything self-started stopped
  const lim = d.limits;
  html += `<section><h2>Limits & what's paused <span class="cnt">per-service budgets${lim&&lim.source==='house'?' · shared house.db':lim?' · her own db':''}</span></h2>`;
  if(lim && (lim.paused||[]).length){
    lim.paused.forEach(p=>{
      html += `<div class="item" style="border-left:3px solid var(--bad);padding-left:10px"><div class="t" style="color:var(--bad)">⏸ ${esc(p.label)} paused</div><div class="d">${esc(p.reason)}${p.resets?` · resets ${esc(p.resets)}`:''}</div></div>`;
    });
  } else {
    html += `<div class="d" style="color:var(--good)">Nothing paused — everything's within budget.</div>`;
  }
  if(lim && (lim.services||[]).length){
    html += `<div style="margin-top:10px">`;
    lim.services.forEach(s=>{
      const col = s.paused?'var(--bad)':s.pct>=80?'var(--warn,#d98a00)':'var(--good)';
      html += `<div class="item"><div class="t" style="display:flex;justify-content:space-between"><span>${esc(s.label)}</span><span class="dim">${s.used}/${s.cap} this month</span></div>`
        + `<div style="height:5px;border-radius:3px;background:var(--panel2);margin:5px 0;overflow:hidden"><span style="display:block;height:100%;width:${s.pct}%;background:${col}"></span></div></div>`;
    });
    html += `</div>`;
  }
  if(lim){
    html += `<div class="d" style="margin-top:8px;color:var(--dim)">Per-item caps: books $${(lim.book_cost||0).toFixed(2)} total · tasks $${(lim.task_cost||0).toFixed(2)} total. No global budget — each service is capped at its own free tier, and only self-started work pauses (her voice never stops).</div>`;
  } else {
    html += `<div class="empty">Limits come online after a restart.</div>`;
  }
  html += `</section>`;

  // reading
  html += `<section><h2>Reading<span class="cnt">${(r.books||[]).length} in library</span></h2>`;
  if(r.current){
    const b = r.current;
    html += `<div class="book"><div class="head"><div class="t" style="font-size:16px;font-weight:700">${esc(b.title)} <span class="dim">— ${esc(b.author)}</span></div>${pill(b.status)}</div>`;
    html += `<div class="d">now on “${esc(b.section_title||'?')}” · section ${b.cursor}/${b.sections}${b.last_feeling?` · feeling ${esc(b.last_feeling)}`:''}</div>`;
    html += bar(b.pct);
    html += `<div class="d" style="margin-top:6px">${b.pct}% · ${b.notes} notes · ${b.sessions} sessions · $${b.cost_usd.toFixed(3)} · last read ${esc(b.last_read||'—')}</div>`;
    if(b.rolling) html += `<div class="d" style="margin-top:8px;color:var(--dim)">“${esc(b.rolling)}${b.rolling.length>=400?'…':''}”</div>`;
    html += `</div>`;
  } else {
    html += `<div class="empty">Not reading anything right now.</div>`;
  }
  // other books
  const others = (r.books||[]).filter(b => !(r.current && b.id===r.current.id));
  if(others.length){
    html += `<div style="margin-top:6px">`;
    others.forEach(b => {
      html += `<div class="item"><div class="head" style="display:flex;justify-content:space-between"><span class="t">${esc(b.title)} <span class="dim">— ${esc(b.author)}</span></span>${pill(b.status)}</div>`;
      html += `<div class="d">${b.cursor}/${b.sections} sections · ${b.notes} notes · $${b.cost_usd.toFixed(3)}${b.last_read?` · ${esc(b.last_read)}`:''}${b.status_reason?` · ${esc(b.status_reason)}`:''}</div></div>`;
    });
    html += `</div>`;
  }
  if(r.wishlist && r.wishlist.length){
    html += `<div style="margin-top:12px" class="muted"><b>Wishlist:</b> `;
    html += r.wishlist.map(w => `<span class="tag">${esc(w.title)}${w.author?' · '+esc(w.author):''}${w.fulfilled?' ✓':''}</span>`).join('');
    html += `</div>`;
  }
  html += `</section>`;

  // becoming
  const bc = d.becoming||{};
  html += `<div class="cols">`;
  html += `<section><h2>Who she's becoming <span class="cnt">traits & values from reading</span></h2>`;
  if(bc.traits && bc.traits.length){
    bc.traits.forEach(t => {
      html += `<div class="item"><div class="t">${esc(t.text)}</div><div class="d">${esc(t.kind)}${t.book?` · from ${esc(t.book)}`:''} · ${strength(t.strength)} ${t.strength}${t.status!=='active'?` · <span class="dim">${esc(t.status)}</span>`:''}${t.veto_reason?` · vetoed: ${esc(t.veto_reason)}`:''}</div></div>`;
    });
  } else html += `<div class="empty">No traits yet — she grows these by finishing books.</div>`;
  if(bc.lessons && bc.lessons.length){
    html += `<h2 style="margin-top:16px">Lessons she applies</h2>`;
    bc.lessons.forEach(le => html += `<div class="item"><div class="t">${esc(le.lesson)}</div><div class="d">${esc(le.kind)} · applied ${le.applied_count}× · ${strength(le.strength)}</div></div>`);
  }
  html += `</section>`;
  html += `<section><h2>Her voice <span class="cnt">turns of phrase</span></h2>`;
  if(bc.voice && bc.voice.length){
    bc.voice.forEach(v => html += `<div class="item"><div class="t">${esc(v.note)} <span class="dim">(${esc(v.kind)})</span></div>${v.example?`<div class="d">e.g. “${esc(v.example)}”</div>`:''}</div>`);
  } else html += `<div class="empty">No borrowed voice yet — comes from reading.</div>`;
  html += `</section>`;
  html += `</div>`;

  // tasks & flows
  const tk = d.tasks||{};
  html += `<section><h2>Tasks & competence <span class="cnt">${(tk.tally&&tk.tally.total)||0} tasks · ${(tk.flows||[]).length} flows</span></h2>`;
  html += `<div class="cols"><div>`;
  html += `<div class="muted" style="margin-bottom:6px">Recent tasks <a href="#tasks" style="font-size:12px">all on the Tasks tab →</a></div>`;
  if(tk.recent && tk.recent.length){
    tk.recent.forEach(t => {
      html += `<a class="tlink" href="#tasks/${esc(d.who)}/${esc(t.full_id||'')}" title="open on the Tasks tab"><div class="item"><div class="t">${esc(t.goal_text)}</div><div class="d">${esc(t.status)}${t.outcome?` · ${esc(t.outcome)}`:''}${t.delegated_to&&t.delegated_to!=='none'?` · ↦ ${esc(t.delegated_to)}`:''}${t.agent_grade?` · grade ${esc(t.agent_grade)}`:''}${t.corrected?` · <span style="color:var(--warn)">corrected</span>`:''} · ${esc(t.created||'')}</div></div></a>`;
    });
  } else html += `<div class="empty">No tasks run yet.</div>`;
  html += `</div><div>`;
  html += `<div class="muted" style="margin-bottom:6px">Flows she's saved</div>`;
  if(tk.flows && tk.flows.length){
    tk.flows.forEach(f => html += `<div class="item"><div class="t">${esc(f.name)} ${f.confirmed_by_user?'<span class="tag on">yours</span>':''}</div><div class="d">${f.run_count}× run · ${f.success_count} ok${f.family?` · ${esc(f.family)}`:''}${f.last_run?` · ${esc(f.last_run)}`:''}</div></div>`);
  } else html += `<div class="empty">No flows yet — she distills these from repeated tasks.</div>`;
  if(tk.prefs && tk.prefs.length){
    html += `<div class="muted" style="margin:14px 0 6px">Learned approach preferences</div>`;
    tk.prefs.forEach(pr => html += `<div class="item"><div class="t">${esc(pr.family)} → ${esc(pr.chosen_approach)}</div>${pr.user_reason?`<div class="d">${esc(pr.user_reason)}</div>`:''}</div>`);
  }
  html += `</div></div></section>`;

  // memory
  const m = d.memory||{};
  html += `<div class="cols">`;
  html += `<section><h2>Memory <span class="cnt">${m.facts_total||0} facts · ${m.messages||0} messages</span></h2>`;
  html += `<div class="chips">`;
  Object.entries(m.facts_by_cat||{}).forEach(([k,v]) => html += `<span class="tag">${esc(k)} · ${v}</span>`);
  html += `</div>`;
  if(m.facts_recent && m.facts_recent.length){
    html += `<div style="margin-top:8px">`;
    m.facts_recent.forEach(f => html += `<div class="item"><div class="t" style="font-weight:500">${esc(f.fact)}</div><div class="d">${esc(f.category)} · ${esc(f.when||'')}</div></div>`);
    html += `</div>`;
  }
  html += `</section>`;
  html += `<section><h2>Recent recaps <span class="cnt">${m.recaps_total||0} sessions logged</span></h2>`;
  if(m.recaps_recent && m.recaps_recent.length){
    m.recaps_recent.forEach(rc => html += `<div class="item"><div class="d">${esc(rc.recap)}${rc.recap && rc.recap.length>=280?'…':''}</div><div class="dim" style="font-size:12px;margin-top:3px">${esc(rc.when||'')}</div></div>`);
  } else html += `<div class="empty">No recaps yet.</div>`;
  html += `</section>`;
  html += `</div>`;

  // capabilities
  const cap = d.capabilities||{};
  html += `<section><h2>Capabilities <span class="cnt">${cap.tool_count||0} tools</span></h2>`;
  (cap.tool_groups||[]).forEach(g => {
    html += `<div class="item"><div class="t">${esc(g.label)} <span class="dim">(${g.tools.length})</span></div><div class="chips" style="margin-top:6px">`;
    g.tools.forEach(t => html += `<span class="tag">${esc(t)}</span>`);
    html += `</div></div>`;
  });
  html += `</section>`;

  // parts / architecture
  html += `<section><h2>Her parts <span class="cnt">${cap.module_count||0} modules</span></h2>`;
  (cap.subsystems||[]).forEach(s => {
    html += `<div class="item"><div class="t">${esc(s.label)} <span class="dim">· ${s.lines} lines</span></div><div class="mods" style="margin-top:8px">`;
    s.modules.forEach(mm => html += `<span class="mod"><b>${esc(mm.file)}</b> <span class="ln">${mm.lines}</span></span>`);
    html += `</div></div>`;
  });
  if(cap.retired && cap.retired.length){
    html += `<div class="item"><div class="dim">Retired: ${cap.retired.map(x=>esc(x.file)+' — '+esc(x.note)).join(', ')}</div></div>`;
  }
  html += `</section>`;

  // live log
  if(p.recent_log && p.recent_log.length){
    html += `<section><h2>Recent activity <span class="cnt">today's log</span></h2><div class="log">${p.recent_log.map(esc).join('\n')}</div></section>`;
  }

  $('content').innerHTML = html;
}

// --- the house transcript ------------------------------------------------------------
function renderConversations(d){
  $('title').textContent = 'Conversations';
  const hh = d.household||{}, h = hh.house||{};
  $('dot').className = 'dot ' + (h.up?'live':'offline');
  $('statusText').textContent = (h.up?'house up':'house down') + (h.up&&h.port?` · :${h.port}`:'') + ((hh.present_names||[]).length?` · ${hh.present_names.join(' & ')} connected`:'');
  $('subtitle').textContent = 'v' + (d.version||'?') + " · everything said in the house, newest first · private sister lines included";
  const conv = d.conversations||{};
  $('banner').innerHTML = d.error ? `<div class="banner">${esc(d.error)}</div>`
    : (conv.available===false ? `<div class="banner soft">No house transcript yet — it appears once the house has run.</div>` : '');
  const ev = d.house_events||[];
  const mode = hh.sister_talk;
  $('stats').innerHTML = [
    statCard(conv.total??0, conv.day?('Lines on '+conv.day):'Lines logged', conv.lines?`${conv.lines.length} shown`:''),
    statCard((conv.lines||[]).filter(l=>!l.audible).length,'Private shown','text-only sister talk'),
    statCard(ev.length,'House events','last 30'),
    statCard(mode?mode:'—','Sister talk', mode==='private'?'sisters talk text-only':mode==='audible'?'you hear them':'house not running'),
  ].join('');

  let html = `<section><h2>Transcript <span class="cnt">${conv.day?esc(conv.day):'all days'}${conv.before?' · older than #'+esc(conv.before):''}</span></h2>`;
  const up = !!h.up;
  renderSayBox(up);   // lives in its own persistent host (#sayhost), so a poll never steals focus mid-type
  html += `<div class="toolbar"><label>Day <select id="daySel"><option value="">all</option>${(conv.days||[]).map(dd=>`<option value="${esc(dd)}"${dd===conv.day?' selected':''}>${esc(dd)}</option>`).join('')}</select></label>`;
  html += `<button id="newestBtn"${conv.before?'':' disabled'}>newest</button><button id="olderBtn"${conv.has_more?'':' disabled'}>older</button>`;
  html += `<span class="dim">${conv.before?'paused on an older page':'live'}</span></div>`;
  if((conv.lines||[]).length){
    let lastDay = null;
    conv.lines.forEach(l => {
      if(!conv.day && l.day !== lastDay){ lastDay = l.day; html += `<div class="dim" style="font-size:12px;margin:12px 0 2px;text-transform:uppercase;letter-spacing:.5px">${esc(l.day)}</div>`; }
      const spk = l.speaker==='user' ? 'you' : esc(l.speaker);
      html += `<div class="line${l.audible?'':' priv'}" title="#${esc(l.id)}${l.turn_id!=null?' · turn '+esc(l.turn_id):''}"><span class="time">${esc(l.time)}</span><span class="spk ${spk}">${esc(l.speaker_name)}</span><span class="arrow">→ ${esc(l.addressee_name)}</span><span class="txt">${esc(l.text)}</span><span class="marks"><span class="badge ${esc(l.badge)}">${esc(l.badge)}</span>${l.interrupted?'<span class="badge cut">cut off</span>':''}${l.kind?`<span class="dim">${esc(l.kind)}</span>`:''}</span></div>`;
    });
  } else {
    html += `<div class="empty">${conv.available===false?'Nothing yet.':(conv.day?'Nothing said that day.':'Nothing said yet.')}</div>`;
  }
  html += `</section>`;

  html += `<section><h2>House <span class="cnt">what the sisters did · last ${ev.length} events</span></h2>`;
  if(ev.length){
    ev.forEach(e => html += `<div class="ev"><span class="time">${esc(e.time)}</span><span class="spk ${esc(e.slug)}" style="font-weight:600">${esc(e.name)}</span><span><span class="tag">${esc(e.kind)}</span> ${esc(e.summary)}${e.capability?` <span class="dim">· ${esc(e.capability)}</span>`:''}${e.relayed_to?` <span class="dim">· told ${esc(e.relayed_to)}</span>`:''}</span></div>`);
  } else {
    html += `<div class="empty">No house events yet — they appear as a sister does, makes, reads or says something worth telling the other.</div>`;
  }
  html += `</section>`;

  $('content').innerHTML = html;
  const sel = $('daySel'); if(sel) sel.onchange = (e) => { CONV.day = e.target.value || null; CONV.before = null; tick(); };
  const nb = $('newestBtn'); if(nb) nb.onclick = () => { CONV.before = null; tick(); };
  const ob = $('olderBtn'); if(ob) ob.onclick = () => { if(conv.next_before){ CONV.before = conv.next_before; tick(); } };
}

// --- the Settings tab ------------------------------------------------------------------
// Every knob in .env, grouped. Edits stay local (SET.dirty) until Save posts ONLY the changed rows to
// /api/settings; config is read at boot, so the reply names what to restart (offered as buttons).
// The cards are rebuilt only when the data changed and nothing is unsaved — a poll never eats an edit.
const hasOwn = (o,k) => Object.prototype.hasOwnProperty.call(o,k);
function fmtSetVal(s, v){
  if(v===null || v===undefined || v==='') return '';
  if(s.type==='bool') return v ? 'on' : 'off';
  if(Array.isArray(v)) return v.join('–');
  return String(v);
}
function restartNames(list, procs){ return (list||[]).map(p => (procs&&procs[p]) || p); }
function renderSettings(d){
  $('title').textContent = 'Settings';
  const hh = d.household||{}, h = hh.house||{};
  SET.hh = hh; SET.procs = d.processes||{};
  $('dot').className = 'dot ' + (h.up?'live':'offline');
  $('statusText').textContent = d.env_present ? ('.env · ' + (d.env_path||'')) : 'no .env yet — Save creates it';
  $('subtitle').textContent = 'v' + (d.version||'?') + ' · every knob the house and the sisters read · read at boot, so a change applies after a restart';
  const sm = d.settings_summary||{};
  $('banner').innerHTML = d.error ? `<div class="banner">${esc(d.error)}</div>`
    : ((sm.invalid||[]).length ? `<div class="banner">These .env values don't parse, so the code falls back to its default: ${sm.invalid.map(esc).join(', ')}</div>` : '');
  $('stats').innerHTML = [
    statCard(sm.total??0, 'Settings', `${sm.editable??0} editable here`),
    statCard(sm.in_env??0, 'Set in .env', 'the rest use the code default'),
    statCard(sm.modified??0, 'Changed', 'from the default'),
    statCard(`${sm.secrets_set??0}/${sm.secrets??0}`, 'API keys set', 'masked · edit .env by hand'),
  ].join('');
  const groups = d.settings||[];
  const sig = JSON.stringify(groups);
  if($('setbar') && SET.sig === sig){ refreshSetBar(); return; }
  if($('setbar') && SET.sig !== null && Object.keys(SET.dirty).length){
    refreshSetBar('.env changed on disk while you were editing — Save writes only the rows you changed');
    return;
  }
  SET.sig = sig; SET.rows = {};
  groups.forEach(g => g.settings.forEach(s => { SET.rows[s.key] = s; }));
  let html = `<div class="setbar" id="setbar"><input id="setFilter" type="search" placeholder="Filter — a key or a few words" value="${esc(SET.filter)}" autocomplete="off">`
    + `<label><input type="checkbox" id="setOnlyMod"${SET.onlyMod?' checked':''}> changed only</label><span id="setCount" class="dim"></span>`
    + `<button class="cbtn" id="setDiscard">Discard</button><button class="cbtn primary" id="setSave">Save</button><div class="setsaved" id="setSaved"></div></div>`;
  groups.forEach(g => {
    html += `<section class="sgroup"><h2>${esc(g.group)} <span class="cnt">${g.settings.length}</span></h2>`;
    g.settings.forEach(s => { html += settingRow(s); });
    html += `</section>`;
  });
  $('content').innerHTML = html;
  $('setFilter').oninput = (e) => { SET.filter = e.target.value; applySetFilter(); };
  $('setOnlyMod').onchange = (e) => { SET.onlyMod = e.target.checked; applySetFilter(); };
  $('setSave').onclick = () => saveSettings();
  $('setDiscard').onclick = () => { SET.dirty = {}; SET.errors = {}; SET.sig = null; tick(); };
  document.querySelectorAll('#content .srow').forEach(wireRow);
  applySetFilter();
  refreshSetBar();
}
function settingFoot(s){
  const dirty = hasOwn(SET.dirty, s.key);
  const bits = [`default: ${esc(s.default_text || fmtSetVal(s, s.default) || 'unset')}`];
  if(s.readonly && s.source_file) bits.push(`in ${esc(s.source_file)}`);
  else if((s.restart||[]).length) bits.push('restart: ' + esc(restartNames(s.restart, SET.procs).join(', ')));
  if(dirty && SET.dirty[s.key]===null) bits.push('<b style="color:var(--warn)">resets to default on Save</b>');
  const canReset = s.editable && (s.source==='.env' || dirty) && !(dirty && SET.dirty[s.key]===null);
  const err = SET.errors[s.key];
  return `<div class="shint">${bits.join(' · ')}${canReset?' <button class="sreset" title="back to the code default">reset</button>':''}</div>`
       + (err ? `<div class="serr">${esc(err)}</div>` : '');
}
function settingRow(s){
  const dirty = hasOwn(SET.dirty, s.key);
  const cur = dirty ? SET.dirty[s.key] : s.value;
  const tags = [`<span class="badge">${esc(s.applies_to)}</span>`];
  if(s.secret) tags.push('<span class="badge sec">secret</span>');
  if(s.readonly) tags.push('<span class="badge ro">read-only</span>');
  if(s.source==='.env') tags.push('<span class="badge env">.env</span>');
  if(s.invalid) tags.push('<span class="badge cut">unparsable</span>');
  let ctl;
  if(s.secret){
    ctl = `<div class="sval">${s.is_set ? esc(s.value) : '<span class="dim">not set</span>'}</div>`;
  } else if(s.readonly){
    ctl = `<div class="sval">${esc(fmtSetVal(s, s.value))}</div>`;
  } else if(s.type==='bool'){
    const on = cur===null ? !!s.default : !!cur;
    ctl = `<label class="sline"><input type="checkbox" data-key="${esc(s.key)}"${on?' checked':''}> <span class="dim" data-label>${on?'on':'off'}</span></label>`;
  } else if(s.type==='choice'){
    const v = cur===null ? s.default : cur;
    ctl = `<div class="sline"><select data-key="${esc(s.key)}">${(s.choices||[]).map(c => `<option value="${esc(c)}"${c===v?' selected':''}>${esc(c)}</option>`).join('')}</select></div>`;
  } else {
    const num = s.type==='int' || s.type==='float';
    const v = cur===null ? (s.default==null ? '' : s.default) : cur;
    const attrs = num ? ` step="${s.type==='int'?'1':'any'}"${s.min!=null?` min="${s.min}"`:''}${s.max!=null?` max="${s.max}"`:''}` : '';
    ctl = `<div class="sline"><input type="${num?'number':'text'}" data-key="${esc(s.key)}" value="${esc(v==null?'':v)}" placeholder="${esc(s.default_text||'')}"${attrs} autocomplete="off" spellcheck="false"></div>`;
  }
  const cls = ['srow'];
  if(dirty) cls.push('dirty'); else if(s.modified) cls.push('mod');
  if(SET.errors[s.key]) cls.push('bad');
  const search = (s.key + ' ' + s.description + ' ' + s.applies_to).toLowerCase();
  return `<div class="${cls.join(' ')}" data-key="${esc(s.key)}" data-search="${esc(search)}">`
    + `<div><div class="sk">${esc(s.key)}</div><div class="sd">${esc(s.description)}</div><div class="sm">${tags.join('')}</div></div>`
    + `<div class="sctl">${ctl}<div class="sfoot">${settingFoot(s)}</div></div></div>`;
}
function rowEl(key){ return document.querySelector(`#content .srow[data-key="${CSS.escape(key)}"]`); }
function wireRow(row){
  if(!row) return;
  const key = row.dataset.key;
  const input = row.querySelector('[data-key]');
  if(input){ input.oninput = () => onSetInput(key, input); input.onchange = () => onSetInput(key, input); }
  const rb = row.querySelector('.sreset');
  if(rb) rb.onclick = () => resetSetting(key);
}
function markRow(key){
  const row = rowEl(key), s = SET.rows[key];
  if(!row || !s) return;
  const dirty = hasOwn(SET.dirty, key);
  row.classList.toggle('dirty', dirty);
  row.classList.toggle('mod', !dirty && !!s.modified);
  row.classList.toggle('bad', !!SET.errors[key]);
  const foot = row.querySelector('.sfoot');
  if(foot){ foot.innerHTML = settingFoot(s); const rb = foot.querySelector('.sreset'); if(rb) rb.onclick = () => resetSetting(key); }
}
function onSetInput(key, el){
  const s = SET.rows[key];
  if(!s) return;
  let v, orig;
  if(s.type==='bool'){
    v = !!el.checked; orig = !!s.value;
    const lab = el.parentNode.querySelector('[data-label]'); if(lab) lab.textContent = v ? 'on' : 'off';
  } else if(s.type==='choice'){
    v = el.value; orig = s.value;
  } else {
    v = el.value.trim(); orig = s.value==null ? '' : String(s.value);
    if(v==='') v = (orig==='' ? '' : null);
  }
  if(v===orig || (v==='' && orig==='') || (v===null && s.source!=='.env')) delete SET.dirty[key]; else SET.dirty[key] = v;
  delete SET.errors[key];
  markRow(key); refreshSetBar();
}
function resetSetting(key){
  const s = SET.rows[key];
  if(!s) return;
  if(s.source==='.env') SET.dirty[key] = null; else delete SET.dirty[key];
  delete SET.errors[key];
  const row = rowEl(key);
  if(row){ row.outerHTML = settingRow(s); wireRow(rowEl(key)); }
  applySetFilter(); refreshSetBar();
}
function applySetFilter(){
  const words = (SET.filter||'').toLowerCase().split(/\s+/).filter(Boolean);
  let shown = 0;
  document.querySelectorAll('#content .sgroup').forEach(sec => {
    let any = 0;
    sec.querySelectorAll('.srow').forEach(row => {
      const s = SET.rows[row.dataset.key] || {};
      const hay = row.dataset.search || '';
      const ok = words.every(w => hay.includes(w)) && (!SET.onlyMod || s.modified || hasOwn(SET.dirty, row.dataset.key));
      row.style.display = ok ? '' : 'none';
      if(ok) any++;
    });
    sec.style.display = any ? '' : 'none';
    shown += any;
  });
  const c = $('setCount'); if(c) c.textContent = `${shown} shown`;
}
function refreshSetBar(note){
  const n = Object.keys(SET.dirty).length;
  const sb = $('setSave'), db = $('setDiscard'), box = $('setSaved');
  if(sb){ sb.disabled = SET.saving || !n; sb.textContent = SET.saving ? 'saving…' : (n ? `Save ${n} change${n>1?'s':''}` : 'Save'); }
  if(db) db.disabled = SET.saving || !n;
  if(!box) return;
  if(note){ box.innerHTML = `<span style="color:var(--warn)">${esc(note)}</span>`; return; }
  const r = SET.saved;
  if(!r){ box.innerHTML = ''; return; }
  const hh = SET.hh||{}, up = !!(hh.house||{}).up, present = hh.present||[];
  const btns = [];
  (r.restart||[]).forEach(p => {
    if(p==='house') btns.push(`<button class="cbtn" data-ract="restart_house"${up?'':' disabled'}>Restart the house</button>`);
    else if(p==='aiva' || p==='saga'){
      const nm = nameOf(p);
      btns.push(present.includes(p) ? `<button class="cbtn" data-ract="restart_sister" data-rtarget="${p}"${up?'':' disabled'}>Restart ${esc(nm)}</button>`
                                    : `<span class="dim">${esc(nm)} isn't running — she'll start with the new values</span>`);
    }
  });
  box.innerHTML = `<span>${esc(r.message||'saved')}</span>${btns.join('')}<button class="sreset" id="setSavedX" title="dismiss">×</button>`;
  box.querySelectorAll('[data-ract]').forEach(b => b.onclick = () => pressControl(b.dataset.ract, b.dataset.rtarget||null,
    b.dataset.ract==='restart_house' ? 'Restart the house? The sisters stay up and reconnect.' : null, b.textContent));
  const x = $('setSavedX'); if(x) x.onclick = () => { SET.saved = null; refreshSetBar(); };
}
async function saveSettings(){
  const updates = Object.assign({}, SET.dirty);
  if(!Object.keys(updates).length || SET.saving) return;
  SET.saving = true; refreshSetBar();
  let res;
  try{
    const r = await fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({updates})});
    res = await r.json();
  }catch(e){ res = {ok:false, error:'dashboard server not reachable'}; }
  SET.saving = false;
  if(res && res.ok){
    SET.dirty = {}; SET.errors = {}; SET.saved = res; SET.sig = null;
    toast(res.message || 'saved', true);
    tick();
  } else {
    SET.errors = (res && res.errors) || {};
    Object.keys(SET.errors).forEach(markRow);
    toast('not saved: ' + ((res && res.error) || 'failed'), false);
    refreshSetBar();
  }
}

// --- typing to them ------------------------------------------------------------------
// The chat box lives in its OWN persistent host (#sayhost), built once and never replaced by a poll,
// so it keeps native focus, caret and draft while you type. renderSayBox only refreshes state
// (enabled/placeholder), never the node. Enter sends (Shift+Enter is left alone).
function renderSayBox(up){
  const host = $('sayhost');
  if(!host) return;
  let box = $('sayBox');
  if(!box){
    host.innerHTML = `<div class="say"><input id="sayBox" type="text" maxlength="__MAXSAY__" autocomplete="off">`
                   + `<button id="sayBtn">Send</button><span class="hint" id="sayHint"></span></div>`;
    box = $('sayBox');
    const btn = $('sayBtn');
    const refresh = () => { if(btn) btn.disabled = SAY.sending || box.disabled || !box.value.trim(); };
    box.oninput = refresh;
    box.onkeydown = (e) => { if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); sendSay(); } };
    if(btn) btn.onclick = () => sendSay();
  }
  const btn = $('sayBtn'), hint = $('sayHint');
  box.placeholder = up ? 'Type to them as you would speak — "Aiva, …", "Saga, …", "both of you, …"'
                       : 'Start the house to type to them';
  box.disabled = SAY.sending || !up;                 // only ever disabled when sending or house down
  if(btn){ btn.disabled = SAY.sending || !up || !box.value.trim(); btn.textContent = SAY.sending?'sending…':'Send'; }
  if(hint) hint.textContent = up ? '' : 'The house is down — nobody is listening.';
}
function clearSayBox(){ const host = $('sayhost'); if(host && host.innerHTML) host.innerHTML = ''; }
async function sendSay(){
  const box = $('sayBox'), btn = $('sayBtn');
  if(!box) return;
  const text = (box.value||'').trim();
  if(!text || SAY.sending) return;
  SAY.sending = true;
  box.disabled = true;
  if(btn){ btn.disabled = true; btn.textContent = 'sending…'; }
  let res;
  try{
    const r = await fetch('/api/say', {method:'POST', headers:{'Content-Type':'application/json'},
                                       body: JSON.stringify({text})});
    res = await r.json();
  }catch(e){ res = {ok:false, error:'dashboard server not reachable'}; }
  SAY.sending = false;
  if(res && res.ok) box.value = '';
  box.disabled = false;
  if(btn){ btn.disabled = !box.value.trim(); btn.textContent = 'Send'; }
  box.focus();                                        // keep the cursor so you can fire off another
  toast(res && res.ok ? 'sent' : ('not sent: ' + ((res&&res.error)||'failed')), !!(res && res.ok));
  tick();
}

// --- the Tasks tab: every task each sister ran, and one in full ------------------------------
// The toolbar, the card list and the detail pane are built ONCE when the tab opens; a poll only
// re-renders the card list when it changed, and re-fetches the open task (/api/task) only when its
// `updated` stamp moved — so an open detail, its expanded <details> and its scroll survive polling.
// The open task lives in the hash (#tasks/<slug>/<id>), so a sister tab's task row can link to it.
const TASK_ID_RE = /^t-\d{8}-\d{6}-[0-9a-f]{4}$/;
let TK = {sister:'', outcome:'', q:'', before:null, sel:null, listSig:null, detailFor:null, detailUpd:null, req:0, inflight:null, showAnn:false, rows:[]};
function readTaskHash(){
  const parts = (location.hash||'').replace(/^#/,'').split('/');
  if(parts[0]==='tasks' && parts.length>=3 && TABS.some(t=>t.who===parts[1]) && TASK_ID_RE.test(parts[2])){
    TK.sel = {who:parts[1], id:parts[2]}; return true;
  }
  return false;
}
const STEP_ICON = {done:'✓', failed:'✗', running:'●', skipped:'↷', not_needed:'–', pending:'○'};
const money = (n) => (n==null||n==='') ? '—' : '$' + Number(n).toFixed(Number(n)>=1?2:4);
function taskMeta(t){
  const bits = [];
  if(t.steps_total) bits.push(`steps ${t.steps_done}/${t.steps_total}${t.steps_failed?` · ${t.steps_failed} failed`:''}`);
  if(t.duration) bits.push(t.duration);
  bits.push(money(t.cost_usd) + (t.claude_cost?` + ${money(t.claude_cost)} Claude`:''));
  if(t.agent_grade) bits.push('grade ' + t.agent_grade);
  if(t.corrected) bits.push('corrected');
  return bits.map(esc).join(' · ');
}
function taskCard(t){
  const on = TK.sel && TK.sel.who===t.sister && TK.sel.id===t.id;
  return `<div class="tcard${on?' on':''}" data-who="${esc(t.sister)}" data-id="${esc(t.id)}">`
    + `<div class="th"><span class="sb ${esc(t.sister)}">${esc(t.sister_name)}</span>${pill(t.bucket)}`
    + `${t.status!==t.bucket&&t.status!=='done'?`<span class="dim">${esc(t.status)}</span>`:''}`
    + `${t.claude?'<span class="badge cc">Claude Code</span>':''}${t.awaiting?'<span class="badge private">waiting for you</span>':''}`
    + `<span class="when" title="${esc(t.created||'')}">${esc(t.created_ago||'')}</span></div>`
    + `<div class="tg">${esc(t.goal)}</div><div class="d">${taskMeta(t)}</div>`
    + (t.summary?`<div class="d dim">${esc(t.summary)}</div>`:'') + `</div>`;
}
function buildTasksShell(d){
  const tk = d.tasks||{};
  const sis = (tk.sisters||[]).map(s=>`<option value="${esc(s.slug)}"${TK.sister===s.slug?' selected':''}>${esc(s.name)}</option>`).join('');
  const outs = (tk.buckets||[]).map(b=>`<option value="${esc(b)}"${TK.outcome===b?' selected':''}>${esc(b)}</option>`).join('');
  $('content').innerHTML = `<section><h2>Tasks <span class="cnt" id="tkCnt"></span></h2>`
    + `<div class="toolbar"><label>Sister <select id="tkSister"><option value="">both</option>${sis}</select></label>`
    + `<label>Outcome <select id="tkOutcome"><option value="">any</option>${outs}</select></label>`
    + `<input type="search" id="tkQ" placeholder="search goals, summaries, ids…" value="${esc(TK.q)}"></div>`
    + `<div class="tkgrid"><div><div class="tklist" id="tkList"></div><div class="toolbar" style="margin-top:10px" id="tkPager"></div></div>`
    + `<div class="tkpane" id="tkDetail"></div></div></section>`;
  $('tkSister').onchange = (e) => { TK.sister = e.target.value; TK.before = null; TK.listSig = null; tick(); };
  $('tkOutcome').onchange = (e) => { TK.outcome = e.target.value; TK.before = null; TK.listSig = null; tick(); };
  let qt = null;
  $('tkQ').oninput = (e) => { clearTimeout(qt); qt = setTimeout(() => { TK.q = e.target.value.trim(); TK.before = null; TK.listSig = null; tick(); }, 300); };
  $('tkList').onclick = (e) => {
    const c = e.target.closest('.tcard'); if(!c) return;
    openTask(c.dataset.who, c.dataset.id, true);
  };
  TK.listSig = null; TK.detailFor = null; TK.detailUpd = null;
  renderTaskPlaceholder();
}
function renderTaskPlaceholder(){
  if(!TK.sel) $('tkDetail').innerHTML = `<div class="empty">Pick a task to see it start to finish — its plan, every step's evidence, the Claude Code sessions it ran and the full event timeline.</div>`;
}
function openTask(who, id, scroll){
  TK.sel = {who, id};
  try{ history.replaceState(null, '', `#tasks/${who}/${id}`); }catch(e){}
  document.querySelectorAll('#tkList .tcard').forEach(c => c.classList.toggle('on', c.dataset.who===who && c.dataset.id===id));
  TK.detailFor = null;
  $('tkDetail').innerHTML = `<div class="empty">loading…</div>`;
  fetchTask(scroll);
}
async function fetchTask(scroll){
  if(!TK.sel) return;
  const {who, id} = TK.sel, key = who+'/'+id, my = ++TK.req;
  TK.inflight = key;
  let res, t;
  try{
    res = await fetch(`/api/task?who=${encodeURIComponent(who)}&id=${encodeURIComponent(id)}`, {cache:'no-store'});
    t = await res.json();
  }catch(e){ t = {error:'dashboard server not reachable'}; }
  if(my !== TK.req) return;
  TK.inflight = null;
  if(WHO!=='tasks' || !$('tkDetail')) return;
  const pane = $('tkDetail');
  const openKeys = new Set([...pane.querySelectorAll('details[data-k]')].filter(x=>x.open).map(x=>x.dataset.k));
  const keepScroll = TK.detailFor===key ? pane.scrollTop : 0;
  const same = TK.detailFor===key;
  pane.innerHTML = t.error ? `<div class="banner">${esc(t.error)}</div>` : renderTaskDetail(t);
  if(same) pane.querySelectorAll('details[data-k]').forEach(x => { x.open = openKeys.has(x.dataset.k); });
  pane.scrollTop = keepScroll;
  const ann = $('tkAnn'); if(ann) ann.onchange = (e) => { TK.showAnn = e.target.checked; pane.querySelectorAll('.tev.sec').forEach(x => x.style.display = TK.showAnn?'':'none'); };
  TK.detailFor = key; TK.detailUpd = t.updated || null;
  if(scroll && window.matchMedia('(max-width:980px)').matches) pane.scrollIntoView({behavior:'smooth', block:'start'});
}
function det(k, summary, body, open){
  return `<details class="more" data-k="${esc(k)}"${open?' open':''}><summary>${summary}</summary>${body}</details>`;
}
function actionBox(st){
  const a = st.action||{};
  if(st.kind==='shell') return `<pre class="box">${esc(a.command||'')}</pre>${a.purpose?`<div class="d">${esc(a.purpose)}</div>`:''}`;
  if(st.kind==='delegate') return `<div class="d mono">${esc(a.project_dir||'')}</div>` + det('prompt-'+st.n, 'Prompt for Claude Code', `<pre class="box">${esc(a.prompt||'')}</pre>`);
  if(st.kind==='ask') return `<div class="d">asks: “${esc(a.question||'')}”</div>`;
  if(st.kind==='tool'){
    const args = Object.entries(a.args||{}).map(([k,v]) => `${k}: ${typeof v==='string'?v:JSON.stringify(v)}`).join('\n');
    return `<div class="d"><span class="tag">${esc(a.tool||'?')}</span></div>${args?`<pre class="box">${esc(args)}</pre>`:''}`;
  }
  const rest = Object.entries(a).map(([k,v]) => `${k}: ${typeof v==='string'?v:JSON.stringify(v)}`).join('\n');
  return rest ? `<pre class="box">${esc(rest)}</pre>` : '';
}
function fileList(arr, cap){
  if(!arr || !arr.length) return '';
  const shown = arr.slice(0, cap||arr.length);
  return `<ul class="flist">${shown.map(f=>`<li>${esc(f.path)}${f.count>1?` <b>×${f.count}</b>`:''}</li>`).join('')}</ul>`
    + (arr.length>shown.length?`<div class="dim" style="font-size:12px">…and ${arr.length-shown.length} more</div>`:'');
}
function renderActivity(a, k){
  if(!a) return `<div class="dim" style="font-size:12.5px;margin-top:4px">No session transcript found on this PC.</div>`;
  if(a.too_big) return `<div class="dim" style="font-size:12.5px;margin-top:4px">The session transcript is too big to summarise here.</div>`;
  const tools = Object.entries(a.tools||{}).map(([n,c]) => `<span class="tag">${esc(n)} ${c}</span>`).join('');
  let h = `<div class="d">${a.turns||0} turns${a.duration_ms?` · ${fmtms(a.duration_ms)}`:''}${a.lines_added!=null?` · <span style="color:var(--good)">+${a.lines_added}</span> <span style="color:var(--bad)">−${a.lines_removed||0}</span> lines`:''}${a.tool_errors?` · ${a.tool_errors} tool errors`:''}${a.model?` · ${esc(a.model)}`:''}</div>`;
  if(tools) h += `<div class="chips" style="margin-top:4px">${tools}</div>`;
  const ed = a.files_edited||[], wr = a.files_written||[], rd = a.files_read||[];
  if(ed.length) h += `<div class="d" style="margin-top:8px;color:var(--ink)">Edited ${ed.length} file${ed.length>1?'s':''}</div>` + fileList(ed);
  if(wr.length) h += `<div class="d" style="margin-top:8px;color:var(--ink)">Wrote ${wr.length} file${wr.length>1?'s':''}</div>` + fileList(wr);
  if(!ed.length && !wr.length) h += `<div class="d" style="margin-top:8px">Changed no files.</div>`;
  if(rd.length) h += det(k+'-read', `Read ${rd.length} file${rd.length>1?'s':''}`, fileList(rd, 80));
  if((a.commands||[]).length) h += det(k+'-cmd', `Ran ${a.commands_total} command${a.commands_total>1?'s':''}`, `<pre class="box">${esc(a.commands.join('\n'))}${a.commands_total>a.commands.length?`\n… and ${a.commands_total-a.commands.length} more`:''}</pre>`);
  if((a.searches||[]).length) h += det(k+'-srch', `Searched ${a.searches_total} time${a.searches_total>1?'s':''}`, `<pre class="box">${esc(a.searches.join('\n'))}</pre>`);
  if(a.final_text) h += det(k+'-final', 'Its final answer this run', `<pre class="box">${esc(a.final_text)}</pre>`);
  return h;
}
function renderClaudeJob(j){
  const k = 'job-'+j.job;
  let h = `<div class="ccjob"><div class="th" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><span class="badge cc">Claude Code</span>`
    + `<span class="mono">${esc(j.job)}</span>${j.status==='done'?pill('success'):j.status==='failed'?pill('failed'):''}`
    + `<span class="dim" style="margin-left:auto">${j.runs.length} run${j.runs.length===1?'':'s'}${j.cost?` · ${money(j.cost)}`:''}</span></div>`;
  if(j.title) h += `<div class="d" style="margin-top:4px">step ${esc(j.n)} · ${esc(j.title)}</div>`;
  if(j.project_dir) h += `<div class="d mono">${esc(j.project_dir)}</div>`;
  if(j.prompt) h += det(k+'-prompt', 'The prompt it was given', `<pre class="box">${esc(j.prompt)}</pre>`);
  j.runs.forEach((r, i) => {
    h += `<div class="ccrun"><div class="d" style="color:var(--ink)">Run ${i+1}${r.started?` · <span title="${esc(r.started)}">${esc((r.started||'').replace('T',' ').slice(0,19))}</span>`:''}${r.took?` · ${esc(r.took)}`:''}${r.status?` · ${esc(r.status)}`:''}${r.cost!=null?` · ${money(r.cost)}`:''}</div>`;
    if(r.session) h += `<div class="d mono" title="${esc(r.transcript_path||'')}">session ${esc(r.session)}${r.inferred?' <span class="dim">(matched by prompt + time)</span>':''}</div>`;
    h += renderActivity(r.activity, k+'-r'+i);
    if(r.transcript_path) h += `<div class="dim mono" style="font-size:11.5px;margin-top:6px;word-break:break-all">${esc(r.transcript_path)}</div>`;
    h += `</div>`;
  });
  const res = j.result;
  if(res){
    h += det(k+'-result', `Saved result <span class="dim">(${res.legacy?'old temp log':'logs/coding'} · ${fmtK(res.chars)} chars${j.runs.length>1?' · last run':''})</span>`,
      `<pre class="box">${esc(res.text)}</pre><div class="dim mono" style="font-size:11.5px;margin-top:4px;word-break:break-all">${esc(res.path)}</div>`, true);
  } else {
    h += `<div class="dim" style="font-size:12.5px;margin-top:8px">No saved result file for this job.</div>`;
  }
  return h + `</div>`;
}
function renderTaskDetail(t){
  let h = `<div class="th" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><span class="sb ${esc(t.sister)}">${esc(t.sister_name)}</span>${pill(t.bucket)}`
    + `${t.status!==t.bucket?`<span class="dim">${esc(t.status)}</span>`:''}${t.claude?'<span class="badge cc">Claude Code</span>':''}`
    + `<span class="dim mono" style="margin-left:auto">${esc(t.id)}</span></div>`;
  h += `<div class="goal">${esc(t.goal)}</div>`;
  h += `<div class="d muted" style="font-size:13px">${esc((t.created||'').replace('T',' ').slice(0,19))} · ${esc(t.created_ago||'')}${t.duration?` · took ${esc(t.duration)}`:''} · ${esc(t.llm_calls)} LLM calls · ${esc(t.tool_calls)} tool calls · ${money(t.cost_usd)}${t.claude_cost?` + ${money(t.claude_cost)} Claude Code`:''}</div>`;
  if(t.pending) h += `<div class="banner soft" style="margin-top:12px"><b>Waiting for you</b> (${esc(t.pending.kind||'')}): ${esc(t.pending.text||'')}</div>`;
  h += `<div style="margin-top:10px">`;
  if(t.details) h += rowKV('Details', t.details);
  if(t.family || t.approach) h += rowKV('Kind', [t.family, t.approach].filter(Boolean).join(' · '));
  if(t.familiarity) h += rowKV('Familiarity', t.familiarity);
  if(t.decision) h += rowKV('Decision', t.decision);
  if(t.result_summary) h += rowKV('Result', t.result_summary);
  if(t.agent_grade) h += rowKV('Grade', t.agent_grade + (t.agent_grade_reason?` — ${t.agent_grade_reason}`:''));
  if(t.user_feedback) h += rowKV('Your feedback', t.user_feedback + (t.corrected?' (corrected)':''));
  else if(t.corrected) h += rowKV('Corrected', 'yes');
  h += `</div>`;

  const steps = t.steps||[];
  h += `<h3>Plan <span class="dim" style="text-transform:none;letter-spacing:0;font-weight:400">${steps.length} step${steps.length===1?'':'s'}</span></h3>`;
  if(steps.length){
    h += `<div class="steps">`;
    steps.forEach(st => {
      h += `<div class="step ${esc(st.status)}"><span class="ic">${STEP_ICON[st.status]||'○'}</span>`
        + `<div class="st">${esc(st.n)}. ${esc(st.title)}</div>`
        + `<div class="d"><span class="tag">${esc(st.kind)}</span>${esc(st.status)}${st.attempts>1?` · ${st.attempts} attempts`:''}${st.risk&&st.risk!=='none'?` · risk ${esc(st.risk)}`:''}${st.took?` · ${esc(st.took)}`:''}${st.started?` · ${esc((st.started||'').slice(11,19))}`:''}</div>`
        + actionBox(st)
        + (st.expect?`<div class="d">expects: ${esc(st.expect)}</div>`:'')
        + (st.failure?`<div class="fail">${esc(st.failure)}</div>`:'')
        + ((st.evidence||[]).length ? det('ev-'+st.n, `Evidence (${st.evidence.length})`, st.evidence.map(e=>`<pre class="box">${esc(e)}</pre>`).join('')) : '')
        + `</div>`;
    });
    h += `</div>`;
  } else h += `<div class="empty">No plan was made.</div>`;

  const jobs = t.claude_jobs||[];
  if(jobs.length){
    h += `<h3>Claude Code <span class="dim" style="text-transform:none;letter-spacing:0;font-weight:400">${jobs.length} job${jobs.length===1?'':'s'}</span></h3>`;
    jobs.forEach(j => h += renderClaudeJob(j));
  }

  const tl = t.timeline||[];
  h += `<h3 style="display:flex;align-items:center;gap:10px">Timeline <span class="dim" style="text-transform:none;letter-spacing:0;font-weight:400">${tl.length} events</span>`
    + (t.announcements?`<label class="dim" style="text-transform:none;letter-spacing:0;font-weight:400;font-size:12px;margin-left:auto;cursor:pointer"><input type="checkbox" id="tkAnn"${TK.showAnn?' checked':''}> show ${t.announcements} announcements</label>`:'')
    + `</h3>`;
  if(tl.length){
    let day = null;
    tl.forEach(e => {
      if(e.day && e.day!==day){ day = e.day; h += `<div class="dim" style="font-size:11px;margin:8px 0 2px;text-transform:uppercase;letter-spacing:.5px">${esc(day)}</div>`; }
      let detail = esc(e.detail);
      if(e.claude){ const c = e.claude; detail = `job <span class="mono">${esc(c.job)}</span> ${esc(c.status||'')}${c.cost!=null?` · ${money(c.cost)}`:''}${c.session?` · session <span class="mono">${esc(c.session.slice(0,8))}</span>`:''}`; }
      h += `<div class="tev${e.secondary?' sec':''}"${e.secondary&&!TK.showAnn?' style="display:none"':''}><span><span class="tdot ${esc(e.tone)}"></span></span><span class="time">${esc(e.time)}</span><span class="evn">${esc(e.event)}</span><span class="dt">${detail}</span></div>`;
    });
  } else h += `<div class="empty">Nothing logged.</div>`;
  return h;
}
function renderTasks(d){
  $('title').textContent = 'Tasks';
  const hh = d.household||{}, h = hh.house||{};
  $('dot').className = 'dot ' + (h.up?'live':'offline');
  $('statusText').textContent = (h.up?'house up':'house down') + ((hh.present_names||[]).length?` · ${hh.present_names.join(' & ')} connected`:'');
  $('subtitle').textContent = 'v' + (d.version||'?') + ' · every task and coding session the sisters ran, start to finish';
  $('banner').innerHTML = d.error ? `<div class="banner">${esc(d.error)}</div>` : '';
  const tk = d.tasks||{}, tl = tk.tally||{};
  const cards = [];
  (tk.sisters||[]).forEach(s => {
    const x = tl[s.slug]||{};
    cards.push(statCard(x.total??0, s.name+' tasks', s.started===false?'not started yet':`${x.success||0} ok · ${x.failed||0} failed · ${(x.cancelled||0)+(x.interrupted||0)} stopped`));
  });
  const all = Object.values(tl);
  const sum = (k) => all.reduce((a,x)=>a+(x[k]||0),0);
  cards.push(statCard(sum('delegated'), 'Claude Code', 'tasks that delegated'));
  cards.push(statCard(money(sum('cost_usd')), 'Her spend', sum('claude_cost')?`+ ${money(sum('claude_cost'))} Claude Code (logged)`:'tasks, lifetime'));
  $('stats').innerHTML = cards.join('');

  if(!$('tkList')) buildTasksShell(d);
  const rows = tk.rows||[];
  TK.rows = rows;
  $('tkCnt').textContent = `${tk.matched||0} match${tk.matched===1?'':'es'}${TK.before?' · older page':''}`;
  const sig = JSON.stringify([rows.map(r=>[r.sister,r.id,r.updated,r.created_ago]), tk.has_more, TK.sel&&TK.sel.id]);
  if(sig !== TK.listSig){
    TK.listSig = sig;
    $('tkList').innerHTML = rows.length ? rows.map(taskCard).join('')
      : `<div class="empty">${(TK.sister||TK.outcome||TK.q)?'No task matches these filters.':'No tasks yet — they appear once a sister runs one (start_task).'}</div>`;
    $('tkPager').innerHTML = (TK.before||tk.has_more) ? `<button id="tkNewest"${TK.before?'':' disabled'}>newest</button><button id="tkOlder"${tk.has_more?'':' disabled'}>older</button>` : '';
    const nb = $('tkNewest'); if(nb) nb.onclick = () => { TK.before = null; TK.listSig = null; tick(); };
    const ob = $('tkOlder'); if(ob) ob.onclick = () => { if(tk.next_before){ TK.before = tk.next_before; TK.listSig = null; tick(); } };
  }
  if(TK.sel){
    const key = TK.sel.who+'/'+TK.sel.id;
    const row = rows.find(r => r.sister===TK.sel.who && r.id===TK.sel.id);
    if(TK.inflight !== key && (TK.detailFor !== key || (row && row.updated !== TK.detailUpd))) fetchTask(false);
  }
}

function rowKV(k,v){ return `<div class="row"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`; }
function fmtms(ms){ ms=ms||0; return ms>=1000 ? (ms/1000).toFixed(ms>=10000?0:1)+'s' : Math.round(ms)+'ms'; }
function dial(label,val,bipolar){
  val = val||0;
  const pct = bipolar ? (val+1)/2*100 : val*100;
  const col = bipolar ? (val>0.05?'var(--good)':val<-0.05?'var(--bad)':'var(--accent)') : 'var(--accent)';
  const shown = bipolar ? ((val>=0?'+':'')+val.toFixed(2)) : val.toFixed(2);
  return `<div class="vital"><div class="lab">${esc(label)}</div><div class="bar"><i style="width:${Math.max(0,Math.min(100,pct))}%;background:${col}"></i></div><div class="dim" style="font-size:12px;margin-top:3px">${shown}</div></div>`;
}
function vital(lab,val){
  const col = val<25?'var(--bad)':val<50?'var(--warn)':'var(--good)';
  return `<div class="vital"><div class="lab">${esc(lab)}</div><div class="bar"><i style="width:${Math.max(0,Math.min(100,val))}%;background:${col}"></i></div><div class="dim" style="font-size:12px;margin-top:3px">${Math.round(val)}</div></div>`;
}

async function tick(){
  const who = WHO;
  let url = '/api/data?who=' + encodeURIComponent(who);
  if(who==='conversations'){
    if(CONV.day) url += '&day=' + encodeURIComponent(CONV.day);
    if(CONV.before) url += '&before=' + encodeURIComponent(CONV.before);
  } else if(who==='tasks'){
    if(TK.sister) url += '&sister=' + encodeURIComponent(TK.sister);
    if(TK.outcome) url += '&outcome=' + encodeURIComponent(TK.outcome);
    if(TK.q) url += '&q=' + encodeURIComponent(TK.q);
    if(TK.before) url += '&before=' + encodeURIComponent(TK.before);
  }
  try{
    const res = await fetch(url, {cache:'no-store'});
    const d = await res.json();
    if(who !== WHO) return;           // the tab changed while this was in flight
    render(d);
  }catch(e){
    $('statusText').textContent = 'dashboard server not reachable';
    $('dot').className = 'dot offline';
  }
}
readTaskHash();
renderTabs();
tick();
setInterval(tick, POLL);
</script>
</body>
</html>"""


def page():
    """The page for the current CHARACTERS/poll settings (built per request; it's cheap)."""
    return (PAGE_TEMPLATE.replace("__POLL__", str(POLL_SECS))
            .replace("__TABS__", json.dumps(tabs()))
            .replace("__MAXSAY__", str(MAX_SAY_CHARS))
            .replace("__TEMP_CORE__", json.dumps(TEMPERAMENT_CORE)))


def api_response(query=""):
    """(status, payload) for `/api/data?<query>` — shared by the HTTP handler and tests.
    `who` = a character slug | 'conversations' (default DEFAULT_WHO); unknown → 404 JSON.
    Conversations also take `day=YYYY-MM-DD`, `before=<id>`, `limit=<n>`; Tasks take `sister`,
    `outcome` (a TASK_BUCKETS value), `q` (text), `before=<created_ts>|<id>`, `limit`."""
    qs = parse_qs(query or "", keep_blank_values=False)
    who = (qs.get("who", [DEFAULT_WHO])[0] or DEFAULT_WHO).strip().lower()
    if who not in (CONVERSATIONS, TASKS_TAB, SETTINGS) and who not in CHARACTERS:
        return 404, {"error": f"unknown who {who!r}", "known": [*CHARACTERS, CONVERSATIONS, TASKS_TAB, SETTINGS],
                     "generated_ts": datetime.now().isoformat()}
    opts = {}
    if who == CONVERSATIONS:
        opts = {"day": qs.get("day", [None])[0], "before": qs.get("before", [None])[0],
                "limit": qs.get("limit", [200])[0]}
    elif who == TASKS_TAB:
        opts = {k: qs.get(k, [None])[0] for k in ("sister", "outcome", "q", "before", "limit")}
    try:
        return 200, collect(who, **opts)
    except Exception as e:  # never let one bad query kill the feed
        return 200, {"error": f"collect failed: {e}", "who": who,
                     "generated_ts": datetime.now().isoformat()}


def control_response(body, send=None, launch=None):
    """(status, payload) for `POST /api/control` — shared by the HTTP handler and tests. `body` is
    the raw JSON (bytes/str) or an already-parsed dict `{"action": <CONTROL_ACTIONS>, "target":
    <CONTROL_TARGETS> | null}`; sister actions need a sister slug, `sister_talk` takes
    audible|private|nothing, everything else takes no target. `start_sister` / `start_house` are
    launched LOCALLY (the .bat) and work with the house down; everything else goes to the house.
    400 = bad request (nothing sent), 503 = the house didn't take it, 200 = sent/launched.
    `send`/`launch` override `send_control`/`launch_bat` (tests)."""
    send = send or send_control
    launch = launch or launch_bat
    if isinstance(body, (bytes, bytearray)):
        body = bytes(body).decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            body = json.loads(body) if body.strip() else None
        except ValueError:
            return 400, {"ok": False, "error": "body is not JSON"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "body must be a JSON object {action, target}"}
    action = str(body.get("action") or "").strip().lower()
    target = body.get("target")
    target = str(target).strip().lower() if target not in (None, "") else None
    if action not in CONTROL_ACTIONS:
        return 400, {"ok": False, "error": f"unknown action {action!r}", "actions": list(CONTROL_ACTIONS)}
    if target is not None and target not in CONTROL_TARGETS:
        return 400, {"ok": False, "error": f"unknown target {target!r}", "targets": list(CONTROL_TARGETS)}
    if action in _SISTER_ACTIONS:
        if target not in CHARACTERS:
            return 400, {"ok": False, "error": f"{action} needs a sister: one of {', '.join(CHARACTERS)}"}
    elif action == "sister_talk":
        if target is not None and target not in _MODE_TARGETS:
            return 400, {"ok": False, "error": "sister_talk takes audible, private or no target"}
    elif target is not None:
        return 400, {"ok": False, "error": f"{action} takes no target"}
    try:
        if action in _LOCAL_LAUNCH_ACTIONS:
            result = launch(target if action == "start_sister" else "house")
        else:
            result = send(action, target)
    except Exception as e:  # a button press must never take the dashboard down
        result = {"ok": False, "error": f"send failed: {e}"}
    if not isinstance(result, dict):
        result = {"ok": False, "error": f"bad result from send: {result!r}"}
    result.setdefault("action", action)
    result.setdefault("target", target)
    return (200 if result.get("ok") else 503), result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # keep the console quiet

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, code, payload):
        self._send(code, json.dumps(payload, ensure_ascii=False, default=str),
                   "application/json; charset=utf-8")

    def _local_client(self):
        """Control presses come from this machine only, unless AIVA_DASHBOARD_CONTROL_REMOTE=1."""
        if os.getenv("AIVA_DASHBOARD_CONTROL_REMOTE", "0") == "1":
            return True
        try:
            host = self.client_address[0]
        except (AttributeError, IndexError, TypeError):
            return False
        return host in ("127.0.0.1", "::1", "::ffff:127.0.0.1")

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path in ("/", "/index.html"):
            self._send(200, page(), "text/html; charset=utf-8")
        elif path == "/api/data":
            code, payload = api_response(query)
            self._json(code, payload)
        elif path == "/api/task":
            code, payload = task_response(query)
            self._json(code, payload)
        elif path == "/healthz":
            self._send(200, "ok", "text/plain")
        elif path in ("/favicon.svg", "/favicon.ico"):
            self._send(200, FAVICON_SVG, "image/svg+xml")
        elif path == "/api/health":
            # who is serving, and which code: the house compares `code_mtime` with dashboard.py on
            # disk and replaces a dashboard that is still running old code
            self._json(200, {"ok": True, "pid": os.getpid(), "version": AIVA_VERSION,
                             "code_mtime": _code_mtime()})
        else:
            self._send(404, "not found", "text/plain")

    def _same_origin(self):
        """A browser POST carries Origin; a settings write must come from this page itself, never
        from another site the browser happens to have open (no Origin = a script/curl on this PC)."""
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = self.headers.get("Host") or ""
        o = origin.split("://", 1)[-1].rstrip("/")
        return bool(host) and o.lower() == host.lower()

    def do_POST(self):
        path, _, _query = self.path.partition("?")
        if path not in ("/api/control", "/api/say", "/api/settings"):
            self._send(404, "not found", "text/plain")
            return
        if not self._local_client():
            self._json(403, {"ok": False, "error": "controls are only accepted from this machine"})
            return
        if path == "/api/settings" and not self._same_origin():
            self._json(403, {"ok": False, "error": "settings are only accepted from this dashboard page"})
            return
        limit = MAX_SETTINGS_BODY if path == "/api/settings" else MAX_CONTROL_BODY
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > limit:
            self._json(400, {"ok": False, "error": f"body must be 0-{limit} bytes of JSON"})
            return
        body = self.rfile.read(n) if n else b""
        if path == "/api/say":
            code, payload = say_response(body)
        elif path == "/api/settings":
            code, payload = settings_response(body)
        else:
            code, payload = control_response(body)
        self._json(code, payload)


def say_response(body, send=None):
    """(status, payload) for `POST /api/say` — the Conversations chat box. `body` is the raw JSON
    (bytes/str) or a parsed dict {"text": "..."}. One line, trimmed, at most MAX_SAY_CHARS; the
    house decides who it was for and she answers out loud. 400 = bad request, 503 = the house
    didn't take it, 200 = sent. `send` overrides `send_say` (tests)."""
    send = send or send_say
    if isinstance(body, (bytes, bytearray)):
        body = bytes(body).decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            body = json.loads(body) if body.strip() else None
        except ValueError:
            return 400, {"ok": False, "error": "body is not JSON"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": 'body must be a JSON object {"text"}'}
    text = body.get("text")
    text = " ".join(str(text).split()) if text is not None else ""
    if not text:
        return 400, {"ok": False, "error": "nothing to say"}
    if len(text) > MAX_SAY_CHARS:
        return 400, {"ok": False, "error": f"keep it under {MAX_SAY_CHARS} characters — she says it out loud"}
    try:
        result = send(text)
    except Exception as e:      # typing at them must never take the dashboard down
        result = {"ok": False, "error": f"send failed: {e}"}
    if not isinstance(result, dict):
        result = {"ok": False, "error": f"bad result from send: {result!r}"}
    result.setdefault("text", text)
    return (200 if result.get("ok") else 503), result


def settings_response(body, env_path=None):
    """(status, payload) for `POST /api/settings {"updates": {KEY: value | null}}` — the Settings
    tab's Save. Every value is validated through the registry FIRST (one bad value → 400 with
    per-key `errors`, nothing written); then one in-place `.env` write (comments/order kept,
    `.env.bak` alongside). `null` resets a key to its code default (its line is removed). Secrets
    and read-only rows are refused. Config is read at boot, so the payload says which process(es)
    must restart (`restart`, `restart_labels`, `message`). `env_path` overrides ENV_PATH (tests)."""
    env_path = env_path or ENV_PATH
    if settings_registry is None:
        return 503, {"ok": False, "error": "modules/settings_registry.py is missing"}
    if isinstance(body, (bytes, bytearray)):
        body = bytes(body).decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            body = json.loads(body) if body.strip() else None
        except ValueError:
            return 400, {"ok": False, "error": "body is not JSON"}
    if not isinstance(body, dict) or not isinstance(body.get("updates"), dict):
        return 400, {"ok": False, "error": 'body must be a JSON object {"updates": {KEY: value}}'}
    try:
        ok, payload = settings_registry.apply_updates(env_path, body["updates"])
    except OSError as e:      # a locked/readonly .env must never take the dashboard down
        return 500, {"ok": False, "error": f"couldn't write .env: {e}"}
    if not ok:
        errs = payload.get("errors") or {}
        return 400, {"ok": False, "errors": errs,
                     "error": "; ".join(errs.values()) if errs else "nothing saved"}
    return 200, {"ok": True, **payload}


def _arg(argv, flag, default=None):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def main(argv=None):
    """`--once [--who aiva|saga|conversations|tasks|settings]` prints one JSON snapshot; otherwise serves.
    `--open` opens the browser on the chosen tab."""
    argv = list(sys.argv[1:] if argv is None else argv)
    who = (_arg(argv, "--who", DEFAULT_WHO) or DEFAULT_WHO).strip().lower()
    known = [*CHARACTERS, CONVERSATIONS, TASKS_TAB, SETTINGS]
    if who not in known:
        print(f"unknown --who {who!r}; known: {', '.join(known)}", file=sys.stderr)
        return 2
    if "--once" in argv:  # print one JSON snapshot and exit (handy for a quick check)
        print(json.dumps(collect(who), ensure_ascii=False, indent=2, default=str))
        return 0
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/#{who}"
    print(f"Aiva dashboard on {url}   (Ctrl+C to stop)")
    for ch in CHARACTERS.values():
        print(f"  {ch['name']}: {ch['db']} read-only")
    print(f"  house: {HOUSE_DB}; polling every {POLL_SECS}s")
    if "--open" in argv:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
