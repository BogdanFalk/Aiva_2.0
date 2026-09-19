"""The household's dashboard — a live window on each sister (everything she is and everything
she's becoming) plus the house's conversation log, and a Controls strip with the house hotkeys.

Run it any time (nobody needs to be awake):

    cd aiva && ..\\.venv\\Scripts\\python dashboard.py
    # then open http://127.0.0.1:8760   (tabs: Aiva | Saga | Conversations)
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

CONVERSATIONS = "conversations"


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
    """The header tab strip, in order: one per character, then Conversations."""
    return ([{"who": slug, "label": ch["name"]} for slug, ch in CHARACTERS.items()]
            + [{"who": CONVERSATIONS, "label": "Conversations"}])


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
                       "monitor": av.get("monitor"), "layer": av.get("layer")})
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
            initiative = {"permission": kv.get("initiative", "ask"), "today": today_count,
                          "last": _ago(kv.get("last_initiative_ts"))}
    return {"mood": _read_mood(inner_state_path), "events": events, "events_total": total,
            "journal": journal, "desires": desires, "desires_total": dtotal,
            "initiative": initiative}


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
                   "restart_sister", "quit_sister", "start_sister", "start_house", "restart_house", "quit")
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
_SISTER_ACTIONS = ("restart_sister", "quit_sister", "start_sister")   # need a sister slug
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
.say{display:flex;gap:8px;align-items:center;margin:0 0 12px}
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
    house hotkeys over its local socket. Polls every __POLL__s · <span id="stamp"></span></footer>
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

function pickWho(){
  const h = (location.hash||'').replace(/^#/,'');
  return TABS.some(t=>t.who===h) ? h : TABS[0].who;
}
function renderTabs(){
  $('tabs').innerHTML = TABS.map(t=>`<button class="tab${t.who===WHO?' on':''}" data-who="${esc(t.who)}">${esc(t.label)}</button>`).join('');
  $('tabs').querySelectorAll('.tab').forEach(b => b.onclick = () => selectTab(b.dataset.who));
}
function selectTab(who){
  if(who===WHO) return;
  WHO = who; CONV = {day:null, before:null};
  if(location.hash !== '#'+who) location.hash = who;
  renderTabs(); tick();
}
window.addEventListener('hashchange', () => { const w = pickWho(); if(w!==WHO){ WHO = w; CONV = {day:null, before:null}; renderTabs(); tick(); } });

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
  if(d.who==='conversations'){ renderConversations(d); } else { clearSayBox(); renderCharacter(d); }
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
    if(inn.initiative){
      const ini = inn.initiative;
      html += `<div class="dim" style="font-size:12px;margin-top:10px">Initiative: <b style="color:var(--muted)">${esc(ini.permission)}</b> · ${ini.today} today${ini.last?` · last ${esc(ini.last)}`:''}</div>`;
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
  html += `<div class="muted" style="margin-bottom:6px">Recent tasks</div>`;
  if(tk.recent && tk.recent.length){
    tk.recent.forEach(t => {
      html += `<div class="item"><div class="t">${esc(t.goal_text)}</div><div class="d">${esc(t.status)}${t.outcome?` · ${esc(t.outcome)}`:''}${t.delegated_to&&t.delegated_to!=='none'?` · ↦ ${esc(t.delegated_to)}`:''}${t.agent_grade?` · grade ${esc(t.agent_grade)}`:''}${t.corrected?` · <span style="color:var(--warn)">corrected</span>`:''} · ${esc(t.created||'')}</div></div>`;
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
    Conversations also take `day=YYYY-MM-DD`, `before=<id>`, `limit=<n>`."""
    qs = parse_qs(query or "", keep_blank_values=False)
    who = (qs.get("who", [DEFAULT_WHO])[0] or DEFAULT_WHO).strip().lower()
    if who != CONVERSATIONS and who not in CHARACTERS:
        return 404, {"error": f"unknown who {who!r}", "known": [*CHARACTERS, CONVERSATIONS],
                     "generated_ts": datetime.now().isoformat()}
    opts = {}
    if who == CONVERSATIONS:
        opts = {"day": qs.get("day", [None])[0], "before": qs.get("before", [None])[0],
                "limit": qs.get("limit", [200])[0]}
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

    def do_POST(self):
        path, _, _query = self.path.partition("?")
        if path not in ("/api/control", "/api/say"):
            self._send(404, "not found", "text/plain")
            return
        if not self._local_client():
            self._json(403, {"ok": False, "error": "controls are only accepted from this machine"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_CONTROL_BODY:
            self._json(400, {"ok": False, "error": f"body must be 0-{MAX_CONTROL_BODY} bytes of JSON"})
            return
        body = self.rfile.read(n) if n else b""
        code, payload = say_response(body) if path == "/api/say" else control_response(body)
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


def _arg(argv, flag, default=None):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def main(argv=None):
    """`--once [--who aiva|saga|conversations]` prints one JSON snapshot; otherwise serves.
    `--open` opens the browser on the chosen tab."""
    argv = list(sys.argv[1:] if argv is None else argv)
    who = (_arg(argv, "--who", DEFAULT_WHO) or DEFAULT_WHO).strip().lower()
    known = [*CHARACTERS, CONVERSATIONS]
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
