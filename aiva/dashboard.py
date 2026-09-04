"""Aiva's dashboard — a live, read-only window on everything she is and everything
she's becoming.

Run it any time (she does NOT need to be awake):

    cd aiva && ..\\.venv\\Scripts\\python dashboard.py
    # then open http://127.0.0.1:8760

It opens her databases READ-ONLY (a fresh connection per request, `mode=ro`
with a busy timeout) so it can never lock her up or change a thing, reads her
state files and today's event log, and serves one self-contained page that
polls itself every few seconds. Nothing leaves the machine — host defaults to
127.0.0.1.

Knobs: AIVA_DASHBOARD_PORT (8760), AIVA_DASHBOARD_HOST (127.0.0.1),
AIVA_DASHBOARD_POLL_SECS (page refresh cadence, 8).

This is a pure observer. It imports no pipecat, no live2d, nothing heavy — just
the standard library — so it starts instantly and can't drag her down.
"""

import ctypes
import glob
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --- paths (resolve relative to THIS file, so it runs from aiva/ or the repo root) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_DB = os.path.join(_HERE, "memory.db")
_AVATAR_STATE = os.path.join(_HERE, "avatar_state.json")
_CAT_STATE = os.path.join(_HERE, "cat_state.json")
_INNER_STATE = os.getenv("AIVA_INNER_STATE") or os.path.join(_HERE, "inner_state.json")
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
]

SUBSYSTEMS = [
    ("Voice pipeline & memory", ["main.py", "memory.py", "persona.py", "utilities.py"]),
    ("Body — Live2D avatar", ["avatar.py", "avatar_renderer.py", "avatar_ipc.py"]),
    ("Desktop cat", ["cat.py", "cat_sprites.py"]),
    ("Computer use", ["computer.py", "app_launcher.py", "file_operations.py", "terminals.py"]),
    ("Tool surface", ["tools.py"]),
    ("Tasks & competence", ["tasks.py", "task_memory.py"]),
    ("Reading & becoming", ["library.py", "reading.py", "becoming.py"]),
]
RETIRED_MODULES = {"vtube_studio.py": "retired — she renders her own avatar now"}


# --- small helpers ------------------------------------------------------------------

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


def _connect():
    """Fresh read-only connection; waits briefly if she's mid-write. Never blocks long."""
    con = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True, timeout=3)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=3000")
    return con


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


# --- the actual data collection -----------------------------------------------------

def collect():
    data = {"generated_ts": datetime.now().isoformat(), "db_present": os.path.exists(_DB),
            "version": AIVA_VERSION}
    data["presence"] = _collect_presence()
    data["cat"] = _collect_cat()
    data["capabilities"] = _collect_capabilities()
    data["inner"] = _collect_inner(None)  # mood-only until a db read replaces it below
    if not data["db_present"]:
        data["headline"] = {}
        return data

    try:
        con = _connect()
    except sqlite3.Error as e:
        data["error"] = f"database busy or unreadable: {e}"
        data["headline"] = {}
        return data
    try:
        data["reading"] = _collect_reading(con)
        data["inner"] = _collect_inner(con)
        data["preferences"] = _collect_preferences(con)
        data["stats"] = _collect_stats(con)
        data["becoming"] = _collect_becoming(con)
        data["tasks"] = _collect_tasks(con)
        data["memory"] = _collect_memory(con)
        data["headline"] = _collect_headline(con, data)
    finally:
        con.close()
    return data


def _collect_presence():
    av = _load_json(_AVATAR_STATE) or {}
    pid = av.get("pid")
    running = _pid_alive(pid)

    # newest log file → recent activity feed + coarse awake/asleep guess
    recent, last_mtime, state = [], None, "unknown"
    logs = sorted(glob.glob(os.path.join(_LOG_DIR, "aiva-*.log")), reverse=True)
    if logs:
        try:
            last_mtime = os.path.getmtime(logs[0])
            with open(logs[0], encoding="utf-8", errors="replace") as f:
                lines = [ln.rstrip("\n") for ln in f if ln.strip()]
            recent = lines[-14:]
            # last real sleep/wake transition wins (ignore rejected/candidate noise)
            for ln in reversed(lines):
                low = ln.lower()
                if "asleep (" in low or "going to sleep" in low:
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


def _collect_cat():
    c = _load_json(_CAT_STATE)
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


def _read_mood():
    """Current mood dials from inner_state.json, decayed forward to now (so the dashboard
    shows what she feels *right now*, not the last-saved value). None until it exists."""
    data = _load_json(_INNER_STATE)
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


def _collect_inner(con):
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
    return {"mood": _read_mood(), "events": events, "events_total": total,
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
    traits = _q(con, "SELECT text, kind, strength, status, source_book_id, adopted_ts, "
                     "announced, veto_reason FROM traits ORDER BY status='active' DESC, strength DESC")
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
    r = data.get("reading", {})
    return {
        "traits": _one(con, "SELECT COUNT(*) FROM traits WHERE status='active'") or 0,
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


# --- the page (self-contained; polls /api/data) -------------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aiva Dashboard</title>
<style>
:root{
  --bg:#0e0b16; --bg2:#161222; --panel:#1b1630; --panel2:#221b3d; --line:#2e2748;
  --ink:#ece8fb; --muted:#a79fce; --dim:#786f9e;
  --accent:#c084fc; --accent2:#f0abfc; --good:#4ade80; --warn:#fbbf24; --bad:#fb7185;
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
.dot{width:9px;height:9px;border-radius:50%;background:var(--dim);box-shadow:0 0 0 0 rgba(0,0,0,0)}
.dot.live{background:var(--good);animation:pulse 2s infinite}
.dot.asleep{background:var(--warn)}
.dot.offline{background:var(--bad)}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(74,222,128,.5)}70%{box-shadow:0 0 0 7px rgba(74,222,128,0)}100%{box-shadow:0 0 0 0 rgba(74,222,128,0)}}
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
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">✦</div>
    <div>
      <h1>Aiva</h1>
      <div class="sub" id="subtitle">loading…</div>
    </div>
    <div class="spacer"></div>
    <div class="status"><span class="dot" id="dot"></span><span id="statusText">connecting…</span></div>
  </header>
  <div id="banner"></div>
  <div class="grid stats" id="stats"></div>
  <div id="content"></div>
  <footer>Reads her databases <b>read-only</b> — safe to keep open. Polls every """ + str(POLL_SECS) + r"""s ·
    <span id="stamp"></span></footer>
</div>
<script>
const POLL = """ + str(POLL_SECS) + r""" * 1000;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s==null?"":s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const bar = (pct) => `<div class="bar"><i style="width:${Math.max(0,Math.min(100,pct))}%"></i></div>`;
const strength = (s) => `<span class="strength"><i style="width:${Math.round((s||0)*100)}%"></i></span>`;

function statCard(n, label, sub){
  return `<div class="stat"><div class="n">${esc(n)}</div><div class="l">${esc(label)}</div>${sub?`<div class="s">${esc(sub)}</div>`:''}</div>`;
}
function pill(status){ return `<span class="pill ${esc(status)}">${esc(status)}</span>`; }

function render(d){
  // status line
  const p = d.presence || {};
  const dot = $('dot'), st = $('statusText');
  dot.className = 'dot ' + (p.state==='awake'?'live':p.state==='asleep'?'asleep':p.state==='offline'?'offline':'');
  const bits = [];
  if(p.state) bits.push(p.state);
  if(p.avatar_running) bits.push('body up (pid '+esc(p.avatar_pid)+')'); else bits.push('body down');
  if(p.last_activity) bits.push('active '+esc(p.last_activity));
  st.textContent = bits.join(' · ');
  $('subtitle').textContent = 'v' + (d.version||'?') + ' · voice companion · ' + (d.capabilities?`${d.capabilities.tool_count} tools · ${d.capabilities.module_count} modules`:'');

  $('banner').innerHTML = d.error ? `<div class="banner">${esc(d.error)}</div>`
    : (!d.db_present ? `<div class="banner">No memory.db yet — she hasn't stored anything. Boot her once.</div>` : '');

  // headline stats
  const h = d.headline||{}, r = d.reading||{};
  const mood = (d.inner&&d.inner.mood)||null;
  $('stats').innerHTML = [
    `<div class="stat"><div class="n" style="font-size:19px;line-height:1.25">${esc(mood?(mood.mood_word||'—'):'—')}</div><div class="l">Mood</div><div class="s">${esc(mood?(mood.focus?('on: '+mood.focus):'resting'):'not live yet')}</div></div>`,
    statCard(h.traits??0,'Traits', (h.voice_notes?`+${h.voice_notes} voice`:'')+(h.lessons?` · ${h.lessons} lessons`:'')),
    statCard(`${h.books_finished??0}/${h.books_total??0}`,'Books', `${h.sections_read??0} sections studied`),
    statCard(h.tasks??0,'Tasks', h.flows?`${h.flows} flows`:'no flows yet'),
    statCard(h.facts??0,'Facts', 'remembered'),
    statCard(h.sessions??0,'Sessions', `${h.messages??0} messages`),
    statCard('$'+(r.total_cost??0).toFixed(2),'Reading spend', 'lifetime'),
  ].join('');

  let html = '';

  // presence + cat
  html += `<div class="cols">`;
  html += `<section><h2>Presence</h2>`;
  html += rowKV('State', p.state||'unknown');
  html += rowKV('Avatar body', p.avatar_running?`running (pid ${esc(p.avatar_pid)})`:'down');
  html += rowKV('Monitor / layer', `${p.monitor??'?'} · ${esc(p.layer||'?')}`);
  html += rowKV('Last activity', p.last_activity||'—');
  html += rowKV('Reading permission', (d.becoming&&d.becoming.permission)||(r.permission)||'—');
  html += `</section>`;
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
    html += `<div class="empty">No inner state yet — she's wired for it; the dials light up the first time you fully restart her (Ctrl+Alt+X).</div>`;
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
    html += `<div class="empty" style="margin-top:14px">No experience logged yet — reading and time-of-day feed this stream once she's restarted; it fills as she lives.</div>`;
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
    html += `<div class="empty">No preferences yet — her baseline tastes seed the first time you fully restart her (Ctrl+Alt+X), then they genuinely evolve from what she reads, does and talks about.</div>`;
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
  html += `<section><h2>Who she's becoming <span class="cnt">traits & values</span></h2>`;
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
  $('stamp').textContent = 'updated ' + new Date().toLocaleTimeString();
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
  try{
    const res = await fetch('/api/data', {cache:'no-store'});
    const d = await res.json();
    render(d);
  }catch(e){
    $('statusText').textContent = 'dashboard server not reachable';
    $('dot').className = 'dot offline';
  }
}
tick();
setInterval(tick, POLL);
</script>
</body>
</html>"""


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

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/data":
            try:
                payload = json.dumps(collect(), ensure_ascii=False, default=str)
            except Exception as e:  # never let one bad query kill the feed
                payload = json.dumps({"error": f"collect failed: {e}",
                                      "generated_ts": datetime.now().isoformat()})
            self._send(200, payload, "application/json; charset=utf-8")
        elif path == "/healthz":
            self._send(200, "ok", "text/plain")
        else:
            self._send(404, "not found", "text/plain")


def main():
    if "--once" in sys.argv:  # print one JSON snapshot and exit (handy for a quick check)
        print(json.dumps(collect(), ensure_ascii=False, indent=2, default=str))
        return 0
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"Aiva dashboard on {url}   (Ctrl+C to stop)")
    print(f"  reading {_DB} read-only; polling every {POLL_SECS}s")
    if "--open" in sys.argv:
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
