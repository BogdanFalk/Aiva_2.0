"""Her projects — v3: long-running, outward things she chose to pursue (see docs/PLAN-v3.md).

Doing builds identity faster than reading: a wish that keeps coming back ("find photos of
serviceberries in bloom", "learn how sourdough starters work") and that she can actually pursue on
her own becomes a PROJECT — worked on in quiet moments over days, felt as progress (drives:
competence/novelty), and shared with him when it's done. Two kinds, both inside her safe envelope
(capabilities auto-tier: web search + keeping finds in her space — nothing that spends, sends or
touches his files):

  - collection — gather N finds on a theme into her space (an image per step)
  - learn      — look something up in a few passes, keeping what she learned in the project notes

PURE helpers (`project_kind`, `step_due`) + `ProjectBook` on the shared connection. The doing itself
(web + space) is injected from main.py, like SelfEngine._act.
"""

import json
import re
from datetime import datetime

MAX_ACTIVE = 3
PROMOTE_STRENGTH = 0.75           # a wish reinforced at least twice in the diary
TARGET = {"collection": 5, "learn": 3}
STEP_GAP_H = 6.0                  # at most one step per project every 6 hours
MAX_STEPS_PER_DAY = 3
STEP_IDLE_S = 10 * 60
QUIET_HOURS = (1, 8)

_COLLECT = re.compile(r"\b(find|collect|gather|keep|save|look for|hunt for)\b.*\b(photos?|pictures?|images?|"
                      r"pics?|art|drawings?|paintings?|examples?|collection)\b", re.I)
_LEARN = re.compile(r"\b(learn|find out|understand|figure out|read up|look into|discover)\b", re.I)
_NOT_PROJECT = re.compile(r"\b(him|his|he|ask|tell|show|buy|order|send|email|install|code|build)\b", re.I)


def project_kind(text):
    """PURE: 'collection' | 'learn' | None — only wishes she can pursue alone, safely."""
    t = text or ""
    if _NOT_PROJECT.search(t):
        return None
    if _COLLECT.search(t):
        return "collection"
    if _LEARN.search(t):
        return "learn"
    return None


def project_title(text):
    """'I want to find photos of serviceberries in bloom.' → 'photos of serviceberries in bloom'."""
    t = re.sub(r"^\s*i\s+(want|would like|'d like|wish)\s+to\s+", "", (text or "").strip(), flags=re.I)
    t = re.sub(r"^(find|collect|gather|keep|save|look for|hunt for|learn( about)?|find out( about)?|"
               r"understand|figure out|read up on|look into|discover)\s+", "", t, flags=re.I)
    return t.rstrip(" .!").strip()[:80] or (text or "")[:80]


def step_due(ctx):
    """PURE: should she work on a project now? ctx: hour, busy, present, voice_idle_s,
    steps_today, hours_since_step, novelty, competence, allowed (web budget)."""
    lo, hi = QUIET_HOURS
    if ctx.get("busy") or not ctx.get("allowed", True):
        return False
    if lo <= ctx.get("hour", 12) < hi:
        return False
    if ctx.get("steps_today", 0) >= MAX_STEPS_PER_DAY:
        return False
    if ctx.get("hours_since_step", 1e9) < STEP_GAP_H:
        return False
    if ctx.get("voice_idle_s", 0) < STEP_IDLE_S:
        return False
    # she works on them because she wants to: restless, or wanting to be useful
    return ctx.get("novelty", 1.0) < 0.6 or ctx.get("competence", 1.0) < 0.6


class ProjectBook:
    def __init__(self, conn, *, log=print, now=None):
        self.conn = conn
        self.log = log
        self._now = now or (lambda: datetime.now().isoformat())
        self._refresh_prompt = lambda: None
        conn.execute("""CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, kind TEXT NOT NULL, why TEXT,
            status TEXT NOT NULL DEFAULT 'active', steps INTEGER NOT NULL DEFAULT 0,
            target INTEGER NOT NULL, created_ts TEXT NOT NULL, last_step_ts TEXT,
            notes_json TEXT NOT NULL DEFAULT '[]', desire_id INTEGER)""")
        conn.commit()

    def attach(self, *, refresh_prompt=None):
        if refresh_prompt:
            self._refresh_prompt = refresh_prompt

    def _refresh(self):
        try:
            self._refresh_prompt()
        except Exception:
            pass

    def active(self):
        rows = self.conn.execute(
            "SELECT id, title, kind, why, steps, target, last_step_ts, notes_json FROM projects "
            "WHERE status = 'active' ORDER BY COALESCE(last_step_ts, created_ts)").fetchall()
        return [{"id": r[0], "title": r[1], "kind": r[2], "why": r[3], "steps": r[4], "target": r[5],
                 "last_step_ts": r[6], "notes": json.loads(r[7] or "[]")} for r in rows]

    def all(self, limit=20):
        rows = self.conn.execute(
            "SELECT id, title, kind, status, steps, target, created_ts, last_step_ts FROM projects "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "title": r[1], "kind": r[2], "status": r[3], "steps": r[4], "target": r[5],
                 "created_ts": r[6], "last_step_ts": r[7]} for r in rows]

    def promote(self, desires):
        """Open wishes strong enough and pursuable → projects (the desire becomes status 'project').
        `desires` = SelfEngine.list_desires('open'). Returns the new projects' titles."""
        made = []
        for d in desires:
            if len(self.active()) >= MAX_ACTIVE:
                break
            if d.get("strength", 0) < PROMOTE_STRENGTH:
                continue
            kind = project_kind(d.get("text", ""))
            if not kind:
                continue
            title = project_title(d["text"])
            if self.conn.execute("SELECT 1 FROM projects WHERE lower(title) = lower(?)", (title,)).fetchone():
                continue
            self.conn.execute(
                "INSERT INTO projects (title, kind, why, target, created_ts, desire_id) VALUES (?,?,?,?,?,?)",
                (title, kind, d["text"][:200], TARGET[kind], self._now(), d.get("id")))
            if d.get("id") is not None:
                self.conn.execute("UPDATE desires SET status = 'project', last_touched_ts = ? WHERE id = ?",
                                  (self._now(), d["id"]))
            self.conn.commit()
            self.log(f"project started: {kind} — {title!r}")
            made.append(title)
        if made:
            self._refresh()
        return made

    def next_project(self):
        a = self.active()
        return a[0] if a else None

    def hours_since_step(self):
        r = self.conn.execute("SELECT MAX(last_step_ts) FROM projects").fetchone()
        if not r or not r[0]:
            return 1e9
        return (datetime.fromisoformat(self._now()) - datetime.fromisoformat(r[0])).total_seconds() / 3600

    def steps_today(self):
        return self.conn.execute(
            "SELECT COALESCE(SUM(1), 0) FROM projects WHERE substr(last_step_ts, 1, 10) = ?",
            (self._now()[:10],)).fetchone()[0]

    def record_step(self, pid, note=None):
        """One unit of progress. Returns 'step' | 'done'."""
        r = self.conn.execute("SELECT steps, target, notes_json FROM projects WHERE id = ?", (pid,)).fetchone()
        if not r:
            return None
        steps, target = r[0] + 1, r[1]
        notes = json.loads(r[2] or "[]")
        if note:
            notes.append(str(note)[:400])
        status = "done" if steps >= target else "active"
        self.conn.execute("UPDATE projects SET steps = ?, status = ?, last_step_ts = ?, notes_json = ? "
                          "WHERE id = ?", (steps, status, self._now(), json.dumps(notes[-10:]), pid))
        self.conn.commit()
        self._refresh()
        return status if status == "done" else "step"

    def touch(self, pid):
        """A step that found nothing still counts against the gap (no hammering the web)."""
        self.conn.execute("UPDATE projects SET last_step_ts = ? WHERE id = ?", (self._now(), pid))
        self.conn.commit()

    def drop(self, text_or_id):
        q = str(text_or_id or "").strip().lower()
        for rid, title in self.conn.execute("SELECT id, title FROM projects WHERE status = 'active'").fetchall():
            if q == str(rid) or (q and q in title.lower()):
                self.conn.execute("UPDATE projects SET status = 'dropped' WHERE id = ?", (rid,))
                self.conn.commit()
                self._refresh()
                return title
        return None

    def prompt_lines(self):
        out = []
        for p in self.active():
            what = "found" if p["kind"] == "collection" else "looked into"
            out.append(f"{p['title']} ({p['kind']}: {what} {p['steps']} of {p['target']} so far)")
        return out
