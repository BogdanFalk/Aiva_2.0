"""Per-service usage caps, so Aiva can't quietly run up a bill.

By design (user decision) there is NO global dollar budget — each metered service is capped at its own
budget / free tier, and when one is exhausted she pauses only the SELF-STARTED work that depends on it
(the live voice never stops). Every near-cap (~80%) and cap (100%) fires an ntfy alert, and the pause
is visible on the dashboard with its reason, so nothing stops silently.

- 'tavily' (web search/finds) is a PAUSING service: at cap, her web finds pause until the month resets.
- 'tts'/'stt' are ALERT-ONLY: counted and alerted, never paused (pausing them would mute her).

Monthly counters key on period=YYYY-MM and auto-reset. State on Memory.conn (event-loop thread only).
`record` is async so the ntfy POST is awaited off the loop.
"""

import os
from datetime import datetime

WARN_FRAC = 0.8

PAUSING_SERVICES = {"tavily"}          # hitting the cap pauses the dependent self-started work
SERVICE_LABEL = {
    "tavily": "Web search & finds",
    "tts": "Voice (ElevenLabs)",
    "stt": "Listening (Deepgram)",
}


def _now():
    return datetime.now().isoformat()


def _period(now_iso):
    return (now_iso or _now())[:7]         # YYYY-MM


def _resets_ts(now_iso):
    d = datetime.fromisoformat(now_iso)
    return (datetime(d.year + 1, 1, 1) if d.month == 12
            else datetime(d.year, d.month + 1, 1)).isoformat()


def caps_from_env():
    """Each service's own budget. 0 = uncapped/off (TTS/STT are off unless a quota is set)."""
    return {
        "tavily": int(os.getenv("AIVA_TAVILY_MONTHLY_CAP", "1000")),
        "tts": int(os.getenv("AIVA_TTS_MONTHLY_CHARS", "0")),
        "stt": int(os.getenv("AIVA_STT_MONTHLY_MINUTES", "0")),
    }


class ServiceLimits:
    def __init__(self, conn, *, notifier=None, caps=None, log=print, now=None):
        self.conn = conn
        self._notifier = notifier
        self.caps = caps if caps is not None else caps_from_env()
        self.log = log
        self._now = now or _now
        self._create_tables()

    def _create_tables(self):
        c = self.conn
        c.execute("""CREATE TABLE IF NOT EXISTS service_usage (
            service TEXT NOT NULL, period TEXT NOT NULL, used REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (service, period))""")
        c.execute("""CREATE TABLE IF NOT EXISTS service_pause (
            service TEXT PRIMARY KEY, period TEXT NOT NULL, reason TEXT NOT NULL,
            since_ts TEXT NOT NULL, resets_ts TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS limits_kv (key TEXT PRIMARY KEY, value TEXT)""")
        c.commit()

    # -- usage --
    def cap(self, service):
        return int(self.caps.get(service, 0) or 0)

    def usage(self, service, now=None):
        p = _period(now or self._now())
        r = self.conn.execute("SELECT used FROM service_usage WHERE service=? AND period=?",
                              (service, p)).fetchone()
        return (r[0] if r else 0) or 0

    def allow(self, service, now=None):
        """Read-only gate: may she use this service right now? (cap 0 = always)."""
        cap = self.cap(service)
        return True if cap <= 0 else self.usage(service, now) < cap

    async def record(self, service, amount=1):
        """Charge `amount` against this month's usage; alert (once) at ~80% and 100%, and pause a
        PAUSING service at its cap. Returns {used, cap, paused}."""
        now = self._now()
        p = _period(now)
        self.conn.execute(
            "INSERT INTO service_usage (service, period, used) VALUES (?,?,?) "
            "ON CONFLICT(service, period) DO UPDATE SET used = used + excluded.used",
            (service, p, amount))
        self.conn.commit()
        used = self.usage(service, now)
        cap = self.cap(service)
        if cap > 0:
            label = SERVICE_LABEL.get(service, service)
            if used >= cap:
                if service in PAUSING_SERVICES:
                    self._pause(service, f"{label} hit its monthly cap ({cap})", now)
                await self._alert_once(
                    f"capped:{service}:{p}",
                    f"{label} hit its monthly cap ({int(used)}/{cap})."
                    + (" Her finds are paused until it resets." if service in PAUSING_SERVICES else ""),
                    title="Aiva: service cap reached", tags=["money_with_wings", "no_entry"])
            elif used >= WARN_FRAC * cap:
                await self._alert_once(
                    f"warn:{service}:{p}",
                    f"{label} is at {int(used)}/{cap} this month ({int(100 * used / cap)}%).",
                    title="Aiva: nearing a service cap", tags=["warning"])
        return {"used": used, "cap": cap, "paused": self.is_paused(service, now)}

    # -- pausing --
    def _pause(self, service, reason, now):
        self.conn.execute(
            "INSERT INTO service_pause (service, period, reason, since_ts, resets_ts) "
            "VALUES (?,?,?,?,?) ON CONFLICT(service) DO UPDATE SET period=excluded.period, "
            "reason=excluded.reason, since_ts=excluded.since_ts, resets_ts=excluded.resets_ts",
            (service, _period(now), reason, now, _resets_ts(now)))
        self.conn.commit()

    def is_paused(self, service, now=None):
        p = _period(now or self._now())
        r = self.conn.execute("SELECT period FROM service_pause WHERE service=?", (service,)).fetchone()
        return bool(r and r[0] == p)               # a stale-period pause is auto-expired

    def paused_reasons(self, now=None):
        p = _period(now or self._now())
        rows = self.conn.execute(
            "SELECT service, reason, resets_ts FROM service_pause WHERE period=?", (p,)).fetchall()
        return [{"service": s, "label": SERVICE_LABEL.get(s, s), "reason": reason,
                 "resets": (resets or "")[:10]} for s, reason, resets in rows]

    def usage_report(self, now=None):
        """For the dashboard: every configured service's usage vs cap + paused flag."""
        out = []
        for service, cap in self.caps.items():
            if cap and cap > 0:
                out.append({"service": service, "label": SERVICE_LABEL.get(service, service),
                            "used": int(self.usage(service, now)), "cap": int(cap),
                            "paused": self.is_paused(service, now)})
        return out

    # -- one-shot alerts (never spam the same threshold twice in a period) --
    async def _alert_once(self, key, message, *, title=None, tags=None):
        if self.conn.execute("SELECT 1 FROM limits_kv WHERE key=?", (key,)).fetchone():
            return
        self.conn.execute("INSERT OR REPLACE INTO limits_kv (key, value) VALUES (?, ?)",
                          (key, self._now()))
        self.conn.commit()
        self.log(f"limits: {message}")
        if self._notifier is not None:
            try:
                await self._notifier.notify_async(message, title=title or "Aiva", tags=tags)
            except Exception as e:
                self.log(f"limits alert failed: {e!r}")

    async def alert(self, message, *, key=None, title=None, tags=None):
        """A general one-shot alert other subsystems (task/book budget hits) can fire through ntfy."""
        await self._alert_once(key or f"alert:{message[:40]}:{_period(self._now())}",
                               message, title=title, tags=tags)
