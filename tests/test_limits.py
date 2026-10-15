"""Offline tests for ServiceLimits: per-service counting, cap gate, warn/cap alerts (once), pausing
vs alert-only services, monthly reset, and the dashboard reports. Fake notifier + fake clock."""
import sqlite3
from datetime import datetime

import pytest

from modules.limits import SERVICE_LABEL, ServiceLimits


class FakeNotifier:
    def __init__(self):
        self.msgs = []

    async def notify_async(self, message, **kw):
        self.msgs.append(message)
        return True


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t.isoformat()


def make(caps, t=datetime(2026, 9, 6, 12, 0, 0)):
    clock = Clock(t)
    n = FakeNotifier()
    L = ServiceLimits(sqlite3.connect(":memory:"), notifier=n, caps=caps, now=clock)
    return L, n, clock


async def test_record_increments_and_allow_flips_at_cap():
    L, n, _ = make({"tavily": 5})
    for _ in range(4):
        await L.record("tavily")
    assert L.usage("tavily") == 4 and L.allow("tavily") is True and L.is_paused("tavily") is False
    await L.record("tavily")                      # hits 5
    assert L.allow("tavily") is False and L.is_paused("tavily") is True


async def test_warn_then_cap_alert_fire_once_each():
    L, n, _ = make({"tavily": 10})
    for _ in range(8):
        await L.record("tavily")                  # 8th crosses 80%
    assert len(n.msgs) == 1 and "80%" in n.msgs[0] or "8/10" in n.msgs[0]
    for _ in range(2):
        await L.record("tavily")                  # hits 10
    assert len(n.msgs) == 2                        # warn once + cap once
    for _ in range(3):
        await L.record("tavily")                  # over cap, no new alerts
    assert len(n.msgs) == 2


async def test_alert_only_service_never_pauses():
    L, n, _ = make({"tts": 5})
    for _ in range(6):
        await L.record("tts")
    assert L.is_paused("tts") is False            # pausing tts would mute her
    assert any("cap" in m.lower() for m in n.msgs)  # but it did alert


async def test_period_reset_clears_usage_and_pause():
    L, n, clock = make({"tavily": 2})
    await L.record("tavily")
    await L.record("tavily")
    assert L.is_paused("tavily") is True
    clock.t = datetime(2026, 10, 1, 9, 0, 0)      # new month
    assert L.usage("tavily") == 0 and L.allow("tavily") is True and L.is_paused("tavily") is False


async def test_cap_zero_is_unlimited_and_silent():
    L, n, _ = make({"tavily": 0})
    for _ in range(50):
        await L.record("tavily")
    assert L.allow("tavily") is True and L.is_paused("tavily") is False and n.msgs == []


async def test_paused_reasons_and_usage_report():
    L, n, _ = make({"tavily": 1, "tts": 100})
    await L.record("tavily")
    pr = L.paused_reasons()
    assert len(pr) == 1 and pr[0]["service"] == "tavily" and pr[0]["resets"]
    rep = {r["service"]: r for r in L.usage_report()}
    assert rep["tavily"]["paused"] is True and rep["tavily"]["cap"] == 1
    assert rep["tts"]["paused"] is False and rep["tts"]["used"] == 0


async def test_per_character_tts_counters_count_silently_and_stay_off_the_cap_report():
    """tts.<slug> has no cap of its own: it is counted (for the dashboard's per-sister chars), never
    alerts or pauses, and — like every uncapped service today — is absent from usage_report. The
    capped, alerting aggregate is still plain 'tts'."""
    L, n, _ = make({"tavily": 5, "tts": 1000})
    await L.record("tts.saga", 250)
    r = await L.record("tts.saga", 250)
    assert r == {"used": 500, "cap": 0, "paused": False}
    assert L.usage("tts.saga") == 500 and L.usage("tts.aiva") == 0
    assert L.allow("tts.saga") is True and L.is_paused("tts.saga") is False and n.msgs == []
    assert "tts.saga" not in {row["service"] for row in L.usage_report()}
    assert SERVICE_LABEL["tts.aiva"] == "Voice — Aiva" and SERVICE_LABEL["tts.saga"] == "Voice — Saga"
    await L.record("tts", 1000)                   # the aggregate still alerts at its cap
    assert any("cap" in m.lower() for m in n.msgs) and L.is_paused("tts") is False
