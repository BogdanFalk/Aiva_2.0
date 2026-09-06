"""Push a short alert to the user off-device via ntfy (ntfy.sh or a self-hosted server).

Used so a spend/usage cap can never be hit silently: the limiter fires one of these when a service
nears or reaches its budget. Deliberately tiny and dependency-injected (a `post` callable, defaulting
to `requests.post`, swapped for a fake in tests). No-op when unconfigured, so nothing breaks if the
user hasn't set a topic. Blocking POST is offloaded via `notify_async` so it never stalls the loop.

Config: NTFY_TOPIC (required to be active), NTFY_URL (default https://ntfy.sh), AIVA_NTFY=1.
"""

import asyncio
import os
from dataclasses import dataclass

import requests


@dataclass
class NotifyConfig:
    enabled: bool = True
    base_url: str = "https://ntfy.sh"
    topic: str = ""

    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.getenv("AIVA_NTFY", "1") == "1",
            base_url=(os.getenv("NTFY_URL", "https://ntfy.sh") or "https://ntfy.sh").rstrip("/"),
            topic=os.getenv("NTFY_TOPIC", ""))


class Notifier:
    def __init__(self, cfg: NotifyConfig, *, post=None, log=print):
        self.cfg = cfg
        self._post = post or requests.post
        self.log = log

    def available(self):
        return bool(self.cfg.enabled and self.cfg.topic)

    def notify(self, message, *, title=None, tags=None, priority=None):
        """Blocking POST to ntfy. Returns True if sent. Title stays ASCII (ntfy header); use `tags`
        (ntfy emoji shortcodes like 'warning','money_with_wings') for icons, and the UTF-8 body for text."""
        if not self.available():
            return False
        headers = {}
        if title:
            headers["Title"] = str(title).encode("ascii", "ignore").decode() or "Aiva"
        if tags:
            headers["Tags"] = ",".join(tags) if isinstance(tags, (list, tuple)) else str(tags)
        if priority:
            headers["Priority"] = str(priority)
        try:
            self._post(f"{self.cfg.base_url}/{self.cfg.topic}",
                       data=(message or "").encode("utf-8"), headers=headers, timeout=10)
            return True
        except Exception as e:
            self.log(f"ntfy notify failed: {e!r}")
            return False

    async def notify_async(self, message, **kw):
        return await asyncio.to_thread(self.notify, message, **kw)
