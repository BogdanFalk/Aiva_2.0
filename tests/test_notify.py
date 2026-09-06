"""Offline tests for the ntfy Notifier: no-op when unconfigured, correct POST shape, injected poster."""
from modules.notify import Notifier, NotifyConfig


class FakePost:
    def __init__(self):
        self.calls = []

    def __call__(self, url, data=None, headers=None, timeout=None):
        self.calls.append({"url": url, "data": data, "headers": headers or {}})
        class R:
            def raise_for_status(self_):
                pass
        return R()


def test_noop_without_topic():
    fp = FakePost()
    n = Notifier(NotifyConfig(topic=""), post=fp)
    assert n.available() is False
    assert n.notify("hi") is False
    assert fp.calls == []


def test_disabled_flag_noops():
    assert Notifier(NotifyConfig(enabled=False, topic="t"), post=FakePost()).notify("x") is False


def test_posts_body_title_and_tags():
    fp = FakePost()
    n = Notifier(NotifyConfig(base_url="https://ntfy.sh", topic="aiva-frank"), post=fp)
    assert n.available() and n.notify("body text", title="Heads up", tags=["warning"]) is True
    c = fp.calls[0]
    assert c["url"] == "https://ntfy.sh/aiva-frank"
    assert c["data"] == "body text".encode("utf-8")
    assert c["headers"]["Title"] == "Heads up" and c["headers"]["Tags"] == "warning"


def test_title_forced_ascii():
    fp = FakePost()
    Notifier(NotifyConfig(topic="t"), post=fp).notify("b", title="cap ⚠ reached")
    assert all(ord(ch) < 128 for ch in fp.calls[0]["headers"]["Title"])


async def test_notify_async_offloads():
    fp = FakePost()
    n = Notifier(NotifyConfig(topic="t"), post=fp)
    assert await n.notify_async("y") is True and fp.calls
