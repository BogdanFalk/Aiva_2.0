"""Offline tests for web.WebClient: search normalization, page extraction fallback, and the
security-critical download gate (content-type allow-list, size cap, ext-from-content-type).
The HTTP layer is injected as fakes — no network."""
import pytest

from modules.web import WebClient, WebConfig


class FakeResp:
    def __init__(self, *, headers=None, json_data=None, chunks=None, text="", status=200):
        self.headers = headers or {}
        self._json = json_data
        self._chunks = chunks or []
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json

    def iter_content(self, chunk_size=65536):
        for c in self._chunks:
            yield c


def make_client(*, get=None, post=None, api_key="k", **cfg_kw):
    cfg = WebConfig(api_key=api_key, **cfg_kw)
    return WebClient(cfg, get=get, post=post, log=lambda m: None)


# --- search --------------------------------------------------------------------

def test_search_normalizes_results_and_images():
    def fake_post(url, **kw):
        return FakeResp(json_data={
            "results": [{"title": "Cute hairclips", "url": "https://shop.example.com/1",
                         "content": "star-shaped enamel clips"}],
            "images": ["https://img.example.com/a.jpg",
                       {"url": "https://img.example.com/b.png", "description": "blue clip"}]})
    c = make_client(post=fake_post)
    out = c.search("enamel hairclip", kind="images")
    assert out["results"][0]["url"].startswith("https://shop")
    assert out["images"][0] == {"url": "https://img.example.com/a.jpg", "description": ""}
    assert out["images"][1]["description"] == "blue clip"


def test_search_without_key_errs():
    c = make_client(api_key="")
    assert c.search("anything")["error"]
    assert not c.available()


# --- extract (fallback path) ---------------------------------------------------

def test_extract_falls_back_to_html_strip_without_key():
    def fake_get(url, **kw):
        return FakeResp(headers={"Content-Type": "text/html"},
                        text="<html><body><script>x()</script><p>Hello <b>world</b></p></body></html>")
    c = make_client(api_key="", get=fake_get)   # no key → tavily skipped → get fallback
    out = c.extract("https://example.com")
    assert out["text"] == "Hello world"          # script stripped, tags removed, whitespace collapsed


def test_extract_rejects_non_url():
    assert make_client().extract("not a url")["error"]


# --- download (the security surface) -------------------------------------------

def test_download_accepts_image_and_hashes(tmp_path):
    def fake_get(url, **kw):
        return FakeResp(headers={"Content-Type": "image/png"}, chunks=[b"\x89PNG", b"data"])
    c = make_client(get=fake_get)
    res = c.download("https://img.example.com/clip", str(tmp_path), base_name="blue hairclip")
    assert res["ok"] and res["content_type"] == "image/png"
    assert res["file_name"].endswith(".png") and res["file_name"].startswith("blue_hairclip")
    assert res["sha256"] and (tmp_path / res["file_name"]).exists()


def test_download_rejects_disallowed_content_type(tmp_path):
    def fake_get(url, **kw):
        return FakeResp(headers={"Content-Type": "application/x-msdownload"}, chunks=[b"MZ..."])
    res = make_client(get=fake_get).download("https://evil.example.com/x.exe", str(tmp_path))
    assert not res["ok"] and "blocked type" in res["error"]
    assert list(tmp_path.iterdir()) == []        # nothing written


def test_download_rejects_oversize_by_content_length(tmp_path):
    def fake_get(url, **kw):
        return FakeResp(headers={"Content-Type": "image/jpeg", "Content-Length": "999999"},
                        chunks=[b"x"])
    c = make_client(get=fake_get, max_bytes=1000)
    res = c.download("https://img.example.com/huge.jpg", str(tmp_path))
    assert not res["ok"] and res["error"] == "too big"


def test_download_rejects_oversize_while_streaming(tmp_path):
    def fake_get(url, **kw):                      # no Content-Length; body exceeds the cap
        return FakeResp(headers={"Content-Type": "image/jpeg"}, chunks=[b"x" * 600, b"x" * 600])
    c = make_client(get=fake_get, max_bytes=1000)
    res = c.download("https://img.example.com/huge.jpg", str(tmp_path))
    assert not res["ok"] and res["error"] == "too big"


def test_download_ext_comes_from_content_type_not_url(tmp_path):
    # url pretends to be an .exe but the server sends a png -> written as .png, never executable
    def fake_get(url, **kw):
        return FakeResp(headers={"Content-Type": "image/png"}, chunks=[b"img"])
    res = make_client(get=fake_get).download("https://x.example.com/trojan.exe", str(tmp_path))
    assert res["ok"] and res["file_name"].endswith(".png")
    assert not res["file_name"].endswith(".exe")
