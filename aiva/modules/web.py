"""Aiva's reach to the internet — search, read a page, and safely download a file.

Provider-agnostic and dependency-injected so it's fully testable offline: the HTTP calls go through
`get`/`post` callables that default to `requests` (already a dependency; see utilities.get_weather)
but are swapped for fakes in tests. Every method is SYNCHRONOUS/blocking on purpose — the tool
handlers wrap them in `asyncio.to_thread` so the voice pipeline never stalls (same pattern main.py
uses for execute_shell).

The download path is the security surface, so it is deliberately strict: only image / text / PDF /
mp4 content-types, a hard size cap, a timeout, sanitized filenames, and the extension is derived from
the CONTENT-TYPE (never the URL) so an executable can never be written. Nothing here ever executes a
file. She finds and fetches; she never buys, signs in, or sends.

First provider: Tavily (search + extract). Add others behind the same `WebClient` interface.
"""

import hashlib
import os
import re
from dataclasses import dataclass

import requests

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
_UA = "Aiva/1.0 (+local companion)"

# download safety
_ALLOWED_EXACT = {"application/pdf", "video/mp4"}
_ALLOWED_PREFIX = ("image/", "text/")
_EXT = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/gif": ".gif",
        "image/webp": ".webp", "image/bmp": ".bmp", "image/svg+xml": ".svg", "video/mp4": ".mp4",
        "application/pdf": ".pdf", "text/plain": ".txt", "text/html": ".html",
        "text/markdown": ".md", "text/csv": ".csv"}


def _content_type_ok(ctype):
    ctype = (ctype or "").split(";")[0].strip().lower()
    return ctype in _ALLOWED_EXACT or any(ctype.startswith(p) for p in _ALLOWED_PREFIX)


def _ext_for(ctype, url=""):
    ctype = (ctype or "").split(";")[0].strip().lower()
    if ctype in _EXT:
        return _EXT[ctype]
    # image/anything -> .img is safe; otherwise a generic, never an executable
    if ctype.startswith("image/"):
        return ".img"
    if ctype.startswith("text/"):
        return ".txt"
    return ".bin"


def _safe_name(name, ext):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "find").strip()) or "find"
    stem = stem[:60].rstrip("._-") or "find"
    if stem.lower().endswith(ext.lower()):
        stem = stem[: -len(ext)]
    return stem + ext


def _strip_html(html):
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html or "")
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class WebConfig:
    provider: str = "tavily"
    api_key: str = ""
    enabled: bool = True
    timeout: float = 15.0
    max_bytes: int = 25 * 1024 * 1024
    max_results: int = 6

    @classmethod
    def from_env(cls):
        max_mb = float(os.getenv("AIVA_WEB_MAX_MB", "25"))
        return cls(
            provider=os.getenv("AIVA_WEB_PROVIDER", "tavily").lower(),
            api_key=os.getenv("TAVILY_API_KEY", "") or os.getenv("SEARCH_API_KEY", ""),
            enabled=os.getenv("AIVA_WEB_SEARCH", "1") == "1",
            timeout=float(os.getenv("AIVA_WEB_TIMEOUT", "15")),
            max_bytes=int(max_mb * 1024 * 1024),
            max_results=int(os.getenv("AIVA_WEB_MAX_RESULTS", "6")))


class WebClient:
    def __init__(self, cfg: WebConfig, *, get=None, post=None, log=print):
        self.cfg = cfg
        self._get = get or requests.get
        self._post = post or requests.post
        self.log = log

    def available(self):
        return bool(self.cfg.enabled and self.cfg.api_key)

    # --- search --------------------------------------------------------------------
    def search(self, query, kind="info", k=None):
        """Return {"results": [{title,url,snippet}], "images": [{url,description}]} or {"error": ...}."""
        query = (query or "").strip()
        if not query:
            return {"error": "empty query"}
        if not self.available():
            return {"error": "no web search configured"}
        k = k or self.cfg.max_results
        want_images = kind == "images"
        try:
            payload = {"api_key": self.cfg.api_key, "query": query,
                       "max_results": k, "search_depth": "basic",
                       "include_images": want_images}
            resp = self._post(TAVILY_SEARCH_URL, json=payload, timeout=self.cfg.timeout,
                              headers={"User-Agent": _UA})
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            self.log(f"web.search failed: {e!r}")
            return {"error": "search failed"}
        results = [{"title": r.get("title") or "", "url": r.get("url") or "",
                    "snippet": (r.get("content") or "")[:400]}
                   for r in (data.get("results") or []) if r.get("url")]
        images = []
        for im in (data.get("images") or []):
            if isinstance(im, str):
                images.append({"url": im, "description": ""})
            elif isinstance(im, dict) and im.get("url"):
                images.append({"url": im["url"], "description": im.get("description") or ""})
        return {"query": query, "kind": kind, "results": results, "images": images}

    # --- read a page ---------------------------------------------------------------
    def extract(self, url):
        """Clean readable text of a page, or {"error": ...}. Tavily /extract, else fetch+strip."""
        url = (url or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            return {"error": "not a web url"}
        if self.available() and self.cfg.provider == "tavily":
            try:
                resp = self._post(TAVILY_EXTRACT_URL,
                                  json={"api_key": self.cfg.api_key, "urls": [url]},
                                  timeout=self.cfg.timeout, headers={"User-Agent": _UA})
                resp.raise_for_status()
                data = resp.json()
                parts = [r.get("raw_content") or "" for r in (data.get("results") or [])]
                text = "\n\n".join(p for p in parts if p).strip()
                if text:
                    return {"url": url, "text": text}
            except Exception as e:
                self.log(f"web.extract (tavily) failed, falling back: {e!r}")
        try:
            resp = self._get(url, timeout=self.cfg.timeout, headers={"User-Agent": _UA})
            resp.raise_for_status()
            ctype = (resp.headers.get("Content-Type") or "").lower()
            body = resp.text if hasattr(resp, "text") else ""
            text = body if ctype.startswith("text/plain") else _strip_html(body)
            return {"url": url, "text": text[:20000]} if text else {"error": "nothing to read there"}
        except Exception as e:
            self.log(f"web.extract failed: {e!r}")
            return {"error": "couldn't read that page"}

    # --- download (the security surface) -------------------------------------------
    def download(self, url, dest_dir, base_name=None):
        """Stream a file to dest_dir under strict content-type + size limits.
        Returns {"ok": True, "path", "file_name", "sha256", "content_type", "bytes"} or
        {"ok": False, "error": <reason>}."""
        url = (url or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            return {"ok": False, "error": "not a web url"}
        try:
            resp = self._get(url, stream=True, timeout=self.cfg.timeout,
                             headers={"User-Agent": _UA})
            resp.raise_for_status()
        except Exception as e:
            self.log(f"web.download request failed: {e!r}")
            return {"ok": False, "error": "download failed"}
        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not _content_type_ok(ctype):
            return {"ok": False, "error": f"blocked type {ctype or 'unknown'}"}
        clen = resp.headers.get("Content-Length")
        if clen:
            try:
                if int(clen) > self.cfg.max_bytes:
                    return {"ok": False, "error": "too big"}
            except ValueError:
                pass
        buf = bytearray()
        try:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                buf.extend(chunk)
                if len(buf) > self.cfg.max_bytes:
                    return {"ok": False, "error": "too big"}
        except Exception as e:
            self.log(f"web.download stream failed: {e!r}")
            return {"ok": False, "error": "download interrupted"}
        if not buf:
            return {"ok": False, "error": "empty file"}
        ext = _ext_for(ctype, url)
        name = base_name or os.path.basename(url.split("?")[0]) or "find"
        file_name = _safe_name(name, ext)
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, file_name)
        i = 1
        while os.path.exists(path):                      # never clobber an existing find
            path = os.path.join(dest_dir, _safe_name(f"{name}-{i}", ext))
            i += 1
        with open(path, "wb") as f:
            f.write(buf)
        return {"ok": True, "path": path, "file_name": os.path.basename(path),
                "sha256": hashlib.sha256(buf).hexdigest(), "content_type": ctype, "bytes": len(buf)}
