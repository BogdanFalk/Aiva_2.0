"""Book files -> clean, ordered sections and retrieval chunks.

Pure functions, no sqlite, no asyncio: everything here is safe to run under
`asyncio.to_thread` while the event loop keeps talking. `reading.py` owns the
database side.

Formats: EPUB (stdlib zip + XML + html.parser), TXT/MD (decode chain with
Project Gutenberg header/footer stripping), PDF (pypdf, lazily imported;
running header/footer removal, de-hyphenation, outline -> chapters). Anything
we genuinely can't read raises `Unreadable` with a message fit to be spoken.

`clean_text` only normalises unicode form and whitespace — never the words —
so quotes Aiva copies from a section can be verified verbatim against it.
"""

import hashlib
import math
import os
import posixpath
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import unquote
import xml.etree.ElementTree as ET

CHARS_PER_TOKEN = 4.0
SUPPORTED = {".pdf", ".epub", ".txt", ".md"}
SECTION_MAX_TOKENS = 6000       # one close-reading call must stay <= ~6k tokens of text
SECTION_MIN_TOKENS = 120        # only genuine scraps merge; real short chapters stay whole
FALLBACK_SECTION_TOKENS = 3500  # when no headings can be found
CHUNK_TOKENS = 750
CHUNK_OVERLAP = 80
RECENT_WRITE_SECS = 5.0         # a file still being copied is skipped this scan


class Unreadable(Exception):
    """User-facing: the message is spoken as-is ('it's a scan with no text layer')."""


@dataclass
class FileInfo:
    path: str
    name: str
    size: int
    mtime: float
    sha256: str


@dataclass
class Section:
    title: str
    text: str
    kind: str = "chapter"       # chapter | part | front | back | split
    words: int = 0
    est_tokens: int = 0

    def __post_init__(self):
        self.text = self.text.strip("\n")
        if not self.words:
            self.words = count_words(self.text)
        if not self.est_tokens:
            self.est_tokens = est_tokens(self.text)


@dataclass
class ParsedBook:
    title: str | None
    author: str | None
    format: str
    sections: list
    words: int = 0
    est_tokens: int = 0
    meta: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def __post_init__(self):
        if not self.words:
            self.words = sum(s.words for s in self.sections)
        if not self.est_tokens:
            self.est_tokens = sum(s.est_tokens for s in self.sections)


# --- small helpers -------------------------------------------------------------------

def est_tokens(text):
    return int(math.ceil(len(text or "") / CHARS_PER_TOKEN))


def count_words(text):
    return len((text or "").split())


_SENT_SPLIT = re.compile(r'(?<=[.!?"”’)])\s+(?=[A-Z"“‘(])')


def split_sentences(text):
    parts = [p.strip() for p in _SENT_SPLIT.split(text or "") if p and p.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])


def clean_text(text):
    """NFC, drop soft hyphens / zero-width chars, normalise whitespace ONLY."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[­​‌‍﻿]", "", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan(library_dir, known=None, now=None):
    """Book files in the library folder (non-recursive). `known` maps path ->
    {"size", "mtime", "sha256"}; unchanged files reuse their hash so a boot
    scan with nothing new costs ~nothing. Files written in the last few
    seconds are skipped (still being copied)."""
    import time as _time
    now = now if now is not None else _time.time()
    known = known or {}
    out = []
    if not os.path.isdir(library_dir):
        return out
    for name in sorted(os.listdir(library_dir)):
        if name.startswith((".", "~$")):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in SUPPORTED:
            continue
        path = os.path.join(library_dir, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if not os.path.isfile(path) or st.st_size == 0:
            continue
        if now - st.st_mtime < RECENT_WRITE_SECS:
            continue
        prev = known.get(path) or known.get(os.path.normcase(path))
        if prev and prev.get("size") == st.st_size and abs(prev.get("mtime", -1) - st.st_mtime) < 1e-6:
            sha = prev["sha256"]
        else:
            sha = sha256_file(path)
        out.append(FileInfo(path=path, name=name, size=st.st_size, mtime=st.st_mtime, sha256=sha))
    return out


def inside(library_dir, path):
    """True if `path` lives directly in the library folder (its own fence)."""
    lib = os.path.normcase(os.path.abspath(library_dir))
    p = os.path.normcase(os.path.abspath(path))
    return os.path.dirname(p) == lib


# --- text decoding & Gutenberg -----------------------------------------------------

def decode_text_bytes(data):
    """(text, encoding). BOMs, utf-8, charset detection, latin-1 last."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace"), "utf-8-sig"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace"), "utf-16"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best is not None:
            return str(best), best.encoding
    except Exception:
        pass
    return data.decode("latin-1", errors="replace"), "latin-1"


_GUT_START = re.compile(r"^\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG.*$", re.M)
_GUT_END = re.compile(r"^\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG.*$", re.M)


def strip_gutenberg(text):
    """Cut the Project Gutenberg header/licence; pull Title:/Author: from it."""
    meta = {}
    m = _GUT_START.search(text)
    if not m:
        return text, meta
    header = text[:m.start()]
    for key in ("Title", "Author"):
        mm = re.search(rf"^{key}:\s*(.+)$", header, re.M)
        if mm:
            meta[key.lower()] = mm.group(1).strip()
    body = text[m.end():]
    e = _GUT_END.search(body)
    if e:
        body = body[:e.start()]
    meta["gutenberg"] = True
    return body, meta


_BAD_TITLES = re.compile(r"^(untitled|microsoft word|document\d*|scan|new document|\.?pdf|book)\b", re.I)


def guess_title_author(file_name, first_lines=(), meta=None):
    """Best-effort (title, author). Metadata wins when it looks real."""
    meta = meta or {}
    title = (meta.get("title") or "").strip()
    author = (meta.get("author") or "").strip() or None
    if title and not _BAD_TITLES.match(title) and len(title) <= 120 and not title.lower().endswith(
            tuple(SUPPORTED)):
        return title, author
    stem = os.path.splitext(os.path.basename(file_name))[0]
    stem = re.sub(r"[_]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    m = re.match(r"^(.+?)\s+-\s+(.+)$", stem)              # "Author - Title" or "Title - Author"
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        # the side with fewer words is more likely the author
        if len(a.split()) <= 3 and len(b.split()) > len(a.split()):
            return b, author or a
        return a, author or b
    m = re.match(r"^(.+?)\s*\((.+?)\)\s*$", stem)          # "Title (Author)"
    if m:
        return m.group(1).strip(), author or m.group(2).strip()
    m = re.match(r"^(.+?)\s+by\s+(.+)$", stem, re.I)       # "Title by Author"
    if m:
        return m.group(1).strip(), author or m.group(2).strip()
    if stem:
        return stem, author
    for line in first_lines:
        line = line.strip()
        if 3 <= len(line) <= 80:
            return line, author
    return "Untitled", author


# --- HTML -> text ----------------------------------------------------------------------

_BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote",
               "section", "article", "header", "footer", "tr", "table", "hr", "pre", "figure", "aside",
               "nav", "dd", "dt", "dl", "body"}
_SKIP_TAGS = {"script", "style", "head", "title", "svg", "math"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.headings = []
        self._skip = 0
        self._heading = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip += 1
        if tag in ("h1", "h2", "h3"):
            self._heading = []
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n" if tag != "br" else "\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        if tag in ("h1", "h2", "h3") and self._heading is not None:
            text = re.sub(r"\s+", " ", "".join(self._heading)).strip()
            if text:
                self.headings.append(text)
            self._heading = None
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n" if tag != "br" else "\n")

    def handle_data(self, data):
        if self._skip:
            return
        self.parts.append(data)
        if self._heading is not None:
            self._heading.append(data)


def html_to_text(html_str):
    """(text with paragraph breaks, [h1-h3 texts in order])."""
    p = _TextExtractor()
    try:
        p.feed(html_str or "")
        p.close()
    except Exception:
        pass
    return clean_text("".join(p.parts)), p.headings


# --- chapter detection -----------------------------------------------------------------

_STYLE_RES = {
    "explicit": re.compile(
        r"^\s*(?:CHAPTER|Chapter|BOOK|Book|PART|Part|LETTER|Letter|CANTO|Canto|ACT|Act|SCENE|Scene|"
        r"STAVE|Stave|SECTION|Section|EPISODE|Episode|STORY|Story)\s+"
        r"(?:[0-9]{1,3}|[IVXLCivxlc]{1,7}|[A-Z][a-z]+(?:-[A-Za-z]+)?)\b.*$"),
    "markdown": re.compile(r"^#{1,3}\s+\S.*$"),
    "roman": re.compile(r"^\s*[IVXLC]{1,7}\.?\s*$"),
    "arabic": re.compile(r"^\s*\d{1,3}\.?\s*$"),
    "caps": re.compile(r"^[A-Z][A-Z0-9 ,.'’:;\-]{3,60}$"),
}
_STYLE_ORDER = ("explicit", "markdown", "roman", "arabic", "caps")
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_to_int(s):
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = _ROMAN.get(ch, 0)
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def _heading_number(style, line):
    if style == "arabic":
        return int(re.sub(r"\D", "", line))
    if style == "roman":
        return _roman_to_int(line.strip().rstrip("."))
    return None


def _candidates(lines, style, markdown):
    rx = _STYLE_RES[style]
    found = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or len(s) > 80:
            continue
        if style == "markdown" and not markdown and sum(1 for l in lines if _STYLE_RES["markdown"].match(l)) < 3:
            return []
        if not rx.match(s):
            continue
        if style == "caps" and (len(s.split()) < 2 or sum(c.isalpha() for c in s) < 4):
            continue
        prev_blank = i == 0 or not lines[i - 1].strip()
        next_blank = i + 1 >= len(lines) or not lines[i + 1].strip()
        if not prev_blank:
            continue
        if not next_blank and style not in ("explicit", "markdown"):
            continue
        found.append((i, s.lstrip("#").strip()))
    return found


def _drop_toc(cands, lines):
    """Headings within 80 words of the previous one are a table of contents;
    a title that recurs keeps its LAST occurrence (the real chapter)."""
    kept = []
    for idx, title in cands:
        if kept:
            prev_idx = kept[-1][0]
            between = count_words("\n".join(lines[prev_idx + 1:idx]))
            if between < 80:
                kept.pop()  # the previous one was a TOC line too
                if kept and count_words("\n".join(lines[kept[-1][0] + 1:idx])) < 80:
                    continue
        kept.append((idx, title))
    seen = {}
    for idx, title in kept:
        seen[title.lower()] = (idx, title)
    last = {v[0] for v in seen.values()}
    return [c for c in kept if c[0] in last]


def _plausible(cands, lines, style):
    n = len(cands)
    if n < 3 or n > 400:
        return False
    sizes = []
    for k, (idx, _) in enumerate(cands):
        end = cands[k + 1][0] if k + 1 < n else len(lines)
        sizes.append(count_words("\n".join(lines[idx + 1:end])))
    sizes.sort()
    median = sizes[len(sizes) // 2]
    # explicit "CHAPTER N" / markdown "#" headings are a strong signal (and TOC runs are
    # already dropped), so trust short chapters; the risky styles need a real chapter's length.
    if style in ("explicit", "markdown"):
        if not (40 <= median <= 15000):
            return False
    elif not (600 <= median <= 15000):
        return False
    if style in ("arabic", "roman"):
        nums = [_heading_number(style, lines[idx]) for idx, _ in cands]
        gaps = sum(1 for a, b in zip(nums, nums[1:]) if b != a + 1)
        if gaps > 1 or nums[0] not in (1, 2):
            return False
    return True


def _fallback_sections(text, target=FALLBACK_SECTION_TOKENS):
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections, cur, cur_tok = [], [], 0
    for p in paras:
        t = est_tokens(p)
        if cur and cur_tok + t > target:
            sections.append(Section(f"Part {len(sections) + 1}", "\n\n".join(cur), "part"))
            cur, cur_tok = [], 0
        cur.append(p)
        cur_tok += t
    if cur:
        sections.append(Section(f"Part {len(sections) + 1}", "\n\n".join(cur), "part"))
    return sections


def segment(text, markdown=False, outline=None):
    """Split a whole text into chapter-like sections. `outline` = [(title, char_offset)]
    (from a PDF's bookmarks) beats every text heuristic when it has >= 3 entries."""
    text = clean_text(text)
    if not text:
        return []
    if outline and len(outline) >= 3:
        pts = sorted((max(0, off), t) for t, off in outline)
        sections = []
        head = text[:pts[0][0]].strip()
        if count_words(head) > 200:
            sections.append(Section("Front matter", head, "front"))
        for k, (off, title) in enumerate(pts):
            end = pts[k + 1][0] if k + 1 < len(pts) else len(text)
            body = text[off:end].strip()
            if body:
                sections.append(Section(title.strip() or f"Section {k + 1}", body, "chapter"))
        if len(sections) >= 3:
            return sections
    lines = text.split("\n")
    for style in _STYLE_ORDER:
        cands = _candidates(lines, style, markdown)
        if not cands:
            continue
        cands = _drop_toc(cands, lines)
        if not _plausible(cands, lines, style):
            continue
        sections = []
        head = "\n".join(lines[:cands[0][0]]).strip()
        if count_words(head) > 200:
            sections.append(Section("Front matter", head, "front"))
        lead_in = head if count_words(head) <= 200 and head else ""
        for k, (idx, title) in enumerate(cands):
            end = cands[k + 1][0] if k + 1 < len(cands) else len(lines)
            body = "\n".join(lines[idx + 1:end]).strip()
            if k == 0 and lead_in:
                body = (lead_in + "\n\n" + body).strip()
            if body:
                sections.append(Section(title, body, "chapter"))
        if len(sections) >= 3:
            return sections
    return _fallback_sections(text)


def segment_spine(items):
    """EPUB spine items [(title|None, text, headings)] -> sections. Tiny items
    (cover, title page, copyright) are folded into the next real one."""
    sections, carry = [], ""
    for k, (title, text, headings) in enumerate(items):
        text = clean_text(text)
        if not text:
            continue
        if count_words(text) < 150:
            carry = (carry + "\n\n" + text).strip()
            continue
        if headings and len(headings) >= 2 and count_words(text) > 4000:
            subs = segment(text)
            if len(subs) >= 2:
                if carry:
                    subs[0] = Section(subs[0].title, carry + "\n\n" + subs[0].text, subs[0].kind)
                    carry = ""
                sections.extend(subs)
                continue
        name = (headings[0] if headings else title) or f"Section {len(sections) + 1}"
        body = (carry + "\n\n" + text).strip() if carry else text
        carry = ""
        sections.append(Section(name[:120], body, "chapter"))
    if carry:
        if sections:
            last = sections[-1]
            sections[-1] = Section(last.title, last.text + "\n\n" + carry, last.kind)
        else:
            sections.append(Section("Section 1", carry, "chapter"))
    return sections


def normalize_sections(sections):
    """Split > SECTION_MAX_TOKENS at paragraph boundaries ('(part 2)'), merge
    < SECTION_MIN_TOKENS into the next, keep front/back tags."""
    out = []
    for s in sections:
        if s.est_tokens <= SECTION_MAX_TOKENS:
            out.append(s)
            continue
        paras = [p for p in re.split(r"\n\s*\n", s.text) if p.strip()]
        parts, cur, cur_tok = [], [], 0
        target = SECTION_MAX_TOKENS * 0.85
        for p in paras:
            t = est_tokens(p)
            if cur and cur_tok + t > target:
                parts.append("\n\n".join(cur))
                cur, cur_tok = [], 0
            cur.append(p)
            cur_tok += t
        if cur:
            parts.append("\n\n".join(cur))
        for i, body in enumerate(parts):
            title = s.title if i == 0 else f"{s.title} (part {i + 1})"
            out.append(Section(title, body, "split" if i else s.kind))
    merged = []
    for s in out:
        if merged and s.est_tokens < SECTION_MIN_TOKENS:
            # a scrap (a few lines): fold it into its neighbour, keeping the
            # larger side's title and never inheriting a 'front' tag
            prev = merged.pop()
            title = prev.title if prev.est_tokens >= s.est_tokens else s.title
            kind = prev.kind if prev.kind != "front" else s.kind
            merged.append(Section(title, prev.text + "\n\n" + s.text, kind))
        else:
            merged.append(s)
    return merged


# --- chunking -----------------------------------------------------------------------------

def chunk(text, target_tokens=CHUNK_TOKENS, overlap_tokens=CHUNK_OVERLAP):
    """Retrieval chunks: paragraphs accumulate to ~target; over-long paragraphs
    split at sentences; each chunk after the first starts with the last whole
    sentences (~overlap tokens) of the previous one."""
    text = clean_text(text)
    if not text:
        return []
    units = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if not p:
            continue
        if est_tokens(p) <= target_tokens:
            units.append(p)
        else:
            cur = []
            for s in split_sentences(p):
                if cur and est_tokens(" ".join(cur + [s])) > target_tokens:
                    units.append(" ".join(cur))
                    cur = []
                cur.append(s)
            if cur:
                units.append(" ".join(cur))
    chunks, cur, cur_tok = [], [], 0
    for u in units:
        t = est_tokens(u)
        if cur and cur_tok + t > target_tokens:
            chunks.append("\n\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(u)
        cur_tok += t
    if cur:
        chunks.append("\n\n".join(cur))
    if overlap_tokens <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, this in zip(chunks, chunks[1:]):
        tail, tail_tok = [], 0
        for s in reversed(split_sentences(prev)):
            if tail_tok + est_tokens(s) > overlap_tokens:
                break
            tail.insert(0, s)
            tail_tok += est_tokens(s)
        out.append((" ".join(tail) + "\n\n" + this) if tail else this)
    return out


# --- readers ---------------------------------------------------------------------------------

def parse_file(path):
    """Dispatch by extension. Raises Unreadable with a spoken-quality message."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        raise Unreadable(f"I can't read {ext or 'that kind of'} files")
    try:
        if ext == ".epub":
            book = read_epub(path)
        elif ext == ".pdf":
            book = read_pdf(path)
        else:
            book = read_text(path)
    except Unreadable:
        raise
    except Exception as e:  # any parser crash becomes an honest reason, never a traceback
        raise Unreadable(f"the file couldn't be parsed ({type(e).__name__})") from e
    if not book.sections or book.words < 50:
        raise Unreadable("there's no readable text in it")
    return book


def read_text(path):
    with open(path, "rb") as f:
        data = f.read()
    text, encoding = decode_text_bytes(data)
    text, gut = strip_gutenberg(text)
    text = clean_text(text)
    if not text:
        raise Unreadable("the file is empty")
    is_md = path.lower().endswith(".md")
    if is_md:
        text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.M)          # bullets
        text = re.sub(r"[*_]{1,3}([^*_\n]+)[*_]{1,3}", r"\1", text)  # emphasis
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)             # images
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)         # links
    first_lines = [l for l in text.split("\n")[:12] if l.strip()]
    title, author = guess_title_author(path, first_lines, gut)
    sections = normalize_sections(segment(text, markdown=is_md))
    meta = {"encoding": encoding, **({"gutenberg": True} if gut.get("gutenberg") else {})}
    return ParsedBook(title, author, "md" if is_md else "txt", sections, meta=meta)


_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container",
       "opf": "http://www.idpf.org/2007/opf",
       "dc": "http://purl.org/dc/elements/1.1/"}


def read_epub(path):
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise Unreadable("the EPUB file is damaged")
    with zf:
        names = set(zf.namelist())
        if "META-INF/encryption.xml" in names:
            raise Unreadable("it's DRM-protected, so I can't open it")
        try:
            container = ET.fromstring(zf.read("META-INF/container.xml"))
            opf_path = container.find(".//c:rootfile", _NS).get("full-path")
        except Exception:
            opf_path = next((n for n in names if n.lower().endswith(".opf")), None)
            if not opf_path:
                raise Unreadable("the EPUB has no package file")
        opf_dir = posixpath.dirname(opf_path)
        opf = ET.fromstring(zf.read(opf_path))
        title = author = None
        t = opf.find(".//dc:title", _NS)
        a = opf.find(".//dc:creator", _NS)
        if t is not None and (t.text or "").strip():
            title = t.text.strip()
        if a is not None and (a.text or "").strip():
            author = a.text.strip()
        manifest = {}
        for item in opf.iter("{%s}item" % _NS["opf"]):
            manifest[item.get("id")] = {"href": item.get("href", ""), "type": item.get("media-type", ""),
                                        "props": (item.get("properties") or "").split()}
        items = []
        for ref in opf.iter("{%s}itemref" % _NS["opf"]):
            if (ref.get("linear") or "yes").lower() == "no":
                continue
            m = manifest.get(ref.get("idref"))
            if not m or "nav" in m["props"]:
                continue
            if "html" not in m["type"] and not m["href"].lower().endswith((".xhtml", ".html", ".htm")):
                continue
            href = posixpath.normpath(posixpath.join(opf_dir, unquote(m["href"]))) if opf_dir else unquote(m["href"])
            href = href.split("#")[0]
            if href not in names:
                alt = next((n for n in names if n.endswith("/" + posixpath.basename(href)) or n == posixpath.basename(href)), None)
                if not alt:
                    continue
                href = alt
            raw = zf.read(href)
            text, headings = html_to_text(decode_text_bytes(raw)[0])
            items.append((headings[0] if headings else None, text, headings))
    if not items:
        raise Unreadable("the EPUB has no readable chapters")
    sections = normalize_sections(segment_spine(items))
    title, author = guess_title_author(path, [], {"title": title or "", "author": author or ""})
    return ParsedBook(title, author, "epub", sections, meta={"spine_items": len(items)})


_PAGE_NUM_RES = (re.compile(r"^\d{1,4}$"), re.compile(r"^page\s+\d+(\s+of\s+\d+)?$", re.I),
                 re.compile(r"^\d+\s*/\s*\d+$"), re.compile(r"^[-–—]\s*\d{1,4}\s*[-–—]$"))


def strip_running_lines(pages):
    """Remove running headers/footers (lines recurring on >30% of pages) and bare page numbers."""
    n = len(pages)
    counts = {}
    norm = lambda s: re.sub(r"\d", "#", s.strip().lower())
    edges = []
    for page in pages:
        lines = [l for l in page.split("\n") if l.strip()]
        edge = set(norm(l) for l in (lines[:2] + lines[-2:]))
        edges.append(edge)
        for e in edge:
            counts[e] = counts.get(e, 0) + 1
    running = {e for e, c in counts.items() if n >= 4 and c > 0.3 * n and len(e) >= 3}
    out = []
    for page in pages:
        kept = []
        lines = page.split("\n")
        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                kept.append("")
                continue
            near_edge = i < 3 or i >= len(lines) - 3
            if near_edge and (norm(s) in running or any(r.match(s) for r in _PAGE_NUM_RES)):
                continue
            kept.append(line)
        out.append("\n".join(kept))
    return out


def reflow_pages(pages):
    """Join wrapped lines, de-hyphenate, keep paragraph breaks; page break = new
    paragraph only if the previous page ended a sentence."""
    paras = []
    cur = ""
    for page in pages:
        for line in page.split("\n"):
            s = line.strip()
            if not s:
                if cur:
                    paras.append(cur)
                    cur = ""
                continue
            if not cur:
                cur = s
            elif cur.endswith("-") and s[:1].islower():
                cur = cur[:-1] + s
            else:
                cur = cur + " " + s
        # page boundary
        if cur and re.search(r'[.!?]["”’)]?$', cur):
            paras.append(cur)
            cur = ""
    if cur:
        paras.append(cur)
    return "\n\n".join(paras)


def read_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise Unreadable("I can't read PDFs yet — the PDF library isn't installed")
    reader = PdfReader(path)
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise Unreadable("it's a password-protected PDF")
        except Exception:
            raise Unreadable("it's a password-protected PDF")
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    n = len(pages)
    if n == 0:
        raise Unreadable("the PDF has no pages")
    total = sum(len(t) for t in pages)
    if total < 60 * n:
        raise Unreadable("it's a scan with no text layer, so I can't read it")
    pages = strip_running_lines(pages)
    offsets, text_parts, pos = [], [], 0
    for page in pages:
        offsets.append(pos)
        pos += len(page) + 2
    text = reflow_pages(pages)
    outline = None
    try:
        flat = []

        def walk(items):
            for it in items:
                if isinstance(it, list):
                    walk(it)
                else:
                    try:
                        flat.append((str(it.title), reader.get_destination_page_number(it)))
                    except Exception:
                        pass
        walk(reader.outline or [])
        if len(flat) >= 3:
            # map page -> char offset in the reflowed text by cumulative page text share
            cum, acc = [], 0
            for page in pages:
                cum.append(acc)
                acc += len(page)
            scale = len(text) / max(1, acc)
            outline = [(t, int(cum[min(pg, n - 1)] * scale)) for t, pg in flat if pg is not None]
    except Exception:
        outline = None
    md = {}
    try:
        info = reader.metadata
        md = {"title": (info.title or "") if info else "", "author": (info.author or "") if info else ""}
    except Exception:
        pass
    warnings = []
    line_lens = [len(l) for p in pages for l in p.split("\n") if l.strip()]
    if line_lens:
        line_lens.sort()
        if line_lens[len(line_lens) // 2] < 35:
            warnings.append("looks like two columns; the text may read out of order")
    title, author = guess_title_author(path, text.split("\n")[:8], md)
    sections = normalize_sections(segment(text, outline=outline))
    return ParsedBook(title, author, "pdf", sections,
                      meta={"pages": n, "outline_used": bool(outline), "chars_per_page": total // n},
                      warnings=warnings)
