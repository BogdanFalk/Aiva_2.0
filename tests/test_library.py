import io
import os
import zipfile

import pytest

from modules import library as lib


# --- token/word/sentence helpers ---------------------------------------------------

def test_est_tokens_and_words():
    assert lib.est_tokens("a" * 400) == 100
    assert lib.count_words("one two three") == 3
    assert lib.split_sentences("Hi there. How are you? Fine!") == ["Hi there.", "How are you?", "Fine!"]


def test_clean_text_keeps_words_drops_junk():
    dirty = "The  quick­ brown\r\n\r\n\r\nfox​."
    out = lib.clean_text(dirty)
    assert "­" not in out and "​" not in out
    assert "quick brown" in out and "\n\n\n" not in out


# --- TXT + Project Gutenberg -------------------------------------------------------

GUTENBERG_TXT = """The Project Gutenberg eBook of A Small Tale

Title: A Small Tale
Author: Jane Doe

*** START OF THE PROJECT GUTENBERG EBOOK A SMALL TALE ***

CHAPTER I

{c1}

CHAPTER II

{c2}

CHAPTER III

{c3}

CHAPTER IV

{c4}

CHAPTER V

{c5}

*** END OF THE PROJECT GUTENBERG EBOOK A SMALL TALE ***

This file should be dropped: licensing boilerplate about redistribution.
"""


def _para(word, n=200):
    return " ".join([word] * n) + "."


def test_txt_gutenberg_chapters(tmp_path):
    body = GUTENBERG_TXT.format(c1=_para("alpha"), c2=_para("bravo"), c3=_para("charlie"),
                                c4=_para("delta"), c5=_para("echo"))
    f = tmp_path / "smalltale.txt"
    f.write_text(body, encoding="utf-8")
    book = lib.parse_file(str(f))
    assert book.format == "txt" and book.title == "A Small Tale" and book.author == "Jane Doe"
    assert book.meta.get("gutenberg") is True
    assert [s.title for s in book.sections] == ["CHAPTER I", "CHAPTER II", "CHAPTER III", "CHAPTER IV", "CHAPTER V"]
    joined = "\n".join(s.text for s in book.sections)
    assert "licensing boilerplate" not in joined and "START OF THE PROJECT" not in joined
    assert "alpha" in book.sections[0].text and "echo" in book.sections[4].text


@pytest.mark.parametrize("headings,style", [
    (["I", "II", "III", "IV"], "roman"),
    (["1", "2", "3", "4"], "arabic"),
    (["THE FIRST DOOR", "THE SECOND DOOR", "THE THIRD DOOR", "THE FOURTH DOOR"], "caps"),
])
def test_heading_styles(tmp_path, headings, style):
    parts = []
    for i, h in enumerate(headings):
        parts.append(h + "\n\n" + _para(f"word{i}", 650))
    f = tmp_path / f"{style}.txt"
    f.write_text("\n\n".join(parts), encoding="utf-8")
    book = lib.parse_file(str(f))
    assert len(book.sections) == len(headings)


def test_markdown_headings(tmp_path):
    md = "\n\n".join(f"# Heading {i}\n\n{_para(f'w{i}')}" for i in range(1, 5))
    f = tmp_path / "notes.md"
    f.write_text(md, encoding="utf-8")
    book = lib.parse_file(str(f))
    assert book.format == "md" and len(book.sections) == 4
    assert book.sections[0].title == "Heading 1"


def test_toc_block_rejected(tmp_path):
    toc = "\n".join(f"Chapter {i}" for i in range(1, 8))     # 7 headings jammed together = a TOC
    real = "\n\n".join(f"Chapter {i}\n\n{_para(f'body{i}')}" for i in range(1, 8))
    f = tmp_path / "withtoc.txt"
    f.write_text(toc + "\n\n" + real, encoding="utf-8")
    book = lib.parse_file(str(f))
    # the TOC block must not become its own tiny sections
    assert all(s.words > 50 for s in book.sections if s.kind == "chapter")


def test_no_headings_fallback(tmp_path):
    text = "\n\n".join(_para(f"lorem{i}", 120) for i in range(40))
    f = tmp_path / "flat.txt"
    f.write_text(text, encoding="utf-8")
    book = lib.parse_file(str(f))
    assert len(book.sections) >= 3
    for s in book.sections:
        assert s.est_tokens <= lib.SECTION_MAX_TOKENS
        assert s.title.startswith("Part ")


def test_decode_cp1252_smart_quotes():
    data = "He said “hello” — really.".encode("cp1252")
    text, enc = lib.decode_text_bytes(data)
    assert "hello" in text and "really" in text


@pytest.mark.parametrize("name,title,author", [
    ("Jane Doe - A Small Tale.txt", "A Small Tale", "Jane Doe"),
    ("A Small Tale (Jane Doe).epub", "A Small Tale", "Jane Doe"),
    ("A Small Tale by Jane Doe.pdf", "A Small Tale", "Jane Doe"),
])
def test_guess_title_author(name, title, author):
    t, a = lib.guess_title_author(name)
    assert t == title and a == author


# --- normalize / chunk -------------------------------------------------------------

def test_split_oversized_and_merge_tiny():
    big = lib.Section("Big", "\n\n".join(_para("chunky", 220) for _ in range(40)))
    tiny = lib.Section("Tiny", "just a few words here", "chapter")
    normal = lib.Section("Normal", _para("y", 300))
    out = lib.normalize_sections([big, tiny, normal])
    assert any("(part 2)" in s.title for s in out)          # big was split
    assert all(s.est_tokens <= lib.SECTION_MAX_TOKENS for s in out)
    assert not any(s.title == "Tiny" and s.est_tokens < lib.SECTION_MIN_TOKENS for s in out)  # tiny merged


def test_chunk_invariants():
    text = " ".join(f"Sentence number {i} carries a few plain words along." for i in range(250))
    chunks = lib.chunk(text, target_tokens=300, overlap_tokens=40)
    assert len(chunks) >= 3
    for c in chunks:
        assert lib.est_tokens(c) <= 300 + 160
    for prev, nxt in zip(chunks, chunks[1:]):
        tail = lib.split_sentences(prev)[-1]
        assert tail in nxt                                  # overlap carried the last sentence forward


# --- EPUB (built in-test) ----------------------------------------------------------

def _make_epub(path, chapters):
    """chapters: [(title, html_body)]. Minimal valid EPUB."""
    container = ('<?xml version="1.0"?><container version="1.0" '
                 'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                 '<rootfiles><rootfile full-path="OEBPS/content.opf" '
                 'media-type="application/oebps-package+xml"/></rootfiles></container>')
    manifest, spine, files = [], [], []
    manifest.append('<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>')
    for i, (title, body) in enumerate(chapters):
        cid = f"c{i}"
        href = f"{cid}.xhtml"
        manifest.append(f'<item id="{cid}" href="{href}" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="{cid}"/>')
        xhtml = (f'<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml">'
                 f'<head><title>{title}</title></head><body>{body}</body></html>')
        files.append((f"OEBPS/{href}", xhtml))
    opf = ('<?xml version="1.0" encoding="utf-8"?>'
           '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">'
           '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
           '<dc:title>The Test Book</dc:title><dc:creator>A. Writer</dc:creator>'
           '<dc:identifier id="id">urn:uuid:1</dc:identifier></metadata>'
           f'<manifest>{"".join(manifest)}'
           '<item id="ncxid" href="toc.ncx" media-type="application/x-dtbncx+xml"/></manifest>'
           f'<spine toc="ncxid"><itemref idref="nav"/>{"".join(spine)}</spine></package>')
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/nav.xhtml", '<html xmlns="http://www.w3.org/1999/xhtml"><body><nav>toc</nav></body></html>')
        zf.writestr("OEBPS/toc.ncx", '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"></ncx>')
        for name, content in files:
            zf.writestr(name, content)


def test_epub_minimal(tmp_path):
    body = lambda w: "<p>" + " ".join([w] * 600) + " &amp; done.</p>"
    chapters = [("Opening", "<h1>Opening</h1>" + body("aa")),
                ("Middle", "<h1>Middle</h1>" + body("bb")),
                ("Ending", "<h1>Ending</h1>" + body("cc"))]
    f = tmp_path / "test.epub"
    _make_epub(str(f), chapters)
    book = lib.parse_file(str(f))
    assert book.format == "epub" and book.title == "The Test Book" and book.author == "A. Writer"
    assert [s.title for s in book.sections] == ["Opening", "Middle", "Ending"]   # spine order, nav skipped
    assert "&" in book.sections[0].text and "&amp;" not in book.sections[0].text  # entity decoded
    assert "aa" in book.sections[0].text


def test_epub_drm_unreadable(tmp_path):
    f = tmp_path / "drm.epub"
    _make_epub(str(f), [("A", "<p>" + " ".join(["x"] * 600) + "</p>")])
    with zipfile.ZipFile(str(f), "a") as zf:
        zf.writestr("META-INF/encryption.xml", "<encryption/>")
    with pytest.raises(lib.Unreadable) as e:
        lib.parse_file(str(f))
    assert "DRM" in str(e.value)


# --- PDF ---------------------------------------------------------------------------

def test_strip_running_lines_and_reflow():
    pages = [f"MY BOOK\n{i}\nThe sen-\ntence wraps here.\nAnother line follows it.\n42"
             for i in range(1, 9)]
    cleaned = lib.strip_running_lines(pages)
    assert all("MY BOOK" not in p for p in cleaned)
    text = lib.reflow_pages(cleaned)
    assert "sentence wraps here" in text                    # de-hyphenated + joined
    assert "MY BOOK" not in text


def test_pdf_scan_detected(tmp_path):
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter
    w = PdfWriter()
    for _ in range(5):
        w.add_blank_page(width=300, height=400)             # no text layer
    f = tmp_path / "scan.pdf"
    with open(f, "wb") as fh:
        w.write(fh)
    with pytest.raises(lib.Unreadable) as e:
        lib.parse_file(str(f))
    assert "scan" in str(e.value)


def test_pdf_smoke_with_text(tmp_path):
    pytest.importorskip("pypdf")
    pytest.importorskip("reportlab")   # only run when a text-PDF builder is available
    from reportlab.pdfgen import canvas
    f = tmp_path / "real.pdf"
    c = canvas.Canvas(str(f))
    for i in range(1, 6):
        for line in [f"CHAPTER {i}"] + [f"word{i} " * 30 for _ in range(20)]:
            c.drawString(72, 720, line)
        c.showPage()
    c.save()
    book = lib.parse_file(str(f))
    assert book.format == "pdf" and book.words > 100
