"""Offline tests for her curated space (SpaceMemory): save/dedupe/recall/drop/status,
in-memory sqlite + a deterministic fake embed. No network, no LLM."""
import sqlite3

import pytest

from modules.space import KINDS, SpaceMemory


def fake_embed_factory(dims=64):
    import zlib

    async def embed(text):
        vec = [0.0] * dims
        for tok in (text or "").lower().split():
            vec[zlib.crc32(tok.encode()) % dims] += 1.0
        n = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / n for v in vec]
    return embed


@pytest.fixture
def space():
    return SpaceMemory(sqlite3.connect(":memory:"), embed=fake_embed_factory(), log=lambda m: None)


async def test_save_and_list(space):
    res = await space.save_find("a blue enamel hairclip", "I love the little star charm",
                                kind="image", source_url="https://shop.example.com/x",
                                path="C:/x/clip.png", file_name="clip.png", sha256="abc123",
                                tags=["hairclip", "cute"])
    assert res["status"] == "saved" and res["id"]
    items = space.list_finds()
    assert len(items) == 1
    it = items[0]
    assert it["kind"] == "image" and it["subject"].startswith("a blue")
    assert it["tags"] == ["hairclip", "cute"] and it["source_url"].startswith("https://")


async def test_dedupe_by_sha256_updates(space):
    await space.save_find("hairclip", "first", kind="image", sha256="samehash")
    res = await space.save_find("hairclip (better shot)", "clearer photo", kind="image",
                                sha256="samehash")
    assert res["status"] == "updated"
    items = space.list_finds()
    assert len(items) == 1 and items[0]["why"] == "clearer photo"   # same file → one row, updated


async def test_dedupe_by_meaning_updates(space):
    await space.save_find("a cozy cat mug", "warm and round", kind="image", sha256="h1")
    res = await space.save_find("a cozy cat mug", "still love it", kind="image", sha256="h2")
    assert res["status"] == "updated"                               # same subject → deduped by embedding
    assert len(space.list_finds()) == 1


async def test_distinct_finds_coexist(space):
    await space.save_find("a cozy cat mug", "warm", kind="image", sha256="h1")
    await space.save_find("a bleak brutalist poster", "stark", kind="image", sha256="h2")
    assert len(space.list_finds()) == 2


async def test_drop_blocks_resave(space):
    await space.save_find("a garish neon sign", "regret", kind="image", sha256="h9")
    dropped = await space.drop_find("garish neon")
    assert dropped and "neon" in dropped
    assert space.list_finds() == []
    res = await space.save_find("a garish neon sign", "again", kind="image", sha256="h9")
    assert res["status"] == "blocked"                              # vetoed file never comes back


async def test_recall_ranks_by_relevance(space):
    await space.save_find("a cozy cat mug", "warm and round", kind="image", sha256="h1")
    await space.save_find("a bleak brutalist poster", "stark", kind="image", sha256="h2")
    hits = await space.recall("cat", k=1)
    assert hits["items"] and "cat" in hits["items"][0]["subject"]


async def test_unknown_kind_falls_back_to_file(space):
    await space.save_find("mystery blob", "dunno", kind="hologram", sha256="h1")
    assert space.list_finds()[0]["kind"] == "file"
    assert "file" in KINDS


async def test_link_without_file_saves(space):
    res = await space.save_find("a tutorial on Live2D recolors", "useful later",
                                kind="link", source_url="https://youtu.be/abc")
    assert res["status"] == "saved"
    it = space.list_finds(kind="link")[0]
    assert it["kind"] == "link" and it["path"] is None


async def test_status_and_lines(space):
    await space.save_find("a cat mug", "warm", kind="image", sha256="h1",
                          source_url="https://shop.example.com/mug")
    await space.save_find("a synth playlist", "energetic", kind="link",
                          source_url="https://music.example.com/p")
    st = space.status()
    assert st["total"] == 2 and st["by_kind"].get("image") == 1 and st["by_kind"].get("link") == 1
    lines = space.status_lines()
    assert any("cat mug" in ln for ln in lines) and any("from " in ln for ln in lines)
