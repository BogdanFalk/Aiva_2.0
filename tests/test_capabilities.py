"""Phase-1 tests for the intention-indexed capability map: tier lookups, wish→capability matching,
the prompt block, and the safety invariant that 'auto' powers only touch reversible tools."""
from modules import capabilities as cap


def test_tiers_are_valid_and_unknown_is_ask():
    for c in cap.CAPABILITIES:
        assert c.tier in cap.TIERS
    assert cap.tier("find_and_keep") == "auto"
    assert cap.tier("build_software") == "ask"
    assert cap.tier("spend_or_send") == "never"
    assert cap.tier("nonexistent_power") == "ask"          # conservative default


def test_auto_capabilities_only_touch_safe_tools():
    # the safety envelope, encoded: nothing she may do on her own may reach a PC/shell/build/self-edit tool
    for c in cap.CAPABILITIES:
        if c.tier == "auto":
            outside = set(c.tools) - cap.SAFE_TOOLS
            assert not outside, f"auto capability {c.name!r} touches unsafe tools {outside}"


def test_match_acquisitive_visual_wish_is_auto_find_and_keep():
    m = cap.match("I wish I had a cute hairclip for my avatar")
    assert m is not None and m.name == "find_and_keep" and m.tier == "auto"


def test_match_build_wish_is_ask():
    m = cap.match("I really want to build a little tool for myself")
    assert m is not None and m.name == "build_software" and m.tier == "ask"


def test_match_self_change_is_ask():
    m = cap.match("I'd love to recolor my hair and change my look")
    assert m is not None and m.tier == "ask"


def test_match_spend_is_never():
    m = cap.match("can you buy and order this for me")
    assert m is not None and m.name == "spend_or_send" and m.tier == "never"


def test_match_gibberish_is_none():
    assert cap.match("qwzx flooble grommit") is None
    assert cap.match("") is None


def test_prompt_lines_cover_powers_and_mark_tiers():
    lines = cap.prompt_lines()
    assert len(lines) == len(cap.CAPABILITIES)
    joined = "\n".join(lines)
    assert "on your own" in joined and "ask first" in joined and "not you" in joined
    assert any("never buy" in ln for ln in lines)          # the hard 'find and show' line is present


def test_for_tool_leads_with_match_and_lists_all():
    out = cap.for_tool("find me a wallpaper")
    assert out["matched"]["name"] == "find_and_keep"
    assert len(out["capabilities"]) == len(cap.CAPABILITIES)
    assert cap.for_tool(None)["matched"] is None


def test_exclude_removes_a_power_from_every_consumer():
    ex = ("tend_cat",)                                         # a sister without the cat
    names = [c.name for c in cap.active(ex)]
    assert "tend_cat" not in names and len(names) == len(cap.CAPABILITIES) - 1
    assert cap.match("feed the cat").name == "tend_cat"
    assert cap.match("feed the cat", exclude=ex) is None       # nothing else claims it
    assert len(cap.prompt_lines(exclude=ex)) == len(cap.CAPABILITIES) - 1
    assert not any("cat" in ln for ln in cap.prompt_lines(exclude=ex))
    out = cap.for_tool("check on the cat", exclude=ex)
    assert out["matched"] is None and len(out["capabilities"]) == len(cap.CAPABILITIES) - 1
    assert cap.prompt_lines() == cap.prompt_lines(exclude=())  # default unchanged


def test_know_sister_is_auto_and_only_recalls():
    m = cap.match("I wonder if my sister has ever done that")
    assert m is not None and m.name == "know_sister" and m.tier == "auto"
    assert set(m.tools) == {"recall_about_sister"} and set(m.tools) <= cap.SAFE_TOOLS
    assert cap.tier("know_sister") == "auto"
