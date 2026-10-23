"""The initiative-probe experiment: finds a module no project notes name, picks the freshest one,
and proposes the one rules file to note it in. Built on a throwaway repo in tmp_path."""
import os

from modules import initiative_probe as ip

RULES = '---\npaths:\n  - "aiva/modules/house_*.py"\n  - "aiva/modules/cat.py"\n---\n\n# notes\n'


def make_repo(tmp_path, modules, claude="", rules=RULES):
    mdir = tmp_path / "aiva" / "modules"
    mdir.mkdir(parents=True)
    for i, name in enumerate(modules):
        p = mdir / f"{name}.py"
        p.write_text("")
        os.utime(p, (1000 + i, 1000 + i))          # later in the list = newer
    (mdir / "__init__.py").write_text("")
    (tmp_path / "CLAUDE.md").write_text(claude)
    (tmp_path / ".claude" / "rules").mkdir(parents=True)
    (tmp_path / ".claude" / "rules" / "house.md").write_text(rules)
    return str(tmp_path)


def test_rules_paths_reads_front_matter_only():
    assert ip.rules_paths(RULES) == ["aiva/modules/house_*.py", "aiva/modules/cat.py"]
    assert ip.rules_paths("# no front matter\n- \"x.py\"") == []


def test_named_modules_are_documented_and_dunder_is_skipped(tmp_path):
    root = make_repo(tmp_path, ["memory", "cat"], claude="`modules.memory` keeps her facts; cat")
    assert ip.undocumented(root) == []
    assert ip.propose(root) is None


def test_a_glob_alone_is_not_a_note(tmp_path):
    # house_*.py loads house.md, but nothing says what house_memory does
    root = make_repo(tmp_path, ["house_memory"])
    p = ip.propose(root)
    assert p.module == "aiva/modules/house_memory.py"
    assert p.target == ".claude/rules/house.md"
    assert "already load" in p.action


def test_stem_must_match_as_a_whole_word(tmp_path):
    root = make_repo(tmp_path, ["cat"], claude="the category of things", rules="# no paths\n")
    assert [s for _, s in ip.undocumented(root)] == ["cat"]


def test_newest_undocumented_module_wins(tmp_path):
    root = make_repo(tmp_path, ["vtube_studio", "app_launcher"])
    p = ip.propose(root)
    assert p.module == "aiva/modules/app_launcher.py"
    assert "1 other module too" in p.pattern


def test_closest_named_module_picks_the_rules_file(tmp_path):
    root = make_repo(tmp_path, ["cat_care"])
    p = ip.propose(root)
    assert p.target == ".claude/rules/house.md"            # lists cat.py
    assert 'Add "aiva/modules/cat_care.py" to the paths:' in p.action


def test_no_matching_rules_file_falls_back_to_claude_md(tmp_path):
    root = make_repo(tmp_path, ["vtube_studio"])
    p = ip.propose(root)
    assert p.target == "CLAUDE.md"
    assert p.pattern == "aiva/modules/vtube_studio.py is not named in any project notes"


def test_probe_never_writes(tmp_path):
    root = make_repo(tmp_path, ["house_memory"])
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    ip.propose(root)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
