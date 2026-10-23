"""A tiny initiative-engine experiment: notice ONE thing in the project worth doing, propose ONE
small step. It only proposes. It never edits, runs or announces anything, and nothing imports it.

The pattern it looks for is a project rule from CLAUDE.md: "when you add or change a module, put
its notes in the matching file under .claude/rules/". A module is UNDOCUMENTED when no notes file
(CLAUDE.md or any .claude/rules/*.md, front matter included) names its stem as a word. A
`house_*.py` glob in a `paths:` list loads the notes, but it doesn't say what the module does.

The proposal picks the most recently modified undocumented module (the freshest drift) and names
the rules file to extend: the first one whose `paths:` globs already match it, else the one that
lists a module sharing the longest name prefix, else CLAUDE.md.

Pure apart from reading files under `root`. `python -m modules.initiative_probe` prints it.
"""

import fnmatch
import os
import re
from dataclasses import dataclass

MODULES_DIR = os.path.join("aiva", "modules")
RULES_DIR = os.path.join(".claude", "rules")


@dataclass(frozen=True)
class Proposal:
    pattern: str        # what it noticed
    module: str         # repo-relative path, forward slashes
    target: str         # the notes file to extend
    action: str         # the one small step


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def rules_paths(text):
    """The `paths:` globs from a rules file's front matter ([] when it has none)."""
    m = re.match(r"---\s*\n(.*?)\n---", text, re.S)
    if not m:
        return []
    return re.findall(r'^\s*-\s*"?([^"\n]+?)"?\s*$', m.group(1), re.M)


def notes(root):
    """{repo-relative notes file: text} for CLAUDE.md and every .claude/rules/*.md."""
    out = {"CLAUDE.md": _read(os.path.join(root, "CLAUDE.md"))}
    rdir = os.path.join(root, RULES_DIR)
    if os.path.isdir(rdir):
        for name in sorted(os.listdir(rdir)):
            if name.endswith(".md"):
                out[f".claude/rules/{name}"] = _read(os.path.join(rdir, name))
    return out


def undocumented(root):
    """[(mtime, stem)] for modules no notes file names, newest first."""
    texts = list(notes(root).values())
    mdir = os.path.join(root, MODULES_DIR)
    found = []
    for name in os.listdir(mdir) if os.path.isdir(mdir) else []:
        stem, ext = os.path.splitext(name)
        if ext != ".py" or stem.startswith("__"):
            continue
        word = re.compile(rf"\b{re.escape(stem)}\b")
        if not any(word.search(t) for t in texts):
            found.append((os.path.getmtime(os.path.join(mdir, name)), stem))
    return sorted(found, key=lambda x: (-x[0], x[1]))


def pick_target(module, files):
    """The notes file to extend for `module` (repo-relative). `files` = notes(root)."""
    best, best_len = "CLAUDE.md", 0
    stem = os.path.splitext(os.path.basename(module))[0]
    for name, text in files.items():
        globs = rules_paths(text)
        if any(fnmatch.fnmatchcase(module, g) for g in globs):
            return name
        for g in globs:
            other = os.path.splitext(os.path.basename(g))[0]
            n = len(os.path.commonprefix([stem, other]))
            if g.startswith("aiva/modules/") and n >= 3 and n > best_len:
                best, best_len = name, n
    return best


def propose(root):
    """The one proposal for the project at `root`, or None when every module has notes."""
    found = undocumented(root)
    if not found:
        return None
    stem = found[0][1]
    module = f"aiva/modules/{stem}.py"
    target = pick_target(module, notes(root))
    others = len(found) - 1
    pattern = (f"{module} is not named in any project notes"
               + (f" ({others} other module{'s' if others != 1 else ''} too)" if others else ""))
    if target == "CLAUDE.md":
        action = f"Add a one-line note for {module} to CLAUDE.md or a new .claude/rules file."
    elif any(fnmatch.fnmatchcase(module, g) for g in rules_paths(notes(root)[target])):
        action = f"Add a bullet describing {module} to {target} (its paths already load it)."
    else:
        action = (f"Add \"{module}\" to the paths: of {target} and a bullet describing it.")
    return Proposal(pattern, module, target, action)


if __name__ == "__main__":
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(here))
    p = propose(root)
    if p is None:
        print("pattern: none. Every module is named in the project notes.")
    else:
        print(f"pattern: {p.pattern}\naction:  {p.action}")
