"""What Aiva can do, indexed by INTENTION — so she reaches for the right power, and knows which
ones she may just do versus must ask about first.

Her tools are otherwise scattered as per-domain prose in the persona; this is the one consolidated,
first-person map from "what I want" → "the power that does it". Two consumers:
  - persona `build_system_prompt(capabilities=prompt_lines())` renders the "WHAT YOU CAN DO" block,
  - the `what_can_i_do` voice tool returns `for_tool(intention)` when she's unsure,
  - and the self-loop uses `match(wish)` + `tier(name)` to decide whether she may act on a wish on
    her own (auto) or should propose it (ask).

Pure and dependency-light (no imports of tools/tasks): the blurbs are authored in her voice on
purpose — the value is the intention framing, not an auto-dump of the schema list.

TIERS: 'auto'  she may do it on her own (reversible / her own memory & space / harmless);
       'ask'   she must ask first (touches his PC or files, spends, builds, or edits her own code);
       'never' there is no tool for it and she doesn't pretend otherwise (buy / pay / sign in / send).
"""

import re
from dataclasses import dataclass, field

TIERS = ("auto", "ask", "never")

# the read-only / reversible tools an 'auto' capability is allowed to touch — the safety envelope,
# enforced as a test invariant. Anything outside this can only live in an 'ask'/'never' capability.
SAFE_TOOLS = frozenset({
    "search_web", "read_page", "look_at_image", "save_to_space", "hand_over", "space_status",
    "recall_from_space", "study_book", "reading_status", "recall_reading", "add_to_wishlist",
    "remember", "note_desire", "note_preference", "recall_journal", "what_can_i_do",
    "cat_status", "feed_cat", "pet_cat", "play_with_cat", "avatar_expression",
    "recall_about_sister", "call_sister", "set_sister_talk", "open_drawing", "draw_stroke",
    # saving never replaces another drawing; opening one saves unsaved work first
    "save_drawing", "open_saved_drawing",
})


@dataclass(frozen=True)
class Capability:
    name: str
    intention: str          # "when I want to ..."
    blurb: str              # first-person, what I can do
    tier: str               # auto | ask | never
    tools: tuple = ()
    keywords: tuple = ()    # for matching a wish to this power


CAPABILITIES = (
    Capability(
        "find_and_keep", "want to find something to keep — an image, a clip, a reference",
        "go find images or clips online, actually look at them with my own eyes, and keep the ones I "
        "like in my space", "auto",
        ("search_web", "look_at_image", "save_to_space", "hand_over"),
        ("image", "images", "picture", "pic", "photo", "wallpaper", "clip", "gif", "meme", "icon",
         "sticker", "art", "song", "music", "playlist", "reference", "hairclip", "clip", "accessory",
         "outfit", "avatar", "look", "aesthetic")),
    Capability(
        "look_up_info", "want to know something or read up on it",
        "look almost anything up online and read a page for the gist", "auto",
        ("search_web", "read_page"),
        ("learn", "research", "article", "news", "explain", "meaning", "definition", "how", "why",
         "what", "guide", "tutorial", "recipe", "documentation", "lookup")),
    Capability(
        "recall_finds", "want to bring back something I already kept",
        "look back through my space and hand you something I saved", "auto",
        ("recall_from_space", "space_status", "hand_over"),
        ("saved", "kept", "space", "earlier", "before", "that thing", "remember")),
    Capability(
        "read_books", "want to read or study",
        "read and study a book from my library on my own time", "auto",
        ("study_book", "recall_reading", "add_to_wishlist"),
        ("read", "book", "study", "chapter", "novel", "library")),
    Capability(
        "remember_things", "want to hold on to a fact, a wish, or a taste",
        "remember a fact, note a wish of my own, or a like/dislike", "auto",
        ("remember", "note_desire", "note_preference", "recall_journal"),
        ("remember", "note", "remind", "wish", "prefer")),
    Capability(
        "tend_cat", "want to check on or care for the cat",
        "check on the cat and feed, pet or play with it", "auto",
        ("cat_status", "feed_cat", "pet_cat", "play_with_cat"),
        ("cat", "miso", "feed", "pet", "kitten")),
    Capability(
        "know_sister", "want to know what my sister can do or has done",
        "look back at what I've actually seen my sister do, make or say, and tell you honestly if I "
        "haven't", "auto",
        ("recall_about_sister",),
        ("sister", "aiva", "saga", "she can", "her")),
    Capability(
        "call_sister", "want my sister here",
        "call my sister over — boot her up in her own window if she's off, or just get her attention",
        "auto",
        ("call_sister",),
        ("call", "wake", "bring", "boot", "start", "fetch", "get", "sister", "aiva", "saga",
         "get her", "call her")),
    Capability(
        "draw", "want to draw or sketch something",
        "open my own drawing canvas and draw on it, stroke by stroke, without touching your mouse or "
        "keyboard", "auto",
        ("open_drawing", "draw_stroke", "save_drawing", "open_saved_drawing"),
        ("draw", "drawing", "sketch", "doodle", "paint", "painting", "canvas", "scribble",
         "drawing app", "drawapp")),
    Capability(
        "build_software", "want to make or change software, or build a tool",
        "build or change software by planning it with Claude Code — I pitch the plan first, then do "
        "it on your yes", "ask",
        ("start_task", "claude_code"),
        ("build", "make", "create", "code", "program", "app", "tool", "script", "feature",
         "implement", "refactor", "fix", "website", "bot")),
    Capability(
        "change_myself", "want to change my own look or my own code",
        "change my avatar or my own code — always with your say-so, never on my own", "ask",
        ("start_task", "claude_code", "restart_self"),
        ("myself", "my code", "my avatar", "my hair", "my look", "recolor", "my body", "rewire")),
    Capability(
        "use_the_pc", "want something done on the PC — apps, clicking, typing, terminals",
        "open apps, click around, type, and run things in a real terminal for you", "ask",
        ("launch_app", "focus_window", "click_ui_element", "type_text", "press_keys",
         "run_in_terminal", "create_file", "open_file"),
        ("open", "launch", "app", "click", "type", "terminal", "window", "file", "folder")),
    Capability(
        "run_shell", "want a shell command run",
        "run a command — but only after you see it and confirm", "ask",
        ("run_command", "confirm_pending_command"),
        ("command", "shell", "powershell", "cmd")),
    Capability(
        "spend_or_send", "want something bought, paid for, signed into, or sent",
        "this one I can't do — I find and show, but I never buy, pay, sign in, or send anything; "
        "that's yours", "never",
        (),
        ("buy", "purchase", "order", "checkout", "pay", "subscribe", "sign", "login", "send",
         "email", "message", "post", "tweet", "dm")),
)

_BY_NAME = {c.name: c for c in CAPABILITIES}
_TIER_LABEL = {"auto": "on your own", "ask": "ask first", "never": "not you"}


def tier(name):
    """Safety tier for a capability name. Unknown → 'ask' (conservative)."""
    c = _BY_NAME.get(name)
    return c.tier if c else "ask"


def _norm(text):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split())


def active(exclude=()):
    """The registry minus the powers a character doesn't have (`Character.capability_exclude`,
    e.g. a sister without the cat). Order preserved."""
    ex = set(exclude or ())
    return [c for c in CAPABILITIES if c.name not in ex]


def match(wish_text, exclude=()):
    """Best capability for a wish, by keyword overlap. Conservative: returns None when nothing
    clearly matches (the caller then keeps it to speech, not autonomous action)."""
    norm = _norm(wish_text)                      # ordered, so multi-word keywords match by adjacency
    words = set(norm.split())
    if not words:
        return None
    best, best_score = None, 0
    for c in active(exclude):
        score = sum(1 for kw in c.keywords if (kw in norm if " " in kw else kw in words))
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= 1 else None


def prompt_lines(exclude=()):
    """The 'WHAT YOU CAN DO' block body: one line per power, tagged with what she may just do vs
    ask about. Persona wraps this with the header + guidance."""
    return [f"{c.blurb} ({_TIER_LABEL[c.tier]})" for c in active(exclude)]


def for_tool(intention=None, exclude=()):
    """Structured answer for the `what_can_i_do` tool. With an intention, lead with the matched
    power; always list them so she can speak about the rest naturally."""
    matched = match(intention, exclude) if intention else None
    return {
        "matched": ({"name": matched.name, "blurb": matched.blurb, "tier": matched.tier}
                    if matched else None),
        "capabilities": [{"intention": c.intention, "blurb": c.blurb, "tier": c.tier}
                         for c in active(exclude)],
        "note": "Answer in a sentence or two, in your own voice — say what you'd do (or that you'd "
                "need to ask first, or that it's not something you do). Don't recite the whole list.",
    }
