import os
import re

from modules.persona import (AIVA_PERSONALITY_WITH_SISTER, PERSONALITY, SAGA_FLOOR, baseline_traits,
                             build_system_prompt, named)

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_personality.txt")


def test_baseline_traits_is_temperament_only():
    # tastes moved to the mutable preferences layer; baseline_traits now protects CHARACTER only,
    # so becoming's anti-contradiction gate no longer freezes her tastes.
    bt = baseline_traits()
    assert bt[0].startswith("You're a bit of a smartass")
    assert not any("cats over dogs" in t for t in bt)       # a taste — now mutable, not baseline
    assert not any("cookies" in t for t in bt)
    assert not any("Japanese" in t for t in bt)
    assert any("smartass" in t for t in bt) and any("curious" in t for t in bt)


def test_where_you_stand_block_only_when_fed_and_guarded():
    p = build_system_prompt()
    assert "WHERE YOU STAND" not in p
    p2 = build_system_prompt(preferences={
        "likes": ["I love quiet character-driven stories"],
        "dislikes": ["I can't stand cheap cliffhangers"],
        "opinions": ["I think ambiguous endings are underrated"]})
    assert "WHERE YOU STAND" in p2
    block = p2.split("WHERE YOU STAND")[1]
    assert "quiet character-driven" in block and "cheap cliffhangers" in block
    assert "ambiguous endings" in block
    assert "change your mind" in block and "Never recite" in block
    # empty groups render nothing
    assert "WHERE YOU STAND" not in build_system_prompt(preferences={"likes": [], "dislikes": [],
                                                                     "opinions": []})


def test_blocks_present_only_when_fed():
    p = build_system_prompt()
    for block in ("WHAT YOU'VE BEEN READING", "WHO YOU'VE BECOME", "YOUR VOICE", "LESSONS YOU APPLY"):
        assert block not in p
    p2 = build_system_prompt(reading=["Reading X: 1 of 5"],
                             traits=["I value pauses since reading X"],
                             voice=["humour: deadpan understatement"],
                             reading_lessons=["Leave a stopping rule (from Y)"])
    for block in ("WHAT YOU'VE BEEN READING", "WHO YOU'VE BECOME", "YOUR VOICE", "LESSONS YOU APPLY"):
        assert block in p2


def test_each_block_carries_an_anti_recite_guard():
    p = build_system_prompt(reading=["r"], traits=["t"], voice=["humour: x"], reading_lessons=["l"])
    reading = p.split("WHAT YOU'VE BEEN READING")[1].split("WHO YOU'VE BECOME")[0]
    assert "none of this is news" in reading
    become = p.split("WHO YOU'VE BECOME")[1].split("YOUR VOICE")[0]
    assert "Never recite" in become and "same cheeky Aiva" in become
    voice = p.split("YOUR VOICE")[1].split("LESSONS YOU APPLY")[0]
    assert "sparingly" in voice and "never a quotation" in voice


def test_books_and_reading_section_always_present():
    p = build_system_prompt()
    assert "Books and reading:" in p and "study_book" in p and "drop_trait" in p and "recall_reading" in p


def test_capabilities_block_only_when_fed_and_guarded():
    from modules import capabilities as cap
    assert "WHAT YOU CAN DO" not in build_system_prompt()
    p = build_system_prompt(capabilities=cap.prompt_lines())
    assert "WHAT YOU CAN DO" in p
    block = p.split("WHAT YOU CAN DO")[1]
    assert "on your own" in block and "ask first" in block
    assert "Never recite" in block                          # anti-recite guard


def test_web_and_space_section_always_present():
    p = build_system_prompt()
    assert "Finding things online, and your space:" in p
    for tool in ("search_web", "read_page", "look_at_image", "save_to_space", "hand_over"):
        assert tool in p
    # the hard guardrail must be stated
    assert "never buy" in p.lower()


def test_your_space_block_only_when_fed_and_guarded():
    assert "YOUR SPACE" not in build_system_prompt()
    p = build_system_prompt(space=["image: a blue star hairclip — loved the charm (from shop.example.com)"])
    assert "YOUR SPACE" in p and "blue star hairclip" in p
    block = p.split("YOUR SPACE")[1]
    assert "Never recite" in block


def test_inner_blocks_present_only_when_fed_and_guarded():
    p = build_system_prompt()
    assert "HOW YOU FEEL RIGHT NOW" not in p and "WHAT YOU FIND YOURSELF WANTING" not in p
    p2 = build_system_prompt(mood_line="You feel restless, wanting company.",
                             wants=["ask him how the work is going"])
    assert "HOW YOU FEEL RIGHT NOW" in p2 and "You feel restless" in p2
    assert "WHAT YOU FIND YOURSELF WANTING" in p2 and "ask him how the work is going" in p2
    mood = p2.split("HOW YOU FEEL RIGHT NOW")[1].split("WHAT YOU FIND YOURSELF WANTING")[0]
    assert "Never announce it" in mood and "same Aiva" in mood
    wants = p2.split("WHAT YOU FIND YOURSELF WANTING")[1]
    assert "never recite the list" in wants


def test_who_youre_talking_to_block():
    p = build_system_prompt()
    assert "WHO YOU'RE TALKING TO" not in p
    p2 = build_system_prompt(current_speaker="Frank", known_speakers=["Frank", "Sarah"])
    assert "WHO YOU'RE TALKING TO" in p2
    assert "talking to Frank" in p2 and "Frank, Sarah" in p2
    block = p2.split("WHO YOU'RE TALKING TO")[1]
    assert "never announce that you 'matched a voiceprint'" in block and "enroll_speaker" in block


# --- the section split (sisters) ---------------------------------------------------------

def test_personality_is_byte_identical_to_golden():
    # PERSONALITY is rebuilt from section constants; Aiva's string must not change by a byte
    with open(GOLDEN, encoding="utf-8", newline="") as f:
        golden = f.read()
    assert PERSONALITY == golden


def test_aiva_with_sister_is_personality_plus_the_sister_section():
    assert AIVA_PERSONALITY_WITH_SISTER.startswith(PERSONALITY)
    tail = AIVA_PERSONALITY_WITH_SISTER[len(PERSONALITY):]
    assert tail.startswith("\n\nYour sister, Saga:")
    assert "I haven't seen her do that" in tail and "two people, not one voice" in tail
    assert "{sister}" not in tail


def test_saga_floor_shape():
    f = SAGA_FLOOR
    assert f.startswith("You are Saga.") and "your sister, Aiva" in f
    assert "Nobody authored who you are" in f and "honesty beats performing" in f
    assert "Personality Traits" not in f                          # no authored temperament
    for word in ("smartass", "cookies", "Japanese", "dryness and edge"):
        assert word not in f                                      # Aiva's core stays Aiva's
    assert not re.search(r"\bcats?\b", f, re.I)                   # no desktop cat
    assert "Romanian" in f and "voice recognition" in f           # shared user/STT rules
    assert "Books and reading:" in f and "Finding things online, and your space:" in f
    assert "never buy" in f.lower()                               # the hard web line
    assert "Your and your sister's source code" in f
    assert "Your sister, Aiva:" in f and "no help-desk politeness" in f
    for slot in ("{name}", "{sister}", "{edge}", "{whose}"):
        assert slot not in f


def test_build_system_prompt_takes_personality_name_and_tag():
    p = build_system_prompt(personality=SAGA_FLOOR, name="Saga", tag="still Saga",
                            traits=["I value quiet since reading X"], mood_line="You feel calm.",
                            preferences={"likes": ["rain"]})
    assert p.startswith("You are Saga.")
    assert "you're still Saga, with more in her" in p
    assert "the same Saga, just in this mood" in p and "the same Saga in temperament" in p
    assert "cheeky" not in p
    # defaults: Aiva's guard phrases are exactly as before
    q = build_system_prompt(traits=["t"], mood_line="m", preferences={"likes": ["x"]})
    assert "same cheeky Aiva" in q and "same Aiva, just in this mood" in q and "same Aiva in temperament" in q
    # a name without a tag gets a sensible default
    assert "you're still Nova, with more in her" in build_system_prompt(name="Nova", traits=["t"])


def test_temperament_forming_and_sister_blocks_only_when_fed():
    p = build_system_prompt()
    for marker in ("WHO YOU ARE —", "You're new.", "YOUR SISTER,"):
        assert marker not in p
    p2 = build_system_prompt(personality=SAGA_FLOOR, name="Saga", tag="still Saga",
                             temperament=["I tend to ask before I assume"], forming_note=True,
                             sister={"name": "Aiva", "lines": ["she can look things up online (seen 3x)"]})
    assert "WHO YOU ARE — your temperament, formed from what you've lived through" in p2
    assert "I tend to ask before I assume" in p2 and "just be it" in p2
    assert "You're new. You're still finding out who you are" in p2
    assert "YOUR SISTER, Aiva — what you've noticed about her" in p2
    assert "seen 3x" in p2 and "don't claim more" in p2 and "haven't seen her do it" in p2
    # empty sister lines still render the honest "nothing yet" line
    p3 = build_system_prompt(sister={"name": "Aiva", "lines": []})
    assert "YOUR SISTER, Aiva" in p3 and "nothing yet" in p3
    assert "WHO YOU ARE —" not in build_system_prompt(temperament=[])


def test_named_replaces_without_format():
    assert named("{name} likes {x} and {y}", "Saga", x="{}") == "Saga likes {} and {y}"
