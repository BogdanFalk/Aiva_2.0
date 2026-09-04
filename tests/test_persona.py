from modules.persona import baseline_traits, build_system_prompt


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
