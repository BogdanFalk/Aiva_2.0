"""Upgrades: she is told what changed in her once, only when it's really on, and can still say
what's new for two weeks after."""
from datetime import date, timedelta
from types import SimpleNamespace

from modules import persona, upgrades
from modules.character import KNOWN
from modules.upgrades import UPGRADES, Upgrade

TODAY = date(2026, 10, 23)


def sister(slug="aiva", llm="gpt-5.6-luna", voice="v1"):
    return SimpleNamespace(slug=slug, llm_model=llm, voice_id=voice)


def ids(ups):
    return [u.id for u in ups]


def test_entries_are_speakable_and_well_formed():
    assert len({u.id for u in UPGRADES}) == len(UPGRADES)            # ids are never reused
    for u in UPGRADES:
        assert u.who and set(u.who) <= set(KNOWN)
        assert u.told.strip() == u.told and u.told.endswith(".")
        assert not any(c in u.told for c in "*#`_\n[]")              # spoken aloud: no markup


def test_gpt6_is_told_only_to_the_sister_running_it():
    env = {"AIVA_LLM_MODEL": "gpt-6-luna", "SAGA_LLM_MODEL": "gpt-5.6-luna"}
    aiva = ids(upgrades.applicable("aiva", env, sister("aiva", "gpt-6-luna")))
    saga = ids(upgrades.applicable("saga", env, sister("saga", "gpt-5.6-luna")))
    assert "2026-10-gpt6-luna" in aiva and "2026-10-sister-gpt6" not in aiva
    assert "2026-10-gpt6-luna" not in saga and "2026-10-sister-gpt6" in saga
    # once Saga moves too, she gets her own upgrade and the note about her sister goes away
    both = ids(upgrades.applicable("saga", {"AIVA_LLM_MODEL": "gpt-6-luna"}, sister("saga", "gpt-6-luna")))
    assert "2026-10-gpt6-luna" in both and "2026-10-sister-gpt6" not in both


def test_nothing_about_gpt6_while_the_env_says_otherwise():
    on = ids(upgrades.applicable("aiva", {"AIVA_LLM_MODEL": "gpt-5.6-luna"}, sister()))
    assert "2026-10-gpt6-luna" not in on and "2026-10-sister-gpt6" not in on


def test_house_features_follow_their_switches():
    on = ids(upgrades.applicable("aiva", {}, sister()))            # defaults: flux, emotion, mhm on
    assert {"2026-10-flux-ear", "2026-10-emotion-hearing", "2026-10-backchannels"} <= set(on)
    off = ids(upgrades.applicable("aiva", {"AIVA_STT": "nova", "AIVA_EMOTION_HEARING": "0"}, sister()))
    assert not {"2026-10-flux-ear", "2026-10-emotion-hearing", "2026-10-backchannels"} & set(off)
    assert "2026-10-backchannels" not in ids(upgrades.applicable("aiva", {"AIVA_BACKCHANNEL": "0"}, sister()))
    assert "2026-10-backchannels" not in ids(upgrades.applicable("aiva", {}, sister(voice=None)))


def test_told_record_round_trips_and_tolerates_garbage():
    for raw in (None, "", "not json", "[1, 2]", "42"):
        assert upgrades.load_told(raw) == {}
    told = upgrades.mark_told({}, UPGRADES[:2], TODAY)
    assert upgrades.load_told(upgrades.dump_told(told)) == told
    later = upgrades.mark_told(told, UPGRADES[:3], TODAY + timedelta(days=5))
    assert later[UPGRADES[0].id] == TODAY.isoformat()             # the first day is kept
    assert later[UPGRADES[2].id] == (TODAY + timedelta(days=5)).isoformat()


def test_pending_and_recent():
    a, b, c = (Upgrade(x, ("aiva",), f"{x}.") for x in "abc")
    told = {"a": (TODAY - timedelta(days=20)).isoformat(), "b": (TODAY - timedelta(days=3)).isoformat()}
    assert ids(upgrades.pending([a, b, c], told)) == ["c"]
    assert ids(upgrades.recent([a, b, c], told, TODAY)) == ["b", "c"]   # a is old news
    assert ids(upgrades.recent([a], {"a": "garbled"}, TODAY)) == []


def test_boot_note_carries_every_new_upgrade_once():
    assert upgrades.boot_note([]) is None
    new = upgrades.applicable("aiva", {"AIVA_LLM_MODEL": "gpt-6-luna"}, sister(llm="gpt-6-luna"))
    note = upgrades.boot_note(new)
    assert note.startswith("Since you last ran, you were upgraded.")
    for u in new:
        assert note.count(u.told) == 1
    assert "own words" in note


def test_prompt_block_renders_only_when_there_is_news():
    assert "WHAT'S NEW IN YOU" not in persona.build_system_prompt()
    assert persona.build_system_prompt(upgrades=[]) == persona.build_system_prompt()
    text = persona.build_system_prompt(upgrades=["You can hear how he sounds now."])
    assert "WHAT'S NEW IN YOU" in text and "- You can hear how he sounds now." in text
