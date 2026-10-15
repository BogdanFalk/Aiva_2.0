"""The house's ear: mic gate semantics (deafen, floor-driven speaking, verify) and name matching."""
from modules.ear import MicController, mentions_any_of


def test_muted_blocks_wakeword_but_not_own_announce():
    m = MicController(ambient=True, log=lambda s: None)
    m.muted = True
    m.wake("wake word", preroll=b"\x00\x00", verify=True, hint="hey_saga")
    assert not m.awake and m.wake_hint is None
    m.wake("background news announce")
    assert m.awake


def test_wake_hint_is_kept_and_cleared_on_reject():
    m = MicController(ambient=True, log=lambda s: None)
    m.wake("wake word", preroll=b"\x00\x00", verify=True, hint="hey_saga")
    assert m.awake and m.verifying and not m.visibly_awake and m.wake_hint == "hey_saga"
    m.reject_wake("no name")
    assert not m.awake and m.wake_hint is None
    m.wake("wake word", preroll=b"\x00\x00", verify=True, hint="hey_aiva")
    m.confirm_wake("heard aiva")
    assert m.visibly_awake and m.wake_hint == "hey_aiva"


def test_floor_speaking_gates_the_mic_like_bot_frames():
    m = MicController(ambient=False, bot_cooldown_secs=10.0)
    assert not m._zeroed()
    m.set_speaking(True)
    assert m.bot_speaking and m._zeroed()
    m.set_speaking(False)
    assert not m.bot_speaking and m._zeroed()      # cooldown for the room's audio tail
    m._bot_stopped_at = 0.0
    assert not m._zeroed()


def test_processing_includes_the_floor_busy_predicate():
    state = {"busy": False}
    m = MicController(ambient=True, busy=lambda: state["busy"])
    assert not m.processing
    state["busy"] = True
    assert m.processing


def test_mentions_any_of_variants_and_loose_shape():
    aiva = ("aiva", "ava", "hey va")
    assert mentions_any_of("Hey Aiva, what's up", aiva, r"[aeiy]{1,3}[vwf]ah?")
    assert mentions_any_of("hey va are you there", aiva)
    assert mentions_any_of("eyva come here", aiva, r"[aeiy]{1,3}[vwf]ah?")
    assert not mentions_any_of("the avocado was heavy", aiva, r"[aeiy]{1,3}[vwf]ah?")
    saga = ("saga", "sagah", "sa ga")
    assert mentions_any_of("Saga, hi", saga, r"s[ae]+r?g+a+h?r?")
    assert mentions_any_of("sarga, hi", saga, r"s[ae]+r?g+a+h?r?")
    assert not mentions_any_of("sage advice about sugar", saga, r"s[ae]+r?g+a+h?r?")


def test_wake_word_is_ignored_while_a_sister_is_speaking():
    """The wake listener has its own mic stream; its pre-roll would carry HER voice into STT."""
    m = MicController(ambient=True, log=lambda s: None, bot_cooldown_secs=10.0)
    m.set_speaking(True)
    m.wake("wake word hey_saga", preroll=b"\x00\x00", verify=True, hint="hey_saga")
    assert not m.awake
    m.set_speaking(False)                       # still inside the cooldown
    m.wake("wake word hey_saga", preroll=b"\x00\x00", verify=True, hint="hey_saga")
    assert not m.awake
    m._bot_stopped_at = 0.0
    m.wake("wake word hey_saga", preroll=b"\x00\x00", verify=True, hint="hey_saga")
    assert m.awake
    # her own announce-style wake (no preroll) is never blocked
    m2 = MicController(ambient=True, log=lambda s: None)
    m2.set_speaking(True)
    m2.wake("aiva has the floor")
    assert m2.awake
