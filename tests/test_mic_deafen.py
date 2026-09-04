"""F9 deafen: while muted, the wake word (which carries a preroll) must NOT wake her, but her
own background announces (no preroll) still can, so she can speak up even while deafened."""
from main import MicController


def test_muted_blocks_wakeword():
    m = MicController(ambient=True)
    assert not m.awake
    m.muted = True
    m.wake("wake word (score=0.92)", preroll=b"\x00\x00", verify=True)
    assert not m.awake            # deafened — the wake word is ignored


def test_muted_still_allows_her_own_announce():
    m = MicController(ambient=True)
    m.muted = True
    m.wake("background news announce")   # no preroll -> her own initiative
    assert m.awake                # she can still wake herself to speak


def test_unmuted_wakeword_wakes_normally():
    m = MicController(ambient=True)
    m.wake("wake word (score=0.92)", preroll=b"\x00\x00", verify=True)
    assert m.awake
