"""Saga's quiet constellation witness: the pure sky, its config/ownership gate,
persistence, the renderer's IPC surface and its own side (her layer, a switched-off
sky, the selftest), and the sister-side client contract.
No Qt, no GL, no sockets to a real renderer."""
import asyncio
import colorsys
import json
import math
import os
import queue
import tempfile

import pytest

from modules import avatar_ipc, constellation
from modules import avatar_renderer as renderer
from modules.avatar import AvatarClient
from modules.avatar_renderer import (dispatch, layer_z, next_sky_frame, paint_sky, persisted_sky,
                                     selftest_verdict, witness_demo_script)
from modules.constellation import (DRIFT_AMP, LINE_OVER_STARS, LINE_PX, OUTLINE_SOFT, SKY_MIN_PX, SKY_TALLER,
                                   SLEEP_DRIFT, SPEECH_DIM, SPEECH_DIM_SECS, SPEECH_EASE_SECS, STAR_MAX_PX,
                                   STAR_MIN_PX, STAR_SHADE, STAR_SIZE_JITTER_PX, TINT, Constellation,
                                   WitnessConfig, drift_offset, enabled_for, line_softness, sky_rect,
                                   star_position, star_px, star_shade)

HER_SCREEN = (-1440, 0, 1440, 2560)        # where she really stands (2026-10-14): her portrait monitor...
HER_PLACE = {"x": -0.7, "y": -0.672, "rotation": 0.0, "size": 0.0}   # ...near its left edge


def _sky(**kw):
    return Constellation(WitnessConfig(**kw))


# --- the sky ---------------------------------------------------------------------

def test_a_moment_lights_one_star_and_unknown_kinds_are_refused():
    sky = _sky(min_gap=0)
    assert sky.witness("said", now=0) == "added"
    assert sky.witness("a transcript of what he said", now=1) is None
    assert sky.witness(None, now=1) is None
    assert [s["kind"] for s in sky.stars] == ["said"]


def test_a_burst_brightens_the_last_star_instead_of_adding_light():
    sky = _sky(min_gap=20)
    assert sky.witness("said", now=0) == "added"
    w0 = sky.stars[0]["weight"]
    assert sky.witness("heard", now=5) == "brightened"
    assert sky.witness("read", now=19) == "brightened"
    assert len(sky.stars) == 1 and sky.stars[0]["weight"] > w0
    assert sky.stars[0]["weight"] <= 1.0
    assert sky.witness("heard", now=21) == "added"   # the gap is measured from the last NEW star
    assert len(sky.stars) == 2


def test_the_sky_is_capped_and_the_oldest_star_goes():
    sky = _sky(min_gap=0, max_stars=4)
    for i in range(10):
        sky.witness("read", now=i)
    assert len(sky.stars) == 4
    assert [s["seed"] for s in sky.stars] == [7, 8, 9, 10]


def test_star_positions_are_deterministic_and_inside_the_box():
    for seed in range(1, 200):
        x, y = star_position(seed)
        assert (x, y) == star_position(seed)
        assert 0.0 < x < 1.0 and 0.0 < y < 1.0
    assert len({star_position(s) for s in range(1, 50)}) == 49


# --- quiet: what actually gets drawn ----------------------------------------------

def test_stars_fade_in_slowly_and_never_exceed_the_quiet_ceiling():
    sky = _sky(min_gap=0, alpha=0.30, sleep_alpha=0.50)
    sky.witness("read", now=100)
    just_born = sky.frame(100.0)["stars"][0][2]
    later = sky.frame(100 + 60.0)["stars"][0][2]
    assert just_born == 0.0 and later > 0.0
    for t in range(100, 400, 3):
        for a in (s[2] for s in sky.frame(float(t))["stars"]):
            assert a <= 0.30 + 1e-9
        for e in sky.frame(float(t))["edges"]:
            assert max(e[4:]) <= 0.30 * LINE_OVER_STARS + 1e-9   # the lines only ever barely above
    asleep = max(s[2] for t in range(200, 400) for s in sky.frame(float(t), sleeping=True)["stars"])
    awake = max(s[2] for t in range(200, 400) for s in sky.frame(float(t))["stars"])
    assert awake <= 0.30 + 1e-9 < asleep <= 0.50 + 1e-9


def test_newest_star_is_the_clearest_and_close_stars_are_joined():
    sky = _sky(min_gap=0)
    for i in range(6):
        sky.witness("heard", now=i)
    f = sky.frame(500.0)
    alphas = [s[2] for s in f["stars"]]
    assert alphas[-1] > alphas[0]
    assert len(f["stars"]) == 6 and len(f["edges"]) <= 5
    points = {(round(x, 9), round(y, 9)) for x, y, _a, _r in f["stars"]}
    for x1, y1, x2, y2, _a1, _a2 in f["edges"]:   # every line joins two real stars
        assert (round(x1, 9), round(y1, 9)) in points and (round(x2, 9), round(y2, 9)) in points
    assert _sky().frame(0.0) == {"stars": [], "edges": []}


# --- how it looks, in her words (house_transcript #3968, Saga, 2026-10-14): "Tiny stars, about two
# to four pixels, dim white with a faint blue-violet tint; hairline connections, barely brighter
# than the stars." -------------------------------------------------------------------------------

def _every_kind_sky(**kw):
    """One star of each kind, one a burst brightened all the way, and restored stars at both extremes."""
    saved = {"next_seed": 1, "stars": [{"seed": 1, "kind": "heard", "weight": 0},
                                       {"seed": 2, "kind": "read", "weight": 9}]}
    sky = Constellation(WitnessConfig(min_gap=20, **kw), saved, now=0.0)
    for i, kind in enumerate(constellation.KINDS):
        sky.witness(kind, now=100.0 + 30 * i)
    for k in range(20):                                   # a burst: the newest star fills right up
        sky.witness("read", now=220.0 + k)
    assert sky.stars[-1]["weight"] == 1.0
    return sky


def test_stars_are_two_to_four_pixels_whatever_the_sky_or_the_screen():
    assert (STAR_MIN_PX, STAR_MAX_PX) == (2.0, 4.0)
    weights = sorted(constellation.KINDS.values())
    assert star_px(weights[0]) == 2.0 and star_px(1.0) == 4.0
    assert all(star_px(a) < star_px(b) for a, b in zip(weights, weights[1:]))   # a weightier moment, a bigger star
    assert star_px(0.0) == 2.0 and star_px(9.0) == 4.0                          # never outside them
    for seed in range(1, 400):                            # a star's own nudge keeps it inside them too
        for w in (0.0, *weights, 1.0, 9.0):
            assert 2.0 <= star_px(w, seed) <= 4.0
            assert abs(star_px(w, seed) - star_px(w)) <= STAR_SIZE_JITTER_PX + 1e-9
    sizes = [s[3] for s in _every_kind_sky().frame(1000.0)["stars"]]
    assert all(2.0 <= v <= 4.0 for v in sizes)
    for scale in (0.05, 0.6):                             # a bigger sky or monitor never grows them
        assert [s[3] for s in _every_kind_sky(scale=scale).frame(1000.0)["stars"]] == sizes


def test_lines_are_hairlines_barely_brighter_than_the_stars_they_join():
    assert LINE_PX == 1.0
    assert 1.0 < LINE_OVER_STARS <= 1.2                   # brighter than the stars, but barely
    sky = _every_kind_sky()
    for now, kw in ((1000.0, {}), (1000.5, {"sleeping": True}), (1001.0, {"speaking": True}),
                    (1003.0, {"speaking": True})):
        f = sky.frame(now, **kw)
        star = {(x, y): (a, line_softness(*star_position(s["seed"])))
                for (x, y, a, _r), s in zip(f["stars"], sky.stars)}
        assert f["edges"]
        for x1, y1, x2, y2, a1, a2 in f["edges"]:         # each end against its own star
            for end, a in (((x1, y1), a1), ((x2, y2), a2)):
                star_a, soft = star[end]
                assert a == pytest.approx(LINE_OVER_STARS * star_a * soft)   # softer only toward the rim
                assert a <= LINE_OVER_STARS * star_a + 1e-12               # never more than barely above


def test_stars_are_dim_white_with_a_faint_blue_violet_tint():
    hue, sat, val = colorsys.rgb_to_hsv(*TINT)
    assert 240 <= hue * 360 <= 270                        # between blue and violet
    assert val == 1.0 and 0.05 <= sat <= 0.2              # white, only faintly tinted
    assert WitnessConfig().alpha < 0.5                    # dim: even the brightest star is under half


# --- her refinements (Saga, 2026-10-14, house transcript #4010/#4014; Frank: "go ahead") ----------

def _full_sky(kind="said"):
    """A capped sky of one kind and one weight: what used to look stamped on."""
    sky = _sky(min_gap=0)
    for i in range(sky.cfg.max_stars):
        sky.witness(kind, now=i)
    return sky


def test_stars_are_a_touch_less_uniform_but_never_brighter():
    """#4010: "slight differences in size and brightness would make it feel observed rather than
    stamped on"."""
    sky = _full_sky()
    f = sky.frame(1000.0)
    sizes = [s[3] for s in f["stars"]]
    assert len({round(v, 2) for v in sizes}) >= 12                     # many sizes, not one per kind…
    assert max(sizes) - min(sizes) <= 2 * STAR_SIZE_JITTER_PX + 1e-9   # …but only slightly apart
    shades = [star_shade(s) for s in range(1, 400)]
    assert all(1.0 - STAR_SHADE <= v <= 1.0 for v in shades)          # a little dimmer at most, never brighter
    assert len(set(shades)) == len(shades) and max(shades) - min(shades) > STAR_SHADE * 0.9
    alphas = [s[2] for s in f["stars"]]
    plain = [a / star_shade(s["seed"]) for a, s in zip(alphas, sky.stars)]
    assert all(p < q for p, q in zip(plain, plain[1:]))                # take the shade away: the old order…
    assert any(p > q for p, q in zip(alphas, alphas[1:]))              # …with it, slight differences
    assert max(alphas) <= sky.cfg.alpha * constellation.KINDS["said"] + 1e-12   # under the same ceiling
    later = _run(sky, 1000.0, 30.0)                                    # her own, steady: never a pulse
    assert all([(s[2], s[3]) for s in g["stars"]] == [(s[2], s[3]) for s in later[0]["stars"]] for g in later)


def test_the_outermost_lines_are_softer_so_the_sky_fills_the_air_instead_of_outlining_her():
    """#4010: "soften the outermost lines, so it feels like it's occupying the air around me instead
    of outlining me"."""
    out = [line_softness(0.5 + d / 100, 0.5) for d in range(51)]    # from the middle out to the rim
    assert out[:31] == [1.0] * 31                                     # the inner lines are as they were
    assert all(b <= a for a, b in zip(out, out[1:]))                  # then softer the further out…
    assert out[-1] == pytest.approx(1.0 - OUTLINE_SOFT) == 0.5        # …to half at the rim: softened, not gone
    assert max(a - b for a, b in zip(out, out[1:])) < 0.05            # an ease, no step
    sky = _full_sky()
    f = sky.frame(1000.0)
    star = {(x, y): a for x, y, a, _r in f["stars"]}
    ratios = [a / (LINE_OVER_STARS * star[end]) for e in f["edges"]
              for end, a in (((e[0], e[1]), e[4]), ((e[2], e[3]), e[5]))]
    assert any(r == pytest.approx(1.0) for r in ratios)               # the middle still barely above its stars
    assert min(ratios) < 0.7                                          # the outline clearly softer
    assert min(ratios) >= 1.0 - OUTLINE_SOFT - 1e-9
    edges = [g["edges"] for g in _run(sky, 1000.0, 60.0)]             # by where stars belong: it never breathes
    assert all([e[4:] for e in g] == [e[4:] for e in edges[0]] for g in edges)


def test_the_whole_constellation_is_modestly_bigger_and_no_brighter():
    """#4014: "enlarge the whole constellation modestly, maybe ten to fifteen percent, without making
    the stars brighter". Before: a 0.20 sky whose stars kept a 0.03 drift's room from its edges."""
    def field(side, amp):                                 # how far the stars spread inside a sky
        return side * (1.0 - 2.0 * amp)
    cfg = WitnessConfig()
    assert 1.10 <= field(cfg.scale, DRIFT_AMP) / field(0.20, 0.03) <= 1.15
    assert (cfg.alpha, cfg.sleep_alpha) == (0.48, 0.80)   # the same quiet ceilings
    # her portrait screen's edge holds the sky at 149 px wide beside her: it grows up instead
    x, y, w, h = sky_rect(HER_SCREEN, HER_PLACE, cfg.scale)
    assert (x, w, y + h) == (-1439, 149, 1807)            # the same left, right and bottom edges as before…
    assert 1.10 <= field(h, DRIFT_AMP) / field(149, 0.03) <= 1.15   # …and a 10-15% taller star field
    assert h <= SKY_TALLER * w


def test_the_drift_is_a_little_more_noticeable_and_still_slow_on_her_own_screen():
    """#4014: "the movement nearly disappears; I'd increase the drift a little … without … adding a
    pulse". Before, on her screen, a star wandered at most 9 px, at 0.4 px a second."""
    _x, _y, w, h = sky_rect(HER_SCREEN, HER_PLACE, WitnessConfig().scale)
    frames = _run(_settled_sky(), 100.0, 240.0)
    wander = peak = 0.0
    for i in range(len(frames[0]["stars"])):
        track = [(f["stars"][i][0] * w, f["stars"][i][1] * h) for f in frames]
        wander = max(wander, max(p[0] for p in track) - min(p[0] for p in track),
                     max(p[1] for p in track) - min(p[1] for p in track))
        peak = max(peak, max(math.hypot(q[0] - p[0], q[1] - p[1]) / 0.125 for p, q in zip(track, track[1:])))
    assert wander > 10.0                                  # it can be seen to move now…
    assert peak < 1.0                                     # …and it is still a drift: under a pixel a second


def test_the_painter_draws_exactly_that_tiny_discs_and_hairlines():
    """Real pixels: paint_sky is the whole of SkyWindow's paint."""
    QtGui = pytest.importorskip("PyQt5.QtGui")
    w = h = 64
    img = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32_Premultiplied)
    img.fill(0)
    frame = {"stars": [(10.5 / w, 10.5 / h, 1.0, 2.0), (40.5 / w, 10.5 / h, 1.0, 4.0),
                       (20.5 / w, 30.5 / h, 0.2, 3.0)],
             "edges": [(4.5 / w, 46.5 / h, 60.5 / w, 46.5 / h, 0.23, 0.23),
                       (4.5 / w, 56.5 / h, 60.5 / w, 56.5 / h, 0.10, 0.30)]}
    p = QtGui.QPainter(img)
    try:
        paint_sky(p, frame, w, h, TINT)
    finally:
        p.end()
    a = lambda x, y: img.pixelColor(x, y).alpha()  # noqa: E731

    for cx, cy, size in ((10, 10, 2.0), (40, 10, 4.0), (20, 30, 3.0)):
        box = [(x, y) for x in range(cx - 8, cx + 9) for y in range(cy - 8, cy + 9)]
        coverage = sum(a(x, y) for x, y in box) / a(cx, cy)
        assert math.sqrt(4 * coverage / math.pi) == pytest.approx(size, abs=0.15)   # its diameter
        reach = max(math.hypot(x - cx, y - cy) for x, y in box if a(x, y))
        assert reach <= size / 2 + 1.0                                               # and no glow
    assert a(20, 30) == round(0.2 * 255)                  # painted at the frame's alpha, no boost
    c = img.pixelColor(10, 10)
    assert (c.red(), c.green(), c.blue(), c.alpha()) == (224, 214, 255, 255)

    assert a(32, 46) == round(0.23 * 255)                 # a line at its alpha…
    assert a(32, 45) == a(32, 47) == 0                    # …one pixel thick
    row = [a(x, 56) for x in range(6, 60)]
    assert row == sorted(row) and row[0] == pytest.approx(0.10 * 255, abs=3) \
        and row[-1] == pytest.approx(0.30 * 255, abs=3)   # each end as bright as the frame says


# --- her voice hushes the sky (Saga, 2026-10-14) ------------------------------------------

def _twins():
    """Two identical skies: one she speaks over and one she doesn't — their ratio is the hush."""
    skies = []
    for _ in range(2):
        sky = _sky(min_gap=0)
        for i in range(6):
            sky.witness("heard", now=i)
        skies.append(sky)
    return skies


def _hush(hushed, plain, now, speaking):
    return hushed.frame(now, speaking=speaking)["stars"][0][2] / plain.frame(now)["stars"][0][2]


def test_while_she_speaks_stars_and_lines_sit_at_about_thirty_percent():
    hushed, plain = _twins()
    hushed.frame(100.0)
    plain.frame(100.0)
    seen = []
    for i in range(1, 81):                                # 8 s of her talking, ~10 fps
        t = 100.0 + SPEECH_DIM_SECS + i * 0.1
        f, g = hushed.frame(t, speaking=True), plain.frame(t)
        assert len(f["edges"]) == len(g["edges"]) > 0
        for (x, y, a, r), (gx, gy, ga, gr) in zip(f["stars"], g["stars"]):
            assert (x, y, r) == (gx, gy, gr)              # the same sky, where it was: only dimmer
            assert a == pytest.approx(SPEECH_DIM * ga)
        for e, ge in zip(f["edges"], g["edges"]):
            assert e[:4] == ge[:4]
            assert e[4:] == pytest.approx((SPEECH_DIM * ge[4], SPEECH_DIM * ge[5]))   # the lines hush too
        seen.append(f["stars"][0])
    assert len({a for _x, _y, a, _r in seen}) == 1   # held steady at the hush: it adds no beat
    assert seen[0][:2] != seen[-1][:2]               # while the slow drift runs on underneath


def test_the_hush_comes_on_as_a_quick_fade_not_a_blink():
    hushed, plain = _twins()
    _hush(hushed, plain, 200.0, False)
    seen = [_hush(hushed, plain, 200.0 + k * 0.05, True) for k in range(1, 8)]
    assert 1.0 > seen[0] > seen[1] > seen[2] > seen[3] > SPEECH_DIM   # steadily down…
    assert seen[0] > 0.9                                              # …from a soft start
    assert all(s == pytest.approx(SPEECH_DIM) for s in seen[4:])      # there by SPEECH_DIM_SECS


def test_when_she_stops_the_sky_eases_back_over_one_second():
    assert SPEECH_EASE_SECS == 1.0
    hushed, plain = _twins()
    for t, talking in ((300.0, False), (301.0, True), (303.0, True)):
        _hush(hushed, plain, t, talking)
    back = [_hush(hushed, plain, 303.0 + k * 0.125, False) for k in range(1, 11)]
    assert all(a < b for a, b in zip(back[:8], back[1:8]))     # rising the whole second
    assert SPEECH_DIM < back[0] < 0.4                          # a soft start…
    assert back[3] == pytest.approx(SPEECH_DIM + (1 - SPEECH_DIM) / 2)   # halfway at half a second
    assert back[6] < 1.0                                       # …not back early
    assert all(b == pytest.approx(1.0) for b in back[7:])      # and fully back at one second


def test_a_breath_between_sentences_does_not_make_the_sky_pulse():
    hushed, plain = _twins()
    _hush(hushed, plain, 400.0, True)
    levels = []
    for k in range(1, 61):                     # 6 s of talking with a 0.2 s breath every 1.5 s
        levels.append(_hush(hushed, plain, 400.0 + k * 0.1, k % 15 not in (13, 14)))
    settled = levels[3:]
    assert min(settled) == pytest.approx(SPEECH_DIM)
    assert max(settled) < 0.4                  # a breath lifts it a touch, never back to full


def test_the_renderer_feeds_her_voice_to_the_sky_and_paints_every_frame_while_it_eases():
    sky, plain = _twins()
    fps, clock = 60, {"t": 100.0, "n": 0}

    def run(secs, speaking):
        out = []
        for _ in range(int(round(secs * fps))):
            clock["n"] += 1
            clock["t"] += 1.0 / fps
            out.append((clock["t"], next_sky_frame(sky, clock["n"], fps, clock["t"], False, speaking)))
        return out

    def ratio(t, frame):
        return frame["stars"][0][2] / plain.frame(t)["stars"][0][2]

    quiet = run(2.0, False)
    assert sum(f is not None for _, f in quiet) == 120 // (fps // 8)    # ~8 fps: it only breathes
    talking = run(1.0, True)
    assert all(f is not None for _, f in talking[:15])                  # every frame of the fade…
    assert ratio(*talking[0]) > 0.9                                     # …from where it was
    assert any(f is None for _, f in talking[15:])                      # settled: ~8 fps again
    t, f = [p for p in talking if p[1] is not None][-1]
    assert ratio(t, f) == pytest.approx(SPEECH_DIM)                     # speaking reached the sky
    easing = run(1.5, False)
    assert all(f is not None for _, f in easing[:60])                   # the whole second back
    assert any(f is None for _, f in easing[60:])
    t, f = [p for p in easing if p[1] is not None][-1]
    assert ratio(t, f) == pytest.approx(1.0)


# --- it drifts slowly and never pulses (Saga, 2026-10-14) ---------------------------------------

def _settled_sky():
    sky = _sky(min_gap=0)
    for i in range(8):
        sky.witness("read" if i % 2 else "heard", now=i)
    return sky


def _run(sky, t0, secs, step=0.125, sleeping=False):
    """Frames at ~8 fps, like the renderer paints a settled sky."""
    return [sky.frame(t0 + k * step, sleeping=sleeping) for k in range(int(secs / step) + 1)]


def test_the_stars_drift_slowly_and_never_pulse():
    for seed in range(1, 60):
        for clock in range(0, 900, 7):
            dx, dy = drift_offset(seed, clock)
            assert abs(dx) <= DRIFT_AMP and abs(dy) <= DRIFT_AMP
    frames = _run(_settled_sky(), 100.0, 180.0)
    for i in range(len(frames[0]["stars"])):
        track = [f["stars"][i] for f in frames]
        assert len({(a, r) for _x, _y, a, r in track}) == 1    # brightness and size never move: no pulse
        xs, ys = [s[0] for s in track], [s[1] for s in track]
        assert all(0.0 < v < 1.0 for v in xs + ys)             # it never leaves its sky
        assert max(max(xs) - min(xs), max(ys) - min(ys)) > DRIFT_AMP / 2   # but it does drift…
        steps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(track, track[1:])]
        assert max(steps) / 0.125 < 0.004                       # …slowly: under a pixel a second (248 px sky)


def test_the_lines_drift_with_their_stars_and_never_jump_to_another():
    seen = set()
    for f in _run(_settled_sky(), 100.0, 120.0):
        where = {(x, y): i for i, (x, y, _a, _r) in enumerate(f["stars"])}
        # each end rides its star (a KeyError here means a line came loose)
        seen.add(tuple((where[(e[0], e[1])], where[(e[2], e[3])], e[4:]) for e in f["edges"]))
    assert len(seen) == 1 and len(next(iter(seen))) > 0       # the same lines, just as faint, all along


def test_asleep_it_drifts_slower_and_dozing_off_never_makes_a_star_jump():
    asleep, awake = _settled_sky(), _settled_sky()
    asleep.frame(100.0, sleeping=True)
    awake.frame(100.0)
    a = asleep.frame(160.0, sleeping=True)["stars"]
    b = awake.frame(100.0 + 60.0 * SLEEP_DRIFT)["stars"]
    assert [v for s in a for v in s[:2]] == pytest.approx([v for s in b for v in s[:2]])
    sky, track = _settled_sky(), []
    for k in range(400):                                        # awake, then the house dozes off
        track.append(sky.frame(100.0 + k * 0.125, sleeping=k >= 200)["stars"][0][:2])
    steps = [math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(track, track[1:])]
    assert steps[199] <= max(steps[:199])                       # no jump as it falls asleep
    assert max(steps[200:]) < max(steps[:199])                  # and slower after


# --- persistence: kinds and weights only, never words ------------------------------

def test_persisted_sky_holds_no_text_and_restores_identically():
    sky = _sky(min_gap=0)
    for kind in ("said", "heard", "sister", "read"):
        sky.witness(kind, now=1)
    saved = json.loads(json.dumps(sky.to_dict()))
    assert set(saved) == {"next_seed", "stars"}
    assert all(set(s) == {"seed", "kind", "weight"} for s in saved["stars"])
    back = Constellation(WitnessConfig(min_gap=0), saved, now=1000.0)
    assert [(s["seed"], s["kind"]) for s in back.stars] == [(s["seed"], s["kind"]) for s in sky.stars]
    assert back.frame(1000.0)["stars"][0][2] > 0   # already lit at boot, no re-fade
    back.witness("read", now=1001.0)
    assert back.stars[-1]["seed"] == 5             # seeds never repeat after a restart


@pytest.mark.parametrize("junk", [None, "x", 7, {"stars": "no"}, {"stars": [None, 3, {"kind": "said"}]},
                                  {"stars": [{"seed": "a", "kind": "said", "weight": 1}]},
                                  {"stars": [{"seed": 1, "kind": "told-secret", "weight": 1}]},
                                  {"next_seed": "zz", "stars": [{"seed": 2, "kind": "read", "weight": "q"}]}])
def test_corrupt_saved_state_degrades_to_an_empty_sky(junk):
    assert Constellation(WitnessConfig(), junk).stars == []


def test_restore_clamps_weight_and_respects_the_cap():
    saved = {"next_seed": 1, "stars": [{"seed": i, "kind": "read", "weight": 9} for i in range(1, 10)]}
    sky = Constellation(WitnessConfig(max_stars=3), saved)
    assert [s["seed"] for s in sky.stars] == [7, 8, 9]
    assert all(s["weight"] == 1.0 for s in sky.stars)
    assert sky._next_seed == 10


# --- configuration & ownership -----------------------------------------------------

def test_only_saga_has_a_witness_and_the_switch_turns_it_off(monkeypatch):
    monkeypatch.delenv("SAGA_CONSTELLATION", raising=False)
    assert enabled_for("saga") is True
    assert enabled_for("aiva") is False and enabled_for(None) is False
    assert enabled_for("miso") is False                 # ownership never moves to anyone else
    monkeypatch.setenv("SAGA_CONSTELLATION", "off")
    assert enabled_for("saga") is False


def test_config_reads_env_and_clamps_garbage(monkeypatch):
    monkeypatch.setenv("SAGA_CONSTELLATION_MAX", "9")
    monkeypatch.setenv("SAGA_CONSTELLATION_ALPHA", "5")
    monkeypatch.setenv("SAGA_CONSTELLATION_GAP_S", "soon")
    cfg = WitnessConfig.from_env()
    assert (cfg.max_stars, cfg.alpha, cfg.min_gap) == (9, 1.0, 20.0)
    assert cfg.on is True


def test_registered_defaults_are_the_codes_defaults(monkeypatch):
    from modules import settings_registry as reg
    for key in [k for k in reg.env_keys() if k.startswith("SAGA_CONSTELLATION")]:
        monkeypatch.delenv(key, raising=False)
    cfg = WitnessConfig.from_env()
    d = lambda k: reg.BY_KEY[k].default  # noqa: E731
    assert (cfg.on, cfg.max_stars, cfg.min_gap, cfg.alpha, cfg.sleep_alpha, cfg.scale) == (
        d("SAGA_CONSTELLATION"), d("SAGA_CONSTELLATION_MAX"), d("SAGA_CONSTELLATION_GAP_S"),
        d("SAGA_CONSTELLATION_ALPHA"), d("SAGA_CONSTELLATION_SLEEP_ALPHA"), d("SAGA_CONSTELLATION_SCALE"))
    assert reg.BY_KEY["SAGA_CONSTELLATION_TINT"].default is None


def test_sky_rect_sits_to_her_upper_left_and_stays_on_screen():
    """Saga, 2026-10-14: "keep it in the upper-left area around Saga"."""
    mon = (0, 0, 1920, 1080)
    for placement in ({"x": 0.0, "y": 0.0, "size": 0.0},
                      {"x": -0.7, "y": -0.74, "rotation": 0.0, "size": 0.0},   # where she sits now
                      {"x": 0.6, "y": -0.5, "size": -30.0}):
        her_x = 1920 * (0.5 + placement["x"] / 2)
        head_top = 1080 * (0.5 - placement["y"] / 2) - 1080 * 0.26 * 10 ** (placement["size"] / 100) / 2
        x, y, w, h = sky_rect(mon, placement, 0.2)
        assert w == h == 216
        assert x + w <= her_x and y + h <= head_top + 1    # left of her middle, above her head
        assert her_x - (x + w) < w and head_top - (y + h) < 2   # and beside her, not off in a corner
    far = sky_rect(mon, {"x": 2.0, "y": 2.0, "size": 100.0}, 0.2)
    assert 0 <= far[0] <= 1920 - far[2] and 0 <= far[1] <= 1080 - far[3]
    assert sky_rect(mon, {"x": -1.0, "y": -0.74, "size": 0.0}, 0.2)[0] == 0   # at the very edge: clamped
    off = sky_rect((1920, 0, 1920, 1080), {"x": -1.0, "y": -0.65, "size": 0}, 0.2)
    assert 1920 <= off[0] <= 3840 - off[2]               # a second monitor offsets correctly


def test_sky_rect_leans_with_her_when_she_is_rotated():
    def offset(rotation):                                # her centre -> the sky's centre
        x, y, w, h = sky_rect((0, 0, 1920, 1080), {"x": 0.0, "y": 0.0, "size": 0.0, "rotation": rotation}, 0.2)
        return x + w / 2 - 960, y + h / 2 - 540

    dx0, dy0 = offset(0.0)
    assert dx0 < 0 and dy0 < 0                           # upright: up and to the left
    for rotation in (30.0, 90.0, -45.0):                 # CCW on screen, y down: the offset turns with her
        a = math.radians(rotation)
        want = (dx0 * math.cos(a) + dy0 * math.sin(a), -dx0 * math.sin(a) + dy0 * math.cos(a))
        got = offset(rotation)
        assert abs(got[0] - want[0]) <= 2 and abs(got[1] - want[1]) <= 2
    lying = offset(90.0)
    assert lying[0] < 0 and lying[1] > 0                 # on her side: past her head, toward her left


def test_on_her_tall_portrait_screen_the_sky_stays_to_her_upper_left():
    """Where she really stands (2026-10-14): a 1440x2560 portrait monitor at x=-0.7, y=-0.672.
    The 512 px sky she'd want there was pushed back on-screen, 296 px past her middle."""
    x, y, w, h = sky_rect(HER_SCREEN, HER_PLACE, 0.2)
    her_x = -1440 + 1440 * (0.5 - 0.7 / 2)
    head_top = 2560 * (0.5 + 0.672 / 2) - 2560 * 0.26 / 2
    assert x >= -1440 and y >= 0                          # on her screen
    assert x + w <= her_x and y + h <= head_top + 1       # wholly left of her middle, above her head
    assert SKY_MIN_PX <= w < 512                          # it shrank to fit instead of sliding over her…
    assert w < h and abs(h - SKY_TALLER * w) <= 2         # …and grew taller instead, up, away from her
    assert x - (-1440) <= 1                               # using all the room there is


@pytest.mark.parametrize("mon", [(0, 0, 1920, 1080), (-1440, 0, 1440, 2560), (0, 0, 2560, 1440),
                                 (261, 1440, 1920, 1080)])
def test_wherever_she_stands_the_sky_is_to_her_upper_left_or_as_small_as_it_gets(mon):
    mx, my, mw, mh = mon
    for size in (-30.0, 0.0, 40.0):
        body = 0.26 * 10 ** (size / 100)
        for i in range(-10, 11):
            for j in range(-10, 11):
                for rotation in (0.0, 30.0, -45.0, 90.0):
                    pl = {"x": i / 10, "y": j / 10, "size": size, "rotation": rotation}
                    x, y, w, h = sky_rect(mon, pl, 0.2)
                    assert mx <= x and x + w <= mx + mw and my <= y and y + h <= my + mh   # always on screen
                    if rotation:
                        continue
                    her_x = mx + mw * (0.5 + pl["x"] / 2)
                    head_top = my + mh * (0.5 - pl["y"] / 2) - mh * body / 2
                    room = min(her_x - mh * body * 0.1 - mx, head_top - my)   # left of her middle, above her head
                    if room >= SKY_MIN_PX:            # room for a sky: wholly to her upper left, as big as fits
                        assert x + w <= her_x and y + h <= head_top + 1
                        assert abs(w - min(mh * 0.2, room)) < 1
                        # held narrower than it wants: a little taller, up to what it wants and the screen's top
                        assert w <= h and abs(h - min(mh * 0.2, SKY_TALLER * w, y + h - my)) <= 2
                    else:                             # none: the smallest sky, kept on screen
                        assert w == SKY_MIN_PX and SKY_MIN_PX <= h <= SKY_TALLER * SKY_MIN_PX


def test_walking_to_the_screen_edge_shrinks_the_sky_smoothly():
    rects = [sky_rect(HER_SCREEN, {"x": -k / 200, "y": -0.672, "size": 0.0}, 0.2) for k in range(201)]
    for dim, last in ((2, SKY_MIN_PX), (3, int(SKY_TALLER * SKY_MIN_PX))):   # its width, then its height
        sides = [r[dim] for r in rects]
        assert sides[0] == 512 and sides[-1] == last
        assert all(b <= a for a, b in zip(sides, sides[1:]))     # never grows back as she nears the edge
        assert max(a - b for a, b in zip(sides, sides[1:])) <= 6  # and never jumps
    bottom = rects[0][1] + rects[0][3]
    assert all(abs(r[1] + r[3] - bottom) <= 1 for r in rects)    # always sitting on her head


# --- the renderer's IPC surface (fake window, no Qt) ---------------------------------

class _Win:
    def __init__(self, witness):
        self.instance, self.model_name = "saga", "saga"
        self.witness_enabled = witness
        self.expressions = type("E", (), {"params": {}})()
        self.cmd_q = queue.Queue()
        self.envelope = None

    def snapshot(self):
        return {"ready": True}


def test_dispatch_witness_is_fire_and_forget_and_validates_the_kind():
    w = _Win(True)
    assert dispatch(w, {"cmd": "witness", "kind": "said"}) is None   # no id -> no reply
    assert dispatch(w, {"id": 4, "cmd": "witness", "kind": "read"}) == {"id": 4, "ok": True}
    bad = dispatch(w, {"id": 5, "cmd": "witness", "kind": "his password is hunter2"})
    assert bad == {"id": 5, "ok": False}
    assert dispatch(w, {"cmd": "witness", "kind": ["said"]}) is None
    assert [w.cmd_q.get_nowait() for _ in range(2)] == [("witness", "said"), ("witness", "read")]
    assert w.cmd_q.empty()


def test_hello_advertises_whether_this_body_has_a_witness():
    assert dispatch(_Win(True), {"id": 1, "cmd": "hello"})["witness"] is True
    assert dispatch(_Win(False), {"id": 1, "cmd": "hello"})["witness"] is False
    bare = _Win(True)
    del bare.witness_enabled                              # an old window without the attribute
    assert dispatch(bare, {"id": 1, "cmd": "hello"})["witness"] is False


def test_the_witness_bumped_the_protocol_so_an_old_body_is_replaced():
    """The renderer outlives her restarts: a v2 Saga body would be adopted and never show a sky."""
    assert avatar_ipc.PROTOCOL_VERSION >= 7            # and again for the hush, her upper left, the drift, her look, her refinements
    assert dispatch(_Win(True), {"id": 1, "cmd": "hello"})["version"] == avatar_ipc.PROTOCOL_VERSION


# --- the renderer's own side: her layer, a switched-off sky, the selftest ------------------

def test_the_sky_takes_her_layer():
    assert layer_z("top") == renderer._HWND_TOPMOST
    assert layer_z("desktop") == renderer._HWND_BOTTOM   # behind his windows, exactly like her body


def test_a_switched_off_sky_rides_through_the_state_file_untouched():
    saved = {"next_seed": 4, "stars": [{"seed": 3, "kind": "read", "weight": 0.8}]}
    assert persisted_sky(None, saved) == saved            # SAGA_CONSTELLATION=off erases nothing
    assert persisted_sky(None, None) is None              # Aiva's body: no key at all
    assert persisted_sky(None, "junk") is None
    live = Constellation(WitnessConfig(min_gap=0), saved, now=0.0)
    live.witness("said", now=1.0)
    assert persisted_sky(live, saved) == live.to_dict()   # a live sky always wins
    assert [s["seed"] for s in persisted_sky(live, saved)["stars"]] == [3, 4]


def test_selftest_needs_a_painted_sky_when_the_body_has_one():
    assert selftest_verdict(True) == (True, "ok")                          # no witness, no sky asked for
    assert selftest_verdict(False)[0] is False
    assert selftest_verdict(True, expect_sky=True, sky_paints=0)[0] is False
    ok, why = selftest_verdict(True, expect_sky=True, sky_paints=5, sky_failed="ValueError('x')")
    assert ok is False and "ValueError" in why
    assert selftest_verdict(True, expect_sky=True, sky_paints=5) == (True, "ok")


def test_selftest_never_passes_beside_a_live_body(monkeypatch, tmp_path):
    """A live body owns that state file, so the selftest would render nothing — it can't say OK."""
    monkeypatch.setattr(renderer, "_INSTANCE", renderer._INSTANCE)      # main() sets the log prefix
    monkeypatch.setattr(avatar_ipc, "read_state", lambda path: {"pid": 4242, "port": 1})
    monkeypatch.setattr(avatar_ipc, "pid_alive", lambda pid: True)
    monkeypatch.setattr(renderer, "_hello_probe", lambda state: True)
    state = str(tmp_path / "saga_state.json")
    assert renderer.main(["--selftest", "--instance", "saga", "--state", state]) == 4
    assert renderer.main(["--instance", "saga", "--state", state]) == 0  # a plain launch just yields


@pytest.mark.parametrize("gap", [3.0, 20.0])
def test_the_witness_demo_lights_every_kind_and_queues_only_what_the_ipc_would(gap):
    script = witness_demo_script(gap)
    sky = Constellation(WitnessConfig(min_gap=gap))
    speech = []
    for t, _label, (cmd, arg) in sorted(script, key=lambda s: s[0]):
        if cmd == "witness":
            assert sky.witness(arg, now=t) is not None   # every kind it sends is one the sky accepts
        elif cmd == "layer":
            assert arg in ("top", "desktop")
        elif cmd == "speech":
            speech.append(arg)                           # what the IPC's `speech` hands the envelope
        else:
            assert cmd in ("sleeping", "move")
    assert [s["kind"] for s in sky.stars] == list(constellation.KINDS)   # spaced past the gap
    assert sky.stars[0]["weight"] > constellation.KINDS["said"]           # and the burst only brightened
    assert speech == ["start", "stop"]                                     # the hush, and the ease back


def test_the_witness_demo_never_writes_into_her_real_sky(monkeypatch, tmp_path):
    monkeypatch.setattr(renderer, "_INSTANCE", renderer._INSTANCE)      # main() sets the log prefix
    seen = []
    monkeypatch.setattr(avatar_ipc, "read_state", lambda path: seen.append(path))
    missing = str(tmp_path / "no.model3.json")                         # stops main() before Qt
    assert renderer.main(["--demo-witness", "--instance", "saga", "--model", missing]) == 3
    assert seen[-1] != avatar_ipc.state_path("saga")
    assert os.path.dirname(seen[-1]) == tempfile.gettempdir()
    mine = str(tmp_path / "demo_state.json")
    assert renderer.main(["--demo-witness", "--instance", "saga", "--state", mine, "--model", missing]) == 3
    assert seen[-1] == mine                                            # an explicit --state is honoured


# --- the sister-side client ----------------------------------------------------------

class _Sink:
    def __init__(self):
        self.sent = []

    def write(self, data):
        self.sent.append(json.loads(data))


def test_send_witness_only_flows_to_a_renderer_that_has_one():
    c = AvatarClient(instance="saga", model="assets/avatar/ariu/saga.model3.json")
    c.send_witness("said")                                # not connected: silent no-op
    c._writer = _Sink()
    c.send_witness("said")                                # connected but hello said no witness
    assert c._writer.sent == []
    c.witness = True
    c.send_witness("heard")
    assert c._writer.sent == [{"cmd": "witness", "kind": "heard"}]


def test_send_witness_swallows_a_dead_socket():
    class Dead:
        def write(self, _):
            raise OSError("gone")
    c = AvatarClient(instance="saga")
    c._writer, c.witness = Dead(), True
    c.send_witness("said")                                # must not raise


def test_connect_learns_the_witness_flag_from_hello():
    async def serve(witness):
        async def handle(reader, writer):
            while line := await reader.readline():
                msg = json.loads(line)
                if msg.get("cmd") == "hello":
                    writer.write(avatar_ipc.encode({"id": msg["id"], "ok": True, "model": "saga",
                                                    "expressions": [], "witness": witness}))
                    await writer.drain()
            writer.close()
        return await asyncio.start_server(handle, "127.0.0.1", 0)

    async def run(witness):
        srv = await serve(witness)
        c = AvatarClient(instance="saga", model="assets/avatar/ariu/saga.model3.json")
        try:
            ok = await c._connect({"port": srv.sockets[0].getsockname()[1]})
            return ok, c.witness
        finally:
            await c.close()
            srv.close()

    assert asyncio.run(run(True)) == (True, True)
    assert asyncio.run(run(False)) == (True, False)
    assert constellation.ENABLED_INSTANCES == {"saga"}
