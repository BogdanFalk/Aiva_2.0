"""Vocal-affect tests: feature extraction on synthetic tones, the pure affect mapping, and the
VocalAffect baseline flow. Deterministic (sine tones), no model."""
import numpy as np
import pytest

from modules.paralinguistics import (SR, AffectConfig, VocalAffect, affect_from,
                                     extract_features)

CFG = AffectConfig()


def sine(freq=150.0, secs=1.5, amp=0.2):
    t = np.arange(int(secs * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16).tobytes()


# -- feature extraction -----------------------------------------------------------

def test_extract_features_reads_pitch_and_energy():
    f = extract_features(sine(150, 1.5, 0.2))
    assert f is not None
    assert 120 < f["f0_mean"] < 180          # ~150 Hz tone
    assert 0.10 < f["rms"] < 0.20            # 0.2 amplitude sine -> ~0.14 rms
    assert f["voiced_ratio"] > 0.8


def test_extract_features_too_short_or_silent():
    assert extract_features(sine(150, 0.1, 0.2)) is None          # too short
    assert extract_features(np.zeros(SR, dtype=np.int16).tobytes()) is None  # silence -> unvoiced


# -- pure affect mapping ----------------------------------------------------------

def test_louder_than_baseline_is_higher_arousal():
    base = {"rms": 0.10, "f0_mean": 150, "f0_std": 10, "voiced_ratio": 0.8}
    loud = {"rms": 0.22, "f0_mean": 180, "f0_std": 20, "voiced_ratio": 0.9}
    quiet = {"rms": 0.05, "f0_mean": 130, "f0_std": 5, "voiced_ratio": 0.7}
    assert affect_from(loud, base, CFG)["arousal"] > 0.1
    assert affect_from(quiet, base, CFG)["arousal"] < 0


def test_salience_grows_with_shift():
    base = {"rms": 0.10, "f0_mean": 150, "f0_std": 10, "voiced_ratio": 0.8}
    calm = affect_from(dict(base), base, CFG)
    big = affect_from({"rms": 0.30, "f0_mean": 210, "f0_std": 25, "voiced_ratio": 0.95}, base, CFG)
    assert big["salience"] > calm["salience"]


def test_labels():
    base = {"rms": 0.10, "f0_mean": 150, "f0_std": 10, "voiced_ratio": 0.8}
    assert affect_from({"rms": 0.30, "f0_mean": 210, "f0_std": 30, "voiced_ratio": 0.95},
                       base, CFG)["label"] in ("animated", "tense")
    assert affect_from({"rms": 0.04, "f0_mean": 120, "f0_std": 3, "voiced_ratio": 0.6},
                       base, CFG)["label"] in ("flat", "subdued")


# -- VocalAffect baseline flow ----------------------------------------------------

def test_first_turn_sets_baseline_then_reports():
    va = VocalAffect(CFG)
    assert va.analyze(sine(150, 1.5, 0.10)) is None      # first voiced turn = baseline, no report
    aff = va.analyze(sine(200, 1.5, 0.30))               # louder + higher pitch
    assert aff is not None and aff["arousal"] > 0.1
    assert aff["label"] in ("animated", "tense")


def test_disabled_returns_none():
    va = VocalAffect(CFG, enabled=False)
    assert va.analyze(sine()) is None
