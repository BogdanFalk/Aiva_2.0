"""Vocal affect (paralinguistics) — how the user SOUNDS, not who they are or what they said.

Cheap DSP on the same per-turn 16 kHz PCM the speaker tap already captures: loudness (RMS),
pitch (f0), pitch variability, and how voiced the turn is → a COARSE arousal/valence estimate
RELATIVE to a running baseline ("more wound-up / flatter than usual"). It emits onto her
nervous-system bus so her mood and tone quietly mirror yours — she gets a little gentler when
you sound tired, a little livelier when you're animated. No model, no new deps (numpy only).

Deliberately coarse: it does AROUSAL (energy) well and valence weakly; nuanced emotion classes
and real laughter detection are the model-based upgrade (emotion2vec / an AudioSet laughter
tagger) that would slot in here as a second analyzer feeding the same bus.

Runs in a worker thread (numpy DSP) and returns a plain dict; main.py does the bus emit.
"""

import math
import os
from dataclasses import dataclass

SR = 16000


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def extract_features(pcm, sr=SR):
    """16 kHz mono int16 PCM -> {rms, f0_mean, f0_std, voiced_ratio} over voiced frames, or None
    if there isn't enough voiced audio. Pure numpy; safe to call in a thread."""
    import numpy as np
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if x.size < int(sr * 0.3):
        return None
    win, hop = int(sr * 0.025), int(sr * 0.010)     # 25 ms / 10 ms frames
    frames = [x[i:i + win] for i in range(0, len(x) - win, hop)]
    if not frames:
        return None
    rms = np.array([float(np.sqrt(np.mean(f * f))) for f in frames])
    gate = max(0.015, float(np.median(rms)) * 0.5)  # voiced = above the noise floor
    voiced_idx = np.where(rms > gate)[0]
    voiced_ratio = len(voiced_idx) / len(frames)
    if len(voiced_idx) < 5:
        return None
    lo, hi = int(sr / 350), int(sr / 80)            # f0 search 80–350 Hz
    f0s = []
    for i in voiced_idx:
        f = frames[i] - frames[i].mean()
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        if seg.size == 0:
            continue
        lag = lo + int(np.argmax(seg))
        if ac[lag] / ac[0] > 0.3:                    # peak strong enough to be pitch
            f0s.append(sr / lag)
    f0s = np.array(f0s)
    return {
        "rms": float(rms[voiced_idx].mean()),
        "f0_mean": float(f0s.mean()) if f0s.size else 0.0,
        "f0_std": float(f0s.std()) if f0s.size > 1 else 0.0,
        "voiced_ratio": float(voiced_ratio),
    }


@dataclass
class AffectConfig:
    ema: float = 0.15            # how fast the baseline adapts to you/the room
    gain: float = 0.6            # scale of the arousal hint
    salience_base: float = 0.2   # a plain turn still nudges a little
    salience_max: float = 0.7
    notable: float = 0.12        # |arousal| above this is a real shift worth more salience

    @classmethod
    def from_env(cls):
        g = os.getenv
        return cls(
            ema=float(g("AIVA_AFFECT_EMA", "0.15")),
            gain=float(g("AIVA_AFFECT_GAIN", "0.6")),
            salience_base=float(g("AIVA_AFFECT_SALIENCE_BASE", "0.2")),
            salience_max=float(g("AIVA_AFFECT_SALIENCE_MAX", "0.7")),
            notable=float(g("AIVA_AFFECT_NOTABLE", "0.12")),
        )


class VocalAffect:
    """Holds a running baseline of your normal voice and reports each turn's affect relative to
    it. `analyze` returns None until it has a baseline (first voiced turn) and for turns too
    short to read."""

    def __init__(self, cfg: AffectConfig = None, enabled=True, log=print):
        self.cfg = cfg or AffectConfig.from_env()
        self.enabled = enabled
        self.log = log
        self.base = None

    def analyze(self, pcm):
        if not self.enabled:
            return None
        f = extract_features(pcm)
        if not f:
            return None
        if self.base is None:
            self.base = dict(f)
            return None                              # first turn just sets the baseline
        aff = affect_from(f, self.base, self.cfg)
        # adapt the baseline toward this turn (EMA), so "normal" tracks you over time
        for k in self.base:
            self.base[k] = (1 - self.cfg.ema) * self.base[k] + self.cfg.ema * f[k]
        return aff


def affect_from(f, base, cfg):
    """Pure mapping: current features vs baseline -> {arousal, valence, salience, label, features}.
    Arousal (energy) is the strong signal; valence is a weak lean from pitch variability."""
    d_rms = f["rms"] / max(base["rms"], 1e-4) - 1.0
    d_f0 = (f["f0_mean"] / base["f0_mean"] - 1.0) if (f["f0_mean"] and base["f0_mean"]) else 0.0
    d_var = (f["f0_std"] / base["f0_std"] - 1.0) if (f["f0_std"] and base["f0_std"]) else 0.0
    arousal = _clamp(cfg.gain * (0.5 * d_rms + 0.3 * d_f0 + 0.2 * d_var), -0.6, 0.6)
    valence = _clamp(0.15 * d_var, -0.25, 0.25)      # melodic/varied ~ engaged; flat ~ down. weak.
    mag = abs(arousal)
    salience = (_clamp(cfg.salience_base + mag, 0.0, cfg.salience_max)
                if mag >= cfg.notable else cfg.salience_base * 0.4)
    return {"arousal": round(arousal, 3), "valence": round(valence, 3),
            "salience": round(salience, 3), "label": _label(arousal, valence), "features": f}


def _label(arousal, valence):
    if arousal >= 0.15:
        return "animated" if valence >= 0 else "tense"
    if arousal <= -0.15:
        return "flat" if valence <= 0 else "subdued"
    return "steady"
