"""Speaker-recognition validation spike (throwaway dev tool — safe to delete).

Confirms, on THIS machine with real voices, the three things the plan gates enabling on:
  1) same-speaker cosine sits well ABOVE cross-speaker cosine, with a visible gap
     (this sets AIVA_SPEAKER_T_HIGH / T_LOW / MARGIN),
  2) the CAM++ ONNX + kaldi-native-fbank path actually produces embeddings,
  3) per-utterance embedding latency is small (tens of ms).

It reuses the REAL embedder (SpeakerRecognizer._embed_onnx), so whatever passes here is what
runs live.

Setup:
  1. pip is already done (kaldi-native-fbank, onnxruntime present).
  2. Download a CAM++ speaker ONNX, e.g. from https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx
     (the .onnx file), save it somewhere, and point to it:
         set AIVA_SPEAKER_MODEL=C:\path\to\campplus.onnx
  3. Run (from the aiva/ dir so `modules` imports):
         cd aiva && ..\.venv\Scripts\python ..\tests\spike_speaker_id.py --speakers 2 --clips 3

You'll be prompted to record a few clips per person. Then it prints the cosine gap + suggested
thresholds + latency. If a model input/shape errors out, that tells us to adjust `_embed_onnx`
for that specific export.
"""
import argparse
import os
import sqlite3
import sys
import time
from itertools import combinations

# make `modules` importable when run from the repo root or aiva/
_AIVA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aiva")
if _AIVA not in sys.path:
    sys.path.insert(0, _AIVA)

from modules.speakers import SpeakerConfig, SpeakerMemory, SpeakerRecognizer, _cosine  # noqa: E402


def record_clip(seconds):
    """Record `seconds` of mic audio, return 16 kHz mono int16 PCM bytes (mirrors WakeWordListener)."""
    import numpy as np
    import pyaudio
    from math import gcd
    from scipy.signal import resample_poly

    pa = pyaudio.PyAudio()
    info = pa.get_default_input_device_info()
    native = int(info["defaultSampleRate"])
    frames = int(native * 0.08)
    g = gcd(16000, native); up, down = 16000 // g, native // g
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=native, input=True,
                     frames_per_buffer=frames)
    out = bytearray()
    n = int(seconds * native)
    got = 0
    try:
        while got < n:
            chunk = stream.read(frames, exception_on_overflow=False)
            got += frames
            audio = np.frombuffer(chunk, dtype=np.int16)
            if native != 16000:
                audio = resample_poly(audio.astype(np.float32), up, down)
                audio = np.clip(audio, -32768, 32767).astype(np.int16)
            out += audio.tobytes()
    finally:
        stream.stop_stream(); stream.close(); pa.terminate()
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, default=2)
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=4.0)
    args = ap.parse_args()

    model = os.getenv("AIVA_SPEAKER_MODEL")
    if not model or not os.path.isfile(model):
        print(f"Set AIVA_SPEAKER_MODEL to a CAM++ .onnx file (got: {model!r}). See the header.")
        return 1

    rec = SpeakerRecognizer(SpeakerMemory(sqlite3.connect(":memory:")),
                            SpeakerConfig(), model_path=model, enabled=True, log=print)

    embeds = {}   # speaker_label -> [vec, ...]
    latencies = []
    for s in range(args.speakers):
        label = input(f"\nSpeaker {s + 1} name (for labels): ").strip() or f"spk{s + 1}"
        embeds[label] = []
        for c in range(args.clips):
            input(f"  [{label}] clip {c + 1}/{args.clips} — press Enter, then say a full sentence...")
            pcm = record_clip(args.seconds)
            t0 = time.perf_counter()
            vec = rec._embed_onnx(pcm)
            latencies.append((time.perf_counter() - t0) * 1000)
            if not vec:
                print("    !! embedding failed — check the model export / _embed_onnx input shape")
                return 1
            embeds[label].append(vec)
            print(f"    ok ({len(vec)}-d, {latencies[-1]:.0f} ms)")

    # same-speaker vs cross-speaker cosine
    same, cross = [], []
    for label, vs in embeds.items():
        for a, b in combinations(vs, 2):
            same.append(_cosine(a, b))
    labels = list(embeds)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            for a in embeds[labels[i]]:
                for b in embeds[labels[j]]:
                    cross.append(_cosine(a, b))

    def stats(xs):
        return (min(xs), sum(xs) / len(xs), max(xs)) if xs else (0, 0, 0)

    smin, savg, smax = stats(same)
    cmin, cavg, cmax = stats(cross)
    print("\n================ RESULTS ================")
    print(f"embedding latency: avg {sum(latencies)/len(latencies):.0f} ms, max {max(latencies):.0f} ms")
    print(f"same-speaker  cosine: min {smin:.3f}  avg {savg:.3f}  max {smax:.3f}  (n={len(same)})")
    print(f"cross-speaker cosine: min {cmin:.3f}  avg {cavg:.3f}  max {cmax:.3f}  (n={len(cross)})")
    gap = smin - cmax
    print(f"GAP (min-same - max-cross): {gap:+.3f}  {'GOOD — clean separation' if gap > 0.1 else 'TIGHT — thresholds will be finicky' if gap > 0 else 'BAD — voices overlap, model/clips need work'}")
    if same and cross:
        t_high = round((smin + cmax) / 2 + 0.02, 2) if gap > 0 else round(savg - 0.05, 2)
        t_low = round(cmax + 0.03, 2)
        print(f"\nSuggested .env (tune to taste):")
        print(f"  AIVA_SPEAKER_T_HIGH={max(t_high, t_low + 0.05)}")
        print(f"  AIVA_SPEAKER_T_LOW={t_low}")
        print(f"  AIVA_SPEAKER_MARGIN=0.06")
    return 0


if __name__ == "__main__":
    sys.exit(main())
