"""Local wake-word detection (openWakeWord) on its own mic stream — one listener, N models.

Runs in a thread; never sends audio anywhere. Fires `on_wake(score, preroll, name)` when a model
scores above threshold for `patience` CONSECUTIVE 80 ms frames — `preroll` is the last
`preroll_secs` of 16 kHz PCM (the wake phrase itself) so STT can verify what was actually said,
and `name` is which model fired (e.g. "hey_aiva" / "hey_saga") so the house knows who was called.

A real spoken "hey Aiva" stays in the model's sliding window for several frames, so it keeps
scoring high; false triggers from game/TV audio are usually a single spiky frame — requiring
consecutive frames filters them out where a higher threshold can't (false spikes overlap the score
range of genuine quiet wakes). With two models the raw false-positive rate roughly doubles; the
two-stage transcript verification downstream absorbs that.
"""

import os
import threading
import time
from collections import deque


def model_name(path_or_name: str) -> str:
    """openWakeWord keys its scores by the model's basename without extension."""
    base = os.path.basename(path_or_name)
    return base.rsplit(".", 1)[0] if "." in base else base


def wake_slugs(models_by_slug) -> dict:
    """{model name: sister slug} for `{model path or name: slug}`. The listener reports which model
    fired ("hey_saga"); the floor routes by slug ("saga"). A name missing here (the hey_jarvis
    fallback) maps to no sister, so `.get(name)` gives None."""
    return {model_name(m): slug for m, slug in models_by_slug.items()}


class WakeWordListener:
    def __init__(self, models, on_wake, device_index=None, threshold=0.5, patience=2,
                 preroll_secs=2.5, log=print):
        self._models = [models] if isinstance(models, str) else [m for m in models if m]
        self._on_wake = on_wake
        self._device_index = device_index
        self._threshold = threshold
        self._patience = max(1, patience)
        self._preroll_bytes = int(preroll_secs * 16000) * 2
        self._log = log
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def models(self):
        return list(self._models)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        from math import gcd

        import numpy as np
        import pyaudio
        from openwakeword.model import Model
        from scipy.signal import resample_poly

        oww = Model(wakeword_models=list(self._models), inference_framework="onnx")
        pa = pyaudio.PyAudio()
        # Capture at the device's NATIVE rate and resample to the 16 kHz the model expects —
        # asking Windows for 16 kHz directly yields silence (MME) or corrupted repeating
        # buffers (DirectSound) on this machine.
        if self._device_index is None:
            info = pa.get_default_input_device_info()
        else:
            info = pa.get_device_info_by_index(self._device_index)
        native = int(info["defaultSampleRate"])
        frames = int(native * 0.08)  # 80 ms, the model's chunk size
        g = gcd(16000, native)
        up, down = 16000 // g, native // g

        stream = pa.open(
            rate=native, channels=1, format=pyaudio.paInt16, input=True,
            frames_per_buffer=frames, input_device_index=self._device_index,
        )
        names = ", ".join(model_name(m) for m in self._models)
        print(f"Wake word armed ({names}) on '{info['name']}' @ {native} Hz "
              f"(threshold={self._threshold}, patience={self._patience} frames)")
        try:
            consecutive = 0
            peak = 0.0
            peak_name = None
            ring = deque()      # recent 16 kHz audio, newest last
            ring_bytes = 0
            while not self._stop.is_set():
                chunk = stream.read(frames, exception_on_overflow=False)
                audio = np.frombuffer(chunk, dtype=np.int16)
                if native != 16000:
                    audio = resample_poly(audio.astype(np.float32), up, down)
                    audio = np.clip(audio, -32768, 32767).astype(np.int16)
                pcm = audio.tobytes()
                ring.append(pcm)
                ring_bytes += len(pcm)
                while ring_bytes > self._preroll_bytes and len(ring) > 1:
                    ring_bytes -= len(ring.popleft())
                scores = oww.predict(audio)
                if scores:
                    name, score = max(scores.items(), key=lambda kv: kv[1])
                else:
                    name, score = None, 0.0
                if score >= self._threshold:
                    consecutive += 1
                    if score >= peak:
                        peak, peak_name = score, name
                    if consecutive >= self._patience:
                        oww.reset()
                        self._log(f"wake word detected ({peak_name}, score={peak:.2f}, "
                                  f"threshold={self._threshold}, frames={consecutive})")
                        self._on_wake(peak, b"".join(ring), peak_name)
                        ring.clear()
                        ring_bytes = 0
                        consecutive = 0
                        peak = 0.0
                        peak_name = None
                        time.sleep(2)  # debounce
                else:
                    if consecutive:
                        # would have woken before the patience filter — keep a trace so the
                        # threshold/patience knobs can be tuned later
                        self._log(f"wake spike ignored ({peak_name}, score={peak:.2f}, "
                                  f"only {consecutive}/{self._patience} frames)")
                    consecutive = 0
                    peak = 0.0
                    peak_name = None
        finally:
            stream.close()
            pa.terminate()
