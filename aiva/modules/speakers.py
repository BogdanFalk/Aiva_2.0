"""Open-set speaker recognition — "who am I talking to?".

She embeds each user turn's audio into a voiceprint, matches it against a growing table of
enrolled voices by cosine similarity, and:
  - greets known people naturally,
  - when a voice is clearly UNKNOWN, stashes it and nudges her to ask "who are you?" then
    enrolls them on the spot via the enroll_speaker tool (open-ended — learned on the fly),
  - tells the LLM who's speaking through an on-change system note + a persona roster.

Design mirrors the other memory subsystems: a `memory.conn`-backed store (event-loop-thread
only), the same BLOB + cosine pattern as task_memory/inner, and a pure decision function that
unit-tests with plain tuples. The heavy audio deps (onnxruntime, kaldi-native-fbank, numpy) are
lazy-imported inside the embedder and the whole thing degrades to a no-op when disabled or the
model is missing — so this module and its tests stay dependency-light.

v1 is ONE speaker per turn. Multiple simultaneous speakers (Deepgram diarize + per-segment
embedding) is a later phase.

Audit:  cd aiva && ..\\.venv\\Scripts\\python -m modules.speakers --show
"""

import asyncio
import json
import math
import os
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime

MODEL_ID_DEFAULT = "campplus-192"

UNKNOWN_NOTE = ("[Unknown voice — you haven't met this person. Warmly ask who they are in one "
                "short sentence, then call enroll_speaker with the name they give.]")


def _now():
    return datetime.now().isoformat()


def _secs_between(then_iso, now_iso):
    try:
        then = datetime.fromisoformat(then_iso)
        now = datetime.fromisoformat(now_iso) if isinstance(now_iso, str) else now_iso
        return max(0.0, (now - then).total_seconds())
    except (ValueError, TypeError):
        return 1e9


# --- tiny vector helpers (stdlib-only; same shape as task_memory/inner) --------------

def _pack_vec(vec):
    return struct.pack(f"<{len(vec)}f", *vec) if vec else None


def _unpack_vec(blob):
    if not blob:
        return None
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _normalize(vec):
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _mean_normalized(centroid, count, vec):
    """Running mean of L2-normalized embeddings, re-normalized."""
    merged = [(c * count + v) / (count + 1) for c, v in zip(centroid, vec)]
    return _normalize(merged)


# --- config + pure decision ----------------------------------------------------------

@dataclass
class SpeakerConfig:
    t_high: float = 0.62        # confident match
    t_low: float = 0.45         # below this, top-1 is clearly not a known voice
    margin: float = 0.06        # top1 - top2 needed to pick between two KNOWN people
    sticky_keep: float = 0.50   # sticky speaker still plausible if scored >= this
    sticky_secs: float = 45.0   # ...and heard within this many seconds
    min_secs: float = 1.5       # skip identification on utterances shorter than this
    max_secs: float = 12.0      # cap buffered audio fed to the model
    embed_timeout_ms: int = 800
    primary_name: str = "Frank"

    @classmethod
    def from_env(cls):
        g = os.getenv
        return cls(
            t_high=float(g("AIVA_SPEAKER_T_HIGH", "0.62")),
            t_low=float(g("AIVA_SPEAKER_T_LOW", "0.45")),
            margin=float(g("AIVA_SPEAKER_MARGIN", "0.06")),
            sticky_keep=float(g("AIVA_SPEAKER_STICKY_KEEP", "0.50")),
            sticky_secs=float(g("AIVA_SPEAKER_STICKY_SECS", "45")),
            min_secs=float(g("AIVA_SPEAKER_MIN_SECS", "1.5")),
            max_secs=float(g("AIVA_SPEAKER_MAX_SECS", "12")),
            embed_timeout_ms=int(g("AIVA_SPEAKER_EMBED_TIMEOUT_MS", "800")),
            primary_name=g("AIVA_SPEAKER_PRIMARY_NAME", "Frank"),
        )


@dataclass
class Decision:
    kind: str                    # "known" | "unknown" | "hold"
    name: str | None = None
    speaker_id: int | None = None
    score: float = 0.0


def _sticky_plausible(sticky, scored, cfg, now):
    """The sticky speaker is still a believable match this turn."""
    if not sticky:
        return False
    sid, _sname, last_ts = sticky
    if _secs_between(last_ts, now) > cfg.sticky_secs:
        return False
    for rid, _n, score in scored:
        if rid == sid:
            return score >= cfg.sticky_keep
    return False


def decide(scored, sticky, cfg, now=None):
    """Pure open-set classifier. `scored` = [(id, name, score), ...] DESC (may be empty);
    `sticky` = (id, name, last_ts) | None. Returns a Decision:
      - known: a confident, unambiguous match (attach identity, improve centroid)
      - hold:  keep the sticky speaker (same person continuing) — no note, no re-ask
      - unknown: clearly not a known voice (or nobody enrolled) — trigger the ask
    """
    now = now or _now()
    if not scored:
        return Decision("unknown")
    tid, tname, tscore = scored[0]
    top2 = scored[1][2] if len(scored) > 1 else -1.0
    margin_ok = len(scored) == 1 or (tscore - top2) >= cfg.margin

    if tscore >= cfg.t_high and margin_ok:
        return Decision("known", name=tname, speaker_id=tid, score=tscore)

    # not a confident match: is the same person plausibly still talking?
    if _sticky_plausible(sticky, scored, cfg, now):
        sid, sname, _ = sticky
        return Decision("hold", name=sname, speaker_id=sid, score=tscore)

    if tscore < cfg.t_low:
        return Decision("unknown", score=tscore)
    # between t_low and t_high with no sticky, or a within-margin tie between knowns:
    # not confident enough to name → treat as unknown (the caller debounces the ask)
    return Decision("unknown", score=tscore)


# --- storage -------------------------------------------------------------------------

class SpeakerMemory:
    """The enrolled-voices table on the shared connection (loop-thread only)."""

    def __init__(self, conn: sqlite3.Connection, model_id=MODEL_ID_DEFAULT):
        self.conn = conn
        self.model_id = model_id
        self._create_tables()

    def _create_tables(self):
        c = self.conn
        c.execute("""CREATE TABLE IF NOT EXISTS speakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, dim INTEGER NOT NULL,
            centroid BLOB NOT NULL, sample_count INTEGER NOT NULL DEFAULT 1,
            created_ts TEXT NOT NULL, last_heard_ts TEXT NOT NULL, model_id TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS speaker_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT, speaker_id INTEGER NOT NULL, embedding BLOB NOT NULL,
            ts TEXT NOT NULL, source TEXT)""")
        c.commit()

    def enroll(self, name, vec, *, source="enroll"):
        """Insert a new speaker, or (if the name is taken) fold the sample into that speaker.
        Returns the speaker id."""
        vec = _normalize(vec)
        row = self.conn.execute("SELECT id FROM speakers WHERE name = ?", (name,)).fetchone()
        if row:
            self.add_sample_and_recentroid(row[0], vec, source=source)
            return row[0]
        now = _now()
        cur = self.conn.execute(
            "INSERT INTO speakers (name, dim, centroid, sample_count, created_ts, last_heard_ts, "
            "model_id) VALUES (?,?,?,?,?,?,?)",
            (name, len(vec), _pack_vec(vec), 1, now, now, self.model_id))
        sid = cur.lastrowid
        self.conn.execute("INSERT INTO speaker_samples (speaker_id, embedding, ts, source) "
                          "VALUES (?,?,?,?)", (sid, _pack_vec(vec), now, source))
        self.conn.commit()
        return sid

    def identify(self, vec):
        """[(id, name, score), ...] DESC over speakers matching this model/dim."""
        rows = self.conn.execute(
            "SELECT id, name, dim, centroid, model_id FROM speakers").fetchall()
        out = []
        for rid, name, dim, blob, model_id in rows:
            if dim != len(vec) or model_id != self.model_id:
                continue
            out.append((rid, name, _cosine(vec, _unpack_vec(blob))))
        out.sort(key=lambda r: r[2], reverse=True)
        return out

    def add_sample_and_recentroid(self, sid, vec, *, source="turn", cap=50):
        vec = _normalize(vec)
        row = self.conn.execute(
            "SELECT centroid, sample_count FROM speakers WHERE id = ?", (sid,)).fetchone()
        if not row:
            return
        centroid, count = _unpack_vec(row[0]), row[1]
        new_centroid = _mean_normalized(centroid, count, vec)
        now = _now()
        self.conn.execute(
            "UPDATE speakers SET centroid = ?, sample_count = sample_count + 1, last_heard_ts = ? "
            "WHERE id = ?", (_pack_vec(new_centroid), now, sid))
        # keep a bounded number of raw samples for future re-centroiding
        self.conn.execute("INSERT INTO speaker_samples (speaker_id, embedding, ts, source) "
                          "VALUES (?,?,?,?)", (sid, _pack_vec(vec), now, source))
        extra = self.conn.execute(
            "SELECT id FROM speaker_samples WHERE speaker_id = ? ORDER BY id DESC LIMIT -1 OFFSET ?",
            (sid, cap)).fetchall()
        for (xid,) in extra:
            self.conn.execute("DELETE FROM speaker_samples WHERE id = ?", (xid,))
        self.conn.commit()

    def rename(self, old, new):
        cur = self.conn.execute("UPDATE speakers SET name = ? WHERE name = ?", (new, old))
        self.conn.commit()
        return cur.rowcount > 0

    def forget(self, name):
        row = self.conn.execute("SELECT id FROM speakers WHERE name = ?", (name,)).fetchone()
        if not row:
            return False
        self.conn.execute("DELETE FROM speaker_samples WHERE speaker_id = ?", (row[0],))
        self.conn.execute("DELETE FROM speakers WHERE id = ?", (row[0],))
        self.conn.commit()
        return True

    def touch(self, sid):
        self.conn.execute("UPDATE speakers SET last_heard_ts = ? WHERE id = ?", (_now(), sid))
        self.conn.commit()

    def names(self):
        return [r[0] for r in self.conn.execute(
            "SELECT name FROM speakers ORDER BY last_heard_ts DESC").fetchall()]

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]


# --- orchestrator --------------------------------------------------------------------

class SpeakerRecognizer:
    """Ties audio → embedding → identify → decide → identity note / enroll flow. `embed` is an
    async callable(pcm_bytes) -> vec | None; when None, the real onnxruntime + kaldi-native-fbank
    path is used (lazy-loaded). Everything no-ops when `enabled` is False."""

    def __init__(self, mem: SpeakerMemory, cfg: SpeakerConfig, *, embed=None, model_path=None,
                 enabled=False, log=print, now=None):
        self.mem = mem
        self.cfg = cfg
        self._embed = embed
        self.model_path = model_path
        self.enabled = enabled
        self.log = log
        self._now = now or _now
        self.current = None            # name of the current speaker (or None)
        self._current_id = None
        self._sticky = None            # (id, name, last_ts)
        self.pending = None            # {"vec", "ts"} unknown voice awaiting a name
        self._last_note_name = None    # for on-change note suppression
        self._refresh_prompt = lambda: None
        self._session = None           # lazy ort.InferenceSession
        self._fbank = None             # lazy kaldi-native-fbank

    def attach(self, *, refresh_prompt=None):
        if refresh_prompt:
            self._refresh_prompt = refresh_prompt

    def _sticky_tuple(self):
        return self._sticky

    # -- the per-turn entry point ----------------------------------------------------

    async def on_turn(self, vec):
        """Given the utterance's voiceprint `vec` (already embedded on a worker thread; None to
        skip), update identity and return a system note to inject, or None. Runs on the loop
        thread (does the sqlite work)."""
        if not self.enabled or not vec:
            return None
        now = self._now()

        # first voice ever → silently seed the primary user (he boots her); no interrogation
        if self.mem.count() == 0:
            sid = self.mem.enroll(self.cfg.primary_name, vec, source="primary-seed")
            self._set_current(sid, self.cfg.primary_name, now)
            self._last_note_name = self.cfg.primary_name   # already "current" — don't announce next turn
            self.log(f"speaker: seeded primary '{self.cfg.primary_name}'")
            return None

        scored = self.mem.identify(vec)
        d = decide(scored, self._sticky_tuple(), self.cfg, now=now)

        if d.kind == "known":
            self._set_current(d.speaker_id, d.name, now)
            self.mem.add_sample_and_recentroid(d.speaker_id, vec)   # improve over time
            self.pending = None
            return self._note_on_change(d.name)

        if d.kind == "hold":
            self._set_current(d.speaker_id, d.name, now)            # keep sticky, refresh ts
            return None                                             # no note, no re-ask

        # unknown
        if self.pending and _cosine(vec, self.pending["vec"]) >= self.cfg.t_high:
            # same unknown person still talking (e.g. giving their name) — fold in a 2nd sample,
            # don't re-ask
            self.pending = {"vec": _mean_normalized(self.pending["vec"], 1, vec), "ts": now}
            return None
        # a fresh unknown voice → stash and ask once
        self.pending = {"vec": _normalize(vec), "ts": now}
        self._set_current(None, None, now)
        return UNKNOWN_NOTE

    def _set_current(self, sid, name, now):
        self._current_id = sid
        self.current = name
        if sid is not None:
            self._sticky = (sid, name, now)

    def _note_on_change(self, name):
        if name == self._last_note_name:
            return None                    # unchanged speaker → no note (near-zero token cost)
        self._last_note_name = name
        return f"[Speaking now: {name}.]"

    # -- enroll / veto flow ----------------------------------------------------------

    def enroll_current(self, name):
        """Commit the pending unknown voice under `name` (called by the enroll_speaker tool)."""
        name = (name or "").strip()
        if not name:
            return {"success": False, "error": "enroll under what name?"}
        if not self.pending:
            return {"success": False, "error": "no new voice to enroll right now"}
        sid = self.mem.enroll(name, self.pending["vec"], source="enroll")
        self._set_current(sid, name, self._now())
        self._last_note_name = name
        self.pending = None
        try:
            self._refresh_prompt()
        except Exception:
            pass
        return {"success": True, "name": name}

    def rename_current(self, new_name):
        new_name = (new_name or "").strip()
        if not new_name or self.current is None:
            return {"success": False, "error": "no current speaker to rename"}
        ok = self.mem.rename(self.current, new_name)
        if ok:
            self.current = new_name
            self._last_note_name = new_name
            if self._sticky:
                self._sticky = (self._sticky[0], new_name, self._sticky[2])
            try:
                self._refresh_prompt()
            except Exception:
                pass
        return {"success": ok, "name": new_name if ok else None}

    def forget(self, name):
        ok = self.mem.forget((name or "").strip())
        if ok and name == self.current:
            self._set_current(None, None, self._now())
            self._last_note_name = None
        try:
            self._refresh_prompt()
        except Exception:
            pass
        return {"success": ok}

    # -- persona ---------------------------------------------------------------------

    def roster(self):
        """Data for the WHO YOU'RE TALKING TO persona block: current speaker + known names."""
        return {"current": self.current, "known": self.mem.names()}

    # -- the real embedder (lazy; heavy deps imported here only) ----------------------

    async def embed_pcm(self, pcm: bytes):
        """16 kHz mono int16 PCM -> voiceprint vec, or None. Uses the injected `embed` if given
        (tests), else onnxruntime + kaldi-native-fbank (lazy). Never raises."""
        if not self.enabled or not pcm:
            return None
        if self._embed is not None:
            try:
                return await self._embed(pcm)
            except Exception as e:
                self.log(f"speaker: injected embed failed ({e!r})")
                return None
        try:
            return await asyncio.to_thread(self._embed_onnx, pcm)   # onnx is CPU-bound: off-loop
        except Exception as e:
            self.log(f"speaker: embed failed ({e!r})")
            return None

    def warm(self):
        """Load the ONNX session up front (the one-time ~cold cost) so the first real turn is fast.
        Meant to run in a worker thread at boot; swallows everything."""
        try:
            import numpy as np
            self._embed_onnx(np.zeros(16000, dtype=np.int16).tobytes())
        except Exception:
            pass

    def _embed_onnx(self, pcm: bytes):
        import numpy as np  # lazy
        if self._session is None:
            import kaldi_native_fbank as knf  # lazy
            import onnxruntime as ort  # lazy
            if not self.model_path or not os.path.isfile(self.model_path):
                self.log(f"speaker: model not found at {self.model_path}; ID disabled")
                self.enabled = False
                return None
            self._session = ort.InferenceSession(self.model_path,
                                                 providers=["CPUExecutionProvider"])
            self._knf = knf
        knf = self._knf
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.dither = 0.0
        opts.mel_opts.num_bins = 80
        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(16000, samples.tolist())
        fbank.input_finished()
        feats = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        feats = feats - feats.mean(axis=0, keepdims=True)          # per-utterance CMVN
        out = self._session.run(None, {self._session.get_inputs()[0].name:
                                       feats[np.newaxis].astype(np.float32)})[0]
        return _normalize(out[0].tolist())


# --- audit ---------------------------------------------------------------------------

def _show():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db = os.path.join(here, "memory.db")
    if not os.path.exists(db):
        print("(no memory.db yet)")
        return
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT name, sample_count, dim, model_id, last_heard_ts "
                           "FROM speakers ORDER BY last_heard_ts DESC").fetchall()
    except sqlite3.Error:
        print("(no speakers table yet — feature not run)")
        return
    if not rows:
        print("(no enrolled speakers yet)")
        return
    print("Enrolled voices:")
    for name, n, dim, model_id, ts in rows:
        print(f"  {name:16} {n} samples  dim={dim} model={model_id}  last {ts[:19]}")


if __name__ == "__main__":
    import sys
    if "--show" in sys.argv:
        _show()
    else:
        print(__doc__)
