"""Pick, verify and preview an ElevenLabs voice for one of the sisters, and
write the choice into .env — without ever starting Aiva.

Read-only against the API (voice list, one voice, the free preview mp3) except
`--say`, which synthesises a line and costs credits: it prints the character
count and the estimated credits BEFORE spending anything (text capped at 200
characters).

  python tools\\pick_voice.py                        list your voices (default)
  python tools\\pick_voice.py --like aiva            ranked by label similarity to ELEVENLABS_VOICE_ID
  python tools\\pick_voice.py --filter gender=female --filter accent=british
  python tools\\pick_voice.py --search warm
  python tools\\pick_voice.py --play 3               free preview of row 3 (or pass a voice id)
  python tools\\pick_voice.py --play-top 3 --inline  first three previews, played in-process
  python tools\\pick_voice.py --verify uSbTIocs0ieRu70Acnzm
  python tools\\pick_voice.py --say "Hi, I'm Saga." --voice uSbTIocs0ieRu70Acnzm   (PAID)
  python tools\\pick_voice.py --set uSbTIocs0ieRu70Acnzm   -> SAGA_ELEVENLABS_VOICE_ID in .env
  python tools\\pick_voice.py --pick                 interactive: list, play, set
  python tools\\pick_voice.py --json                 machine-readable listing

Needs ELEVENLABS_API_KEY in <repo>/.env (or the environment). `.env` is read at
boot, so a changed voice id takes effect the next time she starts.
"""

import argparse
import io
import json
import os
import re
import sys
import tempfile
import wave

import requests
from dotenv import load_dotenv

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENV = os.path.join(REPO_DIR, ".env")
API = "https://api.elevenlabs.io/v1"
PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "aiva_voice_previews")
TARGET_KEY = "SAGA_ELEVENLABS_VOICE_ID"   # what --set writes
ANCHOR_KEY = "ELEVENLABS_VOICE_ID"        # Aiva's voice; new keys go right after it
DEFAULT_MODEL = "eleven_flash_v2_5"
SAY_MAX_CHARS = 200
PCM_RATE = 22050

# how much each label matters when judging "sounds like": who she is (gender,
# age, accent) outweighs the marketing words (description, use case)
LABEL_WEIGHTS = {"gender": 3.0, "age": 2.0, "accent": 2.0, "language": 1.5,
                 "description": 1.0, "descriptive": 1.0, "use_case": 1.0}


# --- API ----------------------------------------------------------------------

def _session(api_key=None):
    key = api_key or os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise SystemExit("ELEVENLABS_API_KEY is not set — put it in .env (or pass --env PATH)")
    s = requests.Session()
    s.headers.update({"xi-api-key": key, "accept": "application/json"})
    return s


def _check(resp, what):
    if resp.ok:
        return resp
    detail = ""
    try:
        detail = resp.json().get("detail")
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("status") or json.dumps(detail)
    except ValueError:
        detail = resp.text[:200]
    raise SystemExit(f"{what} failed: HTTP {resp.status_code} {detail or ''}".rstrip())


def list_voices(session):
    """Every voice in the account's library (premade + cloned/designed)."""
    r = _check(session.get(f"{API}/voices", timeout=30), "listing voices")
    return list(r.json().get("voices") or [])


def get_voice(session, voice_id):
    """One voice by id; exits with a clear message when the id is unknown."""
    r = session.get(f"{API}/voices/{voice_id}", timeout=30)
    if r.status_code == 404 or r.status_code == 400:
        raise SystemExit(f"no voice with id {voice_id!r} in this account")
    return _check(r, f"fetching voice {voice_id}").json()


# --- pure helpers -----------------------------------------------------------------

def labels_of(voice):
    """A voice's labels, keys and values lower-cased and stripped."""
    out = {}
    for k, v in (voice.get("labels") or {}).items():
        if v is None or str(v).strip() == "":
            continue
        out[str(k).strip().lower()] = str(v).strip().lower()
    return out


def similarity(labels_a, labels_b):
    """0..1 — weighted share of label keys (present on either side) whose
    values agree. Two empty label sets are 0.0, not 1.0: nothing is known."""
    keys = set(labels_a) | set(labels_b)
    if not keys:
        return 0.0
    total = matched = 0.0
    for k in keys:
        w = LABEL_WEIGHTS.get(k, 1.0)
        total += w
        a, b = labels_a.get(k), labels_b.get(k)
        if a is not None and b is not None and a == b:
            matched += w
    return round(matched / total, 4) if total else 0.0


def rank_like(voices, ref_labels):
    """[(score, voice), ...] best first; ties broken by name for a stable list."""
    scored = [(similarity(labels_of(v), ref_labels), v) for v in voices]
    scored.sort(key=lambda p: (-p[0], (p[1].get("name") or "").lower()))
    return scored


def parse_filters(items):
    """['gender=female', 'accent=british'] -> {'gender': 'female', ...}"""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--filter expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        k, v = k.strip().lower(), v.strip().lower()
        if not k or not v:
            raise SystemExit(f"--filter expects key=value, got {item!r}")
        out[k] = v
    return out


def apply_filters(voices, filters):
    """Keep voices whose label (or top-level field such as `category`)
    contains each wanted value (case-insensitive substring)."""
    if not filters:
        return list(voices)
    kept = []
    for v in voices:
        labels = labels_of(v)
        ok = True
        for k, want in filters.items():
            have = labels.get(k)
            if have is None:
                top = v.get(k)
                have = str(top).lower() if top is not None else ""
            if want not in have:
                ok = False
                break
        if ok:
            kept.append(v)
    return kept


def search(voices, text):
    """Free-text match over name, description, category and label values."""
    q = (text or "").strip().lower()
    if not q:
        return list(voices)
    hits = []
    for v in voices:
        hay = " ".join([str(v.get("name") or ""), str(v.get("description") or ""),
                        str(v.get("category") or ""), *labels_of(v).values()]).lower()
        if q in hay:
            hits.append(v)
    return hits


def fmt_row(voice, idx=None, score=None):
    lab = labels_of(voice)
    parts = []
    if idx is not None:
        parts.append(f"{idx:>3}.")
    if score is not None:
        parts.append(f"{score:4.2f}")
    parts.append(f"{voice.get('voice_id', '?'):<22}")
    parts.append(f"{(voice.get('name') or '?')[:22]:<22}")
    parts.append(f"{lab.get('gender', '-'):<8}")
    parts.append(f"{lab.get('age', '-'):<12}")
    parts.append(f"{lab.get('accent', '-'):<14}")
    desc = lab.get("description") or lab.get("descriptive") or "-"
    parts.append(f"{desc:<14}")
    parts.append(f"[{voice.get('category') or '-'}]")
    return " ".join(parts)


def credits_for(text, model):
    """Estimated ElevenLabs credits: flash/turbo bill ~0.5 per character,
    the rest ~1 per character."""
    per = 0.5 if any(t in (model or "").lower() for t in ("flash", "turbo")) else 1.0
    return len(text) * per


def pcm16_to_wav(pcm_bytes, rate=PCM_RATE):
    """Raw mono 16-bit PCM -> in-memory WAV (for winsound SND_MEMORY)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm_bytes)
    return buf.getvalue()


def upsert_env(text, key, value, after_key=ANCHOR_KEY):
    """Return `text` (a .env file) with KEY=value set exactly once.

    Order of preference: replace an existing `KEY=` line in place; else turn a
    commented `# KEY=` line into the live one in place; else insert right after
    the `after_key=` line; else append. Idempotent; every other byte, the line
    endings, and the trailing newline are preserved.
    """
    lines = text.splitlines(keepends=True)
    body = f"{key}={value}"
    active = re.compile(rf"^\s*{re.escape(key)}\s*=")
    commented = re.compile(rf"^\s*#\s*{re.escape(key)}\s*=")
    anchor = re.compile(rf"^\s*{re.escape(after_key)}\s*=")
    nl = "\r\n" if "\r\n" in text else "\n"

    def ending(line):
        if line.endswith("\r\n"):
            return "\r\n"
        return "\n" if line.endswith("\n") else ""

    for i, line in enumerate(lines):
        if active.match(line):
            lines[i] = body + ending(line)
            return "".join(lines)
    for i, line in enumerate(lines):
        if commented.match(line):
            lines[i] = body + ending(line)
            return "".join(lines)
    for i, line in enumerate(lines):
        if anchor.match(line):
            if not ending(line):          # anchor was the last, newline-less line
                lines[i] = line + nl
            lines.insert(i + 1, body + ending(lines[i]))
            return "".join(lines)
    if lines and not ending(lines[-1]):
        lines[-1] = lines[-1] + nl
    lines.append(body + nl)
    return "".join(lines)


# --- previews / playback ---------------------------------------------------------

def is_api_url(url):
    """True when `url` is on ElevenLabs' own API host (the only place the
    api key may be sent). Premade voices preview from public storage links;
    generated/cloned ones from api.*.elevenlabs.io/v1/... behind the key."""
    from urllib.parse import urlsplit
    parts = urlsplit(url or "")
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and (host == "elevenlabs.io" or host.endswith(".elevenlabs.io"))


def download_preview(voice, session=None):
    """Cache the voice's free preview mp3 under %TEMP%/aiva_voice_previews;
    None when the voice has no preview. The api key travels only to the API
    host (is_api_url); public storage links are fetched with a bare GET."""
    url = voice.get("preview_url")
    if not url:
        return None
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    path = os.path.join(PREVIEW_DIR, f"{voice.get('voice_id', 'voice')}.mp3")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        getter = session.get if (session is not None and is_api_url(url)) else requests.get
        r = getter(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def play(path, inline=False, timeout_s=60):
    """Play an audio file: the default hands it to the system player
    (os.startfile); --inline plays it here via QtMultimedia and returns when
    it ends (or after timeout_s)."""
    if not inline:
        os.startfile(path)
        return
    from PyQt5.QtCore import QCoreApplication, QTimer, QUrl
    from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
    app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    player = QMediaPlayer()

    def on_status(status):
        if status in (QMediaPlayer.EndOfMedia, QMediaPlayer.InvalidMedia):
            if status == QMediaPlayer.InvalidMedia:
                print(f"  couldn't play {path}")
            app.quit()

    player.mediaStatusChanged.connect(on_status)
    player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(path))))
    player.play()
    QTimer.singleShot(int(timeout_s * 1000), app.quit)
    app.exec_()
    player.stop()


def synth_preview(session, text, voice_id, model=DEFAULT_MODEL, out_path=None,
                  play_audio=True):
    """PAID: synthesise `text` (capped at SAY_MAX_CHARS) on `voice_id`, print
    the cost first, play it via winsound, return the WAV bytes."""
    text = (text or "").strip()
    if not text:
        raise SystemExit("--say needs some text")
    if len(text) > SAY_MAX_CHARS:
        print(f"(trimming to {SAY_MAX_CHARS} characters)")
        text = text[:SAY_MAX_CHARS]
    print(f"synthesising {len(text)} chars with {model} on voice {voice_id} "
          f"— about {credits_for(text, model):g} credits (PAID)", flush=True)
    r = session.post(f"{API}/text-to-speech/{voice_id}",
                     params={"output_format": f"pcm_{PCM_RATE}"},
                     json={"text": text, "model_id": model},
                     headers={"accept": "*/*"}, timeout=120)
    _check(r, "text-to-speech")
    wav = pcm16_to_wav(r.content, PCM_RATE)
    if out_path:
        with open(out_path, "wb") as f:
            f.write(wav)
        print(f"saved {out_path}")
    if play_audio:
        import winsound
        winsound.PlaySound(wav, winsound.SND_MEMORY)
    return wav


# --- .env -------------------------------------------------------------------------

def set_voice(env_path, voice_id, key=TARGET_KEY, session=None):
    """Validate `voice_id` against the API, then write KEY=voice_id into
    `env_path` idempotently (UTF-8, line endings preserved)."""
    session = session or _session()
    voice = get_voice(session, voice_id)
    text = ""
    if os.path.isfile(env_path):
        with open(env_path, encoding="utf-8", newline="") as f:
            text = f.read()
    new = upsert_env(text, key, voice_id)
    if new != text:
        with open(env_path, "w", encoding="utf-8", newline="") as f:
            f.write(new)
        print(f"{key}={voice_id}  ({voice.get('name')}) -> {env_path}")
    else:
        print(f"{key}={voice_id}  ({voice.get('name')}) already in {env_path} — unchanged")
    print(".env is read at boot — she picks the voice up the next time she starts.")
    return voice


def verify(session, voice_id):
    v = get_voice(session, voice_id)
    print(f"voice_id:    {v.get('voice_id')}")
    print(f"name:        {v.get('name')}")
    print(f"category:    {v.get('category')}")
    if v.get("description"):
        print(f"description: {v.get('description')}")
    labels = labels_of(v)
    print(f"labels:      {json.dumps(labels, ensure_ascii=False) if labels else '(none)'}")
    print(f"preview_url: {v.get('preview_url') or '(none)'}")
    return v


# --- CLI ----------------------------------------------------------------------------

def _listing(session, args):
    voices = list_voices(session)
    voices = apply_filters(voices, parse_filters(args.filter))
    if args.search:
        voices = search(voices, args.search)
    scores = None
    if args.like:
        ref_id = os.getenv(ANCHOR_KEY) if args.like.lower() == "aiva" else args.like
        if not ref_id:
            raise SystemExit(f"--like aiva needs {ANCHOR_KEY} in .env")
        ref = next((v for v in voices if v.get("voice_id") == ref_id), None) \
            or get_voice(session, ref_id)
        ranked = rank_like(voices, labels_of(ref))
        voices = [v for _, v in ranked]
        scores = [s for s, _ in ranked]
        print(f"ranked by label similarity to {ref.get('name')!r} "
              f"({json.dumps(labels_of(ref), ensure_ascii=False)})")
    else:
        voices.sort(key=lambda v: (str(v.get("category") or ""),
                                   (v.get("name") or "").lower()))
    return voices, scores


def _print_listing(voices, scores, as_json=False):
    if as_json:
        rows = []
        for i, v in enumerate(voices):
            row = {"voice_id": v.get("voice_id"), "name": v.get("name"),
                   "category": v.get("category"), "labels": labels_of(v),
                   "preview_url": v.get("preview_url")}
            if scores is not None:
                row["score"] = scores[i]
            rows.append(row)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not voices:
        print("(no voices match)")
        return
    for i, v in enumerate(voices, 1):
        print(fmt_row(v, i, scores[i - 1] if scores is not None else None))
    print(f"{len(voices)} voice(s). --play N plays a row's free preview; --set ID writes it to .env.")


def _resolve(token, voices, session):
    """'3' -> third listed voice; a voice id -> that voice (listed or fetched)."""
    if token.isdigit():
        n = int(token)
        if not 1 <= n <= len(voices):
            raise SystemExit(f"row {n} doesn't exist (listing has {len(voices)})")
        return voices[n - 1]
    return next((v for v in voices if v.get("voice_id") == token), None) \
        or get_voice(session, token)


def _play_voice(voice, inline, session):
    path = download_preview(voice, session)
    if not path:
        print(f"  {voice.get('name')}: no free preview available")
        return False
    print(f"  playing preview of {voice.get('name')} ({voice.get('voice_id')})")
    play(path, inline=inline)
    return True


def _pick(session, args, voices, scores):
    _print_listing(voices, scores)
    print("pick: N = play row N, sN = set row N as the voice, q = quit")
    while True:
        try:
            ans = input("> ").strip().lower()
        except EOFError:
            return 0
        if ans in ("q", "quit", ""):
            return 0
        if ans.startswith("s") and ans[1:].isdigit():
            v = _resolve(ans[1:], voices, session)
            set_voice(args.env, v["voice_id"], key=args.key, session=session)
            return 0
        if ans.isdigit():
            try:
                _play_voice(_resolve(ans, voices, session), args.inline, session)
            except SystemExit as e:
                print(e)
            continue
        print("? N, sN or q")


def build_parser():
    p = argparse.ArgumentParser(
        description="Pick/verify/preview an ElevenLabs voice and write it to .env (never starts Aiva)")
    p.add_argument("--list", action="store_true", help="list voices (the default action)")
    p.add_argument("--like", metavar="VOICE_ID|aiva", help="rank by label similarity to this voice")
    p.add_argument("--filter", action="append", default=[], metavar="k=v",
                   help="keep voices whose label k contains v (repeatable)")
    p.add_argument("--search", metavar="TEXT", help="free-text match over name/description/labels")
    p.add_argument("--play", metavar="VOICE_ID|N", help="play the free preview of a voice or listed row")
    p.add_argument("--play-top", type=int, metavar="N", help="play the first N previews of the listing")
    p.add_argument("--inline", action="store_true", help="play in-process (QtMultimedia) instead of the system player")
    p.add_argument("--say", metavar="TEXT", help=f"PAID: synthesise TEXT (<= {SAY_MAX_CHARS} chars) with --voice")
    p.add_argument("--voice", metavar="VOICE_ID", help="voice for --say (default: the .env target key)")
    p.add_argument("--model", default=None, help=f"TTS model for --say (default ELEVENLABS_MODEL or {DEFAULT_MODEL})")
    p.add_argument("--save", metavar="PATH", help="also save --say output as WAV")
    p.add_argument("--verify", metavar="VOICE_ID", help="GET one voice and print name/labels/preview_url")
    p.add_argument("--set", metavar="VOICE_ID", help=f"validate and write {TARGET_KEY}=VOICE_ID into .env")
    p.add_argument("--key", default=TARGET_KEY, help=f"env key --set/--pick write (default {TARGET_KEY})")
    p.add_argument("--env", default=DEFAULT_ENV, help="path of the .env file")
    p.add_argument("--json", action="store_true", help="print the listing as JSON")
    p.add_argument("--pick", action="store_true", help="interactive: list, play rows, set one")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    load_dotenv(args.env)
    session = _session()

    if args.verify:
        verify(session, args.verify)
        return 0
    if args.set:
        set_voice(args.env, args.set, key=args.key, session=session)
        return 0
    if args.say is not None:
        voice_id = args.voice or os.getenv(args.key)
        if not voice_id:
            raise SystemExit(f"--say needs --voice VOICE_ID (or {args.key} in .env)")
        model = args.model or os.getenv("ELEVENLABS_MODEL") or DEFAULT_MODEL
        synth_preview(session, args.say, voice_id, model, out_path=args.save)
        return 0

    voices, scores = _listing(session, args)
    if args.pick:
        return _pick(session, args, voices, scores)
    if args.play:
        v = _resolve(args.play, voices, session)
        print(fmt_row(v))
        return 0 if _play_voice(v, args.inline, session) else 1
    if args.play_top:
        top = voices[:max(0, args.play_top)]
        for i, v in enumerate(top, 1):
            print(fmt_row(v, i, scores[i - 1] if scores is not None else None))
            _play_voice(v, args.inline, session)
            if not args.inline and i < len(top):
                try:
                    if input("  Enter for the next one, q to stop: ").strip().lower() == "q":
                        break
                except EOFError:
                    break
        return 0
    _print_listing(voices, scores, as_json=args.json)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
