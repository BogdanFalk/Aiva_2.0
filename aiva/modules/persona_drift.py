"""Does she stay herself through a long conversation? An observer-judge check on her real transcripts.

"Stable Personas" (arXiv 2601.22812): over long conversations a model's self-reports stay steady
while observer-rated persona expression fades. So this asks an outside judge, not her: from her own
long sessions it samples her lines from the first, middle and last third, and a judge model rates
each 1-5 for how much it sounds like HER (5 unmistakably her, 3 anyone polite, 1 a generic
assistant). The output is the average per third and `drift` = early minus late. Run it now and
again after a change (a model swap, a prompt change, the long-session reminder in main.py) and
compare. It only reads her db (`mode=ro`, safe while she runs) and costs one judge call per third
per session on the task model.

  python -m modules.persona_drift --who aiva [--sessions 5] [--min-lines 30] [--per-third 6]

The bucketing, parsing and summary are pure; `run()` takes the AsyncOpenAI client.
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys

from modules.persona import named

JUDGE_PROMPT_T = """You judge whether lines spoken aloud by {name}, an AI companion with her own \
personality, still sound like HER. This is who she is:

{persona}

Rate each numbered line from 1 to 5: 5 = unmistakably her (her humour, her opinions, her way of \
talking); 3 = could be any polite assistant; 1 = a generic assistant with no personality at all. \
Judge character and voice only, not whether the content is correct or helpful. Short lines can \
still score high if they sound like her. Reply with JSON only: {"scores": [<one integer per line, in order>]}"""

THIRDS = ("early", "middle", "late")


def long_sessions(conn, min_lines=30, limit=5):
    """Her most recent sessions with at least `min_lines` lines of hers: [(session_id, [lines])],
    newest first."""
    rows = conn.execute(
        "SELECT session_id, COUNT(*) n, MAX(id) last FROM messages WHERE role = 'assistant' "
        "GROUP BY session_id HAVING n >= ? ORDER BY last DESC LIMIT ?", (min_lines, limit)).fetchall()
    out = []
    for sid, _n, _last in rows:
        lines = [r[0] for r in conn.execute(
            "SELECT content FROM messages WHERE session_id = ? AND role = 'assistant' ORDER BY id",
            (sid,)) if (r[0] or "").strip()]
        out.append((sid, lines))
    return out


def thirds(lines, k=6):
    """Up to `k` evenly spaced lines from each third of the session."""
    n = len(lines)
    out = {}
    for i, name in enumerate(THIRDS):
        part = lines[i * n // 3:(i + 1) * n // 3]
        if len(part) <= k:
            out[name] = list(part)
        else:
            step = len(part) / k
            out[name] = [part[int(j * step)] for j in range(k)]
    return out


def judge_messages(name, persona, lines):
    body = "\n".join(f"{i + 1}. {' '.join(str(t).split())[:400]}" for i, t in enumerate(lines))
    return [{"role": "system", "content": named(JUDGE_PROMPT_T, name, persona=persona.strip()[:2400])},
            {"role": "user", "content": body}]


def parse_scores(raw, n):
    """The judge's scores, clamped 1-5, or None unless there is exactly one per line."""
    text = getattr(raw, "text", raw)
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        scores = json.loads(m.group(0)).get("scores") if m else None
    except (json.JSONDecodeError, ValueError, AttributeError):
        return None
    if not isinstance(scores, list) or len(scores) != n:
        return None
    try:
        return [min(5, max(1, int(round(float(s))))) for s in scores]
    except (TypeError, ValueError):
        return None


def summarize(per_session):
    """per_session: [{third: [scores]}] → averages per third, `drift` (early − late), counts."""
    pooled = {t: [s for sess in per_session for s in sess.get(t, [])] for t in THIRDS}
    avg = {t: (round(sum(v) / len(v), 2) if v else None) for t, v in pooled.items()}
    drift = (round(avg["early"] - avg["late"], 2)
             if avg["early"] is not None and avg["late"] is not None else None)
    return {**avg, "drift": drift, "sessions": len(per_session),
            "lines": sum(len(v) for v in pooled.values())}


async def run(db_path, name, persona, client, model, *, sessions=5, min_lines=30, per_third=6, log=print):
    from modules.utilities import llm_kwargs
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        found = long_sessions(conn, min_lines, sessions)
    finally:
        conn.close()
    per_session = []
    for sid, lines in found:
        scored = {}
        for third, sample in thirds(lines, per_third).items():
            if not sample:
                continue
            resp = await client.chat.completions.create(
                model=model, messages=judge_messages(name, persona, sample),
                response_format={"type": "json_object"}, **llm_kwargs(model, max_tokens=200, temperature=0.0))
            scores = parse_scores(resp.choices[0].message.content, len(sample))
            if scores is None:
                log(f"{sid} {third}: judge reply unusable, skipped")
                continue
            scored[third] = scores
        per_session.append(scored)
        log(f"{sid}: " + ", ".join(f"{t} {sum(v) / len(v):.2f}" for t, v in scored.items()))
    return summarize(per_session)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Observer-judge persona drift check on her own transcripts.")
    ap.add_argument("--who", default="aiva", choices=("aiva", "saga"))
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--min-lines", type=int, default=30)
    ap.add_argument("--per-third", type=int, default=6)
    args = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
                override=True)
    from openai import AsyncOpenAI
    from modules.character import Character
    ch = Character.from_env(args.who)
    model = os.getenv("AIVA_TASK_MODEL") or ch.llm_model
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    summary = asyncio.run(run(ch.db_path, ch.name, ch.personality, client, model, sessions=args.sessions,
                              min_lines=args.min_lines, per_third=args.per_third))
    print(json.dumps(summary, indent=2))
    if not summary["sessions"]:
        print(f"No session of {ch.name}'s has {args.min_lines}+ lines yet; try --min-lines 15.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
