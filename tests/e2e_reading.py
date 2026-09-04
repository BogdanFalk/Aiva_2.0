"""Semi-live reading check: REAL OpenAI + REAL file parsing, NO Aiva pipeline.

    cd aiva && ..\\.venv\\Scripts\\python ..\\tests\\e2e_reading.py <book.epub|txt|md|pdf> [--sections 2 | --all] [--synth]

Copies the file into a temp library, reads it, and prints each section note, the
est/actual token ratio, quote-verification stats, the session wrap, and running
cost. Uses a throwaway database. Good first target: a short public-domain TXT.
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "aiva"))
os.chdir(os.path.join(os.path.dirname(HERE), "aiva"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(HERE), ".env"))

from openai import AsyncOpenAI  # noqa: E402

from modules.memory import Memory  # noqa: E402
from modules.reading import BookMemory, ReadingEngine, embed_long
from modules.tasks import openai_llm_call  # noqa: E402


async def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("usage: e2e_reading.py <book file> [--sections N | --all] [--synth]")
        return
    src = args[0]
    if not os.path.isfile(src):
        print(f"no such file: {src}")
        return
    until_done = "--all" in sys.argv
    sections = None
    if "--sections" in sys.argv:
        sections = int(sys.argv[sys.argv.index("--sections") + 1])

    oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                      max_retries=int(os.getenv("AIVA_LLM_MAX_RETRIES", "5")))
    study_model = os.getenv("AIVA_STUDY_MODEL") or os.getenv("AIVA_TASK_MODEL", "gpt-5.6-luna")
    synth_model = os.getenv("AIVA_STUDY_SYNTH_MODEL", "gpt-5.6-terra")
    lib = tempfile.mkdtemp(prefix="aiva_lib_")
    shutil.copy(src, os.path.join(lib, os.path.basename(src)))
    for fn in os.listdir(lib):
        os.utime(os.path.join(lib, fn), (1000.0, 1000.0))
    memory = Memory(os.path.join(HERE, "e2e_reading.db"))
    bm = BookMemory(memory.conn, embed=lambda t: embed_long(oai, t))
    eng = ReadingEngine(bm, llm_call=openai_llm_call(oai, study_model),
                        synth_call=openai_llm_call(oai, synth_model), client=oai, library_dir=lib,
                        model_name=study_model, synth_model_name=synth_model, log=print)

    async def announce(text):
        print("\n[ANNOUNCE]\n" + text + "\n")

    eng.attach_voice(announce=announce, is_awake=lambda: True)
    print(f"MODEL: {study_model}  SYNTH: {synth_model}\n")
    print("[scan]", await eng.scan_once())
    book = bm.books()[0]
    print(f"BOOK: {book['title']} — {book['author']} ({book['words']} words, {book['sections']} sections)")
    print("[estimate]", eng.estimate(book["id"], sections))
    res = await eng.start_session(book["id"], sections=sections, until_done=until_done)
    print("[start]", res)
    await eng._task
    for n in bm.notes_for(book["id"]):
        if n["skipped"]:
            continue
        print(f"\n--- section {n['section_idx'] + 1}: feeling={n['feeling']}")
        print("  summary:", n["summary"][:200])
        print("  quotes:", [q["text"][:60] for q in n["quotes"]])
        print("  reflection:", n["reflection"][:160])
    if "--synth" in sys.argv or until_done:
        syn = bm.synthesis_for(book["id"])
        if syn:
            print("\n=== SYNTHESIS ===")
            print("  verdict:", syn["verdict"])
            print("  talk_line:", syn["talk_line"])
            print("  takeaways:", syn["takeaways"])
    b = bm.get_book(book["id"])
    print(f"\nDONE: status={b['status']} cursor={b['cursor_section']}/{b['sections']} cost=${b['cost_usd']:.3f}")
    memory.close()


if __name__ == "__main__":
    asyncio.run(main())
