"""Persona drift: sessions long enough, even thirds, a strict parse, and early-minus-late."""
from modules.memory import Memory
from modules.persona_drift import judge_messages, long_sessions, parse_scores, summarize, thirds


def test_only_long_sessions_newest_first():
    m = Memory(":memory:")
    for sid, n in (("old", 40), ("short", 5), ("new", 31)):
        m.save_new_messages(sid, [{"role": r, "content": f"{sid} {i}"}
                                  for i in range(n) for r in ("user", "assistant")])
        m._saved_count = 0
    found = long_sessions(m.conn, min_lines=30, limit=5)
    assert [sid for sid, _ in found] == ["new", "old"]
    assert found[0][1][0] == "new 0" and len(found[1][1]) == 40


def test_thirds_are_even_and_capped():
    t = thirds([str(i) for i in range(30)], k=3)
    assert t == {"early": ["0", "3", "6"], "middle": ["10", "13", "16"], "late": ["20", "23", "26"]}
    small = thirds(["a", "b", "c", "d"], k=6)
    assert small == {"early": ["a"], "middle": ["b"], "late": ["c", "d"]}


def test_parse_scores_needs_one_per_line():
    assert parse_scores('{"scores": [5, 4.6, 0, 9]}', 4) == [5, 5, 1, 5]
    assert parse_scores('```json\n{"scores": [3, 2]}\n```', 2) == [3, 2]
    assert parse_scores('{"scores": [3]}', 2) is None
    assert parse_scores("nope", 1) is None and parse_scores('{"scores": ["x"]}', 1) is None


def test_summary_and_drift():
    s = summarize([{"early": [5, 5], "middle": [4], "late": [3, 3]}, {"early": [4], "late": [2]}])
    assert s == {"early": 4.67, "middle": 4.0, "late": 2.67, "drift": 2.0, "sessions": 2, "lines": 7}
    assert summarize([])["drift"] is None


def test_judge_messages_number_the_lines_and_carry_who_she_is():
    msgs = judge_messages("Aiva", "You are Aiva. Cheeky.", ["Hi there.", "Fine,\nwhatever."])
    assert "You are Aiva. Cheeky." in msgs[0]["content"] and "sound like HER" in msgs[0]["content"]
    assert msgs[1]["content"] == "1. Hi there.\n2. Fine, whatever."
