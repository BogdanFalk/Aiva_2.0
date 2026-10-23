"""Read him, feel, then speak: the side-call's messages, a tolerant parse, and a note only when sure."""
from modules.appraisal import AppraisalConfig, appraisal_messages, appraisal_note, parse_appraisal


def test_config_defaults_and_switch():
    c = AppraisalConfig.from_env({})
    assert c.enabled and c.min_confidence == 0.5 and c.note_ttl_s == 600.0
    off = AppraisalConfig.from_env({"AIVA_APPRAISAL": "0", "AIVA_APPRAISAL_MIN_CONFIDENCE": "7"})
    assert not off.enabled and off.min_confidence == 1.0          # clamped


def test_messages_carry_the_tail_his_line_and_how_he_sounded():
    tail = [{"role": "system", "content": "persona"}, {"role": "user", "content": "long day"},
            {"role": "assistant", "content": "Want to talk about it?"}, {"role": "user", "content": None}]
    msgs = appraisal_messages("Saga", tail, "nah, just tired", heard="tired")
    assert msgs[0]["role"] == "system" and "Saga's inner sense" in msgs[0]["content"]
    body = msgs[1]["content"]
    assert "Him: long day" in body and "Saga: Want to talk about it?" in body and "persona" not in body
    assert "His newest line: nah, just tired" in body and "(He sounded tired.)" in body


def test_parse_is_tolerant_and_clamps():
    a = parse_appraisal('```json\n{"him": "He wants to vent, not be fixed.", "me": "I feel protective.",'
                        ' "valence": 3, "arousal": -1, "confidence": "0.8"}\n```')
    assert a == {"him": "He wants to vent, not be fixed.", "me": "I feel protective.",
                 "valence": 1.0, "arousal": 0.0, "confidence": 0.8}
    assert parse_appraisal('{"him": "", "me": "x"}') is None
    assert parse_appraisal("no json here") is None and parse_appraisal(None) is None
    long = parse_appraisal('{"him": "' + "a" * 500 + '", "me": "b"}')
    assert len(long["him"]) == 160


def test_note_only_when_confident():
    a = {"him": "He's teasing.", "me": "I want to tease back.", "valence": 0.4, "arousal": 0.6,
         "confidence": 0.7}
    note = appraisal_note(a, 0.5)
    assert note.startswith("[Reading him") and "He's teasing. I want to tease back." in note
    assert "never say this out loud" in note
    assert appraisal_note({**a, "confidence": 0.3}, 0.5) is None and appraisal_note(None) is None
