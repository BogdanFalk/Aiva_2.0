"""The GPT-Live trial's pure parts: it refuses to run beside the house, reads her facts read-only,
tells her it's a trial, prices the session, designs a voice with the documented request."""
import json
import os

import pytest

import live_trial as lt
from modules.memory import Memory


def test_refuses_while_the_house_runs(tmp_path):
    state = tmp_path / "house_state.json"
    assert lt.house_running(str(state)) is None                      # no file
    state.write_text("not json")
    assert lt.house_running(str(state)) is None
    state.write_text(json.dumps({"pid": os.getpid()}))
    assert lt.house_running(str(state)) == os.getpid()               # a live process
    state.write_text(json.dumps({"pid": 0}))
    assert lt.house_running(str(state)) is None


def test_facts_are_read_without_her_sister_and_without_writing(tmp_path):
    db = tmp_path / "memory.db"
    m = Memory(str(db))
    m.upsert_facts([{"category": "person", "fact": "He is Romanian."},
                    {"category": "sister", "fact": "Saga draws cats."}])
    m.close()
    before = db.stat().st_mtime
    assert lt.load_facts(str(db)) == ["He is Romanian."]
    assert db.stat().st_mtime == before
    assert lt.load_facts(str(tmp_path / "missing.db")) == []


def test_instructions_carry_her_persona_facts_the_trial_and_the_voice():
    text = lt.build_instructions("Saga", "You are Saga.", "still Saga", ["He is Romanian."],
                                 voice_description="a calm, low adult voice")
    assert text.startswith("You are Saga.") and "He is Romanian." in text
    assert "THIS IS A TRIAL" in text and "You're still Saga." in text and "Nothing from this conversation is saved" in text
    assert text.endswith("YOUR VOICE in this trial: a calm, low adult voice")
    assert "YOUR VOICE" not in lt.build_instructions("Aiva", "You are Aiva.", "the same cheeky Aiva", [])


def test_cost_is_per_second_of_session():
    assert lt.session_cost(60) == 0.05 and lt.session_cost(90) == 0.075 and lt.session_cost(-5) == 0.0


class _Resp:
    def __init__(self, status, data):
        self.status_code, self._data, self.text = status, data, json.dumps(data)

    def json(self):
        return self._data


def test_voice_design_sends_the_documented_request():
    sent = {}

    def post(url, headers, json, timeout):
        sent.update(url=url, auth=headers["Authorization"], body=json)
        return _Resp(200, {"id": "voice_abc", "name": json["name"]})

    assert lt.design_voice("sk-1", "Saga trial", "x" * 3000, post=post) == "voice_abc"
    assert sent["url"] == lt.VOICES_URL and sent["auth"] == "Bearer sk-1"
    assert sent["body"]["type"] == "prompt" and sent["body"]["name"] == "Saga trial"
    assert len(sent["body"]["prompt"]) == 2000                       # the API's limit
    with pytest.raises(RuntimeError, match="HTTP 400"):
        lt.design_voice("sk-1", "n", "d", post=lambda *a, **k: _Resp(400, {"error": "no"}))
    with pytest.raises(RuntimeError, match="no id"):
        lt.design_voice("sk-1", "n", "d", post=lambda *a, **k: _Resp(200, {}))


def test_args_and_the_service():
    a = lt.parse_args([])
    assert (a.who, a.voice, a.design, a.minutes, a.check) == ("aiva", None, None, 10.0, False)
    s = lt.build_service("sk-test", "You are Aiva.", "marin", "gpt-6-luna")
    assert s._settings.voice == "marin" and s._settings.model == "gpt-live-1"
    from pipecat.services.openai.live.llm import _responses_delegation_config
    assert _responses_delegation_config(s._delegation) == {"model": "gpt-6-luna", "tools": [{"type": "web_search"}]}
