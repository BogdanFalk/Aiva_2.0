from modules.utilities import LLM_RATES, gpt5_shape, llm_kwargs


def test_gpt5_shape():
    kw = llm_kwargs("gpt-5.6-luna", max_tokens=500, temperature=0.7)
    assert kw == {"max_completion_tokens": 500, "reasoning_effort": "none", "temperature": 0.7}
    assert "max_tokens" not in kw


def test_gpt6_luna_takes_the_gpt5_shape():
    # without this a gpt-6-luna turn is a 400 on max_tokens (live call, 2026-10-23)
    kw = llm_kwargs("gpt-6-luna", max_tokens=500, temperature=0.7)
    assert kw == {"max_completion_tokens": 500, "reasoning_effort": "none", "temperature": 0.7}
    assert gpt5_shape("gpt-6-luna") and gpt5_shape("gpt-5.6-luna")


def test_gpt6_1_sol_is_not_given_effort_none():
    # it rejects reasoning_effort "none" (pipecat 1.13 notes)
    assert not gpt5_shape("gpt-6.1-sol")
    assert "reasoning_effort" not in llm_kwargs("gpt-6.1-sol", max_tokens=100)


def test_gpt4_shape():
    assert llm_kwargs("gpt-4.1-mini", max_tokens=400) == {"max_tokens": 400}
    assert llm_kwargs("gpt-4.1", temperature=0.0) == {"temperature": 0.0}
    assert not gpt5_shape("gpt-4.1-mini") and not gpt5_shape(None)


def test_no_tokens_given():
    assert llm_kwargs("gpt-5.4-nano") == {"reasoning_effort": "none"}
    assert llm_kwargs("gpt-4.1") == {}


def test_rates_cover_current_models():
    for m in ("gpt-4.1", "gpt-4.1-mini", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.4-nano", "gpt-6-luna"):
        rate_in, rate_out = LLM_RATES[m]
        assert 0 < rate_in < rate_out
