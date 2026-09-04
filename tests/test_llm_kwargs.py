from modules.utilities import LLM_RATES, llm_kwargs


def test_gpt5_shape():
    kw = llm_kwargs("gpt-5.6-luna", max_tokens=500, temperature=0.7)
    assert kw == {"max_completion_tokens": 500, "reasoning_effort": "none", "temperature": 0.7}
    assert "max_tokens" not in kw


def test_gpt4_shape():
    assert llm_kwargs("gpt-4.1-mini", max_tokens=400) == {"max_tokens": 400}
    assert llm_kwargs("gpt-4.1", temperature=0.0) == {"temperature": 0.0}


def test_no_tokens_given():
    assert llm_kwargs("gpt-5.4-nano") == {"reasoning_effort": "none"}
    assert llm_kwargs("gpt-4.1") == {}


def test_rates_cover_current_models():
    for m in ("gpt-4.1", "gpt-4.1-mini", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.4-nano"):
        rate_in, rate_out = LLM_RATES[m]
        assert 0 < rate_in < rate_out
