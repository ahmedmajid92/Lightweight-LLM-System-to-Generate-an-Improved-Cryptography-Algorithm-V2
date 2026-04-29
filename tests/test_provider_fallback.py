from types import SimpleNamespace

import pytest

from cipherlab.cipher.spec import CipherSpec, ImprovementPatch
from cipherlab.config import Settings
from cipherlab.llm import assistant as assistant_module
from cipherlab.llm.openai_provider import LLMResult, LLMUsage, OpenAIProvider


class DummySchema(ImprovementPatch):
    pass


def _make_provider(with_openrouter: bool = True) -> OpenAIProvider:
    provider = object.__new__(OpenAIProvider)
    provider.client = object()
    provider.openrouter_client = object() if with_openrouter else None
    return provider


def test_generate_text_with_fallback_prefers_primary(monkeypatch):
    provider = _make_provider()
    expected = LLMResult(model="deepseek", text="ok", raw=object(), usage=LLMUsage())

    monkeypatch.setattr(provider, "generate_text_chat", lambda **_: expected)
    monkeypatch.setattr(provider, "generate_text", lambda **_: pytest.fail("fallback should not be called"))

    result, model_used = provider.generate_text_with_fallback(
        openrouter_model="deepseek",
        fallback_model="gpt",
        system="s",
        user="u",
    )

    assert result is expected
    assert model_used == "deepseek"


def test_generate_text_with_fallback_uses_openai_when_primary_errors(monkeypatch):
    provider = _make_provider()
    fallback_result = LLMResult(model="gpt", text="fallback", raw=object(), usage=LLMUsage())

    def _raise_primary(**_):
        raise RuntimeError("primary failed")

    monkeypatch.setattr(provider, "generate_text_chat", _raise_primary)
    monkeypatch.setattr(provider, "generate_text", lambda **_: fallback_result)

    result, model_used = provider.generate_text_with_fallback(
        openrouter_model="deepseek",
        fallback_model="gpt",
        system="s",
        user="u",
    )

    assert result is fallback_result
    assert model_used == "gpt"


def test_generate_structured_with_fallback_uses_openai_when_primary_errors(monkeypatch):
    provider = _make_provider()
    fallback_patch = ImprovementPatch(summary="fallback", rationale=["ok"])

    def _raise_primary(**_):
        raise RuntimeError("primary failed")

    monkeypatch.setattr(provider, "generate_json_chat", _raise_primary)
    monkeypatch.setattr(provider, "generate_structured", lambda **_: (fallback_patch, object()))

    patch, raw, model_used = provider.generate_structured_with_fallback(
        openrouter_model="deepseek",
        fallback_model="gpt",
        system="s",
        user="u",
        schema=DummySchema,
    )

    assert patch is fallback_patch
    assert raw is not None
    assert model_used == "gpt"


def test_generate_text_with_fallback_uses_local_on_connection_failure(monkeypatch):
    provider = _make_provider(with_openrouter=False)
    local_result = LLMResult(model="local-code", text="local", raw=object(), usage=LLMUsage())

    def _raise_connection(**_):
        raise ConnectionError("network is unreachable")

    monkeypatch.setattr(provider, "generate_text", _raise_connection)
    monkeypatch.setattr(provider, "generate_text_local", lambda **_: local_result)

    result, model_used = provider.generate_text_with_fallback(
        openrouter_model=None,
        fallback_model="gpt",
        local_model="local-code",
        local_role="code",
        system="s",
        user="u",
    )

    assert result is local_result
    assert model_used == "local-code"


def test_generate_structured_with_fallback_does_not_use_local_on_schema_failure(monkeypatch):
    provider = _make_provider(with_openrouter=False)

    def _raise_schema(**_):
        raise ValueError("schema validation failed")

    monkeypatch.setattr(provider, "generate_structured", _raise_schema)
    monkeypatch.setattr(provider, "generate_structured_local", lambda **_: pytest.fail("local fallback should not run"))

    with pytest.raises(ValueError, match="schema validation failed"):
        provider.generate_structured_with_fallback(
            openrouter_model=None,
            fallback_model="gpt",
            local_model="local-reasoning",
            local_role="reasoning",
            system="s",
            user="u",
            schema=DummySchema,
        )


def test_generate_text_retries_without_temperature_when_model_rejects_it():
    provider = _make_provider(with_openrouter=False)
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "temperature" in kwargs:
                raise RuntimeError("Unsupported parameter: 'temperature' is not supported with this model.")
            return SimpleNamespace(
                output_text="ok",
                usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            )

    provider.client = SimpleNamespace(responses=FakeResponses())

    result = provider.generate_text(
        model="gpt-5.2-codex",
        system="sys",
        user="usr",
        temperature=0.2,
        max_output_tokens=123,
    )

    assert result.text == "ok"
    assert len(calls) == 2
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]


def test_generate_structured_retries_without_temperature_when_model_rejects_it():
    provider = _make_provider(with_openrouter=False)
    calls = []
    parsed_patch = ImprovementPatch(summary="retry succeeded", rationale=["retry worked"])

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            if "temperature" in kwargs:
                raise RuntimeError("Unsupported parameter: 'temperature' is not supported with this model.")
            return SimpleNamespace(output_parsed=parsed_patch)

    provider.client = SimpleNamespace(responses=FakeResponses())

    patch, raw = provider.generate_structured(
        model="gpt-5.2",
        system="sys",
        user="usr",
        schema=DummySchema,
        temperature=0.0,
        max_output_tokens=321,
    )

    assert patch is parsed_patch
    assert raw is not None
    assert len(calls) == 2
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]


def test_suggest_improvements_reports_fallback_model(monkeypatch):
    patch = ImprovementPatch(summary="fallback patch", rationale=["ok"])

    class FakeProvider:
        def __init__(self, *args, **kwargs):
            pass

        def generate_structured_with_fallback(self, **kwargs):
            return patch, SimpleNamespace(), kwargs["fallback_model"]

    monkeypatch.setattr(assistant_module, "OpenAIProvider", FakeProvider)

    settings = Settings(openai_api_key="x", openrouter_api_key="y")
    spec = CipherSpec(
        name="Test",
        architecture="SPN",
        block_size_bits=128,
        key_size_bits=128,
        rounds=10,
        components={
            "sbox": "sbox.aes",
            "perm": "perm.aes_shiftrows",
            "linear": "linear.aes_mixcolumns",
            "key_schedule": "ks.sha256_kdf",
        },
    )

    returned_patch, raw, model_used = assistant_module.suggest_improvements(
        settings=settings,
        spec=spec,
        metrics={"plaintext_avalanche": {"mean": 0.4}},
        issues=["weak diffusion"],
        rag_context="context",
        model="gpt-fallback",
        openrouter_model="deepseek-primary",
    )

    assert returned_patch is patch
    assert raw is not None
    assert model_used == "gpt-fallback"
