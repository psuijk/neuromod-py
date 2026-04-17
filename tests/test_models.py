import dataclasses

import pytest

from neuromod.models import (
    Model,
    define_model,
    custom_model,
    Claude,
    Google,
    OpenAI,
    XAI,
)


def test_define_model_creates_frozen_model():
    m = define_model("anthropic", "test-model", max_input=100, max_output=50)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.id = "other"  # type: ignore[misc]


def test_define_model_fields():
    m = define_model("openai", "gpt-test", max_input=1000, max_output=500)
    assert m.provider == "openai"
    assert m.id == "gpt-test"
    assert m.max_input_tokens == 1000
    assert m.max_output_tokens == 500


def test_custom_model_defaults():
    m = custom_model("anthropic", "my-fine-tune")
    assert m.provider == "anthropic"
    assert m.id == "my-fine-tune"
    assert m.max_input_tokens == 128_000
    assert m.max_output_tokens == 4_096


def test_custom_model_override_limits():
    m = custom_model("openai", "ft:gpt-4o", max_input=64_000, max_output=8_000)
    assert m.max_input_tokens == 64_000
    assert m.max_output_tokens == 8_000


def test_model_frozen():
    m = Model(provider="anthropic", id="test", max_input_tokens=100, max_output_tokens=50)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.provider = "openai"  # type: ignore[misc]


def test_claude_models_exist():
    assert isinstance(Claude.Haiku4_5, Model)
    assert isinstance(Claude.Sonnet4_6, Model)
    assert isinstance(Claude.Opus4_6, Model)


def test_claude_provider_field():
    assert Claude.Haiku4_5.provider == "anthropic"
    assert Claude.Sonnet4_6.provider == "anthropic"
    assert Claude.Opus4_6.provider == "anthropic"


def test_google_models_exist():
    assert isinstance(Google.Flash2_5, Model)
    assert isinstance(Google.FlashLite2_5, Model)
    assert isinstance(Google.Pro2_5, Model)


def test_google_provider_field():
    assert Google.Flash2_5.provider == "google"
    assert Google.FlashLite2_5.provider == "google"
    assert Google.Pro2_5.provider == "google"


def test_openai_models_exist():
    assert isinstance(OpenAI.GPT4o, Model)
    assert isinstance(OpenAI.GPT4oMini, Model)
    assert isinstance(OpenAI.GPT4_1, Model)
    assert isinstance(OpenAI.GPT4_1Mini, Model)
    assert isinstance(OpenAI.O3, Model)
    assert isinstance(OpenAI.O4Mini, Model)


def test_openai_provider_field():
    assert OpenAI.GPT4o.provider == "openai"
    assert OpenAI.GPT4oMini.provider == "openai"
    assert OpenAI.GPT4_1.provider == "openai"


def test_xai_models_exist():
    assert isinstance(XAI.Grok3, Model)
    assert isinstance(XAI.Grok3Mini, Model)


def test_xai_provider_field():
    assert XAI.Grok3.provider == "xai"
    assert XAI.Grok3Mini.provider == "xai"
