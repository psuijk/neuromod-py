import dataclasses

import pytest

from neuromod.models import (
    Model,
    define_model,
    custom_model,
    Claude,
    Google,
    Ollama,
    OpenAI,
    XAI,
)


def test_define_model_creates_frozen_model():
    m = define_model("anthropic", "test-model", max_input=100, max_tokens=50)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.id = "other"  # type: ignore[misc]


def test_define_model_fields():
    m = define_model("openai", "gpt-test", max_input=1000, max_tokens=500)
    assert m.provider == "openai"
    assert m.id == "gpt-test"
    assert m.max_input_tokens == 1000
    assert m.max_tokens == 500


def test_custom_model_defaults():
    m = custom_model("anthropic", "my-fine-tune")
    assert m.provider == "anthropic"
    assert m.id == "my-fine-tune"
    assert m.max_input_tokens == 128_000
    assert m.max_tokens == 4_096


def test_custom_model_override_limits():
    m = custom_model("openai", "ft:gpt-4o", max_input=64_000, max_tokens=8_000)
    assert m.max_input_tokens == 64_000
    assert m.max_tokens == 8_000


def test_model_frozen():
    m = Model(provider="anthropic", id="test", max_input_tokens=100, max_tokens=50)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.provider = "openai"  # type: ignore[misc]


def test_claude_models_exist():
    assert isinstance(Claude.Haiku4_5, Model)
    assert isinstance(Claude.Sonnet5, Model)
    assert isinstance(Claude.Opus5, Model)


def test_claude_provider_field():
    assert Claude.Haiku4_5.provider == "anthropic"
    assert Claude.Sonnet5.provider == "anthropic"
    assert Claude.Opus5.provider == "anthropic"


def test_google_models_exist():
    assert isinstance(Google.Flash3_5, Model)
    assert isinstance(Google.FlashLite3_5, Model)
    assert isinstance(Google.Pro2_5, Model)
    assert isinstance(Google.Pro3_1Preview, Model)


def test_google_provider_field():
    assert Google.Flash3_5.provider == "google"
    assert Google.FlashLite3_5.provider == "google"
    assert Google.Pro2_5.provider == "google"
    assert Google.Pro3_1Preview.provider == "google"


def test_openai_models_exist():
    assert isinstance(OpenAI.Sol, Model)
    assert isinstance(OpenAI.Terra, Model)
    assert isinstance(OpenAI.Luna, Model)


def test_openai_provider_field():
    assert OpenAI.Sol.provider == "openai"
    assert OpenAI.Terra.provider == "openai"
    assert OpenAI.Luna.provider == "openai"


def test_xai_models_exist():
    assert isinstance(XAI.Grok4_6, Model)
    assert isinstance(XAI.Grok4_5, Model)


def test_xai_provider_field():
    assert XAI.Grok4_6.provider == "xai"
    assert XAI.Grok4_5.provider == "xai"


def test_ollama_models_exist():
    assert isinstance(Ollama.Llama3_2, Model)
    assert isinstance(Ollama.Llama3_1, Model)
    assert isinstance(Ollama.Qwen2_5, Model)
    assert isinstance(Ollama.Mistral, Model)
    assert isinstance(Ollama.DeepSeek_R1, Model)


def test_ollama_provider_field():
    assert Ollama.Llama3_2.provider == "ollama"
    assert Ollama.Llama3_1.provider == "ollama"
    assert Ollama.Qwen2_5.provider == "ollama"


def test_ollama_model_ids():
    assert Ollama.Llama3_2.id == "llama3.2"
    assert Ollama.Mistral.id == "mistral"
    assert Ollama.DeepSeek_R1.id == "deepseek-r1"
