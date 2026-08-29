"""Per-request max_tokens override.

The model's ``max_tokens`` is a default, not a mandate — a caller can cap
output per request. Every provider must honour the override and fall back
to the model's value when none is given.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from neuromod.messages.helpers import user_message
from neuromod.models import Bedrock, Claude, Google, Ollama, OpenAI, custom_model
from neuromod.models.model import Model
from neuromod.providers.anthropic import _build_body as anthropic_body
from neuromod.providers.bedrock import _build_body as bedrock_body
from neuromod.providers.google import _build_body as google_body
from neuromod.providers.ollama import _build_body as ollama_body
from neuromod.providers.openai import _build_body as openai_body
from neuromod.providers.provider import ProviderRequest

# (label, model, body builder, reader pulling the output cap off the built body)
# Google nests its cap under generationConfig; the rest put it at the top level.
PROVIDERS: list[tuple[str, Model, Callable[..., dict[str, Any]], Callable[[dict[str, Any]], int]]] = [
    ("anthropic", Claude.Sonnet5, lambda r: anthropic_body(r, stream=False), lambda b: b["max_tokens"]),
    ("bedrock", Bedrock.Claude3_5_Sonnet_v2, bedrock_body, lambda b: b["max_tokens"]),
    ("openai", OpenAI.Sol, lambda r: openai_body(r, stream=False), lambda b: b["max_tokens"]),
    ("google", Google.Pro2_5, google_body, lambda b: b["generationConfig"]["maxOutputTokens"]),
    ("ollama", Ollama.Llama3_2, lambda r: ollama_body(r, stream=False), lambda b: b["max_tokens"]),
]

IDS = [p[0] for p in PROVIDERS]


@pytest.mark.parametrize("label,model,build,cap_of", PROVIDERS, ids=IDS)
def test_falls_back_to_model_default(label, model, build, cap_of):
    body = build(ProviderRequest(model=model, messages=[user_message("hi")]))
    assert cap_of(body) == model.max_tokens


@pytest.mark.parametrize("label,model,build,cap_of", PROVIDERS, ids=IDS)
def test_request_overrides_model_default(label, model, build, cap_of):
    body = build(ProviderRequest(model=model, messages=[user_message("hi")], max_tokens=64))
    assert cap_of(body) == 64


@pytest.mark.parametrize("label,model,build,cap_of", PROVIDERS, ids=IDS)
def test_override_may_exceed_model_default(label, model, build, cap_of):
    """The library does not clamp — the provider is the authority on its own limits."""
    over = model.max_tokens + 1_000
    body = build(ProviderRequest(model=model, messages=[user_message("hi")], max_tokens=over))
    assert cap_of(body) == over


def test_max_input_tokens_is_never_sent():
    """max_input_tokens is reference data for callers; it must not reach the wire."""
    model = custom_model("anthropic", "test", max_input=12_345, max_tokens=100)
    body = anthropic_body(ProviderRequest(model=model, messages=[user_message("hi")]), stream=False)
    assert 12_345 not in body.values()
    assert "max_input_tokens" not in body
