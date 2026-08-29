from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProviderName = Literal["anthropic", "openai", "google", "xai", "ollama", "bedrock"]


@dataclass(frozen=True)
class Model:
    """Identifies which LLM to call, plus its published token limits.

    ``max_input_tokens`` is reference data — the library never enforces it.
    It is there so callers can budget their own context without hardcoding
    a number per model. Values are for the base model at default tier; if
    your account has different limits (e.g. an extended context window),
    override with ``custom_model()``.

    ``max_tokens`` is the default output budget sent with each request.
    It can be overridden per call via ``model(max_tokens=...)``.
    """

    provider: ProviderName
    id: str
    max_input_tokens: int
    max_tokens: int


def define_model(
    provider: ProviderName,
    id: str,
    *,
    max_input: int,
    max_tokens: int,
) -> Model:
    return Model(provider=provider, id=id, max_input_tokens=max_input, max_tokens=max_tokens)


def custom_model(
    provider: ProviderName,
    id: str,
    *,
    max_input: int = 128_000,
    max_tokens: int = 4_096,
) -> Model:
    return Model(provider=provider, id=id, max_input_tokens=max_input, max_tokens=max_tokens)
