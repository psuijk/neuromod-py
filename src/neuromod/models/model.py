from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProviderName = Literal["anthropic", "openai", "google", "xai"]


@dataclass(frozen=True)
class Model:
    provider: ProviderName
    id: str
    max_input_tokens: int
    max_output_tokens: int


def define_model(
    provider: ProviderName,
    id: str,
    *,
    max_input: int,
    max_output: int,
) -> Model:
    return Model(provider=provider, id=id, max_input_tokens=max_input, max_output_tokens=max_output)


def custom_model(
    provider: ProviderName,
    id: str,
    *,
    max_input: int = 128_000,
    max_output: int = 4_096,
) -> Model:
    return Model(provider=provider, id=id, max_input_tokens=max_input, max_output_tokens=max_output)
