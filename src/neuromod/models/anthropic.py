from __future__ import annotations

from .model import define_model


class Claude:
    Haiku4_5 = define_model("anthropic", "claude-haiku-4-5", max_input=200_000, max_tokens=64_000)
    Sonnet5 = define_model("anthropic", "claude-sonnet-5", max_input=1_000_000, max_tokens=128_000)
    Opus5 = define_model("anthropic", "claude-opus-5", max_input=1_000_000, max_tokens=128_000)
