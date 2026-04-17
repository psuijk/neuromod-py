from __future__ import annotations

from .model import define_model


class Claude:
    Haiku4_5 = define_model("anthropic", "claude-haiku-4-5-20251001", max_input=200_000, max_output=8_096)
    Sonnet4_6 = define_model("anthropic", "claude-sonnet-4-6", max_input=200_000, max_output=64_000)
    Opus4_6 = define_model("anthropic", "claude-opus-4-6", max_input=200_000, max_output=32_000)
