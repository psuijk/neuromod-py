from __future__ import annotations

from .model import define_model


class Claude:
    """Claude models on Anthropic's first-party API.

    Limits are for the base model at the default account tier. The 1M context
    window on several models is tier-gated — if your account differs, override
    with ``custom_model("anthropic", ...)``.

    Claude Mythos 5 and Mythos Preview are omitted: both are invitation-only
    (Project Glasswing). Reach them with ``custom_model`` if you have access.
    """

    # ── Claude 5 family ──
    Fable5 = define_model("anthropic", "claude-fable-5", max_input=1_000_000, max_tokens=128_000)
    Opus5 = define_model("anthropic", "claude-opus-5", max_input=1_000_000, max_tokens=128_000)
    Sonnet5 = define_model("anthropic", "claude-sonnet-5", max_input=1_000_000, max_tokens=128_000)

    # ── Claude 4 family ──
    Opus4_8 = define_model("anthropic", "claude-opus-4-8", max_input=1_000_000, max_tokens=128_000)
    Opus4_7 = define_model("anthropic", "claude-opus-4-7", max_input=1_000_000, max_tokens=128_000)
    Opus4_6 = define_model("anthropic", "claude-opus-4-6", max_input=1_000_000, max_tokens=128_000)
    Sonnet4_6 = define_model("anthropic", "claude-sonnet-4-6", max_input=1_000_000, max_tokens=128_000)
    Haiku4_5 = define_model("anthropic", "claude-haiku-4-5", max_input=200_000, max_tokens=64_000)
