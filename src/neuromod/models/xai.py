from __future__ import annotations

from .model import define_model


class XAI:
    """Grok models on the xAI API.

    Context windows are xAI's published figures. **xAI does not publish a
    max-output-token limit** — output is drawn from the same window as the
    prompt, and reasoning tokens consume part of it. The ``max_tokens``
    values below are therefore a conservative 32,768 default rather than a
    documented ceiling; raise it per request when you need longer output::

        await agent.generate(prompt, max_tokens=65_536)
    """

    _MAX_TOKENS = 32_768  # conservative default — not a published limit

    Grok4_6 = define_model("xai", "grok-4.6", max_input=500_000, max_tokens=_MAX_TOKENS)
    Grok4_5 = define_model("xai", "grok-4.5", max_input=500_000, max_tokens=_MAX_TOKENS)
    Grok4_3 = define_model("xai", "grok-4.3", max_input=1_000_000, max_tokens=_MAX_TOKENS)
    Grok4_20Reasoning = define_model(
        "xai", "grok-4.20-0309-reasoning", max_input=1_000_000, max_tokens=_MAX_TOKENS,
    )
    Grok4_20NonReasoning = define_model(
        "xai", "grok-4.20-0309-non-reasoning", max_input=1_000_000, max_tokens=_MAX_TOKENS,
    )
    Grok4_20MultiAgent = define_model(
        "xai", "grok-4.20-multi-agent-0309", max_input=1_000_000, max_tokens=_MAX_TOKENS,
    )
    GrokBuild0_1 = define_model("xai", "grok-build-0.1", max_input=256_000, max_tokens=_MAX_TOKENS)

