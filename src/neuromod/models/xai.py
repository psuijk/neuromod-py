from __future__ import annotations

from .model import define_model


class XAI:
    Grok3 = define_model("xai", "grok-3", max_input=131_072, max_output=131_072)
    Grok3Mini = define_model("xai", "grok-3-mini", max_input=131_072, max_output=131_072)
