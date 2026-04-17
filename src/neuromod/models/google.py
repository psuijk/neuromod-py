from __future__ import annotations

from .model import define_model


class Google:
    Flash2_5 = define_model("google", "gemini-2.5-flash", max_input=1_000_000, max_output=64_000)
    FlashLite2_5 = define_model("google", "gemini-2.5-flash-lite", max_input=1_000_000, max_output=64_000)
    Pro2_5 = define_model("google", "gemini-2.5-pro", max_input=1_000_000, max_output=64_000)
