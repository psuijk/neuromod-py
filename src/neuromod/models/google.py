from __future__ import annotations

from .model import define_model


class Google:
    Flash3_5 = define_model("google", "gemini-3.5-flash", max_input=1_000_000, max_tokens=64_000)
    FlashLite3_5 = define_model("google", "gemini-3.5-flash-lite", max_input=1_000_000, max_tokens=64_000)
    Pro2_5 = define_model("google", "gemini-2.5-pro", max_input=1_000_000, max_tokens=64_000)
    Pro3_1Preview = define_model("google", "gemini-3.1-pro-preview", max_input=1_000_000, max_tokens=64_000)
