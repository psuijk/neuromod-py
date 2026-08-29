from __future__ import annotations

from .model import define_model


class Google:
    """Gemini models on the Gemini API.

    Limits come from Google's published per-model token limits. The common
    Gemini shape is a 1,048,576-token input window with a 65,536-token output
    cap; entries below use the documented exact values rather than round
    numbers so headroom arithmetic stays honest.

    Preview models are billed, more rate-limited, and deprecated with as
    little as two weeks' notice — pin a stable model for production.
    """

    # ── Stable ──
    Flash3_7 = define_model("google", "gemini-3.7-flash", max_input=1_048_576, max_tokens=65_536)
    Flash3_6 = define_model("google", "gemini-3.6-flash", max_input=1_048_576, max_tokens=65_536)
    Flash3_5 = define_model("google", "gemini-3.5-flash", max_input=1_048_576, max_tokens=65_536)
    FlashLite3_5 = define_model("google", "gemini-3.5-flash-lite", max_input=1_048_576, max_tokens=65_536)
    FlashLite3_1 = define_model("google", "gemini-3.1-flash-lite", max_input=1_048_576, max_tokens=65_536)
    Pro2_5 = define_model("google", "gemini-2.5-pro", max_input=1_048_576, max_tokens=65_536)
    Flash2_5 = define_model("google", "gemini-2.5-flash", max_input=1_048_576, max_tokens=65_536)
    FlashLite2_5 = define_model("google", "gemini-2.5-flash-lite", max_input=1_048_576, max_tokens=65_536)

    # ── Preview ──
    Pro3_1Preview = define_model("google", "gemini-3.1-pro-preview", max_input=1_048_576, max_tokens=65_536)
    Flash3Preview = define_model("google", "gemini-3-flash-preview", max_input=1_048_576, max_tokens=65_536)
