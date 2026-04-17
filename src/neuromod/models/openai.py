from __future__ import annotations

from .model import define_model


class OpenAI:
    GPT4o = define_model("openai", "gpt-4o", max_input=128_000, max_output=16_384)
    GPT4oMini = define_model("openai", "gpt-4o-mini", max_input=128_000, max_output=16_384)
    GPT4_1 = define_model("openai", "gpt-4.1", max_input=1_047_576, max_output=32_768)
    GPT4_1Mini = define_model("openai", "gpt-4.1-mini", max_input=1_047_576, max_output=32_768)
    O3 = define_model("openai", "o3", max_input=200_000, max_output=100_000)
    O4Mini = define_model("openai", "o4-mini", max_input=200_000, max_output=100_000)
