from __future__ import annotations

from .model import define_model


class OpenAI:
    Sol = define_model("openai", "gpt-5.6-sol", max_input=1_050_000, max_output=128_000)
    Terra = define_model("openai", "gpt-5.6-terra", max_input=1_050_000, max_output=128_000)
    Luna = define_model("openai", "gpt-5.6-luna", max_input=1_050_000, max_output=128_000)
