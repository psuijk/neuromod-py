from __future__ import annotations

from .model import define_model


class Ollama:
    Llama3_2 = define_model("ollama", "llama3.2", max_input=128_000, max_output=4_096)
    Llama3_1 = define_model("ollama", "llama3.1", max_input=128_000, max_output=4_096)
    Qwen2_5 = define_model("ollama", "qwen2.5", max_input=128_000, max_output=8_192)
    Mistral = define_model("ollama", "mistral", max_input=32_000, max_output=4_096)
    DeepSeek_R1 = define_model("ollama", "deepseek-r1", max_input=128_000, max_output=8_192)
