from __future__ import annotations

from .model import define_model


class Ollama:
    """Popular models in the Ollama library, by pull count.

    Ollama limits differ from hosted providers in a way worth understanding:
    the numbers below are the *models'* declared windows, but what you
    actually get is whatever ``num_ctx`` your Ollama server is configured
    with, which defaults far lower and is bounded by local RAM/VRAM. They
    also vary by tag — ``gemma3:270m`` is 32K while ``gemma3:27b`` is 128K,
    and quantised tags differ again.

    Treat these as conservative starting points, not guarantees, and
    override per model when you know your own setup::

        custom_model("ollama", "gemma3:27b", max_input=128_000, max_tokens=8_192)

    Tags are supported in the id: ``custom_model("ollama", "qwen3:32b")``.
    """

    # ── Llama ──
    Llama3_3 = define_model("ollama", "llama3.3", max_input=128_000, max_tokens=8_192)
    Llama3_2 = define_model("ollama", "llama3.2", max_input=128_000, max_tokens=4_096)
    Llama3_1 = define_model("ollama", "llama3.1", max_input=128_000, max_tokens=4_096)
    Llama3 = define_model("ollama", "llama3", max_input=8_192, max_tokens=4_096)

    # ── Qwen ──
    Qwen3_5 = define_model("ollama", "qwen3.5", max_input=32_768, max_tokens=8_192)
    Qwen3 = define_model("ollama", "qwen3", max_input=32_768, max_tokens=8_192)
    Qwen2_5 = define_model("ollama", "qwen2.5", max_input=128_000, max_tokens=8_192)
    Qwen2_5Coder = define_model("ollama", "qwen2.5-coder", max_input=32_768, max_tokens=8_192)

    # ── Gemma ──
    Gemma4 = define_model("ollama", "gemma4", max_input=128_000, max_tokens=8_192)
    Gemma3 = define_model("ollama", "gemma3", max_input=128_000, max_tokens=8_192)
    Gemma2 = define_model("ollama", "gemma2", max_input=8_192, max_tokens=4_096)

    # ── Other ──
    DeepSeek_R1 = define_model("ollama", "deepseek-r1", max_input=128_000, max_tokens=8_192)
    Mistral = define_model("ollama", "mistral", max_input=32_000, max_tokens=4_096)
    Phi3 = define_model("ollama", "phi3", max_input=128_000, max_tokens=4_096)
    Llava = define_model("ollama", "llava", max_input=32_768, max_tokens=4_096)
