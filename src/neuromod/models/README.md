# models

Model definitions and factories. A `Model` identifies which LLM to use and its token limits.

## Files

- `model.py` — `Model` dataclass, `ProviderName` literal, `define_model()`, `custom_model()`
- `anthropic.py` — Claude model definitions
- `bedrock.py` — Claude-on-Bedrock model definitions
- `google.py` — Gemini model definitions
- `openai.py` — OpenAI model definitions
- `xai.py` — xAI model definitions

## Model

```python
@dataclass(frozen=True)
class Model:
    provider: ProviderName    # "anthropic" | "openai" | "google" | "xai" | "ollama" | "bedrock"
    id: str                   # e.g. "claude-sonnet-5"
    max_input_tokens: int
    max_output_tokens: int
```

## Pre-defined Models

```python
from neuromod import Claude, Bedrock, Google, OpenAI, XAI

Claude.Haiku4_5    # claude-haiku-4-5
Claude.Sonnet5     # claude-sonnet-5
Claude.Opus5       # claude-opus-5

Bedrock.Claude3_5_Sonnet_v2   # anthropic.claude-3-5-sonnet-20241022-v2:0
Bedrock.Claude3_5_Haiku       # anthropic.claude-3-5-haiku-20241022-v1:0

Google.Flash3_5      # etc.
OpenAI.Sol         # etc.
XAI.Grok3          # etc.
```

## Custom Models

```python
from neuromod import custom_model

my_model = custom_model(
    "openai",
    "ft:gpt-5.6-luna:my-org:custom:id",
    max_input=128_000,
    max_output=4_096,
)
```

## Design Note

A `Model` identifies what to call, not how to call it. The provider implementation (in `providers/`) handles the actual API communication. Switching models within the same provider is free — the provider instance is reused.
