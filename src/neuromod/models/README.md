# models

Model definitions and factories. A `Model` identifies which LLM to use and its token limits.

## Files

- `model.py` — `Model` dataclass, `ProviderName` literal, `define_model()`, `custom_model()`
- `anthropic.py` — Claude model definitions
- `google.py` — Gemini model definitions
- `openai.py` — OpenAI model definitions
- `xai.py` — xAI model definitions

## Model

```python
@dataclass(frozen=True)
class Model:
    provider: ProviderName    # "anthropic" | "openai" | "google" | "xai"
    id: str                   # e.g. "claude-sonnet-4-6"
    max_input_tokens: int
    max_output_tokens: int
```

## Pre-defined Models

```python
from neuromod import Claude, Google, OpenAI, XAI

Claude.Haiku4_5    # claude-haiku-4-5-20251001
Claude.Sonnet4_6   # claude-sonnet-4-6
Claude.Opus4_6     # claude-opus-4-6

Google.Flash2       # etc.
OpenAI.GPT4o       # etc.
XAI.Grok3          # etc.
```

## Custom Models

```python
from neuromod import custom_model

my_model = custom_model(
    "openai",
    "ft:gpt-4o:my-org:custom:id",
    max_input=128_000,
    max_output=4_096,
)
```

## Design Note

A `Model` identifies what to call, not how to call it. The provider implementation (in `providers/`) handles the actual API communication. Switching models within the same provider is free — the provider instance is reused.
