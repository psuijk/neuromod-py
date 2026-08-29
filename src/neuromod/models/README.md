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
    max_input_tokens: int      # reference only — never sent, never enforced
    max_tokens: int            # default output budget, overridable per request
```

### Token limits

The two limits serve different roles.

`max_input_tokens` is **reference data**. The library never reads it and never
enforces it — it is published so callers can budget their own context without
hardcoding a number per model:

```python
count = await agent.count_tokens(conversation)
headroom = Claude.Sonnet5.max_input_tokens - Claude.Sonnet5.max_tokens - count.tokens
```

Values are for the base model at the default account tier. If your account has
different limits (an extended context window, for instance), override them with
`custom_model()`.

`max_tokens` is a **request parameter** — Anthropic and Bedrock require it in
every request body. The model's value is a default, not a mandate; pass
`max_tokens` per call to override it:

```python
agent = Agent(model=Claude.Sonnet5)                  # defaults to Sonnet5.max_tokens
await agent.generate("Answer in one word.", max_tokens=16)
```

The library does not clamp the override against `max_tokens` — the provider is
the authority on its own limits, and its error is more accurate than ours.

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
    max_tokens=4_096,
)
```

## Design Note

A `Model` identifies what to call, not how to call it. It carries published
limits as data, but only `max_tokens` reaches the wire —
`max_input_tokens` exists for the caller's arithmetic, not the library's. The provider implementation (in `providers/`) handles the actual API communication. Switching models within the same provider is free — the provider instance is reused.
