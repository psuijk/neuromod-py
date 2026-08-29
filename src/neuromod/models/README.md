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
from neuromod import Claude, Bedrock, Google, OpenAI, XAI, Ollama
```

| Catalog | Models | Notes |
| --- | --- | --- |
| `Claude` | `Fable5`, `Opus5`, `Sonnet5`, `Opus4_8`, `Opus4_7`, `Opus4_6`, `Sonnet4_6`, `Haiku4_5` | 1M context except Haiku 4.5 (200K) |
| `Bedrock` | `Fable5`, `Opus5`, `Sonnet5`, `Opus4_8`, `Opus4_7`, `Haiku4_5`, `Opus4_6`, `Sonnet4_6`, `Opus4_5`, `Sonnet4_5` (+ legacy Claude 3.x) | Two id families — see below |
| `Google` | `Flash3_7`, `Flash3_6`, `Flash3_5`, `FlashLite3_5`, `FlashLite3_1`, `Pro2_5`, `Flash2_5`, `FlashLite2_5`, `Pro3_1Preview`, `Flash3Preview` | 1,048,576 in / 65,536 out |
| `OpenAI` | `Sol`, `Terra`, `Luna` | GPT-5.6 family, 1.05M in / 128K out |
| `XAI` | `Grok4_6`, `Grok4_5`, `Grok4_3`, `Grok4_20Reasoning`, `Grok4_20NonReasoning`, `Grok4_20MultiAgent`, `GrokBuild0_1` (+ legacy Grok 3) | Output cap is a conservative default |
| `Ollama` | `Llama3_3`, `Llama3_2`, `Llama3_1`, `Llama3`, `Qwen3_5`, `Qwen3`, `Qwen2_5`, `Qwen2_5Coder`, `Gemma4`, `Gemma3`, `Gemma2`, `DeepSeek_R1`, `Mistral`, `Phi3`, `Llava` | Real limit is your server's `num_ctx` |

Each catalog's docstring carries the caveats; the ones worth knowing up front:

**Bedrock has two id families, both reachable through `InvokeModel`.** Models
from Claude Opus 4.7 onward use unversioned ids (`anthropic.claude-opus-5`)
and are passed as-is. Claude Opus 4.6 and earlier are ARN-versioned and served
only through cross-region inference — a bare base id returns HTTP 400 asking
for an inference profile, so those entries carry a `global.` prefix. Swap it
for `us.`/`eu.`/`jp.`/`apac.` (10% premium) when you need data residency.

**This provider only speaks Anthropic's wire format.** Bedrock hosts Nova,
Llama, Mistral, Titan and many more, but they use different request shapes and
are not reachable through this provider.

**xAI publishes no max-output limit.** Output shares the context window with
the prompt. Those entries default to a conservative 32,768 — raise it per
request when you need longer output.

**Ollama limits are advisory.** What you actually get is your server's
`num_ctx`, bounded by local RAM/VRAM, and it varies by tag (`gemma3:270m` is
32K, `gemma3:27b` is 128K).

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
