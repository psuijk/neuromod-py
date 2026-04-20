# providers

The provider abstraction layer. Providers handle the actual HTTP communication with LLM APIs.

## Files

- `provider.py` — `Provider` protocol, `ProviderRequest`, `ProviderResponse`, `ProviderStreamResult`, stream event types, `TokenUsage`, `TokenCount`, `ToolDefinition`, `ToolChoice`
- `errors.py` — Typed error hierarchy: `NeuromodError`, `AuthError`, `RateLimitError`, `NetworkError`, `APIError`
- `factory.py` — `ProviderFactory` with caching and API key resolution
- `anthropic.py` — `ClaudeProvider` implementation using raw httpx

## Provider Protocol

Any class that implements these three methods satisfies the `Provider` protocol:

```python
class Provider(Protocol):
    async def generate(self, request: ProviderRequest) -> ProviderResponse: ...
    def stream(self, request: ProviderRequest) -> ProviderStreamResult: ...
    async def count_tokens(self, request: ProviderRequest) -> TokenCount: ...
```

## ProviderRequest

```python
@dataclass(frozen=True)
class ProviderRequest:
    model: Model                         # required
    messages: list[Message]              # required
    tools: list[ToolDefinition] | None   # tool schemas for the LLM
    tool_choice: ToolChoice | None       # "auto" | "required" | "none"
    system: str | None                   # system prompt
    signal: asyncio.Event | None         # cancellation
    schema: JsonSchema | None            # structured output JSON schema
    temperature: float | None
```

## ProviderResponse / ProviderStreamResult

```python
@dataclass(frozen=True)
class ProviderResponse:
    message: Message       # assistant message with text and/or tool calls
    usage: TokenUsage      # input + output token counts

@dataclass
class ProviderStreamResult:
    events: AsyncIterable[ProviderStreamEvent]  # real-time chunks
    response: Awaitable[ProviderResponse]       # final assembled response
```

## Provider Stream Events

Raw events from the API (no step number — that's added by `model()`):

| Event | Fields | Description |
|-------|--------|-------------|
| `TextDeltaEvent` | `text` | Text chunk arrived |
| `ToolCallStartEvent` | `id, name` | LLM started a tool call |
| `ToolCallDeltaEvent` | `id, arguments_delta` | Partial tool arguments |
| `ToolCallsReadyEvent` | `calls: list[ToolCallInfo]` | All tool calls parsed |

## Error Hierarchy

All errors extend `NeuromodError`. Mapped from HTTP status codes:

| Error | HTTP Status | Fields |
|-------|------------|--------|
| `AuthError` | 401, 403 | `provider` |
| `RateLimitError` | 429 | `provider, retry_after_ms` |
| `NetworkError` | Connection/timeout | `provider` |
| `APIError` | Other | `provider, status_code, body` |

## ProviderFactory

Manages provider instances with caching. One instance per (provider, api_key) pair.

```python
factory = ProviderFactory(ProviderFactoryConfig(
    api_keys={"anthropic": "sk-..."},
    base_urls={"openai": "https://my-proxy.com"},
))
provider = factory.get("anthropic")
```

Typically accessed via `config.get_factory()`, not directly.

## ClaudeProvider (anthropic.py)

Implements the `Provider` protocol for the Anthropic Messages API using raw `httpx` (no SDK).

Key internals:
- `_build_body()` — converts `ProviderRequest` to Anthropic API JSON
- `_convert_messages()` / `_convert_content()` — maps internal types to Anthropic format
- `_parse_response()` / `_parse_message()` — maps Anthropic response back to internal types
- `_parse_sse_stream()` — SSE streaming parser
- `_check_status()` — HTTP status to error hierarchy mapping

Anthropic-specific mappings:
- `tool_call` content → `tool_use` block
- `tool_result` content → `tool_result` block
- `tool_choice="required"` → `{"type": "any"}`
- `tool_choice="none"` → `{"type": "none"}`

## Adding a New Provider

1. Create `providers/my_provider.py` implementing the `Provider` protocol
2. Add a case in `factory.py`'s `_build()` method
3. Add the API key env var to `config.py`'s `_ENV_VAR_NAMES`
4. Add model definitions in `models/my_provider.py`
