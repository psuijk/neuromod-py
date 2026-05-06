# agents

The primary public API. An `Agent` wraps a model configuration and provides convenience methods for generation, streaming, and token counting.

## Files

- `agent.py` — `Agent` class with `generate()`, `stream()`, `count_tokens()`, and `__call__`
- `types.py` — `AgentResponse` and `AgentStreamResult` dataclasses

## Agent

```python
from neuromod import Agent, Claude

agent = Agent(
    model=Claude.Sonnet4_6,       # required — which LLM to use
    system="Be helpful.",          # str or Callable[[ConversationContext], str]
    tools=[my_tool],               # list of Tool instances
    max_steps=10,                  # max tool-loop iterations (default: 10)
    temperature=0.7,               # sampling temperature
    schema=MyPydanticModel,        # structured output schema
    api_key="sk-...",              # per-agent API key override
    base_url="https://...",        # per-agent base URL override
    timeout=300,                   # per-agent request timeout in seconds
)
```

### generate()

Runs the full model loop and returns an `AgentResponse`.

```python
response = await agent.generate(
    "user input",
    thread_id="thread-id",         # persist conversation
    model=Claude.Opus4_6,          # override model
    max_steps=3,                   # override max steps
    system="Be concise.",          # override system prompt
    temperature=0.3,               # override temperature
    tool_choice="required",        # "auto" | "required" | "none"
    tool_call_limits={"search": 5},# per-tool call limits
    tool_approval=my_callback,     # async approval callback
    signal=asyncio.Event(),        # cancellation signal
    on_event=my_handler,           # sync event callback
    timeout=300,                   # request timeout in seconds
)
```

### stream()

Returns immediately with an `AgentStreamResult`. Events are consumed via `async for`.

```python
result = agent.stream("user input", **same_options_as_generate)

async for event in result.events:
    if event.type == "text_delta":
        print(event.text, end="")

response = await result.response  # AgentResponse after completion
```

### count_tokens()

Counts tokens for a given input without generating a response.

```python
count = await agent.count_tokens("How many tokens?")
print(count.tokens, count.exact)
```

### __call__()

Makes the Agent usable as a `StepFunction` in `compose()` and `scope()`.

```python
pipeline = compose(agent_a, agent_b)
result = await pipeline("Hello")
```

When used as a step function, context tools override agent tools. If the context has no tools, falls back to the agent's configured tools.

## AgentResponse

```python
@dataclass(frozen=True)
class AgentResponse:
    text: str               # extracted text from last assistant message
    message: Message        # last assistant message
    messages: list[Message] # full conversation history
    finish_reason: StopReason  # "stop" | "max_steps" | "aborted"
    steps: list[StepResult] # results from each step of execution
    usage: TokenUsage       # accumulated token counts
    duration_ms: float      # wall-clock time in milliseconds
    output: Any | None      # parsed structured output (if schema provided)
```

## AgentStreamResult

```python
@dataclass
class AgentStreamResult:
    events: AsyncIterable[StreamEvent]  # async iterate for real-time events
    response: Awaitable[AgentResponse]  # await for final response
```

## Internal Flow

`generate()` and `stream()` both:
1. Build a `ConversationContext` from the input string
2. Create a `model()` step function with the agent's config + overrides
3. Optionally wrap with `thread()` for persistence
4. Execute the step
5. Build an `AgentResponse` from the final context

The difference: `generate()` awaits the step directly, `stream()` runs it as a background task and pipes events through a `Channel`.
