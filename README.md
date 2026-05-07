# neuromod

A composable, type-safe LLM inference library for Python. Everything is a step function.

## Install

```bash
pip install neuromod
```

For thread persistence with SQLAlchemy:

```bash
pip install neuromod-sqlalchemy
```

## Quick Start

```python
import asyncio
from neuromod import Agent, Claude, configure

configure(api_keys={"anthropic": "sk-ant-..."})

agent = Agent(model=Claude.Sonnet4_6, system="You are a helpful assistant.")

async def main():
    response = await agent.generate("What is Python?")
    print(response.text)

asyncio.run(main())
```

Or use environment variables for zero-config setup:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
agent = Agent(model=Claude.Sonnet4_6)
response = await agent.generate("Hello")
```

## Core Concepts

### Everything is a Step Function

The fundamental primitive is `StepFunction`: an async callable that takes a `ConversationContext` and returns a new `ConversationContext`.

```python
StepFunction = Callable[[ConversationContext], Awaitable[ConversationContext]]
```

Agents, model calls, scopes, and threads are all step functions. They compose together with `compose()` and `scope()`.

### Agents

The primary public API. Configure once, use everywhere.

```python
from neuromod import Agent, Claude, create_tool
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str

search = create_tool(
    name="search",
    description="Search the web",
    schema=SearchParams,
    execute=lambda p: search_web(p.query),
)

agent = Agent(
    model=Claude.Sonnet4_6,
    system="You are a research assistant.",
    tools=[search],
    max_steps=5,
    temperature=0.7,
)

# Simple generation
response = await agent.generate("Find recent papers on transformers")
print(response.text)
print(response.usage)          # TokenUsage(input_tokens=..., output_tokens=...)
print(response.finish_reason)  # "stop" or "max_steps"

# Streaming
result = agent.stream("Explain quantum computing")
async for event in result.events:
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
response = await result.response

# Per-request overrides
response = await agent.generate(
    "Summarize this",
    temperature=0.3,
    max_steps=1,
    tool_choice="none",
)

# Token counting
count = await agent.count_tokens("How long is this prompt?")
print(count.tokens)
```

See [agents module docs](src/neuromod/agents/) for full API reference.

### Composition

Build pipelines by composing step functions.

```python
from neuromod import compose, scope, Inherit, model, when, no_tools_called

# Sequential pipeline
pipeline = compose(agent_a, agent_b, agent_c)
result = await pipeline("Hello")

# Scoped execution with isolation
sub_task = scope(
    research_agent,
    inherit=Inherit.NOTHING,
    tools=[search_tool],
    until=no_tools_called,
)

pipeline = compose(planner, sub_task, writer)

# Agents are step functions — they work in compose/scope
await compose(agent_a, agent_b)("Hello")
```

See [composition module docs](src/neuromod/composition/) for `compose()`, `scope()`, `model()`, `when()`, `tap()`, `retry()`, and loop conditions.

### Tools

Define tools with Pydantic schemas for type-safe argument validation.

```python
from neuromod import create_tool
from pydantic import BaseModel

class WeatherParams(BaseModel):
    city: str
    units: str = "celsius"

weather = create_tool(
    name="get_weather",
    description="Get current weather for a city",
    schema=WeatherParams,
    execute=fetch_weather,
    max_calls=3,
    requires_approval=True,
    retry=2,
)
```

See [tools module docs](src/neuromod/tools/) for tool configuration options.

### Threads (Persistent Conversations)

Attach thread IDs to persist conversation history across calls.

```python
from neuromod import configure, Agent, Claude
from neuromod_sqlalchemy import SQLAlchemyThreadStore, Base

# Setup with SQLAlchemy
engine = create_async_engine("sqlite+aiosqlite:///threads.db")
async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.create_all)

store = SQLAlchemyThreadStore(async_sessionmaker(engine, class_=AsyncSession))
configure(thread_store=store)

agent = Agent(model=Claude.Sonnet4_6)
await agent.generate("My name is Alice", thread_id="user-123")
response = await agent.generate("What's my name?", thread_id="user-123")
# response.text → "Your name is Alice"
```

See [thread docs](src/neuromod/composition/) and [neuromod-sqlalchemy](packages/neuromod-sqlalchemy/).

### Structured Output

Get typed responses using Pydantic models.

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    topics: list[str]

agent = Agent(model=Claude.Sonnet4_6, schema=Analysis)
response = await agent.generate("Analyze: I love this product!")
print(response.output)  # Analysis(sentiment="positive", confidence=0.95, topics=["product review"])
```

### Configuration

Three-layer precedence: per-agent > `configure()` > environment variables.

```python
from neuromod import configure

# Global config
configure(
    api_keys={"anthropic": "sk-...", "openai": "sk-..."},
    base_urls={"openai": "https://my-proxy.com"},
    timeouts={"anthropic": 300},
    thread_store=my_store,
)

# Per-agent override
agent = Agent(model=Claude.Sonnet4_6, api_key="sk-different-key")

# Environment variable fallback
# ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY
```

See [config module docs](src/neuromod/config.py).

### Streaming Events

Subscribe to real-time events during generation.

```python
def handler(event):
    match event.type:
        case "text_delta":
            print(event.text, end="")
        case "tool_call_start":
            print(f"\nCalling {event.name}...")
        case "tool_complete":
            print(f"Tool result: {event.result}")
        case "step_start":
            print(f"\n--- Step {event.step_number} ---")

response = await agent.generate("Research this topic", on_event=handler)
```

See [streaming module docs](src/neuromod/streaming/) for all event types.

### Cancellation

Cancel in-progress generation using `asyncio.Event`.

```python
import asyncio

signal = asyncio.Event()
task = asyncio.create_task(agent.generate("Write a novel", signal=signal))

# Later:
signal.set()  # Generation stops after current step
response = await task
print(response.finish_reason)  # "aborted"
```

## Architecture

```
neuromod/
├── agents/          # Agent class — the primary public API
├── composition/     # Step functions: compose, scope, model, thread, helpers
├── config.py        # Global configuration with contextvars
├── messages/        # Content types, Message with properties, builders
├── models/          # Model definitions (Claude, Gemini, OpenAI, xAI)
├── providers/       # Provider protocol, error hierarchy, factory
├── streaming/       # Stream event types, Channel
└── tools/           # Tool definition with Pydantic schemas
```

| Module | Description | Docs |
|--------|-------------|------|
| [agents](src/neuromod/agents/) | Agent class with generate/stream/count_tokens | [README](src/neuromod/agents/README.md) |
| [composition](src/neuromod/composition/) | compose, scope, model step, thread, helpers | [README](src/neuromod/composition/README.md) |
| [config](src/neuromod/config.py) | configure(), API key resolution, factory management | Inline |
| [messages](src/neuromod/messages/) | Content types, Message with properties, builder functions | [README](src/neuromod/messages/README.md) |
| [models](src/neuromod/models/) | Model dataclass, provider model registries | [README](src/neuromod/models/README.md) |
| [providers](src/neuromod/providers/) | Provider protocol, errors, factory, Anthropic/Google/OpenAI/Ollama impls | [README](src/neuromod/providers/README.md) |
| [streaming](src/neuromod/streaming/) | StreamEvent union, event types, Channel | [README](src/neuromod/streaming/README.md) |
| [tools](src/neuromod/tools/) | Tool definition, create_tool, convert_tools | [README](src/neuromod/tools/README.md) |

## Design Principles

1. **Step functions are the primitive.** Everything composes through `async (ctx) -> ctx`.
2. **Single source of truth.** `ConversationContext.messages` is the truth.
3. **Content is always a list.** `Message.content` is always `list[Content]`, never `str | list`.
4. **Tool calls are content parts.** They live inside `content[]`, not on separate fields. Roles are `system | user | assistant` only.
5. **Provider = API connection, not model binding.** One provider per API, model passed per-request.
6. **Errors typed by HTTP status.** Auth (401/403), RateLimit (429), Network, API.
7. **Immutability where practical.** Models and tools are frozen. Context updates produce new instances.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Requires Python 3.11+. Dependencies: `pydantic`, `httpx`.
