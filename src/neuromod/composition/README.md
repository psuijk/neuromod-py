# composition

The core composition primitives. Everything is a step function: `async (ConversationContext) -> ConversationContext`.

## Files

- `context.py` — `ConversationContext` class, `StepFunction` type alias, `ToolApprovalRequest`, `StopReason`
- `compose.py` — `compose()` function for sequential pipelines
- `model.py` — `model()` step function (LLM call + tool execution loop)
- `scope.py` — `scope()` for isolated execution with inheritance control
- `thread.py` — `ThreadStore` ABC, `InMemoryThreadStore`, `thread()` wrapper
- `helpers.py` — `when()`, `tap()`, `retry()`, `no_tools_called()`, `tool_not_used_recently()`

## ConversationContext

The state object that flows through every step function. Contains messages, tools, usage, and callbacks.

```python
ctx = ConversationContext(
    messages=[user_message("hello")],  # conversation history
    tools=[my_tool],                    # available tools
    tool_call_limits={"search": 5},     # per-tool call limits
    tool_approval=my_callback,          # async approval gate
    on_event=my_handler,                # sync event callback
    usage=TokenUsage(...),              # accumulated token counts
    stop_reason="stop",                 # "stop" | "max_steps" | "aborted"
    signal=asyncio.Event(),             # cancellation signal
)

# Immutable updates — returns new instance
new_ctx = ctx.with_updates(messages=[...], usage=new_usage)

# Computed properties
ctx.last_request   # most recent user message (or None)
ctx.last_response  # most recent assistant message (or None)
```

## compose()

Runs step functions sequentially. Each step receives the context from the previous step.

```python
pipeline = compose(step_a, step_b, step_c)

# Accepts a string (wraps in user message) or a ConversationContext
result = await pipeline("Hello")
result = await pipeline(existing_context)
```

## model()

The core LLM step function. Calls the provider, runs the tool execution loop, and returns the updated context.

```python
step = model(
    model=Claude.Sonnet4_6,
    system="Be helpful.",              # str or Callable[[ctx], str]
    temperature=0.7,
    schema=MyModel,                    # Pydantic model for structured output
    max_steps=10,                      # max tool-loop iterations
    tool_choice="auto",                # "auto" | "required" | "none"
    api_key="sk-...",
    base_url="https://...",
    timeout=300,                       # request timeout in seconds
)

result = await step(ctx)
```

**The tool loop:**
1. Build a `ProviderRequest` from the context
2. If `on_event` is set, use streaming; otherwise use `generate()`
3. Append assistant message to context
4. Extract tool calls — if none, return with `stop_reason="stop"`
5. Execute all tool calls in parallel (`asyncio.gather`)
6. Append tool results as a user message
7. Repeat until no tool calls or `max_steps` reached

Tool execution includes: unknown tool handling, call limits, approval gates, Pydantic validation, and configurable retries.

## scope()

Creates an isolated environment for step functions with inheritance control. Steps are positional args, config options are keyword args.

```python
from neuromod import scope, Inherit, no_tools_called

# Full isolation — fresh conversation, only scope's tools
isolated = scope(
    research_agent,
    inherit=Inherit.NOTHING,
    tools=[research_tool],
    until=no_tools_called,
)

# Inherit conversation but not tools
conversation_only = scope(agent, inherit=Inherit.CONVERSATION)

# Silent — run steps but don't merge results back to parent
background = scope(logging_step, silent=True)
```

**Inherit flags:**
- `Inherit.NOTHING` — empty messages, no parent tools
- `Inherit.CONVERSATION` — inherit messages, not tools
- `Inherit.TOOLS` — inherit tools, not messages
- `Inherit.ALL` — inherit everything (default)

**until:** Loop condition. Steps run at least once (do-while). When `until(ctx)` returns `True`, the loop ends.

**silent:** If `True`, child results are discarded — parent context is returned unchanged.

## thread()

Wraps a step function with conversation persistence.

```python
from neuromod import thread

step = thread("conversation-id", my_step, store=my_store)
```

1. Loads message history from the store
2. Prepends history to the context
3. Runs the wrapped step
4. Saves the full conversation back to the store

If no `store` is passed, reads from `configure(thread_store=...)`. Raises `RuntimeError` if no store is configured.

## Helpers

```python
# Conditional execution
step = when(lambda ctx: len(ctx.messages) > 5, summarize_agent)

# Side effects (doesn't modify context)
step = tap(lambda ctx: print(f"Messages: {len(ctx.messages)}"))

# Retry on failure
step = retry(RetryOptions(times=3), flaky_step)

# Loop conditions for scope(until=...)
no_tools_called             # True when last response has no tool calls
tool_not_used_recently("search", 5)  # True when "search" not called in last 5 messages
```
