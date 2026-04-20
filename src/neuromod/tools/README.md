# tools

Tool definitions for giving LLMs the ability to call functions. Uses Pydantic for schema validation.

## Files

- `tool.py` — `Tool` dataclass, `create_tool()` factory, `convert_tools()` helper

## Tool

```python
@dataclass(frozen=True)
class Tool:
    name: str                           # tool name the LLM sees
    description: str                    # description the LLM sees
    schema: type[BaseModel]             # Pydantic model for argument validation
    execute: Callable[[Any], Awaitable[str]]  # async function that runs the tool
    max_calls: int | None = None        # max times this tool can be called per generation
    requires_approval: bool = False     # if True, triggers approval callback before execution
    retry: int | None = None            # number of retries on failure
```

## create_tool()

Factory function with keyword-only arguments:

```python
from neuromod import create_tool
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str
    max_results: int = 10

async def search(params: SearchParams) -> str:
    results = await fetch_results(params.query, params.max_results)
    return "\n".join(results)

search_tool = create_tool(
    name="search",
    description="Search the web for information",
    schema=SearchParams,
    execute=search,
    max_calls=5,
    requires_approval=False,
    retry=2,
)
```

## Tool Execution Flow

When the LLM returns a tool call:

1. **Lookup** — find the `Tool` by name
2. **Call limit check** — compare call count against `max_calls`
3. **Approval check** — if `requires_approval`, call `ctx.tool_approval` callback
4. **Validation** — parse arguments with `tool.schema(**arguments)` (Pydantic)
5. **Execution** — call `tool.execute(parsed_params)`
6. **Retry** — on failure, retry up to `tool.retry` times
7. **Result** — return as `ToolResultContent` (success or error)

All tool calls within a step execute in parallel via `asyncio.gather`.

## convert_tools()

Converts a list of `Tool` instances to `ToolDefinition` instances for the provider API. Strips execution-related fields, keeps only what the LLM needs (name, description, JSON schema).

```python
from neuromod import convert_tools

tool_defs = convert_tools([search_tool])
# Returns list[ToolDefinition] with JSON schemas from Pydantic models
```

## ToolDefinition vs Tool

- `Tool` — full definition including `execute`, `retry`, `max_calls`. Used internally.
- `ToolDefinition` — name + description + JSON schema only. Sent to the LLM API.

`convert_tools()` bridges the two by calling `schema.model_json_schema()` on each tool's Pydantic model.
