# messages

Content types, message construction, and extraction utilities. Messages are the data that flows through conversations.

## Files

- `types.py` — `Content` union type, `Message` dataclass, `Role` literal
- `helpers.py` — Builder functions and extractors

## Content Types

All content is a discriminated union. `Message.content` is always `list[Content]`, never a raw string.

```python
Content = TextContent | MediaContent | ToolCallContent | ToolResultContent
```

| Type | Fields | Description |
|------|--------|-------------|
| `TextContent` | `text: str` | Plain text |
| `MediaContent` | `data: str, mime_type: str, filename?: str` | Base64-encoded media (images, PDFs) |
| `ToolCallContent` | `id: str, name: str, arguments: dict` | LLM requesting a tool call |
| `ToolResultContent` | `call_id: str, result: str, name?: str, is_error: bool` | Result of a tool execution |

Each has a `type` field discriminator: `"text"`, `"media"`, `"tool_call"`, `"tool_result"`.

## Message

```python
@dataclass(frozen=True)
class Message:
    role: Role       # "system" | "user" | "assistant"
    content: list[Content]
```

## Builders

Convenience functions that create `Message` and `Content` instances.

```python
from neuromod import user_message, assistant_message, system_message, text, tool_call, tool_result

# Messages — accept str or list[Content]
user_message("hello")
user_message([text("hello"), image(data, "image/png")])
assistant_message("hi there")
system_message("You are helpful.")

# Content parts
text("hello")
image(data, "image/png")
audio(data, "audio/wav")
document(data, "application/pdf", filename="doc.pdf")
media(data, mime_type)
tool_call(id="tc_1", name="search", arguments={"query": "test"})
tool_result(call_id="tc_1", result="found it", name="search", is_error=False)
```

## Extractors

Pull specific content types from messages.

```python
from neuromod import get_text, get_media, get_tool_calls, get_tool_results, has_tool_calls

get_text(message)         # concatenated text from all TextContent parts
get_media(message)        # list[MediaContent]
get_tool_calls(message)   # list[ToolCallContent]
get_tool_results(message) # list[ToolResultContent]
has_tool_calls(message)   # bool
```
