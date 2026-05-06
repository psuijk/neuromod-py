# messages

Content types, message construction, and extraction utilities. Messages are the data that flows through conversations.

## Files

- `types.py` — `Content` union type, `Message` dataclass with extractor properties, `Role` literal
- `helpers.py` — Builder functions for creating messages and content parts

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

    # Properties for extracting content by type
    message.text           # str — concatenated text from all TextContent parts
    message.media          # list[MediaContent]
    message.tool_calls     # list[ToolCallContent]
    message.tool_results   # list[ToolResultContent]
    message.has_tool_calls # bool
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

Content extraction is available directly as properties on `Message`:

```python
message.text           # concatenated text from all TextContent parts
message.media          # list[MediaContent]
message.tool_calls     # list[ToolCallContent]
message.tool_results   # list[ToolResultContent]
message.has_tool_calls # bool
```
