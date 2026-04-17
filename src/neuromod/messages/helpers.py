from __future__ import annotations

from typing import Any

from .types import (
    Content,
    MediaContent,
    Message,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)


# ── Content part builders ──────────────────────────


def text(value: str) -> TextContent:
    return TextContent(text=value)


def image(data: str, mime_type: str) -> MediaContent:
    return MediaContent(data=data, mime_type=mime_type)


def audio(data: str, mime_type: str) -> MediaContent:
    return MediaContent(data=data, mime_type=mime_type)


def document(data: str, mime_type: str, filename: str | None = None) -> MediaContent:
    return MediaContent(data=data, mime_type=mime_type, filename=filename)


def media(data: str, mime_type: str, filename: str | None = None) -> MediaContent:
    return MediaContent(data=data, mime_type=mime_type, filename=filename)


def tool_call(id: str, name: str, arguments: dict[str, Any]) -> ToolCallContent:
    return ToolCallContent(id=id, name=name, arguments=arguments)


def tool_result(
    call_id: str,
    result: str,
    *,
    name: str | None = None,
    is_error: bool = False,
) -> ToolResultContent:
    return ToolResultContent(call_id=call_id, result=result, name=name, is_error=is_error)


# ── Message builders ───────────────────────────────


def user_message(input: str | list[Content]) -> Message:
    if isinstance(input, str):
        return Message(role="user", content=[TextContent(text=input)])
    return Message(role="user", content=input)


def assistant_message(input: str | list[Content]) -> Message:
    if isinstance(input, str):
        return Message(role="assistant", content=[TextContent(text=input)])
    return Message(role="assistant", content=input)


def system_message(input: str) -> Message:
    return Message(role="system", content=[TextContent(text=input)])


# ── Extractors ─────────────────────────────────────


def get_text(message: Message) -> str:
    return "".join(part.text for part in message.content if isinstance(part, TextContent))


def get_media(message: Message) -> list[MediaContent]:
    return [part for part in message.content if isinstance(part, MediaContent)]


def get_tool_calls(message: Message) -> list[ToolCallContent]:
    return [part for part in message.content if isinstance(part, ToolCallContent)]


def get_tool_results(message: Message) -> list[ToolResultContent]:
    return [part for part in message.content if isinstance(part, ToolResultContent)]


def has_tool_calls(message: Message) -> bool:
    return any(isinstance(part, ToolCallContent) for part in message.content)
