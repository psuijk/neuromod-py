from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class TextContent:
    type: Literal["text"] = field(default="text", init=False)
    text: str


@dataclass(frozen=True)
class MediaContent:
    type: Literal["media"] = field(default="media", init=False)
    data: str
    mime_type: str
    filename: str | None = None


@dataclass(frozen=True)
class ToolCallContent:
    type: Literal["tool_call"] = field(default="tool_call", init=False)
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResultContent:
    type: Literal["tool_result"] = field(default="tool_result", init=False)
    call_id: str
    result: str
    name: str | None = None
    is_error: bool = False


Content = TextContent | MediaContent | ToolCallContent | ToolResultContent


@dataclass(frozen=True)
class Message:
    role: Role
    content: list[Content]
