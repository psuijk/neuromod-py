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

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.content if isinstance(part, TextContent))

    @property
    def media(self) -> list[MediaContent]:
        return [part for part in self.content if isinstance(part, MediaContent)]

    @property
    def tool_calls(self) -> list[ToolCallContent]:
        return [part for part in self.content if isinstance(part, ToolCallContent)]

    @property
    def tool_results(self) -> list[ToolResultContent]:
        return [part for part in self.content if isinstance(part, ToolResultContent)]

    @property
    def has_tool_calls(self) -> bool:
        return any(isinstance(part, ToolCallContent) for part in self.content)
