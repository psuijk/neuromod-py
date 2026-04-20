from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Literal

from neuromod.messages.types import Message
from neuromod.providers.provider import TokenUsage
from neuromod.streaming.events import StreamEvent
from neuromod.tools.tool import Tool


class _Sentinel(Enum):
    UNSET = "UNSET"


_UNSET: Any = _Sentinel.UNSET


@dataclass(frozen=True)
class ToolApprovalRequest:
    id: str
    name: str
    arguments: dict[str, Any]

StopReason = Literal["stop", "max_steps", "aborted"]

class ConversationContext:
    def __init__(
        self,
        *,
        messages: list[Message] | None = None,
        tools: list[Tool] | None = None,
        tool_call_limits: dict[str, int] | None = None,
        tool_approval: Callable[[ToolApprovalRequest], Awaitable[bool]] | None = None,
        on_event: Callable[[StreamEvent], None] | None = None,
        usage: TokenUsage | None = None,
        stop_reason: StopReason | None = None,
        signal: object | None = None,
    ) -> None:
        self.messages: list[Message] = messages if messages is not None else []
        self.tools: list[Tool] | None = tools
        self.tool_call_limits: dict[str, int] | None = tool_call_limits
        self.tool_approval: Callable[[ToolApprovalRequest], Awaitable[bool]] | None = tool_approval
        self.on_event: Callable[[StreamEvent], None] | None = on_event
        self.usage: TokenUsage | None = usage
        self.stop_reason: StopReason | None = stop_reason
        self.signal: object | None = signal

    @property
    def last_request(self) -> Message | None:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None

    @property
    def last_response(self) -> Message | None:
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg
        return None

    def with_updates(
        self,
        *,
        messages: list[Message] = _UNSET,
        tools: list[Tool] | None = _UNSET,
        tool_call_limits: dict[str, int] | None = _UNSET,
        tool_approval: Callable[[ToolApprovalRequest], Awaitable[bool]] | None = _UNSET,
        on_event: Callable[[StreamEvent], None] | None = _UNSET,
        usage: TokenUsage | None = _UNSET,
        stop_reason: StopReason | None = _UNSET,
        signal: object | None = _UNSET,
    ) -> ConversationContext:
        return ConversationContext(
            messages=self.messages if messages is _UNSET else messages,
            tools=self.tools if tools is _UNSET else tools,
            tool_call_limits=self.tool_call_limits if tool_call_limits is _UNSET else tool_call_limits,
            tool_approval=self.tool_approval if tool_approval is _UNSET else tool_approval,
            on_event=self.on_event if on_event is _UNSET else on_event,
            usage=self.usage if usage is _UNSET else usage,
            stop_reason=self.stop_reason if stop_reason is _UNSET else stop_reason,
            signal=self.signal if signal is _UNSET else signal,
        )


StepFunction = Callable[[ConversationContext], Awaitable[ConversationContext]]
