from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import AsyncIterator, Literal, TypeVar

from neuromod.messages.types import ToolCallContent
from neuromod.providers.provider import TokenUsage

T = TypeVar("T")


# ── Text streaming ──────────────────────────────────


@dataclass(frozen=True)
class TextDeltaStreamEvent:
    type: Literal["text_delta"] = field(default="text_delta", init=False)
    text: str
    step_number: int


# ── Tool lifecycle ──────────────────────────────────


@dataclass(frozen=True)
class ToolCallStartStreamEvent:
    type: Literal["tool_call_start"] = field(default="tool_call_start", init=False)
    id: str
    name: str
    step_number: int


@dataclass(frozen=True)
class ToolCallDeltaStreamEvent:
    type: Literal["tool_call_delta"] = field(default="tool_call_delta", init=False)
    id: str
    arguments_delta: str
    step_number: int


@dataclass(frozen=True)
class ToolCallsReadyStreamEvent:
    type: Literal["tool_calls_ready"] = field(default="tool_calls_ready", init=False)
    calls: list[ToolCallContent]
    step_number: int


@dataclass(frozen=True)
class ToolExecutingStreamEvent:
    type: Literal["tool_executing"] = field(default="tool_executing", init=False)
    name: str
    id: str
    step_number: int


@dataclass(frozen=True)
class ToolCompleteStreamEvent:
    type: Literal["tool_complete"] = field(default="tool_complete", init=False)
    name: str
    id: str
    result: str
    is_error: bool
    duration_ms: int
    step_number: int


# ── Approval ────────────────────────────────────────


@dataclass(frozen=True)
class ToolApprovalPendingStreamEvent:
    type: Literal["tool_approval_pending"] = field(default="tool_approval_pending", init=False)
    name: str
    id: str
    step_number: int


@dataclass(frozen=True)
class ToolApprovalDeniedStreamEvent:
    type: Literal["tool_approval_denied"] = field(default="tool_approval_denied", init=False)
    name: str
    id: str
    step_number: int


# ── Step lifecycle ──────────────────────────────────


@dataclass(frozen=True)
class StepResult:
    step_number: int
    tool_calls: list[ToolCallContent]
    usage: TokenUsage
    duration_ms: int


@dataclass(frozen=True)
class StepStartStreamEvent:
    type: Literal["step_start"] = field(default="step_start", init=False)
    step_number: int


@dataclass(frozen=True)
class StepCompleteStreamEvent:
    type: Literal["step_complete"] = field(default="step_complete", init=False)
    step_number: int
    step: StepResult


StreamEvent = (
    TextDeltaStreamEvent
    | ToolCallStartStreamEvent
    | ToolCallDeltaStreamEvent
    | ToolCallsReadyStreamEvent
    | ToolExecutingStreamEvent
    | ToolCompleteStreamEvent
    | ToolApprovalPendingStreamEvent
    | ToolApprovalDeniedStreamEvent
    | StepStartStreamEvent
    | StepCompleteStreamEvent
)


class EventType(StrEnum):
    TEXT_DELTA = "text_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALLS_READY = "tool_calls_ready"
    TOOL_EXECUTING = "tool_executing"
    TOOL_COMPLETE = "tool_complete"
    TOOL_APPROVAL_PENDING = "tool_approval_pending"
    TOOL_APPROVAL_DENIED = "tool_approval_denied"
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"


class Channel(AsyncIterator[T]):
    def __init__(self) -> None:
        self._queue: list[T] = []
        self._event: asyncio.Event = asyncio.Event()
        self._done: bool = False

    def push(self, value: T) -> None:
        self._queue.append(value)
        self._event.set()

    def close(self) -> None:
        self._done = True
        self._event.set()

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        while True:
            if self._queue:
                return self._queue.pop(0)
            if self._done:
                raise StopAsyncIteration
            self._event.clear()
            await self._event.wait()
