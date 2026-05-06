from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterable, Awaitable, Literal, Protocol, TypeAlias, runtime_checkable

from neuromod.messages.types import Message
from neuromod.models.model import Model

JsonSchema: TypeAlias = dict[str, Any]
ToolChoice = Literal["auto", "required", "none"]


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


@dataclass(frozen=True)
class TokenCount:
    tokens: int
    exact: bool


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: JsonSchema


@dataclass(frozen=True)
class ToolCallInfo:
    id: str
    name: str
    arguments: dict[str, Any]


# ── Provider stream events ─────────────────────────


@dataclass(frozen=True)
class TextDeltaEvent:
    type: Literal["text_delta"] = field(default="text_delta", init=False)
    text: str


@dataclass(frozen=True)
class ToolCallStartEvent:
    type: Literal["tool_call_start"] = field(default="tool_call_start", init=False)
    id: str
    name: str


@dataclass(frozen=True)
class ToolCallDeltaEvent:
    type: Literal["tool_call_delta"] = field(default="tool_call_delta", init=False)
    id: str
    arguments_delta: str


@dataclass(frozen=True)
class ToolCallsReadyEvent:
    type: Literal["tool_calls_ready"] = field(default="tool_calls_ready", init=False)
    calls: list[ToolCallInfo]


ProviderStreamEvent = TextDeltaEvent | ToolCallStartEvent | ToolCallDeltaEvent | ToolCallsReadyEvent


# ── Request / Response ─────────────────────────────


@dataclass(frozen=True)
class ProviderRequest:
    model: Model
    messages: list[Message]
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None
    system: str | None = None
    signal: object | None = None
    schema: JsonSchema | None = None
    temperature: float | None = None
    timeout: float | None = None


@dataclass(frozen=True)
class ProviderResponse:
    message: Message
    usage: TokenUsage


@dataclass
class ProviderStreamResult:
    events: AsyncIterable[ProviderStreamEvent]
    response: Awaitable[ProviderResponse]


# ── Provider protocol ──────────────────────────────


@runtime_checkable
class Provider(Protocol):
    async def generate(self, request: ProviderRequest) -> ProviderResponse: ...
    def stream(self, request: ProviderRequest) -> ProviderStreamResult: ...
    async def count_tokens(self, request: ProviderRequest) -> TokenCount: ...
