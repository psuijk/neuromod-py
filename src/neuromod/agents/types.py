from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterable, Awaitable

from neuromod.composition.context import StopReason
from neuromod.messages.types import Message
from neuromod.providers.provider import TokenUsage
from neuromod.streaming.events import StreamEvent


@dataclass(frozen=True)
class AgentResponse:
    """The result of an Agent.generate() call."""

    text: str
    """Extracted text from the last assistant message."""

    message: Message
    """The last assistant message."""

    messages: list[Message]
    """Full conversation history."""

    finish_reason: StopReason
    """How execution ended: "stop", "max_steps", or "aborted"."""

    usage: TokenUsage
    """Accumulated token usage across all steps."""

    duration_ms: float
    """Wall-clock time in milliseconds."""

    output: Any | None = None
    """Parsed structured output if a schema was provided."""

@dataclass
class AgentStreamResult:
    events: AsyncIterable[StreamEvent]
    response: Awaitable[AgentResponse]