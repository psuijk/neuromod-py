from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Awaitable, Callable

from neuromod.messages.types import ToolCallContent
from neuromod.composition.context import ConversationContext, StepFunction


@dataclass(frozen=True)
class RetryOptions:
    times: int


def when(
    condition: Callable[[ConversationContext], bool],
    step: StepFunction,
) -> StepFunction:
    async def run(ctx: ConversationContext) -> ConversationContext:
        if condition(ctx):
            return await step(ctx)
        return ctx

    return run


def tap(
    fn: Callable[[ConversationContext], None] | Callable[[ConversationContext], Awaitable[None]],
) -> StepFunction:
    async def run(ctx: ConversationContext) -> ConversationContext:
        result = fn(ctx)
        if inspect.isawaitable(result):
            await result
        return ctx

    return run


def retry(
    options: RetryOptions,
    step: StepFunction,
) -> StepFunction:
    async def run(ctx: ConversationContext) -> ConversationContext:
        for attempt in range(options.times + 1):
            try:
                return await step(ctx)
            except Exception:
                if attempt == options.times:
                    raise
        raise RuntimeError("unreachable")

    return run


def no_tools_called(ctx: ConversationContext) -> bool:
    response = ctx.last_response
    if response is None:
        return True
    return len(response.tool_calls) == 0


def tool_not_used_recently(tool_name: str, message_count: int) -> Callable[[ConversationContext], bool]:
    def check(ctx: ConversationContext) -> bool:
        recent = ctx.messages[-message_count:]
        for msg in recent:
            for part in msg.content:
                if isinstance(part, ToolCallContent) and part.name == tool_name:
                    return False
        return True

    return check
