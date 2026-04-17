from __future__ import annotations

from typing import Awaitable, Callable

from neuromod.messages.helpers import user_message
from neuromod.composition.context import ConversationContext, StepFunction


def compose(*steps: StepFunction) -> Callable[[str | ConversationContext], Awaitable[ConversationContext]]:
    async def run(input: str | ConversationContext) -> ConversationContext:
        if isinstance(input, str):
            ctx = ConversationContext(messages=[user_message(input)])
        else:
            ctx = input

        for step in steps:
            ctx = await step(ctx)

        return ctx

    return run
