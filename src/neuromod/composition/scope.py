from enum import Flag, auto
from typing import Awaitable, Callable
from neuromod.composition.context import ConversationContext, StepFunction, ToolApprovalRequest
from neuromod.streaming.events import StreamEvent
from neuromod.tools.tool import Tool

class Inherit(Flag):
    NOTHING = 0
    CONVERSATION = auto()
    TOOLS = auto()
    ALL = CONVERSATION | TOOLS

def scope(
    *,
    inherit: Inherit = Inherit.ALL,
    tools: list[Tool] | None = None,
    tool_call_limits: dict[str, int] | None = None,
    tool_approval: Callable[[ToolApprovalRequest], Awaitable[bool]] | None = None,
    on_event: Callable[[StreamEvent], None] | None = None,
    until: Callable[[ConversationContext], bool] | None = None,
    silent: bool = False,
) -> Callable[..., StepFunction]:

    def with_steps(*steps: StepFunction) -> StepFunction:

        async def run(parent_ctx: ConversationContext) -> ConversationContext:
            # Build child context based on inherit flags
            child_messages = parent_ctx.messages if Inherit.CONVERSATION in inherit else []
            if Inherit.TOOLS in inherit:
                child_tools = (parent_ctx.tools or []) + (tools or [])
            else:
                child_tools = tools or None

            child_ctx = ConversationContext(
                messages=child_messages,
                tools=child_tools if child_tools else None,
                tool_call_limits=tool_call_limits or parent_ctx.tool_call_limits,
                tool_approval=tool_approval or parent_ctx.tool_approval,
                on_event=on_event or parent_ctx.on_event,
                signal=parent_ctx.signal,
            )

            while True:
                for step in steps:
                    child_ctx = await step(child_ctx)
                if until is None or until(child_ctx):
                    break

            if silent:
                return parent_ctx
            return parent_ctx.with_updates(
                messages=child_ctx.messages,
                usage=child_ctx.usage,
                stop_reason=child_ctx.stop_reason,
            )

        return run

    return with_steps
