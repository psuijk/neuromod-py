from __future__ import annotations

import pytest
from pydantic import BaseModel

from neuromod.composition.context import ConversationContext, StepFunction
from neuromod.composition.scope import Inherit, scope
from neuromod.messages.helpers import user_message, assistant_message
from neuromod.messages.types import Message
from neuromod.tools.tool import Tool


# --- Helpers ---

def make_step(text: str) -> StepFunction:
    """Create a step that appends an assistant message."""
    async def step(ctx: ConversationContext) -> ConversationContext:
        return ctx.with_updates(
            messages=[*ctx.messages, assistant_message(text)],
        )
    return step


class DummySchema(BaseModel):
    query: str


async def dummy_execute(params: DummySchema) -> str:
    return "ok"


def make_tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"A {name} tool",
        schema=DummySchema,
        execute=dummy_execute,
    )


def make_ctx(
    messages: list[Message] | None = None,
    tools: list[Tool] | None = None,
) -> ConversationContext:
    return ConversationContext(
        messages=messages or [user_message("hello")],
        tools=tools,
    )


# --- Tests ---

class TestInheritFlag:
    def test_nothing_is_zero(self):
        assert Inherit.NOTHING.value == 0

    def test_all_includes_conversation(self):
        assert Inherit.CONVERSATION in Inherit.ALL

    def test_all_includes_tools(self):
        assert Inherit.TOOLS in Inherit.ALL

    def test_nothing_excludes_both(self):
        assert Inherit.CONVERSATION not in Inherit.NOTHING
        assert Inherit.TOOLS not in Inherit.NOTHING

    def test_combine_flags(self):
        combined = Inherit.CONVERSATION | Inherit.TOOLS
        assert combined == Inherit.ALL


class TestScopeInheritance:
    async def test_inherit_all_passes_messages_and_tools(self):
        tool = make_tool("search")
        ctx = make_ctx(tools=[tool])
        step = make_step("response")

        result = await scope(step, inherit=Inherit.ALL)(ctx)

        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"

    async def test_inherit_nothing_hides_messages(self):
        ctx = make_ctx(messages=[user_message("secret")])
        step = make_step("response")

        scoped = scope(step, inherit=Inherit.NOTHING)
        result = await scoped(ctx)

        # Child didn't see "secret", only added "response"
        # Merged back: child messages replace parent messages
        assert len(result.messages) == 1

    async def test_inherit_nothing_hides_tools(self):
        parent_tool = make_tool("parent_tool")
        ctx = make_ctx(tools=[parent_tool])

        async def check_tools(ctx: ConversationContext) -> ConversationContext:
            assert ctx.tools is None
            return ctx

        await scope(check_tools, inherit=Inherit.NOTHING)(ctx)

    async def test_inherit_conversation_only(self):
        parent_tool = make_tool("parent_tool")
        ctx = make_ctx(
            messages=[user_message("hello")],
            tools=[parent_tool],
        )

        async def check_ctx(ctx: ConversationContext) -> ConversationContext:
            assert len(ctx.messages) == 1
            assert ctx.tools is None
            return ctx

        await scope(check_ctx, inherit=Inherit.CONVERSATION)(ctx)

    async def test_inherit_tools_only(self):
        parent_tool = make_tool("parent_tool")
        ctx = make_ctx(
            messages=[user_message("hello")],
            tools=[parent_tool],
        )

        async def check_ctx(ctx: ConversationContext) -> ConversationContext:
            assert len(ctx.messages) == 0
            assert ctx.tools is not None
            assert len(ctx.tools) == 1
            assert ctx.tools[0].name == "parent_tool"
            return ctx

        await scope(check_ctx, inherit=Inherit.TOOLS)(ctx)


class TestScopeTools:
    async def test_scope_tools_added(self):
        scope_tool = make_tool("scope_tool")
        ctx = make_ctx()

        async def check_tools(ctx: ConversationContext) -> ConversationContext:
            assert ctx.tools is not None
            assert len(ctx.tools) == 1
            assert ctx.tools[0].name == "scope_tool"
            return ctx

        await scope(check_tools, inherit=Inherit.NOTHING, tools=[scope_tool])(ctx)

    async def test_scope_tools_merged_with_parent(self):
        parent_tool = make_tool("parent_tool")
        scope_tool = make_tool("scope_tool")
        ctx = make_ctx(tools=[parent_tool])

        async def check_tools(ctx: ConversationContext) -> ConversationContext:
            assert ctx.tools is not None
            assert len(ctx.tools) == 2
            names = [t.name for t in ctx.tools]
            assert "parent_tool" in names
            assert "scope_tool" in names
            return ctx

        await scope(check_tools, inherit=Inherit.ALL, tools=[scope_tool])(ctx)

    async def test_scope_tools_without_inherit(self):
        parent_tool = make_tool("parent_tool")
        scope_tool = make_tool("scope_tool")
        ctx = make_ctx(tools=[parent_tool])

        async def check_tools(ctx: ConversationContext) -> ConversationContext:
            assert ctx.tools is not None
            assert len(ctx.tools) == 1
            assert ctx.tools[0].name == "scope_tool"
            return ctx

        await scope(check_tools, inherit=Inherit.NOTHING, tools=[scope_tool])(ctx)


class TestScopeUntilLoop:
    async def test_runs_at_least_once(self):
        call_count = 0

        async def counting_step(ctx: ConversationContext) -> ConversationContext:
            nonlocal call_count
            call_count += 1
            return ctx

        ctx = make_ctx()
        await scope(counting_step, until=lambda ctx: True)(ctx)
        assert call_count == 1

    async def test_loops_until_condition_met(self):
        call_count = 0

        async def counting_step(ctx: ConversationContext) -> ConversationContext:
            nonlocal call_count
            call_count += 1
            return ctx.with_updates(
                messages=[*ctx.messages, assistant_message(f"msg-{call_count}")],
            )

        ctx = make_ctx()
        # Stop after 3 iterations (initial message + 3 assistant messages = 4)
        await scope(counting_step, until=lambda ctx: len(ctx.messages) >= 4)(ctx)
        assert call_count == 3

    async def test_no_until_runs_once(self):
        call_count = 0

        async def counting_step(ctx: ConversationContext) -> ConversationContext:
            nonlocal call_count
            call_count += 1
            return ctx

        ctx = make_ctx()
        await scope(counting_step)(ctx)
        assert call_count == 1


class TestScopeSilent:
    async def test_silent_returns_parent_unchanged(self):
        ctx = make_ctx(messages=[user_message("original")])
        step = make_step("new message")

        result = await scope(step, silent=True)(ctx)

        assert len(result.messages) == 1
        assert result.messages[0] is ctx.messages[0]

    async def test_non_silent_merges_back(self):
        ctx = make_ctx(messages=[user_message("original")])
        step = make_step("new message")

        result = await scope(step, silent=False)(ctx)

        assert len(result.messages) == 2


class TestScopeMultipleSteps:
    async def test_runs_steps_in_order(self):
        step_a = make_step("first")
        step_b = make_step("second")
        ctx = make_ctx()

        result = await scope(step_a, step_b)(ctx)

        assert len(result.messages) == 3  # user + first + second


class TestScopeCallbacks:
    async def test_inherits_parent_on_event(self):
        events = []
        ctx = ConversationContext(
            messages=[user_message("hello")],
            on_event=lambda e: events.append(e),
        )

        async def check_on_event(ctx: ConversationContext) -> ConversationContext:
            assert ctx.on_event is not None
            return ctx

        await scope(check_on_event)(ctx)

    async def test_scope_on_event_overrides_parent(self):
        parent_events = []
        scope_events = []
        ctx = ConversationContext(
            messages=[user_message("hello")],
            on_event=lambda e: parent_events.append(e),
        )

        async def check_on_event(ctx: ConversationContext) -> ConversationContext:
            assert ctx.on_event is not None
            ctx.on_event("test_event")
            return ctx

        await scope(check_on_event, on_event=lambda e: scope_events.append(e))(ctx)

        assert len(scope_events) == 1
        assert len(parent_events) == 0

    async def test_inherits_parent_signal(self):
        import asyncio
        signal = asyncio.Event()
        ctx = ConversationContext(
            messages=[user_message("hello")],
            signal=signal,
        )

        async def check_signal(ctx: ConversationContext) -> ConversationContext:
            assert ctx.signal is signal
            return ctx

        await scope(check_signal)(ctx)
