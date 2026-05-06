import pytest

from neuromod.messages import (
    Message,
    TextContent,
    ToolResultContent,
    user_message,
    assistant_message,
    system_message,
)
from neuromod.composition import ConversationContext
from neuromod.tools import create_tool
from pydantic import BaseModel


class DummyParams(BaseModel):
    x: str


async def dummy_exec(params: DummyParams) -> str:
    return "ok"


dummy_tool = create_tool(name="dummy", description="d", schema=DummyParams, execute=dummy_exec)


def test_context_default_construction():
    ctx = ConversationContext()
    assert ctx.messages == []
    assert ctx.tools is None
    assert ctx.tool_call_limits is None
    assert ctx.tool_approval is None
    assert ctx.on_event is None
    assert ctx.usage is None
    assert ctx.stop_reason is None
    assert ctx.signal is None


def test_context_with_messages():
    msgs = [user_message("hello"), assistant_message("hi")]
    ctx = ConversationContext(messages=msgs)
    assert len(ctx.messages) == 2


def test_context_with_tools():
    ctx = ConversationContext(tools=[dummy_tool])
    assert ctx.tools is not None
    assert len(ctx.tools) == 1
    assert ctx.tools[0].name == "dummy"


def test_last_request_returns_last_user_message():
    ctx = ConversationContext(messages=[
        user_message("first"),
        assistant_message("response"),
        user_message("second"),
    ])
    assert ctx.last_request is not None
    assert ctx.last_request.role == "user"
    assert ctx.last_request.text == "second"


def test_last_request_skips_assistant_messages():
    ctx = ConversationContext(messages=[
        user_message("question"),
        assistant_message("answer"),
    ])
    assert ctx.last_request is not None
    assert ctx.last_request.role == "user"


def test_last_request_returns_none_when_no_user_messages():
    ctx = ConversationContext(messages=[
        system_message("system prompt"),
        assistant_message("hello"),
    ])
    assert ctx.last_request is None


def test_last_request_includes_tool_result_messages():
    tr = ToolResultContent(call_id="1", result="ok")
    tool_result_msg = Message(role="user", content=[tr])
    ctx = ConversationContext(messages=[
        user_message("start"),
        assistant_message("calling tool"),
        tool_result_msg,
    ])
    assert ctx.last_request is not None
    assert ctx.last_request is tool_result_msg


def test_last_response_returns_last_assistant_message():
    ctx = ConversationContext(messages=[
        user_message("q"),
        assistant_message("a1"),
        user_message("q2"),
        assistant_message("a2"),
    ])
    assert ctx.last_response is not None
    assert ctx.last_response.text == "a2"


def test_last_response_returns_none_when_no_assistant_messages():
    ctx = ConversationContext(messages=[user_message("hello")])
    assert ctx.last_response is None


def test_with_updates_creates_new_instance():
    ctx = ConversationContext()
    new_ctx = ctx.with_updates(stop_reason="stop")
    assert new_ctx is not ctx


def test_with_updates_preserves_unchanged_fields():
    ctx = ConversationContext(messages=[user_message("hi")], stop_reason="stop")
    new_ctx = ctx.with_updates(stop_reason="max_steps")
    assert len(new_ctx.messages) == 1
    assert new_ctx.tools is None


def test_with_updates_overrides_specified_fields():
    ctx = ConversationContext(stop_reason="stop")
    new_ctx = ctx.with_updates(stop_reason="max_steps")
    assert new_ctx.stop_reason == "max_steps"


def test_with_updates_does_not_mutate_original():
    msgs = [user_message("hi")]
    ctx = ConversationContext(messages=msgs, stop_reason="stop")
    new_msgs = [user_message("hi"), assistant_message("hello")]
    new_ctx = ctx.with_updates(messages=new_msgs, stop_reason="max_steps")
    assert ctx.stop_reason == "stop"
    assert len(ctx.messages) == 1
    assert new_ctx.stop_reason == "max_steps"
    assert len(new_ctx.messages) == 2
