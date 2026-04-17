import asyncio
import dataclasses

import pytest

from neuromod.messages import ToolCallContent
from neuromod.providers import TokenUsage
from neuromod.streaming import (
    TextDeltaStreamEvent,
    ToolCallStartStreamEvent,
    ToolCallDeltaStreamEvent,
    ToolCallsReadyStreamEvent,
    ToolExecutingStreamEvent,
    ToolCompleteStreamEvent,
    ToolApprovalPendingStreamEvent,
    ToolApprovalDeniedStreamEvent,
    StepStartStreamEvent,
    StepCompleteStreamEvent,
    StepResult,
    EventType,
    Channel,
)


# ── Event creation ─────────────────────────────────


def test_text_delta_event():
    e = TextDeltaStreamEvent(text="hello", step_number=1)
    assert e.type == "text_delta"
    assert e.text == "hello"
    assert e.step_number == 1


def test_tool_call_start_event():
    e = ToolCallStartStreamEvent(id="1", name="foo", step_number=0)
    assert e.type == "tool_call_start"
    assert e.id == "1"
    assert e.name == "foo"


def test_tool_call_delta_event():
    e = ToolCallDeltaStreamEvent(id="1", arguments_delta='{"x":', step_number=0)
    assert e.type == "tool_call_delta"
    assert e.arguments_delta == '{"x":'


def test_tool_calls_ready_event():
    tc = ToolCallContent(id="1", name="foo", arguments={"x": 1})
    e = ToolCallsReadyStreamEvent(calls=[tc], step_number=0)
    assert e.type == "tool_calls_ready"
    assert len(e.calls) == 1


def test_tool_executing_event():
    e = ToolExecutingStreamEvent(name="foo", id="1", step_number=0)
    assert e.type == "tool_executing"
    assert e.name == "foo"


def test_tool_complete_event():
    e = ToolCompleteStreamEvent(name="foo", id="1", result="ok", is_error=False, duration_ms=100, step_number=0)
    assert e.type == "tool_complete"
    assert e.result == "ok"
    assert e.duration_ms == 100


def test_tool_approval_pending_event():
    e = ToolApprovalPendingStreamEvent(name="foo", id="1", step_number=0)
    assert e.type == "tool_approval_pending"


def test_tool_approval_denied_event():
    e = ToolApprovalDeniedStreamEvent(name="foo", id="1", step_number=0)
    assert e.type == "tool_approval_denied"


def test_step_start_event():
    e = StepStartStreamEvent(step_number=1)
    assert e.type == "step_start"
    assert e.step_number == 1


def test_step_complete_event():
    usage = TokenUsage(input_tokens=10, output_tokens=5)
    sr = StepResult(step_number=1, tool_calls=[], usage=usage, duration_ms=500)
    e = StepCompleteStreamEvent(step_number=1, step=sr)
    assert e.type == "step_complete"
    assert e.step.duration_ms == 500


def test_step_result_creation():
    usage = TokenUsage(input_tokens=10, output_tokens=5)
    tc = ToolCallContent(id="1", name="foo", arguments={})
    sr = StepResult(step_number=0, tool_calls=[tc], usage=usage, duration_ms=200)
    assert sr.step_number == 0
    assert len(sr.tool_calls) == 1
    assert sr.duration_ms == 200


def test_events_frozen():
    e = TextDeltaStreamEvent(text="hi", step_number=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.text = "bye"  # type: ignore[misc]


# ── EventType constants ────────────────────────────


def test_event_type_values():
    assert EventType.TEXT_DELTA == "text_delta"
    assert EventType.TOOL_CALL_START == "tool_call_start"
    assert EventType.TOOL_CALL_DELTA == "tool_call_delta"
    assert EventType.TOOL_CALLS_READY == "tool_calls_ready"
    assert EventType.TOOL_EXECUTING == "tool_executing"
    assert EventType.TOOL_COMPLETE == "tool_complete"
    assert EventType.TOOL_APPROVAL_PENDING == "tool_approval_pending"
    assert EventType.TOOL_APPROVAL_DENIED == "tool_approval_denied"
    assert EventType.STEP_START == "step_start"
    assert EventType.STEP_COMPLETE == "step_complete"


# ── Channel ────────────────────────────────────────


async def test_channel_push_and_iterate():
    ch: Channel[int] = Channel()
    ch.push(1)
    ch.close()
    result = await anext(ch)
    assert result == 1


async def test_channel_close_ends_iteration():
    ch: Channel[int] = Channel()
    ch.close()
    with pytest.raises(StopAsyncIteration):
        await anext(ch)


async def test_channel_multiple_values():
    ch: Channel[int] = Channel()
    ch.push(1)
    ch.push(2)
    ch.push(3)
    ch.close()
    results = []
    async for val in ch:
        results.append(val)
    assert results == [1, 2, 3]


async def test_channel_push_before_iterate():
    ch: Channel[str] = Channel()
    ch.push("a")
    ch.push("b")
    ch.close()
    results = []
    async for val in ch:
        results.append(val)
    assert results == ["a", "b"]


async def test_channel_push_after_close():
    ch: Channel[int] = Channel()
    ch.close()
    ch.push(1)
    # Queue is checked before done flag, so queued value is still yielded
    result = await anext(ch)
    assert result == 1


async def test_channel_empty_close():
    ch: Channel[int] = Channel()
    ch.close()
    results = []
    async for val in ch:
        results.append(val)
    assert results == []


async def test_channel_async_for_loop():
    ch: Channel[str] = Channel()

    async def producer() -> None:
        await asyncio.sleep(0)
        ch.push("hello")
        ch.push("world")
        ch.close()

    task = asyncio.create_task(producer())
    results = []
    async for val in ch:
        results.append(val)
    await task
    assert results == ["hello", "world"]
