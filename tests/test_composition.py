import pytest

from neuromod.messages import (
    user_message,
    assistant_message,
    get_text,
    TextContent,
    ToolCallContent,
    Message,
)
from neuromod.composition import (
    ConversationContext,
    StepFunction,
    compose,
    when,
    tap,
    retry,
    RetryOptions,
    no_tools_called,
    tool_not_used_recently,
)


# ── Helper step functions for tests ────────────────


async def append_response(ctx: ConversationContext) -> ConversationContext:
    return ctx.with_updates(
        messages=[*ctx.messages, assistant_message("response")]
    )


async def uppercase_step(ctx: ConversationContext) -> ConversationContext:
    last = ctx.last_request
    if last is None:
        return ctx
    text = get_text(last).upper()
    return ctx.with_updates(
        messages=[*ctx.messages, assistant_message(text)]
    )


# ── compose ────────────────────────────────────────


async def test_compose_with_string_input():
    pipeline = compose(append_response)
    result = await pipeline("hello")
    assert len(result.messages) == 2
    assert result.messages[0].role == "user"
    assert result.messages[1].role == "assistant"


async def test_compose_with_context_input():
    ctx = ConversationContext(messages=[user_message("hello")])
    pipeline = compose(append_response)
    result = await pipeline(ctx)
    assert len(result.messages) == 2


async def test_compose_chains_steps_sequentially():
    async def add_user(ctx: ConversationContext) -> ConversationContext:
        return ctx.with_updates(messages=[*ctx.messages, user_message("followup")])

    pipeline = compose(append_response, add_user, append_response)
    result = await pipeline("start")
    assert len(result.messages) == 4
    assert result.messages[0].role == "user"
    assert result.messages[1].role == "assistant"
    assert result.messages[2].role == "user"
    assert result.messages[3].role == "assistant"


async def test_compose_empty_steps():
    pipeline = compose()
    ctx = ConversationContext(messages=[user_message("hello")])
    result = await pipeline(ctx)
    assert len(result.messages) == 1


async def test_compose_single_step():
    pipeline = compose(uppercase_step)
    result = await pipeline("hello")
    assert len(result.messages) == 2
    assert get_text(result.messages[1]) == "HELLO"


# ── when ───────────────────────────────────────────


async def test_when_condition_true():
    step = when(lambda ctx: True, append_response)
    ctx = ConversationContext(messages=[user_message("hello")])
    result = await step(ctx)
    assert len(result.messages) == 2


async def test_when_condition_false():
    step = when(lambda ctx: False, append_response)
    ctx = ConversationContext(messages=[user_message("hello")])
    result = await step(ctx)
    assert len(result.messages) == 1


async def test_when_async_step():
    async def async_step(ctx: ConversationContext) -> ConversationContext:
        return ctx.with_updates(messages=[*ctx.messages, assistant_message("async")])

    step = when(lambda ctx: True, async_step)
    ctx = ConversationContext(messages=[user_message("hello")])
    result = await step(ctx)
    assert len(result.messages) == 2


# ── tap ────────────────────────────────────────────


async def test_tap_sync_function():
    called = []

    def side_effect(ctx: ConversationContext) -> None:
        called.append(True)

    step = tap(side_effect)
    ctx = ConversationContext(messages=[user_message("hello")])
    result = await step(ctx)
    assert called == [True]
    assert result is ctx


async def test_tap_async_function():
    called = []

    async def side_effect(ctx: ConversationContext) -> None:
        called.append(True)

    step = tap(side_effect)
    ctx = ConversationContext(messages=[user_message("hello")])
    result = await step(ctx)
    assert called == [True]
    assert result is ctx


async def test_tap_returns_original_context():
    step = tap(lambda ctx: None)
    ctx = ConversationContext(messages=[user_message("hello")])
    result = await step(ctx)
    assert result is ctx


async def test_tap_side_effect_executes():
    log: list[str] = []

    def logger(ctx: ConversationContext) -> None:
        log.append(f"messages: {len(ctx.messages)}")

    step = tap(logger)
    ctx = ConversationContext(messages=[user_message("a"), user_message("b")])
    await step(ctx)
    assert log == ["messages: 2"]


# ── retry ──────────────────────────────────────────


async def test_retry_succeeds_first_try():
    step = retry(RetryOptions(times=2), append_response)
    ctx = ConversationContext(messages=[user_message("hello")])
    result = await step(ctx)
    assert len(result.messages) == 2


async def test_retry_succeeds_after_failures():
    attempt = [0]

    async def flaky(ctx: ConversationContext) -> ConversationContext:
        attempt[0] += 1
        if attempt[0] < 3:
            raise ValueError("fail")
        return ctx.with_updates(messages=[*ctx.messages, assistant_message("ok")])

    step = retry(RetryOptions(times=3), flaky)
    ctx = ConversationContext(messages=[user_message("hello")])
    result = await step(ctx)
    assert len(result.messages) == 2
    assert attempt[0] == 3


async def test_retry_all_attempts_fail():
    async def always_fail(ctx: ConversationContext) -> ConversationContext:
        raise ValueError("always fails")

    step = retry(RetryOptions(times=2), always_fail)
    ctx = ConversationContext(messages=[user_message("hello")])
    with pytest.raises(ValueError, match="always fails"):
        await step(ctx)


async def test_retry_times_zero():
    async def fail_once(ctx: ConversationContext) -> ConversationContext:
        raise ValueError("fail")

    step = retry(RetryOptions(times=0), fail_once)
    ctx = ConversationContext(messages=[user_message("hello")])
    with pytest.raises(ValueError):
        await step(ctx)


async def test_retry_uses_original_context():
    attempts: list[int] = []

    async def track_messages(ctx: ConversationContext) -> ConversationContext:
        attempts.append(len(ctx.messages))
        if len(attempts) < 3:
            raise ValueError("fail")
        return ctx

    step = retry(RetryOptions(times=3), track_messages)
    ctx = ConversationContext(messages=[user_message("hello")])
    await step(ctx)
    # Each retry gets the same original context with 1 message
    assert attempts == [1, 1, 1]


# ── no_tools_called ────────────────────────────────


def test_no_tools_called_no_response():
    ctx = ConversationContext(messages=[user_message("hello")])
    assert no_tools_called(ctx) is True


def test_no_tools_called_text_only_response():
    ctx = ConversationContext(messages=[
        user_message("hello"),
        assistant_message("hi there"),
    ])
    assert no_tools_called(ctx) is True


def test_no_tools_called_with_tool_calls():
    tc = ToolCallContent(id="1", name="foo", arguments={})
    msg = Message(role="assistant", content=[tc])
    ctx = ConversationContext(messages=[user_message("hello"), msg])
    assert no_tools_called(ctx) is False


# ── tool_not_used_recently ─────────────────────────


def test_tool_not_used_recently_not_found():
    ctx = ConversationContext(messages=[
        user_message("hello"),
        assistant_message("hi"),
    ])
    check = tool_not_used_recently("foo", 5)
    assert check(ctx) is True


def test_tool_not_used_recently_found():
    tc = ToolCallContent(id="1", name="search", arguments={})
    msg = Message(role="assistant", content=[tc])
    ctx = ConversationContext(messages=[user_message("hello"), msg])
    check = tool_not_used_recently("search", 5)
    assert check(ctx) is False


def test_tool_not_used_recently_outside_window():
    tc = ToolCallContent(id="1", name="search", arguments={})
    tool_msg = Message(role="assistant", content=[tc])
    ctx = ConversationContext(messages=[
        user_message("old"),
        tool_msg,
        user_message("q1"),
        assistant_message("a1"),
        user_message("q2"),
        assistant_message("a2"),
    ])
    # Window of 2 only sees the last 2 messages, not the tool_msg
    check = tool_not_used_recently("search", 2)
    assert check(ctx) is True
