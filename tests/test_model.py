from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from neuromod.composition.context import ConversationContext, ToolApprovalRequest
from neuromod.composition.model import model, execute_tools
from neuromod.tools.tool import convert_tools
from neuromod.config import configure, _config, _factory
from neuromod.messages.helpers import user_message, assistant_message, tool_call, tool_result
from neuromod.messages.types import (
    Message,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)
from neuromod.models.anthropic import Claude
from neuromod.providers.provider import (
    Provider,
    ProviderRequest,
    ProviderResponse,
    ProviderStreamResult,
    TokenCount,
    TokenUsage,
)
from neuromod.tools.tool import Tool


# ── Helpers ───────────────────────────────────────


class SearchParams(BaseModel):
    query: str


async def search_execute(params: SearchParams) -> str:
    return f"Results for: {params.query}"


async def failing_execute(params: SearchParams) -> str:
    raise ValueError("Tool failed")


def make_tool(
    name: str = "search",
    max_calls: int | None = None,
    requires_approval: bool = False,
    retry: int | None = None,
    execute: Any = None,
) -> Tool:
    return Tool(
        name=name,
        description=f"A {name} tool",
        schema=SearchParams,
        execute=execute or search_execute,
        max_calls=max_calls,
        requires_approval=requires_approval,
        retry=retry,
    )


def text_response(text: str, usage: TokenUsage | None = None) -> ProviderResponse:
    return ProviderResponse(
        message=assistant_message(text),
        usage=usage or TokenUsage(input_tokens=10, output_tokens=5),
    )


def tool_call_response(
    calls: list[dict[str, Any]],
    usage: TokenUsage | None = None,
) -> ProviderResponse:
    content = [
        ToolCallContent(id=c["id"], name=c["name"], arguments=c.get("arguments", {}))
        for c in calls
    ]
    return ProviderResponse(
        message=Message(role="assistant", content=content),
        usage=usage or TokenUsage(input_tokens=10, output_tokens=5),
    )


class MockProvider:
    """A mock provider that returns responses from a queue."""

    def __init__(self, responses: list[ProviderResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.requests: list[ProviderRequest] = []

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp

    def stream(self, request: ProviderRequest) -> ProviderStreamResult:
        raise NotImplementedError

    async def count_tokens(self, request: ProviderRequest) -> TokenCount:
        return TokenCount(tokens=100, exact=True)


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config and factory before each test."""
    _config.set(None)
    _factory.set(None)
    configure(api_keys={"anthropic": "test-key"})
    yield
    _config.set(None)
    _factory.set(None)


# ── convert_tools ────────────────────────────────


class TestConvertTools:
    def test_none_returns_none(self):
        assert convert_tools(None) is None

    def test_empty_list_returns_none(self):
        assert convert_tools([]) is None

    def test_converts_tool(self):
        tool = make_tool()
        result = convert_tools([tool])
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "search"
        assert result[0].description == "A search tool"
        assert "properties" in result[0].parameters


# ── execute_tools ─────────────────────────────────


class TestExecuteTools:
    async def test_executes_tool_successfully(self):
        tool = make_tool()
        calls = [ToolCallContent(id="tc_1", name="search", arguments={"query": "test"})]
        ctx = ConversationContext()
        results = await execute_tools(calls, [tool], {}, ctx, 1)
        assert len(results) == 1
        assert results[0].result == "Results for: test"
        assert results[0].is_error is False

    async def test_unknown_tool_returns_error(self):
        calls = [ToolCallContent(id="tc_1", name="unknown", arguments={})]
        ctx = ConversationContext()
        results = await execute_tools(calls, [], {}, ctx, 1)
        assert results[0].is_error is True
        assert "Unknown tool" in results[0].result

    async def test_max_calls_enforced(self):
        tool = make_tool(max_calls=1)
        calls = [ToolCallContent(id="tc_1", name="search", arguments={"query": "a"})]
        ctx = ConversationContext()
        call_counts: dict[str, int] = {}

        # First call succeeds
        results = await execute_tools(calls, [tool], call_counts, ctx, 1)
        assert results[0].is_error is False

        # Second call exceeds limit
        calls2 = [ToolCallContent(id="tc_2", name="search", arguments={"query": "b"})]
        results2 = await execute_tools(calls2, [tool], call_counts, ctx, 1)
        assert results2[0].is_error is True
        assert "max calls" in results2[0].result

    async def test_approval_denied(self):
        tool = make_tool(requires_approval=True)
        calls = [ToolCallContent(id="tc_1", name="search", arguments={"query": "test"})]

        async def deny(req: ToolApprovalRequest) -> bool:
            return False

        ctx = ConversationContext(tool_approval=deny)
        results = await execute_tools(calls, [tool], {}, ctx, 1)
        assert results[0].is_error is True
        assert "denied" in results[0].result.lower()

    async def test_approval_approved(self):
        tool = make_tool(requires_approval=True)
        calls = [ToolCallContent(id="tc_1", name="search", arguments={"query": "test"})]

        async def approve(req: ToolApprovalRequest) -> bool:
            return True

        ctx = ConversationContext(tool_approval=approve)
        results = await execute_tools(calls, [tool], {}, ctx, 1)
        assert results[0].is_error is False

    async def test_retry_on_failure(self):
        attempt_count = 0

        async def flaky_execute(params: SearchParams) -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Transient error")
            return "Success"

        tool = make_tool(retry=2, execute=flaky_execute)
        calls = [ToolCallContent(id="tc_1", name="search", arguments={"query": "test"})]
        ctx = ConversationContext()
        results = await execute_tools(calls, [tool], {}, ctx, 1)
        assert results[0].result == "Success"
        assert attempt_count == 3

    async def test_all_retries_exhausted(self):
        tool = make_tool(retry=1, execute=failing_execute)
        calls = [ToolCallContent(id="tc_1", name="search", arguments={"query": "test"})]
        ctx = ConversationContext()
        results = await execute_tools(calls, [tool], {}, ctx, 1)
        assert results[0].is_error is True
        assert "Tool failed" in results[0].result

    async def test_pydantic_validation_error(self):
        tool = make_tool()
        calls = [ToolCallContent(id="tc_1", name="search", arguments={"wrong_field": "test"})]
        ctx = ConversationContext()
        results = await execute_tools(calls, [tool], {}, ctx, 1)
        assert results[0].is_error is True

    async def test_parallel_execution(self):
        execution_order: list[str] = []

        async def slow_execute(params: SearchParams) -> str:
            execution_order.append(f"start:{params.query}")
            await asyncio.sleep(0.01)
            execution_order.append(f"end:{params.query}")
            return f"Result: {params.query}"

        tool = make_tool(execute=slow_execute)
        calls = [
            ToolCallContent(id="tc_1", name="search", arguments={"query": "a"}),
            ToolCallContent(id="tc_2", name="search", arguments={"query": "b"}),
        ]
        ctx = ConversationContext()
        results = await execute_tools(calls, [tool], {}, ctx, 1)

        assert len(results) == 2
        # Both should start before either ends (parallel)
        assert execution_order[0].startswith("start:")
        assert execution_order[1].startswith("start:")


# ── model() step function ─────────────────���──────


class TestModelStep:
    async def test_simple_text_response(self):
        mock = MockProvider([text_response("Hello!")])

        step = model(model=Claude.Sonnet4_6)
        ctx = ConversationContext(messages=[user_message("hi")])

        with _mock_provider(mock):
            result = await step(ctx)

        assert result.stop_reason == "stop"
        assert len(result.messages) == 2  # user + assistant
        assert isinstance(result.messages[1].content[0], TextContent)
        assert result.messages[1].content[0].text == "Hello!"

    async def test_usage_accumulated(self):
        mock = MockProvider([
            text_response("Hi", TokenUsage(input_tokens=10, output_tokens=5)),
        ])

        step = model(model=Claude.Sonnet4_6)
        ctx = ConversationContext(messages=[user_message("hi")])

        with _mock_provider(mock):
            result = await step(ctx)

        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    async def test_tool_call_loop(self):
        mock = MockProvider([
            # Step 1: LLM calls a tool
            tool_call_response([{"id": "tc_1", "name": "search", "arguments": {"query": "python"}}]),
            # Step 2: LLM responds with text (no more tools)
            text_response("Python is a language."),
        ])

        tool = make_tool()
        step = model(model=Claude.Sonnet4_6)
        ctx = ConversationContext(
            messages=[user_message("what is python?")],
            tools=[tool],
        )

        with _mock_provider(mock):
            result = await step(ctx)

        assert result.stop_reason == "stop"
        # user, assistant (tool call), user (tool result), assistant (text)
        assert len(result.messages) == 4
        assert isinstance(result.messages[1].content[0], ToolCallContent)
        assert isinstance(result.messages[2].content[0], ToolResultContent)
        assert result.messages[2].content[0].result == "Results for: python"

    async def test_max_steps_enforced(self):
        # LLM always calls tools — never stops on its own
        def always_tool_call():
            i = 0
            while True:
                i += 1
                yield tool_call_response([{"id": f"tc_{i}", "name": "search", "arguments": {"query": f"q{i}"}}])

        gen = always_tool_call()
        mock = MockProvider([next(gen) for _ in range(5)])

        tool = make_tool()
        step = model(model=Claude.Sonnet4_6, max_steps=3)
        ctx = ConversationContext(
            messages=[user_message("loop forever")],
            tools=[tool],
        )

        with _mock_provider(mock):
            result = await step(ctx)

        assert result.stop_reason == "max_steps"
        assert len(mock.requests) == 3

    async def test_system_prompt_passed(self):
        mock = MockProvider([text_response("Hi")])

        step = model(model=Claude.Sonnet4_6, system="Be helpful")
        ctx = ConversationContext(messages=[user_message("hi")])

        with _mock_provider(mock):
            result = await step(ctx)

        assert mock.requests[0].system == "Be helpful"

    async def test_dynamic_system_prompt(self):
        mock = MockProvider([text_response("Hi")])

        step = model(
            model=Claude.Sonnet4_6,
            system=lambda ctx: f"Messages: {len(ctx.messages)}",
        )
        ctx = ConversationContext(messages=[user_message("hi")])

        with _mock_provider(mock):
            result = await step(ctx)

        assert mock.requests[0].system == "Messages: 1"

    async def test_temperature_passed(self):
        mock = MockProvider([text_response("Hi")])

        step = model(model=Claude.Sonnet4_6, temperature=0.7)
        ctx = ConversationContext(messages=[user_message("hi")])

        with _mock_provider(mock):
            result = await step(ctx)

        assert mock.requests[0].temperature == 0.7

    async def test_usage_accumulates_across_steps(self):
        mock = MockProvider([
            tool_call_response(
                [{"id": "tc_1", "name": "search", "arguments": {"query": "a"}}],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
            text_response(
                "Done",
                usage=TokenUsage(input_tokens=20, output_tokens=10),
            ),
        ])

        tool = make_tool()
        step = model(model=Claude.Sonnet4_6)
        ctx = ConversationContext(
            messages=[user_message("test")],
            tools=[tool],
        )

        with _mock_provider(mock):
            result = await step(ctx)

        assert result.usage is not None
        assert result.usage.input_tokens == 30
        assert result.usage.output_tokens == 15


# ── Test helpers ──────────────────────────────────


from contextlib import contextmanager
from unittest.mock import patch
from neuromod.providers.factory import ProviderFactory


@contextmanager
def _mock_provider(mock_provider: MockProvider):
    """Patch the factory to return our mock provider."""
    original_get = ProviderFactory.get

    def fake_get(self, provider, *, api_key=None, base_url=None):
        return mock_provider

    with patch.object(ProviderFactory, "get", fake_get):
        yield
