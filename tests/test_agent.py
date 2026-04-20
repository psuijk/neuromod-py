from __future__ import annotations

from typing import Any
from unittest.mock import patch
from contextlib import contextmanager

import pytest
from pydantic import BaseModel

from neuromod.agents.agent import Agent
from neuromod.agents.types import AgentResponse, AgentStreamResult
from neuromod.composition.context import ConversationContext, ToolApprovalRequest
from neuromod.config import configure, _config, _factory
from neuromod.messages.helpers import user_message
from neuromod.messages.types import (
    Message,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)
from neuromod.models.anthropic import Claude
from neuromod.providers.provider import (
    ProviderRequest,
    ProviderResponse,
    ProviderStreamEvent,
    ProviderStreamResult,
    TextDeltaEvent,
    TokenCount,
    TokenUsage,
)
from neuromod.providers.factory import ProviderFactory
from neuromod.streaming.events import StreamEvent
from neuromod.tools.tool import Tool


# ── Helpers ───────────────────────────────────────


class SearchParams(BaseModel):
    query: str


async def search_execute(params: SearchParams) -> str:
    return f"Results for: {params.query}"


def make_tool(name: str = "search") -> Tool:
    return Tool(
        name=name,
        description=f"A {name} tool",
        schema=SearchParams,
        execute=search_execute,
    )


def text_response(text: str, usage: TokenUsage | None = None) -> ProviderResponse:
    return ProviderResponse(
        message=Message(role="assistant", content=[TextContent(text=text)]),
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
    def __init__(
        self,
        responses: list[ProviderResponse],
        stream_events: list[list[ProviderStreamEvent]] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._stream_events = stream_events or []
        self._call_count = 0
        self.requests: list[ProviderRequest] = []

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp

    def stream(self, request: ProviderRequest) -> ProviderStreamResult:
        self.requests.append(request)
        idx = self._call_count
        self._call_count += 1
        resp = self._responses[idx]
        events = self._stream_events[idx] if idx < len(self._stream_events) else []

        async def event_iter():
            for e in events:
                yield e

        async def get_response():
            return resp

        return ProviderStreamResult(events=event_iter(), response=get_response())

    async def count_tokens(self, request: ProviderRequest) -> TokenCount:
        return TokenCount(tokens=100, exact=True)


@contextmanager
def mock_provider(mock: MockProvider):
    def fake_get(self, provider, *, api_key=None, base_url=None):
        return mock
    with patch.object(ProviderFactory, "get", fake_get):
        yield


@pytest.fixture(autouse=True)
def reset_config():
    _config.set(None)
    _factory.set(None)
    configure(api_keys={"anthropic": "test-key"})
    yield
    _config.set(None)
    _factory.set(None)


# ── Constructor ───────────────────────────────────


class TestAgentInit:
    def test_stores_config(self):
        agent = Agent(model=Claude.Sonnet4_6, max_steps=5, temperature=0.7)
        assert agent._model == Claude.Sonnet4_6
        assert agent._max_steps == 5
        assert agent._temperature == 0.7

    def test_defaults(self):
        agent = Agent(model=Claude.Sonnet4_6)
        assert agent._max_steps == 10
        assert agent._tools is None
        assert agent._system is None
        assert agent._temperature is None
        assert agent._schema is None


# ── generate() ────────────────────────────────────


class TestAgentGenerate:
    async def test_simple_text_response(self):
        mock = MockProvider([text_response("Hello!")])
        agent = Agent(model=Claude.Sonnet4_6)

        with mock_provider(mock):
            response = await agent.generate("hi")

        assert isinstance(response, AgentResponse)
        assert response.text == "Hello!"
        assert response.finish_reason == "stop"
        assert len(response.messages) == 2

    async def test_returns_usage(self):
        mock = MockProvider([
            text_response("Hi", TokenUsage(input_tokens=100, output_tokens=50)),
        ])
        agent = Agent(model=Claude.Sonnet4_6)

        with mock_provider(mock):
            response = await agent.generate("hi")

        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50

    async def test_duration_tracked(self):
        mock = MockProvider([text_response("Hi")])
        agent = Agent(model=Claude.Sonnet4_6)

        with mock_provider(mock):
            response = await agent.generate("hi")

        assert response.duration_ms >= 0

    async def test_tool_call_loop(self):
        mock = MockProvider([
            tool_call_response([{"id": "tc_1", "name": "search", "arguments": {"query": "python"}}]),
            text_response("Python is great."),
        ])
        tool = make_tool()
        agent = Agent(model=Claude.Sonnet4_6, tools=[tool])

        with mock_provider(mock):
            response = await agent.generate("what is python?")

        assert response.finish_reason == "stop"
        assert response.text == "Python is great."
        # user, assistant (tool call), user (tool result), assistant (text)
        assert len(response.messages) == 4

    async def test_system_prompt_from_agent(self):
        mock = MockProvider([text_response("Hi")])
        agent = Agent(model=Claude.Sonnet4_6, system="Be helpful")

        with mock_provider(mock):
            await agent.generate("hi")

        assert mock.requests[0].system == "Be helpful"

    async def test_system_prompt_override(self):
        mock = MockProvider([text_response("Hi")])
        agent = Agent(model=Claude.Sonnet4_6, system="Be helpful")

        with mock_provider(mock):
            await agent.generate("hi", system="Be concise")

        assert mock.requests[0].system == "Be concise"

    async def test_temperature_override(self):
        mock = MockProvider([text_response("Hi")])
        agent = Agent(model=Claude.Sonnet4_6, temperature=0.5)

        with mock_provider(mock):
            await agent.generate("hi", temperature=0.9)

        assert mock.requests[0].temperature == 0.9

    async def test_max_steps_override(self):
        mock = MockProvider([
            tool_call_response([{"id": f"tc_{i}", "name": "search", "arguments": {"query": f"q{i}"}}])
            for i in range(5)
        ])
        tool = make_tool()
        agent = Agent(model=Claude.Sonnet4_6, tools=[tool], max_steps=10)

        with mock_provider(mock):
            response = await agent.generate("loop", max_steps=2)

        assert response.finish_reason == "max_steps"
        assert len(mock.requests) == 2

    async def test_tool_choice_passed(self):
        mock = MockProvider([text_response("Hi")])
        agent = Agent(model=Claude.Sonnet4_6)

        with mock_provider(mock):
            await agent.generate("hi", tool_choice="required")

        assert mock.requests[0].tool_choice == "required"

    async def test_empty_response_handling(self):
        mock = MockProvider([
            ProviderResponse(
                message=Message(role="assistant", content=[]),
                usage=TokenUsage(input_tokens=5, output_tokens=0),
            ),
        ])
        agent = Agent(model=Claude.Sonnet4_6)

        with mock_provider(mock):
            response = await agent.generate("hi")

        assert response.text == ""
        assert response.finish_reason == "stop"


# ── __call__ (StepFunction) ──────────────────────


class TestAgentCall:
    async def test_callable_as_step_function(self):
        mock = MockProvider([text_response("Hello!")])
        agent = Agent(model=Claude.Sonnet4_6)

        ctx = ConversationContext(messages=[user_message("hi")])

        with mock_provider(mock):
            result = await agent(ctx)

        assert isinstance(result, ConversationContext)
        assert len(result.messages) == 2

    async def test_uses_context_tools_over_agent_tools(self):
        mock = MockProvider([text_response("Hi")])
        agent_tool = make_tool("agent_tool")
        ctx_tool = make_tool("ctx_tool")
        agent = Agent(model=Claude.Sonnet4_6, tools=[agent_tool])

        ctx = ConversationContext(
            messages=[user_message("hi")],
            tools=[ctx_tool],
        )

        with mock_provider(mock):
            result = await agent(ctx)

        # Context tools should win
        assert mock.requests[0].tools is not None
        assert len(mock.requests[0].tools) == 1
        assert mock.requests[0].tools[0].name == "ctx_tool"

    async def test_falls_back_to_agent_tools(self):
        mock = MockProvider([text_response("Hi")])
        agent_tool = make_tool("agent_tool")
        agent = Agent(model=Claude.Sonnet4_6, tools=[agent_tool])

        ctx = ConversationContext(messages=[user_message("hi")])

        with mock_provider(mock):
            result = await agent(ctx)

        assert mock.requests[0].tools is not None
        assert mock.requests[0].tools[0].name == "agent_tool"


# ── Structured output ────────────────────────────


class TestAgentStructuredOutput:
    async def test_parses_structured_output(self):
        class Weather(BaseModel):
            city: str
            temp_f: float

        mock = MockProvider([
            text_response('{"city": "Amsterdam", "temp_f": 52.0}'),
        ])
        agent = Agent(model=Claude.Sonnet4_6, schema=Weather)

        with mock_provider(mock):
            response = await agent.generate("weather?")

        assert response.output is not None
        assert isinstance(response.output, Weather)
        assert response.output.city == "Amsterdam"
        assert response.output.temp_f == 52.0


# ── Thread support ────────────────────────────────


class TestAgentThread:
    async def test_thread_persists_messages(self):
        from neuromod.composition.thread import InMemoryThreadStore
        store = InMemoryThreadStore()
        configure(api_keys={"anthropic": "test-key"}, thread_store=store)

        mock_responses = [
            text_response("First response"),
            text_response("Second response"),
        ]

        agent = Agent(model=Claude.Sonnet4_6)

        with mock_provider(MockProvider([mock_responses[0]])):
            await agent.generate("first message", thread="t1")

        with mock_provider(MockProvider([mock_responses[1]])):
            response = await agent.generate("second message", thread="t1")

        # Second call should include history from first call
        assert len(response.messages) == 4  # first user + first assistant + second user + second assistant


# ── stream() ──────────────────────────────────────


class TestAgentStream:
    async def test_returns_stream_result(self):
        mock = MockProvider(
            responses=[text_response("Hello!")],
            stream_events=[[TextDeltaEvent(text="Hello!")]],
        )
        agent = Agent(model=Claude.Sonnet4_6)

        with mock_provider(mock):
            result = agent.stream("hi")
            assert isinstance(result, AgentStreamResult)
            async for _ in result.events:
                pass
            await result.response

    async def test_events_are_iterable(self):
        mock = MockProvider(
            responses=[text_response("Hello!")],
            stream_events=[[
                TextDeltaEvent(text="Hel"),
                TextDeltaEvent(text="lo!"),
            ]],
        )
        agent = Agent(model=Claude.Sonnet4_6)

        with mock_provider(mock):
            result = agent.stream("hi")
            events: list[StreamEvent] = []
            async for event in result.events:
                events.append(event)

        # At least the text delta events + step_start
        assert len(events) >= 2

    async def test_response_is_awaitable(self):
        mock = MockProvider(
            responses=[text_response("Hello!")],
            stream_events=[[TextDeltaEvent(text="Hello!")]],
        )
        agent = Agent(model=Claude.Sonnet4_6)

        with mock_provider(mock):
            result = agent.stream("hi")
            # Drain events first
            async for _ in result.events:
                pass
            response = await result.response

        assert isinstance(response, AgentResponse)
        assert response.text == "Hello!"
        assert response.finish_reason == "stop"

    async def test_events_contain_text_deltas(self):
        mock = MockProvider(
            responses=[text_response("Hello world")],
            stream_events=[[
                TextDeltaEvent(text="Hello "),
                TextDeltaEvent(text="world"),
            ]],
        )
        agent = Agent(model=Claude.Sonnet4_6)

        with mock_provider(mock):
            result = agent.stream("hi")
            text_events = []
            async for event in result.events:
                if event.type == "text_delta":
                    text_events.append(event)

        assert len(text_events) == 2
        assert text_events[0].text == "Hello "
        assert text_events[1].text == "world"

    async def test_on_event_also_called(self):
        mock = MockProvider(
            responses=[text_response("Hello!")],
            stream_events=[[TextDeltaEvent(text="Hello!")]],
        )
        agent = Agent(model=Claude.Sonnet4_6)

        side_events: list[StreamEvent] = []

        with mock_provider(mock):
            result = agent.stream("hi", on_event=lambda e: side_events.append(e))
            async for _ in result.events:
                pass
            await result.response

        # on_event should have received the same events as the channel
        assert len(side_events) >= 1

    async def test_stream_returns_immediately(self):
        mock = MockProvider(
            responses=[text_response("Hello!")],
            stream_events=[[TextDeltaEvent(text="Hello!")]],
        )
        agent = Agent(model=Claude.Sonnet4_6)

        with mock_provider(mock):
            # stream() is not async — it returns immediately
            result = agent.stream("hi")
            assert result is not None
            # Clean up
            async for _ in result.events:
                pass
            await result.response
