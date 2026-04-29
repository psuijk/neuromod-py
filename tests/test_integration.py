"""Integration tests that make real API calls.

Run with: pytest tests/test_integration.py -v
Skip slow providers: pytest tests/test_integration.py -v -k "not anthropic"

Requires:
- ANTHROPIC_API_KEY env var for Anthropic tests
- GEMINI_API_KEY or GOOGLE_AI_API_KEY env var for Google tests
- Ollama running locally with qwen2.5:0.5b pulled
"""

from __future__ import annotations

import os

import pytest

from neuromod import Agent, Claude, Google, custom_model
from neuromod.agents.types import AgentResponse
from neuromod.messages.types import TextContent, ToolCallContent
from neuromod.providers.errors import AuthError, NetworkError
from neuromod.streaming.events import TextDeltaStreamEvent
from neuromod.tools.tool import create_tool
from pydantic import BaseModel


# -- Markers -------------------------------------------

has_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

has_google_key = pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_AI_API_KEY")),
    reason="GEMINI_API_KEY / GOOGLE_AI_API_KEY not set",
)

ollama_model = custom_model("ollama", "qwen2.5:0.5b", max_input=32_000, max_output=2_048)


def _ollama_available() -> bool:
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


has_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running on localhost:11434",
)

integration = pytest.mark.integration


# -- Shared tool for tool-calling tests -----------------


class AddArgs(BaseModel):
    a: int
    b: int


async def _add_execute(args: AddArgs) -> str:
    return str(args.a + args.b)


add_tool = create_tool(
    name="add",
    description="Add two numbers together",
    schema=AddArgs,
    execute=_add_execute,
)


# -- Anthropic -----------------------------------------


@integration
@has_anthropic_key
class TestAnthropicIntegration:
    async def test_generate_text(self):
        agent = Agent(model=Claude.Haiku4_5)
        response = await agent.generate("Reply with exactly: hello world")
        assert isinstance(response, AgentResponse)
        assert len(response.text) > 0
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    async def test_stream_text(self):
        agent = Agent(model=Claude.Haiku4_5)
        result = agent.stream("Reply with exactly: hi")
        chunks: list[str] = []
        async for event in result.events:
            if isinstance(event, TextDeltaStreamEvent):
                chunks.append(event.text)
        response = await result.response
        assert len(chunks) > 0
        assert len(response.text) > 0
        assert "".join(chunks).strip() in response.text

    async def test_tool_calling(self):
        agent = Agent(
            model=Claude.Haiku4_5,
            tools=[add_tool],
            system="Use the add tool to answer math questions. After getting the result, state the answer.",
        )
        response = await agent.generate("What is 7 + 12?")
        assert "19" in response.text


# -- Google/Gemini -------------------------------------


@integration
@has_google_key
class TestGoogleIntegration:
    async def test_generate_text(self):
        agent = Agent(model=Google.Flash2_5)
        response = await agent.generate("Reply with exactly: hello world")
        assert isinstance(response, AgentResponse)
        assert len(response.text) > 0
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    async def test_stream_text(self):
        agent = Agent(model=Google.Flash2_5)
        result = agent.stream("Reply with exactly: hi")
        chunks: list[str] = []
        async for event in result.events:
            if isinstance(event, TextDeltaStreamEvent):
                chunks.append(event.text)
        response = await result.response
        assert len(chunks) > 0
        assert len(response.text) > 0

    async def test_tool_calling(self):
        agent = Agent(
            model=Google.Flash2_5,
            tools=[add_tool],
            system="Use the add tool to answer math questions. After getting the result, state the answer.",
        )
        response = await agent.generate("What is 7 + 12?")
        assert "19" in response.text

    async def test_count_tokens(self):
        agent = Agent(model=Google.Flash2_5)
        count = await agent.count_tokens("Hello, how are you?")
        assert count.tokens > 0
        assert count.exact is True


# -- Ollama --------------------------------------------


@integration
@has_ollama
class TestOllamaIntegration:
    async def test_generate_text(self):
        agent = Agent(model=ollama_model)
        response = await agent.generate("Say hello in one sentence.")
        assert isinstance(response, AgentResponse)
        assert len(response.text) > 0

    async def test_stream_text(self):
        agent = Agent(model=ollama_model)
        result = agent.stream("Say hello in one sentence.")
        chunks: list[str] = []
        async for event in result.events:
            if isinstance(event, TextDeltaStreamEvent):
                chunks.append(event.text)
        response = await result.response
        assert len(chunks) > 0
        assert len(response.text) > 0

    async def test_custom_base_url(self):
        model = custom_model("ollama", "qwen2.5:0.5b", max_input=32_000, max_output=2_048)
        agent = Agent(model=model, base_url="http://localhost:11434/v1")
        response = await agent.generate("Say hi.")
        assert len(response.text) > 0
