from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from neuromod.messages.helpers import (
    assistant_message,
    tool_call,
    tool_result,
    user_message,
)
from neuromod.messages.types import (
    Message,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)
from neuromod.models.anthropic import Claude
from neuromod.providers.anthropic import (
    ClaudeProvider,
    _build_body,
    _convert_content,
    _convert_messages,
    _convert_tool_choice,
    _parse_message,
    _parse_response,
    _parse_usage,
)
from neuromod.providers.errors import (
    APIError,
    AuthError,
    NetworkError,
    RateLimitError,
)
from neuromod.providers.provider import (
    ProviderRequest,
    TokenUsage,
    ToolDefinition,
)


# ── Helpers ───────────────────────────────────────


def make_request(**overrides: Any) -> ProviderRequest:
    defaults: dict[str, Any] = {
        "model": Claude.Sonnet4_6,
        "messages": [user_message("hello")],
    }
    defaults.update(overrides)
    return ProviderRequest(**defaults)


def make_api_response(
    content: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": content or [{"type": "text", "text": "Hello!"}],
        "usage": usage or {"input_tokens": 10, "output_tokens": 5},
        "stop_reason": "end_turn",
    }


# ── Message conversion ───────────────────────────


class TestConvertMessages:
    def test_user_text_message(self):
        messages = [user_message("hello")]
        result = _convert_messages(messages)
        assert result == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    def test_assistant_text_message(self):
        messages = [assistant_message("hi")]
        result = _convert_messages(messages)
        assert result == [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]

    def test_system_messages_filtered(self):
        messages = [
            Message(role="system", content=[TextContent(text="you are helpful")]),
            user_message("hello"),
        ]
        result = _convert_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_tool_call_content(self):
        msg = assistant_message([tool_call("tc_1", "search", {"query": "test"})])
        result = _convert_messages([msg])
        block = result[0]["content"][0]
        assert block == {
            "type": "tool_use",
            "id": "tc_1",
            "name": "search",
            "input": {"query": "test"},
        }

    def test_tool_result_content(self):
        msg = user_message([tool_result("tc_1", "found it")])
        result = _convert_messages([msg])
        block = result[0]["content"][0]
        assert block == {
            "type": "tool_result",
            "tool_use_id": "tc_1",
            "content": "found it",
            "is_error": False,
        }


# ── Content conversion ───────────────────────────


class TestConvertContent:
    def test_text(self):
        assert _convert_content(TextContent(text="hi")) == {"type": "text", "text": "hi"}

    def test_tool_call(self):
        content = ToolCallContent(id="1", name="fn", arguments={"a": 1})
        result = _convert_content(content)
        assert result["type"] == "tool_use"
        assert result["id"] == "1"
        assert result["input"] == {"a": 1}

    def test_tool_result(self):
        content = ToolResultContent(call_id="1", result="ok")
        result = _convert_content(content)
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "1"

    def test_image_media(self):
        from neuromod.messages.types import MediaContent
        content = MediaContent(data="abc123", mime_type="image/png")
        result = _convert_content(content)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"

    def test_document_media(self):
        from neuromod.messages.types import MediaContent
        content = MediaContent(data="abc123", mime_type="application/pdf")
        result = _convert_content(content)
        assert result["type"] == "document"


# ── Tool choice conversion ───────────────────────


class TestConvertToolChoice:
    def test_required(self):
        assert _convert_tool_choice("required") == {"type": "any"}

    def test_none(self):
        assert _convert_tool_choice("none") == {"type": "none"}

    def test_auto(self):
        assert _convert_tool_choice("auto") == {"type": "auto"}


# ── Request body building ────────────────────────


class TestBuildBody:
    def test_minimal_request(self):
        req = make_request()
        body = _build_body(req, stream=False)
        assert body["model"] == "claude-sonnet-4-6"
        assert body["max_tokens"] == 64_000
        assert len(body["messages"]) == 1
        assert "stream" not in body

    def test_stream_flag(self):
        req = make_request()
        body = _build_body(req, stream=True)
        assert body["stream"] is True

    def test_system_prompt(self):
        req = make_request(system="be helpful")
        body = _build_body(req, stream=False)
        assert body["system"] == "be helpful"

    def test_tools(self):
        tools = [ToolDefinition(name="search", description="Search", parameters={"type": "object"})]
        req = make_request(tools=tools)
        body = _build_body(req, stream=False)
        assert len(body["tools"]) == 1
        assert body["tools"][0]["name"] == "search"

    def test_temperature(self):
        req = make_request(temperature=0.5)
        body = _build_body(req, stream=False)
        assert body["temperature"] == 0.5

    def test_tool_choice(self):
        req = make_request(tool_choice="required")
        body = _build_body(req, stream=False)
        assert body["tool_choice"] == {"type": "any"}


# ── Response parsing ─────────────────────────────


class TestParseResponse:
    def test_text_response(self):
        data = make_api_response()
        resp = _parse_response(data)
        assert resp.message.role == "assistant"
        assert len(resp.message.content) == 1
        assert isinstance(resp.message.content[0], TextContent)
        assert resp.message.content[0].text == "Hello!"

    def test_tool_use_response(self):
        data = make_api_response(content=[
            {"type": "text", "text": "Let me search."},
            {"type": "tool_use", "id": "tc_1", "name": "search", "input": {"q": "test"}},
        ])
        resp = _parse_response(data)
        assert len(resp.message.content) == 2
        assert isinstance(resp.message.content[1], ToolCallContent)
        assert resp.message.content[1].name == "search"
        assert resp.message.content[1].arguments == {"q": "test"}

    def test_usage_parsing(self):
        data = make_api_response(usage={
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 10,
        })
        resp = _parse_response(data)
        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 50
        assert resp.usage.cache_read_tokens == 20
        assert resp.usage.cache_write_tokens == 10


# ── Error handling ────────────────────────────────


class TestErrorMapping:
    async def test_auth_error(self):
        provider = ClaudeProvider(api_key="bad-key")
        mock_resp = httpx.Response(401, request=httpx.Request("POST", "http://test"))
        with patch.object(provider._client, "post", return_value=mock_resp):
            with pytest.raises(AuthError):
                await provider.generate(make_request())

    async def test_rate_limit_error(self):
        provider = ClaudeProvider(api_key="key")
        mock_resp = httpx.Response(
            429,
            request=httpx.Request("POST", "http://test"),
            headers={"retry-after": "2.5"},
        )
        with patch.object(provider._client, "post", return_value=mock_resp):
            with pytest.raises(RateLimitError) as exc_info:
                await provider.generate(make_request())
            assert exc_info.value.retry_after_ms == 2500

    async def test_api_error(self):
        provider = ClaudeProvider(api_key="key")
        mock_resp = httpx.Response(
            500,
            request=httpx.Request("POST", "http://test"),
            text="Internal Server Error",
        )
        with patch.object(provider._client, "post", return_value=mock_resp):
            with pytest.raises(APIError) as exc_info:
                await provider.generate(make_request())
            assert exc_info.value.status_code == 500

    async def test_network_error_on_connect(self):
        provider = ClaudeProvider(api_key="key")
        with patch.object(provider._client, "post", side_effect=httpx.ConnectError("failed")):
            with pytest.raises(NetworkError):
                await provider.generate(make_request())

    async def test_network_error_on_timeout(self):
        provider = ClaudeProvider(api_key="key")
        with patch.object(provider._client, "post", side_effect=httpx.ReadTimeout("timeout")):
            with pytest.raises(NetworkError):
                await provider.generate(make_request())


# ── Generate (mocked HTTP) ───────────────────────


class TestGenerate:
    async def test_generate_returns_response(self):
        provider = ClaudeProvider(api_key="test-key")
        api_data = make_api_response()
        mock_resp = httpx.Response(
            200,
            request=httpx.Request("POST", "http://test"),
            json=api_data,
        )
        with patch.object(provider._client, "post", return_value=mock_resp):
            result = await provider.generate(make_request())
            assert isinstance(result.message.content[0], TextContent)
            assert result.message.content[0].text == "Hello!"
            assert result.usage.input_tokens == 10
