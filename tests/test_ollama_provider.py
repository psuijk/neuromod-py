from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

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
from neuromod.models.ollama import Ollama
from neuromod.providers.errors import (
    APIError,
    AuthError,
    NetworkError,
    RateLimitError,
)
from neuromod.providers.ollama import (
    OllamaProvider,
    _build_body,
    _convert_messages,
    _parse_message,
    _parse_response,
    _parse_usage,
)
from neuromod.providers.provider import (
    ProviderRequest,
    TokenUsage,
    ToolDefinition,
)


# -- Helpers -------------------------------------------


def make_request(**overrides: Any) -> ProviderRequest:
    defaults: dict[str, Any] = {
        "model": Ollama.Llama3_2,
        "messages": [user_message("hello")],
    }
    defaults.update(overrides)
    return ProviderRequest(**defaults)


def make_api_response(
    content: str | None = "Hello!",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
    }


# -- Message conversion --------------------------------


class TestConvertMessages:
    def test_user_text_message(self):
        messages = [user_message("hello")]
        result = _convert_messages(messages)
        assert result == [{"role": "user", "content": "hello"}]

    def test_assistant_text_message(self):
        messages = [assistant_message("hi")]
        result = _convert_messages(messages)
        assert result == [{"role": "assistant", "content": "hi"}]

    def test_system_message_from_param(self):
        messages = [user_message("hello")]
        result = _convert_messages(messages, system="be helpful")
        assert result[0] == {"role": "system", "content": "be helpful"}
        assert result[1] == {"role": "user", "content": "hello"}

    def test_system_message_in_messages(self):
        messages = [
            Message(role="system", content=[TextContent(text="you are helpful")]),
            user_message("hello"),
        ]
        result = _convert_messages(messages)
        assert result[0] == {"role": "system", "content": "you are helpful"}
        assert result[1] == {"role": "user", "content": "hello"}

    def test_tool_call_content(self):
        msg = assistant_message([tool_call("tc_1", "search", {"query": "test"})])
        result = _convert_messages([msg])
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] is None
        assert len(result[0]["tool_calls"]) == 1
        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "tc_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search"
        assert json.loads(tc["function"]["arguments"]) == {"query": "test"}

    def test_tool_result_content(self):
        msg = user_message([tool_result("tc_1", "found it")])
        result = _convert_messages([msg])
        assert result[0] == {
            "role": "tool",
            "tool_call_id": "tc_1",
            "content": "found it",
        }

    def test_mixed_user_message_with_text_and_tool_results(self):
        from neuromod.messages.types import TextContent, ToolResultContent
        msg = Message(role="user", content=[
            TextContent(text="here are the results"),
            ToolResultContent(call_id="tc_1", result="result 1"),
            ToolResultContent(call_id="tc_2", result="result 2"),
        ])
        result = _convert_messages([msg])
        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "here are the results"}
        assert result[1] == {"role": "tool", "tool_call_id": "tc_1", "content": "result 1"}
        assert result[2] == {"role": "tool", "tool_call_id": "tc_2", "content": "result 2"}

    def test_assistant_message_with_text_and_tool_calls(self):
        from neuromod.messages.types import TextContent, ToolCallContent
        msg = Message(role="assistant", content=[
            TextContent(text="Let me search."),
            ToolCallContent(id="tc_1", name="search", arguments={"q": "test"}),
        ])
        result = _convert_messages([msg])
        assert result[0]["content"] == "Let me search."
        assert len(result[0]["tool_calls"]) == 1


# -- Request body building -----------------------------


class TestBuildBody:
    def test_minimal_request(self):
        req = make_request()
        body = _build_body(req, stream=False)
        assert body["model"] == "llama3.2"
        assert body["max_tokens"] == 4_096
        assert len(body["messages"]) == 1
        assert "stream" not in body

    def test_stream_flag(self):
        req = make_request()
        body = _build_body(req, stream=True)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}

    def test_system_prompt(self):
        req = make_request(system="be helpful")
        body = _build_body(req, stream=False)
        assert body["messages"][0] == {"role": "system", "content": "be helpful"}

    def test_tools(self):
        tools = [ToolDefinition(name="search", description="Search", parameters={"type": "object"})]
        req = make_request(tools=tools)
        body = _build_body(req, stream=False)
        assert len(body["tools"]) == 1
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["function"]["name"] == "search"

    def test_temperature(self):
        req = make_request(temperature=0.5)
        body = _build_body(req, stream=False)
        assert body["temperature"] == 0.5

    def test_tool_choice(self):
        req = make_request(tool_choice="required")
        body = _build_body(req, stream=False)
        assert body["tool_choice"] == "required"


# -- Response parsing -----------------------------------


class TestParseResponse:
    def test_text_response(self):
        data = make_api_response()
        resp = _parse_response(data)
        assert resp.message.role == "assistant"
        assert len(resp.message.content) == 1
        assert isinstance(resp.message.content[0], TextContent)
        assert resp.message.content[0].text == "Hello!"

    def test_tool_use_response(self):
        data = make_api_response(
            content="Let me search.",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "test"}'},
            }],
        )
        resp = _parse_response(data)
        assert len(resp.message.content) == 2
        assert isinstance(resp.message.content[0], TextContent)
        assert isinstance(resp.message.content[1], ToolCallContent)
        assert resp.message.content[1].name == "search"
        assert resp.message.content[1].arguments == {"q": "test"}

    def test_usage_parsing(self):
        data = make_api_response(usage={
            "prompt_tokens": 100,
            "completion_tokens": 50,
        })
        resp = _parse_response(data)
        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 50

    def test_usage_defaults_to_zero(self):
        data = make_api_response()
        data["usage"] = {}
        resp = _parse_response(data)
        assert resp.usage.input_tokens == 0
        assert resp.usage.output_tokens == 0


class TestParseMessage:
    def test_text_only(self):
        msg = _parse_message({"role": "assistant", "content": "hello"})
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)

    def test_tool_calls_with_json_string_arguments(self):
        msg = _parse_message({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "fn", "arguments": '{"a": 1}'},
            }],
        })
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ToolCallContent)
        assert msg.content[0].arguments == {"a": 1}

    def test_tool_calls_with_dict_arguments(self):
        msg = _parse_message({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "fn", "arguments": {"a": 1}},
            }],
        })
        assert msg.content[0].arguments == {"a": 1}

    def test_empty_content(self):
        msg = _parse_message({"role": "assistant", "content": None})
        assert len(msg.content) == 0


class TestParseUsage:
    def test_maps_openai_fields(self):
        usage = _parse_usage({"prompt_tokens": 42, "completion_tokens": 10})
        assert usage.input_tokens == 42
        assert usage.output_tokens == 10

    def test_missing_fields(self):
        usage = _parse_usage({})
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


# -- Error handling ------------------------------------


class TestErrorMapping:
    async def test_auth_error(self):
        provider = OllamaProvider()
        mock_resp = httpx.Response(401, request=httpx.Request("POST", "http://test"))
        with patch.object(provider._client, "post", return_value=mock_resp):
            with pytest.raises(AuthError):
                await provider.generate(make_request())

    async def test_rate_limit_error(self):
        provider = OllamaProvider()
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
        provider = OllamaProvider()
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
        provider = OllamaProvider()
        with patch.object(provider._client, "post", side_effect=httpx.ConnectError("failed")):
            with pytest.raises(NetworkError):
                await provider.generate(make_request())

    async def test_network_error_on_timeout(self):
        provider = OllamaProvider()
        with patch.object(provider._client, "post", side_effect=httpx.ReadTimeout("timeout")):
            with pytest.raises(NetworkError):
                await provider.generate(make_request())


# -- Generate (mocked HTTP) ----------------------------


class TestGenerate:
    async def test_generate_returns_response(self):
        provider = OllamaProvider()
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

    async def test_generate_with_tool_calls(self):
        provider = OllamaProvider()
        api_data = make_api_response(
            content=None,
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "test"}'},
            }],
        )
        mock_resp = httpx.Response(
            200,
            request=httpx.Request("POST", "http://test"),
            json=api_data,
        )
        with patch.object(provider._client, "post", return_value=mock_resp):
            result = await provider.generate(make_request())
            assert len(result.message.content) == 1
            assert isinstance(result.message.content[0], ToolCallContent)
            assert result.message.content[0].name == "search"


class TestCountTokens:
    async def test_returns_approximate_zero(self):
        provider = OllamaProvider()
        result = await provider.count_tokens(make_request())
        assert result.tokens == 0
        assert result.exact is False


class TestCustomBaseUrl:
    def test_default_base_url(self):
        provider = OllamaProvider()
        assert provider._base_url == "http://localhost:11434/v1"

    def test_custom_base_url(self):
        provider = OllamaProvider(base_url="http://my-server:8080/v1")
        assert provider._base_url == "http://my-server:8080/v1"
