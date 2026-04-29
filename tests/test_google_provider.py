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
    MediaContent,
    Message,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)
from neuromod.models.google import Google
from neuromod.providers.errors import (
    APIError,
    AuthError,
    NetworkError,
    RateLimitError,
)
from neuromod.providers.google import (
    GeminiProvider,
    _build_body,
    _convert_messages,
    _convert_parts,
    _convert_tool_choice,
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
        "model": Google.Flash2_5,
        "messages": [user_message("hello")],
    }
    defaults.update(overrides)
    return ProviderRequest(**defaults)


def make_api_response(
    parts: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": parts or [{"text": "Hello!"}],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": usage or {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }


# -- Message conversion --------------------------------


class TestConvertMessages:
    def test_user_text_message(self):
        messages = [user_message("hello")]
        result = _convert_messages(messages)
        assert result == [{"role": "user", "parts": [{"text": "hello"}]}]

    def test_assistant_text_message(self):
        messages = [assistant_message("hi")]
        result = _convert_messages(messages)
        assert result == [{"role": "model", "parts": [{"text": "hi"}]}]

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
        part = result[0]["parts"][0]
        assert part == {"functionCall": {"name": "search", "args": {"query": "test"}}}

    def test_tool_result_content(self):
        msg = user_message([tool_result("tc_1", "found it", name="search")])
        result = _convert_messages([msg])
        part = result[0]["parts"][0]
        assert part == {
            "functionResponse": {
                "name": "search",
                "response": {"result": "found it"},
            },
        }

    def test_image_media(self):
        msg = Message(role="user", content=[
            MediaContent(data="abc123", mime_type="image/png"),
        ])
        result = _convert_messages([msg])
        part = result[0]["parts"][0]
        assert part == {"inlineData": {"mimeType": "image/png", "data": "abc123"}}


# -- Content conversion --------------------------------


class TestConvertParts:
    def test_text(self):
        parts = _convert_parts([TextContent(text="hi")])
        assert parts == [{"text": "hi"}]

    def test_tool_call(self):
        parts = _convert_parts([ToolCallContent(id="1", name="fn", arguments={"a": 1})])
        assert parts == [{"functionCall": {"name": "fn", "args": {"a": 1}}}]

    def test_tool_result(self):
        parts = _convert_parts([ToolResultContent(call_id="1", result="ok", name="fn")])
        assert parts == [{"functionResponse": {"name": "fn", "response": {"result": "ok"}}}]

    def test_media(self):
        parts = _convert_parts([MediaContent(data="abc", mime_type="image/jpeg")])
        assert parts == [{"inlineData": {"mimeType": "image/jpeg", "data": "abc"}}]


# -- Tool choice conversion ----------------------------


class TestConvertToolChoice:
    def test_required(self):
        assert _convert_tool_choice("required") == "ANY"

    def test_none(self):
        assert _convert_tool_choice("none") == "NONE"

    def test_auto(self):
        assert _convert_tool_choice("auto") == "AUTO"


# -- Request body building -----------------------------


class TestBuildBody:
    def test_minimal_request(self):
        req = make_request()
        body = _build_body(req)
        assert len(body["contents"]) == 1
        assert body["generationConfig"]["maxOutputTokens"] == 64_000
        assert "systemInstruction" not in body

    def test_system_prompt(self):
        req = make_request(system="be helpful")
        body = _build_body(req)
        assert body["systemInstruction"] == {"parts": [{"text": "be helpful"}]}

    def test_tools(self):
        tools = [ToolDefinition(name="search", description="Search", parameters={"type": "object"})]
        req = make_request(tools=tools)
        body = _build_body(req)
        decls = body["tools"][0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "search"

    def test_temperature(self):
        req = make_request(temperature=0.5)
        body = _build_body(req)
        assert body["generationConfig"]["temperature"] == 0.5

    def test_tool_choice(self):
        req = make_request(tool_choice="required")
        body = _build_body(req)
        assert body["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"


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
        data = make_api_response(parts=[
            {"text": "Let me search."},
            {"functionCall": {"name": "search", "args": {"q": "test"}}},
        ])
        resp = _parse_response(data)
        assert len(resp.message.content) == 2
        assert isinstance(resp.message.content[0], TextContent)
        assert isinstance(resp.message.content[1], ToolCallContent)
        assert resp.message.content[1].name == "search"
        assert resp.message.content[1].arguments == {"q": "test"}
        assert resp.message.content[1].id == "call_0"

    def test_usage_parsing(self):
        data = make_api_response(usage={
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "cachedContentTokenCount": 20,
        })
        resp = _parse_response(data)
        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 50
        assert resp.usage.cache_read_tokens == 20

    def test_usage_defaults_to_zero(self):
        data = make_api_response()
        data["usageMetadata"] = {}
        resp = _parse_response(data)
        assert resp.usage.input_tokens == 0
        assert resp.usage.output_tokens == 0


class TestParseMessage:
    def test_text_only(self):
        msg = _parse_message({"parts": [{"text": "hello"}]})
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)

    def test_function_call(self):
        msg = _parse_message({
            "parts": [{"functionCall": {"name": "fn", "args": {"a": 1}}}],
        })
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ToolCallContent)
        assert msg.content[0].id == "call_0"
        assert msg.content[0].name == "fn"
        assert msg.content[0].arguments == {"a": 1}

    def test_multiple_function_calls(self):
        msg = _parse_message({
            "parts": [
                {"functionCall": {"name": "fn1", "args": {}}},
                {"functionCall": {"name": "fn2", "args": {}}},
            ],
        })
        assert len(msg.content) == 2
        assert msg.content[0].id == "call_0"
        assert msg.content[1].id == "call_1"

    def test_empty_parts(self):
        msg = _parse_message({})
        assert len(msg.content) == 0


class TestParseUsage:
    def test_maps_gemini_fields(self):
        usage = _parse_usage({"promptTokenCount": 42, "candidatesTokenCount": 10})
        assert usage.input_tokens == 42
        assert usage.output_tokens == 10

    def test_cache_tokens(self):
        usage = _parse_usage({
            "promptTokenCount": 42,
            "candidatesTokenCount": 10,
            "cachedContentTokenCount": 5,
        })
        assert usage.cache_read_tokens == 5

    def test_missing_fields(self):
        usage = _parse_usage({})
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


# -- Error handling ------------------------------------


class TestErrorMapping:
    async def test_auth_error(self):
        provider = GeminiProvider(api_key="bad-key")
        mock_resp = httpx.Response(403, request=httpx.Request("POST", "http://test"))
        with patch.object(provider._client, "post", return_value=mock_resp):
            with pytest.raises(AuthError):
                await provider.generate(make_request())

    async def test_rate_limit_error(self):
        provider = GeminiProvider(api_key="key")
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
        provider = GeminiProvider(api_key="key")
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
        provider = GeminiProvider(api_key="key")
        with patch.object(provider._client, "post", side_effect=httpx.ConnectError("failed")):
            with pytest.raises(NetworkError):
                await provider.generate(make_request())

    async def test_network_error_on_timeout(self):
        provider = GeminiProvider(api_key="key")
        with patch.object(provider._client, "post", side_effect=httpx.ReadTimeout("timeout")):
            with pytest.raises(NetworkError):
                await provider.generate(make_request())


# -- Generate (mocked HTTP) ----------------------------


class TestGenerate:
    async def test_generate_returns_response(self):
        provider = GeminiProvider(api_key="test-key")
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
        provider = GeminiProvider(api_key="test-key")
        api_data = make_api_response(parts=[
            {"functionCall": {"name": "search", "args": {"q": "test"}}},
        ])
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
    async def test_count_tokens(self):
        provider = GeminiProvider(api_key="test-key")
        mock_resp = httpx.Response(
            200,
            request=httpx.Request("POST", "http://test"),
            json={"totalTokens": 42},
        )
        with patch.object(provider._client, "post", return_value=mock_resp):
            result = await provider.count_tokens(make_request())
            assert result.tokens == 42
            assert result.exact is True


# -- Factory integration --------------------------------


class TestFactoryIntegration:
    def test_factory_builds_google_provider(self):
        from neuromod.providers.factory import ProviderFactory, ProviderFactoryConfig
        factory = ProviderFactory(ProviderFactoryConfig(api_keys={"google": "test-key"}))
        provider = factory.get("google")
        assert provider is not None
