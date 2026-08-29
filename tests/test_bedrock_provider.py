from __future__ import annotations

import base64
import binascii
import json
from typing import Any, AsyncGenerator
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
)
from neuromod.models.bedrock import Bedrock
from neuromod.providers.bedrock import (
    BedrockProvider,
    _build_body,
    _convert_content,
    _convert_messages,
    _convert_tool_choice,
    _decode_frame,
    _iter_eventstream,
    _model_path,
    _parse_frame_headers,
    _parse_response,
    _parse_usage,
    _resolve_endpoint,
)
from neuromod.providers.errors import (
    APIError,
    AuthError,
    ConfigError,
    NetworkError,
    RateLimitError,
)
from neuromod.providers.provider import (
    ProviderRequest,
    ToolDefinition,
)

_ENDPOINT = "https://bedrock-runtime.us-east-1.amazonaws.com"


# ── Helpers ───────────────────────────────────────


def make_request(**overrides: Any) -> ProviderRequest:
    defaults: dict[str, Any] = {
        "model": Bedrock.Opus5,
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


def make_frame(payload: bytes, headers: dict[str, str]) -> bytes:
    """Encode a single AWS event-stream frame (string headers only)."""
    header_bytes = b""
    for name, value in headers.items():
        nb = name.encode("utf-8")
        vb = value.encode("utf-8")
        header_bytes += bytes([len(nb)]) + nb + bytes([7]) + len(vb).to_bytes(2, "big") + vb

    headers_len = len(header_bytes)
    total_len = 12 + headers_len + len(payload) + 4
    prelude = total_len.to_bytes(4, "big") + headers_len.to_bytes(4, "big")
    prelude_crc = binascii.crc32(prelude).to_bytes(4, "big")
    message = prelude + prelude_crc + header_bytes + payload
    message_crc = binascii.crc32(message).to_bytes(4, "big")
    return message + message_crc


def make_chunk_frame(event: dict[str, Any]) -> bytes:
    """Wrap an Anthropic streaming event dict in a Bedrock 'chunk' frame."""
    inner = base64.b64encode(json.dumps(event).encode("utf-8")).decode("ascii")
    payload = json.dumps({"bytes": inner}).encode("utf-8")
    return make_frame(payload, {
        ":message-type": "event",
        ":event-type": "chunk",
        ":content-type": "application/json",
    })


class FakeStreamResponse:
    def __init__(self, status_code: int, chunks: list[bytes]) -> None:
        self.status_code = status_code
        self._chunks = chunks

    async def aiter_bytes(self) -> AsyncGenerator[bytes, None]:
        for chunk in self._chunks:
            yield chunk

    async def aread(self) -> bytes:
        return b""


class FakeStreamCtx:
    def __init__(self, resp: FakeStreamResponse) -> None:
        self._resp = resp

    async def __aenter__(self) -> FakeStreamResponse:
        return self._resp

    async def __aexit__(self, *args: Any) -> bool:
        return False


# ── Region / endpoint resolution ─────────────────


class TestResolveEndpoint:
    def test_explicit_base_url_wins(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        assert _resolve_endpoint("https://custom.example.com") == "https://custom.example.com"

    def test_base_url_trailing_slash_stripped(self):
        assert _resolve_endpoint("https://custom.example.com/") == "https://custom.example.com"

    def test_bedrock_region_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        monkeypatch.setenv("BEDROCK_REGION", "ap-southeast-2")
        assert _resolve_endpoint(None) == "https://bedrock-runtime.ap-southeast-2.amazonaws.com"

    def test_bedrock_region_takes_priority_over_aws_region(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BEDROCK_REGION", "us-west-2")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        assert _resolve_endpoint(None) == "https://bedrock-runtime.us-west-2.amazonaws.com"

    def test_aws_region_fallback(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("BEDROCK_REGION", raising=False)
        monkeypatch.setenv("AWS_REGION", "eu-central-1")
        assert _resolve_endpoint(None) == "https://bedrock-runtime.eu-central-1.amazonaws.com"

    def test_missing_region_raises_config_error(self, monkeypatch: pytest.MonkeyPatch):
        for var in ("BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ConfigError, match="region"):
            _resolve_endpoint(None)


# ── Model path ────────────────────────────────────


class TestModelPath:
    def test_colon_is_percent_encoded(self):
        path = _model_path("anthropic.claude-3-5-sonnet-20241022-v2:0", "invoke")
        assert path == "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/invoke"

    def test_stream_action(self):
        path = _model_path("anthropic.claude-3-haiku-20240307-v1:0", "invoke-with-response-stream")
        assert path.endswith("/invoke-with-response-stream")
        assert "%3A" in path


# ── Message conversion ───────────────────────────


class TestConvertMessages:
    def test_user_text_message(self):
        result = _convert_messages([user_message("hello")])
        assert result == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

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
        block = _convert_messages([msg])[0]["content"][0]
        assert block == {"type": "tool_use", "id": "tc_1", "name": "search", "input": {"query": "test"}}

    def test_tool_result_content(self):
        msg = user_message([tool_result("tc_1", "found it")])
        block = _convert_messages([msg])[0]["content"][0]
        assert block == {
            "type": "tool_result",
            "tool_use_id": "tc_1",
            "content": "found it",
            "is_error": False,
        }


class TestConvertContent:
    def test_image_media(self):
        from neuromod.messages.types import MediaContent
        result = _convert_content(MediaContent(data="abc123", mime_type="image/png"))
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"

    def test_document_media(self):
        from neuromod.messages.types import MediaContent
        result = _convert_content(MediaContent(data="abc123", mime_type="application/pdf"))
        assert result["type"] == "document"


class TestConvertToolChoice:
    def test_required(self):
        assert _convert_tool_choice("required") == {"type": "any"}

    def test_none(self):
        assert _convert_tool_choice("none") == {"type": "none"}

    def test_auto(self):
        assert _convert_tool_choice("auto") == {"type": "auto"}


# ── Request body building ────────────────────────


class TestBuildBody:
    def test_uses_anthropic_version_not_model(self):
        body = _build_body(make_request())
        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert "model" not in body  # model id goes in the URL, not the body
        assert "stream" not in body  # streaming is chosen by the endpoint

    def test_max_tokens_and_messages(self):
        body = _build_body(make_request())
        assert body["max_tokens"] == Bedrock.Opus5.max_tokens
        assert len(body["messages"]) == 1

    def test_system_prompt(self):
        body = _build_body(make_request(system="be helpful"))
        assert body["system"] == "be helpful"

    def test_tools(self):
        tools = [ToolDefinition(name="search", description="Search", parameters={"type": "object"})]
        body = _build_body(make_request(tools=tools))
        assert body["tools"][0]["name"] == "search"

    def test_temperature(self):
        assert _build_body(make_request(temperature=0.5))["temperature"] == 0.5

    def test_tool_choice(self):
        assert _build_body(make_request(tool_choice="required"))["tool_choice"] == {"type": "any"}

    def test_schema_adds_tool_and_forces_choice(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        body = _build_body(make_request(schema=schema))
        assert body["tool_choice"] == {"type": "tool", "name": "_structured_output"}
        assert body["tools"][-1]["name"] == "_structured_output"

    def test_schema_appends_to_existing_tools(self):
        schema = {"type": "object"}
        tools = [ToolDefinition(name="search", description="Search", parameters={"type": "object"})]
        body = _build_body(make_request(tools=tools, schema=schema))
        names = [t["name"] for t in body["tools"]]
        assert names == ["search", "_structured_output"]


# ── Response parsing ─────────────────────────────


class TestParseResponse:
    def test_text_response(self):
        resp = _parse_response(make_api_response())
        assert resp.message.role == "assistant"
        assert isinstance(resp.message.content[0], TextContent)
        assert resp.message.content[0].text == "Hello!"

    def test_tool_use_response(self):
        data = make_api_response(content=[
            {"type": "text", "text": "Let me search."},
            {"type": "tool_use", "id": "tc_1", "name": "search", "input": {"q": "test"}},
        ])
        resp = _parse_response(data)
        assert isinstance(resp.message.content[1], ToolCallContent)
        assert resp.message.content[1].arguments == {"q": "test"}

    def test_usage_parsing(self):
        usage = _parse_usage({
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 10,
        })
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_tokens == 20
        assert usage.cache_write_tokens == 10


# ── Event-stream decoding ────────────────────────


class TestEventStreamDecoding:
    def test_parse_frame_headers(self):
        frame = make_chunk_frame({"type": "message_stop"})
        headers_len = int.from_bytes(frame[4:8], "big")
        headers = _parse_frame_headers(frame[12:12 + headers_len])
        assert headers[":message-type"] == "event"
        assert headers[":event-type"] == "chunk"

    def test_decode_chunk_frame(self):
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}
        decoded = _decode_frame(make_chunk_frame(event))
        assert decoded == [event]

    def test_decode_non_event_frame_ignored(self):
        payload = json.dumps({"bytes": ""}).encode()
        frame = make_frame(payload, {":message-type": "something-else"})
        assert _decode_frame(frame) == []

    async def test_iter_eventstream_reassembles_split_frames(self):
        frame = make_chunk_frame({"type": "message_stop"})
        # Split the frame across chunk boundaries to exercise buffering.
        resp = FakeStreamResponse(200, [frame[:3], frame[3:10], frame[10:]])
        events = [e async for e in _iter_eventstream(resp)]  # type: ignore[arg-type]
        assert events == [{"type": "message_stop"}]

    async def test_iter_eventstream_multiple_frames_in_one_chunk(self):
        f1 = make_chunk_frame({"type": "a"})
        f2 = make_chunk_frame({"type": "b"})
        resp = FakeStreamResponse(200, [f1 + f2])
        events = [e async for e in _iter_eventstream(resp)]  # type: ignore[arg-type]
        assert events == [{"type": "a"}, {"type": "b"}]

    def test_exception_frame_raises_api_error(self):
        payload = json.dumps({"message": "boom"}).encode()
        frame = make_frame(payload, {
            ":message-type": "exception",
            ":exception-type": "internalServerException",
        })
        with pytest.raises(APIError):
            _decode_frame(frame)

    def test_throttling_exception_frame_raises_rate_limit(self):
        payload = json.dumps({"message": "slow down"}).encode()
        frame = make_frame(payload, {
            ":message-type": "exception",
            ":exception-type": "throttlingException",
        })
        with pytest.raises(RateLimitError):
            _decode_frame(frame)


# ── Error handling ────────────────────────────────


class TestErrorMapping:
    async def test_auth_error(self):
        provider = BedrockProvider(api_key="bad-key", base_url=_ENDPOINT)
        mock_resp = httpx.Response(403, request=httpx.Request("POST", "http://test"))
        with patch.object(provider._client, "post", return_value=mock_resp):
            with pytest.raises(AuthError):
                await provider.generate(make_request())

    async def test_rate_limit_error(self):
        provider = BedrockProvider(api_key="key", base_url=_ENDPOINT)
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
        provider = BedrockProvider(api_key="key", base_url=_ENDPOINT)
        mock_resp = httpx.Response(500, request=httpx.Request("POST", "http://test"), text="boom")
        with patch.object(provider._client, "post", return_value=mock_resp):
            with pytest.raises(APIError) as exc_info:
                await provider.generate(make_request())
            assert exc_info.value.status_code == 500
            assert exc_info.value.provider == "bedrock"

    async def test_network_error_on_connect(self):
        provider = BedrockProvider(api_key="key", base_url=_ENDPOINT)
        with patch.object(provider._client, "post", side_effect=httpx.ConnectError("failed")):
            with pytest.raises(NetworkError):
                await provider.generate(make_request())


# ── Generate / stream (mocked HTTP) ──────────────


class TestGenerate:
    async def test_bearer_auth_header(self):
        provider = BedrockProvider(api_key="secret-token", base_url=_ENDPOINT)
        assert provider._client.headers["authorization"] == "Bearer secret-token"

    async def test_generate_returns_response(self):
        provider = BedrockProvider(api_key="test-key", base_url=_ENDPOINT)
        mock_resp = httpx.Response(
            200,
            request=httpx.Request("POST", "http://test"),
            json=make_api_response(),
        )
        with patch.object(provider._client, "post", return_value=mock_resp):
            result = await provider.generate(make_request())
            assert isinstance(result.message.content[0], TextContent)
            assert result.message.content[0].text == "Hello!"
            assert result.usage.input_tokens == 10

    async def test_generate_unwraps_schema(self):
        provider = BedrockProvider(api_key="key", base_url=_ENDPOINT)
        data = make_api_response(content=[
            {"type": "tool_use", "id": "t1", "name": "_structured_output", "input": {"x": 1}},
        ])
        mock_resp = httpx.Response(200, request=httpx.Request("POST", "http://test"), json=data)
        with patch.object(provider._client, "post", return_value=mock_resp):
            result = await provider.generate(make_request(schema={"type": "object"}))
            assert isinstance(result.message.content[0], TextContent)
            assert json.loads(result.message.content[0].text) == {"x": 1}

    async def test_stream_yields_text_and_final_response(self):
        provider = BedrockProvider(api_key="key", base_url=_ENDPOINT)
        frames = [
            make_chunk_frame({"type": "message_start", "message": {"usage": {"input_tokens": 8, "output_tokens": 0}}}),
            make_chunk_frame({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            make_chunk_frame({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
            make_chunk_frame({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}}),
            make_chunk_frame({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}),
            make_chunk_frame({"type": "message_stop"}),
        ]
        fake = FakeStreamCtx(FakeStreamResponse(200, frames))
        with patch.object(provider._client, "stream", return_value=fake):
            result = provider.stream(make_request())
            texts = [e.text async for e in result.events if e.type == "text_delta"]
            response = await result.response

        assert "".join(texts) == "Hello world"
        assert isinstance(response.message.content[0], TextContent)
        assert response.message.content[0].text == "Hello world"
        assert response.usage.input_tokens == 8
        assert response.usage.output_tokens == 3

    async def test_stream_error_status_raises(self):
        provider = BedrockProvider(api_key="key", base_url=_ENDPOINT)
        fake = FakeStreamCtx(FakeStreamResponse(403, []))
        with patch.object(provider._client, "stream", return_value=fake):
            result = provider.stream(make_request())
            with pytest.raises(AuthError):
                async for _ in result.events:
                    pass
            # The failure must also settle the response future, not hang forever.
            with pytest.raises(AuthError):
                await result.response


# ── count_tokens ─────────────────────────────────


class TestCountTokens:
    async def test_estimate_is_inexact(self):
        provider = BedrockProvider(api_key="key", base_url=_ENDPOINT)
        count = await provider.count_tokens(make_request())
        assert count.exact is False
        assert count.tokens > 0
