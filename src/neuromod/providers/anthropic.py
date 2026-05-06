from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import httpx

from neuromod.messages.types import (
    Content,
    MediaContent,
    Message,
    TextContent,
    ToolCallContent,
)
from neuromod.providers.errors import (
    APIError,
    AuthError,
    NetworkError,
    RateLimitError,
)
from neuromod.providers.provider import (
    ProviderRequest,
    ProviderResponse,
    ProviderStreamEvent,
    ProviderStreamResult,
    TextDeltaEvent,
    TokenCount,
    TokenUsage,
    ToolCallDeltaEvent,
    ToolCallInfo,
    ToolCallsReadyEvent,
    ToolCallStartEvent,
    ToolDefinition,
)


_BASE_URL = "https://api.anthropic.com/v1"
_API_VERSION = "2023-06-01"
_SCHEMA_TOOL_NAME = "_structured_output"


class ClaudeProvider:
    """Provider implementation for the Anthropic Messages API using raw httpx."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url or _BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "content-type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": _API_VERSION,
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    # ── Public API (satisfies Provider protocol) ──────

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        body = _build_body(request, stream=False)
        data = await self._post("/messages", body)
        response = _parse_response(data)
        if request.schema:
            response = _unwrap_schema_tool(response)
        return response

    def stream(self, request: ProviderRequest) -> ProviderStreamResult:
        body = _build_body(request, stream=True)
        response_future: _ResponseFuture = _ResponseFuture()
        has_schema = request.schema is not None

        async def events() -> AsyncGenerator[ProviderStreamEvent, None]:
            schema_tool_id: str | None = None
            async with self._client.stream("POST", "/messages", json=body) as http_resp:
                _check_status(http_resp)
                async for event in _parse_sse_stream(http_resp, response_future):
                    if not has_schema:
                        yield event
                        continue

                    if isinstance(event, ToolCallStartEvent) and event.name == _SCHEMA_TOOL_NAME:
                        schema_tool_id = event.id
                        continue

                    if isinstance(event, ToolCallDeltaEvent) and event.id == schema_tool_id:
                        yield TextDeltaEvent(text=event.arguments_delta)
                        continue

                    if isinstance(event, ToolCallsReadyEvent):
                        remaining = [c for c in event.calls if c.name != _SCHEMA_TOOL_NAME]
                        if remaining:
                            yield ToolCallsReadyEvent(calls=remaining)
                        continue

                    yield event

        async def response() -> ProviderResponse:
            resp = await response_future.wait()
            if has_schema:
                return _unwrap_schema_tool(resp)
            return resp

        return ProviderStreamResult(
            events=events(),
            response=response(),
        )

    async def count_tokens(self, request: ProviderRequest) -> TokenCount:
        body = _build_body(request, stream=False)
        body.pop("stream", None)
        data = await self._post("/messages/count_tokens", body)
        return TokenCount(tokens=data["input_tokens"], exact=True)

    # ── HTTP helpers ──────────────────────────────────

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._client.post(path, json=body)
        except httpx.ConnectError as e:
            raise NetworkError("anthropic", cause=e) from e
        except httpx.TimeoutException as e:
            raise NetworkError("anthropic", cause=e) from e

        _check_status(resp)
        return resp.json()


# ── Request building ──────────────────────────────


def _build_body(request: ProviderRequest, *, stream: bool) -> dict[str, Any]:
    messages = _convert_messages(request.messages)
    body: dict[str, Any] = {
        "model": request.model.id,
        "max_tokens": request.model.max_output_tokens,
        "messages": messages,
    }

    if stream:
        body["stream"] = True

    if request.system:
        body["system"] = request.system

    if request.tools:
        body["tools"] = [_convert_tool_def(t) for t in request.tools]

    if request.tool_choice:
        body["tool_choice"] = _convert_tool_choice(request.tool_choice)

    if request.temperature is not None:
        body["temperature"] = request.temperature

    if request.schema:
        schema_tool = {
            "name": _SCHEMA_TOOL_NAME,
            "description": "Return the structured response.",
            "input_schema": request.schema,
        }
        if "tools" in body:
            body["tools"].append(schema_tool)
        else:
            body["tools"] = [schema_tool]
        body["tool_choice"] = {"type": "tool", "name": _SCHEMA_TOOL_NAME}

    return body


def _unwrap_schema_tool(response: ProviderResponse) -> ProviderResponse:
    """Convert a _structured_output tool call back to JSON text content."""
    new_content: list[Content] = []
    for c in response.message.content:
        if isinstance(c, ToolCallContent) and c.name == _SCHEMA_TOOL_NAME:
            new_content.append(TextContent(text=json.dumps(c.arguments)))
        else:
            new_content.append(c)
    return ProviderResponse(
        message=Message(role="assistant", content=new_content),
        usage=response.usage,
    )


def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            continue  # system prompt handled separately
        result.append({
            "role": msg.role,
            "content": [_convert_content(c) for c in msg.content],
        })
    return result


def _convert_content(content: Content) -> dict[str, Any]:
    if isinstance(content, TextContent):
        return {"type": "text", "text": content.text}

    if isinstance(content, MediaContent):
        if content.mime_type.startswith("image/"):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": content.mime_type,
                    "data": content.data,
                },
            }
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": content.mime_type,
                "data": content.data,
            },
        }

    if isinstance(content, ToolCallContent):
        return {
            "type": "tool_use",
            "id": content.id,
            "name": content.name,
            "input": content.arguments,
        }

    return {
        "type": "tool_result",
        "tool_use_id": content.call_id,
        "content": content.result,
        "is_error": content.is_error,
    }


def _convert_tool_def(tool: ToolDefinition) -> dict[str, Any]:
    schema = dict(tool.parameters)
    schema.pop("$schema", None)
    schema.pop("additionalProperties", None)
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": schema,
    }


def _convert_tool_choice(choice: str) -> dict[str, str]:
    if choice == "required":
        return {"type": "any"}
    if choice == "none":
        return {"type": "none"}
    return {"type": "auto"}


# ── Response parsing ──────────────────────────────


def _parse_response(data: dict[str, Any]) -> ProviderResponse:
    message = _parse_message(data)
    usage = _parse_usage(data.get("usage", {}))
    return ProviderResponse(message=message, usage=usage)


def _parse_message(data: dict[str, Any]) -> Message:
    content: list[Content] = []
    for block in data.get("content", []):
        block_type = block.get("type")
        if block_type == "text":
            content.append(TextContent(text=block["text"]))
        elif block_type == "tool_use":
            content.append(ToolCallContent(
                id=block["id"],
                name=block["name"],
                arguments=block.get("input", {}),
            ))
    return Message(role="assistant", content=content)


def _parse_usage(usage: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_tokens=usage.get("cache_read_input_tokens"),
        cache_write_tokens=usage.get("cache_creation_input_tokens"),
    )


# ── SSE streaming ─────────────────────────────────


class _ResponseFuture:
    """Simple future to pass the final response from the stream generator to the caller."""

    def __init__(self) -> None:
        import asyncio
        self._event = asyncio.Event()
        self._response: ProviderResponse | None = None
        self._error: BaseException | None = None

    def resolve(self, response: ProviderResponse) -> None:
        self._response = response
        self._event.set()

    def reject(self, error: BaseException) -> None:
        self._error = error
        self._event.set()

    async def wait(self) -> ProviderResponse:
        await self._event.wait()
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


async def _parse_sse_stream(
    http_resp: httpx.Response,
    future: _ResponseFuture,
) -> AsyncGenerator[ProviderStreamEvent, None]:
    """Parse SSE events from an httpx streaming response and yield provider events."""

    # Accumulation state
    text_parts: list[str] = []
    tool_calls: dict[str, dict[str, Any]] = {}  # id -> {name, arguments_json}
    usage = TokenUsage(input_tokens=0, output_tokens=0)

    try:
        async for event_data in _iter_sse_events(http_resp):
            event_type = event_data.get("type")

            if event_type == "message_start":
                msg_usage = event_data.get("message", {}).get("usage", {})
                usage = _parse_usage(msg_usage)

            elif event_type == "content_block_start":
                block = event_data.get("content_block", {})
                if block.get("type") == "tool_use":
                    tool_id = block["id"]
                    tool_name = block["name"]
                    tool_calls[tool_id] = {"name": tool_name, "arguments_json": ""}
                    yield ToolCallStartEvent(id=tool_id, name=tool_name)

            elif event_type == "content_block_delta":
                delta = event_data.get("delta", {})
                delta_type = delta.get("type")

                if delta_type == "text_delta":
                    chunk = delta.get("text", "")
                    text_parts.append(chunk)
                    yield TextDeltaEvent(text=chunk)

                elif delta_type == "input_json_delta":
                    partial = delta.get("partial_json", "")
                    # Find which tool call this belongs to (last started)
                    for tool_id in reversed(tool_calls):
                        tool_calls[tool_id]["arguments_json"] += partial
                        yield ToolCallDeltaEvent(id=tool_id, arguments_delta=partial)
                        break

            elif event_type == "message_delta":
                delta_usage = event_data.get("usage", {})
                if "output_tokens" in delta_usage:
                    usage = TokenUsage(
                        input_tokens=usage.input_tokens,
                        output_tokens=delta_usage["output_tokens"],
                        cache_read_tokens=usage.cache_read_tokens,
                        cache_write_tokens=usage.cache_write_tokens,
                    )

        # Build final message
        content: list[Content] = []
        if text_parts:
            content.append(TextContent(text="".join(text_parts)))

        parsed_calls: list[ToolCallInfo] = []
        for tool_id, info in tool_calls.items():
            args: dict[str, Any] = json.loads(info["arguments_json"]) if info["arguments_json"] else {}
            content.append(ToolCallContent(id=tool_id, name=info["name"], arguments=args))
            parsed_calls.append(ToolCallInfo(id=tool_id, name=info["name"], arguments=args))

        if parsed_calls:
            yield ToolCallsReadyEvent(calls=parsed_calls)

        message = Message(role="assistant", content=content)
        future.resolve(ProviderResponse(message=message, usage=usage))

    except Exception as e:
        future.reject(e)
        raise


async def _iter_sse_events(http_resp: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    """Yield parsed JSON objects from an SSE stream."""
    buffer = ""
    async for chunk in http_resp.aiter_text():
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue


# ── Error handling ────────────────────────────────


def _check_status(resp: httpx.Response) -> None:
    if resp.status_code >= 200 and resp.status_code < 300:
        return

    if resp.status_code in (401, 403):
        raise AuthError("anthropic")

    if resp.status_code == 429:
        retry_after = resp.headers.get("retry-after")
        retry_ms = int(float(retry_after) * 1000) if retry_after else None
        raise RateLimitError("anthropic", retry_after_ms=retry_ms)

    try:
        body = resp.text
    except httpx.ResponseNotRead:
        body = ""
    raise APIError("anthropic", resp.status_code, body)
