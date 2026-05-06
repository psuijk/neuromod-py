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
    ToolResultContent,
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


_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider:
    """Provider implementation for the Google Gemini API using raw httpx."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url or _BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            params={"key": self._api_key},
            headers={"content-type": "application/json"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    # -- Public API (satisfies Provider protocol) ------

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        body = _build_body(request)
        path = f"/models/{request.model.id}:generateContent"
        data = await self._post(path, body, timeout=request.timeout)
        return _parse_response(data)

    def stream(self, request: ProviderRequest) -> ProviderStreamResult:
        body = _build_body(request)
        path = f"/models/{request.model.id}:streamGenerateContent"
        response_future: _ResponseFuture = _ResponseFuture()
        timeout = request.timeout

        async def events() -> AsyncGenerator[ProviderStreamEvent, None]:
            stream_kwargs: dict[str, Any] = {
                "json": body, "params": {"key": self._api_key, "alt": "sse"},
            }
            if timeout is not None:
                stream_kwargs["timeout"] = timeout
            async with self._client.stream(
                "POST", path, **stream_kwargs,
            ) as http_resp:
                _check_status(http_resp)
                async for event in _parse_sse_stream(http_resp, response_future):
                    yield event

        return ProviderStreamResult(
            events=events(),
            response=response_future.wait(),
        )

    async def count_tokens(self, request: ProviderRequest) -> TokenCount:
        body = {"contents": _convert_messages(request.messages)}
        if request.system:
            body["systemInstruction"] = {"parts": [{"text": request.system}]}
        path = f"/models/{request.model.id}:countTokens"
        data = await self._post(path, body, timeout=request.timeout)
        return TokenCount(tokens=data.get("totalTokens", 0), exact=True)

    # -- HTTP helpers ----------------------------------

    async def _post(self, path: str, body: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"json": body}
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            resp = await self._client.post(path, **kwargs)
        except httpx.ConnectError as e:
            raise NetworkError("google", cause=e) from e
        except httpx.TimeoutException as e:
            raise NetworkError("google", cause=e) from e

        _check_status(resp)
        return resp.json()


# -- Request building ----------------------------------


def _build_body(request: ProviderRequest) -> dict[str, Any]:
    contents = _convert_messages(request.messages)
    body: dict[str, Any] = {"contents": contents}

    if request.system:
        body["systemInstruction"] = {"parts": [{"text": request.system}]}

    generation_config: dict[str, Any] = {
        "maxOutputTokens": request.model.max_output_tokens,
    }
    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    body["generationConfig"] = generation_config

    if request.tools:
        body["tools"] = [{"functionDeclarations": [_convert_tool_def(t) for t in request.tools]}]

    if request.tool_choice:
        body["toolConfig"] = {
            "functionCallingConfig": {"mode": _convert_tool_choice(request.tool_choice)},
        }

    return body


def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            continue  # system handled separately via systemInstruction

        role = "model" if msg.role == "assistant" else "user"
        parts = _convert_parts(msg.content)
        if parts:
            result.append({"role": role, "parts": parts})

    return result


def _convert_parts(content: list[Content]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []

    for c in content:
        if isinstance(c, TextContent):
            parts.append({"text": c.text})

        elif isinstance(c, MediaContent):
            parts.append({
                "inlineData": {
                    "mimeType": c.mime_type,
                    "data": c.data,
                },
            })

        elif isinstance(c, ToolCallContent):
            parts.append({
                "functionCall": {
                    "name": c.name,
                    "args": c.arguments,
                },
            })

        elif isinstance(c, ToolResultContent):
            parts.append({
                "functionResponse": {
                    "name": c.name or "",
                    "response": {"result": c.result},
                },
            })

    return parts


def _convert_tool_def(tool: ToolDefinition) -> dict[str, Any]:
    schema = dict(tool.parameters)
    schema.pop("$schema", None)
    schema.pop("additionalProperties", None)
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": schema,
    }


def _convert_tool_choice(choice: str) -> str:
    if choice == "required":
        return "ANY"
    if choice == "none":
        return "NONE"
    return "AUTO"


# -- Response parsing ----------------------------------


def _parse_response(data: dict[str, Any]) -> ProviderResponse:
    candidate = (data.get("candidates") or [{}])[0]
    message = _parse_message(candidate.get("content", {}))
    usage = _parse_usage(data.get("usageMetadata", {}))
    return ProviderResponse(message=message, usage=usage)


def _parse_message(content_data: dict[str, Any]) -> Message:
    content: list[Content] = []
    call_index = 0

    for part in content_data.get("parts", []):
        if "text" in part:
            content.append(TextContent(text=part["text"]))

        elif "functionCall" in part:
            fc = part["functionCall"]
            content.append(ToolCallContent(
                id=f"call_{call_index}",
                name=fc.get("name", ""),
                arguments=fc.get("args", {}),
            ))
            call_index += 1

    return Message(role="assistant", content=content)


def _parse_usage(usage: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        input_tokens=usage.get("promptTokenCount", 0),
        output_tokens=usage.get("candidatesTokenCount", 0),
        cache_read_tokens=usage.get("cachedContentTokenCount") or None,
    )


# -- SSE streaming -------------------------------------


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
    """Parse Gemini SSE events from an httpx streaming response."""

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []  # [{name, arguments}]
    usage = TokenUsage(input_tokens=0, output_tokens=0)
    call_index = 0

    try:
        async for event_data in _iter_sse_events(http_resp):
            candidate = (event_data.get("candidates") or [{}])[0]
            content = candidate.get("content", {})

            for part in content.get("parts", []):
                if "text" in part:
                    chunk = part["text"]
                    text_parts.append(chunk)
                    yield TextDeltaEvent(text=chunk)

                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_id = f"call_{call_index}"
                    tool_name = fc.get("name", "")
                    tool_args = fc.get("args", {})
                    tool_calls.append({"id": tool_id, "name": tool_name, "arguments": tool_args})
                    call_index += 1

                    yield ToolCallStartEvent(id=tool_id, name=tool_name)
                    args_json = json.dumps(tool_args)
                    yield ToolCallDeltaEvent(id=tool_id, arguments_delta=args_json)

            chunk_usage = event_data.get("usageMetadata")
            if chunk_usage:
                usage = _parse_usage(chunk_usage)

        # Build final message.
        content_parts: list[Content] = []
        if text_parts:
            content_parts.append(TextContent(text="".join(text_parts)))

        parsed_calls: list[ToolCallInfo] = []
        for tc in tool_calls:
            content_parts.append(ToolCallContent(id=tc["id"], name=tc["name"], arguments=tc["arguments"]))
            parsed_calls.append(ToolCallInfo(id=tc["id"], name=tc["name"], arguments=tc["arguments"]))

        if parsed_calls:
            yield ToolCallsReadyEvent(calls=parsed_calls)

        message = Message(role="assistant", content=content_parts)
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


# -- Error handling ------------------------------------


def _check_status(resp: httpx.Response) -> None:
    if resp.status_code >= 200 and resp.status_code < 300:
        return

    if resp.status_code in (401, 403):
        raise AuthError("google")

    if resp.status_code == 429:
        retry_after = resp.headers.get("retry-after")
        retry_ms = int(float(retry_after) * 1000) if retry_after else None
        raise RateLimitError("google", retry_after_ms=retry_ms)

    try:
        body = resp.text
    except httpx.ResponseNotRead:
        body = ""
    raise APIError("google", resp.status_code, body)
