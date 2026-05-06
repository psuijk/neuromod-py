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


_DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaProvider:
    """Provider implementation for Ollama's OpenAI-compatible API using raw httpx."""

    def __init__(self, api_key: str = "", base_url: str | None = None) -> None:
        self._base_url = base_url or _DEFAULT_BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"content-type": "application/json"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    # -- Public API (satisfies Provider protocol) ------

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        body = _build_body(request, stream=False)
        data = await self._post("/chat/completions", body, timeout=request.timeout)
        return _parse_response(data)

    def stream(self, request: ProviderRequest) -> ProviderStreamResult:
        body = _build_body(request, stream=True)
        response_future: _ResponseFuture = _ResponseFuture()
        timeout = request.timeout

        async def events() -> AsyncGenerator[ProviderStreamEvent, None]:
            async with self._client.stream("POST", "/chat/completions", json=body, timeout=timeout) as http_resp:
                _check_status(http_resp)
                async for event in _parse_sse_stream(http_resp, response_future):
                    yield event

        return ProviderStreamResult(
            events=events(),
            response=response_future.wait(),
        )

    async def count_tokens(self, request: ProviderRequest) -> TokenCount:
        # Ollama does not have a dedicated token counting endpoint.
        return TokenCount(tokens=0, exact=False)

    # -- HTTP helpers ----------------------------------

    async def _post(self, path: str, body: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"json": body}
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            resp = await self._client.post(path, **kwargs)
        except httpx.ConnectError as e:
            raise NetworkError("ollama", cause=e) from e
        except httpx.TimeoutException as e:
            raise NetworkError("ollama", cause=e) from e

        _check_status(resp)
        return resp.json()


# -- Request building ----------------------------------


def _build_body(request: ProviderRequest, *, stream: bool) -> dict[str, Any]:
    messages = _convert_messages(request.messages, system=request.system)
    body: dict[str, Any] = {
        "model": request.model.id,
        "messages": messages,
        "max_tokens": request.model.max_output_tokens,
    }

    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

    if request.tools:
        body["tools"] = [_convert_tool_def(t) for t in request.tools]

    if request.tool_choice:
        body["tool_choice"] = request.tool_choice

    if request.temperature is not None:
        body["temperature"] = request.temperature

    return body


def _convert_messages(
    messages: list[Message], *, system: str | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        if msg.role == "system":
            result.append({"role": "system", "content": _extract_text(msg)})
            continue

        if msg.role == "user":
            # Tool results become separate role:tool messages.
            tool_results = [c for c in msg.content if isinstance(c, ToolResultContent)]
            other_content = [c for c in msg.content if not isinstance(c, ToolResultContent)]

            if other_content:
                result.append({"role": "user", "content": _extract_text_from_content(other_content)})

            for tr in tool_results:
                result.append({
                    "role": "tool",
                    "tool_call_id": tr.call_id,
                    "content": tr.result,
                })
            continue

        # Assistant message: may contain text and/or tool calls.
        text_parts = [c for c in msg.content if isinstance(c, TextContent)]
        tool_calls = [c for c in msg.content if isinstance(c, ToolCallContent)]

        entry: dict[str, Any] = {"role": "assistant"}
        entry["content"] = "".join(t.text for t in text_parts) if text_parts else None

        if tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ]

        result.append(entry)

    return result


def _extract_text(msg: Message) -> str:
    return "".join(c.text for c in msg.content if isinstance(c, TextContent))


def _extract_text_from_content(content: list[Content]) -> str:
    return "".join(c.text for c in content if isinstance(c, TextContent))


def _convert_tool_def(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


# -- Response parsing ----------------------------------


def _parse_response(data: dict[str, Any]) -> ProviderResponse:
    choice = data.get("choices", [{}])[0]
    message = _parse_message(choice.get("message", {}))
    usage = _parse_usage(data.get("usage", {}))
    return ProviderResponse(message=message, usage=usage)


def _parse_message(msg_data: dict[str, Any]) -> Message:
    content: list[Content] = []

    text = msg_data.get("content")
    if text:
        content.append(TextContent(text=text))

    for tc in msg_data.get("tool_calls", []):
        func = tc.get("function", {})
        arguments_raw = func.get("arguments", "{}")
        arguments = json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw
        content.append(ToolCallContent(
            id=tc.get("id", ""),
            name=func.get("name", ""),
            arguments=arguments,
        ))

    return Message(role="assistant", content=content)


def _parse_usage(usage: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
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
    """Parse OpenAI-format SSE events from an httpx streaming response."""

    text_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}  # index -> {id, name, arguments_json}
    usage = TokenUsage(input_tokens=0, output_tokens=0)

    try:
        async for event_data in _iter_sse_events(http_resp):
            choices = event_data.get("choices", [])
            if not choices:
                # Final chunk may have usage but no choices.
                chunk_usage = event_data.get("usage")
                if chunk_usage:
                    usage = _parse_usage(chunk_usage)
                continue

            delta = choices[0].get("delta", {})

            # Text delta
            text_chunk = delta.get("content")
            if text_chunk:
                text_parts.append(text_chunk)
                yield TextDeltaEvent(text=text_chunk)

            # Tool call deltas
            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta.get("index", 0)
                func = tc_delta.get("function", {})

                if idx not in tool_calls:
                    # First chunk for this tool call.
                    tool_id = tc_delta.get("id", f"call_{idx}")
                    tool_name = func.get("name", "")
                    tool_calls[idx] = {"id": tool_id, "name": tool_name, "arguments_json": ""}
                    yield ToolCallStartEvent(id=tool_id, name=tool_name)

                arg_delta = func.get("arguments", "")
                if arg_delta:
                    tool_calls[idx]["arguments_json"] += arg_delta
                    yield ToolCallDeltaEvent(id=tool_calls[idx]["id"], arguments_delta=arg_delta)

            # Usage in final chunk
            chunk_usage = event_data.get("usage")
            if chunk_usage:
                usage = _parse_usage(chunk_usage)

        # Build final message.
        content: list[Content] = []
        if text_parts:
            content.append(TextContent(text="".join(text_parts)))

        parsed_calls: list[ToolCallInfo] = []
        for _idx in sorted(tool_calls):
            info = tool_calls[_idx]
            args: dict[str, Any] = json.loads(info["arguments_json"]) if info["arguments_json"] else {}
            content.append(ToolCallContent(id=info["id"], name=info["name"], arguments=args))
            parsed_calls.append(ToolCallInfo(id=info["id"], name=info["name"], arguments=args))

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


# -- Error handling ------------------------------------


def _check_status(resp: httpx.Response) -> None:
    if resp.status_code >= 200 and resp.status_code < 300:
        return

    if resp.status_code in (401, 403):
        raise AuthError("ollama")

    if resp.status_code == 429:
        retry_after = resp.headers.get("retry-after")
        retry_ms = int(float(retry_after) * 1000) if retry_after else None
        raise RateLimitError("ollama", retry_after_ms=retry_ms)

    try:
        body = resp.text
    except httpx.ResponseNotRead:
        body = ""
    raise APIError("ollama", resp.status_code, body)
