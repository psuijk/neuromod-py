from __future__ import annotations

import base64
import json
from os import environ
from typing import Any, AsyncGenerator, AsyncIterable
from urllib.parse import quote

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
    ConfigError,
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


# Bedrock hosts Anthropic Claude using the Messages format, with two twists:
# the model id lives in the URL (not the body), and the body carries
# ``anthropic_version`` instead of ``model``.
_ANTHROPIC_VERSION = "bedrock-2023-05-31"
_SCHEMA_TOOL_NAME = "_structured_output"

# Region is required (no silent default). Checked in order.
_REGION_ENV_VARS = ("BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")


def _resolve_endpoint(base_url: str | None) -> str:
    """Resolve the regional bedrock-runtime endpoint.

    Precedence: an explicit ``base_url`` (from ``configure(base_urls=...)`` or the
    per-agent override) wins; otherwise a region is read from the environment and
    used to build the host. With neither set, a ``ConfigError`` is raised rather
    than guessing a region.
    """
    if base_url:
        return base_url.rstrip("/")

    for var in _REGION_ENV_VARS:
        region = environ.get(var)
        if region:
            return f"https://bedrock-runtime.{region}.amazonaws.com"

    raise ConfigError(
        "Bedrock requires an AWS region. Set BEDROCK_REGION (or AWS_REGION), "
        "or pass a full endpoint via configure(base_urls={'bedrock': "
        "'https://bedrock-runtime.<region>.amazonaws.com'})."
    )


class BedrockProvider:
    """Provider implementation for Claude on Amazon Bedrock using raw httpx.

    Authenticates with a Bedrock API key (bearer token) — no AWS SigV4 signing
    and no boto3 dependency.
    """

    def __init__(self, api_key: str = "", base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = _resolve_endpoint(base_url)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self._api_key}",
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    # ── Public API (satisfies Provider protocol) ──────

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        body = _build_body(request)
        path = _model_path(request.model.id, "invoke")
        data = await self._post(path, body, timeout=request.timeout)
        response = _parse_response(data)
        if request.schema:
            response = _unwrap_schema_tool(response)
        return response

    def stream(self, request: ProviderRequest) -> ProviderStreamResult:
        body = _build_body(request)
        path = _model_path(request.model.id, "invoke-with-response-stream")
        response_future: _ResponseFuture = _ResponseFuture()
        has_schema = request.schema is not None
        timeout = request.timeout

        async def raw_events() -> AsyncGenerator[ProviderStreamEvent, None]:
            async with self._client.stream("POST", path, json=body, timeout=timeout) as http_resp:
                if http_resp.status_code >= 300:
                    await http_resp.aread()
                _check_status(http_resp)
                async for event in _accumulate(_iter_eventstream(http_resp), response_future):
                    yield event

        async def events() -> AsyncGenerator[ProviderStreamEvent, None]:
            schema_tool_id: str | None = None
            try:
                async for event in raw_events():
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
            except BaseException as e:
                # A failure during setup (e.g. an error HTTP status) happens before
                # _accumulate runs, so the future would otherwise never settle and a
                # caller awaiting `.response` would hang. Reject it here (idempotent —
                # _accumulate's own rejection wins if it already ran).
                response_future.reject(e)
                raise

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
        # Bedrock has no universally-available token counting endpoint; estimate.
        body = _build_body(request)
        estimated = len(json.dumps(body)) // 4
        return TokenCount(tokens=estimated, exact=False)

    # ── HTTP helpers ──────────────────────────────────

    async def _post(self, path: str, body: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"json": body}
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            resp = await self._client.post(path, **kwargs)
        except httpx.ConnectError as e:
            raise NetworkError("bedrock", cause=e) from e
        except httpx.TimeoutException as e:
            raise NetworkError("bedrock", cause=e) from e

        _check_status(resp)
        return resp.json()


def _model_path(model_id: str, action: str) -> str:
    # Model ids contain ':' (e.g. ...-v2:0), which must be percent-encoded in the path.
    return f"/model/{quote(model_id, safe='')}/{action}"


# ── Request building ──────────────────────────────


def _build_body(request: ProviderRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "anthropic_version": _ANTHROPIC_VERSION,
        "max_tokens": request.model.max_output_tokens,
        "messages": _convert_messages(request.messages),
    }

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
        body.setdefault("tools", []).append(schema_tool)
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


# ── Streaming ─────────────────────────────────────


class _ResponseFuture:
    """Simple future to pass the final response from the stream generator to the caller."""

    def __init__(self) -> None:
        import asyncio
        self._event = asyncio.Event()
        self._response: ProviderResponse | None = None
        self._error: BaseException | None = None

    def resolve(self, response: ProviderResponse) -> None:
        if self._event.is_set():
            return
        self._response = response
        self._event.set()

    def reject(self, error: BaseException) -> None:
        if self._event.is_set():
            return
        self._error = error
        self._event.set()

    async def wait(self) -> ProviderResponse:
        await self._event.wait()
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


async def _accumulate(
    events: AsyncIterable[dict[str, Any]],
    future: _ResponseFuture,
) -> AsyncGenerator[ProviderStreamEvent, None]:
    """Consume Anthropic-format streaming event dicts, yield provider events, resolve the future."""

    text_parts: list[str] = []
    tool_calls: dict[str, dict[str, Any]] = {}  # id -> {name, arguments_json}
    usage = TokenUsage(input_tokens=0, output_tokens=0)

    try:
        async for event_data in events:
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


# ── AWS event-stream decoding ─────────────────────
#
# Bedrock's streaming endpoint replies with ``application/vnd.amazon.eventstream``:
# a sequence of binary frames rather than SSE. Each frame carries a chunk whose
# payload is ``{"bytes": "<base64>"}``; decoding the base64 yields the same
# Anthropic streaming event dict a native SSE stream would emit.


async def _iter_eventstream(http_resp: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    buffer = bytearray()
    async for chunk in http_resp.aiter_bytes():
        buffer.extend(chunk)
        while len(buffer) >= 4:
            total_len = int.from_bytes(buffer[0:4], "big")
            if total_len < 16 or len(buffer) < total_len:
                break
            frame = bytes(buffer[:total_len])
            del buffer[:total_len]
            for event in _decode_frame(frame):
                yield event


def _decode_frame(frame: bytes) -> list[dict[str, Any]]:
    total_len = int.from_bytes(frame[0:4], "big")
    headers_len = int.from_bytes(frame[4:8], "big")
    headers = _parse_frame_headers(frame[12:12 + headers_len])
    payload = frame[12 + headers_len:total_len - 4]

    message_type = headers.get(":message-type", "event")
    if message_type == "exception" or ":exception-type" in headers:
        _raise_stream_error(headers, payload)

    if message_type != "event":
        return []

    try:
        outer = json.loads(payload)
    except json.JSONDecodeError:
        return []

    encoded = outer.get("bytes")
    if not encoded:
        return []

    try:
        inner = json.loads(base64.b64decode(encoded))
    except (ValueError, json.JSONDecodeError):
        return []

    return [inner]


def _parse_frame_headers(data: bytes) -> dict[str, str]:
    """Parse event-stream frame headers. Only string-valued headers are retained
    (all Bedrock control headers are strings); other value types are skipped."""
    headers: dict[str, str] = {}
    i = 0
    n = len(data)
    while i < n:
        name_len = data[i]
        i += 1
        name = data[i:i + name_len].decode("utf-8", "replace")
        i += name_len
        value_type = data[i]
        i += 1
        if value_type == 7:  # string
            vlen = int.from_bytes(data[i:i + 2], "big")
            i += 2
            headers[name] = data[i:i + vlen].decode("utf-8", "replace")
            i += vlen
        elif value_type == 6:  # byte array
            vlen = int.from_bytes(data[i:i + 2], "big")
            i += 2 + vlen
        elif value_type in (0, 1):  # bool true / false
            pass
        elif value_type == 2:  # byte
            i += 1
        elif value_type == 3:  # short
            i += 2
        elif value_type == 4:  # int
            i += 4
        elif value_type in (5, 8):  # long / timestamp
            i += 8
        elif value_type == 9:  # uuid
            i += 16
        else:  # unknown type — stop to avoid a misaligned read
            break
    return headers


def _raise_stream_error(headers: dict[str, str], payload: bytes) -> None:
    exc_type = headers.get(":exception-type", "")
    body = payload.decode("utf-8", "replace")
    if "throttl" in exc_type.lower():
        raise RateLimitError("bedrock")
    raise APIError("bedrock", 0, body or exc_type or "bedrock stream error")


# ── Error handling ────────────────────────────────


def _check_status(resp: httpx.Response) -> None:
    if resp.status_code >= 200 and resp.status_code < 300:
        return

    if resp.status_code in (401, 403):
        raise AuthError("bedrock")

    if resp.status_code == 429:
        retry_after = resp.headers.get("retry-after")
        retry_ms = int(float(retry_after) * 1000) if retry_after else None
        raise RateLimitError("bedrock", retry_after_ms=retry_ms)

    try:
        body = resp.text
    except httpx.ResponseNotRead:
        body = ""
    raise APIError("bedrock", resp.status_code, body)
