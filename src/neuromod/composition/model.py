import asyncio
import time
from typing import Callable
from pydantic import BaseModel
from neuromod import config
from neuromod.composition.context import ConversationContext, StepFunction, ToolApprovalRequest
from neuromod.messages.types import Content, Message, ToolCallContent, ToolResultContent
from neuromod.models.model import Model
from neuromod.providers.provider import ProviderRequest, ProviderStreamEvent, TokenUsage, ToolChoice
from neuromod.streaming.events import (
    StepCompleteStreamEvent,
    StepResult,
    StepStartStreamEvent,
    StreamEvent,
    TextDeltaStreamEvent,
    ToolApprovalDeniedStreamEvent,
    ToolApprovalPendingStreamEvent,
    ToolCallDeltaStreamEvent,
    ToolCallStartStreamEvent,
    ToolCallsReadyStreamEvent,
    ToolCompleteStreamEvent,
    ToolExecutingStreamEvent,
)
from neuromod.tools.tool import Tool, convert_tools


def model(
        *,
        model: Model,
        system: str | Callable[[ConversationContext], str] | None = None,
        temperature: float | None = None,
        schema: type[BaseModel] | None = None,
        max_steps: int = 10,
        tool_choice: ToolChoice | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
) -> StepFunction:

    async def run(ctx: ConversationContext) -> ConversationContext:
        resolved_key = config.resolve_api_key(model.provider, api_key)
        resolved_timeout = config.resolve_timeout(model.provider, timeout)
        factory = config.get_factory()
        provider = factory.get(model.provider, api_key=resolved_key, base_url=base_url)
        resolved_sys = system(ctx) if callable(system) else system
        tool_defs = convert_tools(ctx.tools)
        json_schema = schema.model_json_schema() if schema else None
        total_usage = ctx.usage if ctx.usage is not None else TokenUsage(input_tokens=0, output_tokens=0)
        tool_call_counts: dict[str, int] = {}

        for step_number in range(1, max_steps + 1):
            if ctx.signal is not None and ctx.signal.is_set():
                return ctx.with_updates(stop_reason="aborted")

            step_start_time = time.monotonic()

            request = ProviderRequest(
                model=model,
                messages=ctx.messages,
                tools=tool_defs,
                system=resolved_sys,
                temperature=temperature,
                tool_choice=tool_choice,
                schema=json_schema,
                signal=ctx.signal,
                timeout=resolved_timeout,
            )

            if ctx.on_event:
                ctx.on_event(StepStartStreamEvent(step_number=step_number))
                stream_result = provider.stream(request)
                async for event in stream_result.events:
                    stream_event = _map_provider_event(event, step_number)
                    if stream_event:
                        ctx.on_event(stream_event)
                response = await stream_result.response
            else:
                response = await provider.generate(request)

            total_usage = TokenUsage(
                input_tokens=total_usage.input_tokens + response.usage.input_tokens,
                output_tokens=total_usage.output_tokens + response.usage.output_tokens,
                cache_read_tokens=(total_usage.cache_read_tokens or 0) + (response.usage.cache_read_tokens or 0) or None,
                cache_write_tokens=(total_usage.cache_write_tokens or 0) + (response.usage.cache_write_tokens or 0) or None,
            )

            ctx = ctx.with_updates(
                messages=[*ctx.messages, response.message],
                usage=total_usage,
            )

            tool_calls = response.message.tool_calls
            if not tool_calls:
                if ctx.on_event:
                    ctx.on_event(StepCompleteStreamEvent(
                        step_number=step_number,
                        step=StepResult(
                            step_number=step_number,
                            tool_calls=[],
                            usage=response.usage,
                            duration_ms=int((time.monotonic() - step_start_time) * 1000),
                        ),
                    ))
                return ctx.with_updates(stop_reason="stop")

            results = await execute_tools(tool_calls, ctx.tools or [], tool_call_counts, ctx, step_number)
            content: list[Content] = list(results)
            ctx = ctx.with_updates(messages=[*ctx.messages, Message(role="user", content=content)])

            if ctx.on_event:
                ctx.on_event(StepCompleteStreamEvent(
                    step_number=step_number,
                    step=StepResult(
                        step_number=step_number,
                        tool_calls=tool_calls,
                        usage=response.usage,
                        duration_ms=int((time.monotonic() - step_start_time) * 1000),
                    ),
                ))

        return ctx.with_updates(stop_reason="max_steps")

    return run

async def execute_tools(
        tool_calls: list[ToolCallContent],
        tools: list[Tool], call_counts: dict[str, int],
        ctx: ConversationContext,
        step_number: int,
) -> list[ToolResultContent]:
    tool_map = {t.name: t for t in (tools or [])}

    async def execute_one(call: ToolCallContent) -> ToolResultContent:
        tool = tool_map.get(call.name)

        if tool is None:
            if ctx.on_event:
                ctx.on_event(ToolCompleteStreamEvent(
                    name=call.name, id=call.id, result=f"Unknown tool: {call.name}",
                    is_error=True, duration_ms=0, step_number=step_number,
                ))
            return ToolResultContent(
                call_id=call.id,
                result=f"Unknown tool: {call.name}",
                is_error=True)

        count = call_counts.get(call.name, 0) + 1
        call_counts[call.name] = count
        if tool.max_calls is not None and count > tool.max_calls:
            if ctx.on_event:
                ctx.on_event(ToolCompleteStreamEvent(
                    name=call.name, id=call.id, result="Tool exceeded max calls",
                    is_error=True, duration_ms=0, step_number=step_number,
                ))
            return ToolResultContent(
                call_id=call.id,
                result="Tool exceeded max calls",
                is_error=True)

        if tool.requires_approval and ctx.tool_approval:
            if ctx.on_event:
                ctx.on_event(ToolApprovalPendingStreamEvent(
                    name=call.name, id=call.id, step_number=step_number,
                ))
            approved = await ctx.tool_approval(ToolApprovalRequest(id=call.id, name=call.name, arguments=call.arguments))

            if not approved:
                if ctx.on_event:
                    ctx.on_event(ToolApprovalDeniedStreamEvent(
                        name=call.name, id=call.id, step_number=step_number,
                    ))
                return ToolResultContent(call_id=call.id, result="Approval denied", is_error=True)

        if ctx.on_event:
            ctx.on_event(ToolExecutingStreamEvent(
                name=call.name, id=call.id, step_number=step_number,
            ))

        start = time.monotonic()
        max_retries = tool.retry or 0
        last_error = None
        is_error = False
        result_text = ""
        for _ in range(max_retries + 1):
            try:
                parsed = tool.schema(**call.arguments)
                result = await tool.execute(parsed)
                result_text = result
                is_error = False
                last_error = None
                break
            except Exception as e:
                last_error = e

        duration = int((time.monotonic() - start) * 1000)

        if last_error is not None:
            result_text = str(last_error)
            is_error = True

        if ctx.on_event:
            ctx.on_event(ToolCompleteStreamEvent(
                name=call.name, id=call.id, result=result_text,
                is_error=is_error, duration_ms=duration, step_number=step_number,
            ))

        if is_error:
            return ToolResultContent(call_id=call.id, result=result_text, name=call.name, is_error=True)
        return ToolResultContent(call_id=call.id, result=result_text, name=call.name)

    results = await asyncio.gather(*[execute_one(call) for call in tool_calls])
    return list(results)

def _map_provider_event(event: ProviderStreamEvent, step_number: int) -> StreamEvent | None:
    if event.type == "text_delta":
        return TextDeltaStreamEvent(text=event.text, step_number=step_number)
    elif event.type == "tool_call_start":
        return ToolCallStartStreamEvent(id=event.id, name=event.name, step_number=step_number)
    elif event.type == "tool_call_delta":
        return ToolCallDeltaStreamEvent(id=event.id, arguments_delta=event.arguments_delta,
step_number=step_number)
    elif event.type == "tool_calls_ready":
        calls = [ToolCallContent(id=c.id, name=c.name, arguments=c.arguments) for c in
event.calls]
        return ToolCallsReadyStreamEvent(calls=calls, step_number=step_number)
    return None
