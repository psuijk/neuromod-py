import asyncio
from typing import Callable
from pydantic import BaseModel
from neuromod import config
from neuromod.composition.context import ConversationContext, StepFunction, ToolApprovalRequest
from neuromod.messages.helpers import get_tool_calls
from neuromod.messages.types import Content, Message, ToolCallContent, ToolResultContent
from neuromod.models.model import Model
from neuromod.providers.provider import ProviderRequest, TokenUsage, ToolDefinition
from neuromod.tools.tool import Tool


def model(
        *,
        model: Model,
        system: str | Callable[[ConversationContext], str] | None = None,
        temperature: float | None = None,
        schema: type[BaseModel] | None = None,
        max_steps: int = 10,
        api_key: str | None = None,
        base_url: str | None = None,
) -> StepFunction:

    async def run(ctx: ConversationContext) -> ConversationContext:
        resolved_key = config.resolve_api_key(model.provider, api_key)
        factory = config.get_factory()
        provider = factory.get(model.provider, api_key=resolved_key, base_url=base_url)
        resolved_sys = system(ctx) if callable(system) else system
        tool_defs = _convert_tools(ctx.tools)
        json_schema = schema.model_json_schema() if schema else None
        total_usage = ctx.usage if ctx.usage is not None else TokenUsage(input_tokens=0, output_tokens=0)
        tool_call_counts: dict[str, int] = {}

        for _ in range(max_steps):
            request = ProviderRequest(
                model=model,
                messages=ctx.messages,
                tools=tool_defs,
                system=resolved_sys,
                temperature=temperature,
                schema=json_schema,
                signal=ctx.signal,
            )

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

            tool_calls = get_tool_calls(response.message)
            if not tool_calls:
                return ctx.with_updates(stop_reason="stop")

            results = await execute_tools(tool_calls, ctx.tools or [], tool_call_counts, ctx)
            content: list[Content] = list(results)
            ctx = ctx.with_updates(messages=[*ctx.messages, Message(role="user", content=content)])

        return ctx.with_updates(stop_reason="max_steps")

    return run

async def execute_tools(
        tool_calls: list[ToolCallContent], 
        tools: list[Tool], call_counts: dict[str, int], 
        ctx: ConversationContext
) -> list[ToolResultContent]:
    tool_map = {t.name: t for t in (tools or [])}

    async def execute_one(call: ToolCallContent) -> ToolResultContent:
        tool = tool_map.get(call.name)

        if tool is None:
            return ToolResultContent(
                call_id=call.id, 
                result=f"Unknown tool: {call.name}",
                is_error=True)
        
        count = call_counts.get(call.name, 0) + 1
        call_counts[call.name] = count
        if tool.max_calls is not None and count > tool.max_calls:
            return ToolResultContent(
                call_id=call.id, 
                result="Tool exceeded max calls",
                is_error=True)
        
        if tool.requires_approval and ctx.tool_approval:
            approved = await ctx.tool_approval(ToolApprovalRequest(id=call.id, name=call.name, arguments=call.arguments))

            if not approved:
                return ToolResultContent(call_id=call.id, result="Approval denied", is_error=True)
            
        max_retries = tool.retry or 0
        last_error = None
        for _ in range(max_retries + 1):
            try:
                parsed = tool.schema(**call.arguments)
                result = await tool.execute(parsed)
                return ToolResultContent(call_id=call.id, result=result, name=call.name)
            except Exception as e:
                last_error = e

        return ToolResultContent(call_id=call.id, result=str(last_error), name=call.name, is_error=True)
    
    results = await asyncio.gather(*[execute_one(call) for call in tool_calls])
    return list(results)

def _convert_tools(tools: list[Tool] | None) -> list[ToolDefinition] | None:
    if not tools:
        return None
    return [
        ToolDefinition(
            name=t.name,
            description=t.description,
            parameters=t.schema.model_json_schema(),
        )
        for t in tools
    ]

