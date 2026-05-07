import asyncio
import time
from typing import Awaitable, Callable

from pydantic import BaseModel
from neuromod import config
from neuromod.agents.types import AgentResponse, AgentStreamResult
from neuromod.composition.context import ConversationContext, ToolApprovalRequest
from neuromod.composition.model import model as model_step
from neuromod.messages.helpers import user_message
from neuromod.messages.types import Message
from neuromod.models.model import Model
from neuromod.providers.provider import ProviderRequest, TokenCount, TokenUsage, ToolChoice
from neuromod.streaming.events import Channel, StepResult, StreamEvent
from neuromod.tools.tool import Tool, convert_tools
from neuromod.composition.thread import thread as thread_step


class Agent:
    def __init__(
        self,
        *,
        model: Model,
        system: str | Callable[[ConversationContext], str] | None = None,
        tools: list[Tool] | None = None,
        max_steps: int = 10,
        temperature: float | None = None,
        schema: type[BaseModel] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._model = model
        self._system = system
        self._tools = tools
        self._max_steps = max_steps
        self._temperature = temperature
        self._schema = schema
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

    async def __call__(self, ctx: ConversationContext) -> ConversationContext:
        tools = ctx.tools or self._tools
        ctx_with_tools = ctx.with_updates(tools=tools) if tools else ctx
        step = model_step(
            model=self._model,
            system=self._system,
            max_steps=self._max_steps,
            temperature=self._temperature,
            schema=self._schema,
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout)
        return await step(ctx_with_tools)
    
    async def generate(
            self,
            input: str,
            *,
            thread_id: str | None = None,
            model: Model | None = None,
            max_steps: int | None = None,
            system: str | None = None,
            temperature: float | None = None,
            tool_choice: ToolChoice | None = None,
            tool_call_limits: dict[str, int] | None = None,
            tool_approval: Callable[[ToolApprovalRequest], Awaitable[bool]] | None = None,
            signal: asyncio.Event | None = None,
            on_event: Callable[[StreamEvent], None] | None = None,
            timeout: float | None = None,
    ) -> AgentResponse:
        start_time = time.monotonic()
        steps: list[StepResult] = []

        def event_handler(event: StreamEvent) -> None:
            if hasattr(event, 'type') and event.type == "step_complete":
                steps.append(event.step)
            if on_event:
                on_event(event)

        ctx = ConversationContext(
            messages=[user_message(input)],
            tools=self._tools,
            tool_call_limits=tool_call_limits,
            tool_approval=tool_approval,
            on_event=event_handler,
            signal=signal,
        )

        step = model_step(
            model=model or self._model,
            system=system or self._system,
            temperature=temperature or self._temperature,
            max_steps=max_steps or self._max_steps,
            tool_choice=tool_choice,
            schema=self._schema,
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=timeout or self._timeout,
        )

        if thread_id:
            step = thread_step(thread_id, step)

        result = await step(ctx)

        duration = (time.monotonic() - start_time) * 1000
        return self._build_response(result, duration, self._schema, steps)

    def stream(
            self,
            input: str,
            *,
            thread_id: str | None = None,
            model: Model | None = None,
            max_steps: int | None = None,
            system: str | None = None,
            temperature: float | None = None,
            tool_choice: ToolChoice | None = None,
            tool_call_limits: dict[str, int] | None = None,
            tool_approval: Callable[[ToolApprovalRequest], Awaitable[bool]] | None = None,
            signal: asyncio.Event | None = None,
            on_event: Callable[[StreamEvent], None] | None = None,
            timeout: float | None = None,) -> AgentStreamResult:
        channel = Channel[StreamEvent]()
        steps: list[StepResult] = []

        def combined_handler(event: StreamEvent) -> None:
            if hasattr(event, 'type') and event.type == "step_complete":
                steps.append(event.step)
            channel.push(event)
            if on_event:
                on_event(event)

        async def run() -> AgentResponse:
            start_time = time.monotonic()

            ctx = ConversationContext(
                messages=[user_message(input)],
                tools=self._tools,
                tool_call_limits=tool_call_limits,
                tool_approval=tool_approval,
                on_event=combined_handler,
                signal=signal,
            )

            step = model_step(
                model=model or self._model,
                system=system or self._system,
                temperature=temperature or self._temperature,
                max_steps=max_steps or self._max_steps,
                tool_choice=tool_choice,
                schema=self._schema,
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=timeout or self._timeout,
            )

            if thread_id:
                step = thread_step(thread_id, step)

            result = await step(ctx)
            channel.close()

            duration = (time.monotonic() - start_time) * 1000
            return self._build_response(result, duration, self._schema, steps)
        
        task = asyncio.ensure_future(run())
        return AgentStreamResult(events=channel, response=task)
    
    async def count_tokens(self, input: str, *, model: Model | None = None) -> TokenCount:
        resolved_model = model or self._model
        resolved_key = config.resolve_api_key(resolved_model.provider, self._api_key)
        factory = config.get_factory()
        provider = factory.get(resolved_model.provider, api_key=resolved_key, base_url=self._base_url)

        ctx = ConversationContext(messages=[user_message(input)])
        resolved_sys = self._system(ctx) if callable(self._system) else self._system

        request = ProviderRequest(
            model=resolved_model,
            messages=ctx.messages,
            tools=convert_tools(self._tools),
            system=resolved_sys,
        )

        return await provider.count_tokens(request)

    
    def _build_response(self, ctx: ConversationContext, duration_ms: float, schema: type[BaseModel] | None, steps: list[StepResult]) -> AgentResponse:
        last_message = ctx.last_response
        text = last_message.text if last_message else ""

        output = None
        if schema and last_message:
            output = schema.model_validate_json(text)

        return AgentResponse(
            text=text,
            message=last_message or Message(role="assistant", content=[]),
            messages=ctx.messages,
            finish_reason=ctx.stop_reason or "stop",
            steps=steps,
            usage=ctx.usage or TokenUsage(input_tokens=0, output_tokens=0),
            duration_ms=duration_ms,
            output=output
        )
        
