from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

from pydantic import BaseModel

from neuromod.providers.provider import ToolDefinition

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: type[BaseModel]
    execute: Callable[[Any], Awaitable[str]]
    max_calls: int | None = None
    requires_approval: bool = False
    retry: int | None = None


def create_tool(
    *,
    name: str,
    description: str,
    schema: type[T],
    execute: Callable[[T], Awaitable[str]],
    max_calls: int | None = None,
    requires_approval: bool = False,
    retry: int | None = None,
) -> Tool:
    return Tool(
        name=name,
        description=description,
        schema=schema,
        execute=execute,
        max_calls=max_calls,
        requires_approval=requires_approval,
        retry=retry,
    )


def convert_tools(tools: list[Tool] | None) -> list[ToolDefinition] | None:
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
