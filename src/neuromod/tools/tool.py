from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

from pydantic import BaseModel

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
