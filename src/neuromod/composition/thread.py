from __future__ import annotations

from abc import ABC, abstractmethod

from neuromod.messages.types import Message
from neuromod.composition.context import ConversationContext, StepFunction


class ThreadStore(ABC):
    @abstractmethod
    async def load(self, thread_id: str) -> list[Message]:
        ...

    @abstractmethod
    async def save(self, thread_id: str, messages: list[Message]) -> None:
        ...

    @abstractmethod
    async def delete(self, thread_id: str) -> None:
        ...

    @abstractmethod
    async def list(self) -> list[str]:
        ...


class InMemoryThreadStore(ThreadStore):
    def __init__(self) -> None:
        self._threads: dict[str, list[Message]] = {}

    async def load(self, thread_id: str) -> list[Message]:
        return list(self._threads.get(thread_id, []))

    async def save(self, thread_id: str, messages: list[Message]) -> None:
        self._threads[thread_id] = list(messages)

    async def delete(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)

    async def list(self) -> list[str]:
        return list(self._threads.keys())


def _get_thread_store() -> ThreadStore:
    """Get the configured thread store, or raise if none is configured."""
    from neuromod.config import get_config
    config = get_config()
    if config.thread_store is not None:
        return config.thread_store
    from neuromod.providers.errors import ConfigError
    raise ConfigError(
        "No thread store configured. Call configure(thread_store=...) "
        "or pass a store explicitly to thread()."
    )


def thread(thread_id: str, step: StepFunction, *, store: ThreadStore | None = None) -> StepFunction:
    async def run(ctx: ConversationContext) -> ConversationContext:
        resolved_store = store if store is not None else _get_thread_store()
        history = await resolved_store.load(thread_id)
        ctx_with_history = ctx.with_updates(messages=[*history, *ctx.messages])
        result = await step(ctx_with_history)
        await resolved_store.save(thread_id, result.messages)
        return result

    return run
