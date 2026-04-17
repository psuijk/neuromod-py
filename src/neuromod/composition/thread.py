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


_default_store: ThreadStore | None = None


def get_default_thread_store() -> ThreadStore:
    global _default_store
    if _default_store is None:
        _default_store = InMemoryThreadStore()
    return _default_store


def set_default_thread_store(store: ThreadStore) -> None:
    global _default_store
    _default_store = store


def thread(thread_id: str, step: StepFunction, *, store: ThreadStore | None = None) -> StepFunction:
    async def run(ctx: ConversationContext) -> ConversationContext:
        resolved_store = store if store is not None else get_default_thread_store()
        history = await resolved_store.load(thread_id)
        ctx_with_history = ctx.with_updates(messages=[*history, *ctx.messages])
        result = await step(ctx_with_history)
        await resolved_store.save(thread_id, result.messages)
        return result

    return run
