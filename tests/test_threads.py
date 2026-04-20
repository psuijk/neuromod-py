import pytest

from neuromod.messages import user_message, assistant_message, get_text
from neuromod.composition import (
    ConversationContext,
    InMemoryThreadStore,
    thread,
)
from neuromod.config import configure, _config, _factory


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset config before and after each test."""
    _config.set(None)
    _factory.set(None)
    yield
    _config.set(None)
    _factory.set(None)


# ── InMemoryThreadStore ────────────────────────────


async def test_load_empty_thread():
    store = InMemoryThreadStore()
    result = await store.load("nonexistent")
    assert result == []


async def test_save_and_load():
    store = InMemoryThreadStore()
    msgs = [user_message("hello"), assistant_message("hi")]
    await store.save("t1", msgs)
    loaded = await store.load("t1")
    assert len(loaded) == 2
    assert get_text(loaded[0]) == "hello"


async def test_save_overwrites():
    store = InMemoryThreadStore()
    await store.save("t1", [user_message("first")])
    await store.save("t1", [user_message("second")])
    loaded = await store.load("t1")
    assert len(loaded) == 1
    assert get_text(loaded[0]) == "second"


async def test_delete_existing_thread():
    store = InMemoryThreadStore()
    await store.save("t1", [user_message("hello")])
    await store.delete("t1")
    loaded = await store.load("t1")
    assert loaded == []


async def test_delete_nonexistent_thread():
    store = InMemoryThreadStore()
    await store.delete("nonexistent")  # should not raise


async def test_list_empty():
    store = InMemoryThreadStore()
    result = await store.list()
    assert result == []


async def test_list_with_threads():
    store = InMemoryThreadStore()
    await store.save("t1", [user_message("a")])
    await store.save("t2", [user_message("b")])
    result = await store.list()
    assert set(result) == {"t1", "t2"}


async def test_load_returns_copy():
    store = InMemoryThreadStore()
    await store.save("t1", [user_message("hello")])
    loaded = await store.load("t1")
    loaded.append(user_message("extra"))
    loaded2 = await store.load("t1")
    assert len(loaded2) == 1


async def test_save_stores_copy():
    store = InMemoryThreadStore()
    msgs = [user_message("hello")]
    await store.save("t1", msgs)
    msgs.append(user_message("extra"))
    loaded = await store.load("t1")
    assert len(loaded) == 1


# ── thread() composition primitive ─────────────────


async def _echo_step(ctx: ConversationContext) -> ConversationContext:
    return ctx.with_updates(
        messages=[*ctx.messages, assistant_message("echo")]
    )


async def test_thread_loads_history_before_step():
    store = InMemoryThreadStore()
    await store.save("t1", [user_message("old"), assistant_message("old-reply")])

    step = thread("t1", _echo_step, store=store)
    ctx = ConversationContext(messages=[user_message("new")])
    result = await step(ctx)
    # history (2) + new message (1) + echo response (1) = 4
    assert len(result.messages) == 4


async def test_thread_saves_messages_after_step():
    store = InMemoryThreadStore()
    step = thread("t1", _echo_step, store=store)
    ctx = ConversationContext(messages=[user_message("hello")])
    await step(ctx)
    saved = await store.load("t1")
    assert len(saved) == 2  # user_message + echo response


async def test_thread_prepends_history_to_context():
    store = InMemoryThreadStore()
    await store.save("t1", [user_message("history")])

    async def check_step(ctx: ConversationContext) -> ConversationContext:
        assert len(ctx.messages) == 2
        assert get_text(ctx.messages[0]) == "history"
        assert get_text(ctx.messages[1]) == "new"
        return ctx.with_updates(messages=[*ctx.messages, assistant_message("reply")])

    step = thread("t1", check_step, store=store)
    ctx = ConversationContext(messages=[user_message("new")])
    await step(ctx)


async def test_thread_with_empty_history():
    store = InMemoryThreadStore()
    step = thread("t1", _echo_step, store=store)
    ctx = ConversationContext(messages=[user_message("first")])
    result = await step(ctx)
    assert len(result.messages) == 2


async def test_thread_accumulates_across_calls():
    store = InMemoryThreadStore()
    step = thread("t1", _echo_step, store=store)

    ctx1 = ConversationContext(messages=[user_message("first")])
    await step(ctx1)

    ctx2 = ConversationContext(messages=[user_message("second")])
    result = await step(ctx2)
    # history (first + echo) + second + echo = 4
    assert len(result.messages) == 4


async def test_thread_with_explicit_store():
    store = InMemoryThreadStore()
    step = thread("t1", _echo_step, store=store)
    ctx = ConversationContext(messages=[user_message("hello")])
    await step(ctx)
    saved = await store.load("t1")
    assert len(saved) == 2


async def test_thread_uses_configured_store():
    store = InMemoryThreadStore()
    configure(thread_store=store)
    step = thread("configured-test", _echo_step)
    ctx = ConversationContext(messages=[user_message("hello")])
    await step(ctx)
    saved = await store.load("configured-test")
    assert len(saved) == 2


async def test_thread_raises_when_no_store_configured():
    step = thread("no-store-test", _echo_step)
    ctx = ConversationContext(messages=[user_message("hello")])
    with pytest.raises(RuntimeError, match="No thread store configured"):
        await step(ctx)
