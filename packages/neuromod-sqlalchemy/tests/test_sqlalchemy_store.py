from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from neuromod.messages.helpers import assistant_message, user_message
from neuromod.messages.types import (
    MediaContent,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)
from neuromod_sqlalchemy import Base, SQLAlchemyThreadStore


@pytest.fixture
async def store():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield SQLAlchemyThreadStore(session_factory)
    await engine.dispose()


def thread_id() -> str:
    return str(uuid.uuid4())


# ── Basic CRUD ────────────────────────────────────


class TestBasicOperations:
    async def test_load_empty(self, store: SQLAlchemyThreadStore):
        result = await store.load(thread_id())
        assert result == []

    async def test_save_and_load(self, store: SQLAlchemyThreadStore):
        tid = thread_id()
        msgs = [user_message("hello"), assistant_message("hi")]
        await store.save(tid, msgs)
        loaded = await store.load(tid)
        assert len(loaded) == 2
        assert loaded[0].role == "user"
        assert isinstance(loaded[0].content[0], TextContent)
        assert loaded[0].content[0].text == "hello"
        assert loaded[1].role == "assistant"

    async def test_save_overwrites(self, store: SQLAlchemyThreadStore):
        tid = thread_id()
        await store.save(tid, [user_message("first")])
        await store.save(tid, [user_message("second")])
        loaded = await store.load(tid)
        assert len(loaded) == 1
        assert loaded[0].content[0].text == "second"

    async def test_delete(self, store: SQLAlchemyThreadStore):
        tid = thread_id()
        await store.save(tid, [user_message("hello")])
        await store.delete(tid)
        loaded = await store.load(tid)
        assert loaded == []

    async def test_delete_nonexistent(self, store: SQLAlchemyThreadStore):
        await store.delete(thread_id())  # should not raise

    async def test_list_empty(self, store: SQLAlchemyThreadStore):
        result = await store.list()
        assert result == []

    async def test_list_threads(self, store: SQLAlchemyThreadStore):
        tid1 = thread_id()
        tid2 = thread_id()
        await store.save(tid1, [user_message("a")])
        await store.save(tid2, [user_message("b")])
        result = await store.list()
        assert set(result) == {tid1, tid2}


# ── Smart save (append detection) ────────────────


class TestSmartSave:
    async def test_append_only_inserts_new(self, store: SQLAlchemyThreadStore):
        tid = thread_id()
        await store.save(tid, [user_message("first")])
        await store.save(tid, [user_message("first"), assistant_message("reply")])
        loaded = await store.load(tid)
        assert len(loaded) == 2
        assert loaded[0].content[0].text == "first"
        assert loaded[1].content[0].text == "reply"

    async def test_rewrite_on_changed_history(self, store: SQLAlchemyThreadStore):
        tid = thread_id()
        await store.save(tid, [user_message("original")])
        await store.save(tid, [user_message("changed"), assistant_message("reply")])
        loaded = await store.load(tid)
        assert len(loaded) == 2
        assert loaded[0].content[0].text == "changed"

    async def test_multiple_appends(self, store: SQLAlchemyThreadStore):
        tid = thread_id()
        msgs = [user_message("1")]
        await store.save(tid, msgs)

        msgs.append(assistant_message("2"))
        await store.save(tid, msgs)

        msgs.append(user_message("3"))
        await store.save(tid, msgs)

        loaded = await store.load(tid)
        assert len(loaded) == 3


# ── Content type serialization ────────────────────


class TestContentSerialization:
    async def test_text_content(self, store: SQLAlchemyThreadStore):
        tid = thread_id()
        await store.save(tid, [user_message("hello")])
        loaded = await store.load(tid)
        assert isinstance(loaded[0].content[0], TextContent)
        assert loaded[0].content[0].text == "hello"

    async def test_tool_call_content(self, store: SQLAlchemyThreadStore):
        from neuromod.messages.helpers import tool_call
        tid = thread_id()
        msg = assistant_message([tool_call("tc_1", "search", {"query": "test"})])
        await store.save(tid, [msg])
        loaded = await store.load(tid)
        content = loaded[0].content[0]
        assert isinstance(content, ToolCallContent)
        assert content.id == "tc_1"
        assert content.name == "search"
        assert content.arguments == {"query": "test"}

    async def test_tool_result_content(self, store: SQLAlchemyThreadStore):
        from neuromod.messages.helpers import tool_result
        tid = thread_id()
        msg = user_message([tool_result("tc_1", "found it", name="search")])
        await store.save(tid, [msg])
        loaded = await store.load(tid)
        content = loaded[0].content[0]
        assert isinstance(content, ToolResultContent)
        assert content.call_id == "tc_1"
        assert content.result == "found it"
        assert content.name == "search"

    async def test_media_content(self, store: SQLAlchemyThreadStore):
        from neuromod.messages.types import Message
        tid = thread_id()
        msg = Message(role="user", content=[MediaContent(data="abc", mime_type="image/png", filename="test.png")])
        await store.save(tid, [msg])
        loaded = await store.load(tid)
        content = loaded[0].content[0]
        assert isinstance(content, MediaContent)
        assert content.data == "abc"
        assert content.mime_type == "image/png"
        assert content.filename == "test.png"
