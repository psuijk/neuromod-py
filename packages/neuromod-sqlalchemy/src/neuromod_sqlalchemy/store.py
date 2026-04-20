from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from neuromod.composition.thread import ThreadStore
from neuromod.messages.types import (
    Content,
    MediaContent,
    Message,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)

from neuromod_sqlalchemy.schema import NeuromodThread, NeuromodThreadMessage


class SQLAlchemyThreadStore(ThreadStore):
    """ThreadStore implementation backed by SQLAlchemy async sessions."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def load(self, thread_id: str) -> list[Message]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(NeuromodThreadMessage)
                .where(NeuromodThreadMessage.thread_id == uuid.UUID(thread_id))
                .order_by(NeuromodThreadMessage.order)
            )
            rows = result.scalars().all()
            return [
                Message(
                    role=row.role,
                    content=_deserialize_content(row.content),
                )
                for row in rows
            ]

    async def save(self, thread_id: str, messages: list[Message]) -> None:
        tid = uuid.UUID(thread_id)

        async with self._session_factory() as session:
            async with session.begin():
                # Ensure thread exists
                existing_thread = await session.get(NeuromodThread, tid)
                if existing_thread is None:
                    session.add(NeuromodThread(id=tid))
                else:
                    existing_thread.updated_at = datetime.now(timezone.utc)

                # Load existing messages
                result = await session.execute(
                    select(NeuromodThreadMessage)
                    .where(NeuromodThreadMessage.thread_id == tid)
                    .order_by(NeuromodThreadMessage.order)
                )
                existing = result.scalars().all()

                # Detect append vs rewrite
                is_append = (
                    len(messages) >= len(existing)
                    and all(
                        stored.role == messages[i].role
                        and stored.content == _serialize_content(messages[i].content)
                        for i, stored in enumerate(existing)
                    )
                )

                if is_append:
                    new_messages = messages[len(existing):]
                    for i, msg in enumerate(new_messages):
                        session.add(NeuromodThreadMessage(
                            thread_id=tid,
                            role=msg.role,
                            content=_serialize_content(msg.content),
                            order=len(existing) + i,
                        ))
                else:
                    # Rewrite: delete all and re-insert
                    await session.execute(
                        delete(NeuromodThreadMessage)
                        .where(NeuromodThreadMessage.thread_id == tid)
                    )
                    for i, msg in enumerate(messages):
                        session.add(NeuromodThreadMessage(
                            thread_id=tid,
                            role=msg.role,
                            content=_serialize_content(msg.content),
                            order=i,
                        ))

    async def delete(self, thread_id: str) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                thread = await session.get(NeuromodThread, uuid.UUID(thread_id))
                if thread is not None:
                    await session.delete(thread)

    async def list(self) -> list[str]:
        async with self._session_factory() as session:
            result = await session.execute(select(NeuromodThread.id))
            return [str(row[0]) for row in result.all()]


# ── Serialization ─────────────────────────────────


def _serialize_content(content: list[Content]) -> str:
    """Serialize message content to JSON string for storage."""
    parts = []
    for c in content:
        if isinstance(c, TextContent):
            parts.append({"type": "text", "text": c.text})
        elif isinstance(c, MediaContent):
            part = {"type": "media", "data": c.data, "mime_type": c.mime_type}
            if c.filename:
                part["filename"] = c.filename
            parts.append(part)
        elif isinstance(c, ToolCallContent):
            parts.append({
                "type": "tool_call",
                "id": c.id,
                "name": c.name,
                "arguments": c.arguments,
            })
        else:
            parts.append({
                "type": "tool_result",
                "call_id": c.call_id,
                "result": c.result,
                "name": c.name,
                "is_error": c.is_error,
            })
    return json.dumps(parts)


def _deserialize_content(data: str) -> list[Content]:
    """Deserialize JSON string back to message content."""
    parts = json.loads(data)
    result: list[Content] = []
    for part in parts:
        t = part["type"]
        if t == "text":
            result.append(TextContent(text=part["text"]))
        elif t == "media":
            result.append(MediaContent(
                data=part["data"],
                mime_type=part["mime_type"],
                filename=part.get("filename"),
            ))
        elif t == "tool_call":
            result.append(ToolCallContent(
                id=part["id"],
                name=part["name"],
                arguments=part["arguments"],
            ))
        elif t == "tool_result":
            result.append(ToolResultContent(
                call_id=part["call_id"],
                result=part["result"],
                name=part.get("name"),
                is_error=part.get("is_error", False),
            ))
    return result
