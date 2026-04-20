from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class NeuromodThread(Base):
    __tablename__ = "neuromod_threads"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    messages: Mapped[list[NeuromodThreadMessage]] = relationship(
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="NeuromodThreadMessage.order",
    )


class NeuromodThreadMessage(Base):
    __tablename__ = "neuromod_thread_messages"
    __table_args__ = (
        Index("ix_neuromod_thread_order", "thread_id", "order"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4,
    )
    thread_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("neuromod_threads.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    thread: Mapped[NeuromodThread] = relationship(back_populates="messages")
