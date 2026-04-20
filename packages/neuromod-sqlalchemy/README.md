# neuromod-sqlalchemy

SQLAlchemy thread store for [neuromod](../../). Persists conversation history to any SQLAlchemy-supported database.

## Install

```bash
pip install neuromod neuromod-sqlalchemy

# For SQLite:
pip install aiosqlite

# For PostgreSQL:
pip install asyncpg
```

## Quick Start

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from neuromod import Agent, Claude, configure
from neuromod_sqlalchemy import Base, SQLAlchemyThreadStore

# Create engine and tables
engine = create_async_engine("sqlite+aiosqlite:///threads.db")
async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.create_all)

# Create store and configure
session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
store = SQLAlchemyThreadStore(session_factory)
configure(thread_store=store)

# Use threads
agent = Agent(model=Claude.Sonnet4_6)
await agent.generate("My name is Alice", thread="user-123")
response = await agent.generate("What's my name?", thread="user-123")
# "Your name is Alice"
```

## PostgreSQL

```python
engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/mydb")
```

## Schema

Two tables, prefixed with `neuromod_`:

### neuromod_threads
| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key (thread ID) |
| created_at | TIMESTAMP | Creation time |
| updated_at | TIMESTAMP | Last update time |

### neuromod_thread_messages
| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key (auto-generated) |
| thread_id | UUID | Foreign key to threads (CASCADE delete) |
| role | VARCHAR(16) | "system", "user", or "assistant" |
| content | TEXT | JSON-serialized message content |
| order | INTEGER | Message position in conversation |
| created_at | TIMESTAMP | Creation time |

Indexed on `(thread_id, order)` for efficient ordered retrieval.

## Smart Save

The store detects whether a save is an append (new messages added to the end) or a rewrite (history changed). Appends only insert new rows; rewrites delete and re-insert. This avoids unnecessary database churn for the common case of adding new messages to an ongoing conversation.

## Exports

```python
from neuromod_sqlalchemy import (
    SQLAlchemyThreadStore,  # ThreadStore implementation
    Base,                   # SQLAlchemy declarative base (for create_all)
    NeuromodThread,         # Thread table model
    NeuromodThreadMessage,  # Message table model
)
```

## Content Serialization

All message content types (text, media, tool calls, tool results) are serialized as JSON in the `content` column. The store handles serialization/deserialization transparently.
