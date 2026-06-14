from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy import String, Text, Boolean, DateTime, Integer, JSON
from datetime import datetime
from typing import Optional
from config import settings


def get_database_url() -> str:
    """Use Supabase if configured, otherwise fall back to SQLite."""
    if settings.SUPABASE_DB_URL:
        url = settings.SUPABASE_DB_URL
        # Ensure we use asyncpg driver
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        print(f"✅ Database: Supabase PostgreSQL")
        return url
    print("✅ Database: SQLite (local)")
    return settings.DATABASE_URL


DB_URL = get_database_url()
IS_POSTGRES = DB_URL.startswith("postgresql")

engine = create_async_engine(
    DB_URL,
    echo=False,
    pool_pre_ping=True,
    **({"pool_size": 5, "max_overflow": 10} if IS_POSTGRES else {}),
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Memory(Base):
    __tablename__ = "memories"
    id:          Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    fact:        Mapped[str]      = mapped_column(Text, nullable=False)
    category:    Mapped[str]      = mapped_column(String(50), default="general")
    importance:  Mapped[int]      = mapped_column(Integer, default=5)
    created_at:  Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at:  Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Goal(Base):
    __tablename__ = "goals"
    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    text:       Mapped[str]      = mapped_column(Text, nullable=False)
    done:       Mapped[bool]     = mapped_column(Boolean, default=False)
    priority:   Mapped[int]      = mapped_column(Integer, default=5)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Conversation(Base):
    __tablename__ = "conversations"
    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str]      = mapped_column(String(100), nullable=False)
    role:       Mapped[str]      = mapped_column(String(20), nullable=False)
    content:    Mapped[str]      = mapped_column(Text, nullable=False)
    agent:      Mapped[str]      = mapped_column(String(50), default="general")
    task_type:  Mapped[str]      = mapped_column(String(50), default="general")
    model_used: Mapped[str]      = mapped_column(String(100), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Connector(Base):
    __tablename__ = "connectors"
    id:              Mapped[int]           = mapped_column(Integer, primary_key=True, autoincrement=True)
    name:            Mapped[str]           = mapped_column(String(100), nullable=False)
    description:     Mapped[str]           = mapped_column(Text, default="")
    icon:            Mapped[str]           = mapped_column(String(10), default="⚙")
    base_url:        Mapped[str]           = mapped_column(Text, nullable=False)
    endpoint:        Mapped[str]           = mapped_column(Text, default="/")
    method:          Mapped[str]           = mapped_column(String(10), default="GET")
    headers:         Mapped[dict]          = mapped_column(JSON, default=dict)
    params_template: Mapped[dict]          = mapped_column(JSON, default=dict)
    body_template:   Mapped[dict]          = mapped_column(JSON, default=dict)
    api_key:         Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    api_key_header:  Mapped[str]           = mapped_column(String(100), default="Authorization")
    api_key_prefix:  Mapped[str]           = mapped_column(String(20), default="Bearer")
    enabled:         Mapped[bool]          = mapped_column(Boolean, default=True)
    created_at:      Mapped[datetime]      = mapped_column(DateTime, default=datetime.utcnow)


class OptimizationLog(Base):
    __tablename__ = "optimization_logs"
    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    event:      Mapped[str]      = mapped_column(Text, nullable=False)
    module:     Mapped[str]      = mapped_column(String(50), default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DigitalTwin(Base):
    __tablename__ = "digital_twin"
    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    key:        Mapped[str]      = mapped_column(String(100), unique=True, nullable=False)
    value:      Mapped[dict]     = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ All tables created/verified")


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
