"""
database/session.py — SQLAlchemy engine + Motor (async MongoDB) session management.

SQLAlchemy layer:  SQLite (dev) / PostgreSQL (prod) — set DATABASE_URL env var.
Motor layer:       MongoDB — set MONGO_URL env var (default: mongodb://localhost:27017).

The two DB layers are completely independent. Existing code using ``get_db_session``
and ``SessionLocal`` is unaffected. New async routes use ``get_motor_db``.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from core.config import settings
from database.models import Base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLAlchemy — unchanged
# ---------------------------------------------------------------------------

_is_memory_sqlite = settings.DATABASE_URL == "sqlite:///:memory:"

_engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    # StaticPool keeps all connections on the same in-process :memory: db (test only)
    **({"poolclass": StaticPool} if _is_memory_sqlite else {"pool_pre_ping": True}),
)

# Enable WAL mode for SQLite to allow concurrent reads during writes
@event.listens_for(_engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _record):
    if "sqlite" in settings.DATABASE_URL:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def create_tables() -> None:
    """Create all SQLAlchemy tables if they do not exist. Called at application startup."""
    Base.metadata.create_all(bind=_engine)


def get_db_session() -> Session:
    """Return a new SQLAlchemy database session (caller is responsible for closing)."""
    return SessionLocal()


# ---------------------------------------------------------------------------
# Motor (async MongoDB) — second connection layer
# ---------------------------------------------------------------------------

# Module-level client — created lazily by init_motor(), reused for the lifetime
# of the process. None until init_motor() is called.
_motor_client: AsyncIOMotorClient | None = None
_motor_db: AsyncIOMotorDatabase | None = None


def init_motor() -> None:
    """
    Initialise the Motor client and select the working database.

    Call once inside the FastAPI lifespan startup block:

        async with lifespan(app):
            init_motor()

    The database name defaults to ``"visionfood"``; override by appending it to
    MONGO_URL:  ``mongodb://host:27017/my_db_name``.
    """
    global _motor_client, _motor_db

    mongo_url: str = getattr(settings, "MONGO_URL", "mongodb://localhost:27017")
    db_name: str = getattr(settings, "MONGO_DB_NAME", "visionfood")

    _motor_client = AsyncIOMotorClient(mongo_url)
    _motor_db = _motor_client[db_name]
    logger.info("Motor client initialised: url=%s db=%s", mongo_url, db_name)


def close_motor() -> None:
    """Close the Motor client. Call inside the FastAPI lifespan shutdown block."""
    global _motor_client, _motor_db
    if _motor_client is not None:
        _motor_client.close()
        _motor_client = None
        _motor_db = None
        logger.info("Motor client closed.")


async def get_motor_db() -> AsyncGenerator[AsyncIOMotorDatabase, None]:
    """
    FastAPI dependency that yields the Motor database for the current request.

    Usage in a router:
        from database.session import get_motor_db

        @router.get("/")
        async def my_endpoint(db: AsyncIOMotorDatabase = Depends(get_motor_db)):
            ...
    """
    if _motor_db is None:
        raise RuntimeError(
            "Motor database is not initialised. "
            "Ensure init_motor() is called in the FastAPI lifespan startup block."
        )
    yield _motor_db
