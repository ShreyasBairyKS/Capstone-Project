"""
database/session.py — SQLAlchemy engine and session factory.

Dev:  SQLite  (file-backed, zero config)
Prod: PostgreSQL (set DATABASE_URL env var)
"""

from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from core.config import settings
from database.models import Base

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
    """Create all tables if they do not exist. Called at application startup."""
    Base.metadata.create_all(bind=_engine)


def get_db_session() -> Session:
    """Return a new database session (caller is responsible for closing)."""
    return SessionLocal()
