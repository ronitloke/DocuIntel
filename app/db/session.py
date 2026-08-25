"""SQLAlchemy engine and session-factory configuration."""

from __future__ import annotations

from dataclasses import dataclass
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings


@dataclass(frozen=True, slots=True)
class Database:
    """Application-scoped database resources."""

    engine: Engine
    session_factory: sessionmaker[Session]


def create_database(settings: Settings) -> Database | None:
    """Create PostgreSQL resources only when a database URL is configured."""

    database_url = settings.resolved_database_url
    if not database_url:
        return None

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        future=True,
        connect_args={"connect_timeout": 5},
    )
    return Database(
        engine=engine,
        session_factory=sessionmaker(
            bind=engine,
            class_=Session,
            expire_on_commit=False,
            autoflush=False,
        ),
    )
