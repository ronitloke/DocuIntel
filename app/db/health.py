"""Small database readiness helpers."""

import logging

from sqlalchemy import text

from app.db.session import Database


logger = logging.getLogger(__name__)


def check_database(database: Database | None) -> bool:
    """Return whether PostgreSQL accepts a trivial query."""

    if database is None:
        return False
    try:
        with database.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning(
            "PostgreSQL readiness check failed dependency=postgresql "
            "error_type=%s readiness=false",
            type(exc).__name__,
        )
        return False
