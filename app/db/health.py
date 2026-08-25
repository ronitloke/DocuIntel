"""Small database readiness helpers."""

from sqlalchemy import text

from app.db.session import Database


def check_database(database: Database | None) -> bool:
    """Return whether PostgreSQL accepts a trivial query."""

    if database is None:
        return False
    try:
        with database.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
