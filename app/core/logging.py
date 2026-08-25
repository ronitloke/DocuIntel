"""Small logging setup used by the initial application bootstrap."""

import logging


def configure_logging(log_level: str = "INFO") -> None:
    """Configure application logging without exposing configuration secrets."""

    level_name = log_level.upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger().setLevel(level)
