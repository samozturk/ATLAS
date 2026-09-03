"""Structured logging configuration."""

import logging
import sys

import structlog


def configure_logging(level: str) -> None:
    """Configure newline-delimited JSON logs for people and log collectors."""
    numeric_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    # Emit one JSON object per line so logs remain readable locally and can be
    # ingested unchanged by future observability tooling.
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=numeric_level, force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
