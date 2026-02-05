"""Shared logging configuration for agent bots."""

import logging
import os

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_log_level() -> int:
    """Get log level from LOG_LEVEL environment variable.

    Defaults to WARNING if not set.
    """
    level_name = os.getenv("LOG_LEVEL", "WARNING").upper()
    return getattr(logging, level_name, logging.WARNING)


def configure_logging(verbose: bool = False) -> None:
    """
    Configure logging for agent bots.

    Args:
        verbose: If True, override LOG_LEVEL to DEBUG
    """
    level = logging.DEBUG if verbose else get_log_level()

    logging.basicConfig(
        level=level,
        format=DEFAULT_FORMAT,
        force=True,  # Allow reconfiguration
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("fastmcp").setLevel(logging.WARNING)
