"""Logging configuration for consistent and structured application logging."""

from __future__ import annotations

import logging
import sys


def setup_logging(
    log_level: str | None = None,
    logger_name: str | None = None,
) -> logging.Logger:
    """Configure and initialize application logging once.

    Args:
        log_level: Logging level string (e.g. 'DEBUG', 'INFO', 'WARNING').
        logger_name: Optional logger name. If None, configures the root logger.

    Returns:
        Configured logging.Logger instance.
    """
    effective_level = (log_level or "INFO").strip().upper()
    numeric_level = getattr(logging, effective_level, logging.INFO)

    target_logger = logging.getLogger(logger_name)
    target_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is invoked multiple times
    if not target_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        target_logger.addHandler(console_handler)

    return target_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger with project-wide formatting.

    Args:
        name: Name of the logger, typically __name__.

    Returns:
        A logging.Logger instance.
    """
    return logging.getLogger(name)
