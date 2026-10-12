"""Logging configuration utilities.

Sets up structured logging for both console and per-run log files.
Each experiment run gets its own log file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from vae_project.config.models import LoggingConfig


def setup_logging(
    log_config: LoggingConfig,
    log_file: Path | None = None,
    logger_name: str = "vae_project",
) -> logging.Logger:
    """Configure logging for a project run.

    Sets up handlers for:
    - Console (stderr) at the configured console level
    - File (optional) at the configured file level

    Args:
        log_config: Logging configuration object.
        log_file: Optional path to the log file. If None, file logging is skipped.
        logger_name: Root logger name.

    Returns:
        Configured logger instance.
    """
    root_logger = logging.getLogger(logger_name)
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplication on re-initialization
    root_logger.handlers.clear()

    formatter = logging.Formatter(log_config.format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, log_config.console_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_config.file_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger
