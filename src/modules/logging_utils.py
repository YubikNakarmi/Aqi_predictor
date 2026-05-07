"""Logging utilities for the AQI forecasting pipeline, including setup for colored console logging and environment variable configuration."""

import logging
import os
from typing import Optional

try:
    from colorlog import ColoredFormatter
except Exception:
    ColoredFormatter = None

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(name: Optional[str] = None, level: Optional[str | int] = None) -> logging.Logger:
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = level.upper()

    logger = logging.getLogger(name if name else __name__)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    if ColoredFormatter is not None:
        handler.setFormatter(
            ColoredFormatter(
                "%(log_color)s" + DEFAULT_LOG_FORMAT,
                datefmt=DEFAULT_DATE_FORMAT,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    else:
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
