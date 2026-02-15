"""Logging configuration for the Atlas MCP server."""

from __future__ import annotations

import logging
import os
from typing import Optional

_CONFIGURED = False


def configure_logging() -> None:
    """Configure global logging settings once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    debug_enabled = os.getenv("HC_AI_DEBUG", "").lower() in {"1", "true", "yes"}
    level = logging.DEBUG if debug_enabled else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance."""
    configure_logging()
    return logging.getLogger(name)
