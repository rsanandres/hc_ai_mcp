"""Shared utilities for MCP tools."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from logging_config import get_logger

logger = get_logger("atlas.tools")


def error_response(
    error_code: str,
    error: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a standardized error response."""
    payload = {
        "status": "error",
        "error": error,
        "error_code": error_code,
        "details": details or {},
    }
    return payload


def validate_non_empty(name: str, value: Optional[str]) -> Optional[str]:
    """Validate that a string value is non-empty."""
    if value is None or not isinstance(value, str) or not value.strip():
        return f"{name} must be a non-empty string"
    return None


def validate_k(name: str, value: int, minimum: int = 1, maximum: int = 1000) -> Optional[str]:
    """Validate integer bounds for k values."""
    if not isinstance(value, int):
        return f"{name} must be an integer"
    if value < minimum or value > maximum:
        return f"{name} must be between {minimum} and {maximum}"
    return None


def validate_metadata(filter_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Validate filter metadata structure."""
    if filter_metadata is None:
        return None
    if not isinstance(filter_metadata, dict):
        return "filter_metadata must be a dictionary"
    return None


def get_timeout(env_name: str, default: float) -> float:
    """Read timeout seconds from environment."""
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s value: %s", env_name, raw)
        return default
