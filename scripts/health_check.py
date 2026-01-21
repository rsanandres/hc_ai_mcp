#!/usr/bin/env python3
"""Health check script for HC-AI MCP Server."""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List

from config import load_config, validate_env
from logging_config import get_logger

logger = get_logger("hc_ai.health")


async def _check_db() -> Dict[str, Any]:
    from db import get_connection_stats

    try:
        stats = await get_connection_stats()
        return {"ok": True, "stats": stats}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_embeddings() -> Dict[str, Any]:
    from embeddings import test_connection

    status = test_connection()
    return {
        "ok": bool(status.get("ok")),
        "provider": status.get("provider"),
        "errors": status.get("errors", []),
    }


def _check_reranker() -> Dict[str, Any]:
    try:
        from reranker import get_reranker

        reranker = get_reranker()
        return {"ok": True, "model": reranker.model_name, "device": reranker.device}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_agent() -> Dict[str, Any]:
    try:
        from agent import get_agent

        _ = get_agent()
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def run() -> int:
    config = load_config()
    env_errors = validate_env()
    errors: List[str] = []

    if env_errors:
        errors.extend(env_errors)

    results: Dict[str, Any] = {"env": {"ok": not env_errors, "errors": env_errors}}

    results["db"] = await _check_db()
    if not results["db"]["ok"]:
        errors.append("Database health check failed")

    results["embeddings"] = _check_embeddings()
    if not results["embeddings"]["ok"]:
        errors.append("Embeddings health check failed")

    results["reranker"] = _check_reranker()
    if not results["reranker"]["ok"]:
        errors.append("Reranker health check failed")

    results["agent"] = _check_agent()
    if not results["agent"]["ok"]:
        errors.append("Agent health check failed")

    logger.info("Health check results: %s", results)
    if errors:
        logger.error("Health check failed: %s", errors)
        return 1

    logger.info("Health check passed")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run())
    sys.exit(exit_code)
