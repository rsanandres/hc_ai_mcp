"""YAML configuration loader for tool toggles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


_CONFIG_CACHE: Dict[str, Any] | None = None


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses config.yaml in package root.
    
    Returns:
        Configuration dictionary.
    """
    global _CONFIG_CACHE
    
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        # Return default config if file doesn't exist
        return _default_config()
    
    with open(config_path, "r") as f:
        _CONFIG_CACHE = yaml.safe_load(f) or {}
    
    return _CONFIG_CACHE


def _default_config() -> Dict[str, Any]:
    """Return default configuration with all tools enabled."""
    return {
        "server": {
            "name": "HC-AI MCP Server",
            "transport": "stdio",
        },
        "tools": {
            "agent_query": {"enabled": True},
            "agent_clear_session": {"enabled": True},
            "agent_health": {"enabled": True},
            "rerank": {"enabled": True},
            "rerank_with_context": {"enabled": True},
            "batch_rerank": {"enabled": True},
            "session_append_turn": {"enabled": True},
            "session_get": {"enabled": True},
            "session_update_summary": {"enabled": True},
            "session_clear": {"enabled": True},
            "ingest": {"enabled": True},
            "embeddings_health": {"enabled": True},
            "db_stats": {"enabled": True},
            "db_queue": {"enabled": True},
            "db_errors": {"enabled": True},
        },
    }


def is_tool_enabled(config: Dict[str, Any], tool_name: str) -> bool:
    """Check if a tool is enabled in the configuration.
    
    Args:
        config: Configuration dictionary.
        tool_name: Name of the tool to check.
    
    Returns:
        True if the tool is enabled, False otherwise.
    """
    tools = config.get("tools", {})
    tool_config = tools.get(tool_name, {})
    return tool_config.get("enabled", False)


def get_server_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get server configuration.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        Server configuration dictionary.
    """
    return config.get("server", {
        "name": "HC-AI MCP Server",
        "transport": "stdio",
    })


def reload_config() -> Dict[str, Any]:
    """Force reload of configuration from file."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
    return load_config()
