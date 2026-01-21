"""YAML configuration loader for tool toggles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field, ValidationError

from logging_config import get_logger

logger = get_logger("hc_ai.config")


_CONFIG_CACHE: Dict[str, Any] | None = None


class ToolConfig(BaseModel):
    """Tool configuration schema."""

    enabled: bool = False
    description: str | None = None


class ServerConfig(BaseModel):
    """Server configuration schema."""

    name: str = "HC-AI MCP Server"
    transport: str = Field(default="stdio", pattern="^(stdio|streamable-http)$")


class AppConfig(BaseModel):
    """Top-level configuration schema."""

    server: ServerConfig = ServerConfig()
    tools: Dict[str, ToolConfig] = Field(default_factory=dict)


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses config.yaml in package root.
    
    Returns:
        Configuration dictionary.
    """
    global _CONFIG_CACHE
    
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        # Return default config if file doesn't exist
        _CONFIG_CACHE = _default_config()
        return _CONFIG_CACHE
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f) or {}

    try:
        validated = AppConfig.model_validate(raw_config)
        _CONFIG_CACHE = validated.model_dump()
    except ValidationError as exc:
        logger.error("Invalid config.yaml: %s", exc)
        raise ValueError("Invalid configuration file") from exc

    return _CONFIG_CACHE


def _default_config() -> Dict[str, Any]:
    """Return default configuration with all tools enabled."""
    default = {
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
    validated = AppConfig.model_validate(default)
    return validated.model_dump()


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


def validate_env() -> List[str]:
    """Validate required environment variables."""
    errors: List[str] = []

    db_password = os.getenv("DB_PASSWORD")
    if not db_password:
        errors.append("DB_PASSWORD is required")

    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
    if llm_provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        errors.append("ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic")
    if llm_provider == "bedrock" and not os.getenv("AWS_REGION"):
        errors.append("AWS_REGION is required for LLM_PROVIDER=bedrock")

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
    if embedding_provider == "bedrock" and not os.getenv("AWS_REGION"):
        errors.append("AWS_REGION is required for EMBEDDING_PROVIDER=bedrock")

    session_provider = os.getenv("SESSION_PROVIDER", "memory").lower()
    if session_provider == "dynamodb" and not os.getenv("AWS_REGION"):
        errors.append("AWS_REGION is required for SESSION_PROVIDER=dynamodb")

    return errors
