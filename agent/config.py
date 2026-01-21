"""LLM configuration for the HC-AI agent."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.getenv(name)
    if value and value.isdigit():
        return int(value)
    return default


def _float_env(name: str, default: float) -> float:
    """Get float from environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _require_env(name: str) -> str:
    """Get a required environment variable or raise."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable is required")
    return value


def get_llm() -> Any:
    """Return a configured LLM client based on environment variables.
    
    Supports:
    - Ollama (default): Set LLM_PROVIDER=ollama
    - AWS Bedrock: Set LLM_PROVIDER=bedrock
    - OpenAI: Set LLM_PROVIDER=openai
    - Anthropic: Set LLM_PROVIDER=anthropic
    
    Environment variables:
    - LLM_PROVIDER: "ollama", "bedrock", "openai", or "anthropic" (default: "ollama")
    - LLM_MODEL: Model name (default varies by provider)
    - LLM_TEMPERATURE: Temperature for generation (default: 0.1)
    - LLM_MAX_TOKENS: Max tokens for generation (default: 2048)
    - OLLAMA_BASE_URL: Ollama API URL (default: http://localhost:11434)
    - LLM_NUM_CTX: Context window size for Ollama (default: 4096)
    - OPENAI_API_KEY: OpenAI API key (required for OpenAI)
    - OPENAI_MODEL: OpenAI model name (defaults to LLM_MODEL or gpt-4o-mini)
    - ANTHROPIC_API_KEY: Anthropic API key (required for Anthropic)
    - ANTHROPIC_MODEL: Anthropic model name (defaults to LLM_MODEL or claude-3-5-sonnet-20241022)
    
    Returns:
        Configured LLM client.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    temperature = _float_env("LLM_TEMPERATURE", 0.1)
    max_tokens = _int_env("LLM_MAX_TOKENS", 2048)
    
    if provider == "bedrock":
        try:
            from langchain_aws import ChatBedrock
        except ImportError as e:
            raise ImportError(
                "langchain-aws is required for Bedrock. Install with: pip install langchain-aws"
            ) from e
        
        model_name = os.getenv("LLM_MODEL", "haiku").lower()
        
        # Map friendly names to model IDs
        model_map = {
            "sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "haiku": "anthropic.claude-3-5-haiku-20241022-v2:0",
            "opus": "anthropic.claude-3-opus-20240229-v1:0",
        }
        model_id = model_map.get(model_name, model_name)
        
        return ChatBedrock(
            model_id=model_id,
            model_kwargs={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for OpenAI. Install with: pip install langchain-openai"
            ) from e
        _require_env("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "langchain-anthropic is required for Anthropic. Install with: pip install langchain-anthropic"
            ) from e
        _require_env("ANTHROPIC_API_KEY")
        model = os.getenv("ANTHROPIC_MODEL") or os.getenv("LLM_MODEL") or "claude-3-5-sonnet-20241022"
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider not in {"ollama"}:
        raise ValueError(
            "Unsupported LLM_PROVIDER. Use one of: ollama, bedrock, openai, anthropic."
        )
    
    # Default: Ollama
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise ImportError(
            "langchain-ollama is required for Ollama. Install with: pip install langchain-ollama"
        ) from e
    
    model = os.getenv("LLM_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    num_ctx = _int_env("LLM_NUM_CTX", 4096)
    
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
    )
