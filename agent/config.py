"""LLM configuration for the ReAct agent."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value and value.isdigit():
        return int(value)
    return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    return default


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable {name} is required but not set")
    return value


def get_llm() -> Any:
    """Return a configured LLM client based on environment variables.

    Supported providers:
    - ollama (default): Local LLM via Ollama
    - bedrock: AWS Bedrock (Claude models)
    - openai: OpenAI API
    - anthropic: Anthropic API (Claude models)
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    temperature = _float_env("LLM_TEMPERATURE", 0.1)
    max_tokens = _int_env("LLM_MAX_TOKENS", 2048)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = _require_env("OPENAI_API_KEY")
        model = os.getenv("LLM_MODEL", "gpt-4o")

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = _require_env("ANTHROPIC_API_KEY")
        model = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    if provider == "bedrock":
        from langchain_aws import ChatBedrock

        model_name = os.getenv("LLM_MODEL", "haiku").lower()
        # Friendly name mapping
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

    # Default: Ollama
    from langchain_ollama import ChatOllama

    model = os.getenv("LLM_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    num_ctx = _int_env("LLM_NUM_CTX", 4096)
    timeout = _int_env("LLM_TIMEOUT_SECONDS", 60)

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
        timeout=timeout,
    )
