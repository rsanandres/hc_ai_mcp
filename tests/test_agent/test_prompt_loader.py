"""Tests for prompt loader."""

from __future__ import annotations

from agent.prompt_loader import (
    get_researcher_prompt,
    get_validator_prompt,
    get_conversational_prompt,
    get_response_prompt,
    get_metadata,
    load_prompts,
)


def test_prompts_yaml_loads() -> None:
    data = load_prompts()
    assert isinstance(data, dict)
    assert "researcher" in data
    assert "validator" in data
    assert "fragments" in data


def test_researcher_prompt_not_empty() -> None:
    prompt = get_researcher_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 100


def test_researcher_prompt_contains_fragments() -> None:
    prompt = get_researcher_prompt()
    # Should contain HIPAA fragment
    assert "HIPAA" in prompt or "hipaa" in prompt.lower() or "PII" in prompt


def test_researcher_prompt_patient_context() -> None:
    prompt = get_researcher_prompt(patient_id="abc-123")
    assert "abc-123" in prompt


def test_validator_prompt_not_empty() -> None:
    prompt = get_validator_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 50


def test_conversational_prompt_exists() -> None:
    prompt = get_conversational_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 20


def test_response_prompt_exists() -> None:
    prompt = get_response_prompt()
    assert isinstance(prompt, str)
    # Response prompt may be empty if not defined, that's ok
    # but the function should return a string


def test_metadata_has_version() -> None:
    meta = get_metadata()
    assert isinstance(meta, dict)
    assert "version" in meta
