"""Tests for agent graph utilities."""

from __future__ import annotations

from agent.graph import _get_researcher_prompt, _get_validator_prompt


def test_prompts_load() -> None:
    researcher = _get_researcher_prompt()
    validator = _get_validator_prompt()
    assert isinstance(researcher, str)
    assert isinstance(validator, str)
