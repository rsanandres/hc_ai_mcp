"""Tests for agent graph utilities."""

from __future__ import annotations

from agent.prompt_loader import get_researcher_prompt, get_validator_prompt
from agent.graph import AgentState, create_multi_agent_graph


def test_prompts_load() -> None:
    researcher = get_researcher_prompt()
    validator = get_validator_prompt()
    assert isinstance(researcher, str)
    assert len(researcher) > 50
    assert isinstance(validator, str)
    assert len(validator) > 50


def test_prompts_with_patient_id() -> None:
    prompt = get_researcher_prompt(patient_id="test-uuid-1234")
    assert "test-uuid-1234" in prompt


def test_graph_creates() -> None:
    graph = create_multi_agent_graph()
    assert graph is not None


def test_agent_state_fields() -> None:
    state: AgentState = {
        "query": "test",
        "session_id": "s1",
        "iteration_count": 0,
    }
    assert state["query"] == "test"
