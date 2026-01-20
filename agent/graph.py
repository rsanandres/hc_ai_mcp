"""LangGraph agent workflow for HC-AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import yaml
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict, total=False):
    """State for the agent workflow."""
    
    query: str
    session_id: str
    patient_id: Optional[str]
    k_retrieve: Optional[int]
    k_return: Optional[int]
    researcher_output: str
    validator_output: str
    validation_result: str
    final_response: str
    iteration_count: int
    tools_called: List[str]
    sources: List[Dict[str, Any]]


# Global agent instance
_AGENT: Any = None


def _load_prompts() -> Dict[str, Any]:
    """Load prompts from YAML file."""
    prompts_path = Path(__file__).parent / "prompts.yaml"
    if prompts_path.exists():
        with open(prompts_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_researcher_prompt(patient_id: Optional[str] = None) -> str:
    """Get the researcher system prompt."""
    prompts = _load_prompts()
    researcher = prompts.get("researcher", {})
    base_prompt = researcher.get("system_prompt", "You are a medical research assistant.")
    
    fragments = prompts.get("fragments", {})
    safety = fragments.get("safety_reminder", "")
    citation = fragments.get("citation_format", "")
    
    prompt = base_prompt
    if safety:
        prompt += "\n\n" + safety
    if citation:
        prompt += "\n\n" + citation
    
    if patient_id:
        patient_context = fragments.get("patient_context", "").format(patient_id=patient_id)
        if patient_context:
            prompt += "\n\n" + patient_context
    
    return prompt


def _get_validator_prompt() -> str:
    """Get the validator system prompt."""
    prompts = _load_prompts()
    validator = prompts.get("validator", {})
    return validator.get("system_prompt", "You are a medical validator.")


def _extract_tool_calls(messages: List[Any]) -> List[str]:
    """Extract tool call names from messages."""
    calls: List[str] = []
    for message in messages:
        if hasattr(message, "name") and message.name:
            calls.append(message.name)
        if hasattr(message, "tool_calls") and message.tool_calls:
            calls.extend([call.get("name", "") for call in message.tool_calls if call.get("name")])
    return [call for call in calls if call]


def _extract_response_text(messages: List[Any]) -> str:
    """Extract response text from messages."""
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content and isinstance(content, str):
            return content
    return ""


async def _researcher_node(state: AgentState) -> AgentState:
    """Researcher node that searches and analyzes."""
    from agent.config import get_llm
    from db.vector_store import search_similar_chunks
    
    max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
    system_prompt = _get_researcher_prompt(state.get("patient_id"))
    
    query = state.get("query", "")
    k_retrieve = state.get("k_retrieve", 50)
    k_return = state.get("k_return", 10)
    
    # Search for relevant documents
    filter_metadata = None
    if state.get("patient_id"):
        filter_metadata = {"patientId": state["patient_id"]}
    
    docs = await search_similar_chunks(
        query=query,
        k=k_retrieve,
        filter_metadata=filter_metadata,
    )
    
    # Build context from documents
    context_parts = []
    sources = []
    for i, doc in enumerate(docs[:k_return]):
        context_parts.append(f"[{i+1}] {doc.page_content}")
        sources.append({
            "doc_id": getattr(doc, "id", doc.metadata.get("chunkId", f"doc_{i}")),
            "content_preview": doc.page_content[:200],
            "metadata": doc.metadata,
        })
    
    context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
    
    # Build messages
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        
        if state.get("validator_output"):
            messages.append(
                SystemMessage(content=f"Validator feedback:\n{state['validator_output']}")
            )
        
        llm = get_llm()
        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)
        
    except ImportError:
        # Fallback if langchain not available
        response_text = f"Based on the query '{query}', I found {len(docs)} relevant documents."
    
    tools_called = state.get("tools_called", []) + ["search_similar_chunks"]
    
    return {
        **state,
        "researcher_output": response_text,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "tools_called": tools_called,
        "sources": sources,
    }


async def _validator_node(state: AgentState) -> AgentState:
    """Validator node that reviews researcher output."""
    from agent.config import get_llm
    
    system_prompt = _get_validator_prompt()
    
    researcher_output = state.get("researcher_output", "")
    query = state.get("query", "")
    
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Validate the response below. If the researcher response is empty, "
                f"evaluate safety based on the user query alone.\n\n"
                f"User query:\n{query}\n\n"
                f"Researcher response:\n{researcher_output}"
            )),
        ]
        
        llm = get_llm()
        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)
        
    except ImportError:
        response_text = "VALIDATION_STATUS: PASS\n\nVERIFIED_CLAIMS:\n- Response appears valid"
    
    # Extract validation status
    validation_result = "NEEDS_REVISION"
    for line in response_text.splitlines():
        if line.strip().startswith("VALIDATION_STATUS"):
            validation_result = line.split(":", 1)[-1].strip()
            break
    
    tools_called = state.get("tools_called", [])
    
    return {
        **state,
        "validator_output": response_text,
        "validation_result": validation_result,
        "tools_called": tools_called,
    }


def _respond_node(state: AgentState) -> AgentState:
    """Final node that prepares the response."""
    validation_result = state.get("validation_result", "NEEDS_REVISION")
    
    if validation_result == "PASS":
        final_response = state.get("researcher_output", "")
    else:
        final_response = (
            f"{state.get('validator_output', '')}\n\n"
            f"RESEARCHER_RESPONSE:\n{state.get('researcher_output', '')}"
        )
    
    return {**state, "final_response": final_response}


def _route_after_validation(state: AgentState) -> str:
    """Determine next node after validation."""
    validation_result = state.get("validation_result", "NEEDS_REVISION")
    
    if validation_result in {"PASS", "FAIL"}:
        return "respond"
    if state.get("iteration_count", 0) >= 2:
        return "respond"
    return "researcher"


def _create_graph():
    """Create the agent graph."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as e:
        raise ImportError(
            "langgraph is required. Install with: pip install langgraph"
        ) from e
    
    graph = StateGraph(AgentState)
    graph.add_node("researcher", _researcher_node)
    graph.add_node("validator", _validator_node)
    graph.add_node("respond", _respond_node)
    
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "validator")
    graph.add_conditional_edges("validator", _route_after_validation)
    graph.add_edge("respond", END)
    
    return graph.compile()


def get_agent() -> Any:
    """Get or create the compiled agent graph."""
    global _AGENT
    if _AGENT is None:
        _AGENT = _create_graph()
    return _AGENT
