"""LangGraph multi-agent workflow: Classify -> Researcher -> Validator -> Respond."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from agent.config import get_llm

try:
    from langgraph.errors import GraphRecursionError
except ImportError:
    GraphRecursionError = RecursionError

from agent.prompt_loader import (
    get_researcher_prompt,
    get_validator_prompt,
    get_conversational_prompt,
    get_response_prompt,
)
from agent.tools import (
    calculate,
    cross_reference_meds,
    get_current_date,
    get_patient_timeline,
    get_session_context,
    lookup_loinc,
    lookup_rxnorm,
    search_clinical_notes,
    search_icd10,
    search_patient_records,
    validate_icd10_code,
)
from agent.tools.context import set_patient_context

# Import session store for automatic history injection
try:
    from session.store import get_session_store
    SESSION_STORE_AVAILABLE = True
except ImportError:
    SESSION_STORE_AVAILABLE = False
    get_session_store = None

# Import query classifier
from agent.query_classifier import QueryClassifier, QueryType

from logging_config import get_logger

logger = get_logger("atlas.agent")


class AgentState(TypedDict, total=False):
    query: str
    session_id: str
    patient_id: Optional[str]
    # Query classification
    query_type: str  # "conversational" | "medical" | "mixed" | "unclear"
    classification_confidence: float
    classification_method: str
    should_acknowledge_greeting: bool
    # Agent outputs
    researcher_output: str
    validator_output: str
    validation_result: str
    final_response: str
    iteration_count: int
    tools_called: List[str]
    sources: List[Dict[str, Any]]
    # Trajectory tracking for death loop prevention
    search_attempts: List[Dict[str, Any]]
    empty_search_count: int


_RESEARCHER_AGENT: Any = None
_VALIDATOR_AGENT: Any = None
_RESPONSE_AGENT: Any = None


def _extract_tool_calls(messages: List[Any]) -> List[str]:
    calls: List[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.name:
                calls.append(message.name)
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            calls.extend([call.get("name", "") for call in message.tool_calls])
    return [call for call in calls if call]


def _extract_response_text(messages: List[Any]) -> str:
    content = ""
    found = False

    for message in reversed(messages):
        if isinstance(message, AIMessage) and getattr(message, "content", ""):
            content = str(message.content)
            found = True
            break

    if not found:
        for message in reversed(messages):
            if getattr(message, "content", ""):
                content = str(message.content)
                found = True
                break

    if not found and messages:
        last = messages[-1]
        content = getattr(last, "content", "") if hasattr(last, "content") else str(last)

    # Clean internal prompt leakage
    content = re.sub(r'=+\s*VALIDATION TOOLS AVAILABLE\s*=+', '', content, flags=re.IGNORECASE)
    content = re.sub(r'=+\s*OUTPUT FORMAT.*=+', '', content, flags=re.IGNORECASE)

    return content.strip()


def _extract_sources(messages: List[Any]) -> List[Dict[str, Any]]:
    """Extract source documents from ToolMessages."""
    sources: List[Dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            try:
                content_str = str(message.content)
                data = json.loads(content_str)

                if isinstance(data, dict) and "chunks" in data and isinstance(data["chunks"], list):
                    for chunk in data["chunks"]:
                        if isinstance(chunk, dict):
                            source_entry: Dict[str, Any] = {
                                "doc_id": chunk.get("id", ""),
                                "content_preview": chunk.get("content", "") or chunk.get("text", ""),
                                "metadata": chunk.get("metadata", {}),
                            }
                            if "score" in chunk and chunk["score"] is not None:
                                source_entry["score"] = float(chunk["score"])
                            sources.append(source_entry)
            except (json.JSONDecodeError, Exception):
                continue
    return sources


def _get_researcher_agent() -> Any:
    global _RESEARCHER_AGENT
    if _RESEARCHER_AGENT is None:
        llm = get_llm()
        tools = [
            search_patient_records,
            search_clinical_notes,
            get_patient_timeline,
            cross_reference_meds,
            get_session_context,
            search_icd10,
            calculate,
            get_current_date,
        ]
        _RESEARCHER_AGENT = create_react_agent(llm, tools)
    return _RESEARCHER_AGENT


def _get_response_agent() -> Any:
    global _RESPONSE_AGENT
    if _RESPONSE_AGENT is None:
        llm = get_llm()
        _RESPONSE_AGENT = create_react_agent(llm, [])
    return _RESPONSE_AGENT


def _get_validator_agent() -> Any:
    global _VALIDATOR_AGENT
    if _VALIDATOR_AGENT is None:
        llm = get_llm()
        tools = [
            validate_icd10_code,
            lookup_loinc,
            lookup_rxnorm,
            get_current_date,
        ]
        _VALIDATOR_AGENT = create_react_agent(llm, tools)
    return _VALIDATOR_AGENT


def _clean_response(text: str) -> str:
    """Strip internal details that shouldn't be shown to users."""
    if not text:
        return text

    text = re.sub(r'\*?\*?[Vv]alidation[_ ][Ss]tatus:?\*?\*?:?\s*\w+', '', text)
    text = re.sub(r'issues:\s*\n\s*-.*?(?=\n\n|\n[A-Z]|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'\*?\*?Final Output Override:?\*?\*?:?\s*None\s*', '', text)
    text = re.sub(r'\[FHIR:\w+/\d+\]', '', text)
    text = re.sub(r'Please note that this response has been scrubbed for PII.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'\[PATIENT\]|\[DATE\]|\[SSN\]|\[PHONE\]|\[EMAIL\]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _load_conversation_history(session_id: str, patient_id: Optional[str] = None, limit: int = 10) -> List[Any]:
    """Load recent conversation history from session store and convert to messages."""
    if not os.getenv("ENABLE_SESSION_HISTORY", "").lower() == "true":
        logger.debug("Session history injection disabled (set ENABLE_SESSION_HISTORY=true to enable)")
        return []

    if not SESSION_STORE_AVAILABLE:
        logger.warning("Session store not available for history loading")
        return []

    try:
        store = get_session_store()
        fetch_limit = limit * 3 if patient_id else limit
        recent_turns = store.get_recent(session_id, limit=fetch_limit)

        if patient_id:
            filtered_turns = [
                turn for turn in recent_turns
                if turn.get("patient_id") == patient_id or turn.get("patient_id") is None
            ]
        else:
            filtered_turns = recent_turns

        filtered_turns = filtered_turns[:limit]

        history_messages: List[Any] = []
        for turn in reversed(filtered_turns):
            role = turn.get("role", "")
            text = turn.get("text", "")
            turn_patient_id = turn.get("patient_id")
            if not text:
                continue

            if turn_patient_id and patient_id and turn_patient_id == patient_id:
                labeled_text = f"[Previous message about patient {turn_patient_id[:8]}...]\n{text}"
            else:
                labeled_text = text

            if role == "user":
                history_messages.append(HumanMessage(content=labeled_text))
            elif role == "assistant":
                history_messages.append(AIMessage(content=labeled_text))

        logger.info("Loaded %d history messages for session %s", len(history_messages), session_id)
        return history_messages
    except Exception as e:
        logger.error("Error loading history: %s: %s", type(e).__name__, e)
        return []


async def _researcher_node(state: AgentState) -> AgentState:
    max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "15"))
    patient_id = state.get("patient_id")

    # Set patient context for auto-injection into tool calls
    set_patient_context(patient_id)

    system_prompt = get_researcher_prompt(patient_id)

    # Build messages list with conversation history
    messages = [SystemMessage(content=system_prompt)]

    # Automatically inject conversation history if session_id is available
    session_id = state.get("session_id")
    logger.info("Researcher: session_id=%s, patient_id=%s", session_id, patient_id)
    if session_id:
        history_messages = _load_conversation_history(session_id, patient_id=patient_id, limit=10)
        if history_messages:
            logger.info("Injecting %d history messages into researcher context", len(history_messages))
            messages.extend(history_messages)

    # Add current query
    messages.append(HumanMessage(content=state["query"]))

    # Add validator feedback if present (for revision cycles)
    if state.get("validator_output"):
        messages.append(HumanMessage(
            content=(
                "REVISION REQUIRED\n\n"
                "Your previous response was rejected. Fix ONLY these issues:\n\n"
                f"{state['validator_output']}\n\n"
                "Keep everything else the same. Do not rewrite the entire response."
            )
        ))

    # Add greeting instruction if this was a mixed query
    if state.get("should_acknowledge_greeting", False):
        messages.append(SystemMessage(
            content="IMPORTANT: The user included a greeting. Please acknowledge it warmly "
                    "at the beginning of your response before addressing the clinical question."
        ))

    # Inject trajectory if in retry mode (death loop prevention)
    search_attempts = state.get("search_attempts", [])
    empty_count = state.get("empty_search_count", 0)
    current_iteration = state.get("iteration_count", 0)

    if empty_count > 0:
        failed_queries = [f"  - '{a.get('query', 'unknown')}' -> {a.get('results_count', 0)} results"
                          for a in search_attempts[-3:]]
        messages.append(SystemMessage(content=f"""[SYSTEM CONTEXT - Do not echo this in your response]

Previous search attempts returned no useful results:
{chr(10).join(failed_queries)}

ACTION REQUIRED: Try DIFFERENT search terms. Consider:
- Using FHIR resource types: Condition, Observation, MedicationRequest
- Removing specific terms that may not match embeddings
- Broadening the query scope

If you still cannot find data, provide a response stating what you searched for and that no records were found.
DO NOT repeat this system message in your output."""))

    # System-wide step limit check
    if current_iteration >= 12:
        messages.append(SystemMessage(content="""[SYSTEM CONTEXT - Do not echo this in your response]

You are approaching the step limit. Provide your best response NOW based on what you have found.
If you found relevant data, summarize it. If nothing was found, state that clearly.
Do NOT make additional tool calls. DO NOT repeat this system message."""))

    agent = _get_researcher_agent()

    try:
        result = await agent.ainvoke({"messages": messages}, config={"recursion_limit": max_iterations})
        output_messages = result.get("messages", [])
        response_text = _extract_response_text(output_messages)
    except GraphRecursionError as e:
        logger.warning("Hit internal recursion limit (max_iterations=%d): %s", max_iterations, e)
        response_text = (
            f"I encountered a complexity limit after {max_iterations} internal steps while researching this. "
            "I am returning what I have found so far, but it may be incomplete."
        )
        output_messages = []

    # Extract sources from tool outputs
    new_sources = _extract_sources(output_messages)
    all_sources = state.get("sources", []) + new_sources
    tools_called = (state.get("tools_called") or []) + _extract_tool_calls(output_messages)

    # Validate non-empty response and detect echo bugs
    echo_indicators = [
        "RETRY MODE",
        "PREVIOUS FAILED QUERIES",
        "SYSTEM CONTEXT",
        "Do not echo this",
        "ACTION REQUIRED: Try DIFFERENT"
    ]
    is_echo_response = any(indicator in response_text for indicator in echo_indicators)

    if not response_text or len(response_text.strip()) < 20 or is_echo_response:
        if new_sources:
            response_text = (
                f"Based on my search, I found {len(new_sources)} relevant records. "
                "However, I was unable to fully synthesize a response. "
                "Please try rephrasing your question for more specific results."
            )
        else:
            response_text = (
                "I searched the patient records but was unable to generate a complete response. "
                "Please try rephrasing your question or providing more specific clinical terms."
            )

    # Track search attempts for trajectory (death loop prevention)
    search_attempts = list(state.get("search_attempts", []))
    found_results_this_iteration = False

    # Build a map of tool_call_id -> result_count from ToolMessages
    tool_results: Dict[str, int] = {}
    for message in output_messages:
        if isinstance(message, ToolMessage):
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id:
                try:
                    content_str = str(message.content)
                    data = json.loads(content_str)
                    if isinstance(data, dict):
                        chunks = data.get("chunks", [])
                        count = data.get("count", len(chunks) if isinstance(chunks, list) else 0)
                        tool_results[tool_call_id] = count
                        if count > 0:
                            found_results_this_iteration = True
                except (json.JSONDecodeError, Exception):
                    tool_results[tool_call_id] = 0

    # Match tool_calls to their results
    for message in output_messages:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                if call.get("name") == "search_patient_records":
                    args = call.get("args", {})
                    call_id = call.get("id", "")
                    result_count = tool_results.get(call_id, 0)

                    current_attempt = {
                        "query": args.get("query", "unknown"),
                        "patient_id": args.get("patient_id", "unknown"),
                        "results_count": result_count,
                        "iteration": state.get("iteration_count", 0) + 1,
                    }
                    search_attempts.append(current_attempt)
                    logger.debug("Tracked search: '%s' -> %d results", args.get("query", "unknown"), result_count)

                    if result_count > 0:
                        found_results_this_iteration = True

    # Track empty searches
    empty_count = state.get("empty_search_count", 0)
    if not found_results_this_iteration and len(tool_results) > 0:
        empty_count += 1
        logger.info("Empty search count: %d", empty_count)
    elif found_results_this_iteration:
        empty_count = 0

    return {
        **state,
        "researcher_output": response_text,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "tools_called": tools_called,
        "sources": all_sources,
        "search_attempts": search_attempts,
        "empty_search_count": empty_count,
    }


async def _validator_node(state: AgentState) -> AgentState:
    """Validate researcher output with strictness tiers and structured parsing."""
    from agent.output_schemas import parse_validator_output
    from agent.guardrails.validators import validate_output

    # Calculate strictness tier
    current_iter = state.get("iteration_count", 0)
    max_iter = int(os.getenv("AGENT_MAX_ITERATIONS", "15"))
    remaining = max_iter - current_iter

    if remaining > 5:
        strictness_tier = "TIER_STRICT"
    elif remaining > 2:
        strictness_tier = "TIER_RELAXED"
    else:
        strictness_tier = "TIER_EMERGENCY"

    # Run Guardrails AI validation first (if enabled)
    researcher_output = state.get("researcher_output", "")
    guardrails_valid, guardrails_error = validate_output(researcher_output)

    if not guardrails_valid:
        return {
            **state,
            "validator_output": f"Guardrails validation failed:\n{guardrails_error}",
            "validation_result": "FAIL",
            "tools_called": state.get("tools_called") or [],
            "sources": state.get("sources", []),
        }

    # Fast-path: Detect incomplete/fallback responses
    _INCOMPLETE_MARKERS = [
        "unable to generate a complete response",
        "unable to generate complete response",
        "could not find any",
        "no results found",
        "please try rephrasing",
        "please rephrase your question",
    ]
    researcher_lower = researcher_output.lower()
    is_incomplete = any(marker in researcher_lower for marker in _INCOMPLETE_MARKERS)

    if is_incomplete and strictness_tier != "TIER_STRICT":
        return {
            **state,
            "validator_output": "Response indicates no data found. Passing to user (not harmful).",
            "validation_result": "PASS",
            "tools_called": state.get("tools_called") or [],
            "sources": state.get("sources", []),
        }

    # Inject strictness tier into prompt
    patient_id = state.get("patient_id") or "N/A"
    system_prompt = get_validator_prompt().format(
        strictness_tier=strictness_tier,
        patient_id=patient_id,
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Validate the response below.\n\n"
                f"STRICTNESS TIER: {strictness_tier}\n"
                f"REMAINING ATTEMPTS: {remaining}\n\n"
                f"User query:\n{state.get('query', '')}\n\n"
                f"Researcher response:\n{researcher_output}"
            )
        ),
    ]

    agent = _get_validator_agent()
    result = await agent.ainvoke({"messages": messages}, config={"recursion_limit": max_iter})
    output_messages = result.get("messages", [])
    response_text = _extract_response_text(output_messages)

    tools_called = (state.get("tools_called") or []) + _extract_tool_calls(output_messages)

    # Use structured parsing with fallback
    parsed = parse_validator_output(response_text)
    validation_result = parsed.validation_status

    # Handle final_output_override if provided
    final_output = response_text
    if parsed.final_output_override:
        final_output = f"{response_text}\n\n---\nCORRECTED OUTPUT:\n{parsed.final_output_override}"
        if strictness_tier == "TIER_EMERGENCY":
            validation_result = "PASS"

    return {
        **state,
        "validator_output": final_output,
        "validation_result": validation_result,
        "tools_called": tools_called,
        "sources": state.get("sources", []),
    }


async def _respond_node(state: AgentState) -> AgentState:
    """Synthesize the researched information into a user-friendly response."""
    max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "15"))
    system_prompt = get_response_prompt() or get_conversational_prompt()

    researcher_output = state.get("researcher_output", "")
    user_query = state.get("query", "")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"User's question: {user_query}\n\n"
                f"Research findings:\n{researcher_output}\n\n"
                "Synthesize these findings into a clear, conversational response for the user."
            )
        ),
    ]

    # Use response agent to synthesize
    agent = _get_response_agent()
    result = await agent.ainvoke({"messages": messages}, config={"recursion_limit": max_iterations})
    output_messages = result.get("messages", [])
    final_response = _extract_response_text(output_messages)

    # Fallback to researcher output if synthesis fails
    if not final_response or len(final_response.strip()) < 10:
        final_response = researcher_output

    # Clean up any internal details that leaked through
    final_response = _clean_response(final_response)

    # Apply PII masking to final response
    try:
        from agent.pii_masker.factory import create_pii_masker
        masker = create_pii_masker()
        final_response, _ = masker.mask_pii(final_response)
    except Exception:
        pass  # PII masking is best-effort

    return {**state, "final_response": final_response}


def _classify_node(state: AgentState) -> dict:
    """Classify the user query to route to appropriate path."""
    query = state.get("query", "")
    session_id = state.get("session_id", "")

    # Get session context if available
    context = {}
    if SESSION_STORE_AVAILABLE and get_session_store:
        try:
            store = get_session_store()
            summary = store.get_summary(session_id)
            context["last_query_type"] = summary.get("last_query_type")
        except Exception:
            pass

    # Classify query
    classifier = QueryClassifier()
    result = classifier.classify(query, session_context=context)

    # Update session metadata with query type
    if SESSION_STORE_AVAILABLE and get_session_store:
        try:
            store = get_session_store()
            store.update_summary(session_id, {"last_query_type": result.query_type.value})
        except Exception:
            pass

    return {
        **state,
        "query_type": result.query_type.value,
        "classification_confidence": result.confidence,
        "classification_method": result.method,
        "should_acknowledge_greeting": result.should_acknowledge_greeting,
    }


async def _conversational_responder_node(state: AgentState) -> dict:
    """Handle purely conversational queries without RAG."""
    query = state.get("query", "")

    system_prompt = get_conversational_prompt()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    llm = get_llm()
    try:
        response_msg = await llm.ainvoke(messages)
        response = str(response_msg.content)
    except Exception as e:
        response = f"I apologize, but I'm having trouble generating a response right now. (Error: {str(e)})"

    return {
        **state,
        "final_response": response,
        "researcher_output": response,
        "validation_result": "PASS",
    }


def _route_after_classification(state: AgentState) -> str:
    """Route based on query classification."""
    query_type = state.get("query_type", "medical")

    if query_type == "conversational":
        return "conversational_responder"
    # Both "medical" and "mixed" go to researcher
    return "researcher"


def _route_after_validation(state: AgentState) -> str:
    """Route based on validation result."""
    validation_result = state.get("validation_result", "NEEDS_REVISION")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "15"))

    # Conversational queries skip validation
    if state.get("query_type") == "conversational":
        return "respond"

    # PASS = validated answer, go to respond
    if validation_result == "PASS":
        return "respond"

    # Hit max iterations, force respond
    if iteration_count >= max_iterations:
        return "respond"

    # FAIL or NEEDS_REVISION = retry with researcher
    return "researcher"


def create_simple_graph():
    """Create a simple single-agent graph: classify -> researcher -> respond.

    Used for straightforward queries that don't require validation.
    """
    graph = StateGraph(AgentState)
    graph.add_node("classify", _classify_node)
    graph.add_node("researcher", _researcher_node)
    graph.add_node("respond", _respond_node)
    graph.add_node("conversational_responder", _conversational_responder_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", _route_after_classification)
    graph.add_edge("researcher", "respond")
    graph.add_edge("respond", END)
    graph.add_edge("conversational_responder", END)
    return graph.compile()


def create_complex_graph():
    """Create the full multi-agent graph: classify -> researcher -> validator -> respond.

    Used for complex queries requiring validation and iterative refinement.
    """
    graph = StateGraph(AgentState)
    graph.add_node("classify", _classify_node)
    graph.add_node("researcher", _researcher_node)
    graph.add_node("validator", _validator_node)
    graph.add_node("respond", _respond_node)
    graph.add_node("conversational_responder", _conversational_responder_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", _route_after_classification)
    graph.add_edge("researcher", "validator")
    graph.add_conditional_edges("validator", _route_after_validation)
    graph.add_edge("respond", END)
    graph.add_edge("conversational_responder", END)
    return graph.compile()


def create_multi_agent_graph():
    """Create the main graph with configurable complexity.

    Uses AGENT_GRAPH_TYPE environment variable:
    - "simple": classify -> researcher -> respond (fast, no validation)
    - "complex": classify -> researcher -> validator -> respond (validated)

    Defaults to "simple" if not specified.
    """
    graph_type = os.getenv("AGENT_GRAPH_TYPE", "simple").lower()

    if graph_type == "complex":
        logger.info("Creating complex multi-agent graph (classify -> researcher -> validator -> respond)")
        return create_complex_graph()
    else:
        logger.info("Creating simple graph (classify -> researcher -> respond)")
        return create_simple_graph()


# Global agent instance
_AGENT: Any = None


def get_agent() -> Any:
    """Get or create the compiled agent graph."""
    global _AGENT
    if _AGENT is None:
        _AGENT = create_multi_agent_graph()
    return _AGENT
