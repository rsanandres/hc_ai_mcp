"""MCP tools for the HC-AI agent."""

from __future__ import annotations

import os
import asyncio
from typing import Any, Dict, List, Optional

from config import is_tool_enabled
from logging_config import get_logger
from tools.utils import error_response, get_timeout, validate_k, validate_non_empty

logger = get_logger("hc_ai.tools.agent")


def register_agent_tools(mcp: Any, config: Dict[str, Any]) -> None:
    """Register agent tools with the MCP server.
    
    Args:
        mcp: The FastMCP server instance.
        config: Configuration dictionary.
    """
    
    if is_tool_enabled(config, "agent_query"):
        @mcp.tool()
        async def agent_query(
            query: str,
            session_id: str,
            patient_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Query the HC-AI agent with a natural language question.

            The agent uses a multi-agent workflow with query classification,
            tool-based retrieval (hybrid search + reranking), validation,
            and response synthesis.

            Args:
                query: The natural language question to ask.
                session_id: Session identifier for conversation context.
                patient_id: Optional patient ID to filter results.

            Returns:
                Dictionary with response, sources, and metadata.
            """
            from agent import get_agent
            from session import get_session_store

            error = validate_non_empty("query", query)
            if error:
                return error_response("VALIDATION_ERROR", error)
            error = validate_non_empty("session_id", session_id)
            if error:
                return error_response("VALIDATION_ERROR", error)

            try:
                agent = get_agent()
                max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "15"))

                state = {
                    "query": query,
                    "session_id": session_id,
                    "patient_id": patient_id,
                    "iteration_count": 0,
                }

                timeout = get_timeout("AGENT_TIMEOUT", 120.0)
                result = await asyncio.wait_for(
                    agent.ainvoke(state, config={"recursion_limit": max_iterations}),
                    timeout=timeout,
                )

                response_text = result.get("final_response") or result.get("researcher_output", "")

                # Store in session
                try:
                    store = get_session_store()
                    store.append_turn(
                        session_id, role="user", text=query,
                        patient_id=patient_id,
                    )
                    store.append_turn(
                        session_id,
                        role="assistant",
                        text=response_text,
                        meta={"tool_calls": result.get("tools_called", [])},
                        patient_id=patient_id,
                    )
                except Exception as e:
                    logger.warning("Failed to store session turn: %s", e)

                return {
                    "query": query,
                    "response": response_text,
                    "sources": result.get("sources", []),
                    "tool_calls": result.get("tools_called", []),
                    "session_id": session_id,
                    "patient_id": patient_id,
                    "query_type": result.get("query_type"),
                    "validation_result": result.get("validation_result"),
                    "status": "success",
                }
            except asyncio.TimeoutError:
                return error_response(
                    "TIMEOUT_ERROR",
                    "Agent request timed out",
                    {"timeout_seconds": get_timeout("AGENT_TIMEOUT", 120.0)},
                )
            except Exception as e:
                logger.error("agent_query failed: %s", e)
                return error_response("LLM_ERROR", str(e))
    
    if is_tool_enabled(config, "agent_clear_session"):
        @mcp.tool()
        async def agent_clear_session(session_id: str) -> Dict[str, str]:
            """Clear agent session history for a given session ID.
            
            Args:
                session_id: The session ID to clear.
            
            Returns:
                Status dictionary.
            """
            from session import get_session_store
            
            error = validate_non_empty("session_id", session_id)
            if error:
                return error_response("VALIDATION_ERROR", error)
            try:
                store = get_session_store()
                store.clear_session(session_id)
                return {"status": "cleared", "session_id": session_id}
            except Exception as e:
                logger.error("agent_clear_session failed: %s", e)
                return error_response("DB_ERROR", str(e), {"session_id": session_id})
    
    if is_tool_enabled(config, "agent_health"):
        @mcp.tool()
        async def agent_health() -> Dict[str, Any]:
            """Check agent health status and configuration.
            
            Returns:
                Health status and configuration details.
            """
            from embeddings import test_connection
            
            health = {
                "status": "ok",
                "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
                "llm_model": os.getenv("LLM_MODEL", "llama3"),
            }
            
            # Check embedding connection
            embed_status = test_connection()
            health["embeddings"] = {
                "provider": embed_status.get("provider"),
                "ok": embed_status.get("ok", False),
            }
            
            if not embed_status.get("ok"):
                health["status"] = "degraded"
                health["embeddings"]["errors"] = embed_status.get("errors", [])
            
            return health
