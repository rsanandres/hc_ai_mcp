"""MCP tools for the HC-AI agent."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from config import is_tool_enabled


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
            k_retrieve: int = 50,
            k_return: int = 10,
        ) -> Dict[str, Any]:
            """Query the HC-AI agent with a natural language question.
            
            Args:
                query: The natural language question to ask.
                session_id: Session identifier for conversation context.
                patient_id: Optional patient ID to filter results.
                k_retrieve: Number of documents to retrieve (default: 50).
                k_return: Number of documents to return after reranking (default: 10).
            
            Returns:
                Dictionary with response, sources, and metadata.
            """
            from agent import get_agent
            from session import get_session_store
            
            if not query.strip():
                return {"error": "Query is required", "status": "error"}
            
            try:
                agent = get_agent()
                max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
                
                state = {
                    "query": query,
                    "session_id": session_id,
                    "patient_id": patient_id,
                    "k_retrieve": k_retrieve,
                    "k_return": k_return,
                    "iteration_count": 0,
                }
                
                result = await agent.ainvoke(state, config={"recursion_limit": max_iterations})
                
                response_text = result.get("final_response") or result.get("researcher_output", "")
                
                # Store in session
                store = get_session_store()
                store.append_turn(session_id, role="user", text=query)
                store.append_turn(
                    session_id,
                    role="assistant",
                    text=response_text,
                    meta={"tool_calls": result.get("tools_called", [])},
                )
                
                return {
                    "query": query,
                    "response": response_text,
                    "sources": result.get("sources", []),
                    "tool_calls": result.get("tools_called", []),
                    "session_id": session_id,
                    "validation_result": result.get("validation_result"),
                    "status": "success",
                }
            except Exception as e:
                return {"error": str(e), "status": "error"}
    
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
            
            try:
                store = get_session_store()
                store.clear_session(session_id)
                return {"status": "cleared", "session_id": session_id}
            except Exception as e:
                return {"status": "error", "error": str(e)}
    
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
