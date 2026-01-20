"""MCP tools for embeddings and ingestion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import is_tool_enabled


def register_embeddings_tools(mcp: Any, config: Dict[str, Any]) -> None:
    """Register embeddings tools with the MCP server.
    
    Args:
        mcp: The FastMCP server instance.
        config: Configuration dictionary.
    """
    
    if is_tool_enabled(config, "ingest"):
        @mcp.tool()
        async def ingest(
            id: str,
            resource_type: str,
            content: str,
            patient_id: str = "unknown",
            full_url: str = "",
            resource_json: str = "",
            source_file: str = "",
        ) -> Dict[str, Any]:
            """Ingest a clinical note for chunking and embedding.
            
            Args:
                id: Unique identifier for the resource.
                resource_type: FHIR resource type (e.g., "Observation", "Condition").
                content: Text content of the resource.
                patient_id: Patient identifier.
                full_url: Full URL of the resource.
                resource_json: Original JSON for chunking (optional).
                source_file: Source file path (optional).
            
            Returns:
                Ingest result with status and chunk counts.
            """
            from embeddings import process_and_store, IngestRequest
            
            if not content or not content.strip():
                return {
                    "id": id,
                    "status": "error",
                    "error": "Content cannot be empty",
                }
            
            try:
                request = IngestRequest(
                    id=id,
                    resourceType=resource_type,
                    content=content,
                    patientId=patient_id,
                    fullUrl=full_url,
                    resourceJson=resource_json,
                    sourceFile=source_file,
                )
                
                result = await process_and_store(request)
                return result.to_dict()
            
            except Exception as e:
                return {
                    "id": id,
                    "status": "error",
                    "error": str(e),
                }
    
    if is_tool_enabled(config, "embeddings_health"):
        @mcp.tool()
        async def embeddings_health() -> Dict[str, Any]:
            """Check embeddings service health and configuration.
            
            Returns:
                Health status and configuration details.
            """
            from embeddings import test_connection
            
            status = test_connection()
            
            return {
                "status": "ok" if status.get("ok") else "error",
                "provider": status.get("provider"),
                "model": status.get("model"),
                "embed_dimensions": status.get("embed_dimensions"),
                "errors": status.get("errors", []),
            }
    
    if is_tool_enabled(config, "db_stats"):
        @mcp.tool()
        async def db_stats() -> Dict[str, Any]:
            """Get database connection and queue statistics.
            
            Returns:
                Database and queue statistics.
            """
            from db import get_connection_stats
            
            try:
                stats = await get_connection_stats()
                return stats
            except Exception as e:
                return {"error": str(e)}
    
    if is_tool_enabled(config, "db_queue"):
        @mcp.tool()
        async def db_queue() -> Dict[str, Any]:
            """Get ingestion queue status.
            
            Returns:
                Queue statistics.
            """
            from db import get_queue_stats
            
            try:
                stats = await get_queue_stats()
                return stats
            except Exception as e:
                return {"error": str(e)}
    
    if is_tool_enabled(config, "db_errors"):
        @mcp.tool()
        async def db_errors(
            limit: int = 100,
            offset: int = 0,
            error_type: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Get error logs with optional filtering.
            
            Args:
                limit: Maximum number of records to return.
                offset: Pagination offset.
                error_type: Filter by error type (validation, fatal, max_retries, queue_full).
            
            Returns:
                Error logs and pagination info.
            """
            from db.vector_store import get_error_logs
            
            try:
                errors = await get_error_logs(
                    limit=limit,
                    offset=offset,
                    error_type=error_type,
                )
                
                return {
                    "errors": errors,
                    "limit": limit,
                    "offset": offset,
                    "count": len(errors),
                }
            except Exception as e:
                return {"errors": [], "error": str(e)}
