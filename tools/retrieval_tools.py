"""MCP tools for retrieval and reranking."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Dict, List, Optional

from config import is_tool_enabled


def register_retrieval_tools(mcp: Any, config: Dict[str, Any]) -> None:
    """Register retrieval tools with the MCP server.
    
    Args:
        mcp: The FastMCP server instance.
        config: Configuration dictionary.
    """
    
    if is_tool_enabled(config, "rerank"):
        @mcp.tool()
        async def rerank(
            query: str,
            k_retrieve: int = 50,
            k_return: int = 10,
            filter_metadata: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """Rerank documents by relevance to a query.
            
            Args:
                query: The search query.
                k_retrieve: Number of documents to retrieve from vector store.
                k_return: Number of documents to return after reranking.
                filter_metadata: Optional metadata filters.
            
            Returns:
                Dictionary with query and reranked results.
            """
            from db import search_similar_chunks
            from reranker import get_reranker, get_cache, build_cache_key
            
            try:
                # Search for candidates
                candidates = await search_similar_chunks(
                    query=query,
                    k=k_retrieve,
                    filter_metadata=filter_metadata,
                )
                
                if not candidates:
                    return {"query": query, "results": []}
                
                # Build document IDs for caching
                doc_ids = []
                for idx, doc in enumerate(candidates):
                    doc_id = getattr(doc, "id", None)
                    if not doc_id:
                        meta = doc.metadata or {}
                        doc_id = meta.get("chunkId") or meta.get("resourceId") or f"doc_{idx}"
                    doc_ids.append(str(doc_id))
                
                # Check cache
                cache = get_cache()
                cache_key = build_cache_key(query, doc_ids)
                cached = cache.get(cache_key)
                
                if cached:
                    cached_map = {doc_id: score for doc_id, score in cached}
                    if all(doc_id in cached_map for doc_id in doc_ids):
                        # Use cached scores
                        scored = [(idx, doc, doc_ids[idx]) for idx, doc in enumerate(candidates)]
                        scored.sort(key=lambda x: (-cached_map.get(x[2], 0), x[0]))
                        top_docs = scored[:k_return]
                        results = [
                            {
                                "id": doc_ids[idx],
                                "content": doc.page_content,
                                "metadata": doc.metadata or {},
                            }
                            for idx, doc, _ in top_docs
                        ]
                        return {"query": query, "results": results}
                
                # Rerank
                reranker = get_reranker()
                scored_docs = reranker.rerank_with_scores(query, candidates)
                
                # Cache results
                scored_pairs = []
                for i, (doc, score) in enumerate(scored_docs):
                    if i < len(doc_ids):
                        scored_pairs.append((doc_ids[i], score))
                cache.set(cache_key, scored_pairs)
                
                # Build response
                results = []
                for i, (doc, score) in enumerate(scored_docs[:k_return]):
                    doc_id = doc_ids[i] if i < len(doc_ids) else f"doc_{i}"
                    results.append({
                        "id": doc_id,
                        "content": doc.page_content,
                        "metadata": doc.metadata or {},
                        "score": score,
                    })
                
                return {"query": query, "results": results}
            
            except Exception as e:
                return {"query": query, "results": [], "error": str(e)}
    
    if is_tool_enabled(config, "rerank_with_context"):
        @mcp.tool()
        async def rerank_with_context(
            query: str,
            k_retrieve: int = 50,
            k_return: int = 10,
            include_full_json: bool = False,
            filter_metadata: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """Rerank documents with optional full FHIR bundle context.
            
            Args:
                query: The search query.
                k_retrieve: Number of documents to retrieve.
                k_return: Number of documents to return.
                include_full_json: Include full FHIR bundle JSON.
                filter_metadata: Optional metadata filters.
            
            Returns:
                Dictionary with chunks and optionally full documents.
            """
            # Get reranked chunks using the rerank tool logic
            from db import search_similar_chunks
            from reranker import get_reranker
            
            try:
                candidates = await search_similar_chunks(
                    query=query,
                    k=k_retrieve,
                    filter_metadata=filter_metadata,
                )
                
                if not candidates:
                    return {"query": query, "chunks": [], "full_documents": []}
                
                reranker = get_reranker()
                scored_docs = reranker.rerank_with_scores(query, candidates)
                
                chunks = []
                patient_ids = set()
                
                for i, (doc, score) in enumerate(scored_docs[:k_return]):
                    doc_id = getattr(doc, "id", None)
                    if not doc_id:
                        meta = doc.metadata or {}
                        doc_id = meta.get("chunkId") or f"doc_{i}"
                    
                    chunks.append({
                        "id": str(doc_id),
                        "content": doc.page_content,
                        "metadata": doc.metadata or {},
                        "score": score,
                    })
                    
                    if doc.metadata and doc.metadata.get("patientId"):
                        patient_ids.add(doc.metadata["patientId"])
                
                result = {"query": query, "chunks": chunks, "full_documents": []}
                
                # TODO: If include_full_json, fetch full FHIR bundles
                # This would require additional storage/retrieval logic
                
                return result
            
            except Exception as e:
                return {"query": query, "chunks": [], "full_documents": [], "error": str(e)}
    
    if is_tool_enabled(config, "batch_rerank"):
        @mcp.tool()
        async def batch_rerank(
            items: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
            """Batch rerank multiple queries in parallel.
            
            Args:
                items: List of rerank requests, each with query, k_retrieve, k_return.
            
            Returns:
                Dictionary with list of rerank results.
            """
            from db import search_similar_chunks
            from reranker import get_reranker
            
            async def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
                query = item.get("query", "")
                k_retrieve = item.get("k_retrieve", 50)
                k_return = item.get("k_return", 10)
                filter_metadata = item.get("filter_metadata")
                
                try:
                    candidates = await search_similar_chunks(
                        query=query,
                        k=k_retrieve,
                        filter_metadata=filter_metadata,
                    )
                    
                    if not candidates:
                        return {"query": query, "results": []}
                    
                    reranker = get_reranker()
                    scored_docs = reranker.rerank_with_scores(query, candidates)
                    
                    results = []
                    for i, (doc, score) in enumerate(scored_docs[:k_return]):
                        doc_id = getattr(doc, "id", None)
                        if not doc_id:
                            meta = doc.metadata or {}
                            doc_id = meta.get("chunkId") or f"doc_{i}"
                        results.append({
                            "id": str(doc_id),
                            "content": doc.page_content,
                            "metadata": doc.metadata or {},
                            "score": score,
                        })
                    
                    return {"query": query, "results": results}
                
                except Exception as e:
                    return {"query": query, "results": [], "error": str(e)}
            
            try:
                tasks = [process_item(item) for item in items]
                results = await asyncio.gather(*tasks)
                return {"items": list(results)}
            except Exception as e:
                return {"items": [], "error": str(e)}
    
    # Session tools
    if is_tool_enabled(config, "session_append_turn"):
        @mcp.tool()
        async def session_append_turn(
            session_id: str,
            role: str,
            text: str,
            meta: Optional[Dict[str, Any]] = None,
            patient_id: Optional[str] = None,
            return_limit: int = 10,
        ) -> Dict[str, Any]:
            """Append a conversation turn to session history.
            
            Args:
                session_id: Session identifier.
                role: Role of the speaker (e.g., "user", "assistant").
                text: Text content of the turn.
                meta: Additional metadata.
                patient_id: Optional patient ID.
                return_limit: Number of recent turns to return.
            
            Returns:
                Session state with recent turns and summary.
            """
            from session import get_session_store
            
            try:
                store = get_session_store()
                store.append_turn(
                    session_id=session_id,
                    role=role,
                    text=text,
                    meta=meta,
                    patient_id=patient_id,
                )
                
                recent = store.get_recent(session_id, limit=return_limit)
                summary = store.get_summary(session_id)
                
                return {
                    "session_id": session_id,
                    "recent_turns": recent,
                    "summary": summary,
                }
            except Exception as e:
                return {"session_id": session_id, "error": str(e)}
    
    if is_tool_enabled(config, "session_get"):
        @mcp.tool()
        async def session_get(
            session_id: str,
            limit: int = 10,
        ) -> Dict[str, Any]:
            """Get session state including recent turns and summary.
            
            Args:
                session_id: Session identifier.
                limit: Maximum number of recent turns to return.
            
            Returns:
                Session state dictionary.
            """
            from session import get_session_store
            
            try:
                store = get_session_store()
                recent = store.get_recent(session_id, limit=limit)
                summary = store.get_summary(session_id)
                
                return {
                    "session_id": session_id,
                    "recent_turns": recent,
                    "summary": summary,
                }
            except Exception as e:
                return {"session_id": session_id, "error": str(e)}
    
    if is_tool_enabled(config, "session_update_summary"):
        @mcp.tool()
        async def session_update_summary(
            session_id: str,
            summary: Dict[str, Any],
            patient_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Update the session summary.
            
            Args:
                session_id: Session identifier.
                summary: Summary data to store.
                patient_id: Optional patient ID.
            
            Returns:
                Updated summary.
            """
            from session import get_session_store
            
            try:
                store = get_session_store()
                store.update_summary(
                    session_id=session_id,
                    summary=summary,
                    patient_id=patient_id,
                )
                updated = store.get_summary(session_id)
                
                return {"session_id": session_id, "summary": updated}
            except Exception as e:
                return {"session_id": session_id, "error": str(e)}
    
    if is_tool_enabled(config, "session_clear"):
        @mcp.tool()
        async def session_clear(session_id: str) -> Dict[str, str]:
            """Clear all session data for a session ID.
            
            Args:
                session_id: Session identifier.
            
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
