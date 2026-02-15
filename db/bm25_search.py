"""BM25 full-text search using PostgreSQL tsvector.

This module provides keyword-based search using PostgreSQL's built-in
full-text search capabilities, complementing semantic vector search.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from langchain_core.documents import Document

from logging_config import get_logger

logger = get_logger("atlas.bm25")


# Schema and table configuration
SCHEMA_NAME = os.getenv("DB_SCHEMA_NAME", "public")
TABLE_NAME = os.getenv("DB_TABLE_NAME", "hc_ai_chunks")

# Allowed metadata keys for filtering (prevents SQL injection via arbitrary keys)
ALLOWED_METADATA_KEYS = {"patient_id", "resource_type", "effective_date", "encounter_id", "status"}


async def bm25_search(
    query: str,
    k: int = 50,
    filter_metadata: Optional[Dict[str, Any]] = None,
    engine=None,
) -> List[Document]:
    """
    Perform BM25 full-text search using PostgreSQL ts_rank.

    Args:
        query: Search query (will be converted to tsquery)
        k: Number of results to return
        filter_metadata: Optional metadata filters (e.g., {"patient_id": "..."})
        engine: SQLAlchemy async engine (if not provided, uses global)

    Returns:
        List of Document objects sorted by BM25 relevance score
    """
    # Import here to avoid circular imports
    from db.vector_store import _engine, initialize_vector_store

    if engine is None:
        if _engine is None:
            await initialize_vector_store()
        from db.vector_store import _engine
        engine = _engine

    if not engine:
        return []

    # Build the base query with ts_rank for BM25-style scoring
    # Use to_tsquery with OR logic (|) for better recall on natural language queries
    # plainto_tsquery uses AND logic which fails for long questions

    # Simple keyword extraction (split by space, filter small words)
    # in a real app query parsing would be more robust
    keywords = [w for w in query.replace('?', '').replace('.', '').split() if len(w) > 2]
    if not keywords:
        ts_query_func = "plainto_tsquery"
        query_param = query
    else:
        ts_query_func = "to_tsquery"
        query_param = " | ".join(keywords)

    base_sql = f"""
        SELECT
            langchain_id,
            content,
            langchain_metadata,
            ts_rank(ts_content, {ts_query_func}('english', :query)) as rank
        FROM "{SCHEMA_NAME}"."{TABLE_NAME}"
        WHERE ts_content @@ {ts_query_func}('english', :query)
    """

    params: Dict[str, Any] = {"query": query_param, "k": k}

    # Add metadata filters if provided
    where_clauses = []
    if filter_metadata:
        for key, value in filter_metadata.items():
            if key not in ALLOWED_METADATA_KEYS:
                continue
            param_name = f"meta_{key}"
            where_clauses.append(
                f"langchain_metadata->>'{key}' = :{param_name}"
            )
            params[param_name] = value

    if where_clauses:
        base_sql += " AND " + " AND ".join(where_clauses)

    # Order by rank and limit
    base_sql += """
        ORDER BY rank DESC
        LIMIT :k
    """

    try:
        async with engine.begin() as conn:
            result = await conn.execute(text(base_sql), params)
            rows = result.fetchall()

            documents = []
            for row in rows:
                # Handle both tuple and mapping access
                if hasattr(row, '_mapping'):
                    langchain_id = row._mapping['langchain_id']
                    content = row._mapping['content']
                    metadata = row._mapping['langchain_metadata'] or {}
                    rank = row._mapping['rank']
                else:
                    langchain_id, content, metadata, rank = row

                # Add BM25 score to metadata for debugging
                if isinstance(metadata, dict):
                    metadata = {**metadata, "_bm25_score": float(rank)}

                doc = Document(
                    id=str(langchain_id),
                    page_content=content or "",
                    metadata=metadata,
                )
                documents.append(doc)

            return documents

    except Exception as e:
        logger.error("BM25 search error: %s", e)
        return []


async def bm25_search_with_phrase(
    query: str,
    k: int = 50,
    filter_metadata: Optional[Dict[str, Any]] = None,
    engine=None,
) -> List[Document]:
    """
    Perform BM25 search with phrase matching support.

    Uses phraseto_tsquery for queries that should match as phrases.
    Better for exact code matching (e.g., "E11.9", "LOINC 2339-0").
    """
    from db.vector_store import _engine, initialize_vector_store

    if engine is None:
        if _engine is None:
            await initialize_vector_store()
        from db.vector_store import _engine
        engine = _engine

    if not engine:
        return []

    # For short queries or codes, use websearch_to_tsquery which handles special chars
    base_sql = f"""
        SELECT
            langchain_id,
            content,
            langchain_metadata,
            ts_rank(ts_content, websearch_to_tsquery('english', :query)) as rank
        FROM "{SCHEMA_NAME}"."{TABLE_NAME}"
        WHERE ts_content @@ websearch_to_tsquery('english', :query)
    """

    params: Dict[str, Any] = {"query": query, "k": k}

    # Add metadata filters
    where_clauses = []
    if filter_metadata:
        for key, value in filter_metadata.items():
            if key not in ALLOWED_METADATA_KEYS:
                continue
            param_name = f"meta_{key}"
            where_clauses.append(
                f"langchain_metadata->>'{key}' = :{param_name}"
            )
            params[param_name] = value

    if where_clauses:
        base_sql += " AND " + " AND ".join(where_clauses)

    base_sql += """
        ORDER BY rank DESC
        LIMIT :k
    """

    try:
        async with engine.begin() as conn:
            result = await conn.execute(text(base_sql), params)
            rows = result.fetchall()

            documents = []
            for row in rows:
                if hasattr(row, '_mapping'):
                    langchain_id = row._mapping['langchain_id']
                    content = row._mapping['content']
                    metadata = row._mapping['langchain_metadata'] or {}
                    rank = row._mapping['rank']
                else:
                    langchain_id, content, metadata, rank = row

                if isinstance(metadata, dict):
                    metadata = {**metadata, "_bm25_score": float(rank)}

                doc = Document(
                    id=str(langchain_id),
                    page_content=content or "",
                    metadata=metadata,
                )
                documents.append(doc)

            return documents

    except Exception as e:
        logger.error("BM25 phrase search error: %s", e)
        return []
