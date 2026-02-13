"""Database module for PostgreSQL + pgvector operations."""

from .vector_store import (
    initialize_vector_store,
    store_chunk,
    store_chunks_batch,
    search_similar_chunks,
    close_connections,
    get_connection_stats,
    get_queue_stats,
    list_patients,
)
from .models import DocumentChunk, ChunkMetadata

__all__ = [
    "initialize_vector_store",
    "store_chunk",
    "store_chunks_batch",
    "search_similar_chunks",
    "close_connections",
    "get_connection_stats",
    "get_queue_stats",
    "list_patients",
    "DocumentChunk",
    "ChunkMetadata",
]
