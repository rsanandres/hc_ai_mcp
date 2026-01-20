"""Embeddings module for chunking and embedding generation."""

from .embedder import get_embedding, get_embeddings, test_connection
from .chunker import (
    semantic_chunking,
    recursive_json_chunking,
    parent_child_chunking,
    recursive_text_chunking,
)
from .ingest import process_and_store, IngestRequest

__all__ = [
    "get_embedding",
    "get_embeddings",
    "test_connection",
    "semantic_chunking",
    "recursive_json_chunking",
    "parent_child_chunking",
    "recursive_text_chunking",
    "process_and_store",
    "IngestRequest",
]
