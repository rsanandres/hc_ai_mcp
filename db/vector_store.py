"""PostgreSQL + pgvector operations for vector storage and search."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document

# Load environment from package directory
load_dotenv()

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "hc_ai")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

TABLE_NAME = os.getenv("DB_TABLE_NAME", "hc_ai_chunks")
SCHEMA_NAME = os.getenv("DB_SCHEMA_NAME", "public")
VECTOR_SIZE = int(os.getenv("DB_VECTOR_SIZE", "1024"))

# Connection pool configuration
MAX_POOL_SIZE = int(os.getenv("DB_MAX_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "5"))
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))

# Queue configuration
QUEUE_MAX_SIZE = int(os.getenv("CHUNK_QUEUE_MAX_SIZE", "1000"))
MAX_RETRIES = int(os.getenv("CHUNK_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.getenv("CHUNK_RETRY_BASE_DELAY", "1.0"))
RETRY_MAX_DELAY = float(os.getenv("CHUNK_RETRY_MAX_DELAY", "60.0"))

# Error classification keywords for retry logic
RETRYABLE_KEYWORDS = [
    "too many clients",
    "connection",
    "timeout",
    "deadlock",
    "lock timeout",
    "connection refused",
]

# Global state
_engine = None
_pg_engine = None
_vector_store = None
_queue: Optional[asyncio.Queue] = None
_queue_worker_task = None
_queue_stats = {
    "queued": 0,
    "processed": 0,
    "failed": 0,
    "retries": 0,
}
_error_log: List[Dict[str, Any]] = []


class CustomEmbeddings:
    """Custom embeddings wrapper that uses the embedder module."""
    
    def __init__(self):
        from embeddings.embedder import get_embedding, get_embeddings
        self._get_embedding = get_embedding
        self._get_embeddings = get_embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self._get_embeddings(texts)
        if embeddings is None:
            raise ValueError("Failed to generate embeddings")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self._get_embedding(text)
        if embedding is None:
            raise ValueError(f"Failed to generate embedding for query")
        return embedding


def _classify_error(exc: Exception) -> str:
    """Classify an error as retryable, duplicate, or fatal."""
    msg = str(exc).lower()
    for kw in RETRYABLE_KEYWORDS:
        if kw in msg:
            return "retryable"
    if "duplicate key" in msg or "unique constraint" in msg or "conflict" in msg:
        return "duplicate"
    return "fatal"


def _validate_chunk(chunk_text: str, chunk_id: str, metadata: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a chunk before storage."""
    if not chunk_text or not chunk_text.strip():
        return False, "Empty chunk text"
    try:
        uuid.UUID(chunk_id)
    except Exception:
        return False, "Invalid chunk_id (must be UUID)"
    if not isinstance(metadata, dict):
        return False, "Metadata must be a dict"
    if len(chunk_text) > 50000:
        return False, "Chunk text too large (max 50000 chars)"
    return True, ""


async def _log_error(
    chunk_id: str,
    error_type: str,
    error_message: str,
    metadata: Optional[Dict[str, Any]] = None,
    retry_count: int = 0,
) -> None:
    """Log an error to the in-memory error log."""
    _error_log.append({
        "timestamp": time.time(),
        "chunk_id": chunk_id,
        "error_type": error_type,
        "error_message": error_message,
        "metadata": metadata or {},
        "retry_count": retry_count,
    })
    # Keep only last 1000 errors
    if len(_error_log) > 1000:
        _error_log.pop(0)


async def initialize_vector_store():
    """Initialize the PostgreSQL vector store with connection pooling."""
    global _engine, _pg_engine, _vector_store
    
    if _vector_store is not None:
        return _vector_store
    
    # Validate required environment variables
    if not DB_PASSWORD:
        raise ValueError("DB_PASSWORD environment variable is required")
    
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
        from langchain_postgres import PGVectorStore, PGEngine
    except ImportError as e:
        raise ImportError(
            "Required packages not installed. Run: pip install sqlalchemy asyncpg langchain-postgres"
        ) from e
    
    # Create engine if not exists
    if _engine is None:
        connection_string = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        _engine = create_async_engine(
            connection_string,
            pool_size=MAX_POOL_SIZE,
            max_overflow=MAX_OVERFLOW,
            pool_timeout=POOL_TIMEOUT,
            pool_pre_ping=True,
            echo=False,
        )
        _pg_engine = PGEngine.from_engine(engine=_engine)
    
    # Create schema if it doesn't exist
    async with _engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA_NAME}"'))
        # Enable pgvector extension
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    
    # Check if table exists
    async with _engine.begin() as conn:
        result = await conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = :schema_name
                    AND table_name = :table_name
                )
            """),
            {"schema_name": SCHEMA_NAME, "table_name": TABLE_NAME}
        )
        table_exists = result.scalar_one()
    
    # Create table if it doesn't exist
    if not table_exists:
        await _pg_engine.ainit_vectorstore_table(
            table_name=TABLE_NAME,
            vector_size=VECTOR_SIZE,
            schema_name=SCHEMA_NAME,
        )
    
    # Create embeddings instance and vector store
    embedding = CustomEmbeddings()
    _vector_store = await PGVectorStore.create(
        engine=_pg_engine,
        table_name=TABLE_NAME,
        schema_name=SCHEMA_NAME,
        embedding_service=embedding,
    )
    
    # Start queue worker
    await _start_queue_worker()
    
    return _vector_store


async def _start_queue_worker():
    """Start the background queue worker."""
    global _queue, _queue_worker_task
    
    if _queue is None:
        _queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    
    if _queue_worker_task is None or _queue_worker_task.done():
        _queue_worker_task = asyncio.create_task(_queue_worker())


async def _queue_worker():
    """Background worker that retries queued chunks with exponential backoff."""
    from db.models import QueuedChunk
    
    while True:
        try:
            queued_chunk: QueuedChunk = await _queue.get()
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.5)
            continue
        
        delay = min(RETRY_BASE_DELAY * (2 ** queued_chunk.retry_count), RETRY_MAX_DELAY)
        if queued_chunk.retry_count > 0:
            await asyncio.sleep(delay)
        
        try:
            success = await _store_chunk_direct(
                chunk_text=queued_chunk.chunk_text,
                chunk_id=queued_chunk.chunk_id,
                metadata=queued_chunk.metadata,
            )
            if success:
                _queue_stats["processed"] += 1
            else:
                raise Exception("Unknown failure during store_chunk_direct")
        except Exception as e:
            classification = _classify_error(e)
            if classification == "retryable" and queued_chunk.retry_count < MAX_RETRIES:
                queued_chunk.retry_count += 1
                _queue_stats["retries"] += 1
                try:
                    await _queue.put(queued_chunk)
                except asyncio.QueueFull:
                    _queue_stats["failed"] += 1
                    await _log_error(
                        chunk_id=queued_chunk.chunk_id,
                        error_type="queue_full",
                        error_message=f"Queue full after {queued_chunk.retry_count} retries: {str(e)}",
                        metadata=queued_chunk.metadata,
                        retry_count=queued_chunk.retry_count,
                    )
            else:
                _queue_stats["failed"] += 1
                error_type = "max_retries" if queued_chunk.retry_count >= MAX_RETRIES else "fatal"
                await _log_error(
                    chunk_id=queued_chunk.chunk_id,
                    error_type=error_type,
                    error_message=str(e),
                    metadata=queued_chunk.metadata,
                    retry_count=queued_chunk.retry_count,
                )
        finally:
            _queue.task_done()


async def _store_chunk_direct(
    chunk_text: str,
    chunk_id: str,
    metadata: Dict[str, Any],
) -> bool:
    """Store a single chunk directly without queuing."""
    is_valid, msg = _validate_chunk(chunk_text, chunk_id, metadata)
    if not is_valid:
        await _log_error(
            chunk_id=chunk_id,
            error_type="validation",
            error_message=msg,
            metadata=metadata,
        )
        raise ValueError(msg)
    
    vector_store = await initialize_vector_store()
    doc = Document(
        id=chunk_id,
        page_content=chunk_text,
        metadata=metadata,
    )
    await vector_store.aadd_documents([doc])
    return True


async def store_chunk(
    chunk_text: str,
    chunk_id: str,
    metadata: Dict[str, Any],
    use_queue: bool = True,
) -> bool:
    """Store a single chunk with optional queuing on retryable errors.
    
    Args:
        chunk_text: The text content of the chunk.
        chunk_id: Unique identifier (UUID) for the chunk.
        metadata: Dictionary of metadata to store with the chunk.
        use_queue: Whether to queue failed chunks for retry.
    
    Returns:
        True if stored successfully, False if queued for retry.
    """
    from db.models import QueuedChunk
    
    try:
        return await _store_chunk_direct(chunk_text, chunk_id, metadata)
    except Exception as e:
        classification = _classify_error(e)
        if classification == "duplicate":
            return True  # Treat duplicates as success
        if use_queue and classification == "retryable":
            try:
                await _start_queue_worker()
                q_item = QueuedChunk(
                    chunk_text=chunk_text,
                    chunk_id=chunk_id,
                    metadata=metadata,
                    first_queued_at=time.time(),
                )
                await _queue.put(q_item)
                _queue_stats["queued"] += 1
                return False
            except asyncio.QueueFull:
                _queue_stats["failed"] += 1
                await _log_error(
                    chunk_id=chunk_id,
                    error_type="queue_full",
                    error_message=f"Queue full: {str(e)}",
                    metadata=metadata,
                )
                return False
        await _log_error(
            chunk_id=chunk_id,
            error_type="fatal",
            error_message=str(e),
            metadata=metadata,
        )
        raise


async def store_chunks_batch(chunks: List[Dict[str, Any]]) -> int:
    """Store multiple chunks in a batch operation.
    
    Args:
        chunks: List of dictionaries with 'text', 'id', and 'metadata' keys.
    
    Returns:
        Number of successfully stored chunks.
    """
    stored = 0
    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_id = chunk.get("id") or str(uuid.uuid4())
        metadata = chunk.get("metadata", {})
        try:
            success = await store_chunk(chunk_text, chunk_id, metadata, use_queue=True)
            if success:
                stored += 1
        except Exception:
            continue
    return stored


async def search_similar_chunks(
    query: str,
    k: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Search for similar chunks using semantic similarity.
    
    Args:
        query: Search query text.
        k: Number of results to return.
        filter_metadata: Optional metadata filters.
    
    Returns:
        List of similar Document objects.
    """
    try:
        vector_store = await initialize_vector_store()
        results = await vector_store.asimilarity_search(
            query=query,
            k=k,
            filter=filter_metadata,
        )
        return results
    except Exception as e:
        print(f"Error searching chunks: {e}")
        return []


async def get_connection_stats() -> Dict[str, Any]:
    """Get database connection and pool statistics."""
    stats: Dict[str, Any] = {
        "active_connections": 0,
        "max_connections": 0,
        "pool_size": MAX_POOL_SIZE,
        "pool_overflow": MAX_OVERFLOW,
        "pool_checked_out": 0,
        "pool_checked_in": 0,
        "queue_size": _queue.qsize() if _queue else 0,
        "queue_stats": _queue_stats.copy(),
    }
    
    if _engine is None:
        return stats
    
    try:
        from sqlalchemy import text
        async with _engine.begin() as conn:
            result = await conn.execute(
                text("""
                    SELECT 
                        count(*) as active_connections,
                        (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_connections
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                    AND state = 'active'
                """)
            )
            row = result.fetchone()
            if row:
                stats["active_connections"] = row[0]
                stats["max_connections"] = row[1]
        
        if hasattr(_engine, "pool"):
            pool = _engine.pool
            stats["pool_checked_out"] = pool.checkedout()
            stats["pool_checked_in"] = pool.checkedin()
    except Exception as e:
        stats["error"] = str(e)
    
    return stats


async def get_queue_stats() -> Dict[str, Any]:
    """Get queue statistics."""
    return {
        "memory_queue_size": _queue.qsize() if _queue else 0,
        "stats": _queue_stats.copy(),
    }


async def get_error_logs(
    limit: int = 100,
    offset: int = 0,
    error_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get error logs with optional filtering.
    
    Args:
        limit: Maximum number of records to return.
        offset: Offset for pagination.
        error_type: Filter by error type.
    
    Returns:
        List of error log records.
    """
    logs = _error_log.copy()
    
    if error_type:
        logs = [log for log in logs if log.get("error_type") == error_type]
    
    # Sort by timestamp descending (newest first)
    logs.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    
    return logs[offset:offset + limit]


async def close_connections():
    """Close database connections and cleanup."""
    global _engine, _vector_store, _queue_worker_task
    
    if _queue_worker_task:
        _queue_worker_task.cancel()
        try:
            await _queue_worker_task
        except asyncio.CancelledError:
            pass
        _queue_worker_task = None
    
    if _engine:
        await _engine.dispose()
        _engine = None
        _vector_store = None
