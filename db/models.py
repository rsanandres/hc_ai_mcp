"""Database models for vector store operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    
    patient_id: str = ""
    resource_id: str = ""
    resource_type: str = ""
    full_url: str = ""
    source_file: str = ""
    chunk_id: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    chunk_size: int = 0
    effective_date: Optional[str] = None
    status: Optional[str] = None
    last_updated: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage (snake_case keys)."""
        result = {
            "patient_id": self.patient_id,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "full_url": self.full_url,
            "source_file": self.source_file,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "chunk_size": self.chunk_size,
        }
        if self.effective_date:
            result["effective_date"] = self.effective_date
        if self.status:
            result["status"] = self.status
        if self.last_updated:
            result["last_updated"] = self.last_updated
        result.update(self.extra)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        """Create from dictionary (supports snake_case keys)."""
        known_keys = {
            "patient_id", "resource_id", "resource_type", "full_url", "source_file",
            "chunk_id", "chunk_index", "total_chunks", "chunk_size",
            "effective_date", "status", "last_updated",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            patient_id=data.get("patient_id", ""),
            resource_id=data.get("resource_id", ""),
            resource_type=data.get("resource_type", ""),
            full_url=data.get("full_url", ""),
            source_file=data.get("source_file", ""),
            chunk_id=data.get("chunk_id", ""),
            chunk_index=data.get("chunk_index", 0),
            total_chunks=data.get("total_chunks", 1),
            chunk_size=data.get("chunk_size", 0),
            effective_date=data.get("effective_date"),
            status=data.get("status"),
            last_updated=data.get("last_updated"),
            extra=extra,
        )


@dataclass
class DocumentChunk:
    """A chunk of a document with its embedding."""
    
    chunk_id: str
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "score": self.score,
        }


@dataclass
class QueuedChunk:
    """A chunk queued for retry."""
    
    chunk_text: str
    chunk_id: str
    metadata: Dict[str, Any]
    retry_count: int = 0
    first_queued_at: float = 0.0
