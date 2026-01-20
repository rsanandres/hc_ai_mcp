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
        """Convert to dictionary for storage."""
        result = {
            "patientId": self.patient_id,
            "resourceId": self.resource_id,
            "resourceType": self.resource_type,
            "fullUrl": self.full_url,
            "sourceFile": self.source_file,
            "chunkId": self.chunk_id,
            "chunkIndex": self.chunk_index,
            "totalChunks": self.total_chunks,
            "chunkSize": self.chunk_size,
        }
        if self.effective_date:
            result["effectiveDate"] = self.effective_date
        if self.status:
            result["status"] = self.status
        if self.last_updated:
            result["lastUpdated"] = self.last_updated
        result.update(self.extra)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        """Create from dictionary."""
        known_keys = {
            "patientId", "resourceId", "resourceType", "fullUrl", "sourceFile",
            "chunkId", "chunkIndex", "totalChunks", "chunkSize",
            "effectiveDate", "status", "lastUpdated"
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            patient_id=data.get("patientId", ""),
            resource_id=data.get("resourceId", ""),
            resource_type=data.get("resourceType", ""),
            full_url=data.get("fullUrl", ""),
            source_file=data.get("sourceFile", ""),
            chunk_id=data.get("chunkId", ""),
            chunk_index=data.get("chunkIndex", 0),
            total_chunks=data.get("totalChunks", 1),
            chunk_size=data.get("chunkSize", 0),
            effective_date=data.get("effectiveDate"),
            status=data.get("status"),
            last_updated=data.get("lastUpdated"),
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
