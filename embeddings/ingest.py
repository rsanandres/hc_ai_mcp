"""Ingestion pipeline for clinical notes."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    """Request model for ingesting a clinical note."""
    
    id: str = Field(..., description="Unique identifier for the resource")
    full_url: str = Field(default="", alias="fullUrl", description="Full URL of the resource")
    resource_type: str = Field(..., alias="resourceType", description="FHIR resource type")
    content: str = Field(..., min_length=1, description="Text content of the resource")
    patient_id: str = Field(default="unknown", alias="patientId", description="Patient ID")
    resource_json: str = Field(default="", alias="resourceJson", description="Original JSON for chunking")
    source_file: str = Field(default="", alias="sourceFile", description="Source file path")
    
    class Config:
        populate_by_name = True


@dataclass
class IngestResult:
    """Result of ingesting a clinical note."""
    
    id: str
    status: str
    chunks_created: int = 0
    chunks_stored: int = 0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "chunks_created": self.chunks_created,
            "chunks_stored": self.chunks_stored,
            "errors": self.errors,
        }


def _extract_resource_metadata(resource_json: str) -> Dict[str, Any]:
    """Extract common metadata fields from FHIR resource JSON."""
    metadata = {}
    if not resource_json or not resource_json.strip():
        return metadata
    
    try:
        resource = json.loads(resource_json)
        
        # Extract effective date
        date_fields = [
            "effectiveDateTime",
            "date",
            "onsetDateTime",
            "performedDateTime",
            "authoredOn",
            "birthDate",
        ]
        for field_name in date_fields:
            if field_name in resource:
                metadata["effectiveDate"] = resource[field_name]
                break
        
        # Check effectivePeriod
        if "effectiveDate" not in metadata and "effectivePeriod" in resource:
            period = resource["effectivePeriod"]
            if isinstance(period, dict) and "start" in period:
                metadata["effectiveDate"] = period["start"]
        
        # Extract status
        if "status" in resource:
            metadata["status"] = resource["status"]
        elif "clinicalStatus" in resource:
            metadata["status"] = resource["clinicalStatus"]
        
        # Extract lastUpdated from meta
        if "meta" in resource and isinstance(resource["meta"], dict):
            if "lastUpdated" in resource["meta"]:
                metadata["lastUpdated"] = resource["meta"]["lastUpdated"]
    
    except Exception as e:
        logger.debug(f"Could not extract metadata from JSON: {e}")
    
    return metadata


async def process_and_store(request: IngestRequest) -> IngestResult:
    """Process and store a clinical note.
    
    Chunks the content and stores each chunk in the vector store.
    
    Args:
        request: The ingest request with content and metadata.
    
    Returns:
        IngestResult with status and chunk counts.
    """
    from embeddings.chunker import recursive_json_chunking, recursive_text_chunking
    from embeddings.embedder import get_embedding
    from db.vector_store import store_chunk
    
    result = IngestResult(id=request.id, status="processing")
    
    try:
        # Determine chunking method based on available data
        if request.resource_json and request.resource_json.strip():
            chunks = recursive_json_chunking(
                request.resource_json,
                max_chunk_size=1000,
                min_chunk_size=500,
            )
            chunking_method = "RecursiveJsonSplitter"
        else:
            chunks = recursive_text_chunking(
                request.content,
                chunk_size=1000,
                chunk_overlap=100,
            )
            chunking_method = "RecursiveCharacterTextSplitter"
        
        if not chunks:
            result.status = "warning"
            result.errors.append("No chunks created")
            return result
        
        result.chunks_created = len(chunks)
        
        # Extract metadata from resource JSON
        resource_metadata = _extract_resource_metadata(request.resource_json)
        total_chunks = len(chunks)
        
        # Store each chunk
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            chunk_text = chunk.get("text", "")
            
            if not chunk_text or not chunk_text.strip():
                result.errors.append(f"Empty chunk: {chunk.get('chunk_id', 'unknown')}")
                continue
            
            # Build metadata
            metadata = {
                "patientId": request.patient_id,
                "resourceId": request.id,
                "resourceType": request.resource_type,
                "fullUrl": request.full_url,
                "sourceFile": request.source_file,
                "chunkId": f"{request.id}_{chunk.get('chunk_id', chunk_id)}",
                "chunkIndex": chunk.get("chunk_index", 0),
                "totalChunks": total_chunks,
                "chunkSize": chunk.get("chunk_size", len(chunk_text)),
                "chunkingMethod": chunking_method,
            }
            
            # Add extracted metadata
            metadata.update(resource_metadata)
            
            # Store chunk
            try:
                success = await store_chunk(
                    chunk_text=chunk_text,
                    chunk_id=chunk_id,
                    metadata=metadata,
                    use_queue=True,
                )
                if success:
                    result.chunks_stored += 1
            except Exception as e:
                result.errors.append(f"Failed to store chunk {chunk_id}: {str(e)}")
        
        if result.chunks_stored == result.chunks_created:
            result.status = "success"
        elif result.chunks_stored > 0:
            result.status = "partial"
        else:
            result.status = "failed"
        
        logger.info(
            f"Processed {request.id}: {result.chunks_stored}/{result.chunks_created} chunks stored"
        )
        
    except Exception as e:
        result.status = "error"
        result.errors.append(str(e))
        logger.error(f"Error processing {request.id}: {e}")
    
    return result
