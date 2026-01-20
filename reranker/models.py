"""Pydantic models for reranker service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    """Request model for reranking documents."""
    
    query: str = Field(..., min_length=1, description="The search query")
    k_retrieve: int = Field(default=50, ge=1, description="Number of documents to retrieve from vector store")
    k_return: int = Field(default=10, ge=1, description="Number of documents to return after reranking")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters for retrieval")


class RerankWithContextRequest(RerankRequest):
    """Request model for reranking with full document context."""
    
    include_full_json: bool = Field(default=False, description="Include full FHIR bundle JSON in response")


class BatchRerankItem(BaseModel):
    """Single item in a batch rerank request."""
    
    query: str = Field(..., min_length=1, description="The search query")
    k_retrieve: int = Field(default=50, ge=1, description="Number of documents to retrieve")
    k_return: int = Field(default=10, ge=1, description="Number of documents to return")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class BatchRerankRequest(BaseModel):
    """Request model for batch reranking."""
    
    items: List[BatchRerankItem] = Field(..., description="List of rerank requests")


class DocumentResponse(BaseModel):
    """Response model for a single document."""
    
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    score: Optional[float] = Field(default=None, description="Relevance score")


class RerankResponse(BaseModel):
    """Response model for reranking."""
    
    query: str = Field(..., description="The original query")
    results: List[DocumentResponse] = Field(default_factory=list, description="Reranked documents")


class FullDocumentResponse(BaseModel):
    """Response model for full document with FHIR bundle."""
    
    patient_id: str = Field(..., description="Patient ID")
    source_filename: str = Field(default="", description="Source file name")
    bundle_json: Dict[str, Any] = Field(default_factory=dict, description="Full FHIR bundle JSON")


class RerankWithContextResponse(BaseModel):
    """Response model for reranking with context."""
    
    query: str = Field(..., description="The original query")
    chunks: List[DocumentResponse] = Field(default_factory=list, description="Reranked chunks")
    full_documents: List[FullDocumentResponse] = Field(default_factory=list, description="Full documents")


class BatchRerankResponse(BaseModel):
    """Response model for batch reranking."""
    
    items: List[RerankResponse] = Field(default_factory=list, description="Rerank results")


class StatsResponse(BaseModel):
    """Response model for reranker statistics."""
    
    model_name: str = Field(..., description="Reranker model name")
    device: str = Field(..., description="Device being used")
    cache_hits: int = Field(default=0, description="Cache hit count")
    cache_misses: int = Field(default=0, description="Cache miss count")
    cache_size: int = Field(default=0, description="Current cache size")
