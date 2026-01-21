"""Cross-encoder reranker implementation using sentence-transformers."""

from __future__ import annotations

import os
from typing import Any, List, Tuple

from dotenv import load_dotenv

load_dotenv()

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto")

from logging_config import get_logger
from reranker.cache import build_cache_key, get_cache

logger = get_logger("hc_ai.reranker")


def _resolve_device(device: str) -> str:
    """Resolve device string to actual device."""
    if device and device.lower() != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Reranker:
    """Cross-encoder reranker that scores query-document pairs."""
    
    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        device: str = RERANKER_DEVICE,
    ) -> None:
        """Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model.
            device: Device to run on ('cpu', 'cuda', or 'auto').
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            ) from e
        
        resolved_device = _resolve_device(device)
        self._device = resolved_device
        self._model_name = model_name
        self._model = CrossEncoder(model_name, device=resolved_device)
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name
    
    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device
    
    def score(self, query: str, docs: List[str]) -> List[float]:
        """Score query-document pairs.
        
        Args:
            query: The query string.
            docs: List of document strings.
        
        Returns:
            List of scores for each document.
        """
        if not docs:
            return []
        pairs = [(query, doc) for doc in docs]
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]
    
    def rerank(self, query: str, docs: List[Any], top_k: int) -> List[Any]:
        """Rerank documents by relevance to query.
        
        Args:
            query: The query string.
            docs: List of Document objects with page_content attribute.
            top_k: Number of top documents to return.
        
        Returns:
            Top-k documents sorted by relevance.
        """
        if not docs:
            return []
        contents = [doc.page_content for doc in docs]
        scores = self.score(query, contents)
        scored_docs = [(idx, doc, score) for idx, (doc, score) in enumerate(zip(docs, scores))]
        scored_docs.sort(key=lambda item: (-item[2], item[0]))
        return [doc for _idx, doc, _score in scored_docs[:top_k]]
    
    def rerank_with_scores(
        self,
        query: str,
        docs: List[Any],
    ) -> List[Tuple[Any, float]]:
        """Rerank documents and return with scores.
        
        Args:
            query: The query string.
            docs: List of Document objects with page_content attribute.
        
        Returns:
            List of (document, score) tuples sorted by relevance.
        """
        if not docs:
            return []
        contents = [doc.page_content for doc in docs]
        doc_ids: List[str] = []
        for idx, doc in enumerate(docs):
            doc_id = getattr(doc, "id", None)
            if not doc_id:
                meta = getattr(doc, "metadata", {}) or {}
                doc_id = meta.get("chunkId") or meta.get("resourceId") or f"doc_{idx}"
            doc_ids.append(str(doc_id))

        scores: List[float] = []
        cache = get_cache()
        cache_key = build_cache_key(query, doc_ids)
        cached = cache.get(cache_key)
        if cached:
            cached_map = {doc_id: score for doc_id, score in cached}
            if all(doc_id in cached_map for doc_id in doc_ids):
                scores = [float(cached_map[doc_id]) for doc_id in doc_ids]
                logger.debug("Reranker cache hit for %s docs", len(doc_ids))

        if not scores:
            scores = self.score(query, contents)
            cache.set(cache_key, list(zip(doc_ids, scores)))
        scored_docs = [(idx, doc, score) for idx, (doc, score) in enumerate(zip(docs, scores))]
        scored_docs.sort(key=lambda item: (-item[2], item[0]))
        return [(doc, score) for _idx, doc, score in scored_docs]
    
    def rerank_batch(
        self,
        queries: List[str],
        docs_list: List[List[Any]],
        top_k: int,
    ) -> List[List[Any]]:
        """Rerank multiple query-document sets.
        
        Args:
            queries: List of query strings.
            docs_list: List of document lists, one per query.
            top_k: Number of top documents to return per query.
        
        Returns:
            List of reranked document lists.
        """
        if len(queries) != len(docs_list):
            raise ValueError("queries and docs_list length mismatch")
        results: List[List[Any]] = []
        for query, docs in zip(queries, docs_list):
            results.append(self.rerank(query, docs, top_k))
        return results


# Global reranker instance
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """Get or create the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
