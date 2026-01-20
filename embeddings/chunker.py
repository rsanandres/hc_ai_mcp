"""Text chunking utilities for document processing."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass
except ImportError:
    nltk = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    RecursiveCharacterTextSplitter = None
    RecursiveJsonSplitter = None

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    np = None
    cosine_similarity = None


def semantic_chunking(
    text: str,
    threshold: float = 0.7,
) -> List[str]:
    """Perform semantic chunking on text using sentence embeddings.
    
    Falls back to simple sentence-based chunking if embeddings are unavailable.
    
    Args:
        text: Input text to chunk.
        threshold: Similarity threshold for chunking (0-1).
    
    Returns:
        List of text chunks.
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # Split into sentences
    sentences = _split_sentences(text)
    
    if len(sentences) < 2:
        return [text] if text.strip() else []
    
    # Try to use embeddings for semantic chunking
    try:
        from .embedder import get_embeddings
        embeddings = get_embeddings(sentences)
        
        if embeddings and len(embeddings) == len(sentences) and SKLEARN_AVAILABLE:
            return _semantic_chunk_with_embeddings(sentences, embeddings, threshold)
    except Exception as e:
        logger.warning(f"Semantic chunking failed, using fallback: {e}")
    
    # Fallback: simple sentence-based chunking (3-5 sentences per chunk)
    return _simple_sentence_chunking(sentences, chunk_size=4)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    if nltk is not None:
        try:
            return nltk.sent_tokenize(text)
        except Exception:
            pass
    # Fallback: split by periods
    return [s.strip() + "." for s in text.split(".") if s.strip()]


def _semantic_chunk_with_embeddings(
    sentences: List[str],
    embeddings: List[List[float]],
    threshold: float,
) -> List[str]:
    """Chunk sentences based on embedding similarity."""
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(len(sentences) - 1):
        if i >= len(embeddings) or (i + 1) >= len(embeddings):
            current_chunk.extend(sentences[i + 1:])
            break
        
        vec_a = np.array(embeddings[i]).reshape(1, -1)
        vec_b = np.array(embeddings[i + 1]).reshape(1, -1)
        
        similarity = cosine_similarity(vec_a, vec_b)[0][0]
        
        if similarity < threshold:
            # Semantic break - start new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def _simple_sentence_chunking(sentences: List[str], chunk_size: int = 4) -> List[str]:
    """Simple chunking by grouping sentences."""
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def recursive_json_chunking(
    json_text: str,
    max_chunk_size: int = 1000,
    min_chunk_size: int = 500,
) -> List[Dict[str, Any]]:
    """Chunk JSON text using RecursiveJsonSplitter.
    
    Args:
        json_text: JSON string to chunk.
        max_chunk_size: Maximum size of chunks.
        min_chunk_size: Minimum size of chunks.
    
    Returns:
        List of chunk dictionaries with chunk_id, text, etc.
    """
    if not json_text or len(json_text.strip()) == 0:
        return []
    
    if not LANGCHAIN_AVAILABLE or RecursiveJsonSplitter is None:
        # Fallback: simple text splitting
        return _fallback_json_chunking(json_text, max_chunk_size)
    
    try:
        json_splitter = RecursiveJsonSplitter(
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
        )
        split_chunks = json_splitter.split_text(json_text)
        
        chunks = []
        for i, chunk in enumerate(split_chunks):
            if hasattr(chunk, "page_content"):
                chunk_text = str(chunk.page_content)
            elif isinstance(chunk, str):
                chunk_text = chunk
            elif isinstance(chunk, dict):
                chunk_text = json.dumps(chunk, ensure_ascii=False)
            else:
                chunk_text = str(chunk)
            
            if chunk_text.strip():
                chunks.append({
                    "chunk_id": f"chunk_{i}",
                    "chunk_type": "json_chunk",
                    "text": chunk_text,
                    "chunk_size": len(chunk_text),
                    "chunk_index": i,
                })
        
        return chunks
    except Exception as e:
        logger.warning(f"RecursiveJsonSplitter failed: {e}")
        return _fallback_json_chunking(json_text, max_chunk_size)


def _fallback_json_chunking(json_text: str, max_chunk_size: int) -> List[Dict[str, Any]]:
    """Fallback chunking for JSON text."""
    chunks = []
    for i in range(0, len(json_text), max_chunk_size):
        chunk_text = json_text[i:i + max_chunk_size]
        if chunk_text.strip():
            chunks.append({
                "chunk_id": f"chunk_{i // max_chunk_size}",
                "chunk_type": "json_chunk",
                "text": chunk_text,
                "chunk_size": len(chunk_text),
                "chunk_index": i // max_chunk_size,
            })
    return chunks


def recursive_text_chunking(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """Chunk text using RecursiveCharacterTextSplitter.
    
    Args:
        text: Text to chunk.
        chunk_size: Maximum size of chunks.
        chunk_overlap: Overlap between chunks.
    
    Returns:
        List of chunk dictionaries.
    """
    if not text or len(text.strip()) == 0:
        return []
    
    if LANGCHAIN_AVAILABLE and RecursiveCharacterTextSplitter is not None:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
            split_chunks = splitter.split_text(text)
            
            chunks = []
            for i, chunk_text in enumerate(split_chunks):
                if chunk_text.strip():
                    chunks.append({
                        "chunk_id": f"chunk_{i}",
                        "chunk_type": "text_chunk",
                        "text": chunk_text,
                        "chunk_size": len(chunk_text),
                        "chunk_index": i,
                    })
            return chunks
        except Exception as e:
            logger.warning(f"RecursiveCharacterTextSplitter failed: {e}")
    
    # Fallback: simple splitting
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk_text = text[i:i + chunk_size]
        if chunk_text.strip():
            chunks.append({
                "chunk_id": f"chunk_{i // (chunk_size - chunk_overlap)}",
                "chunk_type": "text_chunk",
                "text": chunk_text,
                "chunk_size": len(chunk_text),
                "chunk_index": i // (chunk_size - chunk_overlap),
            })
    return chunks


def parent_child_chunking(
    text: str,
    parent_chunk_size: int = 2000,
    child_chunk_size: int = 500,
    parent_overlap: int = 200,
    child_overlap: int = 50,
    use_semantic_for_children: bool = True,
    semantic_threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """Hybrid parent-child chunking.
    
    Creates larger parent chunks for context and smaller child chunks
    for precise retrieval.
    
    Args:
        text: Input text to chunk.
        parent_chunk_size: Size of parent chunks.
        child_chunk_size: Size of child chunks.
        parent_overlap: Overlap between parent chunks.
        child_overlap: Overlap between child chunks.
        use_semantic_for_children: Use semantic chunking for children if available.
        semantic_threshold: Threshold for semantic similarity.
    
    Returns:
        List of chunk dictionaries with parent-child relationships.
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # Create parent chunks
    if LANGCHAIN_AVAILABLE and RecursiveCharacterTextSplitter is not None:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        parent_chunks = parent_splitter.split_text(text)
    else:
        parent_chunks = [
            text[i:i + parent_chunk_size]
            for i in range(0, len(text), parent_chunk_size - parent_overlap)
        ]
    
    all_chunks = []
    
    for parent_idx, parent_text in enumerate(parent_chunks):
        parent_id = f"parent_{parent_idx}"
        
        # Create child chunks
        if use_semantic_for_children:
            child_texts = semantic_chunking(parent_text, threshold=semantic_threshold)
            # Split large semantic chunks
            refined_children = []
            for child in child_texts:
                if len(child) > child_chunk_size:
                    sub_chunks = recursive_text_chunking(child, child_chunk_size, child_overlap)
                    refined_children.extend([c["text"] for c in sub_chunks])
                else:
                    refined_children.append(child)
            child_texts = refined_children if refined_children else [parent_text]
        else:
            child_chunks = recursive_text_chunking(parent_text, child_chunk_size, child_overlap)
            child_texts = [c["text"] for c in child_chunks]
        
        if not child_texts:
            child_texts = [parent_text]
        
        # Create parent chunk object
        all_chunks.append({
            "chunk_id": parent_id,
            "chunk_type": "parent",
            "text": parent_text,
            "parent_id": None,
            "child_ids": [f"{parent_id}_child_{i}" for i in range(len(child_texts))],
            "chunk_size": len(parent_text),
            "chunk_index": parent_idx,
        })
        
        # Create child chunk objects
        for i, child_text in enumerate(child_texts):
            all_chunks.append({
                "chunk_id": f"{parent_id}_child_{i}",
                "chunk_type": "child",
                "text": child_text,
                "parent_id": parent_id,
                "child_ids": [],
                "chunk_size": len(child_text),
                "chunk_index": i,
                "parent_text": parent_text,
            })
    
    return all_chunks
