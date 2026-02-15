"""Embedding generation using Ollama, Amazon Bedrock, or Nomic API."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

from logging_config import get_logger

logger = get_logger("atlas.embeddings")

# Embedding provider configuration
# Supported: "ollama" (default), "bedrock", "nomic" (fallback)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest")
OLLAMA_EMBED_BATCH_SIZE = int(os.getenv("OLLAMA_EMBED_BATCH_SIZE", "8"))

# Nomic API configuration (fallback)
NOMIC_API_AVAILABLE = False
if EMBEDDING_PROVIDER == "nomic":
    try:
        from nomic import embed
        NOMIC_API_AVAILABLE = True
    except (ImportError, TypeError) as e:
        logger.warning("Could not import nomic API: %s", e)

# Bedrock configuration
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
BEDROCK_EMBED_BATCH_SIZE = int(os.getenv("BEDROCK_EMBED_BATCH_SIZE", "4"))


def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding for a single text.
    
    Args:
        text: Text to embed.
    
    Returns:
        Embedding vector or None if failed.
    """
    embeddings = get_embeddings([text])
    if embeddings and len(embeddings) > 0:
        return embeddings[0]
    return None


def get_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    """Get embeddings for multiple texts.
    
    Args:
        texts: List of texts to embed.
    
    Returns:
        List of embedding vectors or None if failed.
    """
    if not texts:
        return []
    
    if EMBEDDING_PROVIDER == "ollama":
        return _get_embeddings_ollama(texts)
    elif EMBEDDING_PROVIDER == "nomic":
        return _get_embeddings_nomic(texts)
    elif EMBEDDING_PROVIDER == "bedrock":
        return _get_embeddings_bedrock(texts)
    else:
        logger.error("Unknown embedding provider: %s", EMBEDDING_PROVIDER)
        return None


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    """Yield lists of items in batches."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _ollama_embed_single(session: requests.Session, text: str) -> List[float]:
    """Embed a single text via Ollama."""
    response = session.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={
            "model": OLLAMA_EMBED_MODEL,
            "prompt": text,
        },
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    if "embedding" not in result:
        raise ValueError(f"Unexpected response format from Ollama: {result}")
    return result["embedding"]


def _get_embeddings_ollama(texts: List[str]) -> Optional[List[List[float]]]:
    """Get embeddings using Ollama API.

    Args:
        texts: List of texts to embed.

    Returns:
        List of embedding vectors or None if failed.
    """
    # TODO: Future improvement — re-enable ThreadPoolExecutor for parallel
    # embedding after resolving issues with connection pool exhaustion.
    try:
        embeddings: List[List[float]] = []
        batch_size = max(1, OLLAMA_EMBED_BATCH_SIZE)
        with requests.Session() as session:
            for batch in _chunked(texts, batch_size):
                for text in batch:
                    embeddings.append(_ollama_embed_single(session, text))
        return embeddings
    except requests.exceptions.RequestException as e:
        logger.error("Error calling Ollama API: %s", e)
        return None
    except Exception as e:
        logger.error("Ollama embedding error: %s", e)
        return None


def _get_embeddings_bedrock(texts: List[str]) -> Optional[List[List[float]]]:
    """Get embeddings using Amazon Bedrock.
    
    Args:
        texts: List of texts to embed.
    
    Returns:
        List of embedding vectors or None if failed.
    """
    try:
        import boto3
        from botocore.config import Config as BotoConfig

        boto_config = BotoConfig(
            read_timeout=int(os.getenv("BEDROCK_READ_TIMEOUT", "60")),
            connect_timeout=int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "10")),
            retries={"max_attempts": int(os.getenv("BEDROCK_MAX_RETRIES", "2"))},
        )
        client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION, config=boto_config)
        embeddings = []
        batch_size = max(1, BEDROCK_EMBED_BATCH_SIZE)

        for batch in _chunked(texts, batch_size):
            for text in batch:
                body = json.dumps({"inputText": text})
                response = client.invoke_model(
                    modelId=BEDROCK_EMBED_MODEL,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                result = json.loads(response["body"].read())
                if "embedding" in result:
                    embeddings.append(result["embedding"])
                else:
                    logger.warning(f"Unexpected response format from Bedrock: {result}")
                    return None
        return embeddings
    except ImportError:
        logger.error("boto3 is required for Bedrock embeddings. Install with: pip install boto3")
        return None
    except Exception as e:
        logger.error(f"Error calling Bedrock API: {e}")
        return None


def _get_embeddings_nomic(texts: List[str]) -> Optional[List[List[float]]]:
    """Get embeddings using Nomic API (fallback provider)."""
    if not NOMIC_API_AVAILABLE:
        logger.error("Nomic API not available. Install with: pip install nomic")
        return None
    try:
        output = embed.text(
            texts=texts,
            model="nomic-embed-text-v1.5",
            task_type="search_document",
        )
        return output["embeddings"] if output.get("embeddings") else None
    except Exception as e:
        logger.error("Error calling Nomic API: %s", e)
        return None


def test_connection() -> Dict[str, Any]:
    """Test connection to the embedding service.
    
    Returns:
        Dictionary with connection status and details.
    """
    result: Dict[str, Any] = {
        "provider": EMBEDDING_PROVIDER,
        "ok": False,
        "errors": [],
    }
    
    if EMBEDDING_PROVIDER == "ollama":
        result["base_url"] = OLLAMA_BASE_URL
        result["model"] = OLLAMA_EMBED_MODEL
        
        # Check /api/tags
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            result["tags_status"] = resp.status_code
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                result["available_models"] = [m.get("name", "") for m in models]
        except Exception as e:
            result["errors"].append(f"tags_error: {e}")
        
        # Try a simple embedding call
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": "test"},
                timeout=10,
            )
            result["embed_status"] = resp.status_code
            if resp.status_code == 200:
                payload = resp.json()
                if "embedding" in payload:
                    result["embed_dimensions"] = len(payload["embedding"])
                    result["ok"] = True
                else:
                    result["errors"].append("embed_error: missing 'embedding' key")
            else:
                result["errors"].append(f"embed_error: status {resp.status_code}")
        except Exception as e:
            result["errors"].append(f"embed_error: {e}")
    
    elif EMBEDDING_PROVIDER == "bedrock":
        result["region"] = BEDROCK_REGION
        result["model"] = BEDROCK_EMBED_MODEL
        
        try:
            import boto3
            from botocore.config import Config as BotoConfig

            boto_config = BotoConfig(
                read_timeout=int(os.getenv("BEDROCK_READ_TIMEOUT", "60")),
                connect_timeout=int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "10")),
                retries={"max_attempts": int(os.getenv("BEDROCK_MAX_RETRIES", "2"))},
            )
            client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION, config=boto_config)
            # Try a simple embedding
            body = json.dumps({"inputText": "test"})
            response = client.invoke_model(
                modelId=BEDROCK_EMBED_MODEL,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            res = json.loads(response["body"].read())
            if "embedding" in res:
                result["embed_dimensions"] = len(res["embedding"])
                result["ok"] = True
            else:
                result["errors"].append("embed_error: missing 'embedding' key")
        except ImportError:
            result["errors"].append("boto3 not installed")
        except Exception as e:
            result["errors"].append(f"bedrock_error: {e}")
    
    elif EMBEDDING_PROVIDER == "nomic":
        result["model"] = "nomic-embed-text-v1.5"
        if not NOMIC_API_AVAILABLE:
            result["errors"].append("nomic package not installed")
        else:
            try:
                output = embed.text(
                    texts=["test"],
                    model="nomic-embed-text-v1.5",
                    task_type="search_document",
                )
                if output.get("embeddings"):
                    result["embed_dimensions"] = len(output["embeddings"][0])
                    result["ok"] = True
                else:
                    result["errors"].append("embed_error: no embeddings returned")
            except Exception as e:
                result["errors"].append(f"nomic_error: {e}")

    else:
        result["errors"].append(f"Unknown provider: {EMBEDDING_PROVIDER}")

    return result
