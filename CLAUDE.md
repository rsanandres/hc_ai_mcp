# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

atlas_mcp is a standalone Python toolkit for AI-powered search and conversation over FHIR R4 clinical documents, accessible via MCP (Model Context Protocol). It provides 15+ MCP tools for agent queries, document retrieval, session management, and embeddings ingestion. The MCP interface is the delivery mechanism — the value is the full healthcare AI stack underneath (hybrid search, cross-encoder reranking, multi-agent workflow, PII masking, medical API integrations).

Requires Python 3.11+, PostgreSQL 14+ with pgvector, and an LLM provider (Ollama default). Designed to be cloned, configured via `.env`, and run immediately.

## Commands

```bash
# Run server (stdio transport for Claude Desktop/Cursor)
python server.py

# Run server (HTTP transport)
python server.py --transport streamable-http --port 8000

# Run all tests (134 tests, no external services needed)
.venv/bin/python -m pytest

# Run a specific test file
.venv/bin/python -m pytest tests/test_session/test_in_memory_store.py

# Run tests matching a pattern
.venv/bin/python -m pytest -k "query_classifier"

# Run tests for a module
.venv/bin/python -m pytest tests/test_agent/

# Install dependencies
pip install -r requirements.txt

# Database setup
psql -U postgres -d hc_ai -f scripts/setup_db.sql
```

Note: Use `.venv/bin/python` — the project venv has all dependencies including pytest.

## Architecture

The server entry point is `server.py`, which creates a FastMCP instance, registers tool groups, and runs via stdio or HTTP (uvicorn).

### Core Modules

- **`agent/`** — LangGraph multi-agent workflow: Query → Classifier → Researcher → Validator → Response. `graph.py` defines the state machine, `config.py` handles LLM provider abstraction (Ollama/OpenAI/Anthropic/Bedrock), `query_classifier.py` detects query type (medical/conversational/mixed). `pii_masker/` provides PII detection via local regex or AWS Comprehend Medical.

- **`db/`** — AsyncIO PostgreSQL + pgvector client. `vector_store.py` manages connection pooling (with SSL/RDS auto-detection), async queue workers, retry with backoff, hybrid search (vector + BM25), patient timeline, and `list_patients()`. `bm25_search.py` provides full-text search with metadata key whitelisting against SQL injection.

- **`embeddings/`** — Document ingestion pipeline: raw text → chunking → embedding → storage. `chunker.py` supports semantic, recursive JSON (FHIR-aware), and parent-child chunking. `embedder.py` wraps Ollama/Bedrock/Nomic embedding APIs.

- **`reranker/`** — Cross-encoder reranking via sentence-transformers (`cross_encoder.py`). Includes in-memory TTL cache (`cache.py`).

- **`session/`** — Conversation state with dual backends: `InMemorySessionStore` (default, no deps) or `SessionStore` (DynamoDB). Selected via `SESSION_PROVIDER` env var.

- **`tools/`** — MCP tool definitions registered by `server.py`. Split into `agent_tools.py`, `retrieval_tools.py`, `embeddings_tools.py`. `utils.py` provides validation helpers and standard error response format.

- **`config/`** — YAML + env config loading with Pydantic validation (`loader.py`). Global config singleton with `reload_config()`.

### Key Design Patterns

- **Async-first**: All DB operations use asyncpg with `asyncio.Queue` workers. Tools are `async def`. Timeouts via `asyncio.wait_for()`.
- **Multi-provider**: LLM, embeddings, and sessions each select backend via a single env var (`LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `SESSION_PROVIDER`).
- **Snake_case metadata**: All metadata keys use snake_case (`patient_id`, `resource_type`, `effective_date`) across ingest, storage, indexes, and queries.
- **Structured errors**: All tool errors return `{"status": "error", "error_type": "...", "message": "..."}`.
- **Tool toggles**: `config.yaml` enables/disables individual tools (server restart required).

### Configuration

- `config.yaml` — Tool enable/disable toggles and server transport settings.
- `.env` — Credentials and runtime config. `DB_PASSWORD` is the only required variable. See `env.example` for all 67 env vars.
- `HC_AI_DEBUG=true` enables debug-level logging.  <!-- NOTE: env var name kept as HC_AI_DEBUG for backwards compatibility -->

## Testing

Tests use pytest with pytest-asyncio. `tests/conftest.py` auto-sets test environment variables (DB_PASSWORD, LLM_PROVIDER=ollama, SESSION_PROVIDER=memory, etc.) so tests run without external services. Tests are organized by module: `test_agent/`, `test_db/`, `test_embeddings/`, `test_reranker/`, `test_session/`, `test_integration/`, `test_tools/`.

### Important Conventions

- Metadata keys are **always snake_case** — never camelCase. This is critical for ingest→storage→query consistency.
- The BM25 search layer whitelists metadata filter keys in `ALLOWED_METADATA_KEYS` — add new filterable keys there.
- `agent/tools/retrieval.py` auto-detects FHIR resource types from query keywords and strips patient names from queries to avoid embedding mismatches.
- Bedrock clients (both LLM and embedding) use `BotoConfig` for timeouts/retries via shared env vars (`BEDROCK_READ_TIMEOUT`, etc.).
