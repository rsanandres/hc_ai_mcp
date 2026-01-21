# HC-AI MCP Server

A standalone Model Context Protocol (MCP) server that exposes healthcare AI tools for querying patient data, reranking clinical documents, and managing document embeddings.

## Features

- **Agent Tools**: Query an HC-AI agent with natural language questions
- **Retrieval Tools**: Rerank documents using cross-encoder models
- **Session Tools**: Manage conversation history and context
- **Embeddings Tools**: Ingest and manage clinical documents in a vector store
- **YAML Configuration**: Enable/disable individual tools without code changes

## Prerequisites

- Python 3.11+
- PostgreSQL 14+ with pgvector extension
- Ollama (for local LLM and embeddings) OR cloud credentials (Bedrock/OpenAI/Anthropic)

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-username/hc-ai-mcp.git
cd hc-ai-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up PostgreSQL

```bash
# Create database
createdb hc_ai

# Run setup script
psql -U postgres -d hc_ai -f scripts/setup_db.sql
```

### 3. Configure Environment

```bash
# Copy example environment file
cp env.example .env

# Edit .env with your settings
# At minimum, set DB_PASSWORD
```

### 4. Start Ollama (if using local models)

```bash
# Pull embedding model
ollama pull mxbai-embed-large:latest

# Pull LLM model
ollama pull llama3

# Ensure Ollama is running
ollama serve
```

### 5. Run the Server

```bash
# Run with stdio transport (for Claude Desktop, Cursor)
python server.py

# Or run with HTTP transport
python server.py --transport streamable-http --port 8000
```

## Connecting to the MCP Server

### Claude Desktop

Add to your Claude Desktop configuration (`~/.config/claude/config.json` on Linux/Mac):

```json
{
  "mcpServers": {
    "hc-ai": {
      "command": "python",
      "args": ["/path/to/hc-ai-mcp/server.py"],
      "env": {}
    }
  }
}
```

### Cursor IDE

Add to your Cursor MCP settings:

```json
{
  "hc-ai": {
    "command": "python",
    "args": ["/path/to/hc-ai-mcp/server.py"]
  }
}
```

### HTTP Transport

If running with HTTP transport, connect to:
```
http://localhost:8000
```

## Configuration

### Tool Configuration (config.yaml)

Enable or disable individual tools by editing `config.yaml`:

```yaml
tools:
  agent_query:
    enabled: true
  rerank:
    enabled: true
  ingest:
    enabled: false  # Disable ingestion
```

### Environment Variables

See `env.example` for all available configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | PostgreSQL host | `localhost` |
| `DB_PORT` | PostgreSQL port | `5432` |
| `DB_NAME` | Database name | `hc_ai` |
| `DB_PASSWORD` | Database password | (required) |
| `EMBEDDING_PROVIDER` | `ollama` or `bedrock` | `ollama` |
| `LLM_PROVIDER` | `ollama`, `bedrock`, `openai`, `anthropic` | `ollama` |
| `RERANKER_MODEL` | Cross-encoder model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

### Multi-LLM Setup

- **Ollama**: set `LLM_PROVIDER=ollama` and `LLM_MODEL=llama3`
- **Bedrock**: set `LLM_PROVIDER=bedrock`, `AWS_REGION`, and `LLM_MODEL` (e.g., `haiku`, `sonnet`, `opus`)
- **OpenAI**: set `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and `OPENAI_MODEL` (e.g., `gpt-4o-mini`)
- **Anthropic**: set `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_MODEL` (e.g., `claude-3-5-sonnet-20241022`)

### Debug Logging

Enable verbose logging by setting:

```
HC_AI_DEBUG=true
```

### Timeouts

Configure tool timeouts:

```
AGENT_TIMEOUT=60
RERANK_TIMEOUT=30
```

## Available Tools

### Agent Tools

| Tool | Description |
|------|-------------|
| `agent_query` | Query the HC-AI agent with natural language |
| `agent_clear_session` | Clear agent session history |
| `agent_health` | Check agent health status |

### Retrieval Tools

| Tool | Description |
|------|-------------|
| `rerank` | Rerank documents by relevance |
| `rerank_with_context` | Rerank with full FHIR bundle context |
| `batch_rerank` | Batch rerank multiple queries |

### Session Tools

| Tool | Description |
|------|-------------|
| `session_append_turn` | Add a conversation turn |
| `session_get` | Get session state |
| `session_update_summary` | Update session summary |
| `session_clear` | Clear session data |

### Embeddings Tools

| Tool | Description |
|------|-------------|
| `ingest` | Ingest a clinical note |
| `embeddings_health` | Check embeddings service health |
| `db_stats` | Get database statistics |
| `db_queue` | Get ingestion queue status |
| `db_errors` | Get error logs |

## Example Usage

### Querying the Agent

```python
# Using the MCP client
result = await client.call_tool("agent_query", {
    "query": "What medications is patient P123 currently taking?",
    "session_id": "session-001",
    "patient_id": "P123"
})
```

### Reranking Documents

```python
result = await client.call_tool("rerank", {
    "query": "diabetes treatment options",
    "k_retrieve": 50,
    "k_return": 10
})
```

### Ingesting Documents

```python
result = await client.call_tool("ingest", {
    "id": "obs-001",
    "resource_type": "Observation",
    "content": "Blood glucose level: 120 mg/dL",
    "patient_id": "P123"
})
```

## Ingesting FHIR Data

Use the sample ingest script to load FHIR bundles:

```bash
# Ingest a single file
python scripts/sample_ingest.py /path/to/bundle.json

# Ingest a directory of bundles
python scripts/sample_ingest.py /path/to/bundles/
```

## Architecture

```
POC_MCP/
├── server.py           # MCP server entry point
├── config.yaml         # Tool toggle configuration
├── agent/              # HC-AI agent
├── reranker/           # Document reranking
├── embeddings/         # Chunking and embeddings
├── session/            # Session management
├── db/                 # Vector store operations
├── tools/              # MCP tool definitions
└── scripts/            # Setup utilities
```

## Troubleshooting

### Ollama Connection Issues

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Check available models
ollama list
```

### Database Connection Issues

```bash
# Verify PostgreSQL is running
pg_isready -h localhost -p 5432

# Check pgvector extension
psql -U postgres -d hc_ai -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Missing Embeddings

If embeddings are not being generated:
1. Check `EMBEDDING_PROVIDER` in `.env`
2. Verify Ollama model is pulled: `ollama pull mxbai-embed-large:latest`
3. Check logs for connection errors

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
