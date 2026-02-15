# Debugging Guide

## Enable Verbose Logging

Set the following in your `.env`:

```
HC_AI_DEBUG=true
```

This enables DEBUG-level logs across:
- `atlas.agent`
- `atlas.tools`
- `atlas.db`
- `atlas.embeddings`
- `atlas.reranker`
- `atlas.session`

## Health Checks

Run the health check script:

```
python scripts/health_check.py
```

It validates:
- Environment configuration
- Database connectivity
- Embedding provider connectivity
- MCP tool availability

## Common Issues

### Server Fails to Start
- Confirm `DB_PASSWORD` is set.
- If using Bedrock, set `AWS_REGION`.
- If using OpenAI/Anthropic, ensure API keys are set.

### Tool Timeouts
- Increase `AGENT_TIMEOUT` or `RERANK_TIMEOUT` in `.env`.
- Check database connectivity and model responsiveness.

### Missing Embeddings
- Ensure Ollama is running and the embed model is pulled.
- Check `EMBEDDING_PROVIDER` and `OLLAMA_EMBED_MODEL`.

## Performance Profiling

Use `HC_AI_DEBUG=true` and inspect:  <!-- NOTE: env var name kept as HC_AI_DEBUG for backwards compatibility -->
- Agent response timings
- Reranker cache hit/miss behavior
- Embedding latency and batch size impact
