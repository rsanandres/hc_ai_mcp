-- HC-AI MCP Server - Database Setup
-- PostgreSQL with pgvector extension
--
-- Prerequisites:
--   1. PostgreSQL 14+ installed
--   2. pgvector extension available
--
-- Usage:
--   psql -U postgres -d hc_ai -f setup_db.sql

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema
CREATE SCHEMA IF NOT EXISTS public;

-- Create chunks table for storing document embeddings
-- Adjust vector dimensions (1024) based on your embedding model:
--   - mxbai-embed-large: 1024 dimensions
--   - text-embedding-ada-002: 1536 dimensions
--   - amazon.titan-embed-text-v1: 1536 dimensions
CREATE TABLE IF NOT EXISTS public.hc_ai_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding vector(1024),  -- Adjust dimensions as needed
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for vector similarity search
-- Using HNSW index for better performance on large datasets
CREATE INDEX IF NOT EXISTS hc_ai_chunks_embedding_idx 
ON public.hc_ai_chunks 
USING hnsw (embedding vector_cosine_ops);

-- Create index for metadata filtering
CREATE INDEX IF NOT EXISTS hc_ai_chunks_metadata_idx 
ON public.hc_ai_chunks 
USING gin (metadata);

-- Create index for patient_id lookups (common filter)
CREATE INDEX IF NOT EXISTS hc_ai_chunks_patient_id_idx
ON public.hc_ai_chunks ((metadata->>'patient_id'));

-- Create index for resource_type lookups
CREATE INDEX IF NOT EXISTS hc_ai_chunks_resource_type_idx
ON public.hc_ai_chunks ((metadata->>'resource_type'));

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-update updated_at
DROP TRIGGER IF EXISTS update_hc_ai_chunks_updated_at ON public.hc_ai_chunks;
CREATE TRIGGER update_hc_ai_chunks_updated_at
    BEFORE UPDATE ON public.hc_ai_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON TABLE public.hc_ai_chunks TO your_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_app_user;

-- Verify setup
SELECT 
    'pgvector extension' as component,
    CASE WHEN EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') 
         THEN 'OK' ELSE 'MISSING' END as status
UNION ALL
SELECT 
    'hc_ai_chunks table' as component,
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables 
                      WHERE table_schema = 'public' AND table_name = 'hc_ai_chunks') 
         THEN 'OK' ELSE 'MISSING' END as status
UNION ALL
SELECT 
    'embedding index' as component,
    CASE WHEN EXISTS (SELECT 1 FROM pg_indexes 
                      WHERE indexname = 'hc_ai_chunks_embedding_idx') 
         THEN 'OK' ELSE 'MISSING' END as status;

-- Display table info
\d+ public.hc_ai_chunks
