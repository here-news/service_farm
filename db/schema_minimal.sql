-- Minimal PostgreSQL Schema for Content Storage
-- Neo4j is the single source of truth for knowledge graph
-- PostgreSQL stores only: content, embeddings, and extraction status
--
-- Version: 3.0 (Neo4j-centric)
-- Date: 2025-12-08

-- =============================================================================
-- EXTENSIONS
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";       -- pgvector for embeddings

-- =============================================================================
-- SCHEMAS
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS content;   -- Content storage (PostgreSQL)
CREATE SCHEMA IF NOT EXISTS system;    -- Workers, jobs

-- =============================================================================
-- CONTENT SCHEMA: Page content and embeddings
-- =============================================================================

-- Page content (text + embedding only)
-- All metadata (title, description, etc.) stored in Neo4j Page nodes
CREATE TABLE content.pages (
    id UUID PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,

    -- Content (the main payload)
    content_text TEXT,

    -- Embedding for similarity search
    embedding vector(1536),

    -- Processing status
    -- 'stub' -> 'extracted' -> 'knowledge_complete' or 'failed'
    status VARCHAR(50) DEFAULT 'stub',
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for similarity search
CREATE INDEX IF NOT EXISTS idx_pages_embedding
ON content.pages USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_pages_status ON content.pages(status);
CREATE INDEX IF NOT EXISTS idx_pages_url ON content.pages(url);

-- =============================================================================
-- CONTENT SCHEMA: Claim embeddings (optional, for claim similarity)
-- =============================================================================

-- Claim embeddings (claim text stored in Neo4j Claim nodes)
CREATE TABLE content.claim_embeddings (
    claim_id UUID PRIMARY KEY,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_claim_embeddings
ON content.claim_embeddings USING ivfflat (embedding vector_cosine_ops);

-- =============================================================================
-- CONTENT SCHEMA: Event embeddings (for event similarity)
-- =============================================================================

CREATE TABLE content.event_embeddings (
    event_id UUID PRIMARY KEY,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_event_embeddings
ON content.event_embeddings USING ivfflat (embedding vector_cosine_ops);

-- =============================================================================
-- SYSTEM SCHEMA: Rogue extraction tasks (browser-based fallback)
-- =============================================================================

CREATE TABLE system.rogue_extraction_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id UUID NOT NULL,
    url TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    retry_count INT DEFAULT 0,
    processing_started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rogue_tasks_status
ON system.rogue_extraction_tasks(status);

-- =============================================================================
-- UPDATED_AT TRIGGER
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_pages_updated_at
    BEFORE UPDATE ON content.pages
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rogue_tasks_updated_at
    BEFORE UPDATE ON system.rogue_extraction_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON SCHEMA content IS 'Content storage - page text and embeddings only';
COMMENT ON SCHEMA system IS 'System tables for workers and tasks';

COMMENT ON TABLE content.pages IS 'Page content and embeddings. Metadata in Neo4j.';
COMMENT ON TABLE content.claim_embeddings IS 'Claim embeddings for similarity search. Claim text in Neo4j.';
COMMENT ON TABLE content.event_embeddings IS 'Event embeddings for similarity search.';
COMMENT ON TABLE system.rogue_extraction_tasks IS 'Pages requiring browser-based extraction';
