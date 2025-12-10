-- Migration 002: Add core schema for knowledge graph
-- Combines with existing community schema (users, chat_sessions, comments)
--
-- Architecture:
-- - Postgres: page content, embeddings (for search/matching)
-- - Neo4j: graph structure (pages, claims, entities, relationships)

-- =============================================================================
-- EXTENSIONS
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";      -- Fuzzy text search
CREATE EXTENSION IF NOT EXISTS "vector";       -- pgvector for embeddings

-- =============================================================================
-- SCHEMAS
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS core;      -- Knowledge graph metadata
CREATE SCHEMA IF NOT EXISTS content;   -- Embeddings for semantic search

-- =============================================================================
-- CORE.PAGES: Page content and embeddings
-- Neo4j holds: Page nodes, relationships, graph structure
-- Postgres holds: Text content, embeddings for semantic search
-- =============================================================================

CREATE TABLE core.pages (
    id VARCHAR(20) PRIMARY KEY,  -- pg_xxxxxxxx format (short ID)

    -- URLs
    url TEXT NOT NULL,
    canonical_url TEXT,

    -- Content (for extraction and search)
    content_text TEXT,
    word_count INT DEFAULT 0,

    -- Metadata (from iframely)
    title TEXT,
    description TEXT,
    byline TEXT,              -- author/byline
    thumbnail_url TEXT,
    site_name TEXT,
    domain TEXT,

    -- Language
    language VARCHAR(10) DEFAULT 'en',
    language_confidence FLOAT DEFAULT 0.0,

    -- Metadata quality score
    metadata_confidence FLOAT DEFAULT 0.0,

    -- Publication time
    pub_time TIMESTAMPTZ,

    -- Status (synced with Neo4j)
    status VARCHAR(50) DEFAULT 'pending',
    -- Values: pending, extracted, knowledge_complete, event_complete, failed

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Embedding for semantic search (OpenAI ada-002: 1536 dimensions)
    embedding vector(1536)
);

-- Indexes
CREATE INDEX idx_pages_url ON core.pages(url);
CREATE INDEX idx_pages_canonical_url ON core.pages(canonical_url);
CREATE INDEX idx_pages_status ON core.pages(status);
CREATE INDEX idx_pages_domain ON core.pages(domain);
CREATE INDEX idx_pages_created_at ON core.pages(created_at DESC);
CREATE INDEX idx_pages_pub_time ON core.pages(pub_time DESC);
CREATE INDEX idx_pages_embedding ON core.pages USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION core.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_pages_updated_at BEFORE UPDATE ON core.pages
    FOR EACH ROW EXECUTE FUNCTION core.update_updated_at_column();

-- =============================================================================
-- CONTENT.CLAIM_EMBEDDINGS: Claim embeddings for semantic matching
-- Neo4j holds: Claim nodes, EXTRACTED relationships, claim metadata
-- Postgres holds: Only embeddings for claim similarity/corroboration detection
-- =============================================================================

CREATE TABLE content.claim_embeddings (
    claim_id VARCHAR(20) PRIMARY KEY,  -- cl_xxxxxxxx format

    -- Embedding for semantic similarity
    embedding vector(1536) NOT NULL
);

CREATE INDEX idx_claim_embeddings_vector ON content.claim_embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- =============================================================================
-- CORE.ENTITIES: Entity embeddings for matching
-- Neo4j holds: Entity nodes, relationships, wikidata data
-- Postgres holds: Name embeddings for entity resolution
-- =============================================================================

CREATE TABLE core.entities (
    id VARCHAR(20) PRIMARY KEY,  -- en_xxxxxxxx format

    -- Canonical name
    canonical_name TEXT NOT NULL,
    entity_type VARCHAR(50),  -- PERSON, ORGANIZATION, LOCATION, etc.

    -- Wikidata
    wikidata_qid VARCHAR(20),

    -- Embedding for entity matching
    embedding vector(1536),

    -- Statistics
    mention_count INT DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_entities_canonical_name ON core.entities(canonical_name);
CREATE INDEX idx_entities_type ON core.entities(entity_type);
CREATE INDEX idx_entities_wikidata_qid ON core.entities(wikidata_qid) WHERE wikidata_qid IS NOT NULL;
CREATE INDEX idx_entities_embedding ON core.entities USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE TRIGGER update_entities_updated_at BEFORE UPDATE ON core.entities
    FOR EACH ROW EXECUTE FUNCTION core.update_updated_at_column();

-- =============================================================================
-- NOTES
-- =============================================================================
--
-- This schema is intentionally minimal. The knowledge graph structure
-- (relationships, claim metadata, entity properties, etc.) lives in Neo4j.
--
-- Postgres is used only for:
-- 1. Page content storage (for extraction workers)
-- 2. Vector embeddings (for semantic search/matching)
-- 3. Basic metadata (for API queries)
--
-- See Neo4j graph model in: db/neo4j_graph_model.md
