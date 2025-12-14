-- Service Farm Gen2 Schema
-- PostgreSQL-first architecture for knowledge graph
-- Based on ADR-001: Database Architecture Principles
--
-- Version: 2.0
-- Date: 2025-11-29
-- Migration from: Gen1 (Neo4j + PostgreSQL hybrid)

-- =============================================================================
-- EXTENSIONS
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";      -- Fuzzy text search
CREATE EXTENSION IF NOT EXISTS "vector";       -- pgvector for embeddings

-- =============================================================================
-- SCHEMAS
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS core;      -- Knowledge graph (service_farm owns)
CREATE SCHEMA IF NOT EXISTS bridge;    -- Interface for webapp
CREATE SCHEMA IF NOT EXISTS system;    -- Workers, jobs, queues

-- =============================================================================
-- CORE SCHEMA: Knowledge Graph
-- =============================================================================

-- -----------------------------------------------------------------------------
-- PAGES: Artifacts + Content (merged for simplicity)
-- Replaces: Gen1 extraction_tasks + Neo4j Page nodes
-- -----------------------------------------------------------------------------

CREATE TABLE core.pages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- URLs (hot columns, indexed)
    url TEXT NOT NULL,
    canonical_url TEXT UNIQUE NOT NULL,

    -- Content
    title TEXT,
    description TEXT,
    content_text TEXT,
    byline TEXT,
    author TEXT,
    thumbnail_url TEXT,
    site_name TEXT,
    domain TEXT,

    -- Language detection
    language VARCHAR(10) DEFAULT 'en',
    language_confidence FLOAT DEFAULT 0.0,

    -- Metadata quality
    metadata_confidence FLOAT DEFAULT 0.0,  -- 0.0-1.0 based on completeness of title/description/author/thumbnail

    -- Metadata
    word_count INT DEFAULT 0,
    pub_time TIMESTAMPTZ,

    -- Status (hot column, indexed)
    status VARCHAR(50) DEFAULT 'stub',
    -- Possible values: stub, preview, extracted, semantic_complete,
    --                  failed, semantic_failed, event_complete

    -- Processing stage (for debugging)
    current_stage VARCHAR(100),
    error_message TEXT,

    -- Health flags
    is_healthy BOOLEAN DEFAULT TRUE,

    -- Asset flags (GCS storage)
    has_screenshot BOOLEAN DEFAULT FALSE,
    has_html BOOLEAN DEFAULT FALSE,
    has_metadata BOOLEAN DEFAULT FALSE,

    -- Flexible metadata (JSONB pattern from Gen1)
    metadata JSONB DEFAULT '{}'::jsonb,
    -- Example: {"content_health": {...}, "extraction_context": {...}}

    -- Timestamps (hot columns)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    -- Embedding for semantic search
    embedding vector(1536)  -- OpenAI ada-002 dimension
);

-- Indexes
CREATE INDEX idx_pages_canonical_url ON core.pages(canonical_url);
CREATE INDEX idx_pages_status ON core.pages(status);
CREATE INDEX idx_pages_domain ON core.pages(domain);
CREATE INDEX idx_pages_language ON core.pages(language);
CREATE INDEX idx_pages_created_at ON core.pages(created_at DESC);
CREATE INDEX idx_pages_pub_time ON core.pages(pub_time DESC);
CREATE INDEX idx_pages_metadata_gin ON core.pages USING GIN(metadata);
CREATE INDEX idx_pages_embedding ON core.pages USING ivfflat (embedding vector_cosine_ops);

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

-- -----------------------------------------------------------------------------
-- ENTITIES: Canonical knowledge graph nodes
-- Replaces: Neo4j Person, Organization, Location nodes
-- -----------------------------------------------------------------------------

CREATE TYPE core.entity_type AS ENUM (
    'PERSON',
    'ORGANIZATION',
    'LOCATION',
    'EVENT',        -- For event-as-entity references
    'CONCEPT',      -- Abstract concepts
    'PRODUCT',
    'WORK_OF_ART',
    'LAW',
    'OTHER'
);

CREATE TABLE core.entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Canonical naming
    canonical_name TEXT NOT NULL,
    entity_type core.entity_type NOT NULL,

    -- Multi-language support (docs/09.architecture.principles.md lines 169-198)
    names_by_language JSONB DEFAULT '{}'::jsonb,
    -- Example: {"en": ["Hong Kong", "HK"], "zh": ["香港"], "fr": ["Hong Kong"]}

    -- Wikidata integration (ADR-001 hybrid approach)
    wikidata_qid VARCHAR(20),  -- Q8646 for Hong Kong
    wikidata_properties JSONB, -- {"P31": "Q515", "P17": "Q148"}

    -- Confidence (docs/09.architecture.principles.md lines 271-330)
    semantic_confidence FLOAT DEFAULT 0.5,   -- NER + Wikidata match
    structural_confidence FLOAT DEFAULT 0.5, -- Graph topology signals
    temporal_freshness FLOAT DEFAULT 1.0,    -- Decay over time

    -- Combined confidence (computed)
    confidence FLOAT GENERATED ALWAYS AS (
        (semantic_confidence * 0.4 +
         structural_confidence * 0.4 +
         temporal_freshness * 0.2)
    ) STORED,

    -- Entity profile (enrichment from Gen1)
    profile_summary TEXT,
    profile_roles JSONB,         -- ["CEO", "Founder"]
    profile_affiliations JSONB,  -- ["Tesla", "SpaceX"]
    profile_key_facts JSONB,     -- ["Born 1971", "South African"]
    profile_locations JSONB,     -- ["California", "Texas"]

    -- Statistics
    mention_count INT DEFAULT 0,
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),

    -- Enrichment tracking
    last_enriched_at TIMESTAMPTZ,

    -- Status
    status VARCHAR(50) DEFAULT 'stub',
    -- Possible: stub, partial, enriched, validated, canonical, disputed

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Embedding for semantic matching
    embedding vector(1536),

    -- Uniqueness constraint
    UNIQUE(canonical_name, entity_type, wikidata_qid)
);

-- Indexes
CREATE INDEX idx_entities_canonical_name ON core.entities(canonical_name);
CREATE INDEX idx_entities_type ON core.entities(entity_type);
CREATE INDEX idx_entities_wikidata_qid ON core.entities(wikidata_qid) WHERE wikidata_qid IS NOT NULL;
CREATE INDEX idx_entities_confidence ON core.entities(confidence DESC);
CREATE INDEX idx_entities_names_gin ON core.entities USING GIN(names_by_language);
CREATE INDEX idx_entities_embedding ON core.entities USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_entities_canonical_trgm ON core.entities USING GIN(canonical_name gin_trgm_ops);

CREATE TRIGGER update_entities_updated_at BEFORE UPDATE ON core.entities
    FOR EACH ROW EXECUTE FUNCTION core.update_updated_at_column();

-- -----------------------------------------------------------------------------
-- EDGES: Graph relationships (replaces Neo4j relationships)
-- ADR-001: Hybrid Wikidata property integration
-- -----------------------------------------------------------------------------

CREATE TYPE core.edge_type AS ENUM (
    -- Location relationships (P276, P131, P17 → simplified)
    'LOCATED_IN',
    'BORDERS',

    -- Organizational (P108, P463, P361)
    'EMPLOYED_BY',
    'MEMBER_OF',
    'PART_OF',
    'OWNS',

    -- Events (P1344, P828, P156)
    'PARTICIPATED_IN',
    'CAUSED_BY',
    'FOLLOWED_BY',
    'RELATED_TO',

    -- Content relationships (our custom)
    'MENTIONED_IN',
    'CO_OCCURS_WITH',

    -- Disputes
    'CONTRADICTS',
    'SUPPORTS'
);

CREATE TABLE core.edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Edge endpoints (polymorphic: entity→entity, entity→event, etc.)
    from_entity_id UUID REFERENCES core.entities(id) ON DELETE CASCADE,
    to_entity_id UUID REFERENCES core.entities(id) ON DELETE CASCADE,

    -- Relationship type (our simplified ontology)
    edge_type core.edge_type NOT NULL,

    -- Wikidata properties (optional, for enrichment)
    wikidata_properties JSONB,
    -- Example: {"P276": 0.95, "P131": 0.85}

    -- Confidence
    confidence FLOAT DEFAULT 0.5,

    -- Evidence
    evidence_count INT DEFAULT 1,
    evidence_sources UUID[],  -- Array of page IDs
    source VARCHAR(20) DEFAULT 'extracted',
    -- Possible: extracted, wikidata, user, inferred

    -- Temporal decay (docs/09.architecture.principles.md lines 201-227)
    decay_rate FLOAT DEFAULT 0.01,  -- Per-day decay
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    last_reconfirmed_at TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Uniqueness: one edge type per entity pair
    UNIQUE(from_entity_id, to_entity_id, edge_type)
);

-- Indexes
CREATE INDEX idx_edges_from ON core.edges(from_entity_id);
CREATE INDEX idx_edges_to ON core.edges(to_entity_id);
CREATE INDEX idx_edges_type ON core.edges(edge_type);
CREATE INDEX idx_edges_confidence ON core.edges(confidence DESC);
CREATE INDEX idx_edges_last_seen ON core.edges(last_seen DESC);

CREATE TRIGGER update_edges_updated_at BEFORE UPDATE ON core.edges
    FOR EACH ROW EXECUTE FUNCTION core.update_updated_at_column();

-- -----------------------------------------------------------------------------
-- CLAIMS: Atomic facts extracted from pages
-- Replaces: Neo4j Claim nodes
-- -----------------------------------------------------------------------------

CREATE TABLE core.claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source
    page_id UUID REFERENCES core.pages(id) ON DELETE CASCADE,

    -- Claim content
    text TEXT NOT NULL,

    -- Temporal info
    event_time TIMESTAMPTZ,  -- When the claim says something happened

    -- Confidence
    confidence FLOAT DEFAULT 0.5,

    -- Claim type/modality
    modality VARCHAR(50),
    -- Possible: observation, allegation, announcement, action, quote

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Embedding for semantic clustering
    embedding vector(1536)
);

-- Indexes
CREATE INDEX idx_claims_page ON core.claims(page_id);
CREATE INDEX idx_claims_event_time ON core.claims(event_time);
CREATE INDEX idx_claims_confidence ON core.claims(confidence DESC);
CREATE INDEX idx_claims_embedding ON core.claims USING ivfflat (embedding vector_cosine_ops);

-- -----------------------------------------------------------------------------
-- CLAIM_ENTITIES: Which entities are mentioned in which claims
-- -----------------------------------------------------------------------------

CREATE TABLE core.claim_entities (
    claim_id UUID REFERENCES core.claims(id) ON DELETE CASCADE,
    entity_id UUID REFERENCES core.entities(id) ON DELETE CASCADE,
    mention_context TEXT,  -- Sentence/context where entity appears
    PRIMARY KEY (claim_id, entity_id)
);

CREATE INDEX idx_claim_entities_claim ON core.claim_entities(claim_id);
CREATE INDEX idx_claim_entities_entity ON core.claim_entities(entity_id);

-- -----------------------------------------------------------------------------
-- PAGE_ENTITIES: Which entities are mentioned in which pages
-- Replaces: Neo4j Page→Entity relationships
-- -----------------------------------------------------------------------------

CREATE TABLE core.page_entities (
    page_id UUID REFERENCES core.pages(id) ON DELETE CASCADE,
    entity_id UUID REFERENCES core.entities(id) ON DELETE CASCADE,
    mention_count INT DEFAULT 1,
    PRIMARY KEY (page_id, entity_id)
);

CREATE INDEX idx_page_entities_page ON core.page_entities(page_id);
CREATE INDEX idx_page_entities_entity ON core.page_entities(entity_id);

-- -----------------------------------------------------------------------------
-- EVENTS: Factual happenings (docs/09.architecture.principles.md lines 138-142)
-- Replaces: Neo4j Event nodes (new in Gen2)
-- -----------------------------------------------------------------------------

CREATE TABLE core.events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Event identification
    title TEXT NOT NULL,
    summary TEXT,
    event_type VARCHAR(50),
    -- Possible: legal_proceeding, policy_announcement, incident,
    --           investigation, election, disaster, etc.

    -- Temporal
    event_start TIMESTAMPTZ,
    event_end TIMESTAMPTZ,

    -- Location
    location TEXT,

    -- Multi-language support
    languages JSONB DEFAULT '[]'::jsonb,  -- ["en", "zh", "fr"]

    -- Confidence (docs/09.architecture.principles.md event formation)
    confidence FLOAT DEFAULT 0.5,
    coherence_score FLOAT DEFAULT 0.5,  -- How well claims cluster

    -- Hierarchy (IMPLEMENTATION_BASIS.md lines 108-119)
    parent_event_id UUID REFERENCES core.events(id),
    event_scale VARCHAR(20) DEFAULT 'meso',
    -- Possible: micro, meso, macro, story

    relationship_type VARCHAR(50),
    -- Possible: PHASE_OF, PART_OF, CAUSED_BY, FOLLOWS

    -- Statistics
    claims_count INT DEFAULT 0,
    pages_count INT DEFAULT 0,

    -- Status
    status VARCHAR(50) DEFAULT 'active',
    -- Possible: active, merged_into, disputed, archived

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Embedding for semantic matching
    embedding vector(1536)
);

-- Indexes
CREATE INDEX idx_events_event_start ON core.events(event_start DESC);
CREATE INDEX idx_events_event_type ON core.events(event_type);
CREATE INDEX idx_events_parent ON core.events(parent_event_id) WHERE parent_event_id IS NOT NULL;
CREATE INDEX idx_events_confidence ON core.events(confidence DESC);
CREATE INDEX idx_events_embedding ON core.events USING ivfflat (embedding vector_cosine_ops);

CREATE TRIGGER update_events_updated_at BEFORE UPDATE ON core.events
    FOR EACH ROW EXECUTE FUNCTION core.update_updated_at_column();

-- -----------------------------------------------------------------------------
-- PAGE_EVENTS: Which pages discuss which events
-- -----------------------------------------------------------------------------

CREATE TABLE core.page_events (
    page_id UUID REFERENCES core.pages(id) ON DELETE CASCADE,
    event_id UUID REFERENCES core.events(id) ON DELETE CASCADE,
    PRIMARY KEY (page_id, event_id)
);

CREATE INDEX idx_page_events_page ON core.page_events(page_id);
CREATE INDEX idx_page_events_event ON core.page_events(event_id);

-- -----------------------------------------------------------------------------
-- EVENT_ENTITIES: Which entities participated in which events
-- -----------------------------------------------------------------------------

CREATE TABLE core.event_entities (
    event_id UUID REFERENCES core.events(id) ON DELETE CASCADE,
    entity_id UUID REFERENCES core.entities(id) ON DELETE CASCADE,
    role VARCHAR(100),  -- Optional: "defendant", "prosecutor", "witness"
    PRIMARY KEY (event_id, entity_id)
);

CREATE INDEX idx_event_entities_event ON core.event_entities(event_id);
CREATE INDEX idx_event_entities_entity ON core.event_entities(entity_id);

-- =============================================================================
-- BRIDGE SCHEMA: Interface for webapp
-- =============================================================================

-- -----------------------------------------------------------------------------
-- ARTIFACT_METADATA: Who submitted what (soft FK to webapp.users)
-- -----------------------------------------------------------------------------

CREATE TABLE bridge.artifact_metadata (
    artifact_id UUID PRIMARY KEY REFERENCES core.pages(id) ON DELETE CASCADE,

    -- External references (webapp owns users table)
    submitted_by_id UUID,  -- Soft reference to webapp.users.id

    -- Submission context
    submission_source VARCHAR(50),
    -- Possible: browser_extension, api, bulk_import, worker

    -- User-specific metadata (flexible for webapp needs)
    user_metadata JSONB DEFAULT '{}'::jsonb,
    -- Example: {"is_bookmarked": true, "tags": ["important"], "user_notes": "..."}

    -- Timestamps
    submitted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_artifact_metadata_submitted_by ON bridge.artifact_metadata(submitted_by_id);
CREATE INDEX idx_artifact_metadata_source ON bridge.artifact_metadata(submission_source);

-- -----------------------------------------------------------------------------
-- VIEW: Artifacts for users (webapp-facing)
-- -----------------------------------------------------------------------------

CREATE VIEW bridge.artifacts_for_users AS
SELECT
    p.id,
    p.canonical_url,
    p.title,
    p.status,
    p.domain,
    p.language,
    p.word_count,
    p.pub_time,
    p.created_at,
    am.submitted_by_id,
    am.submission_source,
    am.submitted_at,
    am.user_metadata
FROM core.pages p
LEFT JOIN bridge.artifact_metadata am ON p.id = am.artifact_id;

-- =============================================================================
-- SYSTEM SCHEMA: Workers, jobs, queues
-- =============================================================================

-- -----------------------------------------------------------------------------
-- WORKER_JOBS: Async job tracking
-- -----------------------------------------------------------------------------

CREATE TYPE system.job_priority AS ENUM (
    'critical',  -- < 5s
    'high',      -- < 30s
    'normal',    -- < 5min
    'low',       -- < 1hr
    'batch'      -- No limit
);

CREATE TYPE system.job_status AS ENUM (
    'queued',
    'processing',
    'completed',
    'failed',
    'retrying'
);

CREATE TABLE system.worker_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Job details
    job_type VARCHAR(100) NOT NULL,
    -- Possible: extract_url, extract_entities, extract_claims,
    --           cluster_events, enrich_entity, recompute_confidence

    priority system.job_priority DEFAULT 'normal',
    status system.job_status DEFAULT 'queued',

    -- Target
    target_id UUID,  -- page_id, entity_id, event_id, etc.

    -- Payload
    payload JSONB DEFAULT '{}'::jsonb,

    -- Execution
    worker_id VARCHAR(100),
    attempts INT DEFAULT 0,
    max_attempts INT DEFAULT 3,
    error_message TEXT,

    -- Timestamps
    queued_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    next_retry_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX idx_worker_jobs_status ON system.worker_jobs(status);
CREATE INDEX idx_worker_jobs_priority ON system.worker_jobs(priority, queued_at);
CREATE INDEX idx_worker_jobs_target ON system.worker_jobs(target_id);
CREATE INDEX idx_worker_jobs_next_retry ON system.worker_jobs(next_retry_at)
    WHERE status = 'retrying';

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Get page summary with entity/event counts
CREATE OR REPLACE FUNCTION core.get_page_summary(p_page_id UUID)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'page_id', p.id,
        'title', p.title,
        'status', p.status,
        'entities_count', COUNT(DISTINCT pe.entity_id),
        'events_count', COUNT(DISTINCT pev.event_id),
        'claims_count', COUNT(DISTINCT c.id)
    ) INTO result
    FROM core.pages p
    LEFT JOIN core.page_entities pe ON p.id = pe.page_id
    LEFT JOIN core.page_events pev ON p.id = pev.page_id
    LEFT JOIN core.claims c ON p.id = c.page_id
    WHERE p.id = p_page_id
    GROUP BY p.id, p.title, p.status;

    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Graph traversal: Find connected entities (recursive CTE example)
CREATE OR REPLACE FUNCTION core.find_entity_network(
    p_entity_id UUID,
    p_max_depth INT DEFAULT 2,
    p_min_confidence FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    entity_id UUID,
    canonical_name TEXT,
    depth INT,
    path_confidence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE entity_network AS (
        -- Base case
        SELECT
            e.id AS entity_id,
            e.canonical_name,
            0 AS depth,
            1.0::float AS path_confidence
        FROM core.entities e
        WHERE e.id = p_entity_id

        UNION

        -- Recursive case
        SELECT
            e.id AS entity_id,
            e.canonical_name,
            en.depth + 1 AS depth,
            (en.path_confidence * eg.confidence)::float AS path_confidence
        FROM core.entities e
        JOIN core.edges eg ON eg.to_entity_id = e.id
        JOIN entity_network en ON eg.from_entity_id = en.entity_id
        WHERE en.depth < p_max_depth
          AND eg.confidence >= p_min_confidence
    )
    SELECT DISTINCT
        entity_network.entity_id,
        entity_network.canonical_name,
        entity_network.depth,
        entity_network.path_confidence
    FROM entity_network
    ORDER BY depth, path_confidence DESC;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- COMMENTS (Documentation)
-- =============================================================================

COMMENT ON SCHEMA core IS 'Core knowledge graph: pages, entities, events, claims, edges';
COMMENT ON SCHEMA bridge IS 'Interface for webapp: artifact metadata, views';
COMMENT ON SCHEMA system IS 'Worker infrastructure: jobs, queues';

COMMENT ON TABLE core.pages IS 'Artifacts + content (merged from Gen1 extraction_tasks + Neo4j Page)';
COMMENT ON TABLE core.entities IS 'Canonical entities with multi-language support and Wikidata linking';
COMMENT ON TABLE core.edges IS 'Graph relationships with Wikidata property integration';
COMMENT ON TABLE core.claims IS 'Atomic facts extracted from pages';
COMMENT ON TABLE core.events IS 'Factual happenings formed by clustering claims';

COMMENT ON COLUMN core.edges.wikidata_properties IS 'Optional Wikidata P-numbers for enrichment: {"P276": 0.95}';
COMMENT ON COLUMN core.entities.names_by_language IS 'Multi-language aliases: {"en": ["Hong Kong"], "zh": ["香港"]}';
COMMENT ON COLUMN core.entities.confidence IS 'Computed: semantic * 0.4 + structural * 0.4 + temporal * 0.2';
