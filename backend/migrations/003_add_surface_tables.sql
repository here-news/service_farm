-- Migration 003: Add surface tables for REEE weaver
--
-- Architecture:
-- - Neo4j: Surface nodes, CONTAINS edges to Claims
-- - Postgres: Surface centroids for similarity search (pgvector)
--
-- Surfaces are L2 identity clusters - claims about the same referent.

-- =============================================================================
-- CONTENT.SURFACE_CENTROIDS: Surface embeddings for similarity matching
-- Neo4j holds: Surface nodes, claim membership, entity links
-- Postgres holds: Centroid embeddings for surface similarity search
-- =============================================================================

CREATE TABLE IF NOT EXISTS content.surface_centroids (
    surface_id VARCHAR(20) PRIMARY KEY,  -- sf_xxxxxxxx format

    -- Centroid embedding (average of claim embeddings)
    centroid vector(1536) NOT NULL,

    -- Metadata for filtering
    claim_count INT DEFAULT 0,
    source_count INT DEFAULT 0,

    -- Time bounds (for temporal filtering in similarity search)
    time_start TIMESTAMPTZ,
    time_end TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for similarity search (ivfflat)
-- NOTE: Only enable ivfflat when dataset > 10K rows
-- With small datasets, ivfflat's approximate search misses results
-- because it only probes 1 list by default.
--
-- For small datasets, exact search (sequential scan) is fast and accurate.
-- Uncomment when surface count exceeds ~10,000:
--
-- CREATE INDEX IF NOT EXISTS idx_surface_centroids_vector
--     ON content.surface_centroids
--     USING ivfflat (centroid vector_cosine_ops)
--     WITH (lists = 100);

-- Index for time-filtered queries
CREATE INDEX IF NOT EXISTS idx_surface_centroids_time
    ON content.surface_centroids (time_start, time_end);

-- Trigger for updated_at
CREATE TRIGGER update_surface_centroids_updated_at
    BEFORE UPDATE ON content.surface_centroids
    FOR EACH ROW EXECUTE FUNCTION core.update_updated_at_column();

-- =============================================================================
-- CONTENT.CLAIM_SURFACES: Mapping claims to surfaces (for join queries)
-- This enables: "find surfaces for these claims" without Neo4j roundtrip
-- =============================================================================

CREATE TABLE IF NOT EXISTS content.claim_surfaces (
    claim_id VARCHAR(20) PRIMARY KEY,  -- cl_xxxxxxxx format
    surface_id VARCHAR(20) NOT NULL,   -- sf_xxxxxxxx format

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_claim_surfaces_surface
    ON content.claim_surfaces (surface_id);

-- =============================================================================
-- NOTES
-- =============================================================================
--
-- Query patterns:
--
-- 1. Find candidate surfaces by embedding similarity:
--    SELECT surface_id, 1 - (centroid <=> $1::vector) as similarity
--    FROM content.surface_centroids
--    WHERE time_end > $2 AND time_start < $3
--    ORDER BY centroid <=> $1::vector
--    LIMIT 20;
--
-- 2. Find surfaces for a set of claims:
--    SELECT DISTINCT surface_id
--    FROM content.claim_surfaces
--    WHERE claim_id = ANY($1);
--
-- 3. Neo4j still handles:
--    - Surface node properties (entities, anchors, support)
--    - CONTAINS edges to Claims
--    - MENTIONS edges from Claims to Entities
