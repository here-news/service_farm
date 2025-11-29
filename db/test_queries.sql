-- Test Queries for Gen2 Schema

-- =============================================================================
-- Test 1: Basic Data Validation
-- =============================================================================

\echo '=== Test 1: Data Counts ==='
SELECT 'Pages' AS table_name, COUNT(*) AS count FROM core.pages
UNION ALL
SELECT 'Entities', COUNT(*) FROM core.entities
UNION ALL
SELECT 'Edges', COUNT(*) FROM core.edges
UNION ALL
SELECT 'Claims', COUNT(*) FROM core.claims
UNION ALL
SELECT 'Events', COUNT(*) FROM core.events;

-- =============================================================================
-- Test 2: Multi-language Entity Lookup
-- =============================================================================

\echo ''
\echo '=== Test 2: Multi-language Entity Names ==='
SELECT
    canonical_name,
    entity_type,
    wikidata_qid,
    names_by_language
FROM core.entities
WHERE canonical_name = 'Hong Kong';

-- =============================================================================
-- Test 3: Page Summary Function
-- =============================================================================

\echo ''
\echo '=== Test 3: Page Summary Function ==='
SELECT core.get_page_summary('11111111-1111-1111-1111-111111111111'::uuid);

-- =============================================================================
-- Test 4: Event with Cross-Language Pages
-- =============================================================================

\echo ''
\echo '=== Test 4: Event Cross-Language Coverage ==='
SELECT
    e.title,
    e.event_type,
    e.languages,
    e.confidence,
    COUNT(DISTINCT pe.page_id) AS page_count,
    string_agg(DISTINCT p.language, ', ') AS page_languages
FROM core.events e
JOIN core.page_events pe ON e.id = pe.event_id
JOIN core.pages p ON pe.page_id = p.id
GROUP BY e.id, e.title, e.event_type, e.languages, e.confidence;

-- =============================================================================
-- Test 5: Entity Graph Edges (Wikidata integration)
-- =============================================================================

\echo ''
\echo '=== Test 5: Entity Edges with Wikidata Properties ==='
SELECT
    e_from.canonical_name AS from_entity,
    eg.edge_type,
    e_to.canonical_name AS to_entity,
    eg.confidence,
    eg.wikidata_properties
FROM core.edges eg
JOIN core.entities e_from ON eg.from_entity_id = e_from.id
JOIN core.entities e_to ON eg.to_entity_id = e_to.id;

-- =============================================================================
-- Test 6: Computed Confidence
-- =============================================================================

\echo ''
\echo '=== Test 6: Entity Confidence Computation ==='
SELECT
    canonical_name,
    semantic_confidence,
    structural_confidence,
    temporal_freshness,
    confidence AS computed_confidence
FROM core.entities
ORDER BY confidence DESC;

-- =============================================================================
-- Test 7: Bridge View (webapp interface)
-- =============================================================================

\echo ''
\echo '=== Test 7: Bridge View for Webapp ==='
SELECT
    title,
    status,
    language,
    submission_source,
    submitted_by_id
FROM bridge.artifacts_for_users
LIMIT 5;

-- =============================================================================
-- Test 8: Manual Recursive CTE (without function)
-- =============================================================================

\echo ''
\echo '=== Test 8: Recursive Entity Network (Jordan -> Hong Kong) ==='
WITH RECURSIVE entity_network AS (
    -- Base: Start with Jordan
    SELECT
        e.id AS entity_id,
        e.canonical_name,
        0 AS depth,
        1.0::float AS path_confidence
    FROM core.entities e
    WHERE e.canonical_name = 'Jordan, Hong Kong'

    UNION

    -- Recursive: Follow edges
    SELECT
        e.id AS entity_id,
        e.canonical_name,
        en.depth + 1 AS depth,
        (en.path_confidence * eg.confidence)::float AS path_confidence
    FROM core.entities e
    JOIN core.edges eg ON eg.to_entity_id = e.id
    JOIN entity_network en ON eg.from_entity_id = en.entity_id
    WHERE en.depth < 2
      AND eg.confidence >= 0.7
)
SELECT DISTINCT
    canonical_name,
    depth,
    path_confidence
FROM entity_network
ORDER BY depth, path_confidence DESC;
