# Surface Persistence: Neo4j Domain Model

## Overview

Surfaces are L2 epistemic units - bundles of claims connected by identity edges.
They are **derived** from claims + parameters, not source-of-truth.

Key principle: Surfaces are materialized views with version tracking.

## Node Schema

### Surface Node

```cypher
(:Surface {
    // Identity
    id: "sf_xxxxxxxx",              // Surface ID

    // Versioning (critical for reproducibility)
    params_version: 1,               // Parameters version when computed
    computed_at: datetime(),         // When this surface was computed

    // Computed properties
    entropy: 0.45,                   // Semantic dispersion
    mass: 3.2,                       // Weighted claim count
    claim_count: 5,                  // Number of claims
    source_count: 3,                 // Number of distinct sources

    // Centroid (for similarity)
    centroid: [0.1, 0.2, ...],       // Embedding centroid (optional, for search)

    // Time window
    time_start: datetime(),          // Earliest claim timestamp
    time_end: datetime(),            // Latest claim timestamp

    // Semantic interpretation (optional, from LLM)
    canonical_title: "...",          // Human-readable title
    description: "...",              // Summary

    // Typed surface properties (for inquiry emergence)
    question_key: "death_count",     // If surface has typed claims
    canonical_value: 160,            // MAP estimate if typed
    value_entropy: 0.8,              // Uncertainty in value

    // Status
    status: "active"                 // active | stale | superseded
})
```

### Relationships

```cypher
// Surface contains claims (L2 → L0)
(s:Surface)-[:CONTAINS {
    added_at: datetime(),
    confidence: 0.85               // Identity confidence when added
}]->(c:Claim)

// Identity edges within surface (claim-to-claim)
(c1:Claim)-[:IDENTITY {
    relation: "confirms",          // confirms | refines | supersedes | conflicts
    confidence: 0.9,
    params_version: 1
}]->(c2:Claim)

// Surface-to-Surface relations (L2 → L2)
(s1:Surface)-[:SURFACE_REL {
    type: "conflicts",             // confirms | supersedes | conflicts | refines
    confidence: 0.7,
    evidence: {...},
    params_version: 1
}]->(s2:Surface)

// Surface belongs to Event (L2 → L3)
(s:Surface)-[:BELONGS_TO {
    level: "core",                 // core | periphery | quarantine
    score: 0.85,
    attached_at: datetime()
}]->(e:Event)

// Surface has entities (aggregated from claims)
(s:Surface)-[:ABOUT]->(ent:Entity)

// Anchor entities (key identifiers)
(s:Surface)-[:ANCHORED_BY]->(ent:Entity)
```

## Indexes

```cypher
// Primary lookups
CREATE INDEX surface_id IF NOT EXISTS FOR (s:Surface) ON (s.id);
CREATE INDEX surface_params IF NOT EXISTS FOR (s:Surface) ON (s.params_version);
CREATE INDEX surface_status IF NOT EXISTS FOR (s:Surface) ON (s.status);

// Query patterns
CREATE INDEX surface_question_key IF NOT EXISTS FOR (s:Surface) ON (s.question_key);
CREATE INDEX surface_entropy IF NOT EXISTS FOR (s:Surface) ON (s.entropy);

// Composite for versioned queries
CREATE INDEX surface_version_status IF NOT EXISTS FOR (s:Surface) ON (s.params_version, s.status);
```

## Query Patterns

### Get surfaces for event
```cypher
MATCH (e:Event {id: $event_id})<-[:BELONGS_TO]-(s:Surface)
WHERE s.status = 'active' AND s.params_version = $params_version
RETURN s, collect { MATCH (s)-[:CONTAINS]->(c:Claim) RETURN c } as claims
ORDER BY s.mass DESC
```

### Get typed surfaces (for inquiry emergence)
```cypher
MATCH (s:Surface)
WHERE s.question_key IS NOT NULL
  AND s.status = 'active'
  AND s.params_version = $params_version
RETURN s.id, s.question_key, s.canonical_value, s.value_entropy
ORDER BY s.value_entropy DESC
```

### Find conflicting surfaces
```cypher
MATCH (s1:Surface)-[r:SURFACE_REL {type: 'conflicts'}]->(s2:Surface)
WHERE s1.params_version = $params_version
RETURN s1, s2, r.confidence, r.evidence
```

### Get surface with full context
```cypher
MATCH (s:Surface {id: $surface_id})
OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
OPTIONAL MATCH (s)-[:ABOUT]->(ent:Entity)
OPTIONAL MATCH (s)-[:ANCHORED_BY]->(anchor:Entity)
OPTIONAL MATCH (s)-[:BELONGS_TO]->(e:Event)
RETURN s,
       collect(DISTINCT c) as claims,
       collect(DISTINCT ent) as entities,
       collect(DISTINCT anchor) as anchors,
       e as event
```

## Lifecycle

### 1. Surface Creation (from IdentityLinker)

```python
async def persist_surface(self, surface: Surface, claims: Dict[str, Claim]):
    """Persist computed surface to Neo4j."""

    # Create Surface node
    await self.neo4j._execute_write("""
        MERGE (s:Surface {id: $id})
        SET s.params_version = $params_version,
            s.computed_at = datetime(),
            s.entropy = $entropy,
            s.mass = $mass,
            s.claim_count = $claim_count,
            s.source_count = $source_count,
            s.time_start = $time_start,
            s.time_end = $time_end,
            s.question_key = $question_key,
            s.canonical_value = $canonical_value,
            s.status = 'active'
    """, {...})

    # Link to claims
    for claim_id in surface.claim_ids:
        await self.neo4j._execute_write("""
            MATCH (s:Surface {id: $surface_id})
            MATCH (c:Claim {id: $claim_id})
            MERGE (s)-[:CONTAINS]->(c)
        """, {...})
```

### 2. Surface Invalidation (on parameter change)

```python
async def invalidate_surfaces(self, old_version: int):
    """Mark surfaces computed with old params as stale."""

    await self.neo4j._execute_write("""
        MATCH (s:Surface)
        WHERE s.params_version = $old_version AND s.status = 'active'
        SET s.status = 'stale'
    """, {'old_version': old_version})
```

### 3. Surface Recomputation

When parameters change, recompute surfaces for affected events:

```python
async def recompute_surfaces(self, event_id: str, params: Parameters):
    """Recompute surfaces for an event with new parameters."""

    # Get claims for event
    claims = await self.claim_repo.get_by_event(event_id)

    # Run IdentityLinker with new params
    linker = IdentityLinker(params=params)
    for claim in claims:
        await linker.add_claim(claim)

    surfaces = linker.compute_surfaces()

    # Persist new surfaces
    for surface in surfaces.values():
        await self.persist_surface(surface, claims)
```

## Versioning Strategy

**Surfaces are ephemeral by design.** They can be recomputed from:
- L0 claims (immutable)
- Parameters (versioned)

Therefore:
1. Keep surfaces for current `params_version` as `status='active'`
2. On parameter change, mark old surfaces `status='stale'`
3. Optionally keep N historical versions for audit
4. Can always recompute from claims if needed

## Migration Path

1. Add Surface node type and indexes
2. Add `:CONTAINS` relationships (Surface → Claim)
3. Add `:IDENTITY` relationships (Claim → Claim) for edges
4. Add `:SURFACE_REL` relationships (Surface → Surface)
5. Backfill from existing events

## Open Questions

1. **Centroid storage**: Store as property (fast) or separate node (flexible)?
   - Recommendation: Property for now, can migrate later

2. **Identity edge storage**: On Claim nodes or separate?
   - Recommendation: On Claim nodes as `:IDENTITY` relationships
   - Allows querying edges independently of surfaces

3. **Historical surfaces**: Keep how many versions?
   - Recommendation: Current + 1 previous, garbage collect older

4. **Cross-event surfaces**: Can a surface span events?
   - Current: No, surfaces are computed per-event
   - Future: May need to handle event merges
