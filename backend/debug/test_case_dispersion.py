"""
Test dispersion-based CaseView with 3-signal approach.

Loads data directly from Neo4j since principled surfaces use S000 format
and don't have centroids in PostgreSQL yet.

Signals:
- Signal 1: Backbone overlap (dispersion-based)
- Signal 2: Relation backbone (entity pairs that recur)
- Signal 3: Semantic similarity (NOT YET - no centroids for principled surfaces)

2-of-3 gate: Core edges require at least 2 signals.
"""

import asyncio
from services.neo4j_service import Neo4jService
from reee.views.case_view import CaseViewBuilder, EntityRole
from reee.types import Event, Surface


async def test_dispersion_case_view():
    # Initialize services
    neo4j = Neo4jService()
    await neo4j.connect()

    # Load surfaces with entities via HAS_ENTITY relationship
    surface_rows = await neo4j._execute_read("""
        MATCH (s:Surface)
        OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
        OPTIONAL MATCH (s)-[:HAS_ENTITY]->(e:Entity)
        WITH s,
             collect(DISTINCT c.id) as claim_ids,
             collect(DISTINCT e.canonical_name) as entity_names
        RETURN s.id as id, claim_ids, entity_names
    """)

    surfaces = {}
    for row in surface_rows:
        s = Surface(
            id=row["id"],
            claim_ids=set(row["claim_ids"] or []),
            entities=set(e for e in (row["entity_names"] or []) if e),
        )
        surfaces[row["id"]] = s

    print(f"Loaded {len(surfaces)} surfaces")
    print(f"Surfaces with entities: {sum(1 for s in surfaces.values() if s.entities)}")

    # Load events
    event_rows = await neo4j._execute_read("""
        MATCH (e:Event)-[:INCLUDES]->(s:Surface)
        WITH e, collect(s.id) as surface_ids, e.claim_count as claim_count
        RETURN e.id as id, surface_ids, claim_count
    """)

    events = {}
    for row in event_rows:
        event = Event(
            id=row["id"],
            surface_ids=set(row["surface_ids"] or []),
        )
        event.total_claims = row["claim_count"] or len(event.surface_ids)
        events[row["id"]] = event

    print(f"Loaded {len(events)} events")

    # Build cases with 3-signal approach (semantic disabled for now)
    case_builder = CaseViewBuilder(
        min_incidents=2,
        min_claims=3,
        hub_threshold=0.7,
        semantic_threshold=0.5,
        core_threshold=0.4,
    )
    result = await case_builder.build_from_events(events, surfaces)

    print(f"\n{'=' * 70}")
    print("ENTITY DISPERSION ANALYSIS")
    print(f"{'=' * 70}")

    # Show entities with freq >= 2
    frequent_roles = [
        r for r in result.entity_roles.values()
        if r.frequency >= 2
    ]

    # Sort by dispersion
    frequent_roles.sort(key=lambda r: r.dispersion)

    print("\nBACKBONES (low dispersion = co-anchors co-occur):")
    print("-" * 70)
    for role in frequent_roles[:10]:
        if role.role == "backbone":
            print(f"  {role.entity}")
            print(f"    {role.explanation[:100]}...")

    print("\nHUBS (high dispersion = co-anchors are disjoint):")
    print("-" * 70)
    for role in frequent_roles:
        if role.role == "hub":
            print(f"  {role.entity}")
            print(f"    {role.explanation[:100]}...")

    print(f"\n{'=' * 70}")
    print("RELATION BACKBONES (entity pairs that recur)")
    print(f"{'=' * 70}")
    for pair, count in sorted(result.relation_pairs.items(), key=lambda x: -x[1])[:15]:
        print(f"  {pair[0]} ↔ {pair[1]}: {count} events")

    print(f"\n{'=' * 70}")
    print(f"CASE VIEW RESULTS (L4) - 3 Signal Approach")
    print(f"{'=' * 70}")
    print(f"Total entities analyzed: {result.stats['entities_analyzed']}")
    print(f"Backbones identified: {result.stats['backbones']}")
    print(f"Hubs suppressed: {result.stats['hubs']}")
    print(f"Relation pairs: {result.stats['relation_pairs']}")
    print(f"Case edges (2-of-3 gate): {result.stats['case_edges']}")
    print(f"Cases formed: {result.stats['cases']}")

    print(f"\n{'=' * 70}")
    print("CASES WITH EXPLAINABILITY:")
    print(f"{'=' * 70}")
    for case in sorted(result.cases.values(), key=lambda c: -c.total_claims):
        print(f"\n{case.id}: {case.title}")
        print(f"  Events: {len(case.event_ids)} - {list(case.event_ids)[:5]}")
        print(f"  Claims: {case.total_claims}")
        print(f"  Backbone score: {case.backbone_score:.3f}")
        if case.formation_reason:
            print(f"  Formation: {case.formation_reason[:150]}...")

        # Show edge details
        if case.edges:
            print(f"  Edges ({len(case.edges)}):")
            for edge in case.edges[:3]:
                print(f"    {edge.event1_id} ↔ {edge.event2_id}: {edge.explanation[:80]}...")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(test_dispersion_case_view())
