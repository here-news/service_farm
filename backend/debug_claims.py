import asyncio
from services.neo4j_service import Neo4jService
from reee.views.case_view import CaseViewBuilder

async def test_case_view():
    neo4j = Neo4jService()
    await neo4j.connect()

    # Load events and surfaces
    from reee.types import Event, Surface

    # Load surfaces
    surface_rows = await neo4j._execute_read("""
        MATCH (s:Surface)
        OPTIONAL MATCH (s)-[:CONTAINS]->(c:Claim)
        OPTIONAL MATCH (s)-[:HAS_ENTITY]->(e:Entity)
        WITH s,
             collect(DISTINCT c.id) as claim_ids,
             collect(DISTINCT e.canonical_name) as entities
        RETURN s.id as id, claim_ids, entities
    """)

    surfaces = {}
    for row in surface_rows:
        surfaces[row["id"]] = Surface(
            id=row["id"],
            claim_ids=set(row["claim_ids"] or []),
            entities=set(e for e in (row["entities"] or []) if e),
        )

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

    print(f"Loaded {len(events)} events, {len(surfaces)} surfaces")

    # Build cases
    case_builder = CaseViewBuilder(min_incidents=2, min_claims=3)
    result = await case_builder.build_from_events(events, surfaces)

    print(f"\n{'=' * 70}")
    print("CASE VIEW (L4) - Story-level aggregation")
    print(f"{'=' * 70}")
    print(f"Cases formed: {result.stats['cases']}")
    print(f"Backbone entities: {result.stats['backbone_entities']}")
    print(f"Avg events per case: {result.stats['avg_events_per_case']:.1f}")

    print(f"\n{'=' * 70}")
    print("CASES:")
    print(f"{'=' * 70}")
    for case in sorted(result.cases.values(), key=lambda c: -c.total_claims):
        print(f"\n{case.id}: {case.backbone_entity}")
        print(f"  Events: {len(case.event_ids)} - {list(case.event_ids)[:5]}")
        print(f"  Claims: {case.total_claims}")
        print(f"  Entities: {list(case.all_entities)[:6]}")
        if case.time_window[0]:
            print(f"  Time: {case.time_window[0]} to {case.time_window[1]}")

    await neo4j.close()

asyncio.run(test_case_view())
