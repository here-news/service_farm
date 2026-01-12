"""
Test EventJustification bundle.

This tests the explainability improvements:
1. Membrane proof (why grouped)
2. What happened (representative surfaces)
3. Canonical proposition handle (not just entities)
"""

import asyncio
from services.neo4j_service import Neo4jService
from reee.builders import PrincipledEventBuilder, PrincipledSurfaceBuilder
from reee.types import Claim


async def test_event_justification():
    # Initialize services
    neo4j = Neo4jService()
    await neo4j.connect()

    # Load claims from Neo4j
    claim_rows = await neo4j._execute_read("""
        MATCH (c:Claim)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        WITH c, collect(DISTINCT e.canonical_name) as entity_names
        WHERE size(entity_names) >= 2
        RETURN c.id as id, c.text as text, entity_names, c.event_time as event_time
        LIMIT 200
    """)

    claims = []
    for row in claim_rows:
        claims.append(Claim(
            id=row["id"],
            text=row["text"] or "",
            source="test",
            entities=set(e for e in (row["entity_names"] or []) if e),
            event_time=row["event_time"]
        ))

    print(f"Loaded {len(claims)} claims")

    # Build surfaces
    surface_builder = PrincipledSurfaceBuilder()
    surface_result = await surface_builder.build_from_claims(claims)
    print(f"Built {len(surface_result.surfaces)} surfaces")

    # Build events
    event_builder = PrincipledEventBuilder()
    event_result = await event_builder.build_from_surfaces(
        surface_result.surfaces,
        surface_result.ledger,
        min_claims_for_event=2
    )

    print(f"Built {len(event_result.events)} events")
    print(f"\n{'=' * 70}")
    print("EVENT JUSTIFICATIONS")
    print(f"{'=' * 70}")

    for event_id, event in sorted(event_result.events.items(), key=lambda x: -x[1].total_claims)[:5]:
        print(f"\n{event_id}: {event.total_claims} claims, {len(event.surface_ids)} surfaces")

        just = event.justification
        if not just:
            print("  [No justification]")
            continue

        print(f"\n  CANONICAL HANDLE: {just.canonical_handle}")
        print(f"  Citations: {just.handle_citations}")

        print(f"\n  MEMBRANE PROOF (why grouped):")
        if just.core_motifs:
            print(f"    Core motifs ({len(just.core_motifs)}):")
            for m in just.core_motifs[:3]:
                print(f"      {m['entities']} (support={m.get('support', 0)})")

        if just.context_passes:
            print(f"    Context passes ({len(just.context_passes)}):")
            for cp in just.context_passes[:3]:
                print(f"      {cp['surface1']} ↔ {cp['surface2']}: {cp['entity']} ({cp['status']})")

        if just.underpowered_edges:
            print(f"    Underpowered edges ({len(just.underpowered_edges)}):")
            for ue in just.underpowered_edges[:3]:
                print(f"      {ue['entity']}: {ue['reason']}")

        print(f"\n  WHAT HAPPENED:")
        print(f"    Representative surfaces: {just.representative_surfaces}")
        if just.representative_titles:
            print(f"    Titles:")
            for title in just.representative_titles[:3]:
                print(f"      - {title[:80]}...")
        if just.representative_facts:
            print(f"    Key facts:")
            for fact in just.representative_facts[:3]:
                print(f"      - {fact[:80]}...")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(test_event_justification())
