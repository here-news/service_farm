"""
Test Event Weaver (Tier 2) with real claims.

Usage:
    docker exec herenews-app python -m test_eu.test_event_weaver
"""

import asyncio
import os
import sys
sys.path.insert(0, '/app/backend')

import asyncpg

from services.neo4j_service import Neo4jService
from repositories.claim_repository import ClaimRepository
from repositories.event_repository import EventRepository
from test_eu.core.event_weaver import EventWeaver, weave_claims_to_events


async def main():
    print("="*60)
    print("EVENT WEAVER TEST (Tier 2)")
    print("="*60)

    # Connect to databases
    print("\nConnecting to databases...")

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'phi_password_dev'),
        min_size=1,
        max_size=5
    )

    neo4j = Neo4jService(
        uri=os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
        user=os.getenv('NEO4J_USER', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD')
    )
    await neo4j.connect()

    claim_repo = ClaimRepository(db_pool, neo4j)
    event_repo = EventRepository(db_pool, neo4j)

    # Fetch claims from Hong Kong fire event
    print("\nFetching claims from Wang Fuk Court Fire event...")

    event_data = await neo4j._execute_read("""
        MATCH (e:Event)
        WHERE e.canonical_name CONTAINS 'Wang Fuk'
        RETURN e.id as id, e.canonical_name as name
        LIMIT 1
    """, {})

    if not event_data:
        print("Event not found!")
        return

    event = event_data[0]
    print(f"Found: {event['name']} ({event['id']})")

    claims = await event_repo.get_event_claims(event['id'])
    print(f"Loaded {len(claims)} claims")

    # Get embeddings for claims
    print("\nFetching embeddings from PostgreSQL...")
    from pgvector.asyncpg import register_vector

    embeddings = {}
    async with db_pool.acquire() as conn:
        await register_vector(conn)
        for claim in claims:
            result = await conn.fetchval("""
                SELECT embedding FROM core.claim_embeddings WHERE claim_id = $1
            """, claim.id)
            if result is not None and len(result) > 0:
                embeddings[claim.id] = [float(x) for x in result]

    print(f"Found {len(embeddings)} embeddings")

    # Test Event Weaver
    print("\n" + "="*60)
    print("WEAVING CLAIMS INTO EVENTS")
    print("="*60)

    weaver = EventWeaver()

    linked_count = 0
    created_count = 0

    for i, claim in enumerate(claims):
        embedding = embeddings.get(claim.id)
        result = await weaver.weave_claim(claim, embedding)

        if result['action'] == 'linked':
            linked_count += 1
        else:
            created_count += 1

        if (i + 1) % 30 == 0:
            print(f"  Processed {i + 1}/{len(claims)} claims...")
            print(f"    Events: {len(weaver.event_candidates)}, Linked: {linked_count}, Created: {created_count}")

    # Post-process: merge events with shared entities
    print("\nMerging events with shared entities...")
    merges = weaver.merge_events(min_shared_entities=1)
    print(f"  Merges performed: {merges}")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    summary = weaver.summary()
    print(f"\n  Total claims: {len(claims)}")
    print(f"  Event clusters: {summary['total_events']}")
    print(f"  Claims linked: {linked_count}")
    print(f"  New events created: {created_count}")
    print(f"  Compression: {len(claims) / max(summary['total_events'], 1):.1f}x")

    print(f"\n{'='*60}")
    print("EVENT CLUSTERS")
    print(f"{'='*60}")

    # Sort by claim count
    sorted_events = sorted(
        summary['events'],
        key=lambda e: e['claims'],
        reverse=True
    )

    for i, evt in enumerate(sorted_events[:10]):
        entities_str = ', '.join(evt['entities'][:3])
        time_str = f"{evt['time_start']} to {evt['time_end']}" if evt['time_start'] else "unknown"
        print(f"\n  [{i}] {evt['id']}: {evt['claims']} claims")
        print(f"      Entities: {entities_str}")
        print(f"      Time: {time_str}")

    # Temporal relations
    print(f"\n{'='*60}")
    print("TEMPORAL RELATIONS")
    print(f"{'='*60}")

    relations = weaver.compute_temporal_relations()
    for rel in relations[:10]:
        print(f"  {rel['event_a_name']} {rel['relation'].upper()} {rel['event_b_name']}")

    if not relations:
        print("  (No temporal relations - events may lack time data)")

    # Cleanup
    await db_pool.close()
    await neo4j.close()

    print(f"\n✓ Test complete")


if __name__ == '__main__':
    asyncio.run(main())
