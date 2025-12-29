"""
Test Phase 3: Recursive Weaver (Event-to-Event).

Usage:
    docker exec herenews-app python -m test_eu.test_recursive_weaver
"""

import asyncio
import os
import sys
sys.path.insert(0, '/app/backend')

import asyncpg

from services.neo4j_service import Neo4jService
from repositories.event_repository import EventRepository
from test_eu.core.event_weaver import EventWeaver, RecursiveWeaver


async def main():
    print("="*70)
    print("PHASE 3: RECURSIVE WEAVER TEST")
    print("="*70)

    # Connect
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'phi_password_dev'),
        min_size=1, max_size=5
    )

    neo4j = Neo4jService(
        uri=os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
        user=os.getenv('NEO4J_USER', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD')
    )
    await neo4j.connect()

    event_repo = EventRepository(db_pool, neo4j)

    # Get claims from Wang Fuk Court Fire
    print("\nLoading claims...")
    event_data = await neo4j._execute_read("""
        MATCH (e:Event) WHERE e.canonical_name CONTAINS 'Wang Fuk'
        RETURN e.id as id LIMIT 1
    """, {})
    claims = await event_repo.get_event_claims(event_data[0]['id'])
    print(f"Loaded {len(claims)} claims")

    # Get embeddings
    from pgvector.asyncpg import register_vector
    embeddings = {}
    async with db_pool.acquire() as conn:
        await register_vector(conn)
        for claim in claims:
            result = await conn.fetchval(
                "SELECT embedding FROM core.claim_embeddings WHERE claim_id = $1",
                claim.id
            )
            if result is not None and len(result) > 0:
                embeddings[claim.id] = [float(x) for x in result]

    # =========================================================================
    # TIER 2: Weave claims into events
    # =========================================================================
    print("\n" + "="*70)
    print("TIER 2: CLAIM → EVENT WEAVING")
    print("="*70)

    weaver = EventWeaver()
    for claim in claims:
        embedding = embeddings.get(claim.id)
        await weaver.weave_claim(claim, embedding)

    weaver.merge_events(min_shared_entities=1)
    print(f"\nEvents created: {len(weaver.event_candidates)}")

    # Show top events
    sorted_events = sorted(weaver.event_candidates, key=lambda e: len(e.claim_ids), reverse=True)
    for evt in sorted_events[:5]:
        print(f"  {evt.id}: {len(evt.claim_ids)} claims - {', '.join(list(evt.entity_names)[:3])}")

    # =========================================================================
    # PHASE 3: Weave events into meta-events
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 3: EVENT → META-EVENT WEAVING")
    print("="*70)

    recursive = RecursiveWeaver()
    result = await recursive.weave_events(weaver.event_candidates)

    print(f"\nMeta-events created: {result['total_meta_events']}")
    print(f"Event edges: {result['total_event_edges']}")

    print("\n--- META-EVENTS (Narrative Arcs) ---")
    for meta in result['meta_events']:
        print(f"\n  [{meta['id']}] {meta['label']}")
        print(f"    Contains {meta['events']} events")
        print(f"    Entities: {', '.join(meta['entities'][:5])}")
        print(f"    Time: {meta['time_start']} → {meta['time_end']}")

    print("\n--- NARRATIVE CHAINS ---")
    chains = result['narrative_chains']
    if chains:
        for i, chain in enumerate(chains[:5]):
            print(f"\n  Chain {i+1}: {' → '.join(chain)}")
    else:
        print("  (No temporal chains detected)")

    # =========================================================================
    # FULL PIPELINE SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("FULL PIPELINE SUMMARY")
    print("="*70)

    print(f"""
    ┌─────────────────────────────────────────────────────────┐
    │  INPUT: {len(claims)} claims                                     │
    │                          ↓                              │
    │  TIER 2 (EventWeaver): {len(weaver.event_candidates)} events                      │
    │                          ↓                              │
    │  PHASE 3 (RecursiveWeaver): {result['total_meta_events']} meta-events              │
    │                          ↓                              │
    │  Narrative Chains: {len(chains)}                                  │
    └─────────────────────────────────────────────────────────┘

    Compression:
      Claims → Events:      {len(claims) / max(len(weaver.event_candidates), 1):.1f}x
      Events → Meta-events: {len(weaver.event_candidates) / max(result['total_meta_events'], 1):.1f}x
      Total:                {len(claims) / max(result['total_meta_events'], 1):.1f}x
    """)

    await db_pool.close()
    await neo4j.close()

    print("✓ Recursive weaver test complete")


if __name__ == '__main__':
    asyncio.run(main())
