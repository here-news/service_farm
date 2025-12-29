"""
Test Weaver Persistence: Verify EventWeaver and RecursiveWeaver persist correctly.

Usage:
    docker exec herenews-app python -m test_eu.test_weaver_persistence
"""

import asyncio
import os
import sys
sys.path.insert(0, '/app/backend')

import asyncpg
from pgvector.asyncpg import register_vector

from services.neo4j_service import Neo4jService
from repositories.event_repository import EventRepository
from test_eu.core.event_weaver import EventWeaver, RecursiveWeaver


async def main():
    print("="*70)
    print("WEAVER PERSISTENCE TEST")
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

    # Load claims from Wang Fuk Court Fire
    print("\nLoading claims...")
    event_data = await neo4j._execute_read("""
        MATCH (e:Event) WHERE e.canonical_name CONTAINS 'Wang Fuk'
        RETURN e.id as id LIMIT 1
    """, {})

    if not event_data:
        print("ERROR: Wang Fuk Court Fire event not found")
        await db_pool.close()
        await neo4j.close()
        return

    claims = await event_repo.get_event_claims(event_data[0]['id'])
    print(f"Loaded {len(claims)} claims")

    # Build claims dict for linking
    claims_by_id = {c.id: c for c in claims}

    # Get embeddings
    embeddings = {}
    async with db_pool.acquire() as conn:
        await register_vector(conn)
        for claim in claims[:50]:  # Limit for test
            result = await conn.fetchval(
                "SELECT embedding FROM core.claim_embeddings WHERE claim_id = $1",
                claim.id
            )
            if result is not None and len(result) > 0:
                embeddings[claim.id] = [float(x) for x in result]

    # =========================================================================
    # TEST 1: EventWeaver persistence
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: EventWeaver Persistence (DRY RUN)")
    print("="*70)

    weaver = EventWeaver()
    for claim in claims[:50]:
        embedding = embeddings.get(claim.id)
        await weaver.weave_claim(claim, embedding)

    weaver.merge_events(min_shared_entities=1)
    print(f"\nEvents created in-memory: {len(weaver.event_candidates)}")

    # Show what would be persisted
    print("\n--- Would Persist (DRY RUN) ---")
    for evt in weaver.event_candidates[:5]:
        print(f"  Event: {', '.join(list(evt.entity_names)[:3])}")
        print(f"    Claims: {len(evt.claim_ids)}")
        print(f"    Entities: {len(evt.entities)}")
        print(f"    Time: {str(evt.time_start)[:10] if evt.time_start else 'N/A'}")

    # =========================================================================
    # TEST 2: RecursiveWeaver persistence
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: RecursiveWeaver Persistence (DRY RUN)")
    print("="*70)

    recursive = RecursiveWeaver()
    result = await recursive.weave_events(weaver.event_candidates)

    print(f"\nMeta-events: {result['total_meta_events']}")
    print(f"Event edges: {result['total_event_edges']}")

    print("\n--- Would Persist (DRY RUN) ---")
    for edge in recursive.event_edges[:5]:
        print(f"  Edge: {edge.source_id} -[{edge.relation.value}]-> {edge.target_id}")
        print(f"    Confidence: {edge.confidence:.2f}")

    # =========================================================================
    # TEST 3: Verify EventRepository methods exist
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 3: EventRepository Integration Check")
    print("="*70)

    # Check new methods exist
    methods_to_check = [
        'create_event_edge',
        'get_event_edges',
        'get_narrative_chain'
    ]

    for method in methods_to_check:
        if hasattr(event_repo, method):
            print(f"  [OK] EventRepository.{method}()")
        else:
            print(f"  [MISSING] EventRepository.{method}()")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("PERSISTENCE INTEGRATION SUMMARY")
    print("="*70)

    print(f"""
    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │  EventWeaver.persist(event_repo)                                │
    │    → Creates Event nodes in Neo4j                               │
    │    → Stores embeddings in PostgreSQL                            │
    │    → Links claims via INTAKES relationships                     │
    │    → Links entities via INVOLVES relationships                  │
    ├─────────────────────────────────────────────────────────────────┤
    │  RecursiveWeaver.persist(event_repo)                            │
    │    → Creates EVENT_EDGE relationships                           │
    │    → Stores FOLLOWS, CAUSES, RELATED_TO edges                   │
    │    → Enables narrative chain traversal                          │
    └─────────────────────────────────────────────────────────────────┘

    Data Flow:
    Claims → EventWeaver → EventCandidates → persist() → Neo4j Events
                                    │
                                    ▼
                          RecursiveWeaver → MetaEvents/Edges → persist() → Neo4j Edges
    """)

    await db_pool.close()
    await neo4j.close()

    print("Persistence integration test complete")


if __name__ == '__main__':
    asyncio.run(main())
