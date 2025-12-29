"""
Test kernel with pgvector integration using real claims.

Usage:
    docker exec herenews-app python -m test_eu.test_kernel_pgvector
"""

import asyncio
import os
import sys
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/backend/test_eu')

from openai import AsyncOpenAI
import asyncpg

from services.neo4j_service import Neo4jService
from repositories.claim_repository import ClaimRepository
from test_eu.core.kernel import EpistemicKernel, process_domain_claims


async def main():
    # Connect to databases
    print("Connecting to databases...")

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
    openai_client = AsyncOpenAI()

    # Fetch claims from a real event (Hong Kong fire)
    print("\nFetching claims from Hong Kong fire event...")

    # Get event claims with entities
    event_id = await neo4j._execute_read("""
        MATCH (e:Event)
        WHERE e.canonical_name CONTAINS 'Hong Kong'
           OR e.canonical_name CONTAINS 'Wang Fuk'
           OR e.canonical_name CONTAINS 'fire'
        RETURN e.id as id, e.canonical_name as name
        LIMIT 1
    """, {})

    if not event_id:
        print("No Hong Kong fire event found, fetching random claims...")
        # Fallback: get random claims with embeddings
        claims_data = await db_pool.fetch("""
            SELECT c.claim_id, c.embedding
            FROM core.claim_embeddings c
            LIMIT 30
        """)

        claims = []
        for row in claims_data:
            claim = await claim_repo.get_by_id(row['claim_id'])
            if claim:
                # Hydrate entities
                claim = await claim_repo.hydrate_entities(claim)
                claims.append(claim)
    else:
        event = event_id[0]
        print(f"Found event: {event['name']} ({event['id']})")

        # Get claims for this event with entities hydrated
        from repositories.event_repository import EventRepository
        event_repo = EventRepository(db_pool, neo4j)
        claims = await event_repo.get_event_claims(event['id'])
        print(f"Loaded {len(claims)} claims with entities")

    if not claims:
        print("No claims found!")
        return

    # Show sample claims
    print(f"\nSample claims ({len(claims)} total):")
    for i, c in enumerate(claims[:5]):
        entities = [e.canonical_name for e in (c.entities or [])]
        print(f"  [{i}] {c.text[:80]}...")
        if entities:
            print(f"       Entities: {', '.join(entities[:3])}")

    def format_time(t):
        """Format event_time whether it's datetime or string."""
        if t is None:
            return None
        if hasattr(t, 'strftime'):
            return t.strftime('%Y-%m-%d')
        return str(t)[:10]  # String: take first 10 chars

    # FULL SCALE TEST: All claims with pgvector
    print("\n" + "="*60)
    print(f"FULL SCALE TEST: {len(claims)} claims with pgvector")
    print("="*60)

    import time
    start_time = time.time()

    kernel_full = EpistemicKernel(
        llm_client=openai_client,
        claim_repository=claim_repo
    )

    for i, claim in enumerate(claims):
        context_parts = [claim.text]
        if claim.entities:
            context_parts.append(f"Entities: {', '.join(e.canonical_name for e in claim.entities[:3])}")
        if claim.event_time:
            context_parts.append(f"Time: {format_time(claim.event_time)}")
        context = ' | '.join(context_parts)

        source = claim.metadata.get('source_name', 'unknown')
        await kernel_full.process(claim.text, source, context)

        # Progress indicator every 20 claims
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(claims)} claims...")

    elapsed = time.time() - start_time
    summary = kernel_full.summary()

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Claims processed: {summary['total_claims']}")
    print(f"  Nodes created: {summary['total_beliefs']}")
    print(f"  Compression: {summary['compression']:.1f}x")
    print(f"  ")
    print(f"  Confirmed (3+ sources): {summary['confirmed']}")
    print(f"  Corroborated (2 sources): {summary['corroborated']}")
    print(f"  Single source: {summary['single_source']}")
    print(f"  Conflicts: {summary['conflicts']}")
    print(f"  ")
    print(f"  LLM calls: {summary['llm_calls']}")
    print(f"  LLM skipped: {summary['llm_calls_skipped']}")
    print(f"  pgvector queries: {summary['pgvector_queries']}")
    print(f"  ")
    print(f"  Coherence: {summary['coherence']:.2f}")
    print(f"  Entropy: {summary['entropy']:.2f}")
    print(f"  Time: {elapsed:.1f}s ({len(claims)/elapsed:.1f} claims/sec)")
    print(f"  ")
    print(f"  Relations: {summary['relations']}")

    # Show topology
    print("\n" + "="*60)
    print("TOPOLOGY STRUCTURE")
    print("="*60)

    topo = kernel_full.topology()
    print(f"\nNodes: {topo['stats']['total_nodes']}")
    print(f"Edges: {topo['stats']['total_edges']}")
    print(f"Surfaces: {topo['stats']['total_surfaces']}")

    for i, surface in enumerate(topo.get('surfaces', [])[:3]):
        size = surface.get('size', len(surface.get('nodes', [])))
        mass = surface.get('mass', 0)
        print(f"\nSurface {i}: {size} nodes, mass={mass:.2f}")
        # Handle different surface formats
        node_indices = surface.get('node_indices', surface.get('nodes', []))
        for node_idx in node_indices[:3]:
            if isinstance(node_idx, dict):
                print(f"  - {node_idx.get('text', '')[:60]}...")
            elif node_idx < len(topo['nodes']):
                node = topo['nodes'][node_idx]
                print(f"  - {node['text'][:60]}...")

    # Cleanup
    await db_pool.close()
    await neo4j.close()

    print("\n✓ Test complete")


if __name__ == '__main__':
    asyncio.run(main())
