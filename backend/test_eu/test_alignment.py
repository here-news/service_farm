"""
Alignment Check: Verify Tier 1 and Tier 2 relationships are consistent.

Tests:
1. Kernel (Tier 1): Check that CONFIRMS/REFINES claims are semantically similar
2. Weaver (Tier 2): Check that same-event claims share entities
3. Cross-tier: Verify kernel nodes map to weaver events sensibly

Usage:
    docker exec herenews-app python -m test_eu.test_alignment
"""

import asyncio
import os
import sys
sys.path.insert(0, '/app/backend')

from openai import AsyncOpenAI
import asyncpg

from services.neo4j_service import Neo4jService
from repositories.claim_repository import ClaimRepository
from repositories.event_repository import EventRepository
from test_eu.core.kernel import EpistemicKernel
from test_eu.core.event_weaver import EventWeaver


async def main():
    print("="*70)
    print("ALIGNMENT CHECK: Tier 1 (Kernel) ↔ Tier 2 (Weaver)")
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

    claim_repo = ClaimRepository(db_pool, neo4j)
    event_repo = EventRepository(db_pool, neo4j)
    openai_client = AsyncOpenAI()

    # Get claims
    print("\nLoading claims from Wang Fuk Court Fire...")
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
    # TIER 1: Run Kernel
    # =========================================================================
    print("\n" + "="*70)
    print("TIER 1: KERNEL ANALYSIS")
    print("="*70)

    def format_time(t):
        if t is None: return None
        if hasattr(t, 'strftime'): return t.strftime('%Y-%m-%d')
        return str(t)[:10]

    kernel = EpistemicKernel(
        llm_client=openai_client,
        claim_repository=claim_repo
    )

    # Track claim -> result mapping
    claim_results = {}

    for claim in claims[:50]:  # First 50 for speed
        context_parts = [claim.text]
        if claim.entities:
            context_parts.append(f"Entities: {', '.join(e.canonical_name for e in claim.entities[:3])}")
        if claim.event_time:
            context_parts.append(f"Time: {format_time(claim.event_time)}")
        context = ' | '.join(context_parts)

        source = claim.metadata.get('source_name', 'unknown')
        result = await kernel.process(claim.text, source, context)
        claim_results[claim.id] = {
            'text': claim.text[:80],
            'relation': result.get('relation', result.get('applied_relation')),
            'affected_belief': result.get('affected_belief'),
            'similarity': result.get('similarity', 0),
            'entities': [e.canonical_name for e in (claim.entities or [])]
        }

    # Show kernel results
    print(f"\nKernel processed 50 claims → {len(kernel.topo.nodes)} nodes")

    print("\n--- KERNEL NODES (Facts) ---")
    for i, node in enumerate(kernel.topo.nodes):
        print(f"\n[Node {i}] {node.text[:100]}...")
        print(f"  Sources: {node.source_count}")
        print(f"  Claims: {len(node.claim_ids)}")

    # Check CONFIRMS relationships
    print("\n--- CONFIRMS ANALYSIS ---")
    confirms = [(cid, r) for cid, r in claim_results.items() if r['relation'] == 'confirms']
    print(f"Total CONFIRMS: {len(confirms)}")

    # Group by target node
    by_node = {}
    for cid, r in confirms:
        target = r['affected_belief']
        if target not in by_node:
            by_node[target] = []
        by_node[target].append((cid, r))

    for node_idx, confirmers in by_node.items():
        if node_idx is None:
            continue
        node = kernel.topo.nodes[node_idx]
        print(f"\nNode [{node_idx}]: {node.text[:60]}...")
        print(f"  Confirmed by {len(confirmers)} claims:")
        for cid, r in confirmers[:3]:
            print(f"    - sim={r['similarity']:.2f}: {r['text'][:50]}...")

    # =========================================================================
    # TIER 2: Run Weaver
    # =========================================================================
    print("\n" + "="*70)
    print("TIER 2: WEAVER ANALYSIS")
    print("="*70)

    weaver = EventWeaver()

    for claim in claims[:50]:
        embedding = embeddings.get(claim.id)
        await weaver.weave_claim(claim, embedding)

    merges = weaver.merge_events(min_shared_entities=1)
    print(f"\nWeaver processed 50 claims → {len(weaver.event_candidates)} events (after {merges} merges)")

    print("\n--- EVENT CLUSTERS ---")
    sorted_events = sorted(weaver.event_candidates, key=lambda e: len(e.claim_ids), reverse=True)

    for evt in sorted_events[:5]:
        print(f"\n[{evt.id}] {len(evt.claim_ids)} claims")
        print(f"  Entities: {', '.join(list(evt.entity_names)[:5])}")
        print(f"  Time: {evt.time_start} → {evt.time_end}")

    # =========================================================================
    # CROSS-TIER ALIGNMENT CHECK
    # =========================================================================
    print("\n" + "="*70)
    print("CROSS-TIER ALIGNMENT")
    print("="*70)

    # For each kernel node, check which weaver events its claims belong to
    print("\n--- Kernel Nodes → Weaver Events ---")

    for i, node in enumerate(kernel.topo.nodes):
        print(f"\nKernel Node [{i}]: {node.text[:60]}...")

        # Find which claims confirm this node
        node_claims = set()
        for cid, r in claim_results.items():
            if r['affected_belief'] == i or (r['relation'] == 'novel' and len(kernel.topo.nodes) == i + 1):
                node_claims.add(cid)

        # Find which weaver events these claims belong to
        event_counts = {}
        for cid in node_claims:
            if cid in weaver._claim_to_event:
                evt_id = weaver._claim_to_event[cid]
                event_counts[evt_id] = event_counts.get(evt_id, 0) + 1

        if event_counts:
            print(f"  Maps to weaver events:")
            for evt_id, count in sorted(event_counts.items(), key=lambda x: -x[1]):
                evt = next((e for e in weaver.event_candidates if e.id == evt_id), None)
                if evt:
                    ent_str = ', '.join(list(evt.entity_names)[:3])
                    print(f"    {evt_id}: {count} claims (entities: {ent_str})")

    # =========================================================================
    # ALIGNMENT ISSUES CHECK
    # =========================================================================
    print("\n" + "="*70)
    print("ALIGNMENT ISSUES")
    print("="*70)

    issues = []

    # Check 1: CONFIRMS with low similarity
    low_sim_confirms = [(cid, r) for cid, r in claim_results.items()
                        if r['relation'] == 'confirms' and r['similarity'] < 0.4]
    if low_sim_confirms:
        issues.append(f"⚠ {len(low_sim_confirms)} CONFIRMS with similarity < 0.4")
        for cid, r in low_sim_confirms[:2]:
            print(f"  Low sim CONFIRMS: {r['text'][:50]}... (sim={r['similarity']:.2f})")

    # Check 2: Same kernel node but different weaver events (potential fragmentation)
    for i, node in enumerate(kernel.topo.nodes):
        node_claims = [cid for cid, r in claim_results.items() if r['affected_belief'] == i]
        events_for_node = set()
        for cid in node_claims:
            if cid in weaver._claim_to_event:
                events_for_node.add(weaver._claim_to_event[cid])

        if len(events_for_node) > 1:
            issues.append(f"⚠ Kernel node [{i}] spans {len(events_for_node)} weaver events")

    # Check 3: Same weaver event but different kernel nodes (potential under-compression)
    for evt in weaver.event_candidates:
        if len(evt.claim_ids) < 3:
            continue
        kernel_nodes = set()
        for cid in evt.claim_ids:
            if cid in claim_results:
                affected = claim_results[cid]['affected_belief']
                if affected is not None:
                    kernel_nodes.add(affected)

        if len(kernel_nodes) > 2:
            issues.append(f"⚠ Weaver event {evt.id} has claims from {len(kernel_nodes)} kernel nodes")

    if issues:
        print("\nPotential issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No major alignment issues detected")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
  Tier 1 (Kernel):
    - 50 claims → {len(kernel.topo.nodes)} nodes
    - Relations: {kernel.summary()['relations']}
    - LLM skipped: {kernel.llm_calls_skipped}/{kernel.llm_calls + kernel.llm_calls_skipped}

  Tier 2 (Weaver):
    - 50 claims → {len(weaver.event_candidates)} events
    - Largest event: {max(len(e.claim_ids) for e in weaver.event_candidates)} claims

  Alignment: {'✓ GOOD' if len(issues) == 0 else f'⚠ {len(issues)} issues'}
""")

    await db_pool.close()
    await neo4j.close()


if __name__ == '__main__':
    asyncio.run(main())
