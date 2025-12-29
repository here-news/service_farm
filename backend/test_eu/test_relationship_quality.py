"""
Relationship Quality Check: Verify CONFIRMS are truly same-fact.

Usage:
    docker exec herenews-app python -m test_eu.test_relationship_quality
"""

import asyncio
import os
import sys
sys.path.insert(0, '/app/backend')

from openai import AsyncOpenAI
import asyncpg

from services.neo4j_service import Neo4jService
from repositories.event_repository import EventRepository
from test_eu.core.kernel import EpistemicKernel


async def main():
    print("="*70)
    print("RELATIONSHIP QUALITY CHECK")
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
    openai_client = AsyncOpenAI()

    # Get claims
    event_data = await neo4j._execute_read("""
        MATCH (e:Event) WHERE e.canonical_name CONTAINS 'Wang Fuk'
        RETURN e.id as id LIMIT 1
    """, {})
    claims = await event_repo.get_event_claims(event_data[0]['id'])

    print(f"\nProcessing first 30 claims with full tracking...")

    kernel = EpistemicKernel(llm_client=openai_client)

    # Track each claim's processing
    processing_log = []

    def format_time(t):
        if t is None: return None
        if hasattr(t, 'strftime'): return t.strftime('%Y-%m-%d')
        return str(t)[:10]

    for claim in claims[:30]:
        context_parts = [claim.text]
        if claim.entities:
            context_parts.append(f"Entities: {', '.join(e.canonical_name for e in claim.entities[:3])}")
        if claim.event_time:
            context_parts.append(f"Time: {format_time(claim.event_time)}")
        context = ' | '.join(context_parts)

        result = await kernel.process(claim.text, 'test', context)

        processing_log.append({
            'claim_id': claim.id,
            'text': claim.text,
            'entities': [e.canonical_name for e in (claim.entities or [])],
            'relation': result.get('relation', result.get('applied_relation')),
            'affected_belief': result.get('affected_belief'),
            'similarity': result.get('similarity', 0),
            'reasoning': result.get('reasoning', ''),
            'skipped_llm': result.get('skipped_llm', False)
        })

    # Show kernel nodes (the distinct facts)
    print("\n" + "="*70)
    print("KERNEL NODES (Distinct Facts)")
    print("="*70)

    for i, node in enumerate(kernel.topo.nodes):
        print(f"\n[Node {i}] {node.text}")
        print(f"  Sources: {node.source_count}, Claims: {len(node.claim_ids)}")

    # Group claims by relationship
    by_relation = {}
    for log in processing_log:
        rel = log['relation']
        if rel not in by_relation:
            by_relation[rel] = []
        by_relation[rel].append(log)

    print("\n" + "="*70)
    print("RELATIONSHIP BREAKDOWN")
    print("="*70)

    for rel, logs in by_relation.items():
        print(f"\n{rel.upper()}: {len(logs)} claims")

    # Detailed CONFIRMS analysis
    print("\n" + "="*70)
    print("CONFIRMS QUALITY CHECK")
    print("="*70)

    confirms = by_relation.get('confirms', [])
    if confirms:
        # Group by target node
        by_target = {}
        for log in confirms:
            target = log['affected_belief']
            if target not in by_target:
                by_target[target] = []
            by_target[target].append(log)

        for target_idx, logs in by_target.items():
            if target_idx is None:
                continue

            target_node = kernel.topo.nodes[target_idx]
            print(f"\n--- Target Node [{target_idx}] ---")
            print(f"FACT: {target_node.text}")
            print(f"\nConfirmed by {len(logs)} claims:")

            for log in logs[:5]:
                sim = log['similarity']
                skipped = "✓ LLM skipped" if log['skipped_llm'] else "⚡ LLM used"
                print(f"\n  [{sim:.2f}] {skipped}")
                print(f"  CLAIM: {log['text'][:100]}...")
                if log['entities']:
                    print(f"  ENTITIES: {', '.join(log['entities'][:4])}")

            if len(logs) > 5:
                print(f"\n  ... and {len(logs) - 5} more")

    # Check for potential false CONFIRMS (different facts marked as same)
    print("\n" + "="*70)
    print("FALSE CONFIRMS CHECK")
    print("="*70)

    potential_false = []
    for log in confirms:
        # Check if claim mentions different numbers
        target_idx = log['affected_belief']
        if target_idx is None:
            continue

        target_text = kernel.topo.nodes[target_idx].text.lower()
        claim_text = log['text'].lower()

        # Extract numbers
        import re
        target_nums = set(re.findall(r'\b(\d+)\b', target_text))
        claim_nums = set(re.findall(r'\b(\d+)\b', claim_text))

        # If both have numbers but they're different, could be a problem
        if target_nums and claim_nums and target_nums != claim_nums:
            # Check for obvious conflicts (death tolls, etc.)
            death_words = ['dead', 'killed', 'died', 'death', 'fatalities']
            if any(w in target_text for w in death_words) and any(w in claim_text for w in death_words):
                # Both about deaths with different numbers
                potential_false.append({
                    'target': target_text[:80],
                    'target_nums': target_nums,
                    'claim': claim_text[:80],
                    'claim_nums': claim_nums,
                    'similarity': log['similarity']
                })

    if potential_false:
        print(f"\n⚠ Found {len(potential_false)} potential false CONFIRMS (different numbers):")
        for pf in potential_false[:3]:
            print(f"\n  Target nums: {pf['target_nums']}")
            print(f"  Target: {pf['target']}")
            print(f"  Claim nums: {pf['claim_nums']}")
            print(f"  Claim: {pf['claim']}")
            print(f"  Similarity: {pf['similarity']:.2f}")
    else:
        print("\n✓ No obvious false CONFIRMS detected")

    # Show NOVEL claims
    print("\n" + "="*70)
    print("NOVEL CLAIMS (New Facts)")
    print("="*70)

    novels = by_relation.get('novel', [])
    for log in novels:
        print(f"\n  {log['text'][:100]}...")
        if log['entities']:
            print(f"    Entities: {', '.join(log['entities'][:3])}")

    await db_pool.close()
    await neo4j.close()

    print("\n✓ Quality check complete")


if __name__ == '__main__':
    asyncio.run(main())
