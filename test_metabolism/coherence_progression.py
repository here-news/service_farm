"""
Coherence Progression

Show how coherence changed as each source was added to the event.
"""
import asyncio
import sys
from collections import defaultdict
from typing import List, Tuple

sys.path.insert(0, '/app')
from services.neo4j_service import Neo4jService


def calculate_coherence(claims: List[dict]) -> Tuple[float, dict]:
    """Calculate coherence and components"""
    if not claims:
        return 0.5, {'hub_coverage': 0, 'connectivity': 0, 'components': 0, 'hubs': 0}

    # Build entity counts
    claims_entities = {}
    entity_counts = defaultdict(int)

    for claim in claims:
        claim_id = claim['claim_id']
        entities = set(claim['entity_ids'] or [])
        claims_entities[claim_id] = entities
        for eid in entities:
            entity_counts[eid] += 1

    # Hub coverage
    hub_entities = {eid for eid, count in entity_counts.items() if count >= 3}
    claims_touching_hubs = sum(1 for entities in claims_entities.values()
                                if any(eid in hub_entities for eid in entities))
    hub_coverage = claims_touching_hubs / len(claims) if claims else 0

    # Graph connectivity via union-find
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for claim_id, entities in claims_entities.items():
        for entity_id in entities:
            union(f"claim:{claim_id}", f"entity:{entity_id}")

    roots = {find(key) for key in parent.keys()}
    num_components = len(roots) if roots else 1
    connectivity = 1.0 / num_components

    coherence = 0.6 * hub_coverage + 0.4 * connectivity

    return coherence, {
        'hub_coverage': hub_coverage,
        'connectivity': connectivity,
        'components': num_components,
        'hubs': len(hub_entities),
        'claims_touching_hubs': claims_touching_hubs
    }


async def show_progression(event_id: str):
    """Show coherence as each source was added"""

    neo4j = Neo4jService()
    await neo4j.connect()

    print("=" * 80)
    print(f"📊 Coherence Progression for Event: {event_id}")
    print("=" * 80)

    # Get event info
    event = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})
        RETURN e.canonical_name as name, e.coherence as coherence
    """, {'event_id': event_id})

    if not event:
        print(f"❌ Event not found")
        return

    print(f"\n🎯 Event: {event[0]['name']}")
    print(f"   Current stored coherence: {event[0]['coherence']}")

    # Get all claims with source info
    claims_data = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        OPTIONAL MATCH (c)-[:MENTIONS]->(ent:Entity)
        OPTIONAL MATCH (p:Page)-[:CONTAINS]->(c)
        WITH c, p, collect(DISTINCT ent.id) as entity_ids
        RETURN c.id as claim_id,
               c.text as claim_text,
               entity_ids,
               p.domain as source_domain,
               p.created_at as page_created
        ORDER BY p.created_at, c.id
    """, {'event_id': event_id})

    # Group by source domain in order of appearance
    sources_order = []
    claims_by_source = defaultdict(list)

    for claim in claims_data:
        domain = claim['source_domain'] or 'unknown'
        if domain not in sources_order:
            sources_order.append(domain)
        claims_by_source[domain].append(claim)

    print(f"\n📰 Sources (in order of addition):")
    for i, source in enumerate(sources_order, 1):
        print(f"   {i}. {source}: {len(claims_by_source[source])} claims")

    # Calculate coherence progressively
    print(f"\n{'='*80}")
    print(f"📈 Coherence Progression")
    print(f"{'='*80}")
    print(f"\n{'Source':<35} {'Claims':>8} {'Hubs':>6} {'Hub%':>8} {'Comp':>6} {'Conn':>8} {'Coherence':>10}")
    print("-" * 90)

    cumulative_claims = []

    for source in sources_order:
        # Add claims from this source
        cumulative_claims.extend(claims_by_source[source])

        # Calculate coherence
        coherence, details = calculate_coherence(cumulative_claims)

        print(f"{source:<35} {len(cumulative_claims):>8} {details['hubs']:>6} "
              f"{details['hub_coverage']:>7.1%} {details['components']:>6} "
              f"{details['connectivity']:>7.3f} {coherence:>10.3f}")

    # Show the problem: what's causing fragmentation?
    print(f"\n{'='*80}")
    print(f"🔍 Entity Analysis")
    print(f"{'='*80}")

    # Get entity names
    entity_counts = defaultdict(int)
    entity_names = {}

    for claim in claims_data:
        for eid in (claim['entity_ids'] or []):
            entity_counts[eid] += 1

    # Fetch names for top entities
    top_entities = sorted(entity_counts.items(), key=lambda x: -x[1])[:10]
    for eid, count in top_entities:
        result = await neo4j._execute_read(
            "MATCH (e:Entity {id: $eid}) RETURN e.canonical_name as name",
            {'eid': eid}
        )
        name = result[0]['name'] if result else eid[:12]
        entity_names[eid] = name

    print(f"\nTop entities by mention count:")
    for eid, count in top_entities:
        hub = "🌟 HUB" if count >= 3 else ""
        print(f"   {count:>3}x - {entity_names.get(eid, eid)} {hub}")

    # Analyze orphan claims
    orphans = [c for c in claims_data if not c['entity_ids']]
    if orphans:
        print(f"\n⚠️  ORPHAN CLAIMS (no entities): {len(orphans)}")
        by_source = defaultdict(int)
        for c in orphans:
            by_source[c['source_domain']] += 1
        for source, count in by_source.items():
            print(f"   {source}: {count} orphan claims")

    # Analyze why 5 components
    print(f"\n🔗 Component Analysis:")
    print(f"   The graph has 5 connected components because claims/entities")
    print(f"   don't all connect through shared entities.")
    print(f"\n   To improve coherence:")
    print(f"   1. Ensure claims extract entities that connect to hubs")
    print(f"   2. Link related entities (e.g., Tai Po → Hong Kong)")
    print(f"   3. Fix orphan claims that have no entities")

    await neo4j.close()
    print("\n" + "=" * 80)


async def main():
    event_id = sys.argv[1] if len(sys.argv) > 1 else "ev_pth3a8dc"
    await show_progression(event_id)


if __name__ == "__main__":
    asyncio.run(main())
