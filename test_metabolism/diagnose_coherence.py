"""
Diagnose Coherence Drop

Analyze why coherence dropped from 0.500 → 0.380 after adding ABC News article.
Uses same coherence calculation as live_event.py
"""
import asyncio
import sys
from collections import defaultdict

sys.path.insert(0, '/app')
from services.neo4j_service import Neo4jService


async def diagnose_event_coherence(event_id: str):
    """Analyze coherence components for an event"""

    neo4j = Neo4jService()
    await neo4j.connect()

    print("=" * 80)
    print(f"📊 Coherence Diagnosis for Event: {event_id}")
    print("=" * 80)

    # Get event info
    event = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})
        RETURN e.canonical_name as name, e.coherence as coherence, e.claims_count as claims_count
    """, {'event_id': event_id})

    if not event:
        print(f"❌ Event {event_id} not found")
        return

    print(f"\n🎯 Event: {event[0]['name']}")
    print(f"   Current coherence: {event[0]['coherence']}")

    # Get all claims with their entities
    claims_data = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        OPTIONAL MATCH (c)-[:MENTIONS]->(ent:Entity)
        OPTIONAL MATCH (p:Page)-[:CONTAINS]->(c)
        RETURN c.id as claim_id,
               left(c.text, 80) as claim_text,
               c.event_time as event_time,
               collect(DISTINCT ent.id) as entity_ids,
               collect(DISTINCT ent.canonical_name) as entity_names,
               p.domain as source_domain
        ORDER BY c.event_time
    """, {'event_id': event_id})

    if not claims_data:
        print("❌ No claims found")
        return

    print(f"\n📋 Total claims: {len(claims_data)}")

    # Build entity mention counts (like live_event.py)
    entity_mention_counts = defaultdict(int)
    claims_entities = {}  # claim_id -> set of entity_ids

    for claim in claims_data:
        claim_id = claim['claim_id']
        entity_ids = claim['entity_ids'] or []
        claims_entities[claim_id] = set(entity_ids)

        for entity_id in entity_ids:
            entity_mention_counts[entity_id] += 1

    # Identify hub entities (3+ mentions)
    hub_entities = {eid for eid, count in entity_mention_counts.items() if count >= 3}

    print(f"\n🌐 Entity Analysis:")
    print(f"   Total unique entities: {len(entity_mention_counts)}")
    print(f"   Hub entities (3+ mentions): {len(hub_entities)}")

    # Show top entities
    sorted_entities = sorted(entity_mention_counts.items(), key=lambda x: -x[1])[:10]
    print(f"\n   Top 10 entities by mention count:")
    for eid, count in sorted_entities:
        # Get entity name
        ent_info = await neo4j._execute_read("""
            MATCH (e:Entity {id: $eid})
            RETURN e.canonical_name as name
        """, {'eid': eid})
        name = ent_info[0]['name'] if ent_info else eid
        hub_marker = "🌟 HUB" if eid in hub_entities else ""
        print(f"      {count}x - {name} {hub_marker}")

    # Calculate hub coverage
    claims_touching_hubs = 0
    for claim_id, entities in claims_entities.items():
        if any(eid in hub_entities for eid in entities):
            claims_touching_hubs += 1

    hub_coverage = claims_touching_hubs / len(claims_data) if claims_data else 0
    print(f"\n🎯 Hub Coverage: {claims_touching_hubs}/{len(claims_data)} = {hub_coverage:.3f}")

    # Calculate graph connectivity via union-find
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

    # Build union-find from claims and entities
    for claim_id, entities in claims_entities.items():
        for entity_id in entities:
            union(f"claim:{claim_id}", f"entity:{entity_id}")

    # Count connected components
    roots = set()
    for key in parent.keys():
        roots.add(find(key))

    num_components = len(roots)
    graph_connectivity = 1.0 / num_components if num_components > 0 else 0.0

    print(f"\n🔗 Graph Connectivity:")
    print(f"   Connected components: {num_components}")
    print(f"   Connectivity score: {graph_connectivity:.3f}")

    # Calculate coherence
    coherence = 0.6 * hub_coverage + 0.4 * graph_connectivity
    print(f"\n📊 COHERENCE CALCULATION:")
    print(f"   = 0.6 × hub_coverage + 0.4 × graph_connectivity")
    print(f"   = 0.6 × {hub_coverage:.3f} + 0.4 × {graph_connectivity:.3f}")
    print(f"   = {0.6 * hub_coverage:.3f} + {0.4 * graph_connectivity:.3f}")
    print(f"   = {coherence:.3f}")

    # Show claims by source domain
    print(f"\n📰 Claims by Source:")
    claims_by_domain = defaultdict(list)
    for claim in claims_data:
        domain = claim['source_domain'] or 'unknown'
        claims_by_domain[domain].append(claim)

    for domain, domain_claims in claims_by_domain.items():
        print(f"\n   {domain}: {len(domain_claims)} claims")
        for c in domain_claims[:3]:  # Show first 3
            text = c['claim_text']
            entities = ', '.join(c['entity_names'][:3]) if c['entity_names'] else 'none'
            print(f"      • {text}...")
            print(f"        entities: {entities}")

    # Identify problem: claims without entities
    orphan_claims = [c for c in claims_data if not c['entity_ids']]
    if orphan_claims:
        print(f"\n⚠️  ORPHAN CLAIMS (no entities): {len(orphan_claims)}")
        for c in orphan_claims[:5]:
            print(f"      • {c['claim_text']}...")

    # Identify claims not touching hubs
    isolated_claims = []
    for claim in claims_data:
        claim_id = claim['claim_id']
        entities = claims_entities.get(claim_id, set())
        if not any(eid in hub_entities for eid in entities):
            isolated_claims.append(claim)

    if isolated_claims:
        print(f"\n⚠️  ISOLATED CLAIMS (don't touch hubs): {len(isolated_claims)}")
        for c in isolated_claims[:5]:
            entities = ', '.join(c['entity_names'][:3]) if c['entity_names'] else 'none'
            print(f"      • {c['claim_text'][:60]}...")
            print(f"        entities: {entities}")

    await neo4j.close()
    print("\n" + "=" * 80)


async def main():
    # Default to Wang Fuk Court Fire event
    event_id = sys.argv[1] if len(sys.argv) > 1 else "ev_pth3a8dc"
    await diagnose_event_coherence(event_id)


if __name__ == "__main__":
    asyncio.run(main())
