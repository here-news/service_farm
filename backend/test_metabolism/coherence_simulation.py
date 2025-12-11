"""
Coherence Simulation

Simulate coherence before and after adding new claims from a specific source.
This helps understand how new articles affect event coherence.

Usage:
    python coherence_simulation.py <event_id> <source_domain>
    python coherence_simulation.py ev_pth3a8dc abc.net.au
"""
import asyncio
import sys
from collections import defaultdict
from typing import List, Dict, Set, Tuple

sys.path.insert(0, '/app')
from services.neo4j_service import Neo4jService


class CoherenceCalculator:
    """Calculate coherence using same logic as live_event.py"""

    def __init__(self, claims: List[dict]):
        self.claims = claims
        self.claims_entities = {}  # claim_id -> set of entity_ids

        for claim in claims:
            self.claims_entities[claim['claim_id']] = set(claim['entity_ids'] or [])

    def calculate(self) -> Tuple[float, dict]:
        """Calculate coherence and return components"""
        if not self.claims:
            return 0.5, {'hub_coverage': 0, 'connectivity': 0, 'components': 0}

        hub_coverage, hub_details = self._calculate_hub_coverage()
        connectivity, conn_details = self._calculate_graph_connectivity()

        coherence = 0.6 * hub_coverage + 0.4 * connectivity

        return coherence, {
            'hub_coverage': hub_coverage,
            'connectivity': connectivity,
            'hub_entities': hub_details['hub_entities'],
            'claims_touching_hubs': hub_details['claims_touching_hubs'],
            'components': conn_details['components'],
            'entity_counts': hub_details['entity_counts']
        }

    def _calculate_hub_coverage(self) -> Tuple[float, dict]:
        """Calculate % of claims touching hub entities (3+ mentions)"""
        entity_mention_counts = defaultdict(int)

        for entities in self.claims_entities.values():
            for entity_id in entities:
                entity_mention_counts[entity_id] += 1

        hub_entities = {eid for eid, count in entity_mention_counts.items() if count >= 3}

        claims_touching_hubs = 0
        for entities in self.claims_entities.values():
            if any(eid in hub_entities for eid in entities):
                claims_touching_hubs += 1

        hub_coverage = claims_touching_hubs / len(self.claims) if self.claims else 0

        return hub_coverage, {
            'hub_entities': len(hub_entities),
            'claims_touching_hubs': claims_touching_hubs,
            'entity_counts': dict(entity_mention_counts)
        }

    def _calculate_graph_connectivity(self) -> Tuple[float, dict]:
        """Calculate 1/num_components using union-find"""
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

        # Build union-find
        for claim_id, entities in self.claims_entities.items():
            for entity_id in entities:
                union(f"claim:{claim_id}", f"entity:{entity_id}")

        # Count components
        roots = set()
        for key in parent.keys():
            roots.add(find(key))

        num_components = len(roots) if roots else 1
        connectivity = 1.0 / num_components

        return connectivity, {'components': num_components}


async def simulate_coherence(event_id: str, test_domain: str):
    """Simulate coherence before and after adding claims from test_domain"""

    neo4j = Neo4jService()
    await neo4j.connect()

    print("=" * 80)
    print(f"📊 Coherence Simulation")
    print(f"   Event: {event_id}")
    print(f"   Test domain: {test_domain}")
    print("=" * 80)

    # Get event info
    event = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})
        RETURN e.canonical_name as name, e.coherence as coherence
    """, {'event_id': event_id})

    if not event:
        print(f"❌ Event {event_id} not found")
        return

    print(f"\n🎯 Event: {event[0]['name']}")
    print(f"   Stored coherence: {event[0]['coherence']}")

    # Get all claims with their entities and source domain
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
        await neo4j.close()
        return

    # Split claims: before (without test_domain) and all (with test_domain)
    claims_before = [c for c in claims_data if c['source_domain'] != test_domain]
    claims_from_test = [c for c in claims_data if c['source_domain'] == test_domain]

    print(f"\n📋 Claim Distribution:")
    print(f"   Total claims: {len(claims_data)}")
    print(f"   Claims from {test_domain}: {len(claims_from_test)}")
    print(f"   Claims from other sources: {len(claims_before)}")

    # Calculate BEFORE coherence (without test domain)
    print(f"\n{'='*80}")
    print(f"📊 BEFORE (without {test_domain})")
    print(f"{'='*80}")

    calc_before = CoherenceCalculator(claims_before)
    coherence_before, details_before = calc_before.calculate()

    print(f"   Claims: {len(claims_before)}")
    print(f"   Hub entities (3+ mentions): {details_before['hub_entities']}")
    print(f"   Claims touching hubs: {details_before['claims_touching_hubs']}/{len(claims_before)}")
    print(f"   Hub coverage: {details_before['hub_coverage']:.3f}")
    print(f"   Connected components: {details_before['components']}")
    print(f"   Connectivity: {details_before['connectivity']:.3f}")
    print(f"   📊 COHERENCE: {coherence_before:.3f}")

    # Show top entities before
    sorted_entities = sorted(details_before['entity_counts'].items(), key=lambda x: -x[1])[:5]
    if sorted_entities:
        print(f"\n   Top entities:")
        for eid, count in sorted_entities:
            ent_info = await neo4j._execute_read(
                "MATCH (e:Entity {id: $eid}) RETURN e.canonical_name as name",
                {'eid': eid}
            )
            name = ent_info[0]['name'] if ent_info else eid[:12]
            hub = "🌟" if count >= 3 else ""
            print(f"      {count}x - {name} {hub}")

    # Calculate AFTER coherence (with test domain)
    print(f"\n{'='*80}")
    print(f"📊 AFTER (with {test_domain})")
    print(f"{'='*80}")

    calc_after = CoherenceCalculator(claims_data)
    coherence_after, details_after = calc_after.calculate()

    print(f"   Claims: {len(claims_data)}")
    print(f"   Hub entities (3+ mentions): {details_after['hub_entities']}")
    print(f"   Claims touching hubs: {details_after['claims_touching_hubs']}/{len(claims_data)}")
    print(f"   Hub coverage: {details_after['hub_coverage']:.3f}")
    print(f"   Connected components: {details_after['components']}")
    print(f"   Connectivity: {details_after['connectivity']:.3f}")
    print(f"   📊 COHERENCE: {coherence_after:.3f}")

    # Show top entities after
    sorted_entities = sorted(details_after['entity_counts'].items(), key=lambda x: -x[1])[:5]
    if sorted_entities:
        print(f"\n   Top entities:")
        for eid, count in sorted_entities:
            ent_info = await neo4j._execute_read(
                "MATCH (e:Entity {id: $eid}) RETURN e.canonical_name as name",
                {'eid': eid}
            )
            name = ent_info[0]['name'] if ent_info else eid[:12]
            hub = "🌟" if count >= 3 else ""
            print(f"      {count}x - {name} {hub}")

    # Show claims from test domain
    print(f"\n{'='*80}")
    print(f"📰 Claims from {test_domain}")
    print(f"{'='*80}")

    for c in claims_from_test:
        entities = ', '.join(c['entity_names'][:3]) if c['entity_names'] else '⚠️ NO ENTITIES'
        print(f"\n   • {c['claim_text']}")
        print(f"     Entities: {entities}")

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")

    delta = coherence_after - coherence_before
    direction = "📈 INCREASED" if delta > 0 else "📉 DECREASED" if delta < 0 else "➡️ UNCHANGED"

    print(f"   Before: {coherence_before:.3f}")
    print(f"   After:  {coherence_after:.3f}")
    print(f"   Delta:  {delta:+.3f} {direction}")

    # Analyze why
    print(f"\n   Analysis:")

    hub_delta = details_after['hub_coverage'] - details_before['hub_coverage']
    conn_delta = details_after['connectivity'] - details_before['connectivity']

    print(f"   • Hub coverage: {details_before['hub_coverage']:.3f} → {details_after['hub_coverage']:.3f} ({hub_delta:+.3f})")
    print(f"   • Connectivity: {details_before['connectivity']:.3f} → {details_after['connectivity']:.3f} ({conn_delta:+.3f})")

    # Check for orphan claims in test domain
    orphans = [c for c in claims_from_test if not c['entity_ids']]
    if orphans:
        print(f"\n   ⚠️  {len(orphans)} claims from {test_domain} have NO ENTITIES:")
        for c in orphans[:3]:
            print(f"      • {c['claim_text'][:60]}...")

    # Check if test claims touch hubs
    hub_entities_after = {eid for eid, count in details_after['entity_counts'].items() if count >= 3}
    test_touching_hubs = 0
    for c in claims_from_test:
        if any(eid in hub_entities_after for eid in (c['entity_ids'] or [])):
            test_touching_hubs += 1

    print(f"\n   📊 {test_domain} integration:")
    print(f"      {test_touching_hubs}/{len(claims_from_test)} claims touch hub entities")

    await neo4j.close()
    print("\n" + "=" * 80)


async def main():
    event_id = sys.argv[1] if len(sys.argv) > 1 else "ev_pth3a8dc"
    test_domain = sys.argv[2] if len(sys.argv) > 2 else "www.abc.net.au"
    await simulate_coherence(event_id, test_domain)


if __name__ == "__main__":
    asyncio.run(main())
