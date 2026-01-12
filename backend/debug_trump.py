import asyncio
from services.neo4j_service import Neo4jService
from itertools import combinations
from collections import defaultdict

async def analyze_trump_case():
    neo4j = Neo4jService()
    await neo4j.connect()

    # Get all Trump events and their entities
    result = await neo4j._execute_read("""
        MATCH (e:Event)-[:INCLUDES]->(s:Surface)-[:HAS_ENTITY]->(trump:Entity)
        WHERE trump.canonical_name = "Donald Trump"
        WITH e
        MATCH (e)-[:INCLUDES]->(s2:Surface)-[:HAS_ENTITY]->(ent:Entity)
        WITH e, collect(DISTINCT ent.canonical_name) as entities
        RETURN e.id as event_id, e.claim_count as claims, entities
        ORDER BY e.claim_count DESC
    """)

    print("TRUMP EVENTS - looking for binding motifs:")
    print("=" * 70)

    all_entity_sets = []
    for row in result:
        entities = set(row["entities"])
        all_entity_sets.append((row["event_id"], entities, row["claims"]))
        print(f"\n{row['event_id']} ({row['claims']} claims):")
        ent_list = list(entities)
        print(f"  Entities: {ent_list[:8]}")

    # Find k>=2 motifs that appear in multiple events
    print("\n" + "=" * 70)
    print("CROSS-EVENT MOTIFS (k>=2 entity sets appearing in 2+ events):")
    print("=" * 70)

    motif_events = defaultdict(set)
    for event_id, entities, _ in all_entity_sets:
        entity_list = list(entities)
        for k in range(2, min(len(entity_list)+1, 5)):
            for subset in combinations(entity_list, k):
                motif = frozenset(subset)
                motif_events[motif].add(event_id)

    # Filter to motifs in 2+ events
    recurring_motifs = {m: evts for m, evts in motif_events.items() if len(evts) >= 2}

    # Sort by support, skip Trump-only
    print("\nMotifs WITHOUT Trump in them:")
    count = 0
    for motif, events in sorted(recurring_motifs.items(), key=lambda x: -len(x[1])):
        if "Donald Trump" not in motif:
            print(f"  {set(motif)}: {len(events)} events - {events}")
            count += 1
            if count >= 10:
                break

    print("\nMotifs WITH Trump (these are expected):")
    count = 0
    for motif, events in sorted(recurring_motifs.items(), key=lambda x: -len(x[1])):
        if "Donald Trump" in motif and len(motif) >= 2:
            other = set(motif) - {"Donald Trump"}
            print(f"  Trump + {other}: {len(events)} events")
            count += 1
            if count >= 10:
                break

    await neo4j.close()

asyncio.run(analyze_trump_case())
