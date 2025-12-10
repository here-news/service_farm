"""
Calculate coherence for existing Hong Kong fire event
"""
import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService
from services.event_service import EventService
from repositories.event_repository import EventRepository
from repositories.claim_repository import ClaimRepository
from repositories.entity_repository import EntityRepository
from models.domain.live_event import LiveEvent


async def main():
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    neo4j = Neo4jService()
    await neo4j.connect()

    # Create repositories (db_pool first, then neo4j)
    claim_repo = ClaimRepository(db_pool, neo4j)
    entity_repo = EntityRepository(db_pool, neo4j)
    event_repo = EventRepository(db_pool, neo4j)
    event_service = EventService(event_repo, claim_repo, entity_repo)

    print("=" * 80)
    print("🧮 CALCULATING COHERENCE FOR EXISTING EVENT")
    print("=" * 80)
    print()

    # Get the Wang Fuk Court fire event
    event_data = await neo4j._execute_read("""
        MATCH (e:Event {id: 'ev_4uvbwao6'})
        RETURN e.id, e.canonical_name, e.coherence
    """, {})

    if not event_data:
        print("❌ Event not found")
        await db_pool.close()
        await neo4j.close()
        return

    event_id = event_data[0]['e.id']
    event_name = event_data[0]['e.canonical_name']
    old_coherence = event_data[0]['e.coherence']

    print(f"Event: {event_name}")
    print(f"ID: {event_id}")
    print(f"Old coherence: {old_coherence:.3f}" if old_coherence else "Old coherence: Not set")
    print()

    # Get full event object
    event = await event_repo.get_by_id(event_id)
    if not event:
        print("❌ Could not load event")
        await db_pool.close()
        await neo4j.close()
        return

    # Create LiveEvent and hydrate
    print("💧 Hydrating LiveEvent...")
    live_event = LiveEvent(event, event_service)
    await live_event.hydrate(claim_repo)

    print(f"   Claims: {len(live_event.claims)}")
    print(f"   Entities: {len(live_event.entity_ids)}")
    print()

    # Calculate coherence
    print("🧮 Calculating coherence components...")
    hub_coverage = await live_event._calculate_hub_coverage()
    graph_connectivity = await live_event._calculate_graph_connectivity()
    new_coherence = await live_event._calculate_coherence()

    print()
    print("=" * 80)
    print("📊 COHERENCE BREAKDOWN")
    print("=" * 80)
    print(f"Hub coverage:       {hub_coverage:.3f} (60% weight)")
    print(f"Graph connectivity: {graph_connectivity:.3f} (40% weight)")
    print(f"Overall coherence:  {new_coherence:.3f}")
    print()

    if old_coherence:
        delta = new_coherence - old_coherence
        print(f"Change: {old_coherence:.3f} → {new_coherence:.3f} (Δ {delta:+.3f})")
        if abs(delta) > 0.1:
            print("✨ SIGNIFICANT CHANGE - Would trigger narrative regeneration!")
    print()

    # Update in Neo4j
    print("💾 Updating coherence in Neo4j...")
    await event_repo.update_coherence(event_id, new_coherence)
    print("   ✅ Updated")
    print()

    # Get entity mention counts to show hubs
    entity_mentions = {}
    for claim in live_event.claims:
        for entity_id in claim.entity_ids:
            entity_mentions[entity_id] = entity_mentions.get(entity_id, 0) + 1

    hub_entities = {eid: count for eid, count in entity_mentions.items() if count >= 3}

    if hub_entities:
        print("🎯 Hub entities (3+ mentions):")
        # Get entity names
        for entity_id in sorted(hub_entities.keys(), key=lambda x: -hub_entities[x])[:10]:
            entity_data = await neo4j._execute_read("""
                MATCH (e:Entity {id: $entity_id})
                RETURN e.name
            """, {'entity_id': entity_id})
            entity_name = entity_data[0]['e.name'] if entity_data else entity_id
            print(f"   {entity_name}: {hub_entities[entity_id]} mentions")
    else:
        print("⚠️  No hub entities found")

    await db_pool.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
