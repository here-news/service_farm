"""Manually trigger narrative regeneration for event"""
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

    # Create repositories
    claim_repo = ClaimRepository(db_pool, neo4j)
    entity_repo = EntityRepository(db_pool, neo4j)
    event_repo = EventRepository(db_pool, neo4j)
    event_service = EventService(event_repo, claim_repo, entity_repo)

    print("=" * 80)
    print("📝 REGENERATING EVENT NARRATIVE")
    print("=" * 80)
    print()

    # Get the event
    event = await event_repo.get_by_id('ev_4uvbwao6')
    if not event:
        print("❌ Event not found")
        await db_pool.close()
        await neo4j.close()
        return

    # Create LiveEvent and hydrate
    print(f"Event: {event.canonical_name}")
    print(f"Coherence: {event.coherence:.3f}" if event.coherence else "Coherence: Not set")
    print()

    print("💧 Hydrating LiveEvent...")
    live_event = LiveEvent(event, event_service)
    await live_event.hydrate(claim_repo)

    print(f"   Claims: {len(live_event.claims)}")
    print(f"   Entities: {len(live_event.entity_ids)}")
    print()

    # Show claims being used
    print("📋 Sample claims (showing first 10):")
    for i, claim in enumerate(live_event.claims[:10], 1):
        print(f"   {i}. {claim.text[:80]}...")
    print()

    # Regenerate narrative
    print("🔄 Calling _generate_event_narrative()...")
    print()

    old_narrative = event.metadata.get('summary', 'No previous narrative') if event.metadata else 'No previous narrative'

    new_narrative = await event_service._generate_event_narrative(event, live_event.claims)

    print("=" * 80)
    print("OLD NARRATIVE:")
    print("=" * 80)
    print(old_narrative)
    print()
    print("=" * 80)
    print("NEW NARRATIVE:")
    print("=" * 80)
    print(new_narrative)
    print()
    print("=" * 80)

    print(f"\nOld length: {len(old_narrative)} chars")
    print(f"New length: {len(new_narrative)} chars")
    print()

    # Update in storage
    print("💾 Updating narrative in Neo4j...")
    await event_repo.update_narrative(event.id, new_narrative)
    print("   ✅ Updated")

    await db_pool.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
