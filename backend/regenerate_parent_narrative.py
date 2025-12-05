#!/usr/bin/env python3
"""
Script to manually regenerate parent event narrative with correct date
"""
import asyncio
import sys
import os
import asyncpg

from services.neo4j_service import Neo4jService
from repositories.event_repository import EventRepository
from repositories.claim_repository import ClaimRepository
from repositories.entity_repository import EntityRepository
from services.event_service import EventService
from openai import AsyncOpenAI

async def main():
    event_id = "26f37077-ca7c-4bef-80b8-47d593574ea7"  # Wang Fuk Court Fire

    # Initialize postgres pool
    db_pool = await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
        database=os.getenv("POSTGRES_DB", "herenews"),
        min_size=1,
        max_size=2
    )

    # Initialize services
    neo4j_service = Neo4jService()
    await neo4j_service.connect()
    event_repo = EventRepository(db_pool, neo4j_service)
    claim_repo = ClaimRepository(db_pool, neo4j_service)
    entity_repo = EntityRepository(db_pool, neo4j_service)
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    event_service = EventService(event_repo, claim_repo, entity_repo, openai_client)

    # Get the event
    event = await event_repo.get_by_id(event_id)
    if not event:
        print(f"❌ Event {event_id} not found")
        return

    print(f"📖 Regenerating narrative for: {event.canonical_name}")
    print(f"   Current summary (first 100 chars): {event.summary[:100] if event.summary else 'None'}...")

    # Regenerate parent narrative
    sub_events = await event_repo.get_sub_events(event.id)
    print(f"   Found {len(sub_events)} sub-events")

    await event_service._generate_parent_narrative(event, sub_events)

    # Save the updated event
    await event_repo.update(event)

    print(f"✅ Updated narrative (first 100 chars): {event.summary[:100] if event.summary else 'None'}...")

    # Close connections
    await neo4j_service.close()
    await db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())
