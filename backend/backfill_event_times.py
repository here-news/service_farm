#!/usr/bin/env python3
"""
Backfill earliest_time and latest_time for events from their claims
"""
import asyncio
import asyncpg
import os
import json
from datetime import datetime

async def backfill_event_times():
    """Recalculate event times from claims"""
    from services.neo4j_service import Neo4jService

    # Connect to PostgreSQL
    pool = await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
        database=os.getenv("POSTGRES_DB", "herenews"),
        min_size=1,
        max_size=2
    )

    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    try:
        # Get all events from Neo4j
        events = await neo4j._execute_read("""
            MATCH (e:Event)
            RETURN e.id as id, e.canonical_name as name, e.metadata_json as metadata_json
        """)

        print(f"Found {len(events)} events")

        updated = 0
        for event in events:
            try:
                event_id = event['id']
                event_name = event['name']

                # Parse metadata to get claim_ids
                metadata_json = event.get('metadata_json', '{}')
                metadata = json.loads(metadata_json) if metadata_json else {}
                claim_ids = metadata.get('claim_ids', [])

                if not claim_ids:
                    print(f"⏭️  Skipping {event_name} (no claims)")
                    continue

                # Get claim event_times from PostgreSQL
                async with pool.acquire() as conn:
                    claim_times = await conn.fetch("""
                        SELECT event_time
                        FROM core.claims
                        WHERE id = ANY($1::uuid[])
                            AND event_time IS NOT NULL
                        ORDER BY event_time
                    """, claim_ids)

                if not claim_times:
                    print(f"⏭️  Skipping {event_name} (no claims with event_time)")
                    continue

                # Calculate bounds
                event_times = [row['event_time'] for row in claim_times]
                earliest_time = min(event_times)
                latest_time = max(event_times)

                # Update Neo4j
                await neo4j._execute_write("""
                    MATCH (e:Event {id: $event_id})
                    SET e.earliest_time = $earliest_time,
                        e.latest_time = $latest_time,
                        e.updated_at = datetime()
                """, {
                    'event_id': event_id,
                    'earliest_time': earliest_time,
                    'latest_time': latest_time
                })

                updated += 1
                print(f"✅ Updated {event_name}: {earliest_time} to {latest_time}")

            except Exception as e:
                print(f"❌ Failed to update event {event.get('name', event.get('id'))}: {e}")
                continue

        print(f"\n✨ Updated {updated}/{len(events)} events")

    finally:
        await neo4j.close()
        await pool.close()

if __name__ == "__main__":
    asyncio.run(backfill_event_times())
