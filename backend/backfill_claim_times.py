#!/usr/bin/env python3
"""
Backfill event_time for claims that have temporal data in metadata but NULL event_time
"""
import asyncio
import asyncpg
import os
import json
from datetime import datetime

async def backfill_claim_times():
    """Backfill event_time from metadata.when.date"""
    pool = await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
        database=os.getenv("POSTGRES_DB", "herenews"),
        min_size=1,
        max_size=2
    )

    async with pool.acquire() as conn:
        # Find claims with NULL event_time but metadata.when.date
        claims = await conn.fetch("""
            SELECT id, text, metadata
            FROM core.claims
            WHERE event_time IS NULL
                AND metadata->'when'->>'date' IS NOT NULL
        """)

        print(f"Found {len(claims)} claims with NULL event_time but temporal metadata")

        updated = 0
        for claim in claims:
            try:
                metadata = claim['metadata']
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                when_data = metadata.get('when', {})
                date_str = when_data.get('date')

                if date_str:
                    # Parse date and set to midnight UTC
                    event_time = datetime.fromisoformat(date_str + 'T00:00:00+00:00')

                    # Update claim
                    await conn.execute("""
                        UPDATE core.claims
                        SET event_time = $1
                        WHERE id = $2
                    """, event_time, claim['id'])

                    updated += 1
                    print(f"✅ Updated claim {claim['id']}: {date_str} → {event_time}")

            except Exception as e:
                print(f"❌ Failed to update claim {claim['id']}: {e}")
                continue

        print(f"\n✨ Updated {updated}/{len(claims)} claims")

    await pool.close()

if __name__ == "__main__":
    asyncio.run(backfill_claim_times())
