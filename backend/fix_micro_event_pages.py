"""
Fix missing page_events entries for micro events

The micro events created by create_event_hierarchy.py have claims_count > 0
but no page_events entries, causing the API to return 0 claims/pages.

This script adds the missing page_events entries.
"""
import asyncio
import asyncpg
import os
import json

DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'herenews'),
    'user': os.getenv('POSTGRES_USER', 'herenews_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
}


async def fix_micro_event_pages():
    """Add missing page_events entries for micro events"""

    conn = await asyncpg.connect(**DB_CONFIG)

    try:
        # Find micro events with claims_count > 0 but no page_events
        micro_events = await conn.fetch("""
            SELECT e.id, e.title, e.claims_count, e.pages_count
            FROM core.events e
            WHERE e.event_scale = 'micro'
              AND e.claims_count > 0
              AND NOT EXISTS (
                  SELECT 1 FROM core.page_events pe WHERE pe.event_id = e.id
              )
        """)

        print(f"Found {len(micro_events)} micro events with missing page_events")

        for event in micro_events:
            print(f"\n📊 Fixing event: {event['title']} ({event['claims_count']} claims)")

            # Find parent event (via PART_OF relationship)
            parent = await conn.fetchrow("""
                SELECT related_event_id
                FROM core.event_relationships
                WHERE event_id = $1 AND relationship_type = 'PART_OF'
                LIMIT 1
            """, event['id'])

            if not parent:
                print(f"  ⚠️  No parent event found, skipping")
                continue

            parent_id = parent['related_event_id']

            # Get all claims from parent event's pages
            parent_claims = await conn.fetch("""
                SELECT c.id, c.text, c.page_id
                FROM core.claims c
                JOIN core.pages p ON c.page_id = p.id
                JOIN core.page_events pe ON p.id = pe.page_id
                WHERE pe.event_id = $1
            """, parent_id)

            print(f"  Parent event has {len(parent_claims)} total claims")

            # Match claims to this micro event based on event's enriched_json group
            event_full = await conn.fetchrow("""
                SELECT enriched_json FROM core.events WHERE id = $1
            """, event['id'])

            enriched = json.loads(event_full['enriched_json']) if event_full['enriched_json'] else {}
            group_name = enriched.get('group')

            if not group_name:
                print(f"  ⚠️  No group name in enriched_json, skipping")
                continue

            # Keywords from create_event_hierarchy.py
            keywords_map = {
                'fire_outbreak': ['fire broke out', 'started', '2:51 p.m.', 'level 5 alarm', 'upgraded'],
                'casualties': ['died', 'death', 'killed', 'firefighter died', 'missing', 'injured', 'hospitalized'],
                'rescue_operations': ['firefighters', 'temperature', 'difficult', 'grappling', 'intense heat', 'fire authorities'],
                'evacuations': ['evacuated', 'temporary housing', 'road closed', 'buses diverted'],
                'investigation': ['arrested', 'manslaughter', 'police', 'investigation']
            }

            keywords = keywords_map.get(group_name, [])

            # Find matching claims
            matched_page_ids = set()
            matched_claim_count = 0

            for claim in parent_claims:
                text_lower = claim['text'].lower()
                if any(kw.lower() in text_lower for kw in keywords):
                    matched_page_ids.add(claim['page_id'])
                    matched_claim_count += 1

            print(f"  Matched {matched_claim_count} claims from {len(matched_page_ids)} pages")

            # Create page_events entries
            for page_id in matched_page_ids:
                await conn.execute("""
                    INSERT INTO core.page_events (page_id, event_id)
                    VALUES ($1, $2)
                    ON CONFLICT DO NOTHING
                """, page_id, event['id'])

            # Update pages_count
            await conn.execute("""
                UPDATE core.events
                SET pages_count = $2
                WHERE id = $1
            """, event['id'], len(matched_page_ids))

            print(f"  ✅ Added {len(matched_page_ids)} page_events entries")

        print(f"\n✅ Fixed {len(micro_events)} micro events")

    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(fix_micro_event_pages())
