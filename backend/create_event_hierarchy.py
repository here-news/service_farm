"""
Create proper event hierarchy (micro → meso → macro) from existing flat event

Groups claims into micro events, creates relationships
"""
import asyncio
import asyncpg
import os
import json
import uuid
from datetime import datetime
from collections import defaultdict

DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'herenews'),
    'user': os.getenv('POSTGRES_USER', 'herenews_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
}


async def create_event_hierarchy():
    """Create micro/meso/macro event hierarchy from flat event"""

    conn = await asyncpg.connect(**DB_CONFIG)

    try:
        # Get the flat event
        flat_event = await conn.fetchrow("""
            SELECT id, title, event_start, event_end
            FROM core.events
            WHERE id = '0c6bc931-94ea-4e14-9fc9-5c5ed3ebeb2a'
        """)

        if not flat_event:
            print("❌ Event not found")
            return

        print(f"📊 Analyzing event: {flat_event['title']}")

        # Fetch all claims
        claims = await conn.fetch("""
            SELECT c.id, c.text, c.event_time, c.confidence,
                   ARRAY_AGG(DISTINCT ce.entity_id) as entity_ids,
                   ARRAY_AGG(DISTINCT e.canonical_name) FILTER (WHERE e.canonical_name IS NOT NULL) as entity_names
            FROM core.claims c
            JOIN core.pages p ON c.page_id = p.id
            JOIN core.page_events pe ON p.id = pe.page_id
            LEFT JOIN core.claim_entities ce ON c.id = ce.claim_id
            LEFT JOIN core.entities e ON ce.entity_id = e.id
            WHERE pe.event_id = $1
            GROUP BY c.id, c.text, c.event_time, c.confidence
            ORDER BY c.event_time NULLS LAST
        """, flat_event['id'])

        print(f"📝 Found {len(claims)} claims")

        # Group claims into micro events by topic
        micro_groups = {
            'fire_outbreak': [],
            'casualties': [],
            'rescue_operations': [],
            'evacuations': [],
            'investigation': []
        }

        keywords = {
            'fire_outbreak': ['fire broke out', 'started', '2:51 p.m.', 'level 5 alarm', 'upgraded'],
            'casualties': ['died', 'death', 'killed', 'firefighter died', 'missing', 'injured', 'hospitalized'],
            'rescue_operations': ['firefighters', 'temperature', 'difficult', 'grappling', 'intense heat', 'fire authorities'],
            'evacuations': ['evacuated', 'temporary housing', 'road closed', 'buses diverted'],
            'investigation': ['arrested', 'manslaughter', 'police', 'investigation']
        }

        for claim in claims:
            text_lower = claim['text'].lower()
            matched = False

            for group_name, group_keywords in keywords.items():
                if any(kw.lower() in text_lower for kw in group_keywords):
                    micro_groups[group_name].append(dict(claim))
                    matched = True
                    break

            if not matched:
                # Default to fire_outbreak for uncategorized
                micro_groups['fire_outbreak'].append(dict(claim))

        # Filter out empty groups
        micro_groups = {k: v for k, v in micro_groups.items() if v}

        print(f"\n📂 Claim groups:")
        for group_name, group_claims in micro_groups.items():
            print(f"  - {group_name}: {len(group_claims)} claims")

        # Create micro events
        micro_events = {}

        for group_name, group_claims in micro_groups.items():
            if not group_claims:
                continue

            # Generate title
            titles = {
                'fire_outbreak': 'Fire outbreak at Wang Fuk Court',
                'casualties': 'Casualties and deaths',
                'rescue_operations': 'Firefighting and rescue operations',
                'evacuations': 'Evacuations and road closures',
                'investigation': 'Police investigation and arrests'
            }

            title = titles.get(group_name, group_name)

            # Temporal bounds
            times = [c['event_time'] for c in group_claims if c['event_time']]
            event_start = min(times) if times else flat_event['event_start']
            event_end = max(times) if times else flat_event['event_end']

            # Collect entities
            all_entities = set()
            for claim in group_claims:
                if claim['entity_ids']:
                    all_entities.update([eid for eid in claim['entity_ids'] if eid])

            # Create micro event
            micro_event_id = await conn.fetchval("""
                INSERT INTO core.events (
                    title, summary, event_start, event_end,
                    confidence, status, event_scale, claims_count, pages_count,
                    enriched_json
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
            """,
                title,
                f"Micro event: {group_name}",
                event_start,
                event_end,
                0.7,  # Micro events have moderate confidence
                'stable',  # Derived from stable claims
                'micro',
                len(group_claims),
                1,  # Micro events are small
                json.dumps({
                    'scale': {'type': 'micro', 'rationale': f'Single aspect of larger event: {group_name}'},
                    'group': group_name
                })
            )

            # Link entities
            for entity_id in all_entities:
                await conn.execute("""
                    INSERT INTO core.event_entities (event_id, entity_id)
                    VALUES ($1, $2)
                    ON CONFLICT DO NOTHING
                """, micro_event_id, entity_id)

            micro_events[group_name] = {
                'id': micro_event_id,
                'title': title,
                'claims_count': len(group_claims)
            }

            print(f"✅ Created micro event: {title} ({len(group_claims)} claims)")

        # Update flat event to meso/macro
        await conn.execute("""
            UPDATE core.events
            SET event_scale = $2,
                enriched_json = jsonb_set(
                    COALESCE(enriched_json, '{}'::jsonb),
                    '{scale,type}',
                    '"meso"'::jsonb
                )
            WHERE id = $1
        """, flat_event['id'], 'meso')

        print(f"\n✅ Updated parent event to meso scale")

        # Create PART_OF relationships
        for group_name, micro_event in micro_events.items():
            await conn.execute("""
                INSERT INTO core.event_relationships (
                    event_id, related_event_id, relationship_type, confidence, metadata
                )
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT DO NOTHING
            """,
                micro_event['id'],
                flat_event['id'],
                'PART_OF',
                0.9,
                json.dumps({'aspect': group_name})
            )

            print(f"🔗 Linked '{micro_event['title']}' → PART_OF → '{flat_event['title']}'")

        # Show tree structure
        print("\n" + "="*80)
        print("📊 EVENT HIERARCHY:")
        print("="*80)

        tree = await conn.fetch("""
            SELECT
                parent.id as parent_id,
                parent.title as parent_title,
                parent.event_scale as parent_scale,
                parent.claims_count as parent_claims,
                child.id as child_id,
                child.title as child_title,
                child.event_scale as child_scale,
                child.claims_count as child_claims,
                rel.relationship_type
            FROM core.events parent
            LEFT JOIN core.event_relationships rel ON parent.id = rel.related_event_id
            LEFT JOIN core.events child ON rel.event_id = child.id
            WHERE parent.id = $1
            ORDER BY child.event_scale, child.title
        """, flat_event['id'])

        print(f"\n{tree[0]['parent_scale'].upper()} EVENT: {tree[0]['parent_title']}")
        print(f"  └─ {tree[0]['parent_claims']} claims total\n")

        for row in tree:
            if row['child_id']:
                print(f"    ├─ [{row['relationship_type']}] {row['child_scale'].upper()}: {row['child_title']}")
                print(f"    │    └─ {row['child_claims']} claims")

        print("\n" + "="*80)

    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(create_event_hierarchy())
