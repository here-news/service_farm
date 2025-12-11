#!/usr/bin/env python3
"""
Backfill wikidata_label for entities that have QID but missing label.

This fetches the authoritative label from Wikidata and updates:
1. wikidata_label - the authoritative Wikidata name
2. canonical_name - updated to match wikidata_label
3. aliases - old canonical_name added as alias
"""
import asyncio
import aiohttp
import os
from neo4j import AsyncGraphDatabase

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'herenews_neo4j_pass')

WIKIDATA_API = "https://www.wikidata.org/w/api.php"


HEADERS = {
    'User-Agent': 'HereNews/1.0 (https://herenews.app; contact@herenews.app) aiohttp/3.x'
}


async def fetch_wikidata_label(session: aiohttp.ClientSession, qid: str) -> str | None:
    """Fetch label for a QID from Wikidata."""
    params = {
        'action': 'wbgetentities',
        'ids': qid,
        'format': 'json',
        'props': 'labels',
        'languages': 'en'
    }
    try:
        async with session.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=10) as resp:
            if resp.status != 200:
                print(f"  ⚠️ {qid}: HTTP {resp.status}")
                return None
            data = await resp.json()
            entity = data.get('entities', {}).get(qid, {})
            labels = entity.get('labels', {})
            if 'en' in labels:
                return labels['en']['value']
    except Exception as e:
        print(f"  ⚠️ Failed to fetch {qid}: {e}")
    return None


async def backfill_labels(dry_run: bool = True):
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    async with aiohttp.ClientSession() as http_session:
        async with driver.session() as db_session:
            # Find entities with QID but no wikidata_label
            result = await db_session.run("""
                MATCH (e:Entity)
                WHERE e.wikidata_qid IS NOT NULL AND e.wikidata_label IS NULL
                RETURN e.id as id, e.canonical_name as name, e.wikidata_qid as qid
            """)
            entities = [r async for r in result]

            print(f"\n📊 Found {len(entities)} entities to backfill\n")

            updated = 0
            for ent in entities:
                qid = ent['qid']
                current_name = ent['name']
                entity_id = ent['id']

                # Fetch label from Wikidata
                label = await fetch_wikidata_label(http_session, qid)

                if not label:
                    print(f"  ⏭️ {current_name} ({qid}) - no English label found")
                    continue

                if label == current_name:
                    # Just set wikidata_label, no name change needed
                    print(f"  ✓ {current_name} ({qid}) - name already correct")
                    if not dry_run:
                        await db_session.run("""
                            MATCH (e:Entity {id: $id})
                            SET e.wikidata_label = $label
                        """, {'id': entity_id, 'label': label})
                    updated += 1
                else:
                    # Update canonical_name to label, add old name as alias
                    print(f"  🔄 {current_name} → {label} ({qid})")
                    if not dry_run:
                        await db_session.run("""
                            MATCH (e:Entity {id: $id})
                            SET e.wikidata_label = $label,
                                e.canonical_name = $label,
                                e.aliases = CASE
                                    WHEN $old_name IN coalesce(e.aliases, []) THEN e.aliases
                                    ELSE coalesce(e.aliases, []) + $old_name
                                END
                        """, {'id': entity_id, 'label': label, 'old_name': current_name})
                    updated += 1

                # Rate limit
                await asyncio.sleep(0.1)

    await driver.close()

    mode = "DRY RUN" if dry_run else "APPLIED"
    print(f"\n✅ {mode}: Would update {updated}/{len(entities)} entities")

    if dry_run:
        print("\n💡 Run with --apply to actually update the database")


if __name__ == "__main__":
    import sys
    dry_run = "--apply" not in sys.argv
    asyncio.run(backfill_labels(dry_run=dry_run))
