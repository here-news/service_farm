#!/usr/bin/env python3
"""
Backfill image_url and wikidata_description for entities that have QID but missing these fields.

This fetches P18 (image) and description from Wikidata and updates Neo4j.
"""
import asyncio
import aiohttp
import hashlib
import os
from neo4j import AsyncGraphDatabase

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'herenews_neo4j_pass')

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

HEADERS = {
    'User-Agent': 'HereNews/1.0 (https://herenews.app; contact@herenews.app) aiohttp/3.x'
}


async def fetch_wikidata_enrichment(session: aiohttp.ClientSession, qid: str) -> dict:
    """Fetch description and P18 image for a QID from Wikidata."""
    try:
        params = {
            'action': 'wbgetentities',
            'ids': qid,
            'format': 'json',
            'props': 'labels|descriptions|claims',
            'languages': 'en'
        }

        async with session.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=15) as resp:
            if resp.status != 200:
                print(f"  ⚠️ {qid}: HTTP {resp.status}")
                return {}

            data = await resp.json()
            entity = data.get('entities', {}).get(qid, {})

            result = {}

            # Get description
            descriptions = entity.get('descriptions', {})
            if 'en' in descriptions:
                result['description'] = descriptions['en']['value']

            # Get P18 image
            claims = entity.get('claims', {})
            for claim in claims.get('P18', []):
                mainsnak = claim.get('mainsnak', {})
                datavalue = mainsnak.get('datavalue', {})
                if datavalue.get('type') == 'string':
                    filename = datavalue['value']
                    # Convert to Wikimedia Commons URL
                    filename_safe = filename.replace(' ', '_')
                    md5 = hashlib.md5(filename_safe.encode()).hexdigest()
                    thumb_url = f"https://upload.wikimedia.org/wikipedia/commons/thumb/{md5[0]}/{md5[0:2]}/{filename_safe}/250px-{filename_safe}"
                    result['image_url'] = thumb_url
                    break

            return result

    except Exception as e:
        print(f"  ⚠️ Failed to fetch {qid}: {e}")
        return {}


async def backfill_images(dry_run: bool = True):
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    async with aiohttp.ClientSession() as http_session:
        async with driver.session() as db_session:
            # Find entities with QID but missing image_url or wikidata_description
            result = await db_session.run("""
                MATCH (e:Entity)
                WHERE e.wikidata_qid IS NOT NULL
                  AND (e.image_url IS NULL OR e.wikidata_description IS NULL)
                RETURN e.id as id, e.canonical_name as name, e.wikidata_qid as qid,
                       e.image_url as existing_image, e.wikidata_description as existing_desc
            """)
            entities = [r async for r in result]

            print(f"\n📊 Found {len(entities)} entities to backfill\n")

            updated = 0
            for ent in entities:
                qid = ent['qid']
                entity_id = ent['id']
                name = ent['name']
                existing_image = ent['existing_image']
                existing_desc = ent['existing_desc']

                # Fetch from Wikidata
                enrichment = await fetch_wikidata_enrichment(http_session, qid)

                if not enrichment:
                    print(f"  ⏭️ {name} ({qid}) - no enrichment data")
                    continue

                new_image = enrichment.get('image_url')
                new_desc = enrichment.get('description')

                updates = []
                if new_image and not existing_image:
                    updates.append(f"image_url")
                if new_desc and not existing_desc:
                    updates.append(f"description")

                if not updates:
                    print(f"  ✓ {name} ({qid}) - already enriched")
                    continue

                print(f"  🔄 {name} ({qid}) - updating: {', '.join(updates)}")

                if not dry_run:
                    await db_session.run("""
                        MATCH (e:Entity {id: $id})
                        SET e.image_url = COALESCE($image_url, e.image_url),
                            e.wikidata_description = COALESCE($description, e.wikidata_description),
                            e.updated_at = datetime()
                    """, {
                        'id': entity_id,
                        'image_url': new_image,
                        'description': new_desc
                    })
                updated += 1

                # Rate limit
                await asyncio.sleep(0.15)

    await driver.close()

    mode = "DRY RUN" if dry_run else "APPLIED"
    print(f"\n✅ {mode}: Updated {updated}/{len(entities)} entities")

    if dry_run:
        print("\n💡 Run with --apply to actually update the database")


if __name__ == "__main__":
    import sys
    dry_run = "--apply" not in sys.argv
    asyncio.run(backfill_images(dry_run=dry_run))
