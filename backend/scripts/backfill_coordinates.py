#!/usr/bin/env python3
"""
Backfill coordinates for existing location entities from Wikidata.

Uses the existing WikidataClient which handles API calls properly.
"""
import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import AsyncGraphDatabase
from services.wikidata_client import WikidataClient

# Neo4j connection settings
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'herenews_neo4j_pass')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')


async def backfill():
    """Main backfill function."""
    print("=" * 60)
    print("Backfilling coordinates for location entities")
    print("=" * 60)

    # Initialize WikidataClient
    wikidata = WikidataClient()

    # Connect to Neo4j
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    async with driver.session(database=NEO4J_DATABASE) as neo4j_session:
        # Find location entities with QIDs but no coordinates
        result = await neo4j_session.run("""
            MATCH (e:Entity)
            WHERE e.entity_type IN ['GPE', 'LOC', 'LOCATION']
              AND e.wikidata_qid IS NOT NULL
              AND (e.latitude IS NULL OR e.longitude IS NULL)
            RETURN e.id as id, e.canonical_name as name, e.wikidata_qid as qid
            ORDER BY e.canonical_name
        """)

        entities = await result.data()
        print(f"\nFound {len(entities)} location entities needing coordinates\n")

        if not entities:
            print("Nothing to update!")
            await wikidata.close()
            await driver.close()
            return

        # Fetch coordinates from Wikidata
        updated = 0
        failed = 0

        for entity in entities:
            entity_id = entity['id']
            name = entity['name']
            qid = entity['qid']

            print(f"  {name} ({qid})...", end=" ", flush=True)

            try:
                await wikidata._ensure_session()
                coords = await wikidata._fetch_entity_coordinates(qid)

                if coords:
                    # Update Neo4j
                    await neo4j_session.run("""
                        MATCH (e:Entity {id: $entity_id})
                        SET e.latitude = $latitude,
                            e.longitude = $longitude,
                            e.updated_at = datetime()
                    """, {
                        'entity_id': entity_id,
                        'latitude': coords['latitude'],
                        'longitude': coords['longitude']
                    })
                    print(f"✓ ({coords['latitude']:.4f}, {coords['longitude']:.4f})")
                    updated += 1
                else:
                    print("✗ (no coordinates)")
                    failed += 1
            except Exception as e:
                print(f"✗ (error: {e})")
                failed += 1

            # Be nice to Wikidata API
            await asyncio.sleep(0.2)

    await wikidata.close()
    await driver.close()

    print("\n" + "=" * 60)
    print(f"Done! Updated: {updated}, No coordinates: {failed}")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(backfill())
