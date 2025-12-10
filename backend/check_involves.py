"""
Check if events have INVOLVES relationships to entities
"""
import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService
from repositories.event_repository import EventRepository


async def main():
    # Connect to PostgreSQL
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=1,
        max_size=2
    )

    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # Get all events
    result = await neo4j._execute_read("""
        MATCH (e:Event)
        RETURN e.id, e.canonical_name, e.event_type
        ORDER BY e.created_at DESC
        LIMIT 10
    """, {})

    print(f"\n{'='*80}")
    print(f"Found {len(result)} events")
    print(f"{'='*80}\n")

    for row in result:
        event_id = row['e.id']
        event_name = row['e.canonical_name']
        event_type = row['e.event_type']

        # Check INVOLVES relationships
        involves = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[r:INVOLVES]->(entity:Entity)
            RETURN entity.id, entity.canonical_name, entity.entity_type
        """, {'event_id': event_id})

        # Check SUPPORTS relationships (claims)
        claims = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[r:SUPPORTS]->(c:Claim)
            RETURN count(c) as claim_count
        """, {'event_id': event_id})

        claim_count = claims[0]['claim_count'] if claims else 0

        print(f"Event: {event_name} ({event_id})")
        print(f"  Type: {event_type}")
        print(f"  Claims: {claim_count}")
        print(f"  Entities via INVOLVES: {len(involves)}")

        if involves:
            for entity in involves[:5]:  # Show first 5
                print(f"    - {entity['entity.canonical_name']} ({entity['entity.entity_type']})")
        else:
            print(f"    ⚠️  NO INVOLVES RELATIONSHIPS!")

        print()

    await db_pool.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
