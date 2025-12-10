"""
Clean up existing events without INVOLVES relationships
"""
import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


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
        RETURN e.id, e.canonical_name
    """, {})

    print(f"Found {len(result)} events to delete\n")

    for row in result:
        event_id = row['e.id']
        event_name = row['e.canonical_name']

        print(f"Deleting: {event_name} ({event_id})")

        # Delete from Neo4j (with all relationships)
        await neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})
            DETACH DELETE e
        """, {'event_id': event_id})

        # Delete from PostgreSQL embeddings
        async with db_pool.acquire() as conn:
            await conn.execute("""
                DELETE FROM content.event_embeddings WHERE event_id = $1
            """, event_id)

    print(f"\n✅ Deleted {len(result)} events")

    await db_pool.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
