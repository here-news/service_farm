"""
Check event metrics and status
"""
import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # Connect to PostgreSQL
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    # Get event from Neo4j
    result = await neo4j._execute_read("""
        MATCH (e:Event)
        RETURN e.id as id, e.canonical_name as name, e.event_type as event_type,
               e.status as status, e.coherence as coherence,
               e.earliest_time as earliest_time, e.latest_time as latest_time,
               e.created_at as created_at, e.updated_at as updated_at
    """, {})

    if not result:
        print("No events found")
        return

    event = result[0]
    print(f"{'='*80}")
    print(f"Event: {event['name']} ({event['id']})")
    print(f"{'='*80}\n")

    print(f"Status: {event['status']}")
    print(f"Type: {event['event_type']}")
    print(f"Coherence: {event['coherence']}")
    print(f"Time Range: {event['earliest_time']} → {event['latest_time']}")
    print(f"Created: {event['created_at']}")
    print(f"Updated: {event['updated_at']}")
    print()

    # Get narrative from PostgreSQL
    async with db_pool.acquire() as conn:
        narrative_row = await conn.fetchrow("""
            SELECT summary, narrative_version, coherence, updated_at
            FROM content.event_narratives
            WHERE event_id = $1
        """, event['id'])

        if narrative_row:
            print(f"📖 Narrative (version {narrative_row['narrative_version']}):")
            print(f"   Updated: {narrative_row['updated_at']}")
            print(f"   Coherence: {narrative_row['coherence']}")
            print(f"   Length: {len(narrative_row['summary'])} chars")
            print()
            print(f"   Summary:")
            print(f"   {narrative_row['summary'][:400]}...")
            print()
        else:
            print("📖 No narrative found in PostgreSQL")
            print()

    # Get claim count and entity count
    claim_count = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        RETURN count(c) as count
    """, {'event_id': event['id']})

    entity_count = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:INVOLVES]->(en:Entity)
        RETURN count(en) as count
    """, {'event_id': event['id']})

    print(f"📊 Graph Stats:")
    print(f"   Claims: {claim_count[0]['count']}")
    print(f"   Entities: {entity_count[0]['count']}")
    print()

    # Check if event has embedding
    async with db_pool.acquire() as conn:
        emb_row = await conn.fetchrow("""
            SELECT vector_dims(embedding) as dims, created_at
            FROM content.event_embeddings
            WHERE event_id = $1
        """, event['id'])

        if emb_row:
            print(f"🧬 Embedding: {emb_row['dims']} dimensions (created: {emb_row['created_at']})")
        else:
            print(f"🧬 No embedding found")
        print()

    await db_pool.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
