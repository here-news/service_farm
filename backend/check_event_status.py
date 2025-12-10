"""
Check event status and narrative
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

    # Get full event data from Neo4j
    result = await neo4j._execute_read("""
        MATCH (e:Event)
        RETURN e
    """, {})

    if not result:
        print("No events found")
        return

    event = result[0]['e']
    print(f"{'='*80}")
    print(f"Event: {event['canonical_name']} ({event['id']})")
    print(f"{'='*80}\n")

    # Basic info
    print(f"📊 Status & Type:")
    print(f"   Status: {event.get('status', 'N/A')}")
    print(f"   Type: {event.get('event_type', 'N/A')}")
    print(f"   Coherence: {event.get('coherence', 'N/A')}")
    print()

    # Timing
    print(f"⏰ Time Range:")
    print(f"   Earliest: {event.get('earliest_time', 'N/A')}")
    print(f"   Latest: {event.get('latest_time', 'N/A')}")
    print(f"   Created: {event.get('created_at', 'N/A')}")
    print(f"   Updated: {event.get('updated_at', 'N/A')}")
    print()

    # Narrative
    summary = event.get('summary', None)
    if summary:
        print(f"📖 Narrative:")
        print(f"   Length: {len(summary)} chars")
        print(f"   Content:")
        print(f"   {summary}")
        print()
    else:
        print(f"📖 No narrative/summary found")
        print()

    # Metadata
    metadata = event.get('metadata', None)
    if metadata:
        print(f"🔧 Metadata:")
        print(f"   {metadata}")
        print()

    # Get claim count
    claim_count = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        RETURN count(c) as count
    """, {'event_id': event['id']})

    # Get entity count and list
    entities = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:INVOLVES]->(en:Entity)
        RETURN en.canonical_name as name, en.entity_type as type
        ORDER BY en.canonical_name
    """, {'event_id': event['id']})

    print(f"📊 Graph Connections:")
    print(f"   Claims: {claim_count[0]['count']}")
    print(f"   Entities: {len(entities)}")
    if entities:
        for entity in entities:
            print(f"      - {entity['name']} ({entity['type']})")
    print()

    # Check embedding
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

    await db_pool.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
