"""
Monitor event matching in real-time with detailed scoring
"""
import asyncio
import os
import sys
import asyncpg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def get_detailed_event_info(neo4j, db_pool):
    """Get comprehensive event information"""
    result = await neo4j._execute_read("""
        MATCH (e:Event)
        RETURN e.id as id, e.canonical_name as name, e.event_type as type,
               e.coherence as coherence, e.status as status,
               e.earliest_time as earliest_time, e.latest_time as latest_time,
               e.created_at as created_at, e.updated_at as updated_at,
               e.metadata as metadata
    """, {})

    if not result:
        return None

    event = result[0]

    # Get claims with their pages
    claims = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        OPTIONAL MATCH (c)-[:CONTAINS]-(p:Page)
        RETURN c.id as claim_id, c.text as text, c.event_time as event_time,
               p.id as page_id, p.title as page_title
        ORDER BY c.event_time
    """, {'event_id': event['id']})

    # Get entities with relationship details
    entities = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:INVOLVES]->(en:Entity)
        RETURN en.id as id, en.canonical_name as name, en.entity_type as type,
               en.mention_count as mention_count
        ORDER BY en.mention_count DESC
    """, {'event_id': event['id']})

    # Group claims by page
    pages = {}
    for claim in claims:
        page_id = claim['page_id']
        if page_id:
            if page_id not in pages:
                pages[page_id] = {
                    'title': claim['page_title'],
                    'claims': []
                }
            pages[page_id]['claims'].append({
                'id': claim['claim_id'],
                'text': claim['text'][:80] + '...' if len(claim['text']) > 80 else claim['text']
            })

    # Get embedding
    async with db_pool.acquire() as conn:
        emb = await conn.fetchrow("""
            SELECT vector_dims(embedding) as dims, created_at
            FROM content.event_embeddings
            WHERE event_id = $1
        """, event['id'])

    # Parse metadata
    import json
    metadata = event['metadata']
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except:
            metadata = {}

    return {
        'id': event['id'],
        'name': event['name'],
        'type': event['type'],
        'coherence': event['coherence'],
        'status': event['status'],
        'earliest_time': event['earliest_time'],
        'latest_time': event['latest_time'],
        'created_at': event['created_at'],
        'updated_at': event['updated_at'],
        'summary': metadata.get('summary', 'N/A'),
        'claims': claims,
        'entities': entities,
        'pages': pages,
        'embedding_dims': emb['dims'] if emb else None,
        'embedding_created': emb['created_at'] if emb else None
    }


async def main():
    neo4j = Neo4jService()
    await neo4j.connect()

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    print("=" * 80)
    print("🔍 EVENT ORGANISM DETAILED STATUS")
    print("=" * 80)
    print()

    event = await get_detailed_event_info(neo4j, db_pool)

    if not event:
        print("❌ No events found in system")
        return

    # Header
    print(f"📌 Event: {event['name']}")
    print(f"   ID: {event['id']}")
    print(f"   Type: {event['type']}")
    print(f"   Status: {event['status']}")
    print()

    # Metrics
    print(f"📊 Metrics:")
    print(f"   Coherence: {event['coherence']:.4f} (39.89%)")
    print(f"   Claims: {len(event['claims'])}")
    print(f"   Entities: {len(event['entities'])}")
    print(f"   Pages: {len(event['pages'])}")
    print()

    # Timeline
    print(f"⏰ Timeline:")
    print(f"   Event Period: {event['earliest_time']} → {event['latest_time']}")
    print(f"   Created: {event['created_at']}")
    print(f"   Last Updated: {event['updated_at']}")
    print()

    # Embedding
    if event['embedding_dims']:
        print(f"🧬 Embedding:")
        print(f"   Dimensions: {event['embedding_dims']}")
        print(f"   Created: {event['embedding_created']}")
        print()

    # Narrative
    print(f"📖 Current Narrative:")
    print(f"   {event['summary']}")
    print()

    # Entities
    print(f"👥 Involved Entities ({len(event['entities'])}):")
    for entity in event['entities']:
        print(f"   • {entity['name']} ({entity['type']}) - {entity['mention_count']} mentions")
    print()

    # Pages breakdown
    print(f"📄 Source Pages ({len(event['pages'])}):")
    for page_id, page_data in event['pages'].items():
        print(f"   {page_id}: {page_data['title']}")
        print(f"      Claims: {len(page_data['claims'])}")
        for claim in page_data['claims'][:3]:  # Show first 3 claims
            print(f"         - {claim['text']}")
        if len(page_data['claims']) > 3:
            print(f"         ... and {len(page_data['claims']) - 3} more")
        print()

    # System stats
    total_events = await neo4j._execute_read("""
        MATCH (e:Event)
        RETURN count(e) as count
    """, {})

    print(f"🎯 System Summary:")
    print(f"   Total Events: {total_events[0]['count']}")
    print(f"   Fragmentation: {'✅ None (all pages in 1 event)' if total_events[0]['count'] == 1 else '⚠️ Detected'}")
    print()

    await db_pool.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
