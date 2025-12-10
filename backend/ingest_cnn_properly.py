"""
Properly ingest CNN timeline page
"""
import asyncio
import asyncpg
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.job_queue import JobQueue
from services.neo4j_service import Neo4jService
from utils.id_generator import generate_page_id


async def main():
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    neo4j = Neo4jService()
    await neo4j.connect()

    url = "https://ix.cnn.io/dailygraphics/graphics/20251128-hong-kong-fire-timeline/index.html"

    print("=" * 80)
    print("🚀 INGESTING CNN TIMELINE PAGE - PROPER FLOW")
    print("=" * 80)
    print()
    print(f"URL: {url}")
    print()

    # Get current event state
    event = await neo4j._execute_read("MATCH (e:Event) RETURN e.id, e.canonical_name", {})
    if event:
        claims_before = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
            RETURN count(c) as count
        """, {'event_id': event[0]['e.id']})
        print(f"📊 Current Event: {event[0]['e.canonical_name']} ({claims_before[0]['count']} claims)")
        print()

    # Step 1: Create page stub in database
    print("📝 Step 1: Creating page stub in database...")
    page_id = generate_page_id()

    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO core.pages (id, url, canonical_url, status)
            VALUES ($1, $2, $2, 'stub')
        """, page_id, url)

    print(f"   ✅ Created page: {page_id}")
    print()

    # Step 2: Enqueue to extraction worker
    print("📥 Step 2: Enqueuing to extraction worker...")
    await job_queue.enqueue('queue:extraction:high', {
        'page_id': page_id,
        'url': url,
        'priority': 'high'
    })
    print("   ✅ Enqueued")
    print()

    # Monitor extraction
    print("⏳ Step 3: Waiting for extraction (60s)...")
    for i in range(12):
        await asyncio.sleep(5)

        async with db_pool.acquire() as conn:
            page = await conn.fetchrow("""
                SELECT status, content_text IS NOT NULL as has_content
                FROM core.pages WHERE id = $1
            """, page_id)

        if page:
            print(f"   [{i*5}s] status={page['status']}, content={page['has_content']}")

            if page['status'] == 'completed':
                print()
                print("✅ Extraction complete!")
                break
            elif page['status'] == 'failed':
                print()
                print("❌ Extraction failed")
                await cleanup()
                return

    print()

    # Monitor knowledge worker
    print("⏳ Step 4: Waiting for knowledge worker (90s)...")
    for i in range(18):
        await asyncio.sleep(5)

        async with db_pool.acquire() as conn:
            page = await conn.fetchrow("""
                SELECT status, embedding IS NOT NULL as has_embedding
                FROM core.pages WHERE id = $1
            """, page_id)

        if page:
            print(f"   [{i*5}s] status={page['status']}, embedding={page['has_embedding']}")

            if page['status'] == 'knowledge_complete':
                print()
                print("✅ Knowledge extraction complete!")
                break

    # Check claims
    claims = await neo4j._execute_read("""
        MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
        RETURN count(c) as count
    """, {'page_id': page_id})

    print()
    print(f"📊 Claims extracted: {claims[0]['count']}")

    if claims[0]['count'] == 0:
        print("⚠️  No claims - page might be a graphic/timeline without text")
        await db_pool.close()
        await neo4j.close()
        await job_queue.close()
        return

    print()

    # Enqueue to event worker
    print("📥 Step 5: Enqueuing to event worker...")
    await job_queue.enqueue('queue:event:high', {
        'page_id': page_id,
        'url': url
    })
    print("   ✅ Enqueued")
    print()

    # Wait for event processing
    print("⏳ Step 6: Waiting for event worker (30s)...")
    await asyncio.sleep(30)
    print()

    # Check results
    event_link = await neo4j._execute_read("""
        MATCH (e:Event)-[:SUPPORTS]->(c:Claim)<-[:CONTAINS]-(p:Page {id: $page_id})
        RETURN e.canonical_name as name, count(c) as count
    """, {'page_id': page_id})

    if event_link and len(event_link) > 0:
        print(f"✅ SUCCESS!")
        print(f"   Event: {event_link[0]['name']}")
        print(f"   Claims from CNN page: {event_link[0]['count']}")
    else:
        print("⚠️  Not linked to event (might be 100% duplicates)")

    await db_pool.close()
    await neo4j.close()
    await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
