"""
Enqueue CNN timeline through full pipeline and monitor
"""
import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.job_queue import JobQueue
from services.neo4j_service import Neo4jService


async def main():
    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    neo4j = Neo4jService()
    await neo4j.connect()

    url = "https://ix.cnn.io/dailygraphics/graphics/20251128-hong-kong-fire-timeline/index.html"

    print("=" * 80)
    print("🚀 CNN TIMELINE - FULL PIPELINE TEST")
    print("=" * 80)
    print()
    print(f"URL: {url}")
    print()

    # Get current event state
    event = await neo4j._execute_read("""
        MATCH (e:Event)
        RETURN e.id as id, e.canonical_name as name
    """, {})

    if event:
        claims_before = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
            RETURN count(c) as count
        """, {'event_id': event[0]['id']})

        print(f"📊 Current Event State:")
        print(f"   Event: {event[0]['name']}")
        print(f"   Claims: {claims_before[0]['count']}")
        print()

    # Enqueue to extraction
    print("📥 Stage 1: Enqueuing to Extraction Worker...")
    await job_queue.enqueue('queue:extraction:high', {
        'url': url,
        'priority': 'high'
    })
    print("   ✅ Enqueued")
    print()

    # Wait and monitor
    print("⏳ Monitoring progress (60s max)...")
    print()

    page_id = None
    for i in range(12):  # 12 checks x 5 seconds = 60 seconds
        await asyncio.sleep(5)

        # Check if page exists
        async with db_pool.acquire() as conn:
            page = await conn.fetchrow("""
                SELECT id, status, content_text IS NOT NULL as has_content
                FROM core.pages
                WHERE url = $1
            """, url)

        if page:
            page_id = page['id']
            print(f"   [{i*5}s] Page {page_id}: status={page['status']}, content={page['has_content']}")

            if page['status'] == 'knowledge_complete':
                print()
                print("✅ Knowledge Worker completed!")
                break
            elif page['status'] == 'failed':
                print()
                print("❌ Processing failed")
                await db_pool.close()
                await neo4j.close()
                await job_queue.close()
                return

    if not page_id:
        print()
        print("⚠️  Page not found after 60s")
        await db_pool.close()
        await neo4j.close()
        await job_queue.close()
        return

    print()

    # Check claims
    claims = await neo4j._execute_read("""
        MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
        RETURN count(c) as count
    """, {'page_id': page_id})

    print(f"📊 Claims extracted: {claims[0]['count']}")
    print()

    if claims[0]['count'] == 0:
        print("⚠️  No claims extracted")
        await db_pool.close()
        await neo4j.close()
        await job_queue.close()
        return

    # Enqueue to event worker
    print("📥 Stage 2: Enqueuing to Event Worker...")
    await job_queue.enqueue('queue:event:high', {
        'page_id': page_id,
        'url': url
    })
    print("   ✅ Enqueued")
    print()

    # Wait for event processing
    print("⏳ Waiting for Event Worker (30s)...")
    await asyncio.sleep(30)
    print()

    # Check final state
    event_link = await neo4j._execute_read("""
        MATCH (e:Event)-[:SUPPORTS]->(c:Claim)<-[:CONTAINS]-(p:Page {id: $page_id})
        RETURN e.id as event_id, e.canonical_name as event_name, count(c) as claim_count
    """, {'page_id': page_id})

    if event_link and len(event_link) > 0:
        link = event_link[0]
        print(f"✅ SUCCESS - Claims added to event!")
        print(f"   Event: {link['event_name']}")
        print(f"   Claims from this page: {link['claim_count']}")
        print()

        # Get final event state
        total_claims = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
            RETURN count(c) as count
        """, {'event_id': link['event_id']})

        if event:
            added = total_claims[0]['count'] - claims_before[0]['count']
            print(f"📈 Event Growth:")
            print(f"   Before: {claims_before[0]['count']} claims")
            print(f"   After: {total_claims[0]['count']} claims")
            print(f"   Added: {added} unique claims ({link['claim_count'] - added} duplicates)")
    else:
        print("⚠️  Claims not linked to any event")
        print("   Either no match found or still processing")

    await db_pool.close()
    await neo4j.close()
    await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
