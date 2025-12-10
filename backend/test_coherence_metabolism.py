"""
Test coherence-guided metabolism with a new page
"""
import asyncio
import asyncpg
import os
import sys

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

    # Test with a different news source
    url = "https://www.theguardian.com/world/2024/nov/26/hong-kong-high-rise-fire-several-dead"

    print("=" * 80)
    print("🧪 TESTING COHERENCE-GUIDED METABOLISM")
    print("=" * 80)
    print()
    print(f"URL: {url}")
    print()

    # Get current event state
    event = await neo4j._execute_read("""
        MATCH (e:Event {canonical_name: 'Hong Kong High-Rise Fire 2024'})
        RETURN e.id, e.canonical_name, e.coherence
    """, {})

    if event:
        event_id = event[0]['e.id']
        event_name = event[0]['e.canonical_name']
        old_coherence = event[0]['e.coherence']

        claims_before = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
            RETURN count(c) as count
        """, {'event_id': event_id})

        print(f"📊 Current Event: {event_name}")
        print(f"   Claims: {claims_before[0]['count']}")
        print(f"   Coherence: {old_coherence:.3f}" if old_coherence else "   Coherence: Not set")
        print()

    # Step 1: Create page stub
    print("📝 Step 1: Creating page stub...")
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
                await db_pool.close()
                await neo4j.close()
                await job_queue.close()
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
        print("⚠️  No claims extracted")
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

    # Check results - COHERENCE UPDATE!
    event_after = await neo4j._execute_read("""
        MATCH (e:Event {canonical_name: 'Hong Kong High-Rise Fire 2024'})
        RETURN e.id, e.canonical_name, e.coherence
    """, {})

    if event_after:
        event_id = event_after[0]['e.id']
        new_coherence = event_after[0]['e.coherence']

        claims_after = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
            RETURN count(c) as count
        """, {'event_id': event_id})

        claims_from_page = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)<-[:CONTAINS]-(p:Page {id: $page_id})
            RETURN count(c) as count
        """, {'event_id': event_id, 'page_id': page_id})

        print("=" * 80)
        print("📈 METABOLISM RESULTS")
        print("=" * 80)
        print(f"Event: {event_name}")
        print(f"Claims: {claims_before[0]['count']} → {claims_after[0]['count']} (+{claims_after[0]['count'] - claims_before[0]['count']})")
        print(f"Claims from new page: {claims_from_page[0]['count']}")

        if old_coherence and new_coherence:
            delta = new_coherence - old_coherence
            print(f"Coherence: {old_coherence:.3f} → {new_coherence:.3f} (Δ {delta:+.3f})")

            if delta > 0.1:
                print("✨ SIGNIFICANT COHERENCE BOOST - Narrative should have regenerated!")
            elif delta > 0.0:
                print("📊 Minor coherence improvement")
            else:
                print("📉 Coherence decreased or unchanged")
        elif new_coherence:
            print(f"Coherence: {new_coherence:.3f} (newly calculated)")
        else:
            print("⚠️  Coherence not calculated")

    await db_pool.close()
    await neo4j.close()
    await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
