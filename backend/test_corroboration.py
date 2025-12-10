"""Test corroboration mechanism by re-ingesting existing pages"""
import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.job_queue import JobQueue
from services.neo4j_service import Neo4jService


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

    print("=" * 80)
    print("🧪 TESTING CORROBORATION MECHANISM")
    print("=" * 80)
    print()

    # Get pages with knowledge_complete status from previous tests
    async with db_pool.acquire() as conn:
        pages = await conn.fetch("""
            SELECT id, url, status
            FROM core.pages
            WHERE id IN (
                'pg_013v2wny',  -- DW
                'pg_00prszmp',  -- Fox
                'pg_006iquvd',  -- Christianity Today
                'pg_01wzjkk9',  -- Newsweek
                'pg_01euzt1r'   -- NY Post
            )
            ORDER BY id
        """)

    if not pages:
        print("❌ No test pages found")
        await db_pool.close()
        await job_queue.close()
        await neo4j.close()
        return

    print(f"Found {len(pages)} pages to process:")
    for page in pages:
        print(f"  - {page['id']}: {page['url'][:60]}... ({page['status']})")
    print()

    # Reset pages to stub and enqueue for full pipeline
    for i, page in enumerate(pages, 1):
        print(f"📄 Processing page {i}/{len(pages)}: {page['id']}")

        # Reset status
        async with db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.pages
                SET status = 'stub'
                WHERE id = $1
            """, page['id'])

        # Enqueue to extraction worker
        await job_queue.enqueue('queue:extraction:high', {
            'page_id': page['id'],
            'url': page['url'],
            'priority': 'high'
        })
        print(f"  ✅ Enqueued to extraction")

    print()
    print("=" * 80)
    print("📥 All pages enqueued")
    print("=" * 80)
    print()
    print("Waiting for processing...")
    print()

    # Monitor progress
    for round in range(60):  # 5 minutes max
        await asyncio.sleep(5)

        async with db_pool.acquire() as conn:
            status_counts = await conn.fetch("""
                SELECT status, COUNT(*) as count
                FROM core.pages
                WHERE id IN (
                    'pg_013v2wny', 'pg_00prszmp', 'pg_006iquvd',
                    'pg_01wzjkk9', 'pg_01euzt1r'
                )
                GROUP BY status
                ORDER BY status
            """)

        print(f"[{round*5}s] ", end="")
        for row in status_counts:
            print(f"{row['status']}={row['count']} ", end="")
        print()

        # Check if all are knowledge_complete
        all_complete = all(
            row['status'] == 'knowledge_complete'
            for row in status_counts
            if row['count'] > 0
        )

        if all_complete and len(status_counts) == 1:
            print()
            print("✅ All pages reached knowledge_complete")
            break

    # Wait a bit more for event processing
    print()
    print("Waiting for event worker to process...")
    await asyncio.sleep(10)

    # Check results
    print()
    print("=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print()

    # Count events
    events = await neo4j._execute_read("""
        MATCH (e:Event)
        RETURN e.id, e.canonical_name, e.coherence
    """, {})

    print(f"Events created: {len(events)}")
    for event in events:
        coherence = event.get('e.coherence', 0)
        print(f"  - {event['e.canonical_name']} ({event['e.id']}) coherence={coherence:.3f}")
    print()

    if events:
        event_id = events[0]['e.id']

        # Count claims
        claims = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
            RETURN count(c) as count
        """, {'event_id': event_id})
        print(f"Total claims in event: {claims[0]['count']}")

        # Count CORROBORATES relationships
        corroborations = await neo4j._execute_read("""
            MATCH (c1:Claim)-[r:CORROBORATES]->(c2:Claim)
            RETURN count(r) as count
        """, {})
        print(f"CORROBORATES relationships: {corroborations[0]['count']}")

        # Find most corroborated claims
        top_claims = await neo4j._execute_read("""
            MATCH (c:Claim)<-[r:CORROBORATES]-(other:Claim)
            WITH c, count(r) as corr_count, c.confidence as confidence
            ORDER BY corr_count DESC
            LIMIT 5
            RETURN c.id, c.text, corr_count, confidence
        """, {})

        if top_claims:
            print()
            print("Top corroborated claims:")
            for claim in top_claims:
                print(f"  • {claim['c.text'][:80]}...")
                print(f"    Corroborations: {claim['corr_count']}, Confidence: {claim['confidence']:.2f}")
        else:
            print()
            print("⚠️  No corroborated claims found")

    await db_pool.close()
    await job_queue.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
