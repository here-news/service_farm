"""
Reset events: flush all Event nodes and reprocess pages chronologically
"""
import asyncio
import sys
import os
import asyncpg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.neo4j_service import Neo4jService
from services.job_queue import JobQueue


async def main():
    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # Connect to PostgreSQL
    pg_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        user=os.getenv('POSTGRES_USER', 'admin'),
        password=os.getenv('POSTGRES_PASSWORD', 'admin123')
    )

    print("="*80)
    print("RESET EVENTS")
    print("="*80)

    # Step 1: Count current events
    result = await neo4j._execute_read("""
        MATCH (e:Event)
        RETURN count(e) as count
    """)
    event_count = result[0]['count'] if result else 0
    print(f"\n📊 Current events in Neo4j: {event_count}")

    # Step 2: Confirm deletion
    response = input(f"\n⚠️  This will DELETE all {event_count} Event nodes. Continue? [y/N]: ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        await neo4j.close()
        await pg_pool.close()
        return

    # Step 3: Delete all events
    print("\n🗑️  Deleting all Event nodes...")
    await neo4j._execute_write("""
        MATCH (e:Event)
        DETACH DELETE e
    """)
    print("✅ All events deleted")

    # Step 4: Get all pages ordered by pub_time
    print("\n📄 Loading pages in chronological order...")
    async with pg_pool.acquire() as conn:
        pages = await conn.fetch("""
            SELECT id, url, pub_time, title
            FROM core.pages
            WHERE pub_time IS NOT NULL
            ORDER BY pub_time ASC
        """)

    print(f"✅ Found {len(pages)} pages")

    if len(pages) > 0:
        print(f"\n   Earliest: {pages[0]['pub_time']} - {pages[0]['title'][:60]}")
        print(f"   Latest:   {pages[-1]['pub_time']} - {pages[-1]['title'][:60]}")

    # Step 5: Option to reprocess pages
    print(f"\n📋 Next step: Reprocess {len(pages)} pages through event worker")
    print("   This will rebuild the event hierarchy from scratch.")
    print("\n   You can:")
    print("   1. Run event_worker.py manually")
    print("   2. Or queue pages to Redis for workers to process")

    response = input("\nQueue pages to event worker? [y/N]: ")
    if response.lower() == 'y':
        # Connect to Redis job queue
        job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
        await job_queue.connect()

        print("\n📤 Queuing pages to event worker...")
        for page in pages:
            await job_queue.enqueue('queue:event:high', {
                'page_id': str(page['id']),
                'url': page['url']
            })

        print(f"✅ Queued {len(pages)} pages")
        print("\n   Start event worker with:")
        print("   docker exec herenews-api python3 /app/workers/event_worker.py")

        await job_queue.close()

    await neo4j.close()
    await pg_pool.close()

    print("\n✅ Reset complete")


if __name__ == '__main__':
    asyncio.run(main())
