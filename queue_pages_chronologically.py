"""
Queue pages to event worker in chronological order
"""
import asyncio
import sys
import os
import asyncpg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.job_queue import JobQueue


async def main():
    # Connect to PostgreSQL
    pg_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        user=os.getenv('POSTGRES_USER', 'admin'),
        password=os.getenv('POSTGRES_PASSWORD', 'admin123')
    )

    # Get all pages ordered by pub_time
    print("📄 Loading pages in chronological order...")
    async with pg_pool.acquire() as conn:
        pages = await conn.fetch("""
            SELECT id, url, pub_time, title
            FROM core.pages
            WHERE pub_time IS NOT NULL
            ORDER BY pub_time ASC
        """)

    print(f"✅ Found {len(pages)} pages\n")

    for i, page in enumerate(pages):
        print(f"{i+1}. [{page['pub_time']}] {page['title'][:70]}")

    # Connect to Redis job queue
    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    print(f"\n📤 Queuing {len(pages)} pages to event worker...")
    for page in pages:
        await job_queue.enqueue('queue:event:high', {
            'page_id': str(page['id']),
            'url': page['url']
        })

    print(f"✅ Queued {len(pages)} pages")
    print("\nNow start event worker:")
    print("docker exec -d herenews-api python3 /app/workers/event_worker.py")

    await job_queue.close()
    await pg_pool.close()


if __name__ == '__main__':
    asyncio.run(main())
