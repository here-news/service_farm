"""
Test event growth - process second page about same fire
"""
import asyncio
import os
from services.job_queue import JobQueue


async def main():
    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    # Second page: (25 claims)
    page_id = 'pg_00prszmp'

    print(f"🎯 Enqueuing second page {page_id} to test event matching...")
    print(f"Expected: Should match existing 'Wang Fuk Court Fire' event")
    print()

    await job_queue.enqueue('queue:event:high', {
        'page_id': page_id,
        'url': 'https://nypost.com/2025/11/26/world-news/hong-kong-fire-kills-four...',
        'claims_count': 22
    })

    print(f"✅ Page enqueued! Watch event-worker logs...")

    await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
