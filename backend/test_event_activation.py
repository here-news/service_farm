#!/usr/bin/env python3
"""
Test event activation by manually enqueuing a page.

This simulates the knowledge_complete signal that would come from KnowledgeWorker.
"""
import asyncio
import os
import sys
sys.path.insert(0, 'backend')

from services.job_queue import JobQueue

async def main():
    # Connect to Redis
    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    # Enqueue Hong Kong fire page (DW article - has 26 claims)
    page_id = 'pg_013v2wny'

    print(f"🎯 Enqueuing page {page_id} to event worker...")

    await job_queue.enqueue('queue:event:high', {
        'page_id': page_id,
        'url': 'https://dw.com/en/hong-kong-fire-death-toll-rises-as-blaze-engulfs-high-rise/a-74902659',
        'claims_count': 26
    })

    print(f"✅ Page enqueued! Watch event-worker logs...")

if __name__ == "__main__":
    asyncio.run(main())
