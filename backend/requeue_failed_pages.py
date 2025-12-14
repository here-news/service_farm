"""
Requeue failed pages for extraction

This script:
1. Queries all pages with status='failed' from the database
2. Resets their status to 'stub' so they can be retried
3. Adds them back to the extraction queue
"""
import asyncio
import asyncpg
import redis.asyncio as redis
import json
import os
from datetime import datetime

DATABASE_URL = os.getenv('DATABASE_URL', "postgresql://herenews_user:herenews_pass@postgres:5432/herenews")
REDIS_URL = os.getenv('REDIS_URL', "redis://redis:6379")


async def requeue_failed_pages():
    """Requeue all failed pages"""

    # Connect to database
    pool = await asyncpg.create_pool(DATABASE_URL)

    # Connect to Redis
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)

    try:
        # Query failed pages
        failed_pages = await pool.fetch(
            """
            SELECT id, url, status, updated_at
            FROM core.pages
            WHERE status = 'failed'
            ORDER BY updated_at DESC
            """
        )

        print(f"📊 Found {len(failed_pages)} failed pages")

        if not failed_pages:
            print("✅ No failed pages to requeue")
            return

        # Display failed pages
        print("\n🔍 Failed pages:")
        for page in failed_pages:
            print(f"  - {page['url']}")
            print(f"    ID: {page['id']}")
            print(f"    Failed at: {page['updated_at']}")
            print()

        # Reset status and requeue
        requeued = 0
        for page in failed_pages:
            page_id = str(page['id'])
            url = page['url']

            # Reset status to 'stub'
            await pool.execute(
                """
                UPDATE core.pages
                SET status = 'stub', updated_at = $1
                WHERE id = $2
                """,
                datetime.utcnow(),
                page['id']
            )

            # Add to extraction queue
            job = {
                'page_id': page_id,
                'url': url,
                'retry_count': 0
            }
            await redis_client.lpush('queue:extraction:high', json.dumps(job))

            requeued += 1
            print(f"✅ Requeued: {url}")

        print(f"\n🎉 Successfully requeued {requeued} pages")

        # Show queue status
        queue_length = await redis_client.llen('queue:extraction:high')
        print(f"📊 Current extraction queue length: {queue_length}")

    finally:
        await pool.close()
        await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(requeue_failed_pages())
