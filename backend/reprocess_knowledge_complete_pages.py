#!/usr/bin/env python3
"""
Reprocess knowledge_complete pages to rebuild Neo4j graph
"""
import asyncio
import asyncpg
import redis.asyncio as redis

async def main():
    # Connect to Postgres
    pool = await asyncpg.create_pool(
        host='postgres',
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    # Connect to Redis
    r = await redis.from_url('redis://redis:6379')

    # Get knowledge_complete pages
    async with pool.acquire() as conn:
        pages = await conn.fetch("""
            SELECT id, url, word_count
            FROM core.pages
            WHERE status = 'knowledge_complete'
            ORDER BY id
        """)

    print(f"🔄 Found {len(pages)} knowledge_complete pages to reprocess\n")

    # Reset status to 'extracted' and queue to knowledge worker
    for page in pages:
        page_id = page['id']
        url = page['url']
        word_count = page['word_count']

        # Reset status
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.pages
                SET status = 'extracted'
                WHERE id = $1
            """, page_id)

        # Queue to knowledge worker
        import json
        job = json.dumps({'page_id': page_id})
        await r.rpush('queue:semantic:high', job)

        print(f"✅ {page_id} ({word_count} words) - {url[:60]}...")

    print(f"\n🚀 Queued {len(pages)} pages to knowledge worker")
    print("📋 Pipeline: knowledge → event")
    print("\n⏱️  Expected time: ~3-5 minutes for knowledge processing")

    await pool.close()
    await r.aclose()

if __name__ == '__main__':
    asyncio.run(main())
