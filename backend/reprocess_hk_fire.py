"""
Reprocess Hong Kong Fire pages through the event worker

This script:
1. Reads Hong Kong Fire URLs from /tmp/hk_fire_urls.txt
2. Looks up page IDs in PostgreSQL
3. Enqueues pages to event:high queue in chronological order
"""
import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.job_queue import JobQueue

# URLs to reprocess (from /tmp/hk_fire_urls.txt)
HK_FIRE_URLS = [
    "https://livenowfox.com/news/13-killed-more-than-dozen-injured-hong-kong-high-rise-fire",
    "https://nypost.com/2025/11/26/world-news/hong-kong-fire-kills-four-as-blaze-rips-through-multiple-high-rise-towers-in-tai-po",
    "https://newsweek.com/hong-kong-fire-tai-po-high-rise-apartment-11115768",
    "https://dw.com/en/hong-kong-fire-death-toll-rises-as-blaze-engulfs-high-rise/a-74902659",
    "https://rfa.org/cantonese/news/htm/hk-fire-11262025030238.html",
    "https://bbc.com/news/articles/c87dlvdn5n0o",
    "https://aljazeera.com/news/2025/11/27/dozens-killed-in-hong-kong-high-rise-fire",
    "https://straitstimes.com/asia/death-toll-rises-to-at-least-45-in-hong-kong-apartment-fire",
    "https://apnews.com/article/hong-kong-fire-high-rise-13efb10bd5bd13ebcb7ce5cc0fe087cd",
    "https://voanews.com/a/hong-kong-fire-kills-dozens-injures-scores-in-apartment-complex/7902808.html",
    "https://latimes.com/world-nation/story/2025-11-29/hong-kong-building-fire-death-toll"
]


async def main():
    """Main reprocessing entry point"""
    print("🔄 Hong Kong Fire Reprocessing Script")
    print("=" * 60)

    # Connect to PostgreSQL
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=1,
        max_size=2
    )

    # Connect to Redis job queue
    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    print(f"\n📊 Processing {len(HK_FIRE_URLS)} Hong Kong Fire URLs...")
    print()

    pages_found = 0
    pages_enqueued = 0

    async with db_pool.acquire() as conn:
        for i, url in enumerate(HK_FIRE_URLS, 1):
            # Look up page by URL
            page = await conn.fetchrow("""
                SELECT id, url, title, pub_time,
                       (SELECT COUNT(*) FROM core.claims WHERE page_id = p.id) as claims_count
                FROM core.pages p
                WHERE url = $1
            """, url)

            if page:
                pages_found += 1

                # Enqueue to event worker
                await job_queue.enqueue('queue:event:high', {
                    'page_id': str(page['id'])
                })

                pages_enqueued += 1

                print(f"  [{i}/{len(HK_FIRE_URLS)}] ✅ Enqueued: {page['title'][:60]}...")
                print(f"        URL: {url[:80]}")
                print(f"        Page ID: {page['id']}, Claims: {page['claims_count']}")
                print()

            else:
                print(f"  [{i}/{len(HK_FIRE_URLS)}] ⚠️  NOT FOUND: {url}")
                print()

    print("=" * 60)
    print(f"📈 Summary:")
    print(f"  Total URLs:      {len(HK_FIRE_URLS)}")
    print(f"  Pages found:     {pages_found}")
    print(f"  Pages enqueued:  {pages_enqueued}")
    print()

    if pages_enqueued > 0:
        print("🚀 Pages are now being processed by event worker")
        print("   Monitor logs: docker logs -f herenews-worker-event")
        print("   Check Neo4j:  http://localhost:7474")
    else:
        print("⚠️  No pages were enqueued. Check if URLs exist in database.")

    # Cleanup
    await db_pool.close()
    await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
