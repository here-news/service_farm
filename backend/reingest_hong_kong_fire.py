#!/usr/bin/env python3
"""
Re-ingest Hong Kong fire articles through the full pipeline
"""
import asyncio
import redis.asyncio as redis
from services.neo4j_service import Neo4jService
from utils.id_generator import generate_page_id

# Hong Kong fire URLs from the event
URLS = [
    "https://bbc.com/news/live/c2emg1kj1klt",
    "https://livenowfox.com/news/13-killed-more-than-dozen-injured-hong-kong-high-rise-fire",
    "https://dw.com/en/hong-kong-fire-death-toll-rises-as-blaze-engulfs-high-rise/a-74902659",
    "https://nypost.com/2025/11/26/world-news/hong-kong-fire-kills-four-as-blaze-rips-through-multiple-high-rise-towers-in-tai-po",
    "https://newsweek.com/hong-kong-fire-tai-po-high-rise-apartment-11115768",
    "https://apnews.com/article/hong-wang-china-dissent-fire-construction-ff953aec2bc0201b0e3255805a241fc1",
]

async def main():
    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # Connect to Redis
    r = await redis.from_url('redis://redis:6379')

    print("🔥 Re-ingesting Hong Kong fire articles")
    print(f"📊 {len(URLS)} URLs to process\n")

    for url in URLS:
        # Generate page ID
        page_id = generate_page_id()

        # Create Page node in Neo4j
        await neo4j._execute_write("""
            MERGE (p:Page {id: $page_id})
            SET p.url = $url,
                p.status = 'pending',
                p.created_at = datetime()
        """, {'page_id': page_id, 'url': url})

        # Queue to extraction worker (high priority)
        import json
        job = json.dumps({'page_id': page_id, 'url': url, 'retry_count': 0})
        await r.rpush('queue:extraction:high', job)

        print(f"✅ {page_id} - {url}")

    print(f"\n🚀 Queued {len(URLS)} pages to extraction pipeline")
    print("📋 Pipeline: extraction → knowledge → event")
    print("\n⏱️  Expected time: ~5-10 minutes for full pipeline")

    await r.close()
    await neo4j.close()

if __name__ == '__main__':
    asyncio.run(main())
