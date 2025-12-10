#!/usr/bin/env python3
"""
Queue knowledge_complete pages to event worker for metabolism testing
"""
import asyncio
import redis.asyncio as redis
from services.neo4j_service import Neo4jService

async def main():
    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # Connect to Redis
    r = await redis.from_url('redis://redis:6379')

    # Get all knowledge_complete pages
    pages = await neo4j._execute_read("""
        MATCH (p:Page)
        WHERE p.status = 'knowledge_complete'
        RETURN p.id as id, p.url as url
        ORDER BY p.id
    """)

    print(f"Found {len(pages)} knowledge_complete pages")

    # Queue each page to event worker
    for page in pages:
        page_id = page['id']
        url = page['url']

        # Push to event queue
        await r.rpush('queue:event:high', page_id)
        print(f"✅ Queued {page_id} - {url}")

    print(f"\n🚀 Queued {len(pages)} pages to event worker")

    await r.close()
    await neo4j.close()

if __name__ == '__main__':
    asyncio.run(main())
