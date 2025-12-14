#!/usr/bin/env python3
"""
Queue pages with claims to event worker for metabolism testing
"""
import asyncio
import os
import redis.asyncio as redis
from services.neo4j_service import Neo4jService
import json

async def main():
    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # Connect to Redis
    r = await redis.from_url(os.getenv('REDIS_URL', 'redis://redis:6379'))

    # Get pages that have claims
    pages = await neo4j._execute_read("""
        MATCH (p:Page)-[:EXTRACTED]->(c:Claim)
        WITH p, count(c) as claim_count
        WHERE claim_count > 0
        RETURN p.id as page_id, claim_count
        ORDER BY page_id
    """)

    print(f"🔥 Found {len(pages)} pages with claims\n")

    for page in pages:
        page_id = page['page_id']
        claim_count = page['claim_count']

        # Queue to event worker
        job = json.dumps({'page_id': page_id})
        await r.rpush('queue:event:high', job)

        print(f"✅ {page_id} ({claim_count} claims)")

    print(f"\n🚀 Queued {len(pages)} pages to event worker")
    print("📋 Now testing LiveEvent metabolism:")
    print("   - Corroboration detection (claim similarity > 0.85)")
    print("   - Confidence boosting (base + 0.1*√corroboration_count)")
    print("   - Hub entity detection (3+ mentions)")
    print("   - Coherence calculation (hub_coverage + graph_connectivity)")
    print("   - Narrative generation (top 50 corroboration-ranked claims)")

    await neo4j.close()
    await r.aclose()

if __name__ == '__main__':
    asyncio.run(main())
