"""
Re-process pages with malformed claims (no entities)

This will:
1. Delete claims without entities from PostgreSQL and Neo4j
2. Re-queue pages for semantic analysis
3. Test the new wikidata worker (enrichment + merge)
"""
import asyncio
import sys
import os
import asyncpg
from uuid import UUID

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.neo4j_service import Neo4jService
from services.job_queue import JobQueue


async def main():
    # Connect to PostgreSQL
    pg_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        user=os.getenv('POSTGRES_USER', 'admin'),
        password=os.getenv('POSTGRES_PASSWORD', 'admin123')
    )

    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # Connect to Redis job queue
    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    print("="*80)
    print("REPROCESSING MALFORMED PAGES")
    print("="*80)

    # Get all pages
    async with pg_pool.acquire() as conn:
        all_pages = await conn.fetch("""
            SELECT id, url, title
            FROM core.pages
            ORDER BY pub_time ASC
        """)

    print(f"\n📄 Checking {len(all_pages)} pages...")

    pages_to_reprocess = []

    for page in all_pages:
        page_id = page['id']

        # Get claims
        async with pg_pool.acquire() as conn:
            claims = await conn.fetch("""
                SELECT id, text
                FROM core.claims
                WHERE page_id = $1
            """, page_id)

        if not claims:
            continue

        # Check if any claim has no entities
        claims_without_entities = []
        for claim in claims:
            entities = await neo4j._execute_read("""
                MATCH (c:Claim {id: $claim_id})-[:MENTIONS]->(e:Entity)
                RETURN count(e) as entity_count
            """, {'claim_id': str(claim['id'])})

            entity_count = entities[0]['entity_count'] if entities else 0
            if entity_count == 0:
                claims_without_entities.append(claim)

        if claims_without_entities:
            pages_to_reprocess.append({
                'page': page,
                'claims_to_delete': claims_without_entities
            })

    print(f"\n⚠️  Found {len(pages_to_reprocess)} pages with malformed claims")

    if not pages_to_reprocess:
        print("✅ No malformed claims to fix!")
        await neo4j.close()
        await pg_pool.close()
        await job_queue.close()
        return

    # Process each page
    for i, item in enumerate(pages_to_reprocess, 1):
        page = item['page']
        page_id = page['id']
        claims_to_delete = item['claims_to_delete']

        title = (page['title'] or 'Untitled')[:60]
        print(f"\n{i}. {title}")
        print(f"   Deleting {len(claims_to_delete)} claims without entities...")

        # Delete claims from Neo4j first
        for claim in claims_to_delete:
            await neo4j._execute_write("""
                MATCH (c:Claim {id: $claim_id})
                DETACH DELETE c
            """, {'claim_id': str(claim['id'])})

        # Delete claims from PostgreSQL
        claim_ids = [c['id'] for c in claims_to_delete]
        async with pg_pool.acquire() as conn:
            await conn.execute("""
                DELETE FROM core.claims
                WHERE id = ANY($1::uuid[])
            """, claim_ids)

        print(f"   ✅ Deleted {len(claims_to_delete)} malformed claims")

        # Re-queue for semantic analysis
        await job_queue.enqueue('queue:semantic:high', {
            'page_id': str(page_id),
            'url': page['url']
        })
        print(f"   📤 Queued for semantic analysis")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n✅ Reprocessed {len(pages_to_reprocess)} pages")
    print(f"📤 Queued {len(pages_to_reprocess)} jobs to queue:semantic:high")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)

    print("\n1. Semantic worker will re-analyze pages and extract entities")
    print("2. Wikidata worker will enrich entities with QIDs (testing new merged worker!)")
    print("3. Entity merge will consolidate duplicates after enrichment")
    print("4. Then queue pages to event worker for event formation")

    print("\nMonitor progress:")
    print("  docker logs herenews-worker-semantic-1 --tail 20 -f")
    print("  docker logs herenews-worker-wikidata --tail 20 -f")

    await neo4j.close()
    await pg_pool.close()
    await job_queue.close()


if __name__ == '__main__':
    asyncio.run(main())
