"""
Test enrichment worker on Hong Kong fire event

Run this to see if semantic clustering + LLM synthesis produces useful results
"""
import asyncio
import asyncpg
import os
import sys

# Add backend to path
sys.path.insert(0, '/app')

from workers.enrichment_worker import EnrichmentWorker
from services.job_queue import JobQueue


async def test_enrich_hong_kong_fire():
    """Test enrichment on the Hong Kong fire event"""

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=2,
        max_size=5
    )

    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    # Find Hong Kong fire event
    async with db_pool.acquire() as conn:
        event = await conn.fetchrow("""
            SELECT id, title
            FROM core.events
            WHERE title LIKE '%Hong Kong%fire%'
              AND event_scale = 'meso'
            ORDER BY created_at DESC
            LIMIT 1
        """)

        if not event:
            print("❌ Hong Kong fire event not found")
            return

        event_id = event['id']
        print(f"🔥 Testing enrichment on: {event['title']}")
        print(f"   Event ID: {event_id}\n")

    # Run enrichment
    worker = EnrichmentWorker(db_pool, job_queue, worker_id=1)
    await worker.enrich_event(str(event_id))

    # Show results
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow("""
            SELECT enriched_json
            FROM core.events
            WHERE id = $1
        """, event_id)

        if result and result['enriched_json']:
            import json
            enriched = json.loads(result['enriched_json'])

            if 'micro_narratives' in enriched:
                narratives = enriched['micro_narratives']
                print(f"\n{'='*80}")
                print(f"✨ SYNTHESIZED MICRO-NARRATIVES: {len(narratives)}")
                print(f"{'='*80}\n")

                for i, narrative in enumerate(narratives, 1):
                    print(f"📖 Micro-Narrative {i}: {narrative.get('title', 'Untitled')}")
                    print(f"   Claims: {narrative['claim_count']}")
                    print(f"   Confidence: {narrative.get('confidence', 'N/A')}")
                    print(f"\n   Description:")
                    print(f"   {narrative.get('description', 'N/A')}\n")

                    if narrative.get('what'):
                        print(f"   What: {narrative['what']}")
                    if narrative.get('who'):
                        print(f"   Who: {', '.join(narrative['who'][:5])}")
                    if narrative.get('where'):
                        print(f"   Where: {', '.join(narrative['where'])}")
                    if narrative.get('when'):
                        when = narrative['when']
                        print(f"   When: {when.get('start', 'unknown')} (precision: {when.get('precision', 'unknown')})")
                    if narrative.get('why'):
                        print(f"   Why: {narrative['why']}")
                    if narrative.get('contradictions'):
                        print(f"   ⚠️  Contradictions: {len(narrative['contradictions'])}")
                        for c in narrative['contradictions']:
                            print(f"      - {c}")
                    print(f"\n{'-'*80}\n")

            else:
                print("❌ No micro_narratives found in enriched_json")
        else:
            print("❌ No enriched_json found")

    await db_pool.close()
    await job_queue.close()


if __name__ == '__main__':
    asyncio.run(test_enrich_hong_kong_fire())
