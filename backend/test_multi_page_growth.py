"""
Test event growth with multiple pages and emit detailed output
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.job_queue import JobQueue
from services.neo4j_service import Neo4jService
import asyncpg


# Hong Kong fire pages - all about same incident
TEST_PAGES = [
    {
        'page_id': 'pg_013v2wny',
        'url': 'https://newsweek.com/hong-kong-fire-tai-po-high-rise-apartment-11115768',
        'description': 'Newsweek - Initial report'
    },
    {
        'page_id': 'pg_00prszmp',
        'url': 'https://nypost.com/2025/11/26/world-news/hong-kong-fire-kills-four...',
        'description': 'NY Post - Death toll update'
    },
    {
        'page_id': 'pg_013ks2k5',
        'url': 'https://www.christianitytoday.com/2025/12/hong-kong-apartments-fire-church...',
        'description': 'Christianity Today - Church impact'
    },
    {
        'page_id': 'pg_00zbqg7h',
        'url': 'https://dw.com/en/hong-kong-fire-death-toll-rises-as-blaze-engulfs-high-rise/a-74902659',
        'description': 'DW - Death toll rises'
    },
    {
        'page_id': 'pg_01lnezb0',
        'url': 'https://livenowfox.com/news/13-killed-more-than-dozen-injured-hong-kong-high-rise-fire',
        'description': 'Fox - 13 killed report'
    },
    {
        'page_id': 'pg_00r7u1zt',
        'url': 'https://bbc.com/news/live/c2emg1kj1klt',
        'description': 'BBC - Live coverage'
    }
]


async def get_event_status(neo4j, db_pool):
    """Get current event status"""
    result = await neo4j._execute_read("""
        MATCH (e:Event)
        RETURN e.id as id, e.canonical_name as name, e.coherence as coherence,
               e.status as status, e.updated_at as updated_at
    """, {})

    if not result:
        return None

    event = result[0]

    # Get claim count
    claims = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        RETURN count(c) as count
    """, {'event_id': event['id']})

    # Get entity count
    entities = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:INVOLVES]->(en:Entity)
        RETURN count(en) as count
    """, {'event_id': event['id']})

    # Get embedding status
    async with db_pool.acquire() as conn:
        emb = await conn.fetchrow("""
            SELECT vector_dims(embedding) as dims
            FROM content.event_embeddings
            WHERE event_id = $1
        """, event['id'])

    return {
        'id': event['id'],
        'name': event['name'],
        'coherence': event['coherence'],
        'status': event['status'],
        'claims': claims[0]['count'],
        'entities': entities[0]['count'],
        'embedding_dims': emb['dims'] if emb else None,
        'updated_at': event['updated_at']
    }


async def main():
    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    neo4j = Neo4jService()
    await neo4j.connect()

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    print("=" * 80)
    print("🧪 MULTI-PAGE EVENT GROWTH TEST")
    print("=" * 80)
    print()

    # Get initial state
    initial_state = await get_event_status(neo4j, db_pool)
    if initial_state:
        print(f"📊 Initial State:")
        print(f"   Event: {initial_state['name']} ({initial_state['id']})")
        print(f"   Claims: {initial_state['claims']}")
        print(f"   Entities: {initial_state['entities']}")
        print(f"   Coherence: {initial_state['coherence']:.4f}")
        print(f"   Status: {initial_state['status']}")
        print()
    else:
        print("📊 Initial State: No events exist")
        print()

    # Process pages one by one
    for i, page in enumerate(TEST_PAGES, 1):
        print("-" * 80)
        print(f"📄 Page {i}/{len(TEST_PAGES)}: {page['description']}")
        print(f"   ID: {page['page_id']}")
        print(f"   URL: {page['url'][:60]}...")
        print()

        # Enqueue page
        await job_queue.enqueue('queue:event:high', {
            'page_id': page['page_id'],
            'url': page['url']
        })

        print(f"   ✅ Enqueued to event worker")
        print(f"   ⏳ Waiting for processing (30s)...")
        print()

        # Wait for processing
        await asyncio.sleep(30)

        # Get updated state
        current_state = await get_event_status(neo4j, db_pool)

        if current_state:
            # Calculate changes
            if initial_state:
                claim_delta = current_state['claims'] - initial_state['claims']
                entity_delta = current_state['entities'] - initial_state['entities']
                coherence_delta = current_state['coherence'] - initial_state['coherence']

                print(f"   📊 After Processing:")
                print(f"      Event: {current_state['name']}")
                print(f"      Claims: {current_state['claims']} ({claim_delta:+d})")
                print(f"      Entities: {current_state['entities']} ({entity_delta:+d})")
                print(f"      Coherence: {current_state['coherence']:.4f} ({coherence_delta:+.4f})")
                print(f"      Status: {current_state['status']}")

                # Check if same event (matched) or new event (fragmented)
                if current_state['id'] == initial_state['id']:
                    print(f"      ✅ MATCHED existing event (growth)")
                else:
                    print(f"      ⚠️  Created NEW event (fragmentation)")
            else:
                print(f"   📊 After Processing:")
                print(f"      Event: {current_state['name']} ({current_state['id']})")
                print(f"      Claims: {current_state['claims']}")
                print(f"      Entities: {current_state['entities']}")
                print(f"      Coherence: {current_state['coherence']:.4f}")
                print(f"      Status: {current_state['status']}")
                print(f"      ✨ Created FIRST event")
                initial_state = current_state
        else:
            print(f"   ❌ No event found after processing")

        print()

    print("=" * 80)
    print("🏁 TEST COMPLETE")
    print("=" * 80)
    print()

    # Final summary
    final_state = await get_event_status(neo4j, db_pool)
    if final_state and initial_state:
        total_claims_added = final_state['claims'] - initial_state['claims']
        total_entities_added = final_state['entities'] - initial_state['entities']

        print(f"📈 Growth Summary:")
        print(f"   Total Claims Added: {total_claims_added}")
        print(f"   Total Entities Added: {total_entities_added}")
        print(f"   Final Coherence: {final_state['coherence']:.4f}")
        print(f"   Event Status: {final_state['status']}")
        print()

        # Check event count
        event_count = await neo4j._execute_read("""
            MATCH (e:Event)
            RETURN count(e) as count
        """, {})

        print(f"🎯 Result: {event_count[0]['count']} event(s) in system")
        if event_count[0]['count'] == 1:
            print(f"   ✅ SUCCESS: All pages matched to single event organism")
        else:
            print(f"   ⚠️  FRAGMENTATION: Multiple events created")

    await db_pool.close()
    await neo4j.close()
    await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
