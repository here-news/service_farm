"""
Test event growth with REAL pages that exist in PostgreSQL
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.job_queue import JobQueue
from services.neo4j_service import Neo4jService
import asyncpg


# REAL pages that exist in PostgreSQL with knowledge_complete status
REAL_PAGES = [
    {
        'page_id': 'pg_013v2wny',
        'url': 'https://dw.com/en/hong-kong-fire-death-toll-rises-as-blaze-engulfs-high-rise/a-74902659',
        'description': 'DW - Death toll rises'
    },
    {
        'page_id': 'pg_00prszmp',
        'url': 'https://livenowfox.com/news/13-killed-more-than-dozen-injured-hong-kong-high-rise-fire',
        'description': 'Fox - 13 killed report'
    },
    {
        'page_id': 'pg_006iquvd',
        'url': 'https://www.christianitytoday.com/2025/12/hong-kong-apartments-fire-church-christians/',
        'description': 'Christianity Today - Church impact'
    },
    {
        'page_id': 'pg_01wzjkk9',
        'url': 'https://newsweek.com/hong-kong-fire-tai-po-high-rise-apartment-11115768',
        'description': 'Newsweek - Initial report'
    },
    {
        'page_id': 'pg_01euzt1r',
        'url': 'https://nypost.com/2025/11/26/world-news/hong-kong-fire-kills-four-as-blaze-rips...',
        'description': 'NY Post - Death toll update'
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

    # Get unique source pages
    pages = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)<-[:CONTAINS]-(p:Page)
        RETURN DISTINCT p.id as page_id
    """, {'event_id': event['id']})

    return {
        'id': event['id'],
        'name': event['name'],
        'coherence': event['coherence'],
        'status': event['status'],
        'claims': claims[0]['count'],
        'entities': entities[0]['count'],
        'pages': len(pages),
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
    print("🧪 EVENT GROWTH TEST - WITH REAL PAGES")
    print("=" * 80)
    print()

    # Get initial state
    initial_state = await get_event_status(neo4j, db_pool)
    if initial_state:
        print(f"📊 Initial State:")
        print(f"   Event: {initial_state['name']} ({initial_state['id']})")
        print(f"   Claims: {initial_state['claims']}")
        print(f"   Pages: {initial_state['pages']}")
        print(f"   Entities: {initial_state['entities']}")
        print(f"   Coherence: {initial_state['coherence']:.4f}")
        print()
    else:
        print("📊 Initial State: No events exist")
        print()

    # Process each page
    for i, page in enumerate(REAL_PAGES, 1):
        print("-" * 80)
        print(f"📄 Page {i}/{len(REAL_PAGES)}: {page['description']}")
        print(f"   ID: {page['page_id']}")
        print()

        # Enqueue to event worker
        await job_queue.enqueue('queue:event:high', {
            'page_id': page['page_id'],
            'url': page['url']
        })

        print(f"   ✅ Enqueued to event worker")
        print(f"   ⏳ Waiting for processing (25s)...")
        print()

        await asyncio.sleep(25)

        # Get updated state
        current_state = await get_event_status(neo4j, db_pool)

        if current_state:
            if initial_state:
                claim_delta = current_state['claims'] - initial_state['claims']
                page_delta = current_state['pages'] - initial_state['pages']
                entity_delta = current_state['entities'] - initial_state['entities']
                coherence_delta = current_state['coherence'] - initial_state['coherence']

                print(f"   📊 After Processing:")
                print(f"      Claims: {current_state['claims']} ({claim_delta:+d})")
                print(f"      Pages: {current_state['pages']} ({page_delta:+d})")
                print(f"      Entities: {current_state['entities']} ({entity_delta:+d})")
                print(f"      Coherence: {current_state['coherence']:.4f} ({coherence_delta:+.4f})")

                if current_state['id'] == initial_state['id']:
                    print(f"      ✅ MATCHED - Same event")
                else:
                    print(f"      ⚠️  NEW EVENT - Fragmentation detected")
            else:
                print(f"   📊 After Processing:")
                print(f"      Event: {current_state['name']}")
                print(f"      Claims: {current_state['claims']}")
                print(f"      Pages: {current_state['pages']}")
                print(f"      ✨ First event created")
                initial_state = current_state

        print()

    # Final summary
    print("=" * 80)
    print("🏁 TEST COMPLETE")
    print("=" * 80)
    print()

    final_state = await get_event_status(neo4j, db_pool)
    if final_state:
        print(f"📊 Final State:")
        print(f"   Event: {final_state['name']}")
        print(f"   Claims: {final_state['claims']}")
        print(f"   Pages: {final_state['pages']}")
        print(f"   Entities: {final_state['entities']}")
        print(f"   Coherence: {final_state['coherence']:.4f}")
        print()

        # Check fragmentation
        event_count = await neo4j._execute_read("MATCH (e:Event) RETURN count(e) as count", {})
        if event_count[0]['count'] == 1:
            print(f"   ✅ SUCCESS: All pages matched to single event")
        else:
            print(f"   ⚠️  FRAGMENTATION: {event_count[0]['count']} events created")

    await db_pool.close()
    await neo4j.close()
    await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
