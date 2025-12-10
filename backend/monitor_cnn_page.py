"""
Monitor CNN timeline page processing
"""
import asyncio
import asyncpg
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def check_page_status(db_pool, url_pattern):
    """Check page status in PostgreSQL"""
    async with db_pool.acquire() as conn:
        return await conn.fetchrow("""
            SELECT id, url, status, title,
                   content_text IS NOT NULL as has_content,
                   embedding IS NOT NULL as has_embedding
            FROM core.pages
            WHERE url LIKE $1
            ORDER BY created_at DESC
            LIMIT 1
        """, f"%{url_pattern}%")


async def check_claims(neo4j, page_id):
    """Check claims extracted from page"""
    result = await neo4j._execute_read("""
        MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
        RETURN count(c) as count
    """, {'page_id': page_id})
    return result[0]['count'] if result else 0


async def check_event_link(neo4j, page_id):
    """Check if page claims are linked to event"""
    result = await neo4j._execute_read("""
        MATCH (e:Event)-[:SUPPORTS]->(c:Claim)<-[:CONTAINS]-(p:Page {id: $page_id})
        RETURN e.canonical_name as event_name, count(c) as claim_count
    """, {'page_id': page_id})
    return result[0] if result and len(result) > 0 else None


async def main():
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    neo4j = Neo4jService()
    await neo4j.connect()

    url_pattern = "cnn.io/dailygraphics/graphics/20251128-hong-kong-fire-timeline"

    print("=" * 80)
    print("🔍 MONITORING CNN TIMELINE PAGE PROCESSING")
    print("=" * 80)
    print()

    for i in range(20):  # Check for up to 20 iterations (2 minutes)
        print(f"\r⏳ Check {i+1}/20...", end="", flush=True)

        page = await check_page_status(db_pool, url_pattern)

        if page:
            print(f"\n\n✅ PAGE FOUND: {page['id']}")
            print(f"   Status: {page['status']}")
            print(f"   URL: {page['url'][:70]}...")
            print(f"   Title: {page['title']}")
            print(f"   Has content: {page['has_content']}")
            print(f"   Has embedding: {page['has_embedding']}")
            print()

            if page['status'] == 'knowledge_complete':
                # Check claims
                claim_count = await check_claims(neo4j, page['id'])
                print(f"📊 Claims extracted: {claim_count}")
                print()

                if claim_count > 0:
                    # Check event link
                    event_link = await check_event_link(neo4j, page['id'])

                    if event_link:
                        print(f"🎯 Event linked: {event_link['event_name']}")
                        print(f"   Claims in event: {event_link['claim_count']}")
                        print()
                        print("✅ FULL PIPELINE COMPLETE!")
                        break
                    else:
                        print("⏳ Waiting for Event Worker to process...")
            elif page['status'] == 'completed':
                print("⏳ Waiting for Knowledge Worker to extract claims...")
            elif page['status'] == 'failed':
                print("❌ Processing failed")
                break

        await asyncio.sleep(6)  # Check every 6 seconds

    await db_pool.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
