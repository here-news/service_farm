"""
Check which pages have been fully processed
"""
import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
    # Connect to PostgreSQL to check page status
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    neo4j = Neo4jService()
    await neo4j.connect()

    test_pages = [
        ('pg_013v2wny', 'Newsweek - Initial report'),
        ('pg_00prszmp', 'NY Post - Death toll update'),
        ('pg_013ks2k5', 'Christianity Today - Church'),
        ('pg_00zbqg7h', 'DW - Death toll rises'),
        ('pg_01lnezb0', 'Fox - 13 killed'),
        ('pg_00r7u1zt', 'BBC - Live coverage')
    ]

    print("=" * 80)
    print("📋 PAGE PROCESSING STATUS")
    print("=" * 80)
    print()

    for page_id, desc in test_pages:
        print(f"📄 {page_id}: {desc}")

        # Check in PostgreSQL
        async with db_pool.acquire() as conn:
            pg_status = await conn.fetchrow("""
                SELECT status, title, content_text IS NOT NULL as has_content,
                       embedding IS NOT NULL as has_embedding
                FROM core.pages
                WHERE id = $1
            """, page_id)

        if pg_status:
            print(f"   PostgreSQL: ✓ Exists")
            print(f"      Status: {pg_status['status']}")
            print(f"      Title: {pg_status['title']}")
            print(f"      Has content: {pg_status['has_content']}")
            print(f"      Has embedding: {pg_status['has_embedding']}")
        else:
            print(f"   PostgreSQL: ✗ Not found")

        # Check in Neo4j
        neo4j_claims = await neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
            RETURN count(c) as count
        """, {'page_id': page_id})

        if neo4j_claims and neo4j_claims[0]['count'] > 0:
            print(f"   Neo4j: ✓ {neo4j_claims[0]['count']} claims extracted")
        else:
            print(f"   Neo4j: ✗ No claims found")

        # Check event linkage
        event_link = await neo4j._execute_read("""
            MATCH (e:Event)-[:SUPPORTS]->(c:Claim)<-[:CONTAINS]-(p:Page {id: $page_id})
            RETURN e.canonical_name as event_name, count(c) as claim_count
        """, {'page_id': page_id})

        if event_link and len(event_link) > 0:
            print(f"   Event: ✓ Linked to '{event_link[0]['event_name']}' ({event_link[0]['claim_count']} claims)")
        else:
            print(f"   Event: ✗ Not linked to any event")

        print()

    # Summary
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    async with db_pool.acquire() as conn:
        total_in_pg = await conn.fetchval("""
            SELECT count(*) FROM core.pages WHERE id = ANY($1::text[])
        """, [p[0] for p in test_pages])

    total_in_neo4j = 0
    for page_id, _ in test_pages:
        result = await neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
            RETURN count(c) as count
        """, {'page_id': page_id})
        if result and result[0]['count'] > 0:
            total_in_neo4j += 1

    print(f"Pages in PostgreSQL: {total_in_pg}/6")
    print(f"Pages with extracted claims in Neo4j: {total_in_neo4j}/6")
    print()
    print(f"⚠️  Only pages with extracted claims can be added to events!")
    print(f"   Pages must go through KnowledgeWorker first to extract claims.")

    await db_pool.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
