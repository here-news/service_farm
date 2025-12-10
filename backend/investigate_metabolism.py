"""Investigate claim metabolism - why were claims marked as duplicates?"""
import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
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
    print("🔬 INVESTIGATING CLAIM METABOLISM")
    print("=" * 80)
    print()

    # Get pages that should have contributed to event
    test_pages = [
        'pg_013v2wny',  # DW: "Death toll rises" (26 claims)
        'pg_00prszmp',  # Fox: "13 killed report" (25 claims)
        'pg_006iquvd',  # Christianity Today: "Church impact" (24 claims)
        'pg_01wzjkk9',  # Newsweek: "Initial report" (22 claims)
        'pg_01euzt1r',  # NY Post: "Death toll update" (25 claims)
    ]

    event_id = 'ev_4uvbwao6'

    for page_id in test_pages:
        print(f"\n{'='*80}")
        print(f"📄 Page: {page_id}")
        print(f"{'='*80}")

        # Get page info
        async with db_pool.acquire() as conn:
            page = await conn.fetchrow("""
                SELECT url, status, content_text IS NOT NULL as has_content,
                       embedding IS NOT NULL as has_embedding
                FROM core.pages WHERE id = $1
            """, page_id)

        if not page:
            print(f"   ❌ Page not found in PostgreSQL")
            continue

        print(f"   URL: {page['url'][:60]}...")
        print(f"   Status: {page['status']}")
        print(f"   Has content: {page['has_content']}")
        print(f"   Has embedding: {page['has_embedding']}")
        print()

        # Get all claims from this page
        all_claims = await neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
            RETURN count(c) as total
        """, {'page_id': page_id})

        print(f"   Total claims extracted: {all_claims[0]['total']}")

        # Get claims linked to event
        event_claims = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)<-[:CONTAINS]-(p:Page {id: $page_id})
            RETURN count(c) as linked
        """, {'event_id': event_id, 'page_id': page_id})

        linked = event_claims[0]['linked']
        total = all_claims[0]['total']

        print(f"   Claims linked to event: {linked}/{total}")

        if linked == 0 and total > 0:
            print(f"   ⚠️  100% REJECTION - All {total} claims were either duplicates or rejected")

            # Sample some claims to see what they say
            sample_claims = await neo4j._execute_read("""
                MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
                RETURN c.id, c.text
                LIMIT 5
            """, {'page_id': page_id})

            print(f"\n   Sample claims from this page:")
            for claim in sample_claims:
                print(f"      • {claim['c.text'][:100]}...")

            # Check if any were marked as duplicates via EQUIVALENT_TO relationship
            equivalent_claims = await neo4j._execute_read("""
                MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c1:Claim)-[:EQUIVALENT_TO]->(c2:Claim)
                RETURN count(c1) as dup_count
            """, {'page_id': page_id})

            dup_count = equivalent_claims[0]['dup_count']
            print(f"\n   Claims marked EQUIVALENT_TO other claims: {dup_count}/{total}")

        elif linked < total:
            print(f"   ⚠️  PARTIAL REJECTION - {total - linked} claims rejected")

    print()
    print("=" * 80)
    print("📊 OVERALL EVENT STATUS")
    print("=" * 80)

    # Get total claims in event
    total_event_claims = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        RETURN count(c) as total
    """, {'event_id': event_id})

    print(f"Total claims in event: {total_event_claims[0]['total']}")

    # Get unique pages contributing
    unique_pages = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)<-[:CONTAINS]-(p:Page)
        RETURN count(DISTINCT p) as pages
    """, {'event_id': event_id})

    print(f"Unique pages contributing: {unique_pages[0]['pages']}")

    # Check for EQUIVALENT_TO relationships in general
    all_equivalents = await neo4j._execute_read("""
        MATCH (c1:Claim)-[:EQUIVALENT_TO]->(c2:Claim)
        RETURN count(*) as total
    """, {})

    print(f"Total EQUIVALENT_TO relationships: {all_equivalents[0]['total']}")

    await neo4j.close()
    await db_pool.close()


if __name__ == "__main__":
    asyncio.run(main())
