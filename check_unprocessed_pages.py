"""
Check for unprocessed pages and malformed claims
"""
import asyncio
import sys
import os
import asyncpg
from uuid import UUID

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.neo4j_service import Neo4jService


async def main():
    # Connect to PostgreSQL
    pg_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        user=os.getenv('POSTGRES_USER', 'admin'),
        password=os.getenv('POSTGRES_PASSWORD', 'admin123')
    )

    print("="*80)
    print("UNPROCESSED PAGES ANALYSIS")
    print("="*80)

    # Get all pages
    async with pg_pool.acquire() as conn:
        all_pages = await conn.fetch("""
            SELECT id, url, pub_time, title
            FROM core.pages
            ORDER BY pub_time ASC
        """)

    print(f"\n📄 Total pages: {len(all_pages)}")

    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # Check each page for claims and events
    pages_without_claims = []
    pages_without_events = []
    pages_with_malformed_claims = []

    for page in all_pages:
        page_id = page['id']

        # Check claims
        async with pg_pool.acquire() as conn:
            claims = await conn.fetch("""
                SELECT id, text, modality
                FROM core.claims
                WHERE page_id = $1
            """, page_id)

        if not claims:
            pages_without_claims.append(page)
            continue

        # Check for malformed claims (no entities, no text, etc.)
        malformed = []
        for claim in claims:
            issues = []
            if not claim['text'] or len(claim['text'].strip()) < 10:
                issues.append("text too short")

            # Check entities in Neo4j
            entities = await neo4j._execute_read("""
                MATCH (c:Claim {id: $claim_id})-[:MENTIONS]->(e:Entity)
                RETURN e.id as entity_id
            """, {'claim_id': str(claim['id'])})

            if not entities or len(entities) == 0:
                issues.append("no entities")
            if not claim['modality']:
                issues.append("no modality")

            if issues:
                malformed.append({
                    'claim_id': claim['id'],
                    'text': claim['text'][:50] if claim['text'] else None,
                    'issues': issues
                })

        if malformed:
            pages_with_malformed_claims.append({
                'page': page,
                'malformed_claims': malformed
            })

        # Check if page has events (via claims)
        events = await neo4j._execute_read("""
            MATCH (c:Claim {page_id: $page_id})<-[:HAS_CLAIM]-(e:Event)
            RETURN DISTINCT e.id as event_id, e.canonical_name as name
        """, {'page_id': str(page_id)})

        if not events:
            pages_without_events.append({
                'page': page,
                'claims_count': len(claims),
                'has_malformed': len(malformed) > 0
            })

    # Report results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\n❌ Pages without claims: {len(pages_without_claims)}")
    if pages_without_claims:
        for i, page in enumerate(pages_without_claims[:5], 1):
            title = (page['title'] or 'Untitled')[:60]
            print(f"   {i}. [{page['pub_time']}] {title}")
        if len(pages_without_claims) > 5:
            print(f"   ... and {len(pages_without_claims) - 5} more")

    print(f"\n⚠️  Pages with malformed claims: {len(pages_with_malformed_claims)}")
    if pages_with_malformed_claims:
        for i, item in enumerate(pages_with_malformed_claims[:5], 1):
            page = item['page']
            malformed = item['malformed_claims']
            title = (page['title'] or 'Untitled')[:60]
            print(f"   {i}. [{page['pub_time']}] {title}")
            for claim in malformed[:2]:
                print(f"      - {claim['issues']}: {claim['text']}")
        if len(pages_with_malformed_claims) > 5:
            print(f"   ... and {len(pages_with_malformed_claims) - 5} more")

    print(f"\n📊 Pages without events: {len(pages_without_events)}")
    if pages_without_events:
        for i, item in enumerate(pages_without_events[:10], 1):
            page = item['page']
            claims_count = item['claims_count']
            malformed_marker = " (has malformed claims)" if item['has_malformed'] else ""
            title = (page['title'] or 'Untitled')[:50]
            print(f"   {i}. [{page['pub_time']}] {title} - {claims_count} claims{malformed_marker}")
        if len(pages_without_events) > 10:
            print(f"   ... and {len(pages_without_events) - 10} more")

    # Detailed inspection of first unprocessed page
    if pages_without_events:
        print("\n" + "="*80)
        print("DETAILED INSPECTION - First Unprocessed Page")
        print("="*80)

        first = pages_without_events[0]['page']
        page_id = first['id']

        print(f"\n📄 Page: {first['title']}")
        print(f"   URL: {first['url']}")
        print(f"   Published: {first['pub_time']}")

        # Get all claims
        async with pg_pool.acquire() as conn:
            claims = await conn.fetch("""
                SELECT id, text, modality, event_time
                FROM core.claims
                WHERE page_id = $1
                ORDER BY created_at
            """, page_id)

        print(f"\n📝 Claims ({len(claims)}):")
        for i, claim in enumerate(claims, 1):
            # Get entities from Neo4j
            entities = await neo4j._execute_read("""
                MATCH (c:Claim {id: $claim_id})-[:MENTIONS]->(e:Entity)
                RETURN e.id as entity_id, e.name as name, e.wikidata_qid as qid
            """, {'claim_id': str(claim['id'])})

            entity_count = len(entities) if entities else 0
            time_str = claim['event_time'].strftime('%m-%d %H:%M') if claim['event_time'] else 'no-time'
            print(f"   {i}. [{time_str}] {claim['modality'] or 'NO_MODALITY'} ({entity_count} entities)")
            print(f"      {claim['text'][:100]}...")

            # Show first 3 entities
            if entities:
                for entity in entities[:3]:
                    qid_str = f" [{entity['qid']}]" if entity['qid'] else " [no QID]"
                    print(f"         - {entity['name']}{qid_str}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if pages_without_claims:
        print("\n1. Re-run extraction worker for pages without claims")
        print("   docker exec herenews-api python3 /app/queue_extraction_jobs.py")

    if pages_with_malformed_claims:
        print("\n2. Re-run semantic worker for pages with malformed claims")
        print("   - Delete malformed claims")
        print("   - Re-queue semantic analysis")

    if pages_without_events:
        print(f"\n3. Queue {len(pages_without_events)} pages to event worker")
        print("   python3 queue_pages_chronologically.py")

    await neo4j.close()
    await pg_pool.close()


if __name__ == '__main__':
    asyncio.run(main())
