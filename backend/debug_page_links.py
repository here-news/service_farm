"""
Debug page-to-claim relationships
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
    neo4j = Neo4jService()
    await neo4j.connect()

    print("=" * 80)
    print("🔍 DEBUGGING PAGE-CLAIM RELATIONSHIPS")
    print("=" * 80)
    print()

    # Get event
    event = await neo4j._execute_read("""
        MATCH (e:Event)
        RETURN e.id as event_id, e.canonical_name as name
    """, {})

    if not event:
        print("No events found")
        return

    event_id = event[0]['event_id']
    print(f"Event: {event[0]['name']} ({event_id})")
    print()

    # Check all claims for this event
    claims = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        RETURN c.id as claim_id, c.page_id as page_id, c.text as text
        ORDER BY c.page_id
    """, {'event_id': event_id})

    print(f"Total claims: {len(claims)}")
    print()

    # Group by page_id
    by_page = {}
    for claim in claims:
        page_id = claim['page_id']
        if page_id not in by_page:
            by_page[page_id] = []
        by_page[page_id].append(claim['claim_id'])

    print(f"📄 Claims grouped by page_id:")
    for page_id, claim_ids in sorted(by_page.items()):
        print(f"   {page_id}: {len(claim_ids)} claims")
    print()

    # Check if Page nodes exist
    print(f"📊 Checking Page nodes in Neo4j:")
    pages = await neo4j._execute_read("""
        MATCH (p:Page)
        RETURN p.id as page_id, p.title as title
        ORDER BY p.id
    """, {})

    print(f"   Total Page nodes: {len(pages)}")
    for page in pages:
        print(f"      {page['page_id']}: {page['title']}")
    print()

    # Check Page-Claim relationships
    print(f"🔗 Checking Page-[:CONTAINS]->Claim relationships:")
    page_claim_rels = await neo4j._execute_read("""
        MATCH (p:Page)-[:CONTAINS]->(c:Claim)
        RETURN p.id as page_id, count(c) as claim_count
        ORDER BY p.id
    """, {})

    print(f"   Found {len(page_claim_rels)} pages with CONTAINS relationships")
    for rel in page_claim_rels:
        print(f"      {rel['page_id']}: {rel['claim_count']} claims")
    print()

    # Now check for our test pages specifically
    test_pages = [
        'pg_013v2wny',
        'pg_00prszmp',
        'pg_013ks2k5',
        'pg_00zbqg7h',
        'pg_01lnezb0',
        'pg_00r7u1zt'
    ]

    print(f"🧪 Checking test pages in Neo4j:")
    for page_id in test_pages:
        page_exists = await neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})
            RETURN p.id as id, p.title as title
        """, {'page_id': page_id})

        if page_exists:
            # Check claims from this page
            claims_from_page = await neo4j._execute_read("""
                MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
                RETURN count(c) as count
            """, {'page_id': page_id})

            # Check if any claims are linked to our event
            event_claims = await neo4j._execute_read("""
                MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
                WHERE c.page_id = $page_id
                RETURN count(c) as count
            """, {'event_id': event_id, 'page_id': page_id})

            print(f"   ✓ {page_id}: {page_exists[0]['title']}")
            print(f"      CONTAINS claims: {claims_from_page[0]['count']}")
            print(f"      Claims in event: {event_claims[0]['count']}")
        else:
            print(f"   ✗ {page_id}: NOT FOUND in Neo4j")
        print()

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
