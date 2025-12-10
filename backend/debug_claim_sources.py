"""
Debug where claims came from
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
    neo4j = Neo4jService()
    await neo4j.connect()

    # Get event
    event = await neo4j._execute_read("""
        MATCH (e:Event)
        RETURN e.id as id, e.canonical_name as name
    """, {})

    if not event:
        print("No events found")
        return

    event_id = event[0]['id']

    print("=" * 80)
    print(f"Event: {event[0]['name']} ({event_id})")
    print("=" * 80)
    print()

    # Get all claims in this event
    claims = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        RETURN c.id as claim_id
    """, {'event_id': event_id})

    print(f"Total claims in event: {len(claims)}")
    print()

    # For each claim, find its source page
    print("🔍 Finding source pages for each claim:")
    print()

    source_pages = {}
    claims_without_page = []

    for claim in claims:
        claim_id = claim['claim_id']

        # Find page via CONTAINS relationship
        page = await neo4j._execute_read("""
            MATCH (p:Page)-[:CONTAINS]->(c:Claim {id: $claim_id})
            RETURN p.id as page_id
        """, {'claim_id': claim_id})

        if page:
            page_id = page[0]['page_id']
            if page_id not in source_pages:
                source_pages[page_id] = []
            source_pages[page_id].append(claim_id)
        else:
            claims_without_page.append(claim_id)

    print(f"📄 Claims grouped by source page:")
    for page_id, claim_list in sorted(source_pages.items()):
        print(f"   {page_id}: {len(claim_list)} claims")

    if claims_without_page:
        print(f"\n⚠️  {len(claims_without_page)} claims have NO source page:")
        for claim_id in claims_without_page[:5]:
            print(f"   - {claim_id}")
        if len(claims_without_page) > 5:
            print(f"   ... and {len(claims_without_page) - 5} more")

    print()
    print(f"📊 Summary:")
    print(f"   Source pages with claims: {len(source_pages)}")
    print(f"   Claims without page: {len(claims_without_page)}")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
