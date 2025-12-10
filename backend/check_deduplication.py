"""
Check claim deduplication across pages
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
    neo4j = Neo4jService()
    await neo4j.connect()

    test_pages = [
        ('pg_013v2wny', 'DW'),
        ('pg_00prszmp', 'Fox'),
        ('pg_006iquvd', 'Christianity Today'),
        ('pg_01wzjkk9', 'Newsweek'),
        ('pg_01euzt1r', 'NY Post')
    ]

    print("=" * 80)
    print("📊 CLAIM DEDUPLICATION ANALYSIS")
    print("=" * 80)
    print()

    event = await neo4j._execute_read("MATCH (e:Event) RETURN e.id as id", {})
    event_id = event[0]['id'] if event else None

    for page_id, source in test_pages:
        # Total claims from this page
        total_claims = await neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
            RETURN count(c) as count
        """, {'page_id': page_id})

        # Claims from this page that made it into the event
        if event_id:
            event_claims = await neo4j._execute_read("""
                MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)<-[:CONTAINS]-(p:Page {id: $page_id})
                RETURN count(c) as count
            """, {'event_id': event_id, 'page_id': page_id})
            in_event = event_claims[0]['count']
        else:
            in_event = 0

        total = total_claims[0]['count']
        duplicates = total - in_event

        print(f"📄 {page_id} ({source})")
        print(f"   Total claims: {total}")
        print(f"   In event: {in_event}")
        print(f"   Duplicates: {duplicates} ({duplicates/total*100:.1f}%)" if total > 0 else "   Duplicates: 0")
        print()

    # Event summary
    if event_id:
        event_claims = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
            RETURN count(c) as count
        """, {'event_id': event_id})

        print("=" * 80)
        print(f"📈 EVENT SUMMARY")
        print(f"   Total unique claims: {event_claims[0]['count']}")
        print(f"   Source pages: {len(test_pages)}")

        # Calculate total claims across all pages
        total_across_pages = 0
        for page_id, _ in test_pages:
            result = await neo4j._execute_read("""
                MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
                RETURN count(c) as count
            """, {'page_id': page_id})
            total_across_pages += result[0]['count']

        duplicates_removed = total_across_pages - event_claims[0]['count']
        print(f"   Total claims from all pages: {total_across_pages}")
        print(f"   Duplicates removed: {duplicates_removed} ({duplicates_removed/total_across_pages*100:.1f}%)")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
