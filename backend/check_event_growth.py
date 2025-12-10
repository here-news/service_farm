"""
Check if event matched and grew
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
    neo4j = Neo4jService()
    await neo4j.connect()

    # Get all events
    result = await neo4j._execute_read("""
        MATCH (e:Event)
        RETURN e.id, e.canonical_name
    """, {})

    print(f"Found {len(result)} event(s)\n")

    for row in result:
        event_id = row['e.id']
        event_name = row['e.canonical_name']

        # Count claims
        claims = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
            RETURN count(c) as claim_count
        """, {'event_id': event_id})

        # Get pages
        pages = await neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)-[:FROM]->(p:Page)
            RETURN DISTINCT p.id as page_id
        """, {'event_id': event_id})

        print(f"Event: {event_name} ({event_id})")
        print(f"  Claims: {claims[0]['claim_count']}")
        print(f"  Pages: {len(pages)}")
        for page in pages:
            print(f"    - {page['page_id']}")
        print()

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
