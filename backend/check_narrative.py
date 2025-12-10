"""Check event narrative"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
    neo4j = Neo4jService()
    await neo4j.connect()

    # Get the Wang Fuk Court fire event with narrative
    event_data = await neo4j._execute_read("""
        MATCH (e:Event {id: 'ev_4uvbwao6'})
        RETURN e.id, e.canonical_name, e.summary, e.coherence,
               e.status, e.confidence, e.event_type
    """, {})

    if not event_data:
        print("❌ Event not found")
        await neo4j.close()
        return

    event = event_data[0]

    print("=" * 80)
    print("📰 EVENT NARRATIVE")
    print("=" * 80)
    print(f"Event: {event['e.canonical_name']}")
    print(f"ID: {event['e.id']}")
    print(f"Type: {event.get('e.event_type', 'N/A')}")
    print(f"Status: {event.get('e.status', 'N/A')}")
    print(f"Confidence: {event.get('e.confidence', 'N/A')}")
    print(f"Coherence: {event['e.coherence']:.3f}" if event.get('e.coherence') else "Coherence: Not set")
    print()
    print("=" * 80)
    print("NARRATIVE/SUMMARY:")
    print("=" * 80)

    summary = event.get('e.summary')
    if summary:
        print(summary)
    else:
        print("⚠️  No narrative/summary set")

    print()
    print("=" * 80)

    # Get claim count
    claims = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)
        RETURN count(c) as count
    """, {'event_id': event['e.id']})

    print(f"Total claims: {claims[0]['count']}")

    # Get page count
    pages = await neo4j._execute_read("""
        MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c:Claim)<-[:CONTAINS]-(p:Page)
        RETURN count(DISTINCT p) as count
    """, {'event_id': event['e.id']})

    print(f"Source pages: {pages[0]['count']}")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
