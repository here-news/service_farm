"""List existing events"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
    neo4j = Neo4jService()
    await neo4j.connect()

    events = await neo4j._execute_read("""
        MATCH (e:Event)
        RETURN e.id, e.canonical_name, e.coherence
        LIMIT 10
    """, {})

    print(f"Found {len(events)} events:")
    for event in events:
        coherence = event.get('e.coherence', None)
        coh_str = f"{coherence:.3f}" if coherence else "None"
        print(f"  {event['e.canonical_name'][:50]}: {event['e.id']} (coherence={coh_str})")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
