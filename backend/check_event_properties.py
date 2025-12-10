"""Check all properties on the event node"""
import asyncio
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
    neo4j = Neo4jService()
    await neo4j.connect()

    # Get the full event node with all properties
    event_data = await neo4j._execute_read("""
        MATCH (e:Event {id: 'ev_4uvbwao6'})
        RETURN e
    """, {})

    if not event_data:
        print("❌ Event not found")
        await neo4j.close()
        return

    event = event_data[0]['e']

    print("=" * 80)
    print("🔍 ALL EVENT PROPERTIES")
    print("=" * 80)

    for key, value in sorted(event.items()):
        if key == 'metadata':
            print(f"\n{key}:")
            try:
                metadata = json.loads(value) if isinstance(value, str) else value
                for mk, mv in metadata.items():
                    if isinstance(mv, str) and len(mv) > 100:
                        print(f"  {mk}: {mv[:100]}... ({len(mv)} chars)")
                    else:
                        print(f"  {mk}: {mv}")
            except:
                print(f"  {value}")
        elif isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}... ({len(value)} chars)")
        else:
            print(f"{key}: {value}")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
