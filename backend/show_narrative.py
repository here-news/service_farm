"""Show full event narrative from metadata"""
import asyncio
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
    neo4j = Neo4jService()
    await neo4j.connect()

    # Get the full event metadata
    event_data = await neo4j._execute_read("""
        MATCH (e:Event {id: 'ev_4uvbwao6'})
        RETURN e.canonical_name, e.coherence, e.metadata
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
    print(f"Coherence: {event['e.coherence']:.3f}")
    print()
    print("=" * 80)

    metadata_str = event['e.metadata']
    try:
        metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str

        if 'summary' in metadata:
            print(metadata['summary'])
        else:
            print("⚠️  No summary in metadata")
            print(f"Metadata keys: {list(metadata.keys())}")
    except Exception as e:
        print(f"❌ Error parsing metadata: {e}")
        print(f"Raw metadata: {metadata_str}")

    print()
    print("=" * 80)

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
