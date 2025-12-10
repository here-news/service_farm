"""Debug claims and their entities"""
import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
    neo4j = Neo4jService()
    await neo4j.connect()

    # Check claims and their entity relationships
    result = await neo4j._execute_read("""
        MATCH (e:Event {id: 'ev_4uvbwao6'})-[:SUPPORTS]->(c:Claim)
        WITH c LIMIT 5
        OPTIONAL MATCH (c)-[:MENTIONS]->(ent:Entity)
        RETURN c.id, c.text, collect(ent.id) as entity_ids, collect(ent.name) as entity_names
    """, {})

    print("Sample claims from event:")
    for row in result:
        print(f"\nClaim: {row['c.id']}")
        print(f"  Text: {row['c.text'][:100]}...")
        print(f"  Entity IDs: {row['entity_ids']}")
        print(f"  Entity names: {row['entity_names']}")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
