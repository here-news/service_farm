"""
Check where claims are stored
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
    neo4j = Neo4jService()
    await neo4j.connect()

    # Check claims in Neo4j
    result = await neo4j._execute_read("""
        MATCH (c:Claim)
        RETURN count(c) as claim_count
    """, {})

    print(f"Claims in Neo4j: {result[0]['claim_count']}")

    # Sample some claim IDs
    result = await neo4j._execute_read("""
        MATCH (c:Claim)
        RETURN c.id, c.text
        LIMIT 3
    """, {})

    print("\nSample claims:")
    for row in result:
        print(f"  {row['c.id']}: {row['c.text'][:50]}...")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
