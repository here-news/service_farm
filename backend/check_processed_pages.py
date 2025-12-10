"""
Check which Hong Kong fire pages have been fully processed with claims
"""
import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.neo4j_service import Neo4jService


async def main():
    # Connect to PostgreSQL
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=1,
        max_size=2
    )

    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # Get all pages with claims
    result = await neo4j._execute_read("""
        MATCH (p:Page)-[:CONTAINS]->(c:Claim)
        WITH p, count(c) as claim_count
        RETURN p.id, p.title, claim_count
        ORDER BY claim_count DESC
    """, {})

    print(f"\n{'='*80}")
    print(f"Found {len(result)} pages with claims")
    print(f"{'='*80}\n")

    for row in result:
        page_id = row['p.id']
        title = row['p.title']
        claim_count = row['claim_count']

        # Check if page has embedding
        async with db_pool.acquire() as conn:
            has_embedding = await conn.fetchval("""
                SELECT embedding IS NOT NULL
                FROM core.pages
                WHERE id = $1
            """, page_id)

        status = "✅" if has_embedding else "❌"
        print(f"{status} {page_id}: {title}")
        print(f"   {claim_count} claims, embedding: {has_embedding}")
        print()

    await db_pool.close()
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
