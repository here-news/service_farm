"""
Cleanup: Remove Page nodes from Neo4j

Page nodes with embeddings were incorrectly stored in Neo4j.
This script removes them since:
1. Full page data lives in PostgreSQL (data lake)
2. Embeddings should NEVER be in graph databases
3. Claim.page_id in PostgreSQL provides source tracking
"""
import asyncio
from services.neo4j_service import Neo4jService


async def cleanup_page_nodes():
    """Remove all Page nodes and their relationships from Neo4j"""

    neo4j = Neo4jService()
    await neo4j.connect()

    try:
        # Check current count
        result = await neo4j._execute_read("""
            MATCH (p:Page)
            RETURN count(p) as page_count
        """, {})

        page_count = result[0]['page_count'] if result else 0
        print(f"📊 Found {page_count} Page nodes in Neo4j")

        if page_count == 0:
            print("✅ No Page nodes to clean up")
            return

        # Delete Page nodes and their relationships
        result = await neo4j._execute_write("""
            MATCH (p:Page)
            DETACH DELETE p
        """, {})

        print(f"✅ Removed {page_count} Page nodes from Neo4j")
        print("📝 Page data remains in PostgreSQL core.pages table")

    finally:
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(cleanup_page_nodes())
