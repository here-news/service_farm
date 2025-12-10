#!/usr/bin/env python3
"""
Backfill page embeddings for existing pages.

For pages that have claims but no embeddings, generate embeddings from claim texts.
This enables semantic similarity matching for event routing.
"""
import asyncio
import asyncpg
import os
from openai import AsyncOpenAI
from neo4j import AsyncGraphDatabase
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def fetch_page_claims(neo4j_driver, page_id: str):
    """Fetch all claims for a page from Neo4j."""
    async with neo4j_driver.session() as session:
        result = await session.run("""
            MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
            RETURN c.text as text
            ORDER BY c.created_at
        """, page_id=page_id)

        records = await result.values()
        return [record[0] for record in records if record[0]]


async def generate_embedding(openai_client, texts):
    """Generate embedding from list of texts."""
    if not texts:
        return None

    # Combine texts with newlines
    combined_text = "\n".join(texts)

    # Truncate if too long
    if len(combined_text) > 8000:
        combined_text = combined_text[:8000] + "..."

    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=combined_text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return None


async def store_embedding(pg_conn, page_id: str, embedding):
    """Store embedding in PostgreSQL core.pages table."""
    try:
        # Convert list to vector format string
        vector_str = f"[{','.join(map(str, embedding))}]"

        await pg_conn.execute("""
            UPDATE core.pages
            SET embedding = $1::vector,
                updated_at = NOW()
            WHERE id = $2
        """, vector_str, page_id)
        return True
    except Exception as e:
        logger.error(f"Failed to store embedding for {page_id}: {e}")
        return False


async def backfill():
    """Main backfill function."""
    # Connect to PostgreSQL
    pg_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews')
    )

    # Connect to Neo4j
    neo4j_driver = AsyncGraphDatabase.driver(
        os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
        auth=(os.getenv('NEO4J_USER', 'neo4j'),
              os.getenv('NEO4J_PASSWORD', 'herenews_neo4j_pass'))
    )

    # Connect to OpenAI
    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    try:
        # Get all pages from Neo4j (short IDs)
        async with neo4j_driver.session() as session:
            result = await session.run("""
                MATCH (p:Page)-[:CONTAINS]->(c:Claim)
                WITH p, count(c) as claim_count
                WHERE claim_count > 0
                RETURN p.id as page_id, claim_count
                ORDER BY claim_count DESC
            """)
            pages = await result.values()

        logger.info(f"Found {len(pages)} pages with claims")

        # Process each page
        for page_id, claim_count in pages:
            logger.info(f"Processing {page_id} ({claim_count} claims)...")

            # Check if embedding already exists in core.pages
            async with pg_pool.acquire() as conn:
                has_embedding = await conn.fetchval("""
                    SELECT embedding IS NOT NULL
                    FROM core.pages
                    WHERE id = $1
                """, page_id)

                if has_embedding:
                    logger.info(f"  ✓ Already has embedding")
                    continue

                # Fetch claims from Neo4j
                claim_texts = await fetch_page_claims(neo4j_driver, page_id)
                logger.info(f"  📝 Fetched {len(claim_texts)} claim texts")

                if not claim_texts:
                    logger.warning(f"  ⚠️ No claim texts found")
                    continue

                # Generate embedding
                embedding = await generate_embedding(openai_client, claim_texts)

                if not embedding:
                    logger.error(f"  ❌ Failed to generate embedding")
                    continue

                logger.info(f"  📊 Generated embedding ({len(embedding)} dims)")

                # Store embedding directly using short ID
                success = await store_embedding(conn, page_id, embedding)

                if success:
                    logger.info(f"  ✅ Stored embedding")
                else:
                    logger.error(f"  ❌ Failed to store embedding")

        logger.info("✅ Backfill complete!")

    finally:
        await pg_pool.close()
        await neo4j_driver.close()


if __name__ == "__main__":
    asyncio.run(backfill())
