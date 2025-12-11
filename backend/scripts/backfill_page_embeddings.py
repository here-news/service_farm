"""
Backfill Page Embeddings

Generates embeddings for pages that are missing them.
Uses the same logic as KnowledgeWorker STAGE 4e.
"""
import asyncio
import asyncpg
import os
import sys
from openai import AsyncOpenAI
import logging

from pgvector.asyncpg import register_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, '/app')

from services.neo4j_service import Neo4jService


async def get_pages_without_embeddings(conn) -> list:
    """Get pages that have content but no embedding"""
    rows = await conn.fetch("""
        SELECT id, content_text
        FROM core.pages
        WHERE embedding IS NULL
          AND content_text IS NOT NULL
          AND length(content_text) > 100
          AND status IN ('knowledge_complete', 'event_processed')
        ORDER BY updated_at DESC
        LIMIT 100
    """)
    return rows


async def get_claims_for_page(neo4j: Neo4jService, page_id: str) -> list:
    """Get claim texts for a page from Neo4j"""
    result = await neo4j._execute_read("""
        MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
        RETURN c.text as text
    """, {'page_id': page_id})
    return [r['text'] for r in result if r['text']]


async def generate_embedding(client: AsyncOpenAI, texts: list) -> list:
    """Generate embedding from text"""
    combined = "\n".join(texts)
    if len(combined) > 8000:
        combined = combined[:8000] + "..."

    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=combined
    )
    return response.data[0].embedding


async def backfill():
    """Main backfill function"""
    # Connect to PostgreSQL
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=2,
        max_size=5
    )

    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # OpenAI client
    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    try:
        async with db_pool.acquire() as conn:
            await register_vector(conn)

            pages = await get_pages_without_embeddings(conn)
            logger.info(f"Found {len(pages)} pages without embeddings")

            for i, page in enumerate(pages):
                page_id = page['id']

                # Get claims for this page
                claims = await get_claims_for_page(neo4j, page_id)

                if not claims:
                    # Fall back to content text
                    text = page['content_text']
                    if text:
                        claims = [text[:4000]]

                if not claims:
                    logger.warning(f"Skipping {page_id}: no text available")
                    continue

                try:
                    # Generate embedding
                    embedding = await generate_embedding(openai_client, claims)

                    # Store using pgvector native type
                    await conn.execute("""
                        UPDATE core.pages
                        SET embedding = $1,
                            updated_at = NOW()
                        WHERE id = $2
                    """, embedding, page_id)

                    logger.info(f"[{i+1}/{len(pages)}] ✅ {page_id} - {len(claims)} claims")

                except Exception as e:
                    logger.error(f"[{i+1}/{len(pages)}] ❌ {page_id}: {e}")

                # Rate limit
                await asyncio.sleep(0.2)

    finally:
        await db_pool.close()
        await neo4j.close()

    logger.info("Backfill complete!")


if __name__ == "__main__":
    asyncio.run(backfill())
