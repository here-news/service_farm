"""
Test UPDATE vs INSERT for vector embeddings
"""
import asyncio
import asyncpg
import os


async def main():
    pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    test_embedding = [-0.017043846, -0.010912581, 0.028442383] * 512  # 1536 values

    async with pool.acquire() as conn:
        # First create a test page in core.pages
        page_id = 'test_page_vector'
        await conn.execute("""
            INSERT INTO core.pages (id, url, canonical_url, content_text, status)
            VALUES ($1, 'http://test.com', 'http://test.com', 'test', 'completed')
            ON CONFLICT (id) DO NOTHING
        """, page_id)

        # Test UPDATE with list (like knowledge_worker does)
        print("Test UPDATE with list directly (like knowledge_worker)...")
        try:
            await conn.execute("""
                UPDATE core.pages
                SET embedding = $1::vector
                WHERE id = $2
            """, test_embedding, page_id)
            print("  ✅ SUCCESS")

            # Verify it was stored correctly
            row = await conn.fetchrow("SELECT pg_typeof(embedding), vector_dims(embedding) FROM core.pages WHERE id = $1", page_id)
            print(f"  Stored as: type={row['pg_typeof']}, dims={row['vector_dims']}")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")

        # Cleanup
        await conn.execute("DELETE FROM core.pages WHERE id = $1", page_id)

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
