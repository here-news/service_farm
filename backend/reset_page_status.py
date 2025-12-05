#!/usr/bin/env python3
"""Reset page status to stub for re-extraction"""
import asyncio
import asyncpg
import os

async def reset_page_status(page_id: str):
    """Reset page status to stub"""
    pool = await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
        database=os.getenv("POSTGRES_DB", "herenews"),
        min_size=1,
        max_size=2
    )

    async with pool.acquire() as conn:
        # Reset status to stub
        await conn.execute("""
            UPDATE core.pages
            SET status = 'stub',
                word_count = NULL,
                content_text = NULL,
                error_message = NULL
            WHERE id = $1
        """, page_id)

        # Verify
        result = await conn.fetch("""
            SELECT id, status, word_count
            FROM core.pages
            WHERE id = $1
        """, page_id)

        print(f"Updated page {page_id}:")
        for row in result:
            print(f"  Status: {row['status']}, Word count: {row['word_count']}")

    await pool.close()

if __name__ == "__main__":
    asyncio.run(reset_page_status("4fa8ae71-9e10-4072-b85b-0b18c2a0f42b"))
