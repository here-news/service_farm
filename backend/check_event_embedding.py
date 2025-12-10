"""
Check event embedding in PostgreSQL
"""
import asyncio
import asyncpg
import os


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

    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT event_id, embedding
            FROM content.event_embeddings
        """)

        for row in rows:
            emb = row['embedding']
            print(f"Event: {row['event_id']}")
            print(f"  Type: {type(emb)}")
            print(f"  Length: {len(emb) if emb else 'None'}")
            if isinstance(emb, list) and len(emb) > 0:
                print(f"  First 3 values: {emb[:3]}")
            print()

    await db_pool.close()


if __name__ == "__main__":
    asyncio.run(main())
