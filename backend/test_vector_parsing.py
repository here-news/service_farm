"""
Test vector parsing from PostgreSQL
"""
import asyncio
import asyncpg
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))


async def main():
    pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    async with pool.acquire() as conn:
        # Test page embedding
        row = await conn.fetchrow('SELECT embedding FROM core.pages WHERE id = $1', 'pg_013v2wny')
        emb_str = row['embedding']

        print(f"Raw type: {type(emb_str)}")
        print(f"Raw length: {len(emb_str)}")
        print(f"First 100 chars: {emb_str[:100]}")
        print()

        # Test parsing
        if isinstance(emb_str, str) and emb_str.startswith('[') and emb_str.endswith(']'):
            parsed = [float(x.strip()) for x in emb_str[1:-1].split(',')]
            print(f"Parsed length: {len(parsed)}")
            print(f"First 5: {parsed[:5]}")
        else:
            print("Not in expected format")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
