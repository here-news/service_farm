"""
Test different formats for inserting vector embeddings
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

    # Sample embedding (first 3 values from real embedding)
    test_embedding = [-0.017043846, -0.010912581, 0.028442383] * 512  # 1536 values

    async with pool.acquire() as conn:
        # Test 1: Pass list directly
        print("Test 1: Passing list directly...")
        try:
            await conn.execute("""
                INSERT INTO content.event_embeddings (event_id, embedding)
                VALUES ($1, $2::vector)
            """, 'test_1', test_embedding)
            print("  ✅ SUCCESS")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")

        # Test 2: Pass as string with brackets
        print("\nTest 2: Passing as string '[...]'...")
        try:
            embedding_str = '[' + ','.join(str(x) for x in test_embedding) + ']'
            await conn.execute("""
                INSERT INTO content.event_embeddings (event_id, embedding)
                VALUES ($1, $2::vector)
            """, 'test_2', embedding_str)
            print("  ✅ SUCCESS")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")

        # Test 3: Pass as Python str(list)
        print("\nTest 3: Passing as str(list)...")
        try:
            await conn.execute("""
                INSERT INTO content.event_embeddings (event_id, embedding)
                VALUES ($1, $2::vector)
            """, 'test_3', str(test_embedding))
            print("  ✅ SUCCESS")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")

        # Test 4: No cast, just vector column
        print("\nTest 4: No ::vector cast...")
        try:
            await conn.execute("""
                INSERT INTO content.event_embeddings (event_id, embedding)
                VALUES ($1, $2)
            """, 'test_4', test_embedding)
            print("  ✅ SUCCESS")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")

        # Check what worked
        print("\n\nChecking stored embeddings:")
        rows = await conn.fetch("SELECT event_id, pg_typeof(embedding), vector_dims(embedding) FROM content.event_embeddings WHERE event_id LIKE 'test_%'")
        for row in rows:
            print(f"  {row['event_id']}: type={row['pg_typeof']}, dims={row['vector_dims']}")

        # Cleanup
        await conn.execute("DELETE FROM content.event_embeddings WHERE event_id LIKE 'test_%'")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
