"""
Check all Hong Kong fire pages in PostgreSQL
"""
import asyncio
import asyncpg
import os


async def main():
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432,
        user='herenews_user',
        password='herenews_pass',
        database='herenews'
    )

    # Get all pages matching Hong Kong fire
    async with db_pool.acquire() as conn:
        pages = await conn.fetch("""
            SELECT id, url, status, title,
                   content_text IS NOT NULL as has_content,
                   embedding IS NOT NULL as has_embedding,
                   created_at, updated_at
            FROM core.pages
            WHERE url LIKE '%hong-kong%fire%'
               OR url LIKE '%fire%hong-kong%'
               OR url LIKE '%tai-po%'
               OR url LIKE '%wang-fuk%'
            ORDER BY created_at
        """)

    print("=" * 80)
    print("📄 ALL HONG KONG FIRE PAGES IN POSTGRESQL")
    print("=" * 80)
    print()

    if not pages:
        print("❌ No pages found")
    else:
        print(f"Found {len(pages)} page(s):\n")
        for page in pages:
            print(f"ID: {page['id']}")
            print(f"URL: {page['url'][:70]}...")
            print(f"Status: {page['status']}")
            print(f"Title: {page['title']}")
            print(f"Has content: {page['has_content']}")
            print(f"Has embedding: {page['has_embedding']}")
            print(f"Created: {page['created_at']}")
            print(f"Updated: {page['updated_at']}")
            print()

    # Check our specific test pages
    test_pages = [
        'pg_013v2wny',
        'pg_00prszmp',
        'pg_013ks2k5',
        'pg_00zbqg7h',
        'pg_01lnezb0',
        'pg_00r7u1zt'
    ]

    print("=" * 80)
    print("🧪 TEST PAGES STATUS")
    print("=" * 80)
    print()

    async with db_pool.acquire() as conn:
        for page_id in test_pages:
            result = await conn.fetchrow("""
                SELECT id, url, status,
                       content_text IS NOT NULL as has_content,
                       length(content_text) as content_length
                FROM core.pages
                WHERE id = $1
            """, page_id)

            if result:
                print(f"✓ {page_id}")
                print(f"  Status: {result['status']}")
                print(f"  URL: {result['url'][:60]}...")
                print(f"  Content: {result['content_length']} chars")
            else:
                print(f"✗ {page_id} - NOT IN DATABASE")
            print()

    await db_pool.close()


if __name__ == "__main__":
    asyncio.run(main())
