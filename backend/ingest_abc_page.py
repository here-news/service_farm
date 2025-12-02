"""
Manually ingest ABC News page to demonstrate event evolution
"""
import asyncio
import asyncpg
import os
from test_event_network import EventNetworkBuilder
from openai import AsyncOpenAI
import httpx
from bs4 import BeautifulSoup


async def fetch_and_process_abc():
    """Fetch ABC News page and process through event network"""

    url = "https://abcnews.go.com/International/massive-fire-engulfs-hong-kong-high-rise-apartment/story?id=127887923"

    # Fetch page
    print(f"📥 Fetching: {url}\n")
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        response = await client.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        html = response.text

    # Extract metadata
    soup = BeautifulSoup(html, 'html.parser')

    title = soup.find('meta', property='og:title')
    title = title['content'] if title else soup.find('title').text if soup.find('title') else "Untitled"

    description = soup.find('meta', property='og:description')
    description = description['content'] if description else ""

    # Extract main content
    article = soup.find('article')
    if article:
        # Get all paragraph text
        paragraphs = article.find_all('p')
        content_text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
    else:
        content_text = soup.get_text()[:5000]  # Fallback

    print(f"✅ Fetched page:")
    print(f"   Title: {title}")
    print(f"   Description: {description[:100]}...")
    print(f"   Content length: {len(content_text)} chars\n")

    # Connect to DB
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=1, max_size=2
    )

    # Check if page already exists
    async with db_pool.acquire() as conn:
        existing_page = await conn.fetchrow("""
            SELECT id, embedding FROM core.pages WHERE url = $1
        """, url)

    if existing_page:
        print(f"⚠️  Page already exists: {existing_page['id']}")
        page_id = existing_page['id']
        embedding = existing_page['embedding']
    else:
        # Create page record (without embedding for now)
        async with db_pool.acquire() as conn:
            page_id = await conn.fetchval("""
                INSERT INTO core.pages (url, title, description, content_text, created_at)
                VALUES ($1, $2, $3, $4, NOW())
                RETURNING id
            """, url, title, description, content_text)

        print(f"✅ Created page: {page_id}\n")

        # Generate embedding
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.embeddings.create(
            input=f"{title}\n{description}",
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        # Store embedding
        async with db_pool.acquire() as conn:
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            await conn.execute("""
                UPDATE core.pages
                SET embedding = $2::vector
                WHERE id = $1
            """, page_id, embedding_str)

        print(f"✅ Generated embedding\n")

    # Process through event network
    print(f"{'='*80}")
    print("PROCESSING THROUGH EVENT NETWORK")
    print(f"{'='*80}\n")

    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    builder = EventNetworkBuilder(db_pool, openai_client)

    page_data = {
        'id': page_id,
        'title': title,
        'description': description,
        'content_text': content_text,
        'url': url,
        'embedding': embedding,
        'created_at': 'now'
    }

    event_id = await builder.process_page(page_data)

    print(f"\n{'='*80}")
    print(f"RESULT")
    print(f"{'='*80}\n")
    print(f"Event ID: {event_id}")

    # Show updated event
    async with db_pool.acquire() as conn:
        event = await conn.fetchrow("""
            SELECT title, coherence,
                   (SELECT COUNT(*) FROM core.page_events WHERE event_id = e.id) as page_count
            FROM core.events e
            WHERE id = $1
        """, event_id)

        print(f"Event: {event['title']}")
        print(f"Coherence: {event['coherence']:.3f}")
        print(f"Total pages: {event['page_count']}")

    await db_pool.close()


if __name__ == '__main__':
    asyncio.run(fetch_and_process_abc())
