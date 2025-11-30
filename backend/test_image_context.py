"""
Test extracting images with their context (alt text, captions)
"""
import asyncio
import httpx
from bs4 import BeautifulSoup


async def test_image_context(url: str):
    """Extract images with context from a URL"""
    print(f"\nTesting: {url}")
    print("=" * 80)

    # Fetch HTML
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        response = await client.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; HereNewsBot/2.0; +https://here.news)'
        })
        html = response.text

    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')

    # Find all images in article content
    # Look for common article containers
    article = soup.find('article') or soup.find('div', class_=lambda x: x and ('article' in x.lower() or 'content' in x.lower()))

    if not article:
        article = soup  # Fallback to whole page

    images = []
    for img in article.find_all('img'):
        src = img.get('src') or img.get('data-src')
        if not src or 'tracking' in src or 'pixel' in src or '.gif' in src:
            continue  # Skip tracking pixels

        # Extract context
        alt = img.get('alt', '')
        title = img.get('title', '')

        # Look for caption in nearby elements
        caption = ''
        if img.parent:
            # Check for figcaption
            figcaption = img.parent.find('figcaption')
            if figcaption:
                caption = figcaption.get_text(strip=True)
            # Check for caption class
            elif img.parent.find(class_=lambda x: x and 'caption' in x.lower()):
                caption_elem = img.parent.find(class_=lambda x: x and 'caption' in x.lower())
                caption = caption_elem.get_text(strip=True)

        images.append({
            'src': src,
            'alt': alt,
            'title': title,
            'caption': caption
        })

    print(f"\nFound {len(images)} images with context:\n")
    for i, img in enumerate(images[:5], 1):  # Show first 5
        print(f"{i}. {img['src'][:80]}")
        if img['alt']:
            print(f"   Alt: {img['alt']}")
        if img['title']:
            print(f"   Title: {img['title']}")
        if img['caption']:
            print(f"   Caption: {img['caption'][:200]}")
        print()


async def main():
    urls = [
        "https://www.usatoday.com/story/money/2025/11/29/buy-american-manufacturers-household-goods-tariffs-imports/86935613007/",
        "https://www.espn.com/nfl/story/_/id/47097079/team-djs-nba-youngboy-lions-jets-breece-hall-49ers-jets-bills-josh-allen",
    ]

    for url in urls:
        await test_image_context(url)
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
