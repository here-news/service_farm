"""
Test multi-method extraction on difficult URLs
"""
import asyncio
import httpx
from services.multi_extractor import MultiMethodExtractor


async def test_extraction(url: str):
    """Test extraction on a URL"""
    print(f"\n{'='*80}")
    print(f"Testing: {url}")
    print(f"{'='*80}\n")

    # Fetch HTML
    print("📥 Fetching HTML...")
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        try:
            response = await client.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; HereNewsBot/2.0; +https://here.news)'
            })

            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code}")
                return

            html = response.text
            print(f"✅ Fetched {len(html)} bytes\n")

        except Exception as e:
            print(f"❌ Fetch failed: {e}")
            return

    # Try extraction with multiple methods
    extractor = MultiMethodExtractor(min_words=100)
    result = await extractor.extract(url, html)

    if result.success:
        print(f"\n✅ SUCCESS")
        print(f"   Method: {result.method_used}")
        print(f"   Words: {result.word_count}")
        if result.top_image:
            print(f"   Top image: {result.top_image}")
        if result.images:
            print(f"   Total images: {len(result.images)}")
            for i, img in enumerate(result.images[:3]):  # Show first 3
                print(f"      - {img}")
            if len(result.images) > 3:
                print(f"      ... and {len(result.images) - 3} more")
        print(f"   Preview: {result.content[:200]}...")
    else:
        print(f"\n❌ FAILED")
        print(f"   Last method: {result.method_used}")
        print(f"   Error: {result.error_message}")
        if result.content:
            print(f"   Partial content ({result.word_count} words): {result.content[:200]}...")


async def main():
    """Test on difficult URLs"""

    # Test URLs
    urls = [
        # Reuters - blocks most scrapers (403)
        "https://www.reuters.com/business/aerospace-defense/ukrainian-delegation-heads-us-peace-talks-after-lead-negotiators-exit-2025-11-29/",

        # USA Today - should work with any method
        "https://www.usatoday.com/story/money/2025/11/29/buy-american-manufacturers-household-goods-tariffs-imports/86935613007/",

        # ESPN - JavaScript heavy
        "https://www.espn.com/nfl/story/_/id/47097079/team-djs-nba-youngboy-lions-jets-breece-hall-49ers-jets-bills-josh-allen",
    ]

    for url in urls:
        await test_extraction(url)
        await asyncio.sleep(1)  # Be polite


if __name__ == "__main__":
    asyncio.run(main())
