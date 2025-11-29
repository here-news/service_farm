#!/usr/bin/env python3
"""
Bulk import articles from feed_urls.json into your system
The extraction worker will resolve Google News URLs naturally when fetching content
"""
import json
import asyncio
import httpx

async def import_article(article_data, api_base_url="http://localhost:8000"):
    """Import one article into the system via /artifacts/draft endpoint"""

    # Use the Google News URL directly
    # Your extraction worker will follow the redirect when fetching content
    url = article_data['google_news_url']

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{api_base_url}/artifacts/draft",
                params={"url": url}
            )
            response.raise_for_status()
            result = response.json()

            return {
                'success': True,
                'artifact_id': result.get('artifact_id'),
                'title': article_data['title'],
                'timestamp': article_data['timestamp']
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'title': article_data['title']
            }

async def bulk_import(input_json, max_articles=None, concurrent=5):
    """Import multiple articles concurrently"""

    # Load articles
    with open(input_json, 'r') as f:
        articles = json.load(f)

    if max_articles:
        articles = articles[:max_articles]

    print(f"Importing {len(articles)} articles...")
    print(f"Concurrency: {concurrent} at a time\n")

    # Process in batches
    results = []
    for i in range(0, len(articles), concurrent):
        batch = articles[i:i+concurrent]
        batch_num = (i // concurrent) + 1

        print(f"Batch {batch_num}: Processing {len(batch)} articles...")

        tasks = [import_article(article) for article in batch]
        batch_results = await asyncio.gather(*tasks)

        results.extend(batch_results)

        # Show results
        for r in batch_results:
            if r['success']:
                print(f"  ✓ {r['title'][:60]} -> {r['artifact_id']}")
            else:
                print(f"  ✗ {r['title'][:60]} -> {r['error']}")

        # Delay between batches
        if i + concurrent < len(articles):
            await asyncio.sleep(2)

    # Summary
    success_count = sum(1 for r in results if r['success'])
    print(f"\n{'='*70}")
    print(f"Imported {success_count}/{len(articles)} articles successfully")
    print(f"{'='*70}")

    return results

async def main():
    """Import articles from feed"""
    import sys

    max_articles = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    if max_articles == 5:
        print("Testing with first 5 articles")
        print("To import more: python3 bulk_import_articles.py 50")
        print("To import all: python3 bulk_import_articles.py 100\n")

    results = await bulk_import('feed_urls.json', max_articles=max_articles)

    # Save results
    with open('import_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: import_results.json")

if __name__ == "__main__":
    asyncio.run(main())
