#!/usr/bin/env python3
"""
Simple URL resolver using only standard library
"""
import urllib.request
import urllib.error
from urllib.parse import urlparse

def resolve_url(url: str, timeout: int = 10) -> str:
    """
    Resolve URL to final destination after redirects

    Args:
        url: URL to resolve (Google News or direct)
        timeout: Request timeout

    Returns:
        Final URL after following redirects
    """
    try:
        # Convert RSS to web URL if needed
        if '/rss/articles/' in url:
            url = url.replace('/rss/articles/', '/articles/')

        # Create request with browser-like headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

        request = urllib.request.Request(url, headers=headers)

        # Open URL and follow redirects
        with urllib.request.urlopen(request, timeout=timeout) as response:
            final_url = response.geturl()

            # Return the final URL
            return final_url

    except Exception as e:
        print(f"Error resolving {url}: {e}")
        return url


def process_feed_urls(json_file: str, output_file: str):
    """Process feed URLs from JSON and resolve them"""
    import json

    with open(json_file, 'r') as f:
        items = json.load(f)

    results = []
    for i, item in enumerate(items, 1):
        print(f"Resolving {i}/{len(items)}: {item['title'][:50]}...")

        resolved_url = resolve_url(item['google_news_url'])

        results.append({
            'timestamp': item['timestamp'],
            'title': item['title'],
            'google_news_url': item['google_news_url'],
            'publisher_url': resolved_url,
            'is_resolved': 'google.com' not in resolved_url
        })

        # Rate limit
        import time
        time.sleep(1)

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    resolved_count = sum(1 for r in results if r['is_resolved'])
    print(f"\nResolved {resolved_count}/{len(results)} URLs successfully")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    # Test with first item
    import json

    with open('feed_urls.json', 'r') as f:
        items = json.load(f)

    if items:
        test_url = items[0]['google_news_url']
        print(f"Testing URL resolution...")
        print(f"Original: {test_url}")

        resolved = resolve_url(test_url)
        print(f"Resolved: {resolved}")

        if 'google.com' not in resolved:
            print("\n✓ Success! URL resolved to publisher site")
        else:
            print("\n✗ Still on Google - may need browser automation")
