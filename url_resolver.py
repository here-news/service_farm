#!/usr/bin/env python3
"""
URL resolver that handles Google News redirects
Integrate this into your extraction pipeline
"""
import httpx
import asyncio
from urllib.parse import urlparse

async def resolve_google_news_url(url: str, timeout: int = 10) -> str:
    """
    Resolve Google News URL to actual publisher URL

    Args:
        url: Google News URL (can be RSS or web format)
        timeout: Request timeout in seconds

    Returns:
        Real publisher URL after redirect, or original URL if resolution fails
    """
    try:
        # Convert RSS to web URL if needed
        if '/rss/articles/' in url:
            url = url.replace('/rss/articles/', '/articles/')

        # Use httpx with browser-like headers
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://news.google.com/',
            }

            # HEAD request first (faster)
            try:
                response = await client.head(url, headers=headers)
                final_url = str(response.url)
            except:
                # Fall back to GET if HEAD fails
                response = await client.get(url, headers=headers)
                final_url = str(response.url)

            # Check if we actually got redirected away from Google
            if 'google.com' not in final_url:
                return final_url
            else:
                # Still on Google, return original
                return url

    except Exception as e:
        print(f"Error resolving URL {url}: {e}")
        return url


async def batch_resolve_urls(urls: list[str], max_concurrent: int = 5) -> dict[str, str]:
    """
    Resolve multiple Google News URLs concurrently

    Args:
        urls: List of Google News URLs
        max_concurrent: Maximum concurrent requests

    Returns:
        Dict mapping original URL -> resolved URL
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def resolve_with_limit(url):
        async with semaphore:
            resolved = await resolve_google_news_url(url)
            return url, resolved

    tasks = [resolve_with_limit(url) for url in urls]
    results = await asyncio.gather(*tasks)

    return dict(results)


# Example usage
async def main():
    """Test URL resolution"""
    test_urls = [
        "https://news.google.com/articles/CBMilAFBVV95cUxPd1JKVE5kRjdRSzUyTUwyMVB3Z0p4VTkzeWNFYnpPcDJwY3VDRHRMT0lZYWZCY1QwRnFtODdqaVpZeXhzY1dlaTRwZVdGRWFadi10V1NVb2pVZ1NyMzAtcEpLX2FXM19wY09fZEJDV0VzamJWR1huWWdhRWlxREkyUEZ3QzhxUlFVbjhZd3llb0x1R2Y2?oc=5"
    ]

    results = await batch_resolve_urls(test_urls)
    for original, resolved in results.items():
        print(f"Original: {original[:80]}...")
        print(f"Resolved: {resolved}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
