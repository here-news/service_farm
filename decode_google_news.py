#!/usr/bin/env python3
"""
Try to decode Google News article IDs to find embedded URLs
"""
import base64
import json
import re

def try_decode_google_news_url(url):
    """Attempt to decode embedded URL from Google News article ID"""
    try:
        # Extract the article ID (the CBM... part)
        match = re.search(r'articles/([A-Za-z0-9_-]+)', url)
        if not match:
            return None

        article_id = match.group(1)

        # Try base64 decode variants
        for variant in [article_id, article_id + '=', article_id + '==']:
            try:
                decoded = base64.b64decode(variant, validate=True)
                # Look for URL patterns in decoded data
                text = decoded.decode('utf-8', errors='ignore')
                # Search for http/https URLs
                urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
                if urls:
                    return urls[0]
            except:
                continue

        # Try URL-safe base64
        for variant in [article_id, article_id + '=', article_id + '==']:
            try:
                decoded = base64.urlsafe_b64decode(variant)
                text = decoded.decode('utf-8', errors='ignore')
                urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
                if urls:
                    return urls[0]
            except:
                continue

    except Exception as e:
        pass

    return None

# Test with sample URLs
if __name__ == "__main__":
    test_urls = [
        "https://news.google.com/articles/CBMilAFBVV95cUxPd1JKVE5kRjdRSzUyTUwyMVB3Z0p4VTkzeWNFYnpPcDJwY3VDRHRMT0lZYWZCY1QwRnFtODdqaVpZeXhzY1dlaTRwZVdGRWFadi10V1NVb2pVZ1NyMzAtcEpLX2FXM19wY09fZEJDV0VzamJWR1huWWdhRWlxREkyUEZ3QzhxUlFVbjhZd3llb0x1R2Y2?oc=5",
        "https://news.google.com/articles/CBMiogFBVV95cUxPcm94OXBMZjY5MXZ0YXVnRnJIaFIxS1NoR2JpTjNHeUtsT2xiSlcxMHRMVUpHRUFBZTJWZ2NGVXdKSmtvY2Y0RTFZcWNpMEFEZ0JYZExpbEFRQjQwNkl3b3ppS2NoWnhHMFJET3pyZmZoU3VocU95UmtaVnptMUVLUGl0akpnSmZjSkNMWDh2RWlGUXV0S2VjWUwyWG9GZEh0WkE?oc=5",
    ]

    for url in test_urls:
        print(f"\nOriginal: {url[:80]}...")
        decoded = try_decode_google_news_url(url)
        if decoded:
            print(f"Decoded: {decoded}")
        else:
            print("Could not decode")
