#!/usr/bin/env python3
"""
Extract real URLs using undetected-chromedriver to bypass bot detection
"""
import json
import time
import undetected_chromedriver as uc

def setup_undetected_driver():
    """Setup undetected Chrome driver"""
    options = uc.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = uc.Chrome(options=options, version_main=None)
    return driver

def get_real_url(driver, google_news_url, max_wait=10):
    """Get real URL using undetected chromedriver"""
    try:
        print(f"    Fetching: {google_news_url[:80]}...")
        driver.get(google_news_url)

        # Wait for redirect
        for i in range(max_wait):
            time.sleep(1)
            current_url = driver.current_url

            # Check if we've left Google
            if 'google.com' not in current_url:
                print(f"    ✓ Redirected to: {current_url[:80]}...")
                return current_url, True

            # Check if we hit CAPTCHA
            if 'sorry' in current_url:
                print(f"    ✗ CAPTCHA detected")
                return google_news_url, False

        # Timeout - still on Google
        print(f"    ✗ No redirect after {max_wait}s")
        return driver.current_url, False

    except Exception as e:
        print(f"    ✗ Error: {e}")
        return google_news_url, False

def process_urls(input_json, output_json, max_articles=None, delay=3):
    """Process URLs with undetected chromedriver"""

    # Load URLs
    with open(input_json, 'r') as f:
        articles = json.load(f)

    if max_articles:
        articles = articles[:max_articles]

    print(f"Processing {len(articles)} articles with {delay}s delay between requests\n")

    driver = setup_undetected_driver()
    results = []
    success_count = 0

    try:
        for i, article in enumerate(articles, 1):
            print(f"[{i}/{len(articles)}] {article['title'][:60]}...")

            real_url, success = get_real_url(driver, article['google_news_url'])

            if success:
                success_count += 1

            results.append({
                'timestamp': article['timestamp'],
                'title': article['title'],
                'google_news_url': article['google_news_url'],
                'publisher_url': real_url,
                'resolved': success
            })

            # Rate limiting - important!
            if i < len(articles):
                print(f"    Waiting {delay}s...\n")
                time.sleep(delay)

    finally:
        driver.quit()

    # Save results
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results: {success_count}/{len(articles)} URLs successfully resolved")
    print(f"Saved to: {output_json}")
    print(f"{'='*70}")

    return results

if __name__ == "__main__":
    import sys

    # Default: process first 10 with 5s delay
    # Usage: python3 script.py [max_articles] [delay_seconds]
    max_articles = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    delay = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if max_articles == 10:
        print("Testing with first 10 articles and 5s delay")
        print("To process more: python3 extract_urls_undetected.py 50 3")
        print("To process all: python3 extract_urls_undetected.py 999 3\n")

    process_urls(
        'feed_urls.json',
        'feed_resolved_urls.json',
        max_articles=max_articles,
        delay=delay
    )
