#!/usr/bin/env python3
"""
Extract real publisher URLs from Google News using Selenium
"""
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

def setup_driver():
    """Setup headless Chrome driver"""
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    return driver

def get_real_url(driver, google_news_url, timeout=10):
    """Get real publisher URL by following Google News redirect"""
    try:
        # Navigate to the URL
        driver.get(google_news_url)

        # Wait for redirect (Google News redirects quickly)
        time.sleep(3)

        # Get the current URL after redirect
        final_url = driver.current_url

        # Check if we actually left Google
        if 'google.com' not in final_url and 'sorry' not in final_url:
            return final_url, True
        else:
            return google_news_url, False

    except Exception as e:
        print(f"Error: {e}")
        return google_news_url, False

def process_feed(input_json, output_json, max_articles=None):
    """Process all URLs from feed_urls.json"""

    # Load input
    with open(input_json, 'r') as f:
        articles = json.load(f)

    if max_articles:
        articles = articles[:max_articles]

    print(f"Processing {len(articles)} articles...")

    # Setup browser
    driver = setup_driver()

    results = []
    success_count = 0

    try:
        for i, article in enumerate(articles, 1):
            print(f"\n[{i}/{len(articles)}] {article['title'][:60]}...")

            real_url, success = get_real_url(driver, article['google_news_url'])

            if success:
                print(f"  ✓ {real_url}")
                success_count += 1
            else:
                print(f"  ✗ Failed to resolve")

            results.append({
                'timestamp': article['timestamp'],
                'title': article['title'],
                'google_news_url': article['google_news_url'],
                'publisher_url': real_url,
                'resolved': success
            })

            # Rate limiting
            time.sleep(2)

    finally:
        driver.quit()

    # Save results
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results: {success_count}/{len(articles)} URLs successfully resolved")
    print(f"Saved to: {output_json}")
    print(f"{'='*60}")

    return results

if __name__ == "__main__":
    import sys

    # Process first 5 as a test, or all if argument provided
    max_articles = 5 if len(sys.argv) < 2 else None

    if max_articles:
        print("Testing with first 5 articles. Run with 'all' to process everything:")
        print("  python3 extract_real_urls_selenium.py all\n")

    results = process_feed(
        'feed_urls.json',
        'feed_with_real_urls.json',
        max_articles=max_articles
    )
