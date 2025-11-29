#!/usr/bin/env python3
"""
Extract real publisher URLs using Selenium to avoid Google's bot detection
Requires: pip install selenium
"""
import time
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def setup_driver():
    """Set up headless Chrome driver"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(options=chrome_options)
    return driver

def get_real_url_selenium(driver, google_news_url):
    """Get real URL using Selenium"""
    try:
        # Convert RSS to web URL
        web_url = google_news_url.replace('/rss/articles/', '/articles/')

        driver.get(web_url)
        time.sleep(2)  # Wait for redirect

        # Get final URL after redirect
        real_url = driver.current_url

        # Check if we got redirected to actual article
        if 'google.com' not in real_url:
            return real_url
        else:
            return google_news_url  # Redirect failed
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return google_news_url

def process_feed(input_file, output_file, max_articles=None):
    """Process feed file with Selenium"""
    driver = setup_driver()

    try:
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line_num, line in enumerate(f_in, 1):
                if max_articles and line_num > max_articles:
                    break

                line = line.strip()
                if not line:
                    continue

                parts = line.split(' | ')
                if len(parts) != 3:
                    continue

                timestamp = parts[0]
                title_source = parts[1]
                google_url = parts[2]

                print(f"Processing {line_num}: {title_source[:50]}...", file=sys.stderr)
                real_url = get_real_url_selenium(driver, google_url)

                output_line = f"{timestamp} | {title_source} | {real_url}\n"
                f_out.write(output_line)

                if line_num % 10 == 0:
                    print(f"Processed {line_num} articles...", file=sys.stderr)

    finally:
        driver.quit()

if __name__ == "__main__":
    # Process first 10 as a test
    process_feed("sorted_feed.txt", "sorted_feed_with_real_urls_selenium.txt", max_articles=10)
    print("\nDone! Check sorted_feed_with_real_urls_selenium.txt", file=sys.stderr)
