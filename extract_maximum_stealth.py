#!/usr/bin/env python3
"""
Maximum stealth approach to bypass Google's detection
"""
import json
import time
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

def setup_stealth_driver():
    """Setup driver with maximum stealth"""
    options = uc.ChromeOptions()

    # Run headless
    options.add_argument('--headless=new')

    # Stealth settings
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-web-security')
    options.add_argument('--disable-features=VizDisplayCompositor')

    # Random window size
    width = random.randint(1200, 1920)
    height = random.randint(800, 1080)
    options.add_argument(f'--window-size={width},{height}')

    # Create driver
    driver = uc.Chrome(options=options, use_subprocess=True, version_main=None)

    # Additional JavaScript to hide automation
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": driver.execute_script("return navigator.userAgent").replace('Headless', '')
    })

    # Set navigator.webdriver to false
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    return driver

def get_publisher_url(driver, google_news_url):
    """Extract publisher URL with maximum stealth"""
    try:
        # Random delay before request
        time.sleep(random.uniform(1, 3))

        print(f"    Navigating...")
        driver.get(google_news_url)

        # Wait and check URL changes
        for i in range(15):
            time.sleep(1)
            current_url = driver.current_url

            # Success - left Google
            if 'google.com' not in current_url:
                print(f"    ✓ Success: {current_url[:70]}...")
                return current_url, True

            # CAPTCHA
            if '/sorry/' in current_url:
                print(f"    ✗ CAPTCHA page")
                return google_news_url, False

            # Still waiting...
            if i % 3 == 0:
                print(f"    ... waiting ({i+1}s)")

        # Timeout
        print(f"    ✗ Timeout - no redirect")
        return driver.current_url, False

    except Exception as e:
        print(f"    ✗ Error: {str(e)[:50]}")
        return google_news_url, False

def main():
    """Test with one URL"""
    print("Maximum Stealth Test\n")

    with open('feed_urls.json', 'r') as f:
        articles = json.load(f)

    # Test with first article
    test_article = articles[0]

    print(f"Testing: {test_article['title'][:60]}...")
    print(f"URL: {test_article['google_news_url'][:80]}...\n")

    driver = setup_stealth_driver()

    try:
        real_url, success = get_publisher_url(driver, test_article['google_news_url'])

        print(f"\n{'='*70}")
        if success:
            print(f"SUCCESS!")
            print(f"Publisher URL: {real_url}")
        else:
            print(f"FAILED - Google still blocking")
            print(f"Final URL: {real_url[:100]}")
        print(f"{'='*70}")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
