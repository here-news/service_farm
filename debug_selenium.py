#!/usr/bin/env python3
"""
Debug version to see what's happening with Google News URLs
"""
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def test_google_news_redirect():
    """Test one URL to see what happens"""

    # Setup
    chrome_options = Options()
    # Run in headless mode
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    test_url = "https://news.google.com/articles/CBMilAFBVV95cUxPd1JKVE5kRjdRSzUyTUwyMVB3Z0p4VTkzeWNFYnpPcDJwY3VDRHRMT0lZYWZCY1QwRnFtODdqaVpZeXhzY1dlaTRwZVdGRWFadi10V1NVb2pVZ1NyMzAtcEpLX2FXM19wY09fZEJDV0VzamJWR1huWWdhRWlxREkyUEZ3QzhxUlFVbjhZd3llb0x1R2Y2?oc=5"

    print(f"Testing URL: {test_url}\n")

    try:
        driver.get(test_url)
        print(f"Initial URL: {driver.current_url}")

        # Check page source
        time.sleep(2)
        print(f"\nAfter 2s: {driver.current_url}")

        time.sleep(3)
        print(f"After 5s total: {driver.current_url}")

        # Check page title
        print(f"\nPage title: {driver.title}")

        # Save screenshot
        driver.save_screenshot("debug_screenshot.png")
        print("Screenshot saved to debug_screenshot.png")

        # Check if there's a redirect link we need to click
        page_source = driver.page_source
        if 'click here' in page_source.lower() or 'continue' in page_source.lower():
            print("\nPage might require user interaction")

        print(f"\nFinal URL: {driver.current_url}")

    finally:
        driver.quit()

if __name__ == "__main__":
    test_google_news_redirect()
