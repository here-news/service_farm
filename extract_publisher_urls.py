#!/usr/bin/env python3
"""
Extract real publisher URLs from Google News redirect links in sorted_feed.txt
"""
import requests
from urllib.parse import unquote
import time
import sys

def get_real_url(google_news_url):
    """Follow Google News redirect to get actual publisher URL"""
    try:
        # Convert RSS URL to web URL by removing /rss
        web_url = google_news_url.replace('/rss/articles/', '/articles/')

        # Use GET request with headers to simulate a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(web_url, allow_redirects=True, timeout=10, headers=headers)

        # Return the final URL after all redirects
        return response.url
    except Exception as e:
        print(f"Error fetching {google_news_url}: {e}", file=sys.stderr)
        return google_news_url  # Return original if fails

def process_feed_file(input_file, output_file):
    """Process sorted_feed.txt and extract real publisher URLs"""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            # Parse: timestamp | title - source | google_news_url
            parts = line.split(' | ')
            if len(parts) != 3:
                print(f"Skipping malformed line {line_num}", file=sys.stderr)
                continue

            timestamp = parts[0]
            title_source = parts[1]
            google_url = parts[2]

            # Get real URL
            print(f"Processing line {line_num}: {title_source[:50]}...", file=sys.stderr)
            real_url = get_real_url(google_url)

            # Write output
            output_line = f"{timestamp} | {title_source} | {real_url}\n"
            f_out.write(output_line)

            # Rate limit to be respectful
            time.sleep(0.5)

            if line_num % 10 == 0:
                print(f"Processed {line_num} articles...", file=sys.stderr)

if __name__ == "__main__":
    input_file = "sorted_feed.txt"
    output_file = "sorted_feed_with_real_urls.txt"

    print(f"Reading from: {input_file}", file=sys.stderr)
    print(f"Writing to: {output_file}", file=sys.stderr)
    print("This may take a while...\n", file=sys.stderr)

    process_feed_file(input_file, output_file)

    print(f"\nDone! Real URLs saved to {output_file}", file=sys.stderr)
