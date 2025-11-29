#!/usr/bin/env python3
"""
Prepare feed URLs for processing - convert to simple list format
Your content extraction pipeline can handle the Google News redirects
when actually fetching the articles.
"""

def prepare_urls(input_file, output_file):
    """Extract URLs into simple list format"""
    urls = []

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(' | ')
            if len(parts) == 3:
                timestamp = parts[0]
                title_source = parts[1]
                url = parts[2]

                # Convert RSS to web URL (better for redirects)
                web_url = url.replace('/rss/articles/', '/articles/')

                urls.append({
                    'timestamp': timestamp,
                    'title': title_source,
                    'google_news_url': web_url
                })

    # Write as simple format
    with open(output_file, 'w') as f:
        for item in urls:
            f.write(f"{item['google_news_url']}\n")

    # Also write JSON version with metadata
    import json
    json_file = output_file.replace('.txt', '.json')
    with open(json_file, 'w') as f:
        json.dump(urls, f, indent=2)

    print(f"Prepared {len(urls)} URLs")
    print(f"URLs list: {output_file}")
    print(f"JSON with metadata: {json_file}")

if __name__ == "__main__":
    prepare_urls("sorted_feed.txt", "feed_urls.txt")
