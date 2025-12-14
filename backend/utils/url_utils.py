"""
URL normalization utilities

Provides canonical URL normalization for deduplication and caching.
"""
from urllib.parse import urlparse, urlunparse, parse_qs


# Blacklist of tracking parameters to remove
TRACKING_PARAMS = {
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
    'fbclid', 'gclid', 'msclkid', 'mc_cid', 'mc_eid',
    '_ga', '_gl', 'ref', 'source', 'campaign'
}


def normalize_url(url: str) -> str:
    """
    Normalize URL to canonical form for deduplication.

    Removes:
    - www. prefix
    - Trailing slashes (EXCEPT when query string present, to preserve query-driven routes)
    - URL fragments (#)
    - Common tracking parameters (utm_*, fbclid, etc.)

    This deduplication happens at instant tier to:
    1. Prevent duplicate iframely calls (saves API quota)
    2. Prevent duplicate worker jobs (saves compute)
    3. Enable instant cache hits on same content

    Args:
        url: The URL to normalize

    Returns:
        Canonical URL string
    """
    parsed = urlparse(url)

    # Remove www prefix
    netloc = parsed.netloc.lower()
    if netloc.startswith('www.'):
        netloc = netloc[4:]

    # Remove trailing slash ONLY if no query string present
    # (preserves path for query-driven routes like /case-detail/?id=123)
    if parsed.query:
        path = parsed.path  # Keep trailing slash with query strings
    else:
        path = parsed.path.rstrip('/') if parsed.path != '/' else '/'

    # Remove tracking parameters (marketing noise)
    if parsed.query:
        params = parse_qs(parsed.query)

        # Keep only non-tracking params
        clean_params = {k: v for k, v in params.items() if k.lower() not in TRACKING_PARAMS}

        # Rebuild query string (sorted for consistency)
        query = '&'.join(f"{k}={v[0]}" for k, v in sorted(clean_params.items()))
    else:
        query = ''

    canonical = urlunparse((
        parsed.scheme or 'https',
        netloc,
        path,
        '',  # params
        query,
        ''   # fragment removed
    ))
    return canonical


def extract_domain(url: str) -> str:
    """
    Extract domain from URL, stripping www prefix.

    Args:
        url: The URL to extract domain from

    Returns:
        Domain string (e.g., 'example.com')
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except (ValueError, AttributeError):
        return url
