"""
ExtractionWorker - Extract text content from URLs

Responsibilities:
- Fetch HTML from URL
- Extract clean text using trafilatura
- Detect language
- Extract publication date
- Update page status: stub/preview → extracted
- Commission semantic analysis worker

Decision logic:
- Process if status is 'stub' or 'preview' (not yet extracted)
- Skip if status is 'extracted' or 'failed'
"""
import os
import uuid
import logging
import time
import asyncio
from typing import Tuple, Dict, Optional
from datetime import datetime
from urllib.parse import urlparse
import asyncpg
import httpx
import trafilatura
from langdetect import detect
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import sys
import os
# Add parent directory to path for direct imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.job_queue import JobQueue
from services.worker_base import BaseWorker
from services.multi_extractor import MultiMethodExtractor
# Direct import to avoid Neo4j dependency from repositories.__init__
from repositories.page_repository import PageRepository

logger = logging.getLogger(__name__)


class ExtractionWorker(BaseWorker):
    """
    Extract text content from web pages

    Combines:
    - Demo: trafilatura for content extraction
    - Gen1: Language detection, pub_time extraction
    - Gen2: Autonomous decision-making based on page status
    """

    def __init__(self, pool: asyncpg.Pool, job_queue: JobQueue, worker_id: int = 1):
        super().__init__(
            pool=pool,
            job_queue=job_queue,
            worker_name=f"extraction-worker-{worker_id}",
            queue_name="queue:extraction:high"
        )
        self.worker_id = worker_id
        self.multi_extractor = MultiMethodExtractor(min_words=100)
        self.page_repo = PageRepository(pool)

    async def get_state(self, job: dict) -> dict:
        """
        Fetch page state from PageRepository

        Args:
            job: {'page_id': uuid, 'url': str, 'retry_count': int}

        Returns:
            Page state as dict
        """
        page_id = job['page_id']
        page = await self.page_repo.get_by_id(page_id)

        if not page:
            raise ValueError(f"Page {page_id} not found in database")

        # Convert Page model to dict for compatibility with BaseWorker
        return {
            'id': page.id,
            'url': page.url,
            'canonical_url': page.canonical_url,
            'title': page.title,
            'status': page.status,
            'language': page.language,
            'word_count': page.word_count,
            'pub_time': page.pub_time,
            'created_at': page.created_at,
            'updated_at': page.updated_at
        }

    async def should_process(self, state: dict) -> Tuple[bool, float]:
        """
        Decide if page needs extraction

        Logic:
        - Process if status is 'stub' or 'preview' (needs extraction)
        - Skip if 'extracted' (already done)
        - Skip if 'failed' (extraction failed before, don't retry immediately)

        Returns:
            (should_process, confidence)
        """
        status = state.get('status')

        # Already extracted - skip
        if status == 'extracted':
            return (False, 0.0)

        # Failed before - skip (unless retry logic implemented)
        if status == 'failed':
            return (False, 0.1)

        # Stub or preview - needs extraction
        if status in ['stub', 'preview']:
            confidence = 0.9 if status == 'stub' else 0.8
            return (True, confidence)

        # Unknown status - process cautiously
        return (True, 0.5)

    async def process(self, job: dict, state: dict):
        """
        Extract content from URL

        Steps:
        1. Fetch HTML from URL
        2. Extract text with trafilatura
        3. Detect language
        4. Extract publication date
        5. Update database
        6. Commission semantic analysis worker
        """
        page_id = job['page_id']
        url = job['url']

        logger.info(f"[{self.worker_name}] Extracting: {url}")

        try:
            # Step 1: Fetch HTML
            html = await self._fetch_html(url)

            if not html:
                await self._mark_failed(page_id, "Failed to fetch HTML")
                return

            # Step 2: Extract text with multi-method extraction
            result = await self.multi_extractor.extract(url, html)

            if not result.success:
                # Try Playwright as last resort before failing
                logger.warning(f"[{self.worker_name}] ⚠️ Standard extraction failed - trying Playwright fallback")
                playwright_result = await self._extract_with_playwright(url)

                if not playwright_result or not playwright_result['success']:
                    await self._mark_failed(page_id, f"All extraction methods failed: {result.error_message}")
                    return

                # Playwright succeeded! Use its content
                extracted = playwright_result['content']
                word_count = playwright_result['word_count']

                # Detect language
                try:
                    detected_lang = detect(extracted[:1000])
                except:
                    detected_lang = state.get('language', 'en')

                # Get metadata from state
                existing = await self.page_repo.get_by_id(page_id)

                # Update page with playwright-extracted content
                await self.page_repo.update_extracted_content(
                    page_id=page_id,
                    content_text=extracted,
                    title=existing.title,
                    description=existing.description,
                    author=existing.author,
                    thumbnail_url=existing.thumbnail_url,
                    metadata_confidence=existing.metadata_confidence or 0.5,
                    language=detected_lang,
                    word_count=word_count,
                    pub_time=existing.pub_time
                )

                logger.info(f"[{self.worker_name}] ✅ Playwright fallback succeeded after standard extraction failed: {word_count} words from {url}")

                # Commission semantic analysis
                semantic_job = {
                    'page_id': str(page_id),
                    'url': url
                }
                await self.job_queue.enqueue('queue:semantic:high', semantic_job)
                return

            extracted = result.content
            logger.info(f"[{self.worker_name}] Used {result.method_used} for {url}")

            # Step 3: Detect language
            try:
                detected_lang = detect(extracted[:1000])  # Use first 1000 chars
            except:
                detected_lang = state.get('language', 'en')  # Fallback to existing

            # Step 4: Extract publication date (from HTML metadata)
            pub_time = self._extract_pub_time(html)
            # IMPORTANT: If pub_time extraction fails, leave it as None
            # Don't use extraction/scraping time as publication date

            # Step 5: Update database with extracted metadata
            word_count = len(extracted.split())

            # Get metadata from extraction result
            title = result.title
            description = result.description
            author = ', '.join(result.authors) if result.authors else None  # Join multiple authors
            thumbnail_url = result.top_image  # Use top_image as thumbnail
            site_name = result.site_name  # Publisher name from og:site_name

            # Extract domain from URL
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
            except:
                domain = None

            # Get existing metadata from repository (preserve iframely data)
            existing = await self.page_repo.get_by_id(page_id)

            # Use COALESCE to prefer new data, fallback to existing (from iframely)
            final_title = title or existing.title
            final_description = description or existing.description
            final_author = author or existing.author
            final_thumbnail = thumbnail_url or existing.thumbnail_url
            final_site_name = site_name or existing.site_name
            # IMPORTANT: Prefer our HTML extraction for pub_time over iframely
            # Reason: iframely often returns cache/processing timestamps instead of
            # actual publication dates. Our multi-strategy HTML extraction (JSON-LD,
            # meta tags, trafilatura) gets the authoritative date from the publisher.
            # Only fallback to iframely's pub_time if our extraction completely fails.
            final_pub_time = pub_time or existing.pub_time

            # Calculate metadata confidence (0.0-1.0)
            # 1.0 = all fields present, 0.5 = half present, etc.
            # Include pub_time in confidence calculation (important for timeline ordering)
            metadata_fields = [final_title, final_description, final_author, final_thumbnail, final_pub_time]
            metadata_confidence = sum(1 for f in metadata_fields if f) / len(metadata_fields)

            # Update page using repository
            await self.page_repo.update_extracted_content(
                page_id=page_id,
                content_text=extracted,
                title=final_title,
                description=final_description,
                author=final_author,
                thumbnail_url=final_thumbnail,
                metadata_confidence=metadata_confidence,
                language=detected_lang,
                word_count=word_count,
                pub_time=final_pub_time,
                domain=domain,
                site_name=final_site_name
            )

            # Check if word count is suspiciously low (likely paywall/JS-heavy page)
            if word_count < 100:
                logger.warning(
                    f"[{self.worker_name}] ⚠️ Low word count ({word_count} words) - trying Playwright fallback"
                )

                # Try Playwright as fallback for JS-heavy pages
                playwright_result = await self._extract_with_playwright(url)

                if playwright_result and playwright_result['success']:
                    # Playwright succeeded - update with new content
                    extracted = playwright_result['content']
                    word_count = playwright_result['word_count']

                    # Re-detect language with new content
                    try:
                        detected_lang = detect(extracted[:1000])
                    except:
                        pass

                    # Update page with playwright-extracted content
                    await self.page_repo.update_extracted_content(
                        page_id=page_id,
                        content_text=extracted,
                        title=final_title,
                        description=final_description,
                        author=final_author,
                        thumbnail_url=final_thumbnail,
                        metadata_confidence=metadata_confidence,
                        language=detected_lang,
                        word_count=word_count,
                        pub_time=final_pub_time
                    )

                    logger.info(
                        f"[{self.worker_name}] ✅ Playwright fallback succeeded: "
                        f"{word_count} words (lang={detected_lang}) from {url}"
                    )
                else:
                    # Both standard and playwright extraction failed
                    logger.warning(
                        f"[{self.worker_name}] ❌ Both standard and Playwright extraction failed for {url}"
                    )
                    await self._mark_failed(page_id, "Insufficient content after all extraction methods")
                    return
            else:
                logger.info(
                    f"[{self.worker_name}] ✅ Extracted {word_count} words "
                    f"(lang={detected_lang}) from {url}"
                )

            # Step 6: Commission semantic analysis worker (even for low word count - let semantic decide)
            await self.job_queue.enqueue('queue:semantic:high', {
                'page_id': str(page_id),
                'url': url,
                'text': extracted,  # Pass extracted text to semantic worker
                'retry_count': 0
            })

            logger.info(f"[{self.worker_name}] Commissioned semantic analysis for {page_id}")

        except Exception as e:
            logger.error(f"[{self.worker_name}] Extraction failed: {e}", exc_info=True)
            await self._mark_failed(page_id, str(e))
            raise

    async def _fetch_html(self, url: str, timeout: float = 30.0) -> str:
        """
        Fetch HTML from URL

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            HTML content or None if failed
        """
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                })

                if response.status_code == 200:
                    return response.text
                else:
                    logger.warning(
                        f"[{self.worker_name}] HTTP {response.status_code} for {url}"
                    )
                    return None

        except httpx.TimeoutException:
            logger.warning(f"[{self.worker_name}] Timeout fetching {url}")
            return None
        except Exception as e:
            logger.error(f"[{self.worker_name}] Error fetching {url}: {e}")
            return None

    def _extract_pub_time(self, html: str) -> datetime:
        """
        Extract publication time from HTML metadata with multiple fallbacks

        Strategy:
        1. Try trafilatura metadata extraction
        2. Try common meta tags (article:published_time, datePublished, etc.)
        3. Try JSON-LD structured data
        4. Return None if all fail (don't default to current time)

        Args:
            html: HTML content

        Returns:
            Publication datetime or None
        """
        import re
        from dateutil import parser as date_parser

        # Strategy 1: Trafilatura (but check if it has time component)
        trafilatura_date = None
        try:
            from trafilatura import extract_metadata
            metadata = extract_metadata(html)
            if metadata and metadata.date:
                dt = datetime.fromisoformat(metadata.date.replace('Z', '+00:00'))
                # Only use trafilatura if it has a time component (not midnight)
                if dt.hour != 0 or dt.minute != 0 or dt.second != 0:
                    return dt
                else:
                    # Save as fallback but try other strategies first
                    trafilatura_date = dt
        except:
            pass

        # Strategy 2: Common meta tags
        meta_patterns = [
            r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']article:published_time["\']',
            r'<meta[^>]+name=["\']datePublished["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']datePublished["\']',
            r'<meta[^>]+itemprop=["\']datePublished["\'][^>]+content=["\']([^"\']+)["\']',
            r'<time[^>]+datetime=["\']([^"\']+)["\']',
        ]

        for pattern in meta_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    return date_parser.parse(date_str)
                except:
                    continue

        # Strategy 3: JSON-LD structured data
        jsonld_pattern = r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
        jsonld_matches = re.findall(jsonld_pattern, html, re.DOTALL | re.IGNORECASE)

        for jsonld_str in jsonld_matches:
            try:
                import json
                data = json.loads(jsonld_str)

                # Handle both single object and array
                if isinstance(data, list):
                    data = data[0] if data else {}

                # Handle @graph structure (used by BBC and others)
                if '@graph' in data and isinstance(data['@graph'], list) and len(data['@graph']) > 0:
                    data = data['@graph'][0]

                # Try common date fields
                for field in ['datePublished', 'publishDate', 'dateCreated', 'uploadDate']:
                    if field in data:
                        try:
                            return date_parser.parse(data[field])
                        except:
                            continue
            except:
                continue

        # All strategies failed - use trafilatura date-only as last resort
        return trafilatura_date

    async def _extract_with_playwright(self, url: str, timeout_ms: int = 30000) -> Optional[Dict]:
        """
        Extract content using Playwright browser (fallback for JS-heavy pages)

        Args:
            url: URL to extract
            timeout_ms: Browser timeout in milliseconds

        Returns:
            Dict with content, word_count, success, or None if failed
        """
        start_time = time.time()

        try:
            async with async_playwright() as p:
                logger.info(f"[{self.worker_name}] 🎭 Trying Playwright fallback for {url}")

                # Launch browser
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-blink-features=AutomationControlled'
                    ]
                )

                # Use iPhone device emulation with authentic mobile user agent
                # (from gen1 - works better than desktop UA for many sites)
                device = p.devices.get('iPhone 13 Pro', p.devices['iPhone 12 Pro'])
                # Override user_agent in device dict to avoid duplicate argument error
                device_config = {**device, 'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1'}
                context = await browser.new_context(**device_config)
                page = await context.new_page()

                # Add stealth scripts
                await page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                """)

                try:
                    # Navigate to URL
                    await page.goto(url, timeout=timeout_ms, wait_until='domcontentloaded')

                    # Wait for network idle
                    try:
                        await page.wait_for_load_state('networkidle', timeout=5000)
                    except PlaywrightTimeoutError:
                        pass  # Continue anyway

                    # Wait for JS execution
                    await asyncio.sleep(2)

                    # Extract text
                    text_content = await page.inner_text('body')

                    # Clean up text
                    lines = (line.strip() for line in text_content.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    cleaned_text = ' '.join(chunk for chunk in chunks if chunk)

                    word_count = len(cleaned_text.split())
                    extraction_time_ms = (time.time() - start_time) * 1000

                    await browser.close()

                    logger.info(
                        f"[{self.worker_name}] ✅ Playwright extracted {word_count} words "
                        f"from {url} ({extraction_time_ms:.0f}ms)"
                    )

                    return {
                        'success': word_count >= 100,
                        'content': cleaned_text,
                        'word_count': word_count
                    }

                except PlaywrightTimeoutError:
                    await browser.close()
                    logger.warning(f"[{self.worker_name}] ⏱️ Playwright timeout for {url}")
                    return None

        except Exception as e:
            logger.error(f"[{self.worker_name}] ❌ Playwright extraction failed: {e}")
            return None

    async def _mark_failed(self, page_id: uuid.UUID, reason: str):
        """
        Mark page extraction as failed and create rogue extraction task

        For URLs that block scrapers (401/403), create a task for browser extension

        Args:
            page_id: Page UUID
            reason: Failure reason
        """
        # Get page from repository
        page = await self.page_repo.get_by_id(page_id)

        if not page:
            logger.error(f"[{self.worker_name}] Page {page_id} not found")
            return

        # Mark page as failed using repository
        await self.page_repo.mark_failed(page_id)

        # Check if this is a scraper-blocking issue (401/403 or fetch failure)
        is_rogue = any(code in reason for code in ['401', '403', 'Failed to fetch'])

        if is_rogue:
            # Create rogue extraction task for browser extension
            await self.page_repo.create_rogue_task(page_id, page.url)

            logger.info(
                f"[{self.worker_name}] 🔴 Created rogue task for {page.url} (browser extension will handle)"
            )
        else:
            logger.warning(f"[{self.worker_name}] ❌ Marked {page_id} as failed: {reason}")

    async def handle_error(self, job: dict, error: Exception):
        """
        Handle extraction error with retry logic

        Args:
            job: Failed job
            error: Exception that occurred
        """
        retry_count = job.get('retry_count', 0)
        max_retries = 3

        if retry_count < max_retries:
            # Re-enqueue with incremented retry count
            await self.job_queue.enqueue(self.queue_name, {
                **job,
                'retry_count': retry_count + 1
            })
            logger.info(
                f"[{self.worker_name}] Re-enqueued job (retry {retry_count + 1}/{max_retries})"
            )
        else:
            # Max retries exceeded - mark as failed
            try:
                page_id = job['page_id']
                await self._mark_failed(page_id, f"Max retries exceeded: {error}")
            except:
                pass
