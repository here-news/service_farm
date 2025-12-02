"""
Playwright Worker - Browser-based extraction for JS-heavy content

Handles pages that failed semantic analysis due to JavaScript-rendered content.
Extracts full text content + multimedia artifacts (images with captions).
"""
import os
import uuid
import time
import logging
import asyncio
import asyncpg
from typing import Dict, List, Optional
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlaywrightWorker:
    """
    Browser-based content extraction worker

    Triggered when semantic analysis fails due to insufficient content
    but word_count >= 100 (indicating JS-heavy content)
    """

    # iPhone 14 Pro Max device profile (better readability, wider viewport)
    DEVICE_NAME = 'iPhone 14 Pro Max'

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        job_queue,
        worker_id: int = 1,
        timeout: int = 30,
        headless: bool = True
    ):
        """
        Initialize Playwright worker

        Args:
            db_pool: PostgreSQL connection pool
            job_queue: Redis job queue
            worker_id: Worker identifier
            timeout: Request timeout in seconds
            headless: Run browser in headless mode
        """
        self.db_pool = db_pool
        self.job_queue = job_queue
        self.worker_id = worker_id
        self.worker_name = f"playwright-worker-{worker_id}"
        self.timeout_ms = timeout * 1000
        self.headless = headless
        self.running = False

    async def start(self):
        """Start worker loop"""
        self.running = True
        logger.info(f"🎭 {self.worker_name} started, waiting for jobs...")

        while self.running:
            try:
                # Dequeue job from Redis
                job = await self.job_queue.dequeue('playwright_extraction', timeout=5)

                if job:
                    page_id = uuid.UUID(job['page_id'])
                    url = job['url']
                    reason = job.get('reason', 'unknown')

                    logger.info(f"🎭 [{self.worker_name}] Processing: {url} (reason: {reason})")

                    await self._process_page(page_id, url)
                else:
                    # No jobs, wait a bit
                    await asyncio.sleep(1)

            except KeyboardInterrupt:
                logger.info(f"🎭 [{self.worker_name}] Received shutdown signal")
                self.running = False
            except Exception as e:
                logger.error(f"🎭 [{self.worker_name}] Error in main loop: {e}")
                await asyncio.sleep(5)  # Backoff on error

    async def _process_page(self, page_id: uuid.UUID, url: str):
        """
        Extract content from page using Playwright

        Args:
            page_id: Page UUID
            url: URL to extract
        """
        start_time = time.time()

        async with self.db_pool.acquire() as conn:
            # Mark as processing
            await conn.execute("""
                UPDATE core.pages
                SET status = 'playwright_processing',
                    updated_at = NOW()
                WHERE id = $1
            """, page_id)

            try:
                # Extract content with Playwright
                result = await self._extract_with_browser(url)

                if result['success']:
                    # Update page with extracted content
                    await conn.execute("""
                        UPDATE core.pages
                        SET content_text = $2,
                            word_count = $3,
                            status = 'extracted',
                            error_message = NULL,
                            updated_at = NOW()
                        WHERE id = $1
                    """, page_id, result['content'], result['word_count'])

                    # TODO: Store media artifacts in future
                    # for media in result.get('media', []):
                    #     await self._store_media_artifact(conn, page_id, media)

                    extraction_time_ms = (time.time() - start_time) * 1000

                    logger.info(
                        f"✅ [{self.worker_name}] Extracted {result['word_count']} words, "
                        f"{len(result.get('media', []))} media items from {url} "
                        f"({extraction_time_ms:.0f}ms)"
                    )

                    # Re-queue for semantic analysis
                    await self.job_queue.enqueue('semantic_processing', {
                        'page_id': str(page_id)
                    })

                    logger.info(f"🔄 [{self.worker_name}] Re-queued {url} for semantic processing")

                else:
                    # Extraction failed
                    await conn.execute("""
                        UPDATE core.pages
                        SET status = 'playwright_failed',
                            error_message = $2,
                            updated_at = NOW()
                        WHERE id = $1
                    """, page_id, result.get('error_message', 'Unknown error'))

                    logger.error(
                        f"❌ [{self.worker_name}] Playwright extraction failed for {url}: "
                        f"{result.get('error_message')}"
                    )

            except Exception as e:
                logger.error(f"❌ [{self.worker_name}] Error processing {url}: {e}")

                await conn.execute("""
                    UPDATE core.pages
                    SET status = 'playwright_failed',
                        error_message = $2,
                        updated_at = NOW()
                    WHERE id = $1
                """, page_id, f"Worker error: {str(e)}")

    async def _extract_with_browser(self, url: str) -> Dict:
        """
        Extract content using Playwright browser

        Returns comprehensive extraction including:
        - Full text content (JS-rendered)
        - Images with captions
        - Videos with descriptions
        - Metadata

        Args:
            url: URL to extract

        Returns:
            Dict with success, content, word_count, media, error_message
        """
        start_time = time.time()

        try:
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(
                    headless=self.headless,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-blink-features=AutomationControlled'
                    ]
                )

                # Get iPhone device configuration
                device = p.devices[self.DEVICE_NAME]

                # Create context with mobile device emulation
                context = await browser.new_context(**device)

                # Create page
                page = await context.new_page()

                # Add stealth scripts to evade bot detection
                await page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                """)

                try:
                    # Navigate to URL
                    await page.goto(
                        url,
                        timeout=self.timeout_ms,
                        wait_until='domcontentloaded'
                    )

                    # Wait for network to be idle (content loaded)
                    try:
                        await page.wait_for_load_state('networkidle', timeout=5000)
                    except PlaywrightTimeoutError:
                        # Network didn't idle within 5s, continue anyway
                        pass

                    # Wait for dynamic content and JS execution
                    await asyncio.sleep(2)

                    # Extract everything in parallel
                    text_content, media_items = await asyncio.gather(
                        page.inner_text('body'),
                        self._extract_media(page)
                    )

                    # Clean up text
                    lines = (line.strip() for line in text_content.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    cleaned_text = ' '.join(chunk for chunk in chunks if chunk)

                    word_count = len(cleaned_text.split())

                    extraction_time_ms = (time.time() - start_time) * 1000

                    # Cleanup
                    await browser.close()

                    return {
                        'success': word_count >= 100,
                        'content': cleaned_text,
                        'word_count': word_count,
                        'media': media_items,
                        'extraction_time_ms': extraction_time_ms
                    }

                except PlaywrightTimeoutError:
                    extraction_time_ms = (time.time() - start_time) * 1000
                    await browser.close()

                    return {
                        'success': False,
                        'content': '',
                        'word_count': 0,
                        'media': [],
                        'extraction_time_ms': extraction_time_ms,
                        'error_message': f'Page load timed out after {self.timeout_ms}ms'
                    }

                except Exception as e:
                    extraction_time_ms = (time.time() - start_time) * 1000
                    await browser.close()

                    return {
                        'success': False,
                        'content': '',
                        'word_count': 0,
                        'media': [],
                        'extraction_time_ms': extraction_time_ms,
                        'error_message': f'Navigation error: {str(e)}'
                    }

        except Exception as e:
            extraction_time_ms = (time.time() - start_time) * 1000

            return {
                'success': False,
                'content': '',
                'word_count': 0,
                'media': [],
                'extraction_time_ms': extraction_time_ms,
                'error_message': f'Browser initialization error: {str(e)}'
            }

    async def _extract_media(self, page) -> List[Dict]:
        """
        Extract media artifacts (images, videos) with contextual information

        Extracts:
        - <figure> with <figcaption>
        - <img> with data-caption or nearby captions
        - Videos with descriptions

        Args:
            page: Playwright page object

        Returns:
            List of media items with url, caption, alt, type
        """
        media_items = await page.evaluate("""
            () => {
                const media = [];

                // Pattern 1: <figure> with <figcaption>
                document.querySelectorAll('figure').forEach((figure, idx) => {
                    const img = figure.querySelector('img');
                    const video = figure.querySelector('video');
                    const caption = figure.querySelector('figcaption');

                    if (img) {
                        media.push({
                            type: 'image',
                            url: img.src,
                            alt: img.alt || '',
                            caption: caption ? caption.innerText.trim() : '',
                            width: img.naturalWidth,
                            height: img.naturalHeight,
                            position: media.length
                        });
                    } else if (video) {
                        media.push({
                            type: 'video',
                            url: video.src || video.querySelector('source')?.src || '',
                            caption: caption ? caption.innerText.trim() : '',
                            poster: video.poster || '',
                            position: media.length
                        });
                    }
                });

                // Pattern 2: Images with data-caption attribute
                document.querySelectorAll('img[data-caption]').forEach((img) => {
                    // Skip if already captured in figure
                    if (!img.closest('figure')) {
                        media.push({
                            type: 'image',
                            url: img.src,
                            alt: img.alt || '',
                            caption: img.getAttribute('data-caption') || '',
                            width: img.naturalWidth,
                            height: img.naturalHeight,
                            position: media.length
                        });
                    }
                });

                // Pattern 3: WordPress wp-caption
                document.querySelectorAll('.wp-caption').forEach((div) => {
                    const img = div.querySelector('img');
                    const caption = div.querySelector('.wp-caption-text');

                    if (img) {
                        media.push({
                            type: 'image',
                            url: img.src,
                            alt: img.alt || '',
                            caption: caption ? caption.innerText.trim() : '',
                            width: img.naturalWidth,
                            height: img.naturalHeight,
                            position: media.length
                        });
                    }
                });

                return media;
            }
        """)

        return media_items

    async def stop(self):
        """Stop worker gracefully"""
        self.running = False
        logger.info(f"🎭 {self.worker_name} stopped")
