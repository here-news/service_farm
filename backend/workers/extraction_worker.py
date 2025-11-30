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
from typing import Tuple
from datetime import datetime
import asyncpg
import httpx
import trafilatura
from langdetect import detect
from services.job_queue import JobQueue
from services.worker_base import BaseWorker
from services.multi_extractor import MultiMethodExtractor

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

    async def get_state(self, job: dict) -> dict:
        """
        Fetch page state from PostgreSQL

        Args:
            job: {'page_id': uuid, 'url': str, 'retry_count': int}

        Returns:
            Page state from core.pages table
        """
        page_id = uuid.UUID(job['page_id'])

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    id, url, canonical_url, title, status,
                    language, word_count, pub_time,
                    created_at, updated_at
                FROM core.pages
                WHERE id = $1
            """, page_id)

            if not row:
                raise ValueError(f"Page {page_id} not found in database")

            return dict(row)

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
        page_id = uuid.UUID(job['page_id'])
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
                await self._mark_failed(page_id, f"All extraction methods failed: {result.error_message}")
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

            # Step 5: Update database with extracted metadata
            word_count = len(extracted.split())

            # Get metadata from extraction result
            title = result.title
            description = result.description
            author = ', '.join(result.authors) if result.authors else None  # Join multiple authors
            thumbnail_url = result.top_image  # Use top_image as thumbnail

            async with self.pool.acquire() as conn:
                # First, get existing metadata to preserve iframely data
                existing = await conn.fetchrow("""
                    SELECT title, description, author, thumbnail_url
                    FROM core.pages WHERE id = $1
                """, page_id)

                # Use COALESCE to prefer new data, fallback to existing (from iframely)
                final_title = title or existing['title']
                final_description = description or existing['description']
                final_author = author or existing['author']
                final_thumbnail = thumbnail_url or existing['thumbnail_url']

                # Calculate metadata confidence (0.0-1.0)
                # 1.0 = all fields present, 0.5 = half present, etc.
                metadata_fields = [final_title, final_description, final_author, final_thumbnail]
                metadata_confidence = sum(1 for f in metadata_fields if f) / len(metadata_fields)

                await conn.execute("""
                    UPDATE core.pages
                    SET
                        content_text = $2,
                        title = $3,
                        description = $4,
                        author = $5,
                        thumbnail_url = $6,
                        metadata_confidence = $7,
                        status = 'extracted',
                        language = $8,
                        word_count = $9,
                        pub_time = $10,
                        updated_at = NOW()
                    WHERE id = $1
                """, page_id, extracted, final_title, final_description, final_author,
                     final_thumbnail, metadata_confidence, detected_lang, word_count, pub_time)

                # Store extracted content in a separate table (optional)
                # For now, we'll just log success
                logger.info(
                    f"[{self.worker_name}] ✅ Extracted {word_count} words "
                    f"(lang={detected_lang}) from {url}"
                )

            # Step 6: Commission semantic analysis worker
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

    async def _fetch_html(self, url: str, timeout: float = 10.0) -> str:
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
                    'User-Agent': 'Mozilla/5.0 (compatible; HereNewsBot/2.0; +https://here.news)'
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
        Extract publication time from HTML metadata

        Uses trafilatura's built-in date extraction

        Args:
            html: HTML content

        Returns:
            Publication datetime or None
        """
        try:
            from trafilatura import extract_metadata
            metadata = extract_metadata(html)
            if metadata and metadata.date:
                return datetime.fromisoformat(metadata.date.replace('Z', '+00:00'))
        except:
            pass
        return None

    async def _mark_failed(self, page_id: uuid.UUID, reason: str):
        """
        Mark page extraction as failed and create rogue extraction task

        For URLs that block scrapers (401/403), create a task for browser extension

        Args:
            page_id: Page UUID
            reason: Failure reason
        """
        async with self.pool.acquire() as conn:
            # Get page URL
            page = await conn.fetchrow("""
                SELECT url, status FROM core.pages WHERE id = $1
            """, page_id)

            if not page:
                logger.error(f"[{self.worker_name}] Page {page_id} not found")
                return

            # Mark page as failed
            await conn.execute("""
                UPDATE core.pages
                SET status = 'failed', updated_at = NOW()
                WHERE id = $1
            """, page_id)

            # Check if this is a scraper-blocking issue (401/403 or fetch failure)
            is_rogue = any(code in reason for code in ['401', '403', 'Failed to fetch'])

            if is_rogue:
                # Create rogue extraction task for browser extension
                await conn.execute("""
                    INSERT INTO core.rogue_extraction_tasks (page_id, url, status, created_at)
                    VALUES ($1, $2, 'pending', NOW())
                    ON CONFLICT DO NOTHING
                """, page_id, page['url'])

                logger.info(
                    f"[{self.worker_name}] 🔴 Created rogue task for {page['url']} (browser extension will handle)"
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
                page_id = uuid.UUID(job['page_id'])
                await self._mark_failed(page_id, f"Max retries exceeded: {error}")
            except:
                pass
