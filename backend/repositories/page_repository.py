"""
Page Repository - PostgreSQL storage for web page content

Storage strategy:
- PostgreSQL: Primary storage for page content, metadata, embeddings
- No Neo4j involvement (pages are content-only, not graph nodes)
"""
import uuid
import logging
from typing import Optional
import asyncpg
from datetime import datetime

from models.page import Page

logger = logging.getLogger(__name__)


class PageRepository:
    """
    Repository for Page domain model

    PostgreSQL-only storage for web page content
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def get_by_id(self, page_id: uuid.UUID) -> Optional[Page]:
        """
        Retrieve page by ID from PostgreSQL

        Args:
            page_id: Page UUID

        Returns:
            Page model or None
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    id, url, canonical_url, title, description, author,
                    thumbnail_url, content_text, language, word_count,
                    pub_time, metadata_confidence, status,
                    created_at, updated_at
                FROM core.pages
                WHERE id = $1
            """, page_id)

            if not row:
                return None

            return Page(
                id=row['id'],
                url=row['url'],
                canonical_url=row['canonical_url'],
                title=row['title'],
                description=row['description'],
                author=row['author'],
                thumbnail_url=row['thumbnail_url'],
                content_text=row['content_text'],
                language=row['language'],
                word_count=row['word_count'],
                pub_time=row['pub_time'],
                metadata_confidence=row['metadata_confidence'],
                status=row['status'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )

    async def update_extracted_content(
        self,
        page_id: uuid.UUID,
        content_text: str,
        title: str,
        description: Optional[str],
        author: Optional[str],
        thumbnail_url: Optional[str],
        metadata_confidence: float,
        language: str,
        word_count: int,
        pub_time: Optional[datetime]
    ) -> None:
        """
        Update page with extracted content

        Args:
            page_id: Page UUID
            content_text: Extracted text content
            title: Page title
            description: Page description
            author: Page author
            thumbnail_url: Thumbnail image URL
            metadata_confidence: Metadata extraction confidence (0.0-1.0)
            language: Detected language code
            word_count: Word count
            pub_time: Publication timestamp
        """
        async with self.db_pool.acquire() as conn:
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
            """, page_id, content_text, title, description, author,
                 thumbnail_url, metadata_confidence, language, word_count, pub_time)

            logger.debug(f"📄 Updated page {page_id} with extracted content ({word_count} words)")

    async def mark_failed(self, page_id: uuid.UUID) -> None:
        """
        Mark page as failed extraction

        Args:
            page_id: Page UUID
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.pages
                SET status = 'failed', updated_at = NOW()
                WHERE id = $1
            """, page_id)

            logger.info(f"❌ Marked page {page_id} as failed")

    async def create_rogue_task(self, page_id: uuid.UUID, url: str) -> None:
        """
        Create rogue extraction task for pages that need browser-based extraction
        (e.g., paywalled, bot-blocked, or low content)

        Args:
            page_id: Page UUID
            url: Page URL
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO core.rogue_extraction_tasks (page_id, url, status, created_at)
                VALUES ($1, $2, 'pending', NOW())
                ON CONFLICT DO NOTHING
            """, page_id, url)

            logger.info(f"🔴 Created rogue extraction task for {url}")

    async def get_by_canonical_url(self, canonical_url: str) -> Optional[Page]:
        """
        Retrieve page by canonical URL for deduplication

        Args:
            canonical_url: Normalized canonical URL

        Returns:
            Page model or None
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    id, url, canonical_url, title, description, author,
                    thumbnail_url, content_text, language, word_count,
                    pub_time, metadata_confidence, status,
                    created_at, updated_at
                FROM core.pages
                WHERE canonical_url = $1
            """, canonical_url)

            if not row:
                return None

            return Page(
                id=row['id'],
                url=row['url'],
                canonical_url=row['canonical_url'],
                title=row['title'],
                description=row['description'],
                author=row['author'],
                thumbnail_url=row['thumbnail_url'],
                content_text=row['content_text'],
                language=row['language'],
                word_count=row['word_count'],
                pub_time=row['pub_time'],
                metadata_confidence=row['metadata_confidence'],
                status=row['status'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )

    async def create(self, page: Page) -> Page:
        """
        Create new page in PostgreSQL

        Args:
            page: Page domain model

        Returns:
            Created page with timestamps
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO core.pages (
                    id, url, canonical_url, title, description, author,
                    thumbnail_url, content_text, language, word_count,
                    pub_time, metadata_confidence, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """, page.id, page.url, page.canonical_url, page.title, page.description,
                page.author, page.thumbnail_url, page.content_text, page.language,
                page.word_count, page.pub_time, page.metadata_confidence, page.status)

            # Fetch timestamps
            row = await conn.fetchrow("""
                SELECT created_at, updated_at FROM core.pages WHERE id = $1
            """, page.id)

            page.created_at = row['created_at']
            page.updated_at = row['updated_at']

            logger.debug(f"📄 Created page {page.id}: {page.title or page.url}")
            return page

    async def update_status(self, page_id: uuid.UUID, status: str) -> None:
        """
        Update page status

        Args:
            page_id: Page UUID
            status: New status (queued, extraction_complete, semantic_complete, etc.)
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.pages
                SET status = $2, updated_at = NOW()
                WHERE id = $1
            """, page_id, status)

            logger.debug(f"📄 Updated page {page_id} status: {status}")

    async def get_entity_count(self, page_id: uuid.UUID) -> int:
        """
        Get count of entities linked to a page

        Args:
            page_id: Page UUID

        Returns:
            Count of entities
        """
        async with self.db_pool.acquire() as conn:
            count = await conn.fetchval("""
                SELECT COUNT(DISTINCT entity_id)
                FROM core.claim_entity_links
                WHERE claim_id IN (
                    SELECT id FROM core.claims WHERE page_id = $1
                )
            """, page_id)

            return count or 0

    async def get_claim_count(self, page_id: uuid.UUID) -> int:
        """
        Get count of claims extracted from a page

        Args:
            page_id: Page UUID

        Returns:
            Count of claims
        """
        async with self.db_pool.acquire() as conn:
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM core.claims WHERE page_id = $1
            """, page_id)

            return count or 0

    async def get_entities(self, page_id: uuid.UUID):
        """
        Get all entities linked to a page's claims

        Args:
            page_id: Page UUID

        Returns:
            List of entity dictionaries
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT
                    e.id, e.canonical_name, e.entity_type,
                    e.mention_count, e.confidence, e.status
                FROM core.entities e
                JOIN core.claim_entity_links cel ON e.id = cel.entity_id
                JOIN core.claims c ON cel.claim_id = c.id
                WHERE c.page_id = $1
                ORDER BY e.mention_count DESC, e.canonical_name
            """, page_id)

            return [dict(row) for row in rows]

    async def get_claims(self, page_id: uuid.UUID):
        """
        Get all claims extracted from a page

        Args:
            page_id: Page UUID

        Returns:
            List of claim dictionaries
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    id, text, modality, confidence, event_time,
                    created_at
                FROM core.claims
                WHERE page_id = $1
                ORDER BY created_at
            """, page_id)

            return [dict(row) for row in rows]
