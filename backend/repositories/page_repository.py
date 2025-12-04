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
