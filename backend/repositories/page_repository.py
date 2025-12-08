"""
Page Repository - Split storage for web page data

Storage strategy:
- PostgreSQL: Content storage (content_text, embedding, status)
- Neo4j: Metadata and graph relationships (title, url, claims, entities)

The repository abstracts this split from consumers - they work with Page domain model.
"""
import uuid
import logging
from typing import Optional, List
import asyncpg
from datetime import datetime

from models.page import Page
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class PageRepository:
    """
    Repository for Page domain model

    Handles split storage:
    - PostgreSQL: content_text, embedding, processing status
    - Neo4j: metadata, graph relationships
    """

    def __init__(self, db_pool: asyncpg.Pool, neo4j_service: Optional[Neo4jService] = None):
        self.db_pool = db_pool
        self.neo4j = neo4j_service

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    async def get_by_id(self, page_id: uuid.UUID) -> Optional[Page]:
        """
        Retrieve page by ID.

        Fetches from PostgreSQL (handles both old and new schemas).
        Optionally enriches with Neo4j metadata if available.

        Args:
            page_id: Page UUID

        Returns:
            Page model or None
        """
        # Get from PostgreSQL (query all columns for backward compatibility)
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, url, canonical_url, content_text, status,
                       title, description, author, thumbnail_url,
                       language, word_count, pub_time, metadata_confidence,
                       created_at, updated_at
                FROM core.pages
                WHERE id = $1
            """, page_id)

            if not row:
                return None

        # Build Page from PostgreSQL data (works with old schema)
        page = Page(
            id=row['id'],
            url=row['url'],
            canonical_url=row['canonical_url'],
            content_text=row['content_text'],
            status=row['status'],
            title=row.get('title'),
            description=row.get('description'),
            author=row.get('author'),
            thumbnail_url=row.get('thumbnail_url'),
            language=row.get('language'),
            word_count=row.get('word_count', 0) or 0,
            pub_time=row.get('pub_time'),
            metadata_confidence=row.get('metadata_confidence', 0.0) or 0.0,
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

        # Optionally enrich with Neo4j metadata (for new architecture)
        if self.neo4j:
            try:
                neo4j_data = await self.neo4j.get_page_by_id(str(page_id))
                if neo4j_data:
                    # Neo4j data takes precedence if available
                    page.title = neo4j_data.get('title') or page.title
                    page.word_count = neo4j_data.get('word_count') or page.word_count
            except Exception:
                pass  # Neo4j not critical for reads

        return page

    async def get_by_canonical_url(self, canonical_url: str) -> Optional[Page]:
        """
        Retrieve page by canonical URL for deduplication.

        Args:
            canonical_url: Normalized canonical URL

        Returns:
            Page model or None
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id FROM core.pages WHERE canonical_url = $1
            """, canonical_url)

            if not row:
                return None

            return await self.get_by_id(row['id'])

    async def get_content(self, page_id: uuid.UUID) -> Optional[str]:
        """
        Get just the content text (for workers that only need content).

        Args:
            page_id: Page UUID

        Returns:
            Content text or None
        """
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval("""
                SELECT content_text FROM core.pages WHERE id = $1
            """, page_id)

    # =========================================================================
    # WRITE OPERATIONS
    # =========================================================================

    async def create(self, page: Page) -> Page:
        """
        Create new page.

        Creates stub in PostgreSQL. Metadata stored in Neo4j during extraction.

        Args:
            page: Page domain model

        Returns:
            Created page with timestamps
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO core.pages (id, url, canonical_url, status)
                VALUES ($1, $2, $3, $4)
            """, page.id, page.url, page.canonical_url or page.url, page.status)

            row = await conn.fetchrow("""
                SELECT created_at, updated_at FROM core.pages WHERE id = $1
            """, page.id)

            page.created_at = row['created_at']
            page.updated_at = row['updated_at']

            logger.debug(f"📄 Created page stub {page.id}: {page.url}")
            return page

    async def update_content(
        self,
        page_id: uuid.UUID,
        content_text: str,
        status: str = 'extracted'
    ) -> None:
        """
        Update page content in PostgreSQL.

        Args:
            page_id: Page UUID
            content_text: Extracted text content
            status: New status
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.pages
                SET content_text = $2, status = $3, updated_at = NOW()
                WHERE id = $1
            """, page_id, content_text, status)

            logger.debug(f"📄 Updated page {page_id} content")

    async def update_metadata(
        self,
        page_id: uuid.UUID,
        title: str,
        description: Optional[str],
        author: Optional[str],
        thumbnail_url: Optional[str],
        language: str,
        word_count: int,
        pub_time: Optional[datetime],
        metadata_confidence: float,
        domain: str
    ) -> None:
        """
        Update page metadata in Neo4j.

        Args:
            page_id: Page UUID
            title: Page title
            description: Meta description
            author: Author name
            thumbnail_url: Featured image URL
            language: Detected language
            word_count: Content word count
            pub_time: Publication timestamp
            metadata_confidence: Extraction confidence
            domain: Source domain
        """
        if not self.neo4j:
            logger.warning("Neo4j not available - skipping metadata update")
            return

        await self.neo4j.create_or_update_page(
            page_id=str(page_id),
            url=None,  # Already set on create
            title=title,
            domain=domain,
            status='extracted',
            word_count=word_count,
            metadata_confidence=metadata_confidence,
            description=description,
            author=author,
            thumbnail_url=thumbnail_url,
            language=language,
            pub_time=pub_time.isoformat() if pub_time else None
        )

        logger.debug(f"📄 Updated page {page_id} metadata in Neo4j")

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
        pub_time: Optional[datetime],
        domain: str = None
    ) -> None:
        """
        Update page with extracted content (both PostgreSQL and Neo4j).

        This is a convenience method that updates both stores.

        Args:
            page_id: Page UUID
            content_text: Extracted text content
            title: Page title
            description: Page description
            author: Page author
            thumbnail_url: Thumbnail image URL
            metadata_confidence: Metadata extraction confidence
            language: Detected language
            word_count: Word count
            pub_time: Publication timestamp
            domain: Source domain
        """
        # Update content in PostgreSQL
        await self.update_content(page_id, content_text, status='extracted')

        # Update metadata in Neo4j
        if self.neo4j and domain:
            await self.update_metadata(
                page_id=page_id,
                title=title,
                description=description,
                author=author,
                thumbnail_url=thumbnail_url,
                language=language,
                word_count=word_count,
                pub_time=pub_time,
                metadata_confidence=metadata_confidence,
                domain=domain
            )

        logger.debug(f"📄 Updated page {page_id} with extracted content ({word_count} words)")

    async def update_status(self, page_id: uuid.UUID, status: str) -> None:
        """
        Update page processing status.

        Updates both PostgreSQL (for worker coordination) and Neo4j (for queries).

        Args:
            page_id: Page UUID
            status: New status
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.pages
                SET status = $2, updated_at = NOW()
                WHERE id = $1
            """, page_id, status)

        if self.neo4j:
            await self.neo4j.update_page_status(str(page_id), status)

        logger.debug(f"📄 Updated page {page_id} status: {status}")

    async def mark_failed(self, page_id: uuid.UUID, error_message: str = None) -> None:
        """
        Mark page as failed.

        Args:
            page_id: Page UUID
            error_message: Optional error details
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.pages
                SET status = 'failed', updated_at = NOW()
                WHERE id = $1
            """, page_id)

        if self.neo4j:
            await self.neo4j.update_page_status(str(page_id), 'failed')

        logger.info(f"❌ Marked page {page_id} as failed")

    # =========================================================================
    # GRAPH QUERIES (via Neo4j)
    # =========================================================================

    async def get_claim_count(self, page_id: uuid.UUID) -> int:
        """
        Get count of claims linked to this page.

        Args:
            page_id: Page UUID

        Returns:
            Count of claims
        """
        if not self.neo4j:
            return 0

        result = await self.neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
            RETURN count(c) as count
        """, {'page_id': str(page_id)})

        return result[0]['count'] if result else 0

    async def get_entity_count(self, page_id: uuid.UUID) -> int:
        """
        Get count of unique entities mentioned in this page's claims.

        Args:
            page_id: Page UUID

        Returns:
            Count of unique entities
        """
        if not self.neo4j:
            return 0

        result = await self.neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)-[:MENTIONS]->(e:Entity)
            RETURN count(DISTINCT e) as count
        """, {'page_id': str(page_id)})

        return result[0]['count'] if result else 0

    async def get_claims(self, page_id: uuid.UUID) -> List[dict]:
        """
        Get all claims for this page.

        Args:
            page_id: Page UUID

        Returns:
            List of claim dictionaries
        """
        if not self.neo4j:
            return []

        results = await self.neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
            RETURN c.id as id, c.text as text, c.confidence as confidence,
                   c.event_time as event_time, c.created_at as created_at
            ORDER BY c.created_at
        """, {'page_id': str(page_id)})

        return [dict(r) for r in results]

    async def get_entities(self, page_id: uuid.UUID) -> List[dict]:
        """
        Get all entities mentioned in this page's claims.

        Args:
            page_id: Page UUID

        Returns:
            List of entity dictionaries
        """
        if not self.neo4j:
            return []

        results = await self.neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)-[:MENTIONS]->(e:Entity)
            WITH DISTINCT e
            RETURN e.id as id, e.canonical_name as canonical_name,
                   e.entity_type as entity_type, e.wikidata_qid as wikidata_qid,
                   e.mention_count as mention_count, e.confidence as confidence
            ORDER BY e.mention_count DESC
        """, {'page_id': str(page_id)})

        return [dict(r) for r in results]

    # =========================================================================
    # UTILITY
    # =========================================================================

    async def create_rogue_task(self, page_id: uuid.UUID, url: str) -> None:
        """
        Create rogue extraction task for pages needing browser-based extraction.

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
